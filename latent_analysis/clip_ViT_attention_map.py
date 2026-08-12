"""
Raw attention-map extraction for the vision tower of this OpenCLIP fork.

Targets `VisionTransformer` from the attached transformer.py, whose
`Transformer` always builds `ResidualAttentionBlock`s wrapping a plain
`nn.MultiheadAttention` (never the custom `Attention`/`CustomTransformer`
path, which is only used elsewhere e.g. for text/CoCa cross-attn blocks).

`ResidualAttentionBlock.attention()` calls:
    self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

need_weights=False is hardcoded at the call site, so the weight matrix is
never returned even though it's computed on the non-fused path. Instead of
patching your source, we intercept the call with a forward *pre*-hook that
flips need_weights/average_attn_weights before `nn.MultiheadAttention.forward`
runs, and a forward hook that grabs the returned weights.

Only the vision tower is touched. Nothing text-related is hooked.
"""

import os
import types
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class ViTAttentionHooker:
    """
    Registers hooks on every ResidualAttentionBlock.attn in
    `visual.transformer.resblocks` to capture full, per-head raw attention
    weights during a forward pass.

    Usage:
        hooker = ViTAttentionHooker(model.visual)
        hooker.register()
        with torch.no_grad():
            image_features = model.encode_image(pixel_values)
        attn = hooker.attn_weights   # {layer_idx: [B, heads, tokens, tokens]}
        hooker.remove()              # restore normal (fused/fast) inference
    """

    def __init__(self, visual_transformer: torch.nn.Module):
        self.visual = visual_transformer
        self.attn_weights: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _pre_hook(module, args, kwargs):
        # Force the underlying nn.MultiheadAttention onto the path that
        # actually computes and returns the weight matrix, per-head.
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False  # keep [B, heads, L, L], don't collapse heads
        return args, kwargs

    def _make_post_hook(self, layer_idx: int):
        def _hook(module, args, kwargs, output):
            # output = (attn_output, attn_output_weights) now that need_weights=True
            _, attn_output_weights = output
            self.attn_weights[layer_idx] = attn_output_weights.detach().to("cpu")
            return output  # unchanged; downstream code only uses output[0]
        return _hook

    def register(self):
        self.remove()
        for i, block in enumerate(self.visual.transformer.resblocks):
            h_pre = block.attn.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            h_post = block.attn.register_forward_hook(self._make_post_hook(i), with_kwargs=True)
            self._handles.extend([h_pre, h_post])

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        self.attn_weights = {}


# --------------------------------------------------------------------------
# Turning raw per-layer attention into a token-wise map over image patches
# --------------------------------------------------------------------------

def token_attention_map(
    attn_weights: Dict[int, torch.Tensor],
    layer_idx: int,
    head: Optional[int] = None,
    batch_idx: int = 0,
) -> torch.Tensor:
    """
    Token-to-patch attention for a single layer.

    Args:
        attn_weights: output of ViTAttentionHooker.attn_weights
        layer_idx: which resblock to read (0-indexed)
        head: specific head index, or None to average over heads
        batch_idx: which item in the batch
        target_idx: index of the target token for attention rollout (0=CLS, 1..N=patches)
    Returns:
        CLS, 1 ... N x N attention map
    """
    w = attn_weights[layer_idx][batch_idx]  # [heads, tokens, tokens]
    w = w.mean(dim=0) if head is None else w[head]
    return w[:, 1:]  # return map for all tokens, drop CLS->CLS entry


def attention_rollout(
    attn_weights: Dict[int, torch.Tensor],
    batch_idx: int = 0,
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """
    Attention Rollout (Abnar & Zuidema, 2020): multiplies head-averaged
    attention matrices across layers, adding the identity at each layer to
    account for the residual stream, and row-normalizing. Gives a less
    noisy "effective attention" signal than any single layer alone.

    Args:
        discard_ratio: if >0, zero out the lowest-`discard_ratio` fraction
            of attention weights per row before rolling out (reduces noise
            from near-uniform low-attention entries). 0 disables this.
        target_idx: index of the target token for attention rollout (0=CLS, 1..N=patches)
    Returns:
        CLS, 1 ... N x N attention map
    """
    result = None
    layer_indices = sorted(attn_weights.keys())
    for i in layer_indices:
        w = attn_weights[i][batch_idx].mean(dim=0)  # [tokens, tokens], head-avg

        if discard_ratio > 0:
            flat = w.view(w.shape[0], -1)
            k = int(flat.shape[-1] * discard_ratio)
            if k > 0:
                lowest = torch.topk(flat, k, dim=-1, largest=False).indices
                flat.scatter_(-1, lowest, 0.0)
            w = flat.view_as(w)

        w = w + torch.eye(w.shape[-1])
        w = w / w.sum(dim=-1, keepdim=True)
        result = w if result is None else w @ result

    return result[:, 1:]  # return map for all tokens, drop CLS->CLS entry


def to_spatial_heatmap(
    patch_vec: torch.Tensor,
    grid_size: tuple,
    image_size: Optional[tuple] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Reshape a flat per-patch vector into a 2D grid and optionally upsample
    to pixel resolution for overlay on the original image.

    Args:
        patch_vec: 1D tensor, length grid_size[0] * grid_size[1]
        grid_size: (H, W) patch grid, e.g. model.visual.grid_size
        image_size: (H, W) in pixels to upsample to; None = leave at grid res
        normalize: min-max normalize to [0, 1] for visualization
    """
    h, w = grid_size
    heat = patch_vec.reshape(h, w)
    if normalize:
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    if image_size is not None:
        heat = F.interpolate(
            heat[None, None], size=image_size, mode="bilinear", align_corners=False
        )[0, 0]
    return heat


# --------------------------------------------------------------------------
# End-to-end example
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import open_clip  # adjust this import if you're using a local/vendored fork
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    MODEL_NAME = "ViT-bigG-14"          # replace with your model
    PRETRAINED = "/u/fdammeier/checkpoints/mindeyev2/open_clip_pytorch_model.bin"  # replace with your checkpoint/tag or local path
    IMAGE_PATH = "/u/fdammeier/repositories/NeuroFlow/sample_data/images/63.png"        # replace with your image
    TARGET_IDX = 0                       # 0=CLS, 1..N=patches

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()

    hooker = ViTAttentionHooker(model.visual)
    hooker.register()

    # read image ids from filenames
    image_ids = [int(filename.split('.')[0]) for filename in os.listdir('../sample_data/images') if filename.endswith('.png')]
    image_ids.sort()  # sort to ensure consistent order

    # create a list of image paths
    image_paths = [os.path.join('../sample_data/images', f"{image_id}.png") for image_id in image_ids]

    # img = Image.open(IMAGE_PATH).convert("RGB")
    # pixel_values = preprocess(img).unsqueeze(0)

    # with torch.no_grad():
    #     image_features = model.encode_image(pixel_values)  # image latent, unaffected by hooks

    # hooker.remove()  # restore fast/fused inference for any subsequent calls
    images = [Image.open(p).convert("RGB") for p in image_paths[:5]]

    pixel_values = torch.stack([preprocess(image) for image in images])  # [B, 3, H, W]

    with torch.no_grad():
        image_features = model.encode_image(pixel_values)  # [B, output_dim]
    hooker.remove()

    # per-image maps, same hooker.attn_weights dict, just vary batch_idx
    # for b in range(pixel_values.shape[0]):
    #     rollout_map = attention_rollout(hooker.attn_weights, batch_idx=b, target_idx=0)

    num_layers = len(model.visual.transformer.resblocks)
    grid_size = model.visual.grid_size  # (H_patches, W_patches)

    # Last-layer attention, head-averaged
    batch_last_layer_maps = [
        token_attention_map(hooker.attn_weights, batch_idx=b, layer_idx=num_layers - 1)
        for b in range(pixel_values.shape[0])
    ]
    batch_last_layer_heats = [
        [
            to_spatial_heatmap(last_layer_map, grid_size)
            for last_layer_map in last_layer_maps
        ]
        for last_layer_maps in batch_last_layer_maps
    ]

    # Full rollout across all layers
    batch_rollout_maps = [
        attention_rollout(hooker.attn_weights, batch_idx=b)
        for b in range(pixel_values.shape[0])
    ]
    batch_rollout_heats = [
        [
            to_spatial_heatmap(rollout_map, grid_size)
            for rollout_map in rollout_maps
        ]
        for rollout_maps in batch_rollout_maps
    ]

    # Sanity check: are rollout maps for two tokens idential?
    token_1 = batch_rollout_maps[0][1]
    token_2 = batch_rollout_maps[0][2]
    print("Difference between rollout maps for token 1 and token 2: ", torch.sum(torch.abs(token_1 - token_2)).item())

    # plot single example for the specified TARGET_IDX (CLS = 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(images[0])
    axes[0].set_title("Input")
    # axes[1].imshow(img)
    axes[1].imshow(batch_last_layer_heats[0][TARGET_IDX].numpy(), cmap="viridis")
    axes[1].set_title(f"Layer {num_layers - 1}, Token {TARGET_IDX} attention")
    # axes[2].imshow(img)
    axes[2].imshow(batch_rollout_heats[0][TARGET_IDX].numpy(), cmap="viridis")
    axes[2].set_title(f"Attention rollout (all layers), Token {TARGET_IDX}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"attention_maps_{TARGET_IDX}.png", dpi=150)
    print("image_features shape:", image_features.shape)
    print(f"Saved attention_map_{TARGET_IDX}.png")

    # plot all non-CLS tokens in a grid
    num_tokens = grid_size[0] * grid_size[1]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    for i in range(num_tokens):
        row, col = divmod(i, grid_size[1])
        # axes[row, col].imshow(img)
        axes[row, col].imshow(batch_last_layer_heats[0][i + 1].numpy(), cmap="viridis")
        # axes[row, col].set_title(f"Token {i + 1}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(f"attention_maps_all_tokens.png", dpi=150)
    print("Saved attention_maps_all_tokens.png")

    # plot all non-CLS tokens in a grid for rollout
    num_tokens = grid_size[0] * grid_size[1]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    for i in range(num_tokens):
        row, col = divmod(i, grid_size[1])
        # axes[row, col].imshow(img)
        axes[row, col].imshow(batch_rollout_heats[0][i + 1].numpy(), cmap="viridis")
        # axes[row, col].set_title(f"Token {i + 1}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(f"attention_rollout_all_tokens.png", dpi=150)
    print("Saved attention_rollout_all_tokens.png")

    # plot entire batch
    plt.figure(figsize=(20, 12))
    for i in range(pixel_values.shape[0]):
        # Display the image
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        
        # Attention heatmap
        plt.subplot(3, 5, i + 6)
        plt.imshow(batch_last_layer_heats[i][TARGET_IDX].numpy(), cmap="viridis")
        # plt.colorbar()
        plt.axis('off')

        # Attention rollout heatmap
        plt.subplot(3, 5, i + 11)
        plt.imshow(batch_rollout_heats[i][TARGET_IDX].numpy(), cmap="viridis")
        # plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"attention_batch.png", dpi=150)
    print("Saved attention_batch.png")