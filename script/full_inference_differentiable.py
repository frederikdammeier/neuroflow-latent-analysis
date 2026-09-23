"""
This is the differentiable version of the NeuroFlow inference pipeline.

`full_inference.py` runs each stage (CLIP encode, XFM, NeuroVAE, SDXL UnCLIP) as an independent
step that reads its input from disk and writes its output back to disk. That is incompatible with
gradient-based XAI (saliency, Integrated Gradients, ...): saving/loading a tensor detaches it from
the autograd graph, so gradients could never flow from a final reconstruction back to the original
input.

This script instead keeps every stage as a plain in-memory function (`Stage.forward`) and chains
them directly, so a single `run_pipeline(...)` call is one differentiable function from input to
output. Stages are still assembled from the same config toggles as `full_inference.py`
(encode_images, clip_to_neurovae, neurovae_to_clip, decode_fmri, decode_images, encode_fmri), so
configurability is retained — but toggles must form one connected chain (see `build_pipeline`),
since there is no longer a disk-based "hand-off point" that lets independent blocks be rewired
after the fact.

Differentiability status per stage:
- clip_encode, xfm_forward, xfm_backward, neurovae_encode, neurovae_decode: fully differentiable.
- sdxl_decode: NOT YET differentiable. `unclip_recon` (script/inference_utils.py) and
  `decode_first_stage` (script/sdxl/generative_models/sgm/models/diffusion.py) still run under
  `torch.no_grad()`. It is included so image->fmri configs remain complete, but gradients will
  stop at this stage until that vendored code is revisited.
"""
import argparse
import os
from typing import Protocol

import torch
import open_clip

from inference_utils import get_image_paths, CLIPImageDataset, load_pretrained_sdxl_unclip, \
    unclip_recon, load_neurovae, preprocess_image_for_clip, Config, setup_run, \
    expected_gradients

from xfm.sit import SiT
from xfm.samplers import euler_sampler_fwd, euler_sampler_bwd

from PIL import Image
import torchvision.transforms.functional as TF

from captum.attr import IntegratedGradients


class Stage(Protocol):
    """A single differentiable model, split into a one-time `load` and a pure `forward`."""
    name: str

    def load(self, device: str) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class ClipEncodeStage():
    """Images -> OpenCLIP image-token embeddings [B, 256, 1664]. Fully differentiable."""
    name = "clip_encode"

    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None

    def load(self, device: str) -> None:
        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name, device=device, pretrained=self.pretrained
        )
        # from generative_models.sgm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder
        model.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        model.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        model.eval().requires_grad_(False)
        del model.transformer  # text tower is unused for image embeddings
        model.visual.output_tokens = True
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = preprocess_image_for_clip(x)
        _cls, img_tokens = self.model.visual(x)
        return img_tokens


class XfmForwardStage:
    """OpenCLIP embeddings -> NeuroVAE embeddings via 20-step Euler XFM. Differentiable."""
    name = "xfm_forward"

    def __init__(self, checkpoint_path: str, num_steps: int = 20):
        self.checkpoint_path = checkpoint_path
        self.num_steps = num_steps
        self.model = None

    def load(self, device: str) -> None:
        self.model = _load_sit(self.checkpoint_path, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return euler_sampler_fwd(self.model, x, num_steps=self.num_steps, heun=False,
                                  grad_enabled=True)


class XfmBackwardStage:
    """NeuroVAE embeddings -> OpenCLIP embeddings via 20-step Euler XFM. Differentiable."""
    name = "xfm_backward"

    def __init__(self, checkpoint_path: str, num_steps: int = 20):
        self.checkpoint_path = checkpoint_path
        self.num_steps = num_steps
        self.model = None

    def load(self, device: str) -> None:
        self.model = _load_sit(self.checkpoint_path, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return euler_sampler_bwd(self.model, x, num_steps=self.num_steps, heun=False,
                                  grad_enabled=True)


def _load_sit(checkpoint_path: str, device: str):
    # hard-coded architecture args, as in full_inference.py's cross_modal_flow_matching
    block_kwargs = {"fused_attn": True, "qk_norm": False}
    model = SiT(num_patches=256, embed_size=1664, hidden_size=1664, depth=12, num_heads=13,
                **block_kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["ema"])
    return model.to(device).eval().requires_grad_(False)


class NeuroVaeEncodeStage:
    """fMRI z-scores -> NeuroVAE embeddings [B, 256, 1664]. Fully differentiable."""
    name = "neurovae_encode"

    def __init__(self, checkpoint_path: str, config_path: str):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None

    def load(self, device: str) -> None:
        model = load_neurovae(checkpoint_path=self.checkpoint_path, config_path=self.config_path)
        self.model = model.to(device).eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_fmri, _z_fmri_clip = self.model.encode(x, sample=False)  # deterministic posterior mode
        return z_fmri


class NeuroVaeDecodeStage:
    """NeuroVAE embeddings -> fMRI z-scores. Fully differentiable."""
    name = "neurovae_decode"

    def __init__(self, checkpoint_path: str, config_path: str):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None

    def load(self, device: str) -> None:
        model = load_neurovae(checkpoint_path=self.checkpoint_path, config_path=self.config_path)
        self.model = model.to(device).eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.generate(x)


class SdxlDecodeStage:
    """NeuroVAE-mapped CLIP embeddings -> images via SDXL UnCLIP. NOT YET differentiable."""
    name = "sdxl_decode"

    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.diffusion_engine = None
        self.vector_suffix = None

    def load(self, device: str) -> None:
        self.diffusion_engine, self.vector_suffix = load_pretrained_sdxl_unclip(
            config_path=self.config_path, checkpoint_path=self.checkpoint_path, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # unclip_recon only accepts one sample at a time; 
        # still runs under torch.no_grad() internally
        images = [
            unclip_recon(
                x[i:i+1],
                self.diffusion_engine,
                self.vector_suffix,
                num_samples=1,
                device=x.device,
                grad_enabled=torch.is_grad_enabled()
            )
            for i in range(x.shape[0])
        ]
        return torch.cat(images, dim=0)


def build_pipeline(config: Config) -> list[Stage]:
    """
    Assemble the ordered stage chain implied by `config`'s toggles.
    This complex if/else wiring is undesirable, but we accept it for now to be able to use the same
    config class as in full_inference.py.

    Unlike full_inference.py, toggles can no longer be enabled independently of one another: since
    stage outputs are wired directly (in-memory) into the next stage's input, the enabled toggles
    must form exactly one connected chain from a supported starting point to a supported end point.
    Raises ValueError for combinations that don't form such a chain (e.g. requesting both XFM
    directions at once, which would need two independent inputs/pipelines run separately instead).
    """
    stages: list[Stage] = []

    if config.encode_images and config.encode_fmri:
        raise ValueError("Enable only one of encode_images / encode_fmri per pipeline run — "
                          "run the two directions as separate invocations.")

    if config.encode_images:
        stages.append(ClipEncodeStage(config.clip.model_name, config.clip.pretrained_path))
        if config.clip_to_neurovae:
            stages.append(XfmForwardStage(config.xfm.checkpoint_path))
            if config.decode_fmri:
                stages.append(NeuroVaeDecodeStage(config.neurovae.checkpoint_path,
                                                   config.neurovae.config_path))
        elif config.decode_images:
            stages.append(
                SdxlDecodeStage(config.unclip.config_path,
                                config.unclip.checkpoint_path))
        else:
            raise ValueError("encode_images requires clip_to_neurovae (-> optionally decode_fmri) "
                              "or decode_images to form a complete chain.")
    elif config.encode_fmri:
        stages.append(NeuroVaeEncodeStage(config.neurovae.checkpoint_path,
                                           config.neurovae.config_path))
        if config.neurovae_to_clip:
            stages.append(XfmBackwardStage(config.xfm.checkpoint_path))
            if config.decode_images:
                stages.append(SdxlDecodeStage(config.unclip.config_path,
                                               config.unclip.checkpoint_path))
        elif config.decode_fmri:
            stages.append(NeuroVaeDecodeStage(config.neurovae.checkpoint_path,
                                               config.neurovae.config_path))
        else:
            raise ValueError("encode_fmri requires neurovae_to_clip (-> optionally decode_images) "
                              "or decode_fmri to form a complete chain.")
    else:
        raise ValueError(
            "Enable one of encode_images / encode_fmri as the pipeline's entry point."
        )

    return stages


def run_pipeline(
        x: torch.Tensor,
        stages: list[Stage],
        device: str,
        taps: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Run `x` through `stages` in memory, preserving the autograd graph end to end.

    `taps`, if provided, collects a detached/cpu copy of each stage's output for inspection or
    debugging — this side channel never touches the live tensor used for the forward pass.
    """
    for stage in stages:
        stage.load(device)
        torch.cuda.reset_peak_memory_stats(device)
        x = stage.forward(x)
        torch.cuda.synchronize()
        print(f"{stage.name}: peak {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")
        if taps is not None:
            taps[stage.name] = x.detach().cpu()
    return x


def load_pipeline_input(config: Config, device: str) -> torch.Tensor:
    """Load the tensor that seeds the pipeline (raw images or fMRI z-scores) based on toggles."""
    if config.encode_images:
        image_paths = get_image_paths(config.image_path)
        dataset = CLIPImageDataset(image_paths)
        return torch.stack([dataset[i] for i in range(len(dataset))]).to(device)
    else:
        fmri_zscores = torch.load(config.fmri_zscores_path).to(torch.float32)
        return fmri_zscores.mean(dim=1).unsqueeze(1).to(device)  # average over trials, as before

class MaskedMeanStage():
    """Compute the mean of the input tensor `x` along the second dimension,
    considering only the elements where `mask` is non-zero."""
    name = "masked_mean"

    def __init__(self, mask):
        self.mask = mask

    def load(self, device: str) -> None:
        self.mask = self.mask.to(device)

    def forward(self, x):
        masked_x = x * self.mask
        sum_masked_x = masked_x.sum(dim=-1)
        sum_mask = self.mask.sum(dim=-1)
        mean_masked_x = sum_masked_x / (sum_mask + 1e-8)
        return mean_masked_x

class MaskedRelativeMeanStage():
    """Compute the mean of the input tensor `x` along the second dimension,
    considering only the elements where `mask` is non-zero, 
    and contrast it with the mean of all unmasked voxels."""
    name = "masked_relative_mean"

    def __init__(self, mask):
        self.mask = mask

    def load(self, device: str) -> None:
        self.mask = self.mask.to(device)

    def forward(self, x):
        masked_x = x * self.mask
        mean_masked_x = masked_x.sum(dim=-1) / (self.mask.sum(dim=-1) + 1e-8)

        inverse_mask = 1 - self.mask
        inverse_masked_x = x * inverse_mask
        mean_inverse_masked_x = inverse_masked_x.sum(dim=-1) / (inverse_mask.sum(dim=-1) + 1e-8)
        return mean_masked_x - mean_inverse_masked_x

class MaskedPositiveMeanStage():
    """Compute the mean of the positive elements of the input tensor `x` along the second
    dimension, considering only the elements where `mask` is non-zero."""
    name = "masked_positive_mean"

    def __init__(self, mask):
        self.mask = mask

    def load(self, device: str) -> None:
        self.mask = self.mask.to(device)

    def forward(self, x):
        masked_x = x * self.mask
        masked_x = torch.clamp(masked_x, min=0)
        sum_masked_x = masked_x.sum(dim=-1)
        sum_mask = self.mask.sum(dim=-1)
        mean_masked_x = sum_masked_x / (sum_mask + 1e-8)
        return mean_masked_x

class MaskedNegativeMeanStage():
    """Compute the mean of the negative elements of the input tensor `x` along the second
    dimension, considering only the elements where `mask` is non-zero."""
    name = "masked_negative_mean"

    def __init__(self, mask):
        self.mask = mask

    def load(self, device: str) -> None:
        self.mask = self.mask.to(device)

    def forward(self, x):
        masked_x = x * self.mask
        masked_x = torch.clamp(masked_x, max=0)
        sum_masked_x = masked_x.sum(dim=-1)
        sum_mask = self.mask.sum(dim=-1)
        mean_masked_x = sum_masked_x / (sum_mask + 1e-8)
        return mean_masked_x

def main(config: Config, run_path: str):
    ig_config = config.ig

    roi_mask_tensor = torch.load(ig_config.roi_mask_dir)
    
    if ig_config.target == "masked_mean":
        mms = MaskedMeanStage(roi_mask_tensor)
    elif ig_config.target == "masked_positive_mean":
        mms = MaskedPositiveMeanStage(roi_mask_tensor)
    elif ig_config.target == "masked_negative_mean":
        mms = MaskedNegativeMeanStage(roi_mask_tensor)
    elif ig_config.target == "masked_relative_mean":
        mms = MaskedRelativeMeanStage(roi_mask_tensor)
    else:
        raise ValueError(f"Unknown target: {ig_config.target}")

    stages = build_pipeline(config)
    stages.append(mms)
    # x = load_pipeline_input(config, config.device)
    # x.requires_grad_(True)
    # load XAI sample
    img = Image.open(config.image_path)
    x = torch.stack([TF.to_tensor(img)]).float().to(config.device)
    x.requires_grad_(True)

    taps: dict[str, torch.Tensor] = {}
    run_pipeline_parameterized = lambda x: run_pipeline(x, stages, config.device, taps=taps)

    if ig_config.use_expected_gradients:
        # EG expects multiple images as baseline
        baseline_pool = torch.stack(
            [
                TF.to_tensor(Image.open(img_dir))
                for img_dir in get_image_paths(ig_config.baseline_dir)
            ]
        ).float().to(config.device)
        baseline_pool.requires_grad_(True)

        

        attributions = expected_gradients(
            run_pipeline_parameterized,
            x,
            baseline_pool,
            k_samples=ig_config.n_steps,
            batch_size=ig_config.internal_batch_size,
            device=config.device
        )

        delta = 0.0 # TODO not implemented

    else:
        # load baseline
        # IG expects single image baseline
        baseline = Image.open(ig_config.baseline_dir)
        baseline = torch.stack([TF.to_tensor(baseline)]).float().to(config.device)
        baseline.requires_grad_(True)

        ig = IntegratedGradients(run_pipeline_parameterized)
        attributions, delta = ig.attribute(
            x, 
            baselines=baseline,
            n_steps=ig_config.n_steps,
            internal_batch_size=ig_config.internal_batch_size,
            return_convergence_delta=ig_config.return_convergence_delta)

    # output = run_pipeline(x, stages, config.device, taps=taps)

    # demonstrates the pipeline is differentiable end to end; 
    # real XAI target functions replace this
    # output.sum().backward()
    # assert x.grad is not None, "Gradient did not reach the pipeline input."

    os.makedirs(run_path, exist_ok=True)
    torch.save(attributions.detach().cpu(), os.path.join(run_path, "attributions.pt"))
    torch.save(torch.tensor(delta).detach().cpu(), os.path.join(run_path, "delta.pt"))
    # torch.save(output.detach().cpu(), os.path.join(run_path, "output.pt"))
    # torch.save(x.grad.detach().cpu(), os.path.join(run_path, "input_grad.pt"))
    for name, tensor in taps.items():
        torch.save(tensor, os.path.join(run_path, f"{name}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentiable full inference pipeline for NeuroFlow"
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config, run_path = setup_run(args.config_path)

    main(config, run_path)
