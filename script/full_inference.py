"""
In general, we want to be able to easily run experiments in the form of 'what happens if we
feed such images / fMRI scans to the pipeline?' 'How do latents/activations/relevance? behave?'

In order to do this, we need to have full control over the entire inference pipeline, such that
we can easily run the entire thing for new images and capture relevant sections along the way.

The original repository (this folder) does this in a rather unintuitive way, by seperating out
different aspects of the inference into different scripts. This script here is to be a unified
entrypoint to do everything.
"""
# imports
import random
import numpy as np
import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from tqdm import tqdm
import os
import open_clip
from PIL import Image

from inference_utils import get_image_paths, CLIPImageDataset, load_pretrained_sdxl_unclip, \
    unclip_recon, load_neurovae, begin_timed_block, end_timed_block, preprocess_image_for_clip

from vae_utils import requires_grad

from xfm.sit import SiT
from xfm.samplers import euler_sampler_fwd, euler_sampler_bwd

# configs
"""
We're not getting around making this configurable, even though it's invevitably getting a bit
messy.
"""
# Start by collecting relevant configs as constants, to be be extracted into a file
# later on.
# Generic
SEED = 0 # same as in generate.py
DEVICE = "cuda" # "cpu" or "cuda"
ARTIFACTS_DIR = "/u/fdammeier/artifacts"

# Images
IMAGE_DIR = "/u/fdammeier/repositories/NeuroFlow/sample_data/images"
IMAGE_PATHS = get_image_paths(IMAGE_DIR)

# OpenCLIP
OPENCLIP_MODEL_NAME = "ViT-bigG-14"
OPENCLIP_PRETRAINED = "/u/fdammeier/checkpoints/mindeyev2/open_clip_pytorch_model.bin"

# XFM
XFM_CHECKPOINT_PATH = \
    "/u/fdammeier/checkpoints/train_logs/" + \
    "fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/last.pt"

# NeuroVAE
FMRI_ZSCORES_PATH = "/u/fdammeier/repositories/NeuroFlow/sample_data/fmri_zscores.pt"
NEUROVAE_EMBEDDINGS_PATH = "/u/fdammeier/artifacts/neurovae_embeddings.pt"
NEUROVAE_CONFIG_PATH = "/u/fdammeier/repositories/NeuroFlow/configs/neurovae.yaml"
NEUROVAE_CHECKPOINT_PATH = \
    "/u/fdammeier/checkpoints/train_logs/neurovae-nsd-s1-bs64-d1664-zscore-v10-cycle-proj/last.pth"

# SDXL UnCLIP
# SDXL can use any N x 256T x 1664D embeddings for conditioning.
IMAGE_EMBEDDINGS_PATH = "/u/fdammeier/artifacts/img_embeddings.pt"
SDXL_UNCLIP_CONFIG_PATH = \
    "/u/fdammeier/repositories/NeuroFlow/script/sdxl/generative_models/configs/unclip6.yaml"
SDXL_UNCLIP_CHECKPOINT_PATH = "/u/fdammeier/checkpoints/mindeyev2/unclip6_epoch0_step110000.ckpt"

"""
Booleans to toggle steps. Steps are arranged in a way that out of the box support the most common
scenarios:
- (1) Encode images -> XFM to NeuroVAE embeddings -> Decode fMRI
- (2) Encode fMRI -> XFM to OpenCLIP embeddings -> Decode images
- and both together in a single run of the script.

Scenarios like:
- (3) Encode images -> Decode images
- (4) Encode fMRI -> Decode fMRI

are also supported, but have to be considered separately from (1) and (2).
"""
ENCODE_IMAGES = True        # Runs OpenCLIP encoder on images to get embeddings
DECODE_IMAGES = True        # Runs SDXL UnCLIP decoder on embeddings to get images
ENCODE_FMRI = True          # Runs NeuroVAE encoder on fMRI scans to get embeddings
DECODE_FMRI = True          # Runs NeuroVAE decoder on embeddings to get fMRI scansd
NEUROVAE_TO_CLIP = True     # Runs XFM to map NeuroVAE embeddings to OpenCLIP embeddings
CLIP_TO_NEUROVAE = True     # Runs XFM to map OpenCLIP embeddings to NeuroVAE embeddings


def encode_images_with_openclip(
        image_paths: list,
        model_name: str = OPENCLIP_MODEL_NAME,
        pretrained: str = OPENCLIP_PRETRAINED,
        device: str = DEVICE,
        batch_size: int = 32,
):
    """
    Encodes images using OpenCLIP to obtain embeddings.
    The original code in NeuroFlow uses 
    generative_models.sgm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder
    to obtain embeddings, but we use OpenCLIP directly to retain full flexibility.
    If there is unexpected behavior, double-check the configurations in the original code.
    """
    # Load OpenCLIP model
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        device=device,
        pretrained=pretrained
    )

    # from generative_models.sgm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder
    model.register_buffer(
        "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
    )
    model.register_buffer(
        "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
    )

    model.eval()

    del model.transformer # the text transformer is not needed for image embeddings
    model.visual.output_tokens = True

    dataset = CLIPImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    cls_token_embeddings = []
    img_token_embeddings = []
    print(f"Encoding {len(image_paths)} images with OpenCLIP model {model_name}...")
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        # replicates @autocast behavior from FrozenOpenCLIPImageEmbedder.forward()
        with torch.no_grad(), torch.amp.autocast(
            device,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled()
        ):
            preprocessed_batch = preprocess_image_for_clip(batch)
            cls, img_tokens = model.visual(preprocessed_batch) # returns (cls_token, pos_tokens)

            # FrozenOpenCLIPImageEmbedder.forward() would upcast cls to full precision, but we 
            # keep it in the same precision as img_tokens for consistency.
        
        cls_token_embeddings.append(cls.cpu())
        img_token_embeddings.append(img_tokens.cpu())

    cls_token_embeddings = torch.cat(cls_token_embeddings, dim=0)
    img_token_embeddings = torch.cat(img_token_embeddings, dim=0)

    del model, dataset, dataloader # free up memory

    return cls_token_embeddings, img_token_embeddings

def decode_embeddings_with_sdxl_unclip(
        embeddings: torch.Tensor,
        config_path: str = SDXL_UNCLIP_CONFIG_PATH,
        checkpoint_path: str = SDXL_UNCLIP_CHECKPOINT_PATH,
        device: str = DEVICE,
        batch_size: int = 32,
):
    """
    Decodes CLIP image embeddings (256 img tokens) to images using SDXL UnCLIP.

    This is originally from MindEyeV2.

    This section is quite complicated, so we proceed with minimal adaptations from NeuroFLow.
    We will probably have to dig deeper in order to follow the reconstruction process in detail.
    """
    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx]

    embedding_dataset = EmbeddingDataset(embeddings)
    embedding_dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)

    diffusion_engine, vector_suffix = load_pretrained_sdxl_unclip(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )
    requires_grad(diffusion_engine, False)

    reconstructed_images = []

    # nested loop directly taken from generate.py. Not sure if necessary.
    print(f"Decoding {len(embeddings)} embeddings with SDXL UnCLIP...")
    for batch in tqdm(embedding_dataloader):
        for i in range(len(batch)):
            recon_img = unclip_recon(
                batch[i].unsqueeze(0).to(device),
                diffusion_engine,
                vector_suffix,
                num_samples=1,    # recon is understood to be robust enough for a single sample
                device=device,
            )
            
            reconstructed_images.append(recon_img.cpu())

    reconstructed_images = torch.vstack(reconstructed_images)

    # returned image size is 768 x 768. NeuroFlow resizes everything to 256 x 256
    # which we will follow here. Question this later!

    img_size = 256

    resize_transform = Resize((img_size, img_size))
    reconstructed_images = resize_transform(reconstructed_images)

    del diffusion_engine, vector_suffix, embedding_dataset, embedding_dataloader # free up memory

    return reconstructed_images

def encode_fmri_with_neurovae(
        fmri_zscores: torch.Tensor,
        checkpoint_path: str = NEUROVAE_CHECKPOINT_PATH,
        config_path: str = NEUROVAE_CONFIG_PATH,
        device: str = DEVICE,
):
    """
    Encodes fMRI scans using NeuroVAE to obtain embeddings.
    Embeddings are in the form N x 256T x 1664D
    """
    brain_enc = load_neurovae(checkpoint_path=checkpoint_path, config_path=config_path)
    brain_enc = brain_enc.to(device).eval()
    requires_grad(brain_enc, False)

    class FmriDataset(Dataset):
        def __init__(self, fmri_zscores):
            self.fmri_zscores = fmri_zscores

        def __len__(self):
            return len(self.fmri_zscores)

        def __getitem__(self, idx):
            return self.fmri_zscores[idx]

    dataset = FmriDataset(fmri_zscores)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings = []
    print(f"Encoding {len(fmri_zscores)} fMRI scans with NeuroVAE...")
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            z_fmri, z_fmri_clip = brain_enc.encode(batch)
            del z_fmri_clip # we don't need the clip embeddings
            embeddings.append(z_fmri.cpu())

    del brain_enc, dataset, dataloader # free up memory

    embeddings = torch.vstack(embeddings)
    return embeddings

def decode_fmri_with_neurovae(
        embeddings: torch.Tensor,
        checkpoint_path: str = NEUROVAE_CHECKPOINT_PATH,
        config_path: str = NEUROVAE_CONFIG_PATH,
        device: str = DEVICE,
):
    """
    Decodes NeuroVAE embeddings to fMRI scans.
    Embeddings are in the form N x 256T x 1664D
    """
    brain_dec = load_neurovae(checkpoint_path=checkpoint_path, config_path=config_path)
    brain_dec = brain_dec.to(device).eval()
    requires_grad(brain_dec, False)

    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx]

    dataset = EmbeddingDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    fmri_zscores = []
    print(f"Decoding {len(embeddings)} embeddings with NeuroVAE...")
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            fmri_zscores_batch = brain_dec.generate(batch)
            fmri_zscores.append(fmri_zscores_batch.cpu())

    del brain_dec, dataset, dataloader # free up memory

    fmri_zscores = torch.vstack(fmri_zscores)
    return fmri_zscores

def cross_modal_flow_matching(
        embeddings_start: torch.Tensor,
        direction: str = "forward", # "forward" or "backward"
        checkpoint_path: str = XFM_CHECKPOINT_PATH,
        device: str = DEVICE,
        num_steps: int = 20,
):
    """
    Runs cross-modal flow-matching (XFM) to map embeddings from one modality to another.
    Use `direction = "forward"` to map from OpenCLIP to NeuroVAE embeddings.
    Use `direction = "backward"` to map from NeuroVAE to OpenCLIP embeddings.

    Remains to be seen where euler sampling is best implemented. For now, use function defined in
    xfm.samplers.py. Might not be enough for desired tracking purposes.

    Parameters that are for us considered hard-coded (as defined in generate.py) are:
    - fused_attn = True
    - qk_norm = False
    - depth = 12
    - num_heads = 13
    - heun = False
    """
    block_kwargs = {"fused_attn": True, "qk_norm": False}
    model = SiT(        
            num_patches=256,
            embed_size=1664,
            hidden_size=1664,
            depth=12,
            num_heads=13,
            **block_kwargs)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['ema'])

    model = model.to(device).eval()
    requires_grad(model, False)

    if direction == "forward":
        sample = euler_sampler_fwd(
            model, 
            embeddings_start.to(device),
            num_steps=num_steps, 
            heun=False
        )
    elif direction == "backward":
        sample = euler_sampler_bwd(
            model, 
            embeddings_start.to(device),
            num_steps=num_steps, 
            heun=False
        )

    del model, ckpt # free up memory
    
    return sample

def main():
    """
    Main function to run the full inference pipeline based on the specified configurations.
    """
    set_seed(SEED)

    # Load models and checkpoints as needed based on the toggles
    if ENCODE_IMAGES:
        block_title = "Image encoding with OpenCLIP"
        t = begin_timed_block(block_title)

        cls_embeddings, img_embeddings = encode_images_with_openclip(
            IMAGE_PATHS,
            model_name=OPENCLIP_MODEL_NAME,
            pretrained=OPENCLIP_PRETRAINED,
            device=DEVICE
        )

        print(f"Encoded {len(cls_embeddings)} images. CLS embeddings shape: "
              f"{cls_embeddings.shape}, Image token embeddings shape: {img_embeddings.shape}")

        # Save embeddings to artifacts directory
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        torch.save(
            cls_embeddings.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "cls_embeddings.pt")
        )
        torch.save(
            img_embeddings.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "img_embeddings.pt")
        )

        del cls_embeddings, img_embeddings # free up memory
        end_timed_block(block_title, t)

    if ENCODE_FMRI:
        block_title = "fMRI encoding with NeuroVAE"
        t = begin_timed_block(block_title)
        
        fmri_zscores = torch.load(FMRI_ZSCORES_PATH).to(torch.float32) # ensure correct dtype

        # fMRI is of shape N x T x V, where T is the number of trials (3).
        # While testing, we take the mean across trials to get a single representation per sample.
        fmri_zscores = fmri_zscores.mean(dim=1).unsqueeze(1) # shape: N x 1 x V

        neurovae_embeddings = encode_fmri_with_neurovae(
            fmri_zscores,
            checkpoint_path=NEUROVAE_CHECKPOINT_PATH,
            config_path=NEUROVAE_CONFIG_PATH,
            device=DEVICE
        )

        torch.save(
            neurovae_embeddings.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "neurovae_embeddings.pt")
        )
        end_timed_block(block_title, t)

    if NEUROVAE_TO_CLIP:
        block_title = "Mapping NeuroVAE embeddings to OpenCLIP embeddings with XFM"
        t = begin_timed_block(block_title)

        neurovae_embeddings = torch.load(NEUROVAE_EMBEDDINGS_PATH).to(torch.float32)

        img_embeddings = cross_modal_flow_matching(
            neurovae_embeddings,
            direction="backward",
            checkpoint_path=XFM_CHECKPOINT_PATH,
            device=DEVICE,
            num_steps=20
        )

        torch.save(
            img_embeddings.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "img_embeddings_from_neurovae.pt")
        )

        del neurovae_embeddings, img_embeddings # free up memory
        end_timed_block(block_title, t)

    if CLIP_TO_NEUROVAE:
        block_title = "Mapping OpenCLIP embeddings to NeuroVAE embeddings with XFM"
        t = begin_timed_block(block_title)
        
        img_embeddings = torch.load(IMAGE_EMBEDDINGS_PATH).to(torch.float32)

        neurovae_embeddings = cross_modal_flow_matching(
            img_embeddings,
            direction="forward",
            checkpoint_path=XFM_CHECKPOINT_PATH,
            device=DEVICE,
            num_steps=20
        )

        torch.save(
            neurovae_embeddings.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "neurovae_embeddings_from_img.pt")
        )

        del img_embeddings, neurovae_embeddings # free up memory
        end_timed_block(block_title, t)

    if DECODE_IMAGES:
        block_title = "Image decoding with SDXL UnCLIP"
        t = begin_timed_block(block_title)

        img_embeddings = torch.load(
            os.path.join(ARTIFACTS_DIR, "img_embeddings_from_neurovae.pt")
        ).to(torch.float32)
        
        reconstructed_images = decode_embeddings_with_sdxl_unclip(
            img_embeddings,
            config_path=SDXL_UNCLIP_CONFIG_PATH,
            checkpoint_path=SDXL_UNCLIP_CHECKPOINT_PATH,
            device=DEVICE
        )

        print(f"Decoded {len(reconstructed_images)} images. "
                f"Reconstructed images shape: {reconstructed_images.shape}")

        # save reconstructed images to artifacts directory
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        torch.save(
            reconstructed_images.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "reconstructed_images.pt")
        )
        # also save individual images as PNGs for easy viewing
        os.makedirs(os.path.join(ARTIFACTS_DIR, "reconstructed_images"), exist_ok=True)

        # beware of the image id - this might not be the same as the original,
        # as were're not keeping track.
        for i in range(len(reconstructed_images)):
            img = reconstructed_images[i].permute(1, 2, 0).cpu().numpy() # convert to HWC
            img = (img * 255).astype(np.uint8) # convert to uint8
            Image.fromarray(img).save(
                os.path.join(ARTIFACTS_DIR, "reconstructed_images", f"{i}.png")
            )
            del img

        del img_embeddings, reconstructed_images # free up memory
        end_timed_block(block_title, t)

    if DECODE_FMRI:
        block_title = "fMRI decoding with NeuroVAE"
        t = begin_timed_block(block_title)

        neurovae_embeddings = torch.load(
            os.path.join(ARTIFACTS_DIR, "neurovae_embeddings_from_img.pt")
        ).to(torch.float32)

        reconstructed_fmri = decode_fmri_with_neurovae(
            neurovae_embeddings,
            checkpoint_path=NEUROVAE_CHECKPOINT_PATH,
            config_path=NEUROVAE_CONFIG_PATH,
            device=DEVICE
        )

        torch.save(
            reconstructed_fmri.to(torch.device('cpu')), 
            os.path.join(ARTIFACTS_DIR, "reconstructed_fmri.pt")
        )
        end_timed_block(block_title, t)

if __name__ == "__main__":
    main()
