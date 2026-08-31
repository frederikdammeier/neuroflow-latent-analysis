"""
Primarily thought of as an improvement to vae_utils.py, as many functions therein are not
configurable enough.
"""
# imports 
import hashlib
import os
import subprocess
import sys
import dacite
import datetime
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import yaml
import copy
import time
import kornia
from dataclasses import dataclass, asdict, field
from accelerate.utils import set_seed

@dataclass
class CLIPConfig:
    model_name: str = "ViT-bigG-14"
    pretrained_path: str = "/u/fdammeier/checkpoints/mindeyev2/open_clip_pytorch_model.bin"

@dataclass
class NeuroVAEConfig:
    config_path: str = "/u/fdammeier/repositories/NeuroFlow/configs/neurovae.yaml"
    checkpoint_path: str = \
    "/u/fdammeier/checkpoints/train_logs/neurovae-nsd-s1-bs64-d1664-zscore-v10-cycle-proj/last.pth"

@dataclass
class UNCLIPConfig:
    config_path: str = \
    "/u/fdammeier/repositories/NeuroFlow/script/sdxl/generative_models/configs/unclip6.yaml"
    checkpoint_path: str = \
    "/u/fdammeier/checkpoints/mindeyev2/unclip6_epoch0_step110000.ckpt"

@dataclass
class XFMConfig:
    checkpoint_path: str = \
    "/u/fdammeier/checkpoints/train_logs/" + \
    "fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/last.pt"

@dataclass
class Config:
    """
    This class holds all configurables for full_inference.py.
    
    We adopt a light hierarchy do account for the modular nature of the inference pipeline.
    Reproducability is ensured through run-specifc saving; changes can be made either here
    or in dedicated yaml files.
    """
    # General settings
    seed: int = 0
    device: str = "cuda"
    root_dir: str = "/u/fdammeier/artifacts"
    notes: str = ""

    # Data paths
    image_path: str = "/u/fdammeier/repositories/NeuroFlow/sample_data/images"
    fmri_zscores_path: str = "/u/fdammeier/repositories/NeuroFlow/sample_data/fmri_zscores.pt"

    # Latent paths (if None - assumes default location based on root_dir)
    image_embeddings_path: str | None = None
    neurovae_embeddings_path: str | None = None
    img_embeddings_from_neurovae_path: str | None = None
    neurovae_embeddings_from_img_path: str | None = None

    # Component settings
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    neurovae: NeuroVAEConfig = field(default_factory=NeuroVAEConfig)
    unclip: UNCLIPConfig = field(default_factory=UNCLIPConfig)
    xfm: XFMConfig = field(default_factory=XFMConfig)

    # Component toggles
    encode_images: bool = True        # Runs OpenCLIP encoder on images to get embeddings
    decode_images: bool = True        # Runs SDXL UnCLIP decoder on embeddings to get images
    encode_fmri: bool = True          # Runs NeuroVAE encoder on fMRI scans to get embeddings
    decode_fmri: bool = True          # Runs NeuroVAE decoder on embeddings to get fMRI scansd
    neurovae_to_clip: bool = True     # Runs XFM to map NeuroVAE embeddings to OpenCLIP embeddings
    clip_to_neurovae: bool = True     # Runs XFM to map OpenCLIP embeddings to NeuroVAE embeddings

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

def setup_run(config_path: str):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = dacite.from_dict(data_class=Config, data=config_dict)

    set_seed(config.seed)

    # Make run unique
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.new('sha256')
    h.update(config_path.encode()) # if course this is not unique, but users are expected to know
                                   # that is does not make sense to run the same config twice at
                                   # the same time.
    config_hash = h.hexdigest()[-6:]

    run_id = f"full_inference_{time}_{config_hash}"

    run_dir = os.path.join(config.root_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Set embedding default paths if not provided
    if config.image_embeddings_path is None:
        config.image_embeddings_path = os.path.join(run_dir, "image_embeddings.pt")
    if config.neurovae_embeddings_path is None:
        config.neurovae_embeddings_path = os.path.join(run_dir, "neurovae_embeddings.pt")
    if config.img_embeddings_from_neurovae_path is None:
        config.img_embeddings_from_neurovae_path = os.path.join(
            run_dir, 
            "img_embeddings_from_neurovae.pt"
        )
    if config.neurovae_embeddings_from_img_path is None:
        config.neurovae_embeddings_from_img_path = os.path.join(
            run_dir, 
            "neurovae_embeddings_from_img.pt"
        )

    # Save config
    config.save(os.path.join(run_dir, "config.yaml"))

    # Save git state
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        diff = subprocess.check_output(["git", "diff"]).decode()
        with open(os.path.join(run_dir, "git_state.txt"), "w") as f:
            f.write(f"commit: {commit}\n\n--- dirty diff ---\n{diff}")
    except Exception:
        pass  # not a git repo, or git not installed — don't crash the run over it

    return config, run_dir


def get_image_paths(image_dir):
    """
    Get all image paths from the given directory. If the directory contains images with integer
    Names (e.g., 0.png, 1.png, 2.png), they will be sorted in ascending order. If the directory
    contains images with non-integer names, all files will be returned in the order they are
    listed in the directory.
    """
    # handle the case where the image_dir is a single image path
    if os.path.isfile(image_dir):
        return [image_dir]
    else:
        # default case: image_dir contains png images labeled numerically.
        # we want the images to be sorted in ascending order, so we sort the list of image paths.
        try:
            image_ids = [
                int(filename.split('.')[0])
                for filename in os.listdir(image_dir) 
                if filename.endswith('.png')
            ]
            image_ids.sort()

            return [os.path.join(image_dir, f"{image_id}.png") for image_id in image_ids]
        except Exception as e: # we assume the directory contains different image formats.
            print(f"Error while parsing integer image names: {e}")
            print("Falling back to returning all files in the directory.")
            return [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        
class CLIPImageDataset(Dataset):
    """
    Directly taken from "data/extract_features_sdxl_unclip.py".
    """
    def __init__(self, image_paths):
        self.img_data = image_paths

    def __getitem__(self, idx):
        img = Image.open(self.img_data[idx])
        img = TF.to_tensor(img).float()
        return img

    def __len__(self):
        return len(self.img_data)

def preprocess_image_for_clip(x):
    """
    Replicate the preprocessing pipeline used in the original NeuroFlow codebase.
    """
    # from generative_models.sgm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(x.device)
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(x.device)

    x = kornia.geometry.resize(
        x,
        (224, 224),
        interpolation="bicubic",
        align_corners=True,
        antialias=True,
    )
    x = kornia.enhance.normalize(x, mean, std)

    return x

def load_pretrained_sdxl_unclip(
        config_path: str = \
            "/u/fdammeier/repositories/NeuroFlow/script/" + \
            "sdxl/generative_models/configs/unclip6.yaml",
        checkpoint_path: str = "/u/fdammeier/checkpoints/mindeyev2/unclip6_epoch0_step110000.ckpt",
        device: str = "cuda"
):
    """
    taken from vae_utils.py
    """
    # This unfortunate piece of code is necessary to avoid circular import issues when 
    # importing DiffusionEngine from sdxl.
    file_path = os.path.abspath(__file__)
    execution_path = os.path.dirname(file_path)
    sdxl_path = os.path.join(execution_path, "sdxl")
    generative_models_path = os.path.join(sdxl_path, "generative_models")
    sys.path.append(sdxl_path)
    sys.path.append(generative_models_path)
    from sdxl.generative_models.sgm.models.diffusion import DiffusionEngine

    # prep unCLIP
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
    offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(network_config=network_config,
                        denoiser_config=denoiser_config,
                        first_stage_config=first_stage_config,
                        conditioner_config=conditioner_config,
                        sampler_config=sampler_config,
                        scale_factor=scale_factor,
                        disable_first_stage_autocast=disable_first_stage_autocast)
    
    # set to inference
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    ckpt_path = checkpoint_path
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion_engine.load_state_dict(ckpt['state_dict'])

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)
    
    return diffusion_engine, vector_suffix

def unclip_recon(
        x, 
        diffusion_engine, 
        vector_suffix,
        num_samples=1, 
        offset_noise_level=0.04,
        device="cuda"
    ):
    # This unfortunate piece of code is necessary to avoid circular import issues when importing 
    # from sdxl.
    file_path = os.path.abspath(__file__)
    execution_path = os.path.dirname(file_path)
    sdxl_path = os.path.join(execution_path, "sdxl")
    generative_models_path = os.path.join(sdxl_path, "generative_models")
    sys.path.append(sdxl_path)
    sys.path.append(generative_models_path)
    from sdxl.generative_models.sgm.util import append_dims

    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), \
         torch.amp.autocast('cuda', dtype=torch.float16), \
         diffusion_engine.ema_scope():

        # starting noise, can change to VAE outputs of initial image for img2img
        z = torch.randn(num_samples,4,96,96).to(device)

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples

def load_neurovae(
        checkpoint_path: str, 
        config_path: str,
    ):
    """
    Load `NeuroVAE_P` from `checkpoint_path`.
    """
    # This unfortunate piece of code is necessary due to the package structure.
    # Otherwise imports fail in vae/neurovae.py.
    file_path = os.path.abspath(__file__)
    execution_path = os.path.dirname(file_path)
    vae_path = os.path.join(execution_path, "vae")
    sys.path.append(vae_path)

    # Lazy import to keep utils import lightweight.
    from vae.neurovae import NeuroVAE_P

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ddconfig = copy.deepcopy(config['model']['params']['ddconfig'])

    model = NeuroVAE_P(ddconfig=ddconfig)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    return model

# logging untility
def begin_timed_block(name):
    print(f"Starting '{name}'...")
    start_time = time.time()
    return start_time

def end_timed_block(name, start_time):
    end_time = time.time()
    print(f"Finished '{name}' in {end_time - start_time:.2f} seconds.")