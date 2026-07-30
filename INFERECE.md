Written here is how to execute the entire pipeline

`Image -> CLIP-encoding -> Cross-modal flow-matching (XFM) -> fMRI-decoding -> raw fMRI Betas`

and the reversal

`raw fMRI Betas -> fMRI-encoding -> Cross-modal flow-matching (XFM) -> CLIP-encoding -> Image`.

# Images

## Sources

- `/u/fdammeier/data/NeuroFlow/annots/COCO_73k_annots_curated.npy`  contains up to five one-sentence descriptions of each image in COCO (file on NeuroFlow HuggingFace)
- `/u/fdammeier/data/NeuroFlow/annots/coco/annotations/` contains the regular coco annotations

Raw images (obtained from nsd stimuli)
- Test: `/u/fdammeier/data/NeuroFlow/nsd/subj0X/test_img/`
- Train: `/u/fdammeier/data/NeuroFlow/nsd/subj0X/train_img/`


# CLIP encoding

## Sources

Raw images (obtained from nsd stimuli)
- Test: `/u/fdammeier/data/NeuroFlow/nsd/subj0X/test_img/`
- Train: `/u/fdammeier/data/NeuroFlow/nsd/subj0X/train_img/`

Model (extracted from `data/extract_features_sdxl_unclip.py`)
- Dataset and loader defined therein
- model: `sdxl.generative_models.sgm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder`

## Execution

```python
# clip_seq_dim = 256
# clip_emb_dim = 1664

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="/u/fdammeier/checkpoints/mindeyev2/open_clip_pytorch_model.bin",
    output_tokens=True,
    only_tokens=True,
)
```

## Outputs

can be found in `/u/fdammeier/data/NeuroFlow/nsd/subj0X/`. E.g. `nsd_sdxl_clip_test_sub1.npy` contains the clip embeddings of the NSD test set

# NeuroVAE

## Sources

- initializer function `vae_utils.load_neurovae_v10_proj` (`vae_utils.py`)
- checkpoint `/u/fdammeier/checkpoints/train_logs/neurovae-nsd-s1-bs64-d1664-zscore-v10-cycle-proj/last.pth`

## Execution

```python
brain_enc = load_neurovae_v10_proj(args, checkpoint_root=args.ckpt_path).to(device).eval()
```

## Outputs

### Encoder

are located in `/u/fdammeier/generations/evals/` under the respective project configuration. E.g. `fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/sub1/single_s1_all_zfmri.pt` contains VAE latents (`zfmri`).

### Decoder

are located in `/u/fdammeier/generations/evals/` under the respective project configuration. E.g. `fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/sub1/single_s1_all_recon_fmri.pt`

# XFM

## Sources

Model (extracted from generate.py)
- model `xfm.sit.SiT`
- checkpoint `/u/fdammeier/checkpoints/train_logs/fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/last.pth`

## Execution

```python
block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

ema = SiT(        
        num_patches=256,
        embed_size=1664,
        hidden_size=1664,
        depth=args.model_depth,
        num_heads=args.model_head,
        **block_kwargs)
```

## Outputs

are located in `/u/fdammeier/generations/evals/` under the respective project configuration. E.g. `fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/sub1/single_s1_all_sample_clip.pt` contains the CLIP latents obtained from passing fMRI latents through XFM. `fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/sub1/single_s1_all_sample_fmri.pt` contains the fMRI latents obtained from passing CLIP latents through XFM.

# CLIP Decoding

## Sources

- initializer function `vae_utils.load_pretrained_sdxl_unclip` (`vae_utils.py`)

## Outputs

are located in `/u/fdammeier/generations/evals/` under the respective project configuration. E.g. `fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj/sub1/single_s1_all_recon_img.pt`





