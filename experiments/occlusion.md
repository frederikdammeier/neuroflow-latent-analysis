In this file I plan and document design decisions relating to occlusion experiments on NeuroFlow.

# Measuring the Effect of Occlusions

General: calculate delta to non-occluded samples via some distance measure

## on Images

Pixel level metrics probably not useful on the generations. Could employ semantic metrics from NeuroFlow

## on fMRI

Most obvious: correlation as distance measure

# What to occlude

## on Images

General options:
- Patch occlusions
- Gaussian blur occlusions
- Inpainting occlusions

Advanced options:
- Segmentation-based occlusion - specifically occlude objects (E.g. Meta's Segment Anything Model)
- Use genAI to generate samples with certain desired properties.

## on fMRI Betas

Naive: block-out certain regions of the brain - see how much signal remains in image generation.

## Occlusion on Latent Representations

# Additional Ideas
- Targeted semantic experiments (e.g. 'Cats vs Dogs'). I would suspect that the model is not powerful enough to derive any meaningful result here - and it would be quite some work.

