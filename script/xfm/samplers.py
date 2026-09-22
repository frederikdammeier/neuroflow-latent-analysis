import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from functools import partial
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F


def euler_sampler(
        model,
        latents,
        t_steps,
        num_steps=20,
        heun=False,
        grad_enabled=False,
        ):
    # setup conditioning
    _dtype = latents.dtype    
    t_steps = t_steps
    x_next = latents.to(torch.float64)
    device = x_next.device
    
    # grad_enabled=True keeps the unrolled loop in the autograd graph (needed for gradient-based XAI)
    ctx = nullcontext() if grad_enabled else torch.no_grad()
    with ctx:
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            model_input = x_cur
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype)
                ).to(torch.float64)   #! not [0] except REPA            
            x_next = x_cur + (t_next - t_cur) * d_cur
            
            # Using heun
            if heun and (i < num_steps - 1):
                model_input = x_next
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype)
                    ).to(torch.float64)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                
    return x_next


def euler_sampler_cycle(model, fmri, image, num_steps, heun=False, grad_enabled=False):

    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler(model, z0, fw_steps, num_steps, heun, grad_enabled)
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun, grad_enabled)
    
    return z0_pred.float(), z1_pred.float()


def euler_sampler_bwd(model, fmri, num_steps, heun=False, grad_enabled=False):
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun, grad_enabled)
    
    return z0_pred.float()

def euler_sampler_fwd(model, image, num_steps, heun=False, grad_enabled=False):
    
    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler(model, z0, fw_steps, num_steps, heun, grad_enabled)
    
    return z1_pred.float()