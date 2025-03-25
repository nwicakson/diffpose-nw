import torch
import numpy as np
import logging

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """
    Get beta schedule for diffusion process
    """
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        timesteps = (
            torch.arange(num_diffusion_timesteps + 1, dtype=torch.float64) /
            num_diffusion_timesteps + 0.008
        )
        alphas = torch.cos((timesteps + 0.008) / 1.008 * np.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
        betas = betas.numpy()
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def generalized_steps(x, mask, seq, model, betas, eta=0.0):
    """
    Generalized steps for the reverse diffusion process without verbose logging
    
    Args:
        x: Initial tensor (typically noisy)
        mask: Attention mask
        seq: Sequence of timesteps to use
        model: Diffusion model
        betas: Beta schedule
        eta: Parameter for stochastic sampling (0 = deterministic)
    
    Returns:
        A list of [x_0, x_1, ..., x_T] at different timesteps
    """
    with torch.no_grad():
        # Get alphas from betas
        alphas = 1. - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).cuda(), alphas_cumprod[:-1]])
        
        # Initialize sequence iterator and results
        n = x.size(0)
        x0_preds = []
        xs = [x]
        
        # Process sequence in reverse order
        for i, j in enumerate(reversed(seq)):
            t = (torch.ones(n) * j).cuda()
            
            # Get current timestep values
            a_t = alphas_cumprod.index_select(0, t.long())
            a_prev = alphas_cumprod_prev.index_select(0, t.long())
            
            # Ensure correct dimensions
            a_t = a_t.view(-1, 1, 1)
            a_prev = a_prev.view(-1, 1, 1)
            
            # Predict noise at current timestep
            x_t = xs[-1]
            e_t = model(x_t, mask, t)
            
            # Predict original signal x_0
            x0_from_e = (1.0 / a_t.sqrt()) * (x_t - (1 - a_t).sqrt() * e_t)
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e)
            
            # Setup for next step
            c1 = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            c2 = ((1 - a_prev) - c1 ** 2).sqrt()
            
            # Calculate next timestep value
            x_next = a_prev.sqrt() * x0_from_e + c1 * torch.randn_like(x_t) + c2 * e_t
            xs.append(x_next)
    
    return x0_preds, xs