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

def generalized_steps(x, src_mask, seq, model, betas, eta=0.0):
    """
    Original generalized steps function for compatibility
    """
    return simplified_diffusion_sampling(x, src_mask, model, betas, num_steps=len(seq), use_ignn=False)

def simplified_diffusion_sampling(x, mask, model, betas, num_steps=10, use_ignn=False, logging_freq=10, verbose=True):
    """
    A simplified implementation of diffusion sampling that should work reliably
    with both GCN and IGNN models.
    
    Args:
        x: Input tensor (noisy data) of shape [batch_size, num_nodes, feature_dim]
        mask: Attention mask
        model: The diffusion model (GCN or IGNN)
        betas: Noise schedule (numpy array or torch tensor)
        num_steps: Number of denoising steps to perform
        use_ignn: Whether using IGNN model
        logging_freq: How often to log progress (every N steps)
        verbose: Whether to log the diffusion steps 
        
    Returns:
        Denoised tensor
    """
    with torch.no_grad():
        device = x.device
        batch_size = x.shape[0]
        
        # Ensure betas is a numpy array
        if isinstance(betas, torch.Tensor):
            betas = betas.cpu().numpy()
        
        # Limit number of steps to the length of the noise schedule
        total_steps = len(betas)
        num_steps = min(num_steps, total_steps - 1)
        
        # Compute diffusion parameters
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        # Move to torch and to device
        alphas = torch.from_numpy(alphas).float().to(device)
        alphas_cumprod = torch.from_numpy(alphas_cumprod).float().to(device)
        alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float().to(device)
        
        # Calculate timesteps (evenly spaced)
        step_size = max(1, total_steps // num_steps)
        timesteps = list(range(total_steps-1, 0, -step_size))[:num_steps]
        if not timesteps:
            timesteps = [1]  # At least one step
            
        # Log timesteps only once and only if verbose
        if verbose:
            logging.info(f"Using {len(timesteps)} diffusion steps: {timesteps}")
        
        # Start from the input
        xt = x
        
        # Save all intermediate steps (for compatibility with original code)
        all_xs = [xt]
        all_x0_preds = []
        
        # Iteratively denoise
        for i, t in enumerate(timesteps):
            # Create timestep tensor (careful with CUDA and type consistency)
            t_tensor = torch.ones((batch_size,), device=device, dtype=torch.long) * t
            
            try:
                # Predict noise
                if use_ignn:
                    # For IGNN, ensure the timestep is correctly handled
                    predicted_noise = model(xt, mask, t_tensor.float(), None)
                else:
                    # Original GCN model
                    predicted_noise = model(xt, mask, t_tensor.float(), 0)
                
                # Single-step denoising
                alpha_t = alphas_cumprod[t]
                alpha_prev = alphas_cumprod_prev[t]
                
                # Compute variance
                sigma_t = ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
                
                # Predict x0
                pred_x0 = (xt - predicted_noise * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
                all_x0_preds.append(pred_x0)
                
                # Direction to next step
                dir_xt = (1 - alpha_prev - sigma_t**2).sqrt() * predicted_noise
                
                # Update xt (deterministic version)
                xt = alpha_prev.sqrt() * pred_x0 + dir_xt
                all_xs.append(xt)
                
                # Log progress less frequently and only if verbose
                if verbose and ((i+1) % logging_freq == 0 or i == len(timesteps) - 1):
                    logging.info(f"Completed diffusion step {i+1}/{len(timesteps)}, timestep {t}")
                
            except Exception as e:
                logging.error(f"Error in diffusion step {i+1}, timestep {t}: {str(e)}")
                logging.error(f"Tensor shapes - x: {xt.shape}, mask: {mask.shape}, t: {t_tensor.shape}")
                raise e
                
        return all_xs, all_x0_preds