from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.GraFormer import *

### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    
class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ### time embedding ###
        self.temb_proj = torch.nn.Linear(emd_dim,hid_dim)

    def forward(self, x, temb):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out

class AdaptiveGCNdiff(nn.Module):
    """
    An adaptive implicit version of GCNdiff that maintains the same interface
    but uses implicit techniques internally for better convergence.
    """
    def __init__(self, adj, config):
        super(AdaptiveGCNdiff, self).__init__()
        
        self.adj = adj
        self.config = config
        ### load gcn configuration ###
        con_gcn = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, n_pts = \
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim, \
                con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
                
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim*4
                
        ### Generate Graphformer  ###
        self.n_layers = num_layers

        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = self.hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)
        
        ### diffusion configuration (compatibility layer) ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim,self.emd_dim),
            torch.nn.Linear(self.emd_dim,self.emd_dim),
        ])
        
        ### Adaptive implicit configuration ###
        # Get implicit parameters from config (with defaults)
        self.use_adaptive = getattr(config, 'use_adaptive', True)
        
        if self.use_adaptive:
            implicit_config = getattr(config, 'implicit', None)
            if implicit_config is None:
                # Default values if no config is provided
                self.implicit_solver = "anderson"
                self.implicit_iters = 20
                self.implicit_tol = 1e-5
                self.anderson_m = 5
                self.anderson_beta = 1.0
                self.anderson_lam = 1e-4
                self.use_adaptive_alpha = True
                self.init_alpha = 0.5
                self.min_alpha = 0.1
                self.max_alpha = 0.9
                self.use_progressive_tol = True
                self.init_tol = 1e-3
                self.final_tol = 1e-5
                self.tol_decay_steps = 5000
                self.use_warm_start = True
                self.warm_start_momentum = 0.9
            else:
                # Load from config
                self.implicit_solver = getattr(implicit_config, 'solver', 'anderson')
                self.implicit_iters = getattr(implicit_config, 'max_iterations', 20)
                self.implicit_tol = getattr(implicit_config, 'tolerance', 1e-5)
                self.anderson_m = getattr(implicit_config, 'anderson_m', 5)
                self.anderson_beta = getattr(implicit_config, 'anderson_beta', 1.0)
                self.anderson_lam = getattr(implicit_config, 'anderson_lambda', 1e-4)
                self.use_adaptive_alpha = getattr(implicit_config, 'use_adaptive_alpha', True)
                self.init_alpha = getattr(implicit_config, 'init_alpha', 0.5)
                self.min_alpha = getattr(implicit_config, 'min_alpha', 0.1)
                self.max_alpha = getattr(implicit_config, 'max_alpha', 0.9)
                self.use_progressive_tol = getattr(implicit_config, 'use_progressive_tol', True)
                self.init_tol = getattr(implicit_config, 'init_tol', 1e-3)
                self.final_tol = getattr(implicit_config, 'final_tol', 1e-5)
                self.tol_decay_steps = getattr(implicit_config, 'tol_decay_steps', 5000)
                self.use_warm_start = getattr(implicit_config, 'use_warm_start', True)
                self.warm_start_momentum = getattr(implicit_config, 'warm_start_momentum', 0.9)
        
            # Current adaptive state
            self.step_count = 0
            self.current_tol = self.init_tol
            self.prev_solution = None
            
            # Register buffer for the last fixed point (for warm starting)
            self.register_buffer('last_fixed_point', None)
        
        # Iterations tracking
        self.last_iteration_count = 0

    def forward(self, x, mask, t, cemd=0):
        """
        Forward pass that maintains the same interface as GCNdiff but uses
        adaptive implicit techniques internally when enabled.
        
        Args:
            x: Input tensor
            mask: Mask tensor
            t: Timestep tensor (compatibility with diffusion API)
            cemd: Additional embedding info (kept for compatibility)
        """
        if self.use_adaptive:
            # Update step count and tolerance for progressive schedule
            self.step_count += 1
            if self.use_progressive_tol:
                progress = min(1.0, self.step_count / self.tol_decay_steps)
                self.current_tol = self.init_tol * (1 - progress) + self.final_tol * progress
            
            # Choose the solver
            if self.implicit_solver == "anderson":
                return self.forward_anderson(x, mask, t)
            else:
                return self.forward_fixed_point(x, mask, t)
        else:
            # Fallback to standard diffusion approach
            return self.forward_standard(x, mask, t, cemd)
        
    def forward_standard(self, x, mask, t, cemd=0):
        """Standard GCNdiff forward pass for compatibility"""
        # Timestep embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        out = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out, temb)
        out = self.gconv_output(out, self.adj)
        return out
        
    def forward_fixed_point(self, x, mask, t):
        """
        Implicit fixed-point iteration with adaptive parameters
        """
        device = x.device
        self.adj = self.adj.to(device)
        
        # Get time embedding for compatibility
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        # Input processing
        out = self.gconv_input(x, self.adj)
        
        # Initialize fixed point state
        if self.use_warm_start and self.last_fixed_point is not None and x.size(0) == self.last_fixed_point.size(0):
            # Warm start with previous solution
            z = self.warm_start_momentum * self.last_fixed_point + (1 - self.warm_start_momentum) * out
        else:
            z = out.clone()
        
        # Adaptive alpha (relaxation parameter)
        alpha = self.init_alpha
        prev_residual = float('inf')
        
        # Find fixed point with adaptive relaxation
        iteration_count = 0
        for i in range(self.implicit_iters):
            iteration_count = i + 1
            
            # Store previous iteration
            z_prev = z.clone()
            
            # Apply one iteration of the model
            current = z_prev.clone()
            for j in range(self.n_layers):
                current = self.atten_layers[j](current, mask)
                current = self.gconv_layers[j](current, temb)
            
            # Calculate residual
            residual = torch.norm(current - z_prev)
            
            # Adjust alpha if using adaptive relaxation
            if self.use_adaptive_alpha and i > 0:
                if residual > prev_residual:
                    # Slow down if diverging
                    alpha = max(self.min_alpha, alpha * 0.9)
                else:
                    # Speed up if converging
                    alpha = min(self.max_alpha, alpha * 1.1)
            
            # Update with relaxation
            z = (1 - alpha) * z_prev + alpha * current
            
            # Store residual for next iteration
            prev_residual = residual
            
            # Check convergence
            error = torch.norm(z - z_prev) / (torch.norm(z_prev) + 1e-8)
            error_val = error.item()
            
            if error_val < self.current_tol:
                break
        
        # Save metrics and state
        self.last_iteration_count = iteration_count
        if self.use_warm_start:
            self.last_fixed_point = z.detach()
        
        # Output layer
        out = self.gconv_output(z, self.adj)
        return out
        
    def forward_anderson(self, x, mask, t):
        """
        Anderson acceleration for faster convergence
        """
        device = x.device
        self.adj = self.adj.to(device)
        
        # Get time embedding for compatibility
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        # Input processing
        out = self.gconv_input(x, self.adj)
        
        # Initialize fixed point state
        if self.use_warm_start and self.last_fixed_point is not None and x.size(0) == self.last_fixed_point.size(0):
            z = self.warm_start_momentum * self.last_fixed_point + (1 - self.warm_start_momentum) * out
        else:
            z = out.clone()
        
        batch_size, n_joints, feat_dim = z.shape
        
        # Initialize storage for Anderson acceleration
        m = min(self.anderson_m, self.implicit_iters)
        X = torch.zeros(m, batch_size * n_joints * feat_dim, device=device)
        F = torch.zeros(m, batch_size * n_joints * feat_dim, device=device)
        
        # Find fixed point with Anderson acceleration
        iteration_count = 0
        for i in range(self.implicit_iters):
            iteration_count = i + 1
            
            # Store previous z for convergence check
            z_prev = z.clone()
            
            # Apply model to get F(z)
            current = z.clone()
            for j in range(self.n_layers):
                current = self.atten_layers[j](current, mask)
                current = self.gconv_layers[j](current, temb)
            
            # Calculate residual: F(z) - z
            residual = current - z
            
            # Flatten for Anderson calculations
            z_flat = z.reshape(-1)
            residual_flat = residual.reshape(-1)
            
            # Store history
            if i < m:
                X[i] = z_flat
                F[i] = residual_flat
            else:
                # Shift history and add newest vectors
                X = torch.cat([X[1:], z_flat.unsqueeze(0)], dim=0)
                F = torch.cat([F[1:], residual_flat.unsqueeze(0)], dim=0)
            
            # Apply Anderson acceleration after collecting enough history
            if i >= 1:  # Need at least 2 points for meaningful acceleration
                n = min(i+1, m)
                
                # Calculate differences
                dX = X[:n] - X[n-1]  # (n, dim)
                dF = F[:n] - F[n-1]  # (n, dim)
                
                # Skip acceleration if differences are too small (numerical stability)
                if torch.norm(dF) < 1e-10:
                    z = z + self.anderson_beta * residual
                    continue
                
                # Compute coefficients for Anderson acceleration
                if n == 1:
                    alpha = torch.tensor([1.0], device=device)
                else:
                    # Regularized least squares problem: minimize ||dF @ alpha - (-F[n-1])||^2 + lambda * ||alpha||^2
                    gram = torch.matmul(dF, dF.t())  # (n, n)
                    reg = self.anderson_lam * torch.eye(n, device=device)  # Regularization
                    rhs = -torch.matmul(F[n-1], dF.t())  # (n,)
                    
                    try:
                        alpha = torch.linalg.solve(gram + reg, rhs)  # (n,)
                    except:
                        # Fallback if solve fails
                        alpha = torch.ones(n, device=device) / n
                
                # Ensure alpha sums to 1
                alpha = alpha / alpha.sum()
                
                # Compute new estimate
                new_z = torch.matmul(alpha, X[:n])
                new_f = torch.matmul(alpha, F[:n])
                
                # Update z with damping
                z_new = new_z.reshape(z.shape) + self.anderson_beta * new_f.reshape(residual.shape)
                z = z_new
            else:
                # Simple update for first iteration
                z = z + self.anderson_beta * residual
            
            # Check convergence against adaptive tolerance
            error = torch.norm(z - z_prev) / (torch.norm(z_prev) + 1e-8)
            error_val = error.item()
            
            if error_val < self.current_tol:
                break
        
        # Store for metrics and warm starting
        self.last_iteration_count = iteration_count
        if self.use_warm_start:
            self.last_fixed_point = z.detach()
        
        # Final output transformation
        out = self.gconv_output(z, self.adj)
        
        return out
    
    def reset_history(self):
        """Reset warm starting history (useful between epochs)"""
        if self.use_adaptive and self.use_warm_start:
            self.last_fixed_point = None
    
    # These are compatibility methods for integration with the diffusion framework
    def get_diffusion_output(self, x, mask, t, cemd=0):
        """
        Compatibility method for the diffusion sampling process.
        Returns output directly without diffusion steps.
        """
        return self.forward(x, mask, t, cemd)