from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy, math
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.egraformer import *

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
        self.temb_proj = torch.nn.Linear(emd_dim, hid_dim)

    def forward(self, x, temb):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out

class IGCN(nn.Module):
    """
    An implicit version of GCNdiff with chunked attention processing
    to reduce memory usage while maintaining accuracy.
    """
    def __init__(self, adj, config):
        super(IGCN, self).__init__()
        
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
        attn = MemoryEfficientMultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)
        
        # Add batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(self.hid_dim)
        
        ### diffusion configuration (compatibility layer) ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim, self.emd_dim),
            torch.nn.Linear(self.emd_dim, self.emd_dim),
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
                self.anderson_lambda = 1e-4
                self.use_warm_start = False  # Warm start OFF by default
                self.warm_start_momentum = 0.5
                self.chunk_size = 256  # Default chunk size for attention
            else:
                # Load from config
                self.implicit_solver = getattr(implicit_config, 'solver', 'anderson')
                self.implicit_iters = getattr(implicit_config, 'max_iterations', 20)
                self.implicit_tol = getattr(implicit_config, 'tolerance', 1e-5)
                self.anderson_m = getattr(implicit_config, 'anderson_m', 5)
                self.anderson_beta = getattr(implicit_config, 'anderson_beta', 1.0)
                self.anderson_lambda = getattr(implicit_config, 'anderson_lambda', 1e-4)
                self.use_warm_start = getattr(implicit_config, 'use_warm_start', False)
                self.warm_start_momentum = getattr(implicit_config, 'warm_start_momentum', 0.5)
                self.chunk_size = getattr(implicit_config, 'chunk_size', 256)
                
            # Register buffer for the last fixed point (for warm starting)
            self.register_buffer('last_fixed_point', None)
        
        # Iterations tracking
        self.last_iteration_count = 0

    def _set_batchnorm_eval(self):
        """Set all BatchNorm layers to eval mode"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()
                
    def _set_batchnorm_train(self):
        """Set all BatchNorm layers back to train mode"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.train()

    def process_attention_in_chunks(self, x, mask, layer_idx):
        """
        Process attention in chunks to save memory
        
        Args:
            x: Input tensor
            mask: Attention mask
            layer_idx: Index of the attention layer to use
        """
        # Get appropriate attention layer
        layer = self.atten_layers[layer_idx]
        
        # If batch size is small enough, process directly
        if x.size(0) <= self.chunk_size:
            return layer(x, mask)
            
        # Save original batch norm state
        training = self.training
        
        # Process in chunks
        outputs = []
        for i in range(0, x.size(0), self.chunk_size):
            # Get current chunk
            chunk = x[i:i+self.chunk_size]
            
            # Handle mask - expand if needed
            chunk_mask = mask.expand(chunk.size(0), -1, -1) if mask.size(0) == 1 else mask[i:i+self.chunk_size]
            
            # Process chunk
            outputs.append(layer(chunk, chunk_mask))
            
            # Free memory
            if i % (self.chunk_size * 2) == 0 and i > 0:
                torch.cuda.empty_cache()
                
        # Combine chunks
        return torch.cat(outputs, dim=0)

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
            # Choose the solver
            if self.implicit_solver == "anderson":
                return self.forward_anderson_optimized(x, mask, t)
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
            out = self.process_attention_in_chunks(out, mask, i)
            out = self.gconv_layers[i](out, temb)
        out = self.gconv_output(out, self.adj)
        return out
        
    def forward_fixed_point(self, x, mask, t):
        """
        Implicit fixed-point iteration with chunked processing
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
        
        # Initialize fixed point state with optional warm starting
        if self.use_warm_start and self.last_fixed_point is not None and x.size(0) == self.last_fixed_point.size(0):
            z = self.warm_start_momentum * self.last_fixed_point + (1 - self.warm_start_momentum) * out
        else:
            z = out.clone()
        
        # Find fixed point with iteration
        iteration_count = 0
        min_iterations = 10  # Force at least this many iterations
        for i in range(self.implicit_iters):
            iteration_count = i + 1
            
            # Store previous iteration
            z_prev = z.clone()
            
            # Apply one iteration of the model - using chunked attention
            current = z_prev.clone()
            for j in range(self.n_layers):
                # Process attention in chunks to save memory
                current = self.process_attention_in_chunks(current, mask, j)
                # Apply graph convolution
                current = self.gconv_layers[j](current, temb)
            
            # Apply batch normalization for stability
            current_shape = current.shape
            current_flat = current.reshape(-1, self.hid_dim)
            current_flat = self.batch_norm(current_flat)
            current = current_flat.reshape(current_shape)
            
            # Simple update with fixed relaxation parameter
            alpha = 0.5  # Fixed relaxation parameter
            z = (1 - alpha) * z_prev + alpha * current
            
            # Check convergence (only after minimum iterations)
            if i >= min_iterations:
                error = torch.norm(z - z_prev) / (torch.norm(z_prev) + 1e-8)
                if error < self.implicit_tol:
                    break
                    
            # Periodically free memory
            if i % 5 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        # Save metrics and state
        self.last_iteration_count = iteration_count
        if self.use_warm_start:
            self.last_fixed_point = z.detach()
        
        # Output layer
        out = self.gconv_output(z, self.adj)
        return out
    
    def forward_anderson_optimized(self, x, mask, t):
        """
        Optimized Anderson acceleration with chunked processing for memory efficiency
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
        
        # Initialize fixed point state with optional warm starting
        if self.use_warm_start and self.last_fixed_point is not None and x.size(0) == self.last_fixed_point.size(0):
            z = self.warm_start_momentum * self.last_fixed_point + (1 - self.warm_start_momentum) * out
        else:
            z = out.clone()
        
        batch_size, n_joints, feat_dim = z.shape
        
        # Initialize storage for Anderson acceleration
        m = min(self.anderson_m, self.implicit_iters)
        flat_dim = batch_size * n_joints * feat_dim
        X = torch.zeros(m, flat_dim, device=device)
        F = torch.zeros(m, flat_dim, device=device)
        
        # Precompute model features for first iteration to save time
        current = z.clone()
        for j in range(self.n_layers):
            # Process attention in chunks to save memory
            current = self.process_attention_in_chunks(current, mask, j)
            # Apply graph convolution
            current = self.gconv_layers[j](current, temb)
        
        # Apply batch normalization for stability
        current_shape = current.shape
        current_flat = current.reshape(-1, self.hid_dim)
        current_flat = self.batch_norm(current_flat)
        current = current_flat.reshape(current_shape)
        
        # Find fixed point with Anderson acceleration
        iteration_count = 0
        min_iterations = 10  # Force at least this many iterations
        for i in range(self.implicit_iters):
            iteration_count = i + 1
            
            # Store previous z for convergence check
            z_prev = z.clone()
            
            # Calculate residual: F(z) - z from precomputed result
            residual = current - z
            
            # Flatten for Anderson calculations
            z_flat = z.reshape(-1)
            residual_flat = residual.reshape(-1)
            
            # Store history efficiently
            if i < m:
                X[i] = z_flat
                F[i] = residual_flat
            else:
                # Fast update for history storage
                X = torch.cat([X[1:], z_flat.unsqueeze(0)], dim=0)
                F = torch.cat([F[1:], residual_flat.unsqueeze(0)], dim=0)
            
            # Apply Anderson acceleration after collecting enough history
            if i >= 1:  # Need at least 2 points for meaningful acceleration
                n = min(i+1, m)
                
                # Calculate differences
                dX = X[:n] - X[n-1]  # (n, dim)
                dF = F[:n] - F[n-1]  # (n, dim)
                
                # Skip acceleration if differences are too small (numerical stability)
                dF_norm = torch.norm(dF)
                if dF_norm < 1e-10:
                    z = z + self.anderson_beta * residual
                else:
                    # Compute coefficients for Anderson acceleration
                    if n == 1:
                        alpha = torch.tensor([1.0], device=device)
                    else:
                        # Solve least squares problem
                        gram = torch.matmul(dF, dF.t())  # (n, n)
                        reg = self.anderson_lambda * torch.eye(n, device=device)  # Regularization
                        rhs = -torch.matmul(F[n-1], dF.t())  # (n,)
                        
                        try:
                            # Use stable solver
                            alpha = torch.linalg.solve(gram + reg, rhs)  # (n,)
                        except:
                            # Fallback
                            alpha = torch.ones(n, device=device) / n
                        
                        # Ensure alpha sums to 1
                        alpha_sum = alpha.sum()
                        if abs(alpha_sum) > 1e-10:  # Avoid division by zero
                            alpha = alpha / alpha_sum
                        else:
                            alpha = torch.ones(n, device=device) / n
                        
                        # Compute new estimate with Anderson mixing
                        new_z = torch.matmul(alpha, X[:n])
                        new_f = torch.matmul(alpha, F[:n])
                        z = new_z.reshape(z.shape) + self.anderson_beta * new_f.reshape(residual.shape)
            else:
                # Simple update for first iteration
                z = z + self.anderson_beta * residual
            
            # Precompute next iteration model features with chunked attention
            current = z.clone()
            for j in range(self.n_layers):
                # Process attention in chunks to save memory
                current = self.process_attention_in_chunks(current, mask, j)
                # Apply graph convolution
                current = self.gconv_layers[j](current, temb)
            
            # Apply batch normalization for stability
            current_shape = current.shape
            current_flat = current.reshape(-1, self.hid_dim)
            current_flat = self.batch_norm(current_flat)
            current = current_flat.reshape(current_shape)
            
            # Check convergence (only after minimum iterations)
            if i >= min_iterations:
                error = torch.norm(z - z_prev) / (torch.norm(z_prev) + 1e-8)
                if error < self.implicit_tol:
                    break
            
            # Periodically free memory
            if i % 3 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        # Store for metrics and warm starting
        self.last_iteration_count = iteration_count
        if self.use_warm_start:
            self.last_fixed_point = z.detach()
        
        # Final output transformation
        out = self.gconv_output(z, self.adj)
        
        return out
    
    def reset_history(self):
        """Reset warm starting history (useful between epochs)"""
        if self.use_warm_start:
            self.last_fixed_point = None
            
        # Clear any temporary tensors
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def optimize_memory(self):
        """
        Optimize memory usage by clearing unnecessary stored tensors
        """
        # Only keep the most recent fixed point if using warm start
        if self.use_warm_start and hasattr(self, 'last_fixed_point') and self.last_fixed_point is not None:
            # Ensure it's not keeping computational graph
            self.last_fixed_point = self.last_fixed_point.detach()
            
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # These are compatibility methods for integration with the diffusion framework
    def get_diffusion_output(self, x, mask, t, cemd=0):
        """
        Compatibility method for the diffusion sampling process.
        Returns output directly without diffusion steps.
        """
        return self.forward(x, mask, t, cemd)