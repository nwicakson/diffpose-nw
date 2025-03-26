import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def memory_efficient_attention(Q, K, V, mask=None, dropout=None):
    """
    Memory-efficient attention implementation that avoids storing the full attention matrix.
    Uses PyTorch's scaled_dot_product_attention if available (PyTorch 2.0+),
    otherwise falls back to a custom chunk-based implementation.
    
    Args:
        Q, K, V: Query, Key, Value tensors
        mask: Optional attention mask
        dropout: Optional dropout module
    
    Returns:
        attention output, attention weights
    """
    # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
    if hasattr(F, 'scaled_dot_product_attention'):
        # Use PyTorch's optimized implementation
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=dropout.p if dropout is not None else 0.0,
            is_causal=False
        )
        # For compatibility with the original implementation
        attn_weights = None  # PyTorch's implementation doesn't return weights
        return attn_output, attn_weights
    
    # Fall back to chunk-based implementation for older PyTorch versions
    d_k = Q.size(-1)
    batch_size, num_heads, seq_len, _ = Q.size()
    
    # Determine chunk size based on available memory
    # This is a heuristic - adjust chunk_size based on your GPU memory
    chunk_size = min(128, seq_len)
    
    attn_output = torch.zeros_like(V)
    attn_weights = None  # We won't compute the full attention matrix
    
    # Process in chunks to save memory
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        
        # Get current chunk of queries
        Q_chunk = Q[:, :, i:end_idx, :]
        
        # Compute attention scores for this chunk
        scores_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Adjust mask for current chunk
            if mask.dim() == 4:  # batch_size x num_heads x seq_len x seq_len
                mask_chunk = mask[:, :, i:end_idx, :]
            else:  # seq_len x seq_len
                mask_chunk = mask[i:end_idx, :]
            
            scores_chunk = scores_chunk.masked_fill(mask_chunk == 0, -1e9)
        
        # Apply softmax and dropout
        attn_chunk = F.softmax(scores_chunk, dim=-1)
        if dropout is not None:
            attn_chunk = dropout(attn_chunk)
        
        # Apply attention to values
        output_chunk = torch.matmul(attn_chunk, V)
        
        # Store in the output tensor
        attn_output[:, :, i:end_idx, :] = output_chunk
    
    return attn_output, attn_weights

class MemoryEfficientMultiHeadedAttention(nn.Module):
    """
    Memory-efficient implementation of multi-headed attention.
    Drop-in replacement for the original MultiHeadedAttention.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MemoryEfficientMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        
        # Linear projections
        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        
        # Apply attention using the memory-efficient implementation
        x, self.attn = memory_efficient_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # Concatenate heads and run through final linear layer
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)