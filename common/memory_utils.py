import logging
import torch
import numpy as np

def get_dynamic_chunk_size(model, config, device, test_input=None, mask=None):
    """
    Determine the optimal chunk size based on available GPU memory.
    
    Args:
        model: The model being used
        config: Configuration object
        device: Device to use
        test_input: Optional test input tensor for more precise measurement
        mask: Optional attention mask for model
        
    Returns:
        optimal_chunk_size: The optimal chunk size for processing
    """
    # Get user-defined limits from config or set defaults
    min_size = getattr(config, 'min_chunk_size', 32)
    max_size = getattr(config, 'max_chunk_size', 1024)
    target_usage = getattr(config, 'target_memory_usage', 0.75)
    
    # Return default for CPU
    if not torch.cuda.is_available():
        logging.info(f"Dynamic chunk sizing not available on CPU. Using default: {min_size}")
        return min_size
    
    try:
        # Clear cache and get memory stats
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        total_mem = torch.cuda.get_device_properties(device).total_memory
        reserved_mem = torch.cuda.memory_reserved(device)
        allocated_mem = torch.cuda.memory_allocated(device)
        
        # Calculate available memory
        free_mem = total_mem - reserved_mem
        usable_mem = free_mem * target_usage
        
        # Calculate chunk size based on model size or test input
        if test_input is not None and mask is not None:
            # Use actual test input for precise measurement
            chunk_size = estimate_chunk_size_with_test(model, test_input, mask, usable_mem, 
                                                      min_size, max_size, device)
        else:
            # Use model parameters as a heuristic
            chunk_size = estimate_chunk_size_from_params(model, usable_mem, min_size, max_size)
            
        # Apply bounds and rounding for hardware efficiency
        chunk_size = max(min_size, min(chunk_size, max_size))
        chunk_size = (chunk_size // 8) * 8  # Round to multiple of 8
        
        logging.info(f"Dynamic chunk size determined: {chunk_size} (Memory: {free_mem/1e9:.2f}GB free, {usable_mem/1e9:.2f}GB usable)")
        return chunk_size
        
    except Exception as e:
        logging.warning(f"Error determining dynamic chunk size: {e}")
        logging.warning(f"Falling back to minimum chunk size: {min_size}")
        return min_size  # Conservative fallback

def estimate_chunk_size_from_params(model, usable_mem, min_size, max_size):
    """Estimate chunk size based on model parameters"""
    # Get model size in bytes
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Heuristic: each sample in forward pass needs ~5-10x parameter memory
    # This is very model dependent, so we use a conservative estimate
    estimated_bytes_per_sample = model_size_bytes * 8
    
    # Calculate chunk size (with safety margin)
    chunk_size = int(usable_mem / (estimated_bytes_per_sample * 1.5))
    return chunk_size

def estimate_chunk_size_with_test(model, test_input, mask, usable_mem, min_size, max_size, device):
    """Estimate chunk size by running a small test batch"""
    # Start with a small test size
    test_size = min(16, min_size)
    
    # Create a small batch for testing
    if test_input.shape[0] >= test_size:
        small_batch = test_input[:test_size].clone()
        small_mask = mask.expand(test_size, -1, -1) if mask.size(0) == 1 else mask[:test_size].clone()
    else:
        # If test_input is too small, repeat it
        repeats = (test_size + test_input.shape[0] - 1) // test_input.shape[0]
        small_batch = test_input.repeat(repeats, 1, 1)[:test_size]
        small_mask = mask.expand(test_size, -1, -1) if mask.size(0) == 1 else mask.repeat(repeats, 1, 1)[:test_size]
    
    # Measure memory usage for this small batch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device)
    
    # Run a test forward pass
    with torch.no_grad():
        t = torch.ones(test_size, device=device)
        _ = model(small_batch, small_mask, t)
    
    torch.cuda.synchronize()
    mem_after = torch.cuda.max_memory_allocated(device)
    
    # Calculate memory per sample
    mem_per_sample = (mem_after - mem_before) / test_size
    
    # Calculate chunk size with safety margin (0.8)
    chunk_size = int(usable_mem / (mem_per_sample * 1.2))
    
    return chunk_size

def adaptive_chunk_processing(model, inputs, mask, t, process_fn, max_chunk_size=None, device=None):
    """
    Process inputs in adaptively-sized chunks based on current memory conditions
    
    Args:
        model: The model to run
        inputs: Input tensor of shape [batch_size, ...]
        mask: Attention mask
        t: Timestep tensor
        process_fn: Function that processes a chunk (takes chunk, mask, t)
        max_chunk_size: Maximum allowed chunk size
        device: Device to use
        
    Returns:
        Concatenated outputs from all chunks
    """
    if not torch.cuda.is_available():
        # If not on GPU, process everything at once
        return process_fn(inputs, mask, t)
    
    if device is None:
        device = inputs.device
    
    batch_size = inputs.shape[0]
    if max_chunk_size is None:
        max_chunk_size = batch_size
    
    # Get current memory stats
    torch.cuda.synchronize()
    total_mem = torch.cuda.get_device_properties(device).total_memory
    current_mem = torch.cuda.memory_allocated(device)
    reserved_mem = torch.cuda.memory_reserved(device)
    
    # Calculate available memory
    available_mem = total_mem - reserved_mem
    target_available = available_mem * 0.7  # Use 70% of available memory
    
    # Estimate memory per sample (very rough heuristic)
    # This would be more accurate with a test batch, but we want to avoid the overhead
    sample_mem = 4 * sum(p.numel() * p.element_size() for p in model.parameters()) / 100
    
    # Calculate adaptive chunk size
    adaptive_size = int(target_available / sample_mem)
    chunk_size = min(adaptive_size, max_chunk_size, batch_size)
    chunk_size = max(1, chunk_size)  # Ensure at least 1
    
    # Round to multiple of 8 for efficiency
    chunk_size = (chunk_size // 8) * 8
    if chunk_size == 0:
        chunk_size = 8  # Minimum chunk size
    
    # Process in chunks
    outputs = []
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        # Use clone to avoid in-place op issues
        chunk = inputs[i:end_idx].clone()
        
        # Handle mask expansion if needed
        if mask.size(0) == 1:
            chunk_mask = mask.expand(end_idx - i, -1, -1)
        else:
            chunk_mask = mask[i:end_idx].clone()
            
        # Handle timestep tensor
        if isinstance(t, torch.Tensor) and t.size(0) > 1:
            chunk_t = t[i:end_idx].clone()
        else:
            chunk_t = t
        
        # Process chunk
        outputs.append(process_fn(chunk, chunk_mask, chunk_t))
        
        # Free memory
        torch.cuda.empty_cache()
    
    # Combine outputs
    return torch.cat(outputs, dim=0)