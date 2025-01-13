def get_device(skip_mps: bool = False) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        skip_mps: If True, skip MPS even if available (for operations that don't support it)
        
    Returns:
        torch.device: Best available device (cuda > mps > cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif not skip_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu") 