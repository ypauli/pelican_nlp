from typing import Optional
import torch

class ForcedAligner:
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the forced aligner.
        
        Args:
            device: Device to use for inference (default: best available excluding MPS)
        """
        # Force skip MPS as it's not supported for alignment
        self.device = device if device is not None else get_device(skip_mps=True)
        print(f"Initializing ForcedAligner on device: {self.device}") 