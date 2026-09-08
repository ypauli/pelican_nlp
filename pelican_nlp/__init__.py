import os

# Must be set before the CUDA allocator starts. Harmless if CUDA is unused.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Version and metadata
from ._version import __version__
__author__ = "Yves Pauli"


def __getattr__(name):
    if name == "Pelican":
        from .main import Pelican
        return Pelican
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Pelican", "__version__"]
