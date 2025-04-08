# Version and metadata
__version__ = "0.1.0"
__author__ = "Yves Pauli"

try:
    from .main import Pelican
except ImportError as e:
    print(f"Warning: Could not import Pelican class: {e}")

