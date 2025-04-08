# Version and metadata
from ._version import __version__
__author__ = "Yves Pauli"

try:
    from .main import Pelican
except ImportError as e:
    print(f"Warning: Could not import Pelican class: {e}")

