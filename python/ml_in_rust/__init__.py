"""ml_in_rust: a research codebase, not a library.

Five algorithms, each built more than one way (on top of NumPy, in Rust, on
the Apple GPU with MLX), so they can be raced against scikit-learn. See
README.md and the notebooks/ folder for what the races show.

This project began as the first version of autoalgo
(github.com/herodthegreater/autoalgo, tag speed-study-v0.2).
"""

from . import _rust
from ._common import mlx_available
from .cluster import KMeans
from .decomposition import PCA
from .linear_model import Ridge
from .neighbors import KNeighborsClassifier
from .tree import DecisionTreeClassifier

__all__ = [
    "DecisionTreeClassifier", "KMeans", "KNeighborsClassifier", "PCA", "Ridge",
    "available_backends", "gemm_library",
]


def available_backends():
    """Engines this machine can run. MLX is optional (pip install mlx)."""
    return ["numpy", "rust", "rust-blas"] + (["mlx"] if mlx_available() else [])


def gemm_library():
    """Matrix library behind the Rust "-blas" paths: Apple Accelerate on macOS, matrixmultiply elsewhere."""
    return _rust.gemm_library()
__version__ = "0.2.0"
