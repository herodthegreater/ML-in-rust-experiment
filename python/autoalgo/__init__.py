"""autoalgo v0.2: a speed experiment, not a library yet.

Five algorithms, each built more than one way (on top of NumPy, in Rust, on
the Apple GPU with MLX), so they can be raced against scikit-learn. See
README.md and the notebooks/ folder for what the races show.
"""

from .cluster import KMeans
from .decomposition import PCA
from .linear_model import Ridge
from .neighbors import KNeighborsClassifier
from .tree import DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier", "KMeans", "KNeighborsClassifier", "PCA", "Ridge"]
__version__ = "0.2.0"
