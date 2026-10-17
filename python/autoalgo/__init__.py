"""autoalgo v0.1: a speed experiment, not a library yet.

One algorithm (k-nearest neighbors) built two ways so it can be raced against
scikit-learn. See README.md for the question this answers.
"""

from .neighbors import KNeighborsClassifier

__all__ = ["KNeighborsClassifier"]
__version__ = "0.1.0"
