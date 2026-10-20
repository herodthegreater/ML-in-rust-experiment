"""PCA through the covariance matrix, with two engines (and no Rust one).

For tall data (many more rows than features) PCA is: build the d x d
covariance matrix with one big multiply, then take its eigenvectors, which is
cheap because d is small. Like Ridge, the cost is one multiply that a Rust
engine could only hand to the same BLAS routine NumPy uses.

backend="numpy": X^T X on NumPy's BLAS, centering X one cache-sized chunk at a
time (never a full copy) and adding chunks into a float64 total.
backend="mlx": the same, with the big multiply on the Apple GPU.

Component signs follow scikit-learn's convention for this solver: each
component's largest-magnitude entry is positive.
"""

import numpy as np

from ._common import as_float32_2d, centered_stats, check_backend

BACKENDS = ("numpy", "mlx")


class PCA:
    def __init__(self, n_components=None, backend="numpy"):
        check_backend(backend, BACKENDS)
        self.n_components = n_components
        self.backend = backend

    def fit(self, X, y=None):
        X = as_float32_2d(X)
        n, d = X.shape
        k = d if self.n_components is None else self.n_components
        if not 1 <= k <= d:
            raise ValueError(f"n_components must be between 1 and {d}")
        gram, mean, _, _ = centered_stats(X, backend=self.backend)
        cov = gram / (n - 1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1][:k]
        components = vectors[:, order].T
        flip = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
        components *= flip[:, None]

        self.mean_ = mean.astype(np.float32)
        self.components_ = components.astype(np.float32)
        self.explained_variance_ = np.maximum(values[order], 0)
        total = np.maximum(values, 0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.n_components_ = k
        return self

    def transform(self, X):
        return (as_float32_2d(X) - self.mean_) @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
