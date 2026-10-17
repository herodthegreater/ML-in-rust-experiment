"""k-nearest neighbors classification with two interchangeable engines.

backend="numpy": built on top of existing libraries. Distances come from one
big matrix multiply, which NumPy hands to Apple Accelerate (or whatever BLAS
it was built with); neighbor selection uses np.argpartition.

backend="rust": a from-scratch compiled engine (src/lib.rs). Tiled distance
loops written for SIMD, every core via rayon, and the GIL released.

backend="rust-blas": the same Rust engine, except each tile of distance math
is one Apple Accelerate matrix multiply (macOS only). It separates "Rust is
faster" from "the chip's matrix hardware is faster".

All engines rank neighbors by ||x||^2 - 2 q.x, which orders points exactly
like squared Euclidean distance (the query's own norm is the same for every
candidate, so it is dropped). Ties in the vote go to the smallest class.
"""

import numpy as np

from . import _rust

# Cap on the temporary distance block the NumPy engine materializes per chunk.
_CHUNK_BYTES = 128 * 1024 * 1024

BACKENDS = ("numpy", "rust", "rust-blas")


def _pad(X, multiple=4):
    extra = -X.shape[1] % multiple
    if extra == 0:
        return X
    return np.ascontiguousarray(np.pad(X, ((0, 0), (0, extra))))


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, backend="rust"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.n_neighbors = n_neighbors
        self.backend = backend

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if not 1 <= self.n_neighbors <= len(X):
            raise ValueError(f"n_neighbors must be between 1 and {len(X)}")
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self._X = X
        self._y = y_idx.astype(np.int64)
        self._sq = np.einsum("ij,ij->i", X, X)
        if self.backend.startswith("rust"):
            # Zero columns change no distances but let the SIMD loop skip its scalar tail.
            self._X_padded = _pad(X)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._X.shape[1]:
            raise ValueError(f"expected {self._X.shape[1]} features, got {X.shape[1]}")
        if self.backend.startswith("rust"):
            idx = _rust.knn_predict(
                self._X_padded, self._sq, self._y, len(self.classes_), _pad(X), self.n_neighbors,
                use_blas=self.backend == "rust-blas",
            )
        else:
            idx = self._predict_numpy(X)
        return self.classes_[idx]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _predict_numpy(self, X):
        k = self.n_neighbors
        n_classes = len(self.classes_)
        out = np.empty(len(X), dtype=np.int64)
        chunk = max(1, _CHUNK_BYTES // (4 * len(self._X)))
        for start in range(0, len(X), chunk):
            q = X[start : start + chunk]
            d = q @ self._X.T
            d *= -2.0
            d += self._sq
            if k < d.shape[1]:
                nn = np.argpartition(d, k - 1, axis=1)[:, :k]
            else:
                nn = np.broadcast_to(np.arange(d.shape[1]), d.shape)
            votes = self._y[nn]
            counts = np.zeros((len(q), n_classes), dtype=np.int32)
            rows = np.arange(len(q))
            for j in range(votes.shape[1]):
                counts[rows, votes[:, j]] += 1
            out[start : start + chunk] = counts.argmax(axis=1)
        return out
