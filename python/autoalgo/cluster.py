"""k-means clustering (Lloyd's algorithm) with three engines.

backend="numpy": built on top of NumPy. Distances to the centroids are one
matrix multiply on Accelerate; centroid sums use np.bincount per feature.

backend="rust": the whole loop runs in Rust (src/kmeans.rs). Each block of
rows is scored with one Accelerate multiply, assigned, and summed in the same
pass, across every core.

backend="mlx": pure Python on the Apple GPU. Distances and centroid sums are
both matrix multiplies (sums via a one-hot matrix), so everything stays on
the GPU; only the "did anything change?" check comes back to the CPU. The GPU
sums in float32 (the CPU engines use float64), so on large data a few points
on a cluster boundary can land differently.

All engines run the same algorithm as scikit-learn's KMeans(algorithm="lloyd",
n_init=1, tol=0): stop when no label changes or after max_iter iterations.
"""

import numpy as np

from . import _rust
from ._common import as_float32_2d, check_backend, mlx

BACKENDS = ("numpy", "rust", "mlx")


class KMeans:
    def __init__(self, n_clusters=8, init="k-means++", max_iter=300, random_state=None, backend="rust"):
        check_backend(backend, BACKENDS)
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.backend = backend

    def fit(self, X, y=None):
        X = as_float32_2d(X)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError(f"n_clusters must be between 1 and {len(X)}")
        if isinstance(self.init, str):
            if self.init != "k-means++":
                raise ValueError("init must be 'k-means++' or an array of centers")
            centers = _kmeans_plusplus(X, self.n_clusters, np.random.default_rng(self.random_state))
        else:
            centers = np.array(self.init, dtype=np.float32, copy=True, order="C")
            if centers.shape != (self.n_clusters, X.shape[1]):
                raise ValueError("init must have shape (n_clusters, n_features)")

        run = {"rust": _rust.kmeans_lloyd, "numpy": _lloyd_numpy, "mlx": _lloyd_mlx}[self.backend]
        centers, labels, n_iter, inertia = run(X, centers, self.max_iter)
        self.cluster_centers_ = np.asarray(centers, dtype=np.float32)
        self.labels_ = np.asarray(labels, dtype=np.int64)
        self.n_iter_ = int(n_iter)
        self.inertia_ = float(inertia)
        return self

    def predict(self, X):
        X = as_float32_2d(X)
        c = self.cluster_centers_
        return (np.einsum("ij,ij->i", c, c) - 2.0 * (X @ c.T)).argmin(axis=1)


def _kmeans_plusplus(X, k, rng):
    """Greedy-free k-means++: each new center is drawn proportional to squared distance."""
    centers = [X[rng.integers(len(X))]]
    closest = ((X - centers[0]) ** 2).sum(axis=1, dtype=np.float64)
    for _ in range(1, k):
        total = closest.sum()
        i = rng.choice(len(X), p=closest / total) if total > 0 else rng.integers(len(X))
        centers.append(X[i])
        closest = np.minimum(closest, ((X - X[i]) ** 2).sum(axis=1, dtype=np.float64))
    return np.array(centers, dtype=np.float32)


def _lloyd_numpy(X, centers, max_iter):
    n, d = X.shape
    k = len(centers)
    x_sq = np.einsum("ij,ij->i", X, X)
    cols = np.ascontiguousarray(X.T)  # one feature per row, for fast per-feature sums
    rows = np.arange(n)

    def assign(c):
        s = np.einsum("ij,ij->i", c, c) - 2.0 * (X @ c.T)
        lab = s.argmin(axis=1)
        inertia = np.maximum(s[rows, lab] + x_sq, 0).sum(dtype=np.float64)
        return lab, float(inertia)

    labels = np.full(n, -1)
    converged = False
    for it in range(max_iter):
        new, inertia = assign(centers)
        changed = np.count_nonzero(new != labels)
        labels = new
        n_iter = it + 1
        if it > 0 and changed == 0:
            converged = True
            break
        counts = np.bincount(labels, minlength=k)
        sums = np.stack([np.bincount(labels, weights=cols[f], minlength=k) for f in range(d)], axis=1)
        keep = counts > 0
        centers = centers.copy()
        centers[keep] = (sums[keep] / counts[keep, None]).astype(np.float32)
    if not converged:
        labels, inertia = assign(centers)
    return centers, labels, n_iter, inertia


def _lloyd_mlx(X, centers, max_iter):
    mx = mlx()
    n = len(X)
    k = len(centers)
    Xm = mx.array(X)
    x_sq = (Xm * Xm).sum(axis=1)
    C = mx.array(centers)
    ids = mx.arange(k)

    def assign(c):
        s = (c * c).sum(axis=1) - 2.0 * (Xm @ c.T)
        lab = mx.argmin(s, axis=1)
        best = mx.take_along_axis(s, lab[:, None], axis=1)[:, 0]
        inertia = mx.maximum(best + x_sq, 0).sum()
        return lab, inertia

    labels = mx.full((n,), k, dtype=mx.uint32)  # k is never a real label
    converged = False
    for it in range(max_iter):
        new, inertia = assign(C)
        changed = (new != labels).sum()
        mx.eval(new, changed, inertia)
        labels = new
        n_iter = it + 1
        if it > 0 and changed.item() == 0:
            converged = True
            break
        onehot = (labels[:, None] == ids).astype(mx.float32)
        sums = onehot.T @ Xm
        counts = onehot.sum(axis=0)[:, None]
        C = mx.where(counts > 0, sums / mx.maximum(counts, 1), C)
    if not converged:
        labels, inertia = assign(C)
    mx.eval(C, labels, inertia)
    return np.array(C), np.array(labels).astype(np.int64), n_iter, inertia.item()
