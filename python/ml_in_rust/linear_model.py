"""Ridge regression with two engines, and why there is no Rust one.

Ridge has a closed-form answer: solve (X^T X + alpha I) w = X^T y. Nearly all
the work is the single n x d by d x n multiply that builds X^T X; the solve
afterwards is a tiny d x d problem. A Rust engine would just call the same
BLAS multiply NumPy uses, so there is nothing for compiled code to add.

backend="numpy": X^T X on NumPy's BLAS, centering X one cache-sized chunk at a
time (never a full copy) and adding chunks into a float64 total.
backend="mlx": the same, with the big multiply on the Apple GPU.
"""

import numpy as np

from ._common import as_float32_2d, centered_stats, check_backend, mlx

BACKENDS = ("numpy", "mlx")


class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True, backend="numpy"):
        check_backend(backend, BACKENDS)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.backend = backend

    def fit(self, X, y):
        X = as_float32_2d(X)
        y = np.asarray(y, dtype=np.float32).ravel()
        if self.fit_intercept:
            gram, mean, xty, y_mean = centered_stats(X, y, self.backend)
        else:
            gram, xty = _raw_stats(X, y, self.backend)
        gram[np.diag_indices_from(gram)] += self.alpha
        coef = np.linalg.solve(gram, xty)
        self.coef_ = coef.astype(np.float32)
        self.intercept_ = float(y_mean - mean @ coef) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        return as_float32_2d(X) @ self.coef_ + self.intercept_


def _raw_stats(X, y, backend):
    if backend == "mlx":
        mx = mlx()
        Xm = mx.array(X)
        g, xty = Xm.T @ Xm, Xm.T @ mx.array(y)
        mx.eval(g, xty)
        return np.array(g, dtype=np.float64), np.array(xty, dtype=np.float64)
    return (X.T @ X).astype(np.float64), (X.T @ y).astype(np.float64)
