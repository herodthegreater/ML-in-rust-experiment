import numpy as np


def mlx():
    """Import MLX lazily so it stays an optional dependency."""
    try:
        import mlx.core as mx
    except ImportError as e:
        raise ImportError("backend='mlx' needs MLX on Apple silicon: pip install mlx") from e
    return mx


def check_backend(backend, allowed):
    if backend not in allowed:
        raise ValueError(f"backend must be one of {allowed}, got {backend!r}")


def as_float32_2d(X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    return X


def centered_stats(X, y=None, backend="numpy"):
    """Return (Xc^T Xc, mean, Xc^T yc, y_mean) in float64, where Xc is X centered.

    PCA and Ridge only need these small d x d and d-length results, so X is
    never copied as a whole. NumPy engine: X is centered one cache-sized chunk
    at a time, each chunk's float32 multiply is added into a float64 total.
    Centering before multiplying avoids the precision loss of the shortcut
    X^T X - n * mean mean^T, and the float64 running total keeps long sums
    accurate. MLX engine: the multiply runs on the GPU in one call.
    """
    n, d = X.shape
    mean = X.sum(axis=0, dtype=np.float64) / n
    y_mean = float(y.sum(dtype=np.float64) / n) if y is not None else None
    mu = mean.astype(np.float32)
    # mu is the mean rounded to float32; centering by it leaves a tiny offset
    # delta, which adds n * delta delta^T to the result. Removed at the end.
    delta = mean - mu
    if backend == "mlx":
        mx = mlx()
        Xc = mx.array(X) - mx.array(mu)
        g = Xc.T @ Xc
        xty = Xc.T @ (mx.array(y) - y_mean) if y is not None else None
        mx.eval(g, *([xty] if xty is not None else []))
        gram = np.array(g, dtype=np.float64) - n * np.outer(delta, delta)
        return gram, mean, np.array(xty, dtype=np.float64) if xty is not None else None, y_mean

    rows = max(4096, (32 * 1024 * 1024) // (4 * d))
    gram = np.zeros((d, d))
    xty = np.zeros(d) if y is not None else None
    for s in range(0, n, rows):
        c = X[s : s + rows] - mu
        gram += c.T @ c
        if y is not None:
            xty += c.T @ (y[s : s + rows] - np.float32(y_mean))
    gram -= n * np.outer(delta, delta)
    return gram, mean, xty, y_mean
