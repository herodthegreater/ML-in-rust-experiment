"""Quick KNN race: scikit-learn vs every this project engine on Covertype.

Run with the project's .venv interpreter after `maturin develop --release`.
Change TRAIN_ROWS to race at a different size (max 579,012).
"""

import time

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.neighbors import KNeighborsClassifier as SkKNN

from ml_in_rust import KNeighborsClassifier, available_backends

TRAIN_ROWS = 200_000
QUERIES = 2_000
K = 5

data = fetch_covtype(data_home="data")
X = data.data.astype(np.float32)
X = (X - X.mean(0)) / X.std(0)
order = np.random.default_rng(0).permutation(len(X))
X, y = X[order], data.target[order]
Xq, yq = X[:QUERIES], y[:QUERIES]
Xtr, ytr = X[QUERIES : QUERIES + TRAIN_ROWS], y[QUERIES : QUERIES + TRAIN_ROWS]

engines = {
    "sklearn": SkKNN(n_neighbors=K, n_jobs=-1),
    "numpy": KNeighborsClassifier(K, backend="numpy"),
    "rust": KNeighborsClassifier(K, backend="rust"),
    "rust-blas": KNeighborsClassifier(K, backend="rust-blas"),
    "mlx": KNeighborsClassifier(K, backend="mlx"),
}
engines = {k: v for k, v in engines.items() if k == "sklearn" or k in available_backends()}

print(f"{len(Xtr):,} training rows, {QUERIES:,} queries, k={K}\n")
print(f"{'engine':<20}{'predict':>12}{'vs sklearn':>12}{'accuracy':>10}")
baseline = None
for name, model in engines.items():
    model.fit(Xtr, ytr)
    model.predict(Xq)  # warm-up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pred = model.predict(Xq)
        times.append(time.perf_counter() - t0)
    best = min(times)
    baseline = baseline or best
    acc = np.mean(pred == yq)
    print(f"{name:<20}{best * 1e3:>10.1f}ms{baseline / best:>11.2f}x{acc:>10.4f}")
