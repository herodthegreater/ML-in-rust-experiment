"""Histogram decision tree classifier with two engines.

Each feature is cut into at most 256 bins (edges from quantiles, or
midpoints between values when a feature has few distinct values). A split
is then "bin <= t", and finding the best one only needs (bin, class) counts.
This is the trick LightGBM and scikit-learn's HistGradientBoosting use.

backend="numpy": built on top of NumPy. One np.bincount per node builds the
histogram for every feature at once; the sweep over cut points is vectorized.
The tree itself is grown by a Python recursion, one call per node.

backend="rust": the same algorithm in Rust (src/tree.rs), with features
scored in parallel at large nodes.

Prediction compares raw values against the stored bin edges ("bin <= t" is
the same test as "value < edge t"), so new data is never binned. The split rule matches the NumPy engine
exactly, so both engines grow identical trees.

There is no MLX engine: tree growing is branchy, uneven work that GPUs and
matrix hardware can't speed up.
"""

import numpy as np

from . import _rust
from ._common import check_backend

BACKENDS = ("numpy", "rust")
_BINS = 256
_MIN_GAIN = 1e-9
_EDGE_SAMPLE = 200_000


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_leaf=1, backend="rust", random_state=0):
        check_backend(backend, BACKENDS)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.backend = backend
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        y_idx = y_idx.astype(np.int64)
        self._edges = _bin_edges(X, np.random.default_rng(self.random_state))
        Xb = _apply_bins(X, self._edges)
        depth = 2**31 if self.max_depth is None else int(self.max_depth)
        n_classes = len(self.classes_)

        if self.backend == "rust":
            feature, threshold, left, right, counts = _rust.tree_fit(
                Xb, y_idx, n_classes, depth, self.min_samples_leaf
            )
        else:
            feature, threshold, left, right, counts = _grow_numpy(
                Xb, y_idx, n_classes, depth, self.min_samples_leaf
            )
        self.tree_ = {
            "feature": np.asarray(feature, dtype=np.int64),
            "threshold": np.asarray(threshold, dtype=np.int64),
            "left": np.asarray(left, dtype=np.int64),
            "right": np.asarray(right, dtype=np.int64),
            "counts": np.asarray(counts, dtype=np.int64),
        }
        self._leaf_class = self.tree_["counts"].argmax(axis=1).astype(np.int64)
        # "bin <= t" is the same test as "value < edge t", so store the edge and
        # predict on raw values without binning the input.
        f, thr = self.tree_["feature"], self.tree_["threshold"]
        self._cut = np.array(
            [self._edges[fi][ti] if fi >= 0 else np.inf for fi, ti in zip(f, thr)], dtype=np.float32
        )
        return self

    @property
    def node_count(self):
        return len(self.tree_["feature"])

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self._edges):
            raise ValueError(f"expected {len(self._edges)} features, got {X.shape[1]}")
        t = self.tree_
        if self.backend == "rust":
            idx = _rust.tree_predict(X, t["feature"], self._cut, t["left"], t["right"], self._leaf_class)
        else:
            idx = _predict_numpy(X, t["feature"], self._cut, t["left"], t["right"], self._leaf_class)
        return self.classes_[idx]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def _bin_edges(X, rng):
    sample = X if len(X) <= _EDGE_SAMPLE else X[rng.choice(len(X), _EDGE_SAMPLE, replace=False)]
    edges = []
    for f in range(X.shape[1]):
        values = np.unique(sample[:, f])
        if len(values) <= _BINS:
            edges.append((values[:-1] + values[1:]) / 2)
        else:
            qs = np.quantile(sample[:, f], np.linspace(0, 1, _BINS + 1)[1:-1])
            edges.append(np.unique(qs.astype(np.float32)))
    return edges


def _apply_bins(X, edges):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(edges):
        raise ValueError(f"expected {len(edges)} features, got {X.shape[1]}")
    Xb = np.empty(X.shape, dtype=np.uint8)
    for f, e in enumerate(edges):
        Xb[:, f] = np.searchsorted(e, X[:, f], side="right")
    return Xb


def _grow_numpy(Xb, y, n_classes, max_depth, min_leaf):
    n, d = Xb.shape
    C = n_classes
    min_leaf = max(1, min_leaf)
    offsets = np.arange(d, dtype=np.int64) * (_BINS * C)
    feature, threshold, left, right, counts = [], [], [], [], []

    def build(idx, depth):
        node = len(feature)
        yi = y[idx]
        totals = np.bincount(yi, minlength=C).astype(np.int64)
        feature.append(-1)
        threshold.append(-1)
        left.append(-1)
        right.append(-1)
        counts.append(totals)
        n_node = len(idx)
        if depth >= max_depth or n_node < 2 * min_leaf or np.count_nonzero(totals) <= 1:
            return node

        parent = float(np.int64((totals * totals).sum())) / n_node
        codes = Xb[idx].astype(np.int64) * C + yi[:, None] + offsets
        hist = np.bincount(codes.ravel(), minlength=d * _BINS * C).reshape(d, _BINS, C)
        lc = np.cumsum(hist[:, :-1, :], axis=1)
        n_left = lc.sum(axis=2)
        n_right = n_node - n_left
        rc = totals - lc
        sl = (lc * lc).sum(axis=2)
        sr = (rc * rc).sum(axis=2)
        valid = (n_left >= min_leaf) & (n_right >= min_leaf)
        score = np.full(n_left.shape, -np.inf)
        score[valid] = sl[valid] / n_left[valid] + sr[valid] / n_right[valid]
        f, t = np.unravel_index(np.argmax(score), score.shape)
        if not score[f, t] > parent + _MIN_GAIN:
            return node

        go_left = Xb[idx, f] <= t
        feature[node] = int(f)
        threshold[node] = int(t)
        left[node] = build(idx[go_left], depth + 1)
        right[node] = build(idx[~go_left], depth + 1)
        return node

    build(np.arange(n), 0)
    return feature, threshold, left, right, np.array(counts).reshape(len(feature), C)


def _predict_numpy(X, feature, cut, left, right, leaf_class):
    node = np.zeros(len(X), dtype=np.int64)
    # A tree that never split is a single leaf: nothing to walk.
    active = np.arange(len(X)) if feature[0] >= 0 else np.arange(0)
    while len(active):
        nf = node[active]
        go_left = X[active, feature[nf]] < cut[nf]
        node[active] = np.where(go_left, left[nf], right[nf])
        active = active[feature[node[active]] >= 0]
    return leaf_class[node]
