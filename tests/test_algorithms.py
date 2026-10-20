import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import Ridge as SkRidge
from sklearn.tree import DecisionTreeClassifier as SkTree

import ml_in_rust as mr


@pytest.fixture(scope="module")
def blobs():
    # Well-separated clusters: no point is equidistant from two centroids, so
    # every engine must follow exactly the same path as scikit-learn.
    X, _ = make_blobs(8000, n_features=12, centers=6, cluster_std=0.6, random_state=0)
    X = X.astype(np.float32)
    init = X[np.random.default_rng(0).choice(len(X), 6, replace=False)]
    return X, init


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(1)
    # Each feature gets a different spread so PCA's directions are well defined.
    X = (rng.standard_normal((20000, 25)) * np.linspace(0.5, 6, 25) + 4).astype(np.float32)
    y = X @ rng.standard_normal(25).astype(np.float32) + 2 + rng.standard_normal(20000).astype(np.float32)
    return X, y


# --- KMeans -----------------------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy", "rust", "mlx"])
def test_kmeans_matches_sklearn(blobs, backend):
    X, init = blobs
    ref = SkKMeans(6, init=init, n_init=1, max_iter=100, tol=0, algorithm="lloyd").fit(X)
    ours = mr.KMeans(6, init=init, max_iter=100, backend=backend).fit(X)
    # scikit-learn itself is not bit-reproducible across processes: its threads
    # add up centroid sums in whatever order they finish, and on this data it
    # lands on one of two answers that differ by ~9 boundary points. So the
    # comparison allows that much, and determinism is tested separately below.
    assert ours.n_iter_ == ref.n_iter_
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-5)
    assert np.mean(ours.labels_ == ref.labels_) >= 0.998
    assert np.allclose(ours.cluster_centers_, ref.cluster_centers_, atol=0.05)


@pytest.mark.parametrize("backend", ["numpy", "rust", "mlx"])
def test_kmeans_is_reproducible(blobs, backend):
    X, init = blobs
    runs = [mr.KMeans(6, init=init, max_iter=100, backend=backend).fit(X) for _ in range(3)]
    for r in runs[1:]:
        assert np.array_equal(r.labels_, runs[0].labels_)
        assert np.array_equal(r.cluster_centers_, runs[0].cluster_centers_)


@pytest.mark.parametrize("backend", ["numpy", "rust", "mlx"])
def test_kmeans_overlapping_clusters_reach_an_equally_good_answer(backend):
    # With overlapping clusters, some points tie to the last float32 digit and
    # engines may break the tie differently, then drift to a neighboring
    # solution. The answer must be just as good, not bit-identical.
    X, _ = make_blobs(8000, n_features=12, centers=6, random_state=0)
    X = X.astype(np.float32)
    init = X[np.random.default_rng(0).choice(len(X), 6, replace=False)]
    ref = SkKMeans(6, init=init, n_init=1, max_iter=100, tol=0, algorithm="lloyd").fit(X)
    ours = mr.KMeans(6, init=init, max_iter=100, backend=backend).fit(X)
    assert np.mean(ours.labels_ == ref.labels_) >= 0.99
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-5)


@pytest.mark.parametrize("backend", ["numpy", "rust", "mlx"])
def test_kmeans_stops_at_max_iter_with_consistent_labels(blobs, backend):
    X, init = blobs
    m = mr.KMeans(6, init=init, max_iter=2, backend=backend).fit(X)
    assert m.n_iter_ == 2
    assert np.array_equal(m.predict(X), m.labels_)


def test_kmeans_default_init_is_seeded():
    X, _ = make_blobs(2000, n_features=4, centers=3, random_state=2)
    a = mr.KMeans(3, random_state=7, backend="rust").fit(X)
    b = mr.KMeans(3, random_state=7, backend="numpy").fit(X)
    assert np.mean(a.labels_ == b.labels_) >= 0.999


# --- Ridge ------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy", "mlx"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_matches_sklearn(regression, backend, fit_intercept):
    X, y = regression
    ref = SkRidge(alpha=2.0, fit_intercept=fit_intercept).fit(X, y)
    ours = mr.Ridge(alpha=2.0, fit_intercept=fit_intercept, backend=backend).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=2e-3)
    assert ours.intercept_ == pytest.approx(ref.intercept_, abs=2e-2)
    assert np.allclose(ours.predict(X[:50]), ref.predict(X[:50]), atol=5e-2)


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_ridge_at_least_as_accurate_as_sklearn(backend):
    # Off-center data and many rows: the case where 32-bit shortcuts lose precision.
    rng = np.random.default_rng(5)
    X = (rng.standard_normal((400_000, 40)) * np.linspace(0.5, 3, 40) + 6).astype(np.float32)
    y = (X @ rng.standard_normal(40) + 1 + rng.standard_normal(400_000)).astype(np.float32)
    X64, y64 = X.astype(np.float64), y.astype(np.float64)
    Xc = X64 - X64.mean(0)
    exact = np.linalg.solve(Xc.T @ Xc + np.eye(40), Xc.T @ (y64 - y64.mean()))
    sk_err = np.abs(SkRidge(1.0).fit(X, y).coef_ - exact).max()
    our_err = np.abs(mr.Ridge(1.0, backend=backend).fit(X, y).coef_ - exact).max()
    assert our_err <= sk_err


# --- PCA --------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_pca_matches_sklearn(regression, backend):
    X, _ = regression
    ref = SkPCA(8, svd_solver="full").fit(X)
    ours = mr.PCA(8, backend=backend).fit(X)
    assert np.allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-3)
    assert np.allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-3)
    # Eigenvectors are only defined up to sign; each pair must point the same way or exactly opposite.
    assert np.abs((ours.components_ * ref.components_).sum(axis=1)).min() > 0.999
    assert np.allclose(np.abs(ours.transform(X[:20])), np.abs(ref.transform(X[:20])), atol=1e-2)


# --- Decision tree ----------------------------------------------------------

@pytest.mark.parametrize("max_depth", [1, 4, 9, None])
@pytest.mark.parametrize("min_leaf", [1, 20])
def test_tree_engines_grow_identical_trees(max_depth, min_leaf):
    X, y = make_classification(6000, 15, n_informative=6, n_classes=3, random_state=3)
    a = mr.DecisionTreeClassifier(max_depth, min_samples_leaf=min_leaf, backend="numpy").fit(X, y)
    b = mr.DecisionTreeClassifier(max_depth, min_samples_leaf=min_leaf, backend="rust").fit(X, y)
    for key in a.tree_:
        assert np.array_equal(a.tree_[key], b.tree_[key]), key
    assert np.array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_tree_accuracy_close_to_sklearn(backend):
    X, y = make_classification(20000, 20, n_informative=8, n_classes=4, random_state=0)
    ref = SkTree(max_depth=8, random_state=0).fit(X[:16000], y[:16000]).score(X[16000:], y[16000:])
    ours = mr.DecisionTreeClassifier(8, backend=backend).fit(X[:16000], y[:16000]).score(X[16000:], y[16000:])
    assert ours >= ref - 0.02


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_tree_fits_iris_perfectly_and_handles_strings(backend):
    X, y = load_iris(return_X_y=True)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    m = mr.DecisionTreeClassifier(None, backend=backend).fit(X, names)
    assert m.score(X, names) == 1.0
    assert m.predict(X[0])[0] == "setosa"


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_tree_single_class_is_one_leaf(backend):
    X = np.random.default_rng(0).standard_normal((50, 3))
    m = mr.DecisionTreeClassifier(5, backend=backend).fit(X, np.zeros(50))
    assert m.node_count == 1
    assert (m.predict(X) == 0).all()


def test_bad_backends_are_rejected():
    for cls, bad in [(mr.KMeans, "gpu"), (mr.Ridge, "rust"), (mr.PCA, "rust"), (mr.DecisionTreeClassifier, "mlx")]:
        with pytest.raises(ValueError):
            cls(backend=bad)
