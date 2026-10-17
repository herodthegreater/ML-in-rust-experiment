import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.neighbors import KNeighborsClassifier as SkKNN

from autoalgo import KNeighborsClassifier

BACKENDS = ["numpy", "rust", "rust-blas", "mlx"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_sklearn_on_iris(backend):
    X, y = load_iris(return_X_y=True)
    ours = KNeighborsClassifier(5, backend=backend).fit(X, y).predict(X)
    ref = SkKNN(5, algorithm="brute").fit(X, y).predict(X)
    assert np.array_equal(ours, ref)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_features", [3, 8, 13, 54])
@pytest.mark.parametrize("k", [1, 5, 16])
def test_agrees_with_sklearn_on_random_data(backend, n_features, k):
    X, y = make_classification(
        n_samples=3000, n_features=n_features, n_informative=min(n_features, 3),
        n_redundant=0, n_classes=3, random_state=0,
    )
    Xtr, Xte, ytr = X[:2500], X[2500:], y[:2500]
    ours = KNeighborsClassifier(k, backend=backend).fit(Xtr, ytr).predict(Xte)
    ref = SkKNN(k, algorithm="brute").fit(Xtr, ytr).predict(Xte)
    # float32 vs float64 distances can reorder exact near-ties, nothing more.
    assert np.mean(ours == ref) >= 0.995


def test_engines_agree_with_each_other():
    X, y = make_classification(n_samples=5000, n_features=20, random_state=1)
    a = KNeighborsClassifier(7, backend="numpy").fit(X[:4000], y[:4000]).predict(X[4000:])
    b = KNeighborsClassifier(7, backend="rust").fit(X[:4000], y[:4000]).predict(X[4000:])
    assert np.mean(a == b) >= 0.999


@pytest.mark.parametrize("backend", BACKENDS)
def test_string_labels_and_single_row(backend):
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    y = np.array(["cat", "cat", "dog", "dog"])
    model = KNeighborsClassifier(1, backend=backend).fit(X, y)
    assert model.predict(np.array([4.9, 5.0]))[0] == "dog"
    assert list(model.predict(X)) == list(y)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        KNeighborsClassifier(backend="gpu")
    with pytest.raises(ValueError):
        KNeighborsClassifier(10).fit(np.zeros((3, 2)), [0, 1, 2])
    model = KNeighborsClassifier(1).fit(np.zeros((3, 2)), [0, 1, 2])
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, 5)))
