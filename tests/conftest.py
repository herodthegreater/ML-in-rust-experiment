import pytest

from ml_in_rust import mlx_available


def pytest_collection_modifyitems(config, items):
    """Skip MLX cases on machines without MLX (for example Linux on Intel)."""
    if mlx_available():
        return
    skip = pytest.mark.skip(reason="MLX is not installed on this machine")
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("backend") == "mlx":
            item.add_marker(skip)
