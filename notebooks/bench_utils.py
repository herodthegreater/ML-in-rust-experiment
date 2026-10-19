"""Shared helpers for the benchmark notebooks: timing, data, results, charts."""

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
RESULTS = Path(__file__).resolve().parent / "results"

# Color follows the engine in every notebook. scikit-learn is the neutral
# baseline (gray, with line style telling its variants apart); autoalgo's
# engines take the first four categorical slots in a fixed order.
ENGINE_COLORS = {
    "autoalgo-numpy": "#2a78d6",
    "autoalgo-rust": "#eb6834",
    "autoalgo-rust-blas": "#1baf7a",
    "autoalgo-mlx": "#eda100",
}
BASELINE_GRAY = "#52514e"
BASELINE_STYLES = ["-", "--", ":", "-."]


def machine_info():
    def sysctl(key):
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "?"

    try:
        import mlx.core as mx
        mlx_version = mx.__version__
    except ImportError:
        mlx_version = "not installed"
    mem = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "cores": f"{sysctl('hw.perflevel0.physicalcpu')} performance + {sysctl('hw.perflevel1.physicalcpu')} efficiency",
        "ram_gb": round(int(mem) / 2**30) if mem.isdigit() else "?",
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": f"{np.__version__} ({np.show_config(mode='dicts')['Build Dependencies']['blas']['name']})",
        "sklearn": sklearn.__version__,
        "mlx": mlx_version,
    }


def timeit(fn, repeats=3, warmup=True):
    """Median wall-clock seconds of fn() after one untimed warm-up call."""
    if warmup:
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def load_covtype():
    """Covertype (581,012 rows, 54 features, 7 classes), standardized float32, shuffled."""
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype(data_home=DATA)
    X = data.data.astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    order = np.random.default_rng(0).permutation(len(X))
    return X[order], data.target[order]


def add_speedup(df, time_col, group_cols, baseline="sklearn"):
    """Add 'vs sklearn' = baseline time / engine time within each group (>1 means faster)."""
    df = df.copy()
    base = df[df["engine"] == baseline].set_index(group_cols)[time_col]
    df["vs sklearn"] = [
        base.loc[tuple(r) if len(group_cols) > 1 else r[0]] / t
        for r, t in zip(df[group_cols].itertuples(index=False), df[time_col])
    ]
    return df


def save_results(name, df, notes=None):
    RESULTS.mkdir(exist_ok=True)
    payload = {"machine": machine_info(), "notes": notes or {}, "rows": df.to_dict(orient="records")}
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))


def load_results(name):
    payload = json.loads((RESULTS / f"{name}.json").read_text())
    return pd.DataFrame(payload["rows"]), payload


def _style(engine, baseline_index):
    if engine in ENGINE_COLORS:
        return dict(color=ENGINE_COLORS[engine], linestyle="-")
    return dict(color=BASELINE_GRAY, linestyle=BASELINE_STYLES[baseline_index % len(BASELINE_STYLES)])


def line_chart(df, x, y, title, xlabel, ylabel, logx=True, logy=True, ax=None):
    """One line per engine, labeled at its right end (and in the legend)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.6))
    baselines = 0
    engines = list(dict.fromkeys(df["engine"]))
    ends = []
    for engine in engines:
        part = df[df["engine"] == engine].sort_values(x)
        style = _style(engine, baselines)
        if engine not in ENGINE_COLORS:
            baselines += 1
        ax.plot(part[x], part[y], marker="o", markersize=5, linewidth=2, label=engine, **style)
        ends.append((engine, part[x].iloc[-1], part[y].iloc[-1]))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.margins(x=0.18)
    _label_line_ends(ax, ends)
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return ax


def _label_line_ends(ax, ends, min_gap_px=13):
    """Name each line at its last point, nudging labels apart so none overlap."""
    ax.autoscale_view()
    to_pts = 72 / ax.figure.dpi
    placed = sorted(
        ((ax.transData.transform((x, y))[1], name, x, y) for name, x, y in ends), key=lambda t: t[0]
    )
    targets = [p[0] for p in placed]
    for i in range(1, len(targets)):
        targets[i] = max(targets[i], targets[i - 1] + min_gap_px)
    for (orig, name, x, y), target in zip(placed, targets):
        ax.annotate(name, (x, y), xytext=(6, (target - orig) * to_pts), textcoords="offset points",
                    va="center", fontsize=9, color="#0b0b0b")


def fmt_ms(seconds):
    return f"{seconds * 1e3:,.1f} ms"
