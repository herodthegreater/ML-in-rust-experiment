"""Shared helpers for the benchmark notebooks: timing, data, results, charts."""

import json
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# Color follows the engine in every notebook. scikit-learn is the neutral
# baseline (gray, with line style telling its variants apart); this project's
# engines take the first four categorical slots in a fixed order.
ENGINE_COLORS = {
    "numpy": "#2a78d6",
    "rust": "#eb6834",
    "rust-blas": "#1baf7a",
    "mlx": "#eda100",
}
BASELINE_GRAY = "#52514e"
BASELINE_STYLES = ["-", "--", ":", "-."]


def _sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def cpu_name():
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine_slug():
    """Folder name for this machine's results, e.g. macos-arm64-apple-m4."""
    system = {"Darwin": "macos"}.get(platform.system(), platform.system().lower())
    cpu = re.sub(r"\(r\)|\(tm\)|cpu|@.*$", "", cpu_name().lower())
    cpu = re.sub(r"[^a-z0-9]+", "-", cpu).strip("-")
    return f"{system}-{platform.machine().lower()}-{cpu}"


def machine_info():
    import ml_in_rust as mr

    if platform.system() == "Darwin":
        cores = f"{_sysctl('hw.perflevel0.physicalcpu')} performance + {_sysctl('hw.perflevel1.physicalcpu')} efficiency"
        mem = _sysctl("hw.memsize")
        ram = round(int(mem) / 2**30) if mem.isdigit() else "?"
    else:
        cores = f"{os.cpu_count()} logical"
        ram = "?"
        try:
            kb = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal"))
            ram = round(kb / 2**20)
        except (OSError, StopIteration):
            pass
    try:
        import mlx.core as mx
        mlx_version = f"{mx.__version__} ({mx.default_device()})"
    except ImportError:
        mlx_version = "not installed"
    return {
        "machine": machine_slug(),
        "cpu": cpu_name(),
        "cores": cores,
        "ram_gb": ram,
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": f"{np.__version__} (BLAS: {np.show_config(mode='dicts')['Build Dependencies']['blas']['name']})",
        "sklearn": sklearn.__version__,
        "rust-blas matrix library": mr.gemm_library(),
        "mlx": mlx_version,
    }


def only_available(engines):
    """Drop engines this machine can't run (MLX off Apple silicon)."""
    from ml_in_rust import available_backends

    ok = set(available_backends())
    return {k: v for k, v in engines.items() if k.startswith("sklearn") or k in ok}


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
    """Save one notebook's numbers under results/<this machine>/<name>.json."""
    folder = RESULTS_ROOT / machine_slug()
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine_info(), "notes": notes or {}, "rows": df.to_dict(orient="records")}
    (folder / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))


def machines_with_results():
    return sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()) if RESULTS_ROOT.exists() else []


def load_results(name, machine=None):
    payload = json.loads((RESULTS_ROOT / (machine or machine_slug()) / f"{name}.json").read_text())
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
