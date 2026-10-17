"""Race k-nearest neighbors three ways on the Covertype dataset.

Engines:
  sklearn         scikit-learn's KNeighborsClassifier, default settings, all cores
  autoalgo-numpy  autoalgo built on top of NumPy + the system BLAS
  autoalgo-rust   autoalgo's from-scratch Rust engine
  autoalgo-rust-blas  the same Rust engine with Accelerate doing the matrix math

Measures, per training size: fit time, time to predict a fixed batch of
queries, accuracy, and agreement with scikit-learn. Then single-row latency,
the cost a live service or robot pays per prediction.

Usage:  python bench/run_bench.py [--quick]
Writes: bench/results.json and bench/results.md
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import fetch_covtype
from sklearn.neighbors import KNeighborsClassifier as SkKNN

from autoalgo import KNeighborsClassifier

HERE = Path(__file__).parent
K = 5
N_QUERIES = 2000
SEED = 0


def engines():
    return {
        "sklearn": lambda: SkKNN(n_neighbors=K, n_jobs=-1),
        "autoalgo-numpy": lambda: KNeighborsClassifier(K, backend="numpy"),
        "autoalgo-rust": lambda: KNeighborsClassifier(K, backend="rust"),
        "autoalgo-rust-blas": lambda: KNeighborsClassifier(K, backend="rust-blas"),
    }


def timed(fn, repeats):
    fn()  # warm-up: first call pays one-time costs (thread pools, page faults)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def machine_info():
    def sysctl(key):
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "?"

    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        "ram_gb": round(int(sysctl("hw.memsize")) / 2**30) if sysctl("hw.memsize").isdigit() else "?",
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numpy_blas": blas,
        "sklearn": sklearn.__version__,
    }


def load():
    data = fetch_covtype(data_home=HERE / "data")
    X = data.data.astype(np.float32)
    y = data.target
    X = (X - X.mean(0)) / X.std(0)
    order = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[order], y[order]
    return X[N_QUERIES:], y[N_QUERIES:], X[:N_QUERIES], y[:N_QUERIES]


def batch_race(Xpool, ypool, Xq, yq, sizes, repeats):
    rows = []
    for n in sizes:
        Xtr, ytr = Xpool[:n], ypool[:n]
        ref_pred = None
        for name, make in engines().items():
            model = make()
            fit_s = timed(lambda: model.fit(Xtr, ytr), repeats)
            pred_s = timed(lambda: model.predict(Xq), repeats)
            pred = model.predict(Xq)
            if name == "sklearn":
                ref_pred = pred
                method = model._fit_method
            row = {
                "n_train": n,
                "engine": name,
                "fit_s": fit_s,
                "predict_s": pred_s,
                "queries_per_s": len(Xq) / pred_s,
                "accuracy": float(np.mean(pred == yq)),
                "agreement_with_sklearn": float(np.mean(pred == ref_pred)),
            }
            if name == "sklearn":
                row["sklearn_method"] = method
            rows.append(row)
            print(f"  n={n:>7,}  {name:<19} predict {pred_s * 1e3:9.1f} ms   "
                  f"acc {row['accuracy']:.4f}   agree {row['agreement_with_sklearn']:.4f}")
    return rows


def latency_race(Xpool, ypool, Xq, n_train, calls):
    rows = []
    Xtr, ytr = Xpool[:n_train], ypool[:n_train]
    for name, make in engines().items():
        model = make().fit(Xtr, ytr)
        singles = [Xq[i : i + 1] for i in range(calls)]
        model.predict(singles[0])
        times = []
        for q in singles:
            t0 = time.perf_counter()
            model.predict(q)
            times.append(time.perf_counter() - t0)
        rows.append({
            "n_train": n_train,
            "engine": name,
            "median_us": statistics.median(times) * 1e6,
            "p95_us": float(np.percentile(times, 95)) * 1e6,
        })
        print(f"  single row, n={n_train:,}  {name:<19} median {rows[-1]['median_us']:8.0f} us")
    return rows


def write_markdown(info, batch, latency, path):
    def speedup(rows, n, engine, field):
        base = next(r for r in rows if r["n_train"] == n and r["engine"] == "sklearn")[field]
        mine = next(r for r in rows if r["n_train"] == n and r["engine"] == engine)[field]
        return base / mine

    lines = ["# Benchmark results", ""]
    lines += [f"- **{k}:** {v}" for k, v in info.items()]
    lines += ["", f"k = {K}, {N_QUERIES:,} held-out queries, Covertype (54 features, 7 classes), "
              "standardized float32. Times are medians after a warm-up call.", ""]
    lines += ["## Batch prediction", "",
              "| train rows | engine | fit | predict 2,000 | queries/s | accuracy | agrees with sklearn | speedup vs sklearn |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in batch:
        sp = speedup(batch, r["n_train"], r["engine"], "predict_s")
        name = r["engine"] + (f" ({r['sklearn_method']})" if "sklearn_method" in r else "")
        lines.append(
            f"| {r['n_train']:,} | {name} | {r['fit_s'] * 1e3:.1f} ms | {r['predict_s'] * 1e3:.1f} ms | "
            f"{r['queries_per_s']:,.0f} | {r['accuracy']:.4f} | {r['agreement_with_sklearn']:.4f} | {sp:.2f}x |"
        )
    lines += ["", "## Single-row latency", "",
              "| train rows | engine | median | p95 | speedup vs sklearn |", "|---:|---|---:|---:|---:|"]
    for r in latency:
        sp = speedup(latency, r["n_train"], r["engine"], "median_us")
        lines.append(f"| {r['n_train']:,} | {r['engine']} | {r['median_us']:,.0f} µs | {r['p95_us']:,.0f} µs | {sp:.2f}x |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes for a smoke test")
    args = ap.parse_args()

    Xpool, ypool, Xq, yq = load()
    sizes = [10_000, 50_000] if args.quick else [10_000, 50_000, 200_000, len(Xpool)]
    repeats = 1 if args.quick else 3
    info = machine_info()
    print("machine:", info)

    print("batch prediction")
    batch = batch_race(Xpool, ypool, Xq, yq, sizes, repeats)
    print("single-row latency")
    latency = []
    for n in ([10_000] if args.quick else [10_000, 200_000]):
        latency += latency_race(Xpool, ypool, Xq, n, calls=50 if args.quick else 300)

    out = {"machine": info, "batch": batch, "latency": latency}
    suffix = "-quick" if args.quick else ""
    (HERE / f"results{suffix}.json").write_text(json.dumps(out, indent=2))
    write_markdown(info, batch, latency, HERE / f"results{suffix}.md")
    print(f"wrote bench/results{suffix}.md")


if __name__ == "__main__":
    os.chdir(HERE.parent)
    main()
