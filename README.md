# autoalgo v0.2: is a faster ML engine worth building?

This repo is an experiment, not a library yet. It answers one question before more work goes into autoalgo: **can a new engine beat scikit-learn on speed, where, and at what cost?**

Five algorithms are each built more than one way and raced against scikit-learn on an Apple M4. Every engine is checked against scikit-learn's answers (94 tests), and every race lives in a Jupyter notebook you can re-run.

| Algorithm | autoalgo engines | Bottleneck it tests |
|---|---|---|
| k-nearest neighbors | numpy, rust, rust-blas, mlx | matrix math, then picking the k smallest (memory) |
| k-means | numpy, rust, mlx | matrix math every iteration, plus summing rows per cluster |
| Ridge regression | numpy, mlx | one big matrix multiply, then a tiny solve |
| PCA | numpy, mlx | one big matrix multiply, then a tiny eigen-solve |
| Decision tree | numpy, rust | branchy integer counting, no matrix math at all |

- **numpy:** autoalgo built on top of existing libraries (Python plus NumPy, which hands matrix math to Apple Accelerate).
- **rust:** compiled Rust (`src/`), every core, GIL released. k-means also calls Accelerate for its matrix math.
- **rust-blas:** Rust that hands KNN's distance math to Accelerate, which can use the M4's matrix units.
- **mlx:** pure Python on the Apple GPU.

## Scoreboard (largest size raced, Apple M4, 24 GB)

Speed relative to scikit-learn's default; above 1× is faster. Full tables, charts and every intermediate size are in [`notebooks/`](notebooks/).

| Algorithm | numpy | mlx | rust / rust-blas |
|---|---:|---:|---:|
| KNN, predict 2,000 rows (579k train) | 0.12× | 0.89× | 0.61× / **3.02×** |
| KNN, one row at a time (10k train) | 5.5× | 1.3× | 10.6× / **14.0×** |
| k-means, per iteration (2M rows) | 0.09× | 0.37× | **1.68×** |
| Ridge, fit (4M rows) | **1.19×** | 1.11× | (none: no headroom) |
| PCA, fit (3M rows) | 0.56× | 0.55× | (none: no headroom) |
| Decision tree, fit (531k rows, depth 20) | 0.33× | (none) | **2.30×** |
| Decision tree, predict (50k rows, depth 16) | 0.72× | (none) | **7.3×** |

What it says:

1. **Wrapping libraries in Python loses.** The NumPy engines were 6 to 12× slower than scikit-learn whenever a memory-bound step had to run in Python.
2. **Python on the GPU (MLX) ties at best.** It still writes big intermediate matrices to memory and reads them back.
3. **Compiled Rust wins in 3 of 5:** KNN (3× batch, up to 14× single row), k-means (1.7×), and trees (2.3× fit, 7× predict).
4. **Ridge and PCA have no real headroom.** They're one big matrix multiply that scikit-learn already sends to the same hardware. scikit-learn's default PCA beat autoalgo. autoalgo's Ridge is only slightly faster, but on large data it's 40 to 400× *more accurate*, because it centers in chunks and sums in 64-bit.
5. **The biggest gap wasn't an engine.** scikit-learn's PCA took 0.14 s or 5.8 s on the same data depending only on `svd_solver` (43×). The best engine win anywhere in these races was under 4× on batch work.

**Where that points:** the measured speed wins are real but narrow, take compiled code, and need careful engineering and testing per algorithm. The single largest win came from choosing the right solver, which is a navigation problem rather than an engine problem. [`notebooks/06_summary.ipynb`](notebooks/06_summary.ipynb) lays out the two paths (an alternative to scikit-learn vs a navigator for it) against this evidence.

## Correctness notes worth knowing

- **k-means:** scikit-learn is not bit-reproducible across processes (its threads add up sums in varying order). autoalgo's engines are, and that's tested.
- **Ridge:** a first version subtracted the mean *after* multiplying in 32-bit floats and silently lost precision. The current version centers in chunks and sums in 64-bit, and a test holds it to at least scikit-learn's accuracy against an exact 64-bit answer.
- **Decision tree:** the NumPy and Rust engines grow node-for-node identical trees (tested). Prediction compares raw values against stored cutoffs instead of binning every input, which made Rust prediction 40× faster than the first version.

## Run it yourself

Requires macOS on Apple Silicon (for the Accelerate and MLX engines), Python 3.10+, and a Rust toolchain.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy scikit-learn pytest mlx jupyter matplotlib pandas
maturin develop --release
pytest -q
```

Then open `notebooks/00_start_here.ipynb` with the `.venv` interpreter and run the notebooks in order. Each one saves its numbers to `notebooks/results/`, which `06_summary` reads. Each notebook has a `QUICK = False` switch for a fast smoke run. Covertype (about 11 MB) downloads into `bench/data/` on first use.

For a quick KNN-only race from a script: `python race.py` (one size) or `python bench/run_bench.py` (four sizes plus latency).

## Layout

```text
src/                     Rust engines (PyO3): knn.rs, kmeans.rs, tree.rs, accelerate.rs
python/autoalgo/         scikit-learn-style estimators: neighbors, cluster, linear_model, decomposition, tree
tests/                   parity tests against scikit-learn (94)
notebooks/               one benchmark notebook per algorithm, plus 00_start_here and 06_summary
notebooks/results/       the numbers each notebook saved on the run committed here
bench/run_bench.py       the original v0.1 KNN race
race.py                  a quick one-size KNN race to run from an IDE
```
