# autoalgo v0.1: does optimizing ML for speed matter?

This repo is an experiment, not a library yet. It answers one question before more work goes into autoalgo: **can a new engine actually beat scikit-learn on speed, and what does it take?**

To find out, it builds one algorithm, k-nearest neighbors classification, four ways and races them against scikit-learn on a real dataset:

| Engine | What it is |
|---|---|
| `sklearn` | scikit-learn's `KNeighborsClassifier`, default settings, all cores (`n_jobs=-1`) |
| `autoalgo-numpy` | autoalgo built on top of existing libraries: one big NumPy matrix multiply (Apple Accelerate underneath) plus `np.argpartition` |
| `autoalgo-rust` | a from-scratch Rust engine: hand-tiled SIMD distance loops, every core via rayon, GIL released |
| `autoalgo-rust-blas` | the same Rust engine, but each tile of distance math is one Apple Accelerate matrix multiply; Rust does the neighbor search and voting |
| `autoalgo-mlx` | pure Python on the Apple GPU through MLX: one matrix multiply, then a two-stage partial sort (best k per block of 128 rows, then best k overall) |

All five return the same predictions as scikit-learn on every benchmark run (agreement 1.0000).

## Results (Apple M4, 24 GB, macOS 26)

The dataset is Covertype: 581,012 rows, 54 features, 7 classes, standardized to float32. Each run predicts 2,000 held-out rows with k = 5, and every time is a median after a warm-up call. Full tables are in [bench/results.md](bench/results.md).

**Predicting 2,000 rows** (lower is better; speedup is relative to scikit-learn):

| train rows | sklearn | autoalgo-numpy | autoalgo-rust | autoalgo-rust-blas | autoalgo-mlx |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 17.9 ms | 132.9 ms (0.13x) | 17.0 ms (1.05x) | **4.4 ms (4.07x)** | 10.7 ms (1.67x) |
| 50,000 | 57.8 ms | 720.6 ms (0.08x) | 82.4 ms (0.70x) | **18.5 ms (3.12x)** | 54.0 ms (1.07x) |
| 200,000 | 193.8 ms | 1,550 ms (0.13x) | 333.0 ms (0.58x) | **74.8 ms (2.59x)** | 217.7 ms (0.89x) |
| 579,012 | 537.2 ms | 4,549 ms (0.12x) | 974.9 ms (0.55x) | **216.9 ms (2.48x)** | 655.6 ms (0.82x) |

**Predicting one row at a time** (the cost a live service or a robot pays per prediction; median):

| train rows | sklearn | autoalgo-numpy | autoalgo-rust | autoalgo-rust-blas | autoalgo-mlx |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 657 µs | 95 µs (6.9x) | 47 µs (14.0x) | **37 µs (17.8x)** | 283 µs (2.3x) |
| 200,000 | 2,693 µs | 1,283 µs (2.1x) | **748 µs (3.6x)** | 862 µs (3.1x) | 959 µs (2.8x) |

## What this says

1. **Wrapping existing libraries is not enough.** The NumPy engine computes distances with the same fast matrix multiply, yet it is 8 to 13 times slower than scikit-learn on batches. The time goes into `argpartition`, which picks the nearest rows out of a huge distance matrix after it is fully built. That step is limited by memory, not math. scikit-learn avoids it by working in chunks and keeping only the best candidates as it goes.
2. **Rust alone is not enough either.** Hand-written Rust ties scikit-learn on small data and falls to about half its speed on large data. The language is not the advantage. scikit-learn's inner loop is already compiled, and its matrix math runs on Apple's tuned BLAS.
3. **Rust plus the chip's matrix hardware wins clearly.** Routing the distance math through Accelerate, which can use the M4's dedicated matrix units, and doing everything else in Rust makes autoalgo 2.5 to 3.7 times faster than scikit-learn at every size, with identical answers.
4. **Per-call overhead is the biggest gap.** For single predictions, autoalgo is 14 to 18 times faster than scikit-learn on 10k rows, because scikit-learn spends most of each call on Python-side checks and setup. This is the number that matters for robots, games and live APIs.
5. **Python on the GPU (MLX) ties scikit-learn, but doesn't get close to Rust + Accelerate.** MLX is 1.7x faster than scikit-learn on small data and 0.8x on the full dataset, about 3x slower than Rust + Accelerate throughout. Its first version was much slower (2,980 ms at full size): `argpartition` over each 579k-wide row of distances dominated, just like the NumPy engine. Selecting the best 5 inside blocks of 128 rows first made it 4.5x faster. What's left is memory traffic. MLX must write the whole distance matrix out to memory and read it back to select from it, while the Rust engine scores and selects in the same pass and only keeps 5 candidates per query. Closing that gap from Python would mean a custom GPU kernel (`mx.fast.metal_kernel`), which is Metal shader code rather than Python. On single rows, launching GPU work costs about 280 µs, so the GPU is the wrong tool for one-at-a-time predictions.

**Verdict: speed optimization is worth pursuing, but only done one way.** The wins come from reaching the fastest hardware path on each device (here, Accelerate's matrix units) and from cutting per-call overhead, with a compiled core tying it together. Gluing libraries together in Python loses, and a compiled language by itself only ties. Staying in pure Python, even on the GPU, gets you to roughly scikit-learn speed, not past it. That also means a "fastest on any device" library needs a different fast path per device (Accelerate on Apple, another BLAS or a GPU elsewhere) and a way to choose between them.

**Limits of this test:** one algorithm (brute-force KNN), one dataset, one machine. scikit-learn's tree-based neighbor search (`kd_tree`, `ball_tree`) wasn't raced, because its default chose brute force for 54 features. The Accelerate and MLX engines are macOS-only. The NumPy engine is a straightforward implementation, and a cleverer one could close part of its gap.

## Run it yourself

Requires macOS on Apple Silicon (for the Accelerate engine), Python 3.10+, and a Rust toolchain.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy scikit-learn pytest mlx
maturin develop --release
pytest -q
python bench/run_bench.py
```

The benchmark downloads Covertype (about 11 MB) into `bench/data/` on its first run and writes `bench/results.md` and `bench/results.json`. Use `--quick` for a short smoke run.

## Layout

```text
src/lib.rs                     Rust engine (PyO3): SIMD tiles, rayon, optional Accelerate GEMM
python/autoalgo/neighbors.py   KNeighborsClassifier with backend="numpy" | "rust" | "rust-blas" | "mlx"
tests/test_knn.py              parity tests against scikit-learn
bench/run_bench.py             the full race (four sizes plus single-row latency)
race.py                        a quick one-size race to run from an IDE
bench/results.md               latest results from the machine above
```
