# Predictions for the second machine

Written on 2026-10-19, **before** running anything on the second machine, and committed
so the git history shows they came first. The point is to test whether the findings
from the Apple M4 are about Rust or about Apple's hardware.

**Machine A (done):** Apple M4, macOS. NumPy and scikit-learn use Apple Accelerate, which
can reach Apple's matrix units. `rust-blas` also uses Accelerate. MLX runs on the GPU.

**Machine B (next):** Intel Mac running Linux (x86-64). NumPy and scikit-learn use
OpenBLAS. `rust-blas` uses the pure-Rust `matrixmultiply` crate. Rust is compiled with
`target-cpu=native`, so it gets AVX2 if the CPU has it. MLX is not available.

All numbers are speed relative to scikit-learn **on the same machine**, at the largest size raced.

| # | Prediction for machine B | Machine A result | Why |
|---|---|---|---|
| 1 | `rust-blas` KNN falls to **1.5x or less** | 3.0x | Its M4 lead came largely from Accelerate's matrix units. On Linux it trades a tuned BLAS for a pure-Rust multiply, while scikit-learn keeps OpenBLAS. |
| 2 | Plain `rust` KNN stays **between 0.5x and 1.2x** | 0.6x | Hand-written SIMD against a tuned BLAS, same as on the M4. |
| 3 | `rust` k-means falls to **between 0.8x and 1.5x** | 1.7x | Its matrix math moves from Accelerate to `matrixmultiply`; the fused single pass still helps. |
| 4 | `rust` decision tree fit stays **at 2x or more** | 2.3x | No matrix math at all, so the BLAS change shouldn't matter. |
| 5 | `rust` decision tree predict stays **at 4x or more** | 7.3x | Raw-value comparisons on every core; no matrix math. |
| 6 | Single-row KNN latency: `rust` stays **at 5x or more** (10k rows) | 10.6x | That gap is scikit-learn's per-call Python overhead, which doesn't depend on the chip. |
| 7 | Ridge and PCA: `numpy` stays **within 0.7x to 1.3x** | 1.19x, 0.56x | One big multiply on the same BLAS scikit-learn uses: no headroom on any machine. |
| 8 | scikit-learn's PCA solver gap (`full` vs default) stays **above 10x** | 43x | It's an algorithm difference, not a hardware one. |
| 9 | The `numpy` engines still lose to scikit-learn by **3x or more** on KNN and k-means | 8x, 11x | Same memory-bound steps running in Python. |

If 1 and 3 hold while 4, 5 and 6 hold too, the conclusion is: **Rust's wins on matrix-heavy
algorithms were really Apple Accelerate's wins, while Rust's wins on branchy and
overhead-bound work are portable.** If 1 and 3 fail, Rust plus a pure-Rust multiply is
competitive even without Apple's hardware, which would be the more surprising result.

The machine A numbers in this table are from the committed v0.2 run. Machine B's results
will be checked against this table once its folder is in `notebooks/results/`.
