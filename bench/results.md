# Benchmark results

- **chip:** Apple M4
- **performance_cores:** 4
- **efficiency_cores:** 6
- **ram_gb:** 24
- **os:** macOS-26.6.1-arm64-arm-64bit-Mach-O
- **python:** 3.13.5
- **numpy:** 2.5.4
- **numpy_blas:** accelerate
- **sklearn:** 1.9.1
- **mlx:** 0.32.4

k = 5, 2,000 held-out queries, Covertype (54 features, 7 classes), standardized float32. Times are medians after a warm-up call.

## Batch prediction

| train rows | engine | fit | predict 2,000 | queries/s | accuracy | agrees with sklearn | speedup vs sklearn |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 | sklearn (brute) | 0.4 ms | 17.9 ms | 111,970 | 0.7615 | 1.0000 | 1.00x |
| 10,000 | autoalgo-numpy | 0.2 ms | 132.9 ms | 15,053 | 0.7615 | 1.0000 | 0.13x |
| 10,000 | autoalgo-rust | 0.4 ms | 17.0 ms | 117,806 | 0.7615 | 1.0000 | 1.05x |
| 10,000 | autoalgo-rust-blas | 0.3 ms | 4.4 ms | 451,442 | 0.7615 | 1.0000 | 4.03x |
| 10,000 | autoalgo-mlx | 0.6 ms | 10.7 ms | 187,644 | 0.7615 | 1.0000 | 1.68x |
| 50,000 | sklearn (brute) | 1.4 ms | 57.8 ms | 34,613 | 0.8445 | 1.0000 | 1.00x |
| 50,000 | autoalgo-numpy | 0.9 ms | 720.6 ms | 2,775 | 0.8445 | 1.0000 | 0.08x |
| 50,000 | autoalgo-rust | 1.7 ms | 82.4 ms | 24,279 | 0.8445 | 1.0000 | 0.70x |
| 50,000 | autoalgo-rust-blas | 1.5 ms | 18.5 ms | 108,127 | 0.8445 | 1.0000 | 3.12x |
| 50,000 | autoalgo-mlx | 1.6 ms | 54.0 ms | 37,067 | 0.8445 | 1.0000 | 1.07x |
| 200,000 | sklearn (brute) | 5.6 ms | 193.8 ms | 10,321 | 0.8980 | 1.0000 | 1.00x |
| 200,000 | autoalgo-numpy | 4.2 ms | 1549.6 ms | 1,291 | 0.8980 | 1.0000 | 0.13x |
| 200,000 | autoalgo-rust | 10.0 ms | 333.0 ms | 6,005 | 0.8980 | 1.0000 | 0.58x |
| 200,000 | autoalgo-rust-blas | 9.2 ms | 74.8 ms | 26,749 | 0.8980 | 1.0000 | 2.59x |
| 200,000 | autoalgo-mlx | 8.8 ms | 217.7 ms | 9,187 | 0.8980 | 1.0000 | 0.89x |
| 579,012 | sklearn (brute) | 16.1 ms | 537.2 ms | 3,723 | 0.9250 | 1.0000 | 1.00x |
| 579,012 | autoalgo-numpy | 12.3 ms | 4548.7 ms | 440 | 0.9250 | 1.0000 | 0.12x |
| 579,012 | autoalgo-rust | 26.8 ms | 974.9 ms | 2,052 | 0.9250 | 1.0000 | 0.55x |
| 579,012 | autoalgo-rust-blas | 30.2 ms | 216.9 ms | 9,219 | 0.9250 | 1.0000 | 2.48x |
| 579,012 | autoalgo-mlx | 24.8 ms | 655.6 ms | 3,051 | 0.9250 | 1.0000 | 0.82x |

## Single-row latency

| train rows | engine | median | p95 | speedup vs sklearn |
|---:|---|---:|---:|---:|
| 10,000 | sklearn | 657 µs | 1,084 µs | 1.00x |
| 10,000 | autoalgo-numpy | 95 µs | 129 µs | 6.91x |
| 10,000 | autoalgo-rust | 47 µs | 49 µs | 13.92x |
| 10,000 | autoalgo-rust-blas | 37 µs | 39 µs | 17.68x |
| 10,000 | autoalgo-mlx | 283 µs | 592 µs | 2.32x |
| 200,000 | sklearn | 2,693 µs | 3,374 µs | 1.00x |
| 200,000 | autoalgo-numpy | 1,283 µs | 1,939 µs | 2.10x |
| 200,000 | autoalgo-rust | 748 µs | 841 µs | 3.60x |
| 200,000 | autoalgo-rust-blas | 862 µs | 906 µs | 3.13x |
| 200,000 | autoalgo-mlx | 959 µs | 1,439 µs | 2.81x |
