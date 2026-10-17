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

k = 5, 2,000 held-out queries, Covertype (54 features, 7 classes), standardized float32. Times are medians after a warm-up call.

## Batch prediction

| train rows | engine | fit | predict 2,000 | queries/s | accuracy | agrees with sklearn | speedup vs sklearn |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 | sklearn (brute) | 0.4 ms | 17.6 ms | 113,348 | 0.7615 | 1.0000 | 1.00x |
| 10,000 | autoalgo-numpy | 0.2 ms | 136.7 ms | 14,627 | 0.7615 | 1.0000 | 0.13x |
| 10,000 | autoalgo-rust | 0.4 ms | 16.8 ms | 118,942 | 0.7615 | 1.0000 | 1.05x |
| 10,000 | autoalgo-rust-blas | 0.4 ms | 4.8 ms | 418,855 | 0.7615 | 1.0000 | 3.70x |
| 50,000 | sklearn (brute) | 1.4 ms | 57.4 ms | 34,857 | 0.8445 | 1.0000 | 1.00x |
| 50,000 | autoalgo-numpy | 0.9 ms | 737.6 ms | 2,711 | 0.8445 | 1.0000 | 0.08x |
| 50,000 | autoalgo-rust | 1.4 ms | 82.9 ms | 24,116 | 0.8445 | 1.0000 | 0.69x |
| 50,000 | autoalgo-rust-blas | 1.6 ms | 18.2 ms | 109,716 | 0.8445 | 1.0000 | 3.15x |
| 200,000 | sklearn (brute) | 5.7 ms | 195.6 ms | 10,225 | 0.8980 | 1.0000 | 1.00x |
| 200,000 | autoalgo-numpy | 4.1 ms | 1608.3 ms | 1,244 | 0.8980 | 1.0000 | 0.12x |
| 200,000 | autoalgo-rust | 9.5 ms | 327.8 ms | 6,101 | 0.8980 | 1.0000 | 0.60x |
| 200,000 | autoalgo-rust-blas | 10.0 ms | 73.3 ms | 27,301 | 0.8980 | 1.0000 | 2.67x |
| 579,012 | sklearn (brute) | 16.4 ms | 525.9 ms | 3,803 | 0.9250 | 1.0000 | 1.00x |
| 579,012 | autoalgo-numpy | 12.2 ms | 4697.5 ms | 426 | 0.9250 | 1.0000 | 0.11x |
| 579,012 | autoalgo-rust | 28.0 ms | 962.2 ms | 2,079 | 0.9250 | 1.0000 | 0.55x |
| 579,012 | autoalgo-rust-blas | 28.8 ms | 206.9 ms | 9,666 | 0.9250 | 1.0000 | 2.54x |

## Single-row latency

| train rows | engine | median | p95 | speedup vs sklearn |
|---:|---|---:|---:|---:|
| 10,000 | sklearn | 743 µs | 1,345 µs | 1.00x |
| 10,000 | autoalgo-numpy | 93 µs | 125 µs | 7.98x |
| 10,000 | autoalgo-rust | 47 µs | 49 µs | 15.88x |
| 10,000 | autoalgo-rust-blas | 37 µs | 38 µs | 20.29x |
| 200,000 | sklearn | 2,613 µs | 3,326 µs | 1.00x |
| 200,000 | autoalgo-numpy | 1,342 µs | 1,972 µs | 1.95x |
| 200,000 | autoalgo-rust | 747 µs | 839 µs | 3.50x |
| 200,000 | autoalgo-rust-blas | 859 µs | 904 µs | 3.04x |
