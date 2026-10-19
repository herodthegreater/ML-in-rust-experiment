//! k-means (Lloyd's algorithm), the whole loop in Rust.
//!
//! Each iteration splits the rows into blocks that run in parallel. A block
//! scores its rows against every centroid with one Accelerate matrix
//! multiply, picks the nearest centroid, and adds the row into that block's
//! running sums, all in one pass over the data. The per-block sums are then
//! combined, in a fixed order so results are reproducible, into the new
//! centroids. Python is only touched at the start and the end.

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::accelerate;

const ROW_BLOCK: usize = 2048;

struct Partial {
    sums: Vec<f64>,
    counts: Vec<u64>,
    inertia: f64,
    changed: usize,
}

impl Partial {
    fn zero(k: usize, d: usize) -> Self {
        Partial { sums: vec![0.0; k * d], counts: vec![0; k], inertia: 0.0, changed: 0 }
    }

    fn merge(mut self, other: Partial) -> Partial {
        self.sums.iter_mut().zip(&other.sums).for_each(|(a, b)| *a += b);
        self.counts.iter_mut().zip(&other.counts).for_each(|(a, b)| *a += b);
        self.inertia += other.inertia;
        self.changed += other.changed;
        self
    }
}

/// Assign every row to its nearest centroid; return the sums for the next update.
fn assign(x: &[f32], x_sq: &[f32], centers: &[f32], labels: &mut [i64], d: usize, k: usize) -> Partial {
    let c_sq: Vec<f32> = centers.chunks_exact(d).map(|c| c.iter().map(|v| v * v).sum()).collect();
    labels
        .par_chunks_mut(ROW_BLOCK)
        .zip(x.par_chunks(ROW_BLOCK * d))
        .zip(x_sq.par_chunks(ROW_BLOCK))
        .map(|((lab, rows), rows_sq)| {
            let m = lab.len();
            let mut dots = vec![0.0f32; m * k];
            accelerate::gemm_abt(rows, centers, &mut dots, m, k, d);
            let mut p = Partial::zero(k, d);
            for r in 0..m {
                let mut best = 0;
                let mut best_s = f32::INFINITY;
                for j in 0..k {
                    let s = c_sq[j] - 2.0 * dots[r * k + j];
                    if s < best_s {
                        best_s = s;
                        best = j;
                    }
                }
                if lab[r] != best as i64 {
                    p.changed += 1;
                    lab[r] = best as i64;
                }
                p.counts[best] += 1;
                p.inertia += (best_s + rows_sq[r]).max(0.0) as f64;
                let sum = &mut p.sums[best * d..(best + 1) * d];
                for (s, v) in sum.iter_mut().zip(&rows[r * d..(r + 1) * d]) {
                    *s += *v as f64;
                }
            }
            p
        })
        .collect::<Vec<_>>()
        // Merge in block order, not thread-finish order, so floating-point sums
        // (and therefore every result) are identical from run to run.
        .into_iter()
        .fold(Partial::zero(k, d), Partial::merge)
}

/// Run Lloyd iterations from `init`. Returns (centers, labels, n_iter, inertia).
#[pyfunction]
pub fn kmeans_lloyd<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    init: PyReadonlyArray2<'py, f32>,
    max_iter: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>, usize, f64)> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let k = init.shape()[0];
    if init.shape()[1] != d {
        return Err(PyValueError::new_err("init has the wrong number of features"));
    }
    if k == 0 || k > n {
        return Err(PyValueError::new_err("n_clusters must be between 1 and the number of rows"));
    }
    let data = x.as_slice()?;
    let mut centers = init.as_slice()?.to_vec();

    let (labels, n_iter, inertia) = py.allow_threads(|| {
        let x_sq: Vec<f32> = data.par_chunks(d).map(|r| r.iter().map(|v| v * v).sum()).collect();
        let mut labels = vec![-1i64; n];
        let mut n_iter = 0;
        let mut converged = false;
        let mut inertia = 0.0;
        for it in 0..max_iter {
            let p = assign(data, &x_sq, &centers, &mut labels, d, k);
            n_iter = it + 1;
            inertia = p.inertia;
            // Same labels as last time means the centers would not move: done.
            if it > 0 && p.changed == 0 {
                converged = true;
                break;
            }
            for j in 0..k {
                if p.counts[j] > 0 {
                    for f in 0..d {
                        centers[j * d + f] = (p.sums[j * d + f] / p.counts[j] as f64) as f32;
                    }
                }
            }
        }
        if !converged {
            // Labels and inertia must describe the final centers.
            inertia = assign(data, &x_sq, &centers, &mut labels, d, k).inertia;
        }
        (labels, n_iter, inertia)
    });

    let centers = PyArray1::from_vec(py, centers).reshape([k, d])?;
    Ok((centers, labels.into_pyarray(py), n_iter, inertia))
}
