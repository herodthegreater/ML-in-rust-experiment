//! k-nearest neighbors prediction, written from scratch.
//!
//! Queries are split into blocks that run in parallel on every core. Inside a
//! block, the training set is walked in tiles small enough to stay in cache,
//! and each training row is scored against four queries at once so one load
//! of the row feeds four dot products. The dot products use fixed-width
//! chunks so the compiler can turn them into NEON (SIMD) instructions.

use numpy::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const LANES: usize = 8;
const QUERY_BLOCK: usize = 16;
const TRAIN_TILE: usize = 512;

/// The k best (smallest) scores seen so far, kept sorted ascending.
struct TopK {
    dist: Vec<f32>,
    idx: Vec<u32>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK { dist: vec![f32::INFINITY; k], idx: vec![0; k] }
    }

    #[inline(always)]
    fn push(&mut self, d: f32, i: u32) {
        let k = self.dist.len();
        if d >= self.dist[k - 1] {
            return;
        }
        let mut pos = k - 1;
        while pos > 0 && self.dist[pos - 1] > d {
            self.dist[pos] = self.dist[pos - 1];
            self.idx[pos] = self.idx[pos - 1];
            pos -= 1;
        }
        self.dist[pos] = d;
        self.idx[pos] = i;
    }
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        let x: &[f32; LANES] = x.try_into().unwrap();
        let y: &[f32; LANES] = y.try_into().unwrap();
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Dot products of one training row against four queries, sharing each load of the row.
#[inline(always)]
fn dot4(q: [&[f32]; 4], x: &[f32]) -> [f32; 4] {
    let mut acc = [[0.0f32; LANES]; 4];
    let xs = x.chunks_exact(LANES);
    let rem = xs.remainder().len();
    let qs = q.map(|qi| qi.chunks_exact(LANES));
    let [mut q0, mut q1, mut q2, mut q3] = qs;
    for xc in xs {
        let xc: &[f32; LANES] = xc.try_into().unwrap();
        let c0: &[f32; LANES] = q0.next().unwrap().try_into().unwrap();
        let c1: &[f32; LANES] = q1.next().unwrap().try_into().unwrap();
        let c2: &[f32; LANES] = q2.next().unwrap().try_into().unwrap();
        let c3: &[f32; LANES] = q3.next().unwrap().try_into().unwrap();
        for l in 0..LANES {
            let xv = xc[l];
            acc[0][l] += c0[l] * xv;
            acc[1][l] += c1[l] * xv;
            acc[2][l] += c2[l] * xv;
            acc[3][l] += c3[l] * xv;
        }
    }
    let mut out = [0.0f32; 4];
    let tail = x.len() - rem;
    for r in 0..4 {
        let mut s: f32 = acc[r].iter().sum();
        for j in tail..x.len() {
            s += q[r][j] * x[j];
        }
        out[r] = s;
    }
    out
}

fn vote(top: &TopK, y: &[i64], n_classes: usize, counts: &mut [u32]) -> i64 {
    counts.iter_mut().for_each(|c| *c = 0);
    for &i in &top.idx {
        counts[y[i as usize] as usize] += 1;
    }
    // First maximum wins, so ties go to the smallest class (matches scikit-learn).
    let mut best = 0;
    for c in 1..n_classes {
        if counts[c] > counts[best] {
            best = c;
        }
    }
    best as i64
}

fn predict_block(
    train: &[f32],
    sq: &[f32],
    y: &[i64],
    n_classes: usize,
    queries: &[f32],
    d: usize,
    k: usize,
    out: &mut [i64],
) {
    let n_train = sq.len();
    let nq = queries.len() / d;
    let mut tops: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
    let row = |i: usize| &queries[i * d..(i + 1) * d];

    for tile_start in (0..n_train).step_by(TRAIN_TILE) {
        let tile_end = (tile_start + TRAIN_TILE).min(n_train);
        let mut qi = 0;
        while qi + 4 <= nq {
            let group = [row(qi), row(qi + 1), row(qi + 2), row(qi + 3)];
            for j in tile_start..tile_end {
                let dots = dot4(group, &train[j * d..(j + 1) * d]);
                for r in 0..4 {
                    tops[qi + r].push(sq[j] - 2.0 * dots[r], j as u32);
                }
            }
            qi += 4;
        }
        for q in qi..nq {
            let qr = row(q);
            for j in tile_start..tile_end {
                let s = sq[j] - 2.0 * dot(qr, &train[j * d..(j + 1) * d]);
                tops[q].push(s, j as u32);
            }
        }
    }

    let mut counts = vec![0u32; n_classes];
    for (q, top) in tops.iter().enumerate() {
        out[q] = vote(top, y, n_classes, &mut counts);
    }
}

/// Apple's Accelerate framework: its matrix multiply can use the chip's
/// dedicated matrix hardware, which hand-written NEON code cannot reach.
#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::{c_float, c_int};

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: c_int, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int,
            alpha: c_float, a: *const c_float, lda: c_int, b: *const c_float, ldb: c_int,
            beta: c_float, c: *mut c_float, ldc: c_int,
        );
    }

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    /// c (m x n) = a (m x k) times the transpose of b (n x k), all row-major.
    pub fn gemm_abt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert!(a.len() >= m * k && b.len() >= n * k && c.len() >= m * n);
        // SAFETY: the assert above guarantees every buffer covers the extent BLAS touches.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS, m as c_int, n as c_int, k as c_int, 1.0,
                a.as_ptr(), k as c_int, b.as_ptr(), k as c_int, 0.0, c.as_mut_ptr(), n as c_int,
            );
        }
    }
}

const BLAS_TILE: usize = 4096;
const BLAS_QUERY_BLOCK: usize = 128;

/// Same result as `predict_block`, but each tile of dot products is one
/// Accelerate matrix multiply; Rust only does the top-k scan and the vote.
#[cfg(target_os = "macos")]
fn predict_block_blas(
    train: &[f32],
    sq: &[f32],
    y: &[i64],
    n_classes: usize,
    queries: &[f32],
    d: usize,
    k: usize,
    out: &mut [i64],
) {
    let n_train = sq.len();
    let nq = queries.len() / d;
    let mut tops: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
    let mut dots = vec![0.0f32; nq * BLAS_TILE];

    for tile_start in (0..n_train).step_by(BLAS_TILE) {
        let tile_end = (tile_start + BLAS_TILE).min(n_train);
        let nt = tile_end - tile_start;
        accelerate::gemm_abt(queries, &train[tile_start * d..tile_end * d], &mut dots, nq, nt, d);
        for (q, top) in tops.iter_mut().enumerate() {
            let row = &dots[q * nt..(q + 1) * nt];
            for (t, &v) in row.iter().enumerate() {
                top.push(sq[tile_start + t] - 2.0 * v, (tile_start + t) as u32);
            }
        }
    }

    let mut counts = vec![0u32; n_classes];
    for (q, top) in tops.iter().enumerate() {
        out[q] = vote(top, y, n_classes, &mut counts);
    }
}

/// Predict class indices for `x_query` from a fitted training set.
/// `use_blas` routes the distance math through Apple Accelerate (macOS only).
#[pyfunction]
#[pyo3(signature = (x_train, sq_norms, y, n_classes, x_query, k, use_blas=false))]
fn knn_predict<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f32>,
    sq_norms: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, i64>,
    n_classes: usize,
    x_query: PyReadonlyArray2<'py, f32>,
    k: usize,
    use_blas: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if use_blas && !cfg!(target_os = "macos") {
        return Err(PyValueError::new_err("the Accelerate path is only available on macOS"));
    }
    let d = x_train.shape()[1];
    let n_train = x_train.shape()[0];
    if x_query.shape()[1] != d {
        return Err(PyValueError::new_err("feature count mismatch"));
    }
    if k == 0 || k > n_train {
        return Err(PyValueError::new_err("k must be between 1 and the training size"));
    }
    if n_train > u32::MAX as usize {
        return Err(PyValueError::new_err("training set too large"));
    }
    let train = x_train.as_slice()?;
    let sq = sq_norms.as_slice()?;
    let labels = y.as_slice()?;
    let queries = x_query.as_slice()?;
    let n_query = x_query.shape()[0];

    let mut out = vec![0i64; n_query];
    py.allow_threads(|| {
        #[cfg(target_os = "macos")]
        if use_blas {
            out.par_chunks_mut(BLAS_QUERY_BLOCK)
                .zip(queries.par_chunks(BLAS_QUERY_BLOCK * d.max(1)))
                .for_each(|(o, q)| predict_block_blas(train, sq, labels, n_classes, q, d, k, o));
            return;
        }
        out.par_chunks_mut(QUERY_BLOCK)
            .zip(queries.par_chunks(QUERY_BLOCK * d.max(1)))
            .for_each(|(o, q)| predict_block(train, sq, labels, n_classes, q, d, k, o));
    });
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_predict, m)?)?;
    Ok(())
}
