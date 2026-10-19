//! Histogram decision tree (classification, Gini impurity).
//!
//! Features arrive pre-binned to 0..=255 (the Python side computes the bin
//! edges), so finding the best split at a node is: count (bin, class) pairs
//! for each feature, sweep the bins left to right, and score each cut. That
//! is branchy, integer-heavy work that GPUs and BLAS can't help with, which
//! is exactly where compiled code is supposed to pay off.
//!
//! The split rule is written to match the NumPy engine bit for bit: the same
//! integer counts, the same float64 score, and ties broken toward the lowest
//! feature and then the lowest bin. Both engines grow identical trees.

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const BINS: usize = 256;
/// Below this many (rows x features) at a node, splitting work across threads costs more than it saves.
const PARALLEL_WORK: usize = 1 << 16;
/// A split must improve the score by more than this to count.
const MIN_GAIN: f64 = 1e-9;

struct Builder<'a> {
    cols: &'a [u8],
    y: &'a [i64],
    n: usize,
    d: usize,
    c: usize,
    max_depth: usize,
    min_leaf: u64,
    feature: Vec<i64>,
    threshold: Vec<i64>,
    left: Vec<i64>,
    right: Vec<i64>,
    counts: Vec<i64>,
}

impl Builder<'_> {
    /// Best cut for one feature: (score, bin). Higher score means purer children.
    fn best_cut(&self, f: usize, idx: &[u32], totals: &[u64], n_node: u64) -> (f64, usize) {
        let c = self.c;
        let col = &self.cols[f * self.n..(f + 1) * self.n];
        let mut hist = vec![0u64; BINS * c];
        for &i in idx {
            let i = i as usize;
            hist[col[i] as usize * c + self.y[i] as usize] += 1;
        }
        let mut left = vec![0u64; c];
        let mut n_left = 0u64;
        let mut best = (f64::NEG_INFINITY, 0);
        for t in 0..BINS - 1 {
            for k in 0..c {
                left[k] += hist[t * c + k];
                n_left += hist[t * c + k];
            }
            let n_right = n_node - n_left;
            if n_left < self.min_leaf || n_right < self.min_leaf {
                continue;
            }
            let (mut sl, mut sr) = (0u64, 0u64);
            for k in 0..c {
                let r = totals[k] - left[k];
                sl += left[k] * left[k];
                sr += r * r;
            }
            let score = sl as f64 / n_left as f64 + sr as f64 / n_right as f64;
            if score > best.0 {
                best = (score, t);
            }
        }
        best
    }

    fn build(&mut self, idx: &mut [u32], depth: usize) -> i64 {
        let node = self.feature.len() as i64;
        let c = self.c;
        let mut totals = vec![0u64; c];
        for &i in idx.iter() {
            totals[self.y[i as usize] as usize] += 1;
        }
        self.feature.push(-1);
        self.threshold.push(-1);
        self.left.push(-1);
        self.right.push(-1);
        self.counts.extend(totals.iter().map(|&t| t as i64));

        let n_node = idx.len() as u64;
        let pure = totals.iter().filter(|&&t| t > 0).count() <= 1;
        if depth >= self.max_depth || n_node < 2 * self.min_leaf || pure {
            return node;
        }

        let parent: f64 = totals.iter().map(|&t| t * t).sum::<u64>() as f64 / n_node as f64;
        let cuts: Vec<(f64, usize)> = if idx.len() * self.d >= PARALLEL_WORK {
            (0..self.d).into_par_iter().map(|f| self.best_cut(f, idx, &totals, n_node)).collect()
        } else {
            (0..self.d).map(|f| self.best_cut(f, idx, &totals, n_node)).collect()
        };
        let mut best_f = 0;
        for f in 1..self.d {
            if cuts[f].0 > cuts[best_f].0 {
                best_f = f;
            }
        }
        let (score, t) = cuts[best_f];
        if !(score > parent + MIN_GAIN) {
            return node;
        }

        // Partition rows in place: bin <= t goes left.
        let col = &self.cols[best_f * self.n..(best_f + 1) * self.n];
        let mut mid = 0;
        for j in 0..idx.len() {
            if (col[idx[j] as usize] as usize) <= t {
                idx.swap(mid, j);
                mid += 1;
            }
        }
        self.feature[node as usize] = best_f as i64;
        self.threshold[node as usize] = t as i64;
        let (l, r) = idx.split_at_mut(mid);
        let left = self.build(l, depth + 1);
        let right = self.build(r, depth + 1);
        self.left[node as usize] = left;
        self.right[node as usize] = right;
        node
    }
}

type TreeArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<i64>>,
);

/// Grow a tree on binned features. Returns (feature, threshold, left, right, class_counts).
#[pyfunction]
pub fn tree_fit<'py>(
    py: Python<'py>,
    x_binned: PyReadonlyArray2<'py, u8>,
    y: PyReadonlyArray1<'py, i64>,
    n_classes: usize,
    max_depth: usize,
    min_samples_leaf: usize,
) -> PyResult<TreeArrays<'py>> {
    let (n, d) = (x_binned.shape()[0], x_binned.shape()[1]);
    if n == 0 || d == 0 || n > u32::MAX as usize {
        return Err(PyValueError::new_err("need between 1 and 2^32 rows and at least one feature"));
    }
    let xb = x_binned.as_slice()?;
    let labels = y.as_slice()?;
    if labels.iter().any(|&v| v < 0 || v as usize >= n_classes) {
        return Err(PyValueError::new_err("labels must be in 0..n_classes"));
    }

    let (feature, threshold, left, right, counts) = py.allow_threads(|| {
        // Column-major copy so each feature's bins sit next to each other in memory.
        let mut cols = vec![0u8; n * d];
        cols.par_chunks_mut(n).enumerate().for_each(|(f, col)| {
            for (i, v) in col.iter_mut().enumerate() {
                *v = xb[i * d + f];
            }
        });
        let mut b = Builder {
            cols: &cols,
            y: labels,
            n,
            d,
            c: n_classes,
            max_depth,
            min_leaf: min_samples_leaf.max(1) as u64,
            feature: Vec::new(),
            threshold: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            counts: Vec::new(),
        };
        let mut idx: Vec<u32> = (0..n as u32).collect();
        b.build(&mut idx, 0);
        (b.feature, b.threshold, b.left, b.right, b.counts)
    });

    let n_nodes = feature.len();
    Ok((
        feature.into_pyarray(py),
        threshold.into_pyarray(py),
        left.into_pyarray(py),
        right.into_pyarray(py),
        PyArray1::from_vec(py, counts).reshape([n_nodes, n_classes])?,
    ))
}

/// Walk each row down the tree and return the class index of its leaf.
/// Works on raw feature values: a split "bin <= t" is stored as "value < cut",
/// where cut is the bin edge, so prediction never has to bin its input.
#[pyfunction]
pub fn tree_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    feature: PyReadonlyArray1<'py, i64>,
    cut: PyReadonlyArray1<'py, f32>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    leaf_class: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice()?;
    let (feat, cut) = (feature.as_slice()?, cut.as_slice()?);
    let (lft, rgt, cls) = (left.as_slice()?, right.as_slice()?, leaf_class.as_slice()?);
    if feat.iter().any(|&f| f >= d as i64) {
        return Err(PyValueError::new_err("tree uses more features than the input has"));
    }
    let mut out = vec![0i64; n];
    py.allow_threads(|| {
        out.par_chunks_mut(1024).enumerate().for_each(|(chunk, o)| {
            for (r, slot) in o.iter_mut().enumerate() {
                let row = &xs[(chunk * 1024 + r) * d..(chunk * 1024 + r + 1) * d];
                let mut node = 0usize;
                while feat[node] >= 0 {
                    node = if row[feat[node] as usize] < cut[node] {
                        lft[node] as usize
                    } else {
                        rgt[node] as usize
                    };
                }
                *slot = cls[node];
            }
        });
    });
    Ok(out.into_pyarray(py))
}
