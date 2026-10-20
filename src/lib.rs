//! The compiled engines, exposed to Python as `ml_in_rust._rust`.

mod gemm;
mod kmeans;
mod knn;
mod tree;

use pyo3::prelude::*;

/// Which matrix library the "-blas" engines use on this machine.
#[pyfunction]
fn gemm_library() -> &'static str {
    gemm::NAME
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gemm_library, m)?)?;
    m.add_function(wrap_pyfunction!(knn::knn_predict, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans::kmeans_lloyd, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_fit, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_predict, m)?)?;
    Ok(())
}
