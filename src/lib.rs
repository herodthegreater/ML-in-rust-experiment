//! autoalgo's compiled engines, exposed to Python as `autoalgo._rust`.

mod accelerate;
mod kmeans;
mod knn;
mod tree;

use pyo3::prelude::*;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn::knn_predict, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans::kmeans_lloyd, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_fit, m)?)?;
    m.add_function(wrap_pyfunction!(tree::tree_predict, m)?)?;
    Ok(())
}
