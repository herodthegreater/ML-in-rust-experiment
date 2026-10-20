//! Single-precision matrix multiply for the "-blas" engines.
//!
//! On macOS this calls Apple's Accelerate framework, which can use the chip's
//! dedicated matrix hardware that hand-written SIMD code cannot reach. Everywhere
//! else it calls the `matrixmultiply` crate, a pure-Rust multiply that picks
//! AVX2/FMA or NEON kernels at runtime, so nothing has to be installed on Linux.
//! `NAME` records which one ran, so benchmark results say what they measured.

#[cfg(target_os = "macos")]
pub const NAME: &str = "Apple Accelerate";
#[cfg(not(target_os = "macos"))]
pub const NAME: &str = "matrixmultiply (pure Rust)";

#[cfg(target_os = "macos")]
mod imp {
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

#[cfg(not(target_os = "macos"))]
mod imp {
    pub fn gemm_abt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert!(a.len() >= m * k && b.len() >= n * k && c.len() >= m * n);
        // a is m x k row-major (row stride k); b is n x k row-major, read as its
        // k x n transpose (row stride 1, column stride k); c is m x n row-major.
        // SAFETY: the assert above guarantees every buffer covers those extents.
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0,
                a.as_ptr(), k as isize, 1,
                b.as_ptr(), 1, k as isize,
                0.0, c.as_mut_ptr(), n as isize, 1,
            );
        }
    }
}

/// c (m x n) = a (m x k) times the transpose of b (n x k), all row-major.
pub use imp::gemm_abt;
