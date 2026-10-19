//! Single-precision matrix multiply. On macOS it goes through Apple's
//! Accelerate framework, which can use the chip's dedicated matrix hardware
//! that hand-written NEON code cannot reach. Elsewhere it falls back to plain
//! loops so the crate still builds; those paths are correct but not fast.

pub const AVAILABLE: bool = cfg!(target_os = "macos");

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
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|t| a[i * k + t] * b[j * k + t]).sum();
            }
        }
    }
}

/// c (m x n) = a (m x k) times the transpose of b (n x k), all row-major.
pub use imp::gemm_abt;
