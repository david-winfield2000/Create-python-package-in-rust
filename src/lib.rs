use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Matrix multiplication in Rust
#[pyfunction]
fn matmul(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();

    // Ensure dimensions match for matrix multiplication: A(n x k) * B(k x m) = C(n x m)
    if a[0].len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid matrix dimensions"));
    }

    let mut result = vec![vec![0; m]; n];

    for i in 0..n {
        for j in 0..m {
            for l in 0..k {
                result[i][j] += a[i][l] * b[l][j];
            }
        }
    }

    Ok(result)
}

/// Create a Python module
#[pymodule]
fn my_rust_library(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
