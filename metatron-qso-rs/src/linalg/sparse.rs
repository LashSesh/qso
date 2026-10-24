//! Compressed sparse row (CSR) matrices with real entries.
//!
//! Graph Hamiltonians (adjacency and Laplacian matrices) are real and
//! symmetric; CSR storage lets the walk module scale to graphs with millions
//! of nodes, using [`CsrMatrix::matvec`] inside Krylov methods.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::parallel::for_each_mut;

/// Square real matrix in CSR format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CsrMatrix {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Build from `(row, col, value)` triplets; duplicates are summed and
    /// explicit zeros dropped.
    pub fn from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> Result<Self> {
        let mut sorted: Vec<(usize, usize, f64)> = Vec::with_capacity(triplets.len());
        for &(r, c, v) in triplets {
            if r >= n || c >= n {
                return Err(QsoError::DimensionMismatch {
                    expected: n,
                    actual: r.max(c) + 1,
                });
            }
            if !v.is_finite() {
                return Err(QsoError::invalid("value", "non-finite matrix entry"));
            }
            sorted.push((r, c, v));
        }
        sorted.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::with_capacity(sorted.len());
        let mut values: Vec<f64> = Vec::with_capacity(sorted.len());
        let mut last: Option<(usize, usize)> = None;
        for (r, c, v) in sorted {
            if last == Some((r, c)) {
                *values.last_mut().expect("previous entry") += v;
            } else {
                indices.push(c);
                values.push(v);
                indptr[r + 1] += 1;
                last = Some((r, c));
            }
        }
        for r in 0..n {
            indptr[r + 1] += indptr[r];
        }
        let mut m = Self {
            n,
            indptr,
            indices,
            values,
        };
        m.drop_zeros();
        Ok(m)
    }

    fn drop_zeros(&mut self) {
        if self.values.iter().all(|&v| v != 0.0) {
            return;
        }
        let mut indptr = vec![0usize; self.n + 1];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..self.n {
            for k in self.indptr[r]..self.indptr[r + 1] {
                if self.values[k] != 0.0 {
                    indices.push(self.indices[k]);
                    values.push(self.values[k]);
                }
            }
            indptr[r + 1] = indices.len();
        }
        self.indptr = indptr;
        self.indices = indices;
        self.values = values;
    }

    /// Dimension `n` of the `n × n` matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate over `(column, value)` of row `r`.
    pub fn row(&self, r: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        (self.indptr[r]..self.indptr[r + 1]).map(move |k| (self.indices[k], self.values[k]))
    }

    /// `y = A·x` for complex vectors.
    pub fn matvec(&self, x: &[C64], y: &mut [C64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        for_each_mut(y, |r, out| {
            let mut acc = C64::new(0.0, 0.0);
            for k in self.indptr[r]..self.indptr[r + 1] {
                acc += x[self.indices[k]] * self.values[k];
            }
            *out = acc;
        });
    }

    /// `y = A·x` for real vectors.
    pub fn matvec_real(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        for_each_mut(y, |r, out| {
            let mut acc = 0.0;
            for k in self.indptr[r]..self.indptr[r + 1] {
                acc += x[self.indices[k]] * self.values[k];
            }
            *out = acc;
        });
    }

    /// Multiply all entries by `s`.
    pub fn scaled(&self, s: f64) -> Self {
        let mut m = self.clone();
        m.values.iter_mut().for_each(|v| *v *= s);
        m.drop_zeros();
        m
    }

    /// Whether `A = Aᵀ` exactly.
    pub fn is_symmetric(&self) -> bool {
        (0..self.n).all(|r| self.row(r).all(|(c, v)| self.get(c, r) == v))
    }

    /// Entry `A[r, c]` (binary search within the row).
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let cols = &self.indices[self.indptr[r]..self.indptr[r + 1]];
        match cols.binary_search(&c) {
            Ok(k) => self.values[self.indptr[r] + k],
            Err(_) => 0.0,
        }
    }

    /// Dense copy (for small matrices).
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut m = nalgebra::DMatrix::zeros(self.n, self.n);
        for r in 0..self.n {
            for (c, v) in self.row(r) {
                m[(r, c)] = v;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_are_merged_and_multiplied() {
        let m = CsrMatrix::from_triplets(
            3,
            &[
                (0, 1, 1.0),
                (1, 0, 1.0),
                (0, 1, 1.0),
                (2, 2, 3.0),
                (1, 1, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.get(0, 1), 2.0);
        assert!(!m.is_symmetric());
        let x = [C64::new(1.0, 0.0), C64::new(0.0, 1.0), C64::new(2.0, 0.0)];
        let mut y = [C64::new(0.0, 0.0); 3];
        m.matvec(&x, &mut y);
        assert_eq!(y[0], C64::new(0.0, 2.0));
        assert_eq!(y[1], C64::new(1.0, 0.0));
        assert_eq!(y[2], C64::new(6.0, 0.0));
    }
}
