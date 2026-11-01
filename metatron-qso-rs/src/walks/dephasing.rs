//! Quantum walks with site dephasing.
//!
//! The walker's density matrix obeys the Lindblad equation
//!
//! ```text
//! dρ/dt = −i[H, ρ] + κ Σ_j (|j⟩⟨j| ρ |j⟩⟨j| − ½{|j⟩⟨j|, ρ})
//!       = −i[H, ρ] − κ (ρ − diag ρ)
//! ```
//!
//! with dephasing rate `κ`. For `κ = 0` this is the coherent walk; for
//! `κ ≫ γ` coherences are suppressed and the populations follow the
//! classical master equation `dp/dt = −(2γ²/κ)·L·p` (quantum Zeno regime).
//! On a connected graph the unique steady state is the maximally mixed
//! state, i.e. the uniform distribution.
//!
//! Integration uses the classical Runge–Kutta method with step doubling and
//! Richardson extrapolation; the local error per step is kept below the
//! requested tolerance.

use num_complex::Complex64 as C64;

use super::{DENSE_LIMIT, Generator};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::CMatrix;

/// Open quantum walk with site dephasing.
#[derive(Clone, Debug)]
pub struct DephasingWalk {
    n: usize,
    h: CMatrix,
    rate: f64,
    tol: f64,
}

impl DephasingWalk {
    /// Walk on `graph` with hopping rate `gamma` and dephasing rate `rate`.
    pub fn new(graph: &Graph, generator: Generator, gamma: f64, rate: f64) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 || n > DENSE_LIMIT / 4 {
            return Err(QsoError::invalid(
                "graph",
                format!("need 1..={} nodes", DENSE_LIMIT / 4),
            ));
        }
        if !(gamma.is_finite() && gamma > 0.0) || !(rate.is_finite() && rate >= 0.0) {
            return Err(QsoError::invalid(
                "gamma/rate",
                "need gamma > 0 and rate ≥ 0",
            ));
        }
        let h_real = match generator {
            Generator::Laplacian => graph.laplacian_matrix() * gamma,
            Generator::Adjacency => graph.adjacency_matrix() * -gamma,
        };
        Ok(Self {
            n,
            h: h_real.map(|x| C64::new(x, 0.0)),
            rate,
            tol: 1e-11,
        })
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Set the per-step error tolerance (default 1e-11).
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-15);
        self
    }

    fn derivative(&self, rho: &CMatrix) -> CMatrix {
        let i = C64::new(0.0, 1.0);
        let comm = &self.h * rho - rho * &self.h;
        let mut d = comm * -i;
        if self.rate > 0.0 {
            for r in 0..self.n {
                for c in 0..self.n {
                    if r != c {
                        d[(r, c)] -= rho[(r, c)] * self.rate;
                    }
                }
            }
        }
        d
    }

    fn rk4(&self, rho: &CMatrix, dt: f64) -> CMatrix {
        let k1 = self.derivative(rho);
        let k2 = self.derivative(&(rho + &k1 * C64::new(dt / 2.0, 0.0)));
        let k3 = self.derivative(&(rho + &k2 * C64::new(dt / 2.0, 0.0)));
        let k4 = self.derivative(&(rho + &k3 * C64::new(dt, 0.0)));
        rho + (k1 + k2 * C64::new(2.0, 0.0) + k3 * C64::new(2.0, 0.0) + k4)
            * C64::new(dt / 6.0, 0.0)
    }

    /// Evolve the density matrix `rho0` for time `t ≥ 0`.
    pub fn evolve(&self, rho0: &CMatrix, t: f64) -> Result<CMatrix> {
        if rho0.nrows() != self.n || rho0.ncols() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: rho0.nrows(),
            });
        }
        if !(t >= 0.0 && t.is_finite()) {
            return Err(QsoError::invalid(
                "t",
                "time must be finite and non-negative",
            ));
        }
        let scale = self.h.iter().map(|z| z.norm()).fold(0.0, f64::max) * self.n as f64 + self.rate;
        let mut dt = (0.1 / scale.max(1e-12)).min(t.max(1e-12));
        let mut rho = rho0.clone();
        let mut now = 0.0;
        let mut guard = 0usize;
        while now < t {
            guard += 1;
            if guard > 10_000_000 {
                return Err(QsoError::NoConvergence {
                    method: "dephasing RK4",
                    iterations: guard,
                    residual: t - now,
                });
            }
            let step = dt.min(t - now);
            let full = self.rk4(&rho, step);
            let half = self.rk4(&self.rk4(&rho, step / 2.0), step / 2.0);
            let diff = &half - &full;
            let err = diff.iter().map(|z| z.norm()).fold(0.0, f64::max) / 15.0;
            if err <= self.tol || step < 1e-14 {
                rho = half + diff * C64::new(1.0 / 15.0, 0.0);
                now += step;
                let grow = if err > 0.0 {
                    0.9 * (self.tol / err).powf(0.2)
                } else {
                    2.0
                };
                dt = step * grow.clamp(0.2, 2.0);
            } else {
                dt = step * (0.9 * (self.tol / err).powf(0.2)).clamp(0.1, 0.9);
            }
        }
        Ok(rho)
    }

    /// Node populations at time `t` for a walker started at `start`.
    pub fn populations(&self, start: usize, t: f64) -> Result<Vec<f64>> {
        if start >= self.n {
            return Err(QsoError::NodeOutOfRange {
                node: start,
                n_nodes: self.n,
            });
        }
        let mut rho0 = CMatrix::zeros(self.n, self.n);
        rho0[(start, start)] = C64::new(1.0, 0.0);
        let rho = self.evolve(&rho0, t)?;
        Ok((0..self.n).map(|j| rho[(j, j)].re).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{generators, metatron};
    use crate::walks::{ContinuousWalk, classical_walk_probabilities};

    #[test]
    fn zero_dephasing_reproduces_coherent_walk() {
        let g = metatron::fruit_of_life();
        let open = DephasingWalk::new(&g, Generator::Laplacian, 1.0, 0.0).unwrap();
        let closed = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        let a = open.populations(0, 2.7).unwrap();
        let b = closed.probabilities(0, 2.7).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn dephasing_preserves_trace_and_hermiticity_and_mixes_to_uniform() {
        let g = metatron::fruit_of_life();
        let walk = DephasingWalk::new(&g, Generator::Laplacian, 1.0, 0.5).unwrap();
        let mut rho0 = CMatrix::zeros(13, 13);
        rho0[(7, 7)] = C64::new(1.0, 0.0);
        let rho = walk.evolve(&rho0, 3.0).unwrap();
        let trace: C64 = (0..13).map(|j| rho[(j, j)]).sum();
        assert!((trace - C64::new(1.0, 0.0)).norm() < 1e-9);
        assert!((&rho - rho.adjoint()).norm() < 1e-9);
        let p = walk.populations(7, 400.0).unwrap();
        assert!(p.iter().all(|&x| (x - 1.0 / 13.0).abs() < 1e-4), "{p:?}");
    }

    #[test]
    fn strong_dephasing_approaches_classical_walk() {
        // Zeno regime: populations follow dp/dt = −(2γ²/κ)Lp.
        let g = generators::cycle(6);
        let (gamma, kappa) = (1.0, 80.0);
        let walk = DephasingWalk::new(&g, Generator::Laplacian, gamma, kappa)
            .unwrap()
            .with_tolerance(1e-9);
        let t = 30.0;
        let q = walk.populations(0, t).unwrap();
        let c = classical_walk_probabilities(&g, 0, t, 2.0 * gamma * gamma / kappa).unwrap();
        let tv = crate::walks::total_variation(&q, &c);
        assert!(tv < 0.02, "tv = {tv}");
    }
}
