//! Variational quantum classifier with data re-uploading
//! (Pérez-Salinas et al., Quantum 4, 226, 2020).
//!
//! Each of `layers` blocks encodes the (scaled) features as `RY(x_i)` on
//! qubit `i mod n`, followed by trainable `RY·RZ` rotations on every qubit
//! and a CX chain. The model output is `p(y = 1 | x) = (1 − ⟨Z₀⟩)/2`.
//! Training minimises the binary cross-entropy over the full batch with
//! exact adjoint gradients.

use serde::{Deserialize, Serialize};

use super::gradients::{GradientMethod, expectation, expectation_and_gradient};
use crate::error::{QsoError, Result};
use crate::optimize::{Method, Objective, Options, minimize};
use crate::parallel::map_tasks;
use crate::quantum::{Circuit, Parameter, Pauli, PauliString, PauliSum};
use crate::rng::Rng;

/// Model architecture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariationalClassifier {
    pub n_qubits: usize,
    pub n_features: usize,
    pub layers: usize,
    /// Trained parameters (empty before training).
    pub parameters: Vec<f64>,
}

/// Training summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingReport {
    pub loss: f64,
    pub accuracy: f64,
    pub iterations: usize,
    pub evaluations: usize,
}

const EPS: f64 = 1e-9;

impl VariationalClassifier {
    pub fn new(n_qubits: usize, n_features: usize, layers: usize) -> Result<Self> {
        if n_qubits == 0 || n_features == 0 || layers == 0 {
            return Err(QsoError::invalid(
                "shape",
                "qubits, features and layers must be positive",
            ));
        }
        Ok(Self {
            n_qubits,
            n_features,
            layers,
            parameters: Vec::new(),
        })
    }

    /// Number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        2 * self.n_qubits * self.layers
    }

    fn circuit(&self, x: &[f64]) -> Circuit {
        let n = self.n_qubits;
        let mut c = Circuit::new(n);
        c.reserve_parameters(self.num_parameters());
        let mut k = 0;
        for _ in 0..self.layers {
            for (i, &xi) in x.iter().enumerate() {
                c.ry(i % n, xi);
            }
            for q in 0..n {
                c.ry(q, Parameter::new(k));
                c.rz(q, Parameter::new(k + 1));
                k += 2;
            }
            for q in 0..n.saturating_sub(1) {
                c.cx(q, q + 1);
            }
        }
        c
    }

    fn readout() -> PauliSum {
        PauliString::single(0, Pauli::Z).into()
    }

    fn check(&self, x: &[f64]) -> Result<()> {
        if x.len() != self.n_features {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_features,
                actual: x.len(),
            });
        }
        Ok(())
    }

    fn probability_with(&self, x: &[f64], theta: &[f64]) -> Result<f64> {
        let z = expectation(&self.circuit(x), &Self::readout(), theta, None)?;
        Ok(((1.0 - z) / 2.0).clamp(0.0, 1.0))
    }

    /// `p(y = 1 | x)` for the trained model.
    pub fn predict_proba(&self, x: &[f64]) -> Result<f64> {
        self.check(x)?;
        if self.parameters.len() != self.num_parameters() {
            return Err(QsoError::invalid(
                "model",
                "classifier has not been trained",
            ));
        }
        self.probability_with(x, &self.parameters)
    }

    /// Class prediction (`p ≥ ½`).
    pub fn predict(&self, x: &[f64]) -> Result<bool> {
        Ok(self.predict_proba(x)? >= 0.5)
    }

    /// Fraction of correctly classified samples.
    pub fn accuracy(&self, xs: &[Vec<f64>], ys: &[bool]) -> Result<f64> {
        let mut correct = 0usize;
        for (x, &y) in xs.iter().zip(ys) {
            if self.predict(x)? == y {
                correct += 1;
            }
        }
        Ok(correct as f64 / xs.len().max(1) as f64)
    }

    /// Train on `(xs, ys)` with L-BFGS from a seeded random start.
    pub fn fit(
        &mut self,
        xs: &[Vec<f64>],
        ys: &[bool],
        options: &Options,
        seed: u64,
    ) -> Result<TrainingReport> {
        if xs.is_empty() || xs.len() != ys.len() {
            return Err(QsoError::invalid(
                "data",
                "need equally many (non-zero) samples and labels",
            ));
        }
        for x in xs {
            self.check(x)?;
        }
        let mut rng = Rng::seed_from_u64(seed);
        let x0: Vec<f64> = (0..self.num_parameters())
            .map(|_| rng.uniform_range(-0.5, 0.5))
            .collect();
        let loss = CrossEntropy {
            model: self,
            xs,
            ys,
        };
        let res = minimize(&loss, &x0, &Method::lbfgs(), options)?;
        self.parameters = res.x;
        Ok(TrainingReport {
            loss: res.value,
            accuracy: self.accuracy(xs, ys)?,
            iterations: res.iterations,
            evaluations: res.evaluations,
        })
    }
}

struct CrossEntropy<'a> {
    model: &'a VariationalClassifier,
    xs: &'a [Vec<f64>],
    ys: &'a [bool],
}

impl Objective for CrossEntropy<'_> {
    fn dim(&self) -> usize {
        self.model.num_parameters()
    }

    fn value(&self, theta: &[f64]) -> f64 {
        self.value_and_gradient(theta)
            .map(|(v, _)| v)
            .unwrap_or(f64::INFINITY)
    }

    fn value_and_gradient(&self, theta: &[f64]) -> Option<(f64, Vec<f64>)> {
        let readout = VariationalClassifier::readout();
        let per_sample = map_tasks(self.xs.len(), |i| {
            let circuit = self.model.circuit(&self.xs[i]);
            expectation_and_gradient(&circuit, &readout, theta, None, GradientMethod::Adjoint)
        });
        let m = self.xs.len() as f64;
        let mut loss = 0.0;
        let mut grad = vec![0.0; theta.len()];
        for (result, &y) in per_sample.into_iter().zip(self.ys) {
            let (z, dz) = result.ok()?;
            let p = ((1.0 - z) / 2.0).clamp(EPS, 1.0 - EPS);
            // L = −[y ln p + (1−y) ln(1−p)],  dL/dp = (p − y)/(p(1−p)),  dp/dz = −½
            let target = if y { 1.0 } else { 0.0 };
            loss -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
            let dl_dz = -0.5 * (p - target) / (p * (1.0 - p));
            grad.iter_mut().zip(&dz).for_each(|(g, d)| *g += dl_dz * d);
        }
        Some((loss / m, grad.into_iter().map(|g| g / m).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<bool>) {
        let mut rng = Rng::seed_from_u64(seed);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..n {
            let y = i % 2 == 0;
            let c = if y { [0.8, 0.8] } else { [-0.8, -0.8] };
            xs.push(vec![c[0] + 0.35 * rng.normal(), c[1] + 0.35 * rng.normal()]);
            ys.push(y);
        }
        (xs, ys)
    }

    fn circle(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<bool>) {
        let mut rng = Rng::seed_from_u64(seed);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        while xs.len() < n {
            let x = [rng.uniform_range(-1.0, 1.0), rng.uniform_range(-1.0, 1.0)];
            let r2 = x[0] * x[0] + x[1] * x[1];
            if (r2 - 0.45).abs() < 0.08 {
                continue; // margin around the boundary
            }
            xs.push(vec![2.0 * x[0], 2.0 * x[1]]);
            ys.push(r2 < 0.45);
        }
        (xs, ys)
    }

    #[test]
    fn learns_linearly_separable_blobs() {
        let (xs, ys) = blobs(40, 1);
        let (xt, yt) = blobs(40, 2);
        let mut model = VariationalClassifier::new(2, 2, 2).unwrap();
        let report = model
            .fit(
                &xs,
                &ys,
                &Options {
                    max_iterations: 200,
                    ..Default::default()
                },
                0,
            )
            .unwrap();
        assert!(report.accuracy >= 0.95, "{report:?}");
        assert!(model.accuracy(&xt, &yt).unwrap() >= 0.9);
    }

    #[test]
    fn learns_nonlinear_circle_with_reuploading() {
        let (xs, ys) = circle(60, 3);
        let (xt, yt) = circle(60, 4);
        let mut model = VariationalClassifier::new(2, 2, 4).unwrap();
        let report = model
            .fit(
                &xs,
                &ys,
                &Options {
                    max_iterations: 400,
                    ..Default::default()
                },
                1,
            )
            .unwrap();
        assert!(report.accuracy >= 0.9, "{report:?}");
        assert!(model.accuracy(&xt, &yt).unwrap() >= 0.85);
    }

    #[test]
    fn untrained_model_refuses_to_predict() {
        let model = VariationalClassifier::new(2, 2, 1).unwrap();
        assert!(model.predict(&[0.0, 0.0]).is_err());
        assert!(VariationalClassifier::new(0, 2, 1).is_err());
    }
}
