//! Excited states: variational quantum deflation and subspace-search VQE.
//!
//! * VQD (Higgott, Wang and Brierley, Quantum 3, 156 (2019)) finds the
//!   states one after another. State `k` minimises
//!   `⟨H⟩_θ + Σ_{j<k} β_j |⟨ψ_j|ψ(θ)⟩|²`; with every `β_j` larger than the
//!   gap `E_k − E_j`, the minimum is the k-th eigenstate. The default
//!   `β = 3 Σ|c|` (three times the coefficient one-norm of `H`) exceeds
//!   every gap, which is at most `2 Σ|c|`.
//! * Weighted SSVQE (Nakanishi, Mitarai and Fujii, Phys. Rev. Research 1,
//!   033062 (2019)) applies one circuit to the orthogonal basis states
//!   `|0⟩, |1⟩, …, |k−1⟩` and minimises `Σ_j w_j ⟨j|U†HU|j⟩` with weights
//!   `w_0 > w_1 > …`; at the minimum `U|j⟩` is the j-th eigenstate.
//!
//! Both use L-BFGS with exact adjoint gradients (the overlap penalty of VQD
//! is the observable `Σ β_j |ψ_j⟩⟨ψ_j|`) and seeded random restarts.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::GradientMethod;
use super::gradients::{adjoint_with, expectation_and_gradient, prepare};
use crate::error::{QsoError, Result};
use crate::linalg::inner;
use crate::optimize::{Method, Objective, Options, minimize};
use crate::parallel::map_tasks;
use crate::quantum::{Circuit, MAX_QUBITS, PauliSum, StateVector};
use crate::rng::Rng;

/// Settings shared by [`vqd`] and [`ssvqe`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExcitedStateOptions {
    /// Random restarts per optimisation (the best is kept).
    pub restarts: usize,
    pub seed: u64,
    /// VQD penalties `β_j` (default `3 Σ|c|` for every state).
    pub betas: Option<Vec<f64>>,
    /// SSVQE weights (default `k, k−1, …, 1`).
    pub weights: Option<Vec<f64>>,
    pub optimizer: Options,
}

impl Default for ExcitedStateOptions {
    fn default() -> Self {
        Self {
            restarts: 4,
            seed: 0,
            betas: None,
            weights: None,
            optimizer: Options::default(),
        }
    }
}

impl ExcitedStateOptions {
    pub fn with_restarts(mut self, restarts: usize) -> Self {
        self.restarts = restarts;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_betas(mut self, betas: Vec<f64>) -> Self {
        self.betas = Some(betas);
        self
    }

    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }
}

/// Energies and parameters of the states found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExcitedStatesResult {
    /// `⟨H⟩` of each state (without penalties).
    pub energies: Vec<f64>,
    /// Parameters per state (VQD), or the one shared vector repeated
    /// (SSVQE).
    pub parameters: Vec<Vec<f64>>,
}

fn check(
    h: &PauliSum,
    ansatz: &Circuit,
    n_states: usize,
    options: &ExcitedStateOptions,
) -> Result<()> {
    if ansatz.n_qubits() > MAX_QUBITS {
        return Err(QsoError::TooManyQubits {
            what: "excited-state solver",
            max: MAX_QUBITS,
            requested: ansatz.n_qubits(),
        });
    }
    if h.min_qubits() > ansatz.n_qubits() {
        return Err(QsoError::QubitOutOfRange {
            qubit: h.min_qubits() - 1,
            n_qubits: ansatz.n_qubits(),
        });
    }
    if ansatz.num_parameters() == 0 {
        return Err(QsoError::invalid("ansatz", "ansatz has no parameters"));
    }
    if n_states == 0 || n_states > 1 << ansatz.n_qubits() {
        return Err(QsoError::invalid(
            "n_states",
            format!("need 1 to 2^{} states", ansatz.n_qubits()),
        ));
    }
    if options.restarts == 0 {
        return Err(QsoError::invalid("restarts", "need at least one"));
    }
    crate::optimize::check_restarts(options.restarts)?;
    if let Some(w) = &options.weights
        && w.iter()
            .take(n_states)
            .zip(w.iter().skip(1).take(n_states.saturating_sub(1)))
            .any(|(a, b)| a <= b)
    {
        return Err(QsoError::invalid(
            "weights",
            "SSVQE needs strictly decreasing weights; with equal weights the energies are not eigenvalues",
        ));
    }
    for (what, v) in [("betas", &options.betas), ("weights", &options.weights)] {
        if let Some(v) = v
            && (v.len() < n_states || v.iter().any(|x| !(x.is_finite() && *x > 0.0)))
        {
            return Err(QsoError::invalid(
                what,
                format!("need {n_states} finite positive values"),
            ));
        }
    }
    Ok(())
}

/// Best of `restarts` L-BFGS runs from random starts in `[−π, π]`.
fn optimise(
    objective: &dyn Objective,
    options: &ExcitedStateOptions,
    stream: u64,
) -> Result<Vec<f64>> {
    let dim = objective.dim();
    let base = Rng::seed_from_u64(options.seed).fork(stream);
    let runs = map_tasks(options.restarts, |r| {
        let mut rng = base.fork(r as u64);
        let x0: Vec<f64> = (0..dim)
            .map(|_| rng.uniform_range(-std::f64::consts::PI, std::f64::consts::PI))
            .collect();
        minimize(objective, &x0, &Method::lbfgs(), &options.optimizer)
    });
    let runs = runs.into_iter().collect::<Result<Vec<_>>>()?;
    let best = runs
        .into_iter()
        .min_by(|a, b| a.value.total_cmp(&b.value))
        .expect("at least one restart");
    Ok(best.x)
}

struct Deflated<'a> {
    h: &'a PauliSum,
    ansatz: &'a Circuit,
    found: &'a [(f64, Vec<C64>)],
}

impl Objective for Deflated<'_> {
    fn dim(&self) -> usize {
        self.ansatz.num_parameters()
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.value_and_gradient(x).map_or(f64::INFINITY, |v| v.0)
    }

    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        let mut psi = prepare(self.ansatz, self.h, x, None).ok()?;
        psi.apply_circuit(self.ansatz, x).ok()?;
        Some(adjoint_with(
            self.ansatz,
            x,
            psi.into_amplitudes(),
            |v, out| {
                self.h.apply_amplitudes(v, out);
                for (beta, phi) in self.found {
                    let overlap = inner(phi, v) * beta;
                    for (o, p) in out.iter_mut().zip(phi) {
                        *o += overlap * p;
                    }
                }
            },
        ))
    }
}

/// The lowest `n_states` eigenstates of `h` by VQD with `ansatz`.
pub fn vqd(
    h: &PauliSum,
    ansatz: &Circuit,
    n_states: usize,
    options: &ExcitedStateOptions,
) -> Result<ExcitedStatesResult> {
    check(h, ansatz, n_states, options)?;
    // E_k − E_j ≤ 2Σ|c|, with equality for a single Pauli string, so the
    // default exceeds every gap by a margin.
    let default_beta = 3.0
        * h.terms()
            .iter()
            .filter(|(_, s)| !s.is_identity())
            .map(|(c, _)| c.abs())
            .sum::<f64>();
    let mut found: Vec<(f64, Vec<C64>)> = Vec::new();
    let mut result = ExcitedStatesResult {
        energies: Vec::new(),
        parameters: Vec::new(),
    };
    for k in 0..n_states {
        let objective = Deflated {
            h,
            ansatz,
            found: &found,
        };
        let x = optimise(&objective, options, k as u64)?;
        let psi = StateVector::from_circuit(ansatz, &x)?;
        result.energies.push(h.expectation(&psi)?);
        result.parameters.push(x);
        let beta = options
            .betas
            .as_ref()
            .map_or(default_beta.max(1e-3), |b| b[k]);
        found.push((beta, psi.into_amplitudes()));
    }
    Ok(result)
}

struct Subspace<'a> {
    h: &'a PauliSum,
    ansatz: &'a Circuit,
    inputs: Vec<StateVector>,
    weights: Vec<f64>,
}

impl Objective for Subspace<'_> {
    fn dim(&self) -> usize {
        self.ansatz.num_parameters()
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.value_and_gradient(x).map_or(f64::INFINITY, |v| v.0)
    }

    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        let mut value = 0.0;
        let mut grad = vec![0.0; x.len()];
        for (input, w) in self.inputs.iter().zip(&self.weights) {
            let (e, g) = expectation_and_gradient(
                self.ansatz,
                self.h,
                x,
                Some(input),
                GradientMethod::Adjoint,
            )
            .ok()?;
            value += w * e;
            for (a, b) in grad.iter_mut().zip(&g) {
                *a += w * b;
            }
        }
        Some((value, grad))
    }
}

/// The lowest `n_states` eigenvalues of `h` by weighted SSVQE with
/// `ansatz` applied to `|0⟩, …, |n_states − 1⟩`.
pub fn ssvqe(
    h: &PauliSum,
    ansatz: &Circuit,
    n_states: usize,
    options: &ExcitedStateOptions,
) -> Result<ExcitedStatesResult> {
    check(h, ansatz, n_states, options)?;
    let n = ansatz.n_qubits();
    let inputs = (0..n_states)
        .map(|j| StateVector::basis(n, j))
        .collect::<Result<Vec<_>>>()?;
    let weights = options
        .weights
        .clone()
        .map(|w| w[..n_states].to_vec())
        .unwrap_or_else(|| (0..n_states).map(|j| (n_states - j) as f64).collect());
    let objective = Subspace {
        h,
        ansatz,
        inputs,
        weights,
    };
    let x = optimise(&objective, options, 0)?;
    let mut result = ExcitedStatesResult {
        energies: Vec::new(),
        parameters: Vec::new(),
    };
    for input in &objective.inputs {
        let mut psi = input.clone();
        psi.apply_circuit(ansatz, &x)?;
        result.energies.push(h.expectation(&psi)?);
        result.parameters.push(x.clone());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::ansatz::{Entanglement, efficient_su2};
    use crate::algorithms::models;
    use crate::linalg::eigh;

    #[test]
    fn vqd_and_ssvqe_match_exact_diagonalisation() {
        // H2 (two qubits): all four levels; transverse-field Ising (three
        // qubits): the three lowest.
        let cases = [
            (
                models::h2_sto3g().hamiltonian,
                efficient_su2(2, 2, Entanglement::Linear),
                4,
            ),
            (
                models::transverse_field_ising(3, 1.0, 0.7, false),
                // SSVQE maps three inputs at once and needs more layers.
                efficient_su2(3, 4, Entanglement::Full),
                3,
            ),
        ];
        for (h, ansatz, k) in cases {
            let n = ansatz.n_qubits();
            let (exact, _) = eigh(&h.to_dense(n).unwrap());
            let opts = ExcitedStateOptions::default().with_restarts(6).with_seed(2);
            let v = vqd(&h, &ansatz, k, &opts).unwrap();
            let s = ssvqe(&h, &ansatz, k, &opts).unwrap();
            for (j, e) in exact.iter().enumerate().take(k) {
                assert!(
                    (v.energies[j] - e).abs() < 1e-6,
                    "VQD {j}: {} vs {}",
                    v.energies[j],
                    e
                );
                assert!(
                    (s.energies[j] - e).abs() < 1e-6,
                    "SSVQE {j}: {} vs {}",
                    s.energies[j],
                    e
                );
            }
        }
    }

    #[test]
    fn default_penalty_exceeds_the_largest_gap() {
        // H = Z0 has the gap 2 = 2Σ|c|; a penalty of exactly 2Σ|c| leaves
        // the deflated landscape flat.
        let h: PauliSum = "Z0".parse().unwrap();
        let ansatz = crate::algorithms::ansatz::real_amplitudes(1, 1, Entanglement::Linear);
        for seed in 0..4 {
            let r = vqd(
                &h,
                &ansatz,
                2,
                &ExcitedStateOptions::default().with_seed(seed),
            )
            .unwrap();
            assert!(
                (r.energies[0] + 1.0).abs() < 1e-8 && (r.energies[1] - 1.0).abs() < 1e-6,
                "{r:?}"
            );
        }
    }

    #[test]
    fn deflation_gradient_matches_finite_differences() {
        let h = models::transverse_field_ising(2, 1.0, 0.5, false);
        let ansatz = efficient_su2(2, 1, Entanglement::Linear);
        let phi = StateVector::from_circuit(&ansatz, &vec![0.3; ansatz.num_parameters()]).unwrap();
        let found = vec![(1.7, phi.into_amplitudes())];
        let obj = Deflated {
            h: &h,
            ansatz: &ansatz,
            found: &found,
        };
        let x: Vec<f64> = (0..ansatz.num_parameters())
            .map(|i| 0.2 * i as f64 - 0.5)
            .collect();
        let (v, g) = obj.value_and_gradient(&x).unwrap();
        let e = h
            .expectation(&StateVector::from_circuit(&ansatz, &x).unwrap())
            .unwrap();
        let psi = StateVector::from_circuit(&ansatz, &x).unwrap();
        let overlap = inner(&found[0].1, psi.amplitudes()).norm_sqr();
        assert!((v - (e + 1.7 * overlap)).abs() < 1e-12);
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += 1e-6;
            let mut xm = x.clone();
            xm[i] -= 1e-6;
            let fd = (obj.value(&xp) - obj.value(&xm)) / 2e-6;
            assert!((fd - g[i]).abs() < 1e-6);
        }
        let opts = ExcitedStateOptions::default().with_betas(vec![1.0]);
        assert!(vqd(&h, &ansatz, 2, &opts).is_err());
        assert!(vqd(&h, &ansatz, 0, &ExcitedStateOptions::default()).is_err());
        assert!(ssvqe(&h, &Circuit::new(2), 1, &ExcitedStateOptions::default()).is_err());
    }
}
