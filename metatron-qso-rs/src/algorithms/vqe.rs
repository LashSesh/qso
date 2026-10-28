//! Variational quantum eigensolver.
//!
//! Minimises `E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩` over the parameters of an ansatz circuit
//! with exact (adjoint) gradients and seeded multi-start.
//!
//! ```
//! use metatron_qso::algorithms::{ansatz, models, vqe::Vqe};
//!
//! let h2 = models::h2_sto3g();
//! let circuit = ansatz::real_amplitudes(2, 1, ansatz::Entanglement::Linear);
//! let result = Vqe::new(h2.hamiltonian.clone(), circuit).unwrap().with_restarts(4).run().unwrap();
//! let total = result.energy + h2.nuclear_repulsion;
//! assert!((total - h2.reference_total_energy).abs() < 1e-5);
//! ```

use serde::{Deserialize, Serialize};

use super::gradients::{
    GradientMethod, expectation, expectation_and_gradient, expectation_and_gradient_single,
    expectation_single,
};
use crate::error::{QsoError, Result};
use crate::optimize::{Method, Objective, Options, minimize};
use crate::parallel::map_tasks;
use crate::quantum::{Circuit, MAX_QUBITS, PauliSum, Precision, StateVector, StateVector32};
use crate::rng::Rng;

/// VQE problem and solver configuration.
#[derive(Clone, Debug)]
pub struct Vqe {
    hamiltonian: PauliSum,
    ansatz: Circuit,
    method: Method,
    options: Options,
    gradient: GradientMethod,
    restarts: usize,
    seed: u64,
    init_range: f64,
    initial_parameters: Option<Vec<f64>>,
    initial_state: Option<StateVector>,
    precision: Precision,
}

/// Outcome of [`Vqe::run`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VqeResult {
    /// Lowest energy found.
    pub energy: f64,
    pub parameters: Vec<f64>,
    /// Iterations of the best run.
    pub iterations: usize,
    /// Objective evaluations summed over all restarts.
    pub evaluations: usize,
    pub converged: bool,
    /// Energy after each iteration of the best run.
    pub history: Vec<f64>,
    /// Final energy of every restart.
    pub restart_energies: Vec<f64>,
}

struct Energy<'a> {
    vqe: &'a Vqe,
}

impl Objective for Energy<'_> {
    fn dim(&self) -> usize {
        self.vqe.ansatz.num_parameters()
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.vqe.energy(x).unwrap_or(f64::INFINITY)
    }

    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        self.vqe.energy_and_gradient(x).ok()
    }
}

impl Vqe {
    /// VQE for `hamiltonian` with the parametrised `ansatz`.
    pub fn new(hamiltonian: PauliSum, ansatz: Circuit) -> Result<Self> {
        if ansatz.n_qubits() > MAX_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "VQE",
                max: MAX_QUBITS,
                requested: ansatz.n_qubits(),
            });
        }
        if hamiltonian.min_qubits() > ansatz.n_qubits() {
            return Err(QsoError::QubitOutOfRange {
                qubit: hamiltonian.min_qubits() - 1,
                n_qubits: ansatz.n_qubits(),
            });
        }
        if ansatz.num_parameters() == 0 {
            return Err(QsoError::invalid("ansatz", "ansatz has no parameters"));
        }
        Ok(Self {
            hamiltonian,
            ansatz,
            method: Method::lbfgs(),
            options: Options::default(),
            gradient: GradientMethod::Adjoint,
            restarts: 1,
            seed: 0,
            init_range: std::f64::consts::PI,
            initial_parameters: None,
            initial_state: None,
            precision: Precision::Double,
        })
    }

    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub fn with_options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }

    pub fn with_gradient(mut self, gradient: GradientMethod) -> Self {
        self.gradient = gradient;
        self
    }

    /// Number of independent random starts (run in parallel).
    pub fn with_restarts(mut self, restarts: usize) -> Self {
        self.restarts = restarts.max(1);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Random initial parameters are drawn uniformly from `[−r, r]`.
    pub fn with_init_range(mut self, r: f64) -> Self {
        self.init_range = r.abs();
        self
    }

    /// Use these parameters for the first start.
    pub fn with_initial_parameters(mut self, theta: Vec<f64>) -> Result<Self> {
        self.ansatz.check_parameters(&theta)?;
        self.initial_parameters = Some(theta);
        Ok(self)
    }

    /// Reference state the ansatz acts on (default `|0…0⟩`).
    pub fn with_initial_state(mut self, state: StateVector) -> Result<Self> {
        if state.n_qubits() != self.ansatz.n_qubits() {
            return Err(QsoError::DimensionMismatch {
                expected: self.ansatz.n_qubits(),
                actual: state.n_qubits(),
            });
        }
        self.initial_state = Some(state);
        Ok(self)
    }

    pub fn hamiltonian(&self) -> &PauliSum {
        &self.hamiltonian
    }

    pub fn ansatz(&self) -> &Circuit {
        &self.ansatz
    }

    fn method_uses_gradient(&self) -> bool {
        matches!(
            self.method,
            Method::Lbfgs { .. } | Method::Adam { .. } | Method::GradientDescent { .. }
        )
    }

    /// Simulate in single precision (half the memory, adjoint gradients
    /// only) or double precision (the default).
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// The precision of the simulations.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    fn initial_single(&self) -> Option<StateVector32> {
        self.initial_state.as_ref().map(StateVector32::from_state)
    }

    /// Energy at parameters `theta`.
    pub fn energy(&self, theta: &[f64]) -> Result<f64> {
        match self.precision {
            Precision::Double => expectation(
                &self.ansatz,
                &self.hamiltonian,
                theta,
                self.initial_state.as_ref(),
            ),
            _ => expectation_single(
                &self.ansatz,
                &self.hamiltonian,
                theta,
                self.initial_single().as_ref(),
            ),
        }
    }

    /// Energy and its gradient at parameters `theta` (with the configured
    /// gradient method; single precision needs the adjoint method).
    pub fn energy_and_gradient(&self, theta: &[f64]) -> Result<(f64, Vec<f64>)> {
        match (self.precision, self.gradient) {
            (Precision::Double, method) => expectation_and_gradient(
                &self.ansatz,
                &self.hamiltonian,
                theta,
                self.initial_state.as_ref(),
                method,
            ),
            (_, GradientMethod::Adjoint) => expectation_and_gradient_single(
                &self.ansatz,
                &self.hamiltonian,
                theta,
                self.initial_single().as_ref(),
            ),
            _ => Err(QsoError::invalid(
                "gradient",
                "single precision supports adjoint gradients only",
            )),
        }
    }

    /// Optimised state for parameters `theta`.
    pub fn state(&self, theta: &[f64]) -> Result<StateVector> {
        let mut s = self
            .initial_state
            .clone()
            .unwrap_or_else(|| StateVector::zero(self.ansatz.n_qubits()));
        s.apply_circuit(&self.ansatz, theta)?;
        Ok(s)
    }

    /// Exact ground-state energy of the Hamiltonian on the ansatz register
    /// (Lanczos), for judging the variational result.
    pub fn exact_energy(&self) -> Result<f64> {
        self.hamiltonian.ground_state_energy(self.ansatz.n_qubits())
    }

    /// Run all restarts and return the best result.
    pub fn run(&self) -> Result<VqeResult> {
        crate::optimize::check_restarts(self.restarts)?;
        let dim = self.ansatz.num_parameters();
        // Surface an unsupported gradient method (e.g. parameter shift on a
        // doubly-controlled rotation) instead of silently falling back to
        // finite differences inside the optimiser.
        if self.method_uses_gradient() {
            let probe = vec![0.1; dim];
            self.energy_and_gradient(&probe)?;
        }
        let base = Rng::seed_from_u64(self.seed);
        let runs = map_tasks(self.restarts, |r| {
            let x0 = match (&self.initial_parameters, r) {
                (Some(theta), 0) => theta.clone(),
                _ => {
                    let mut rng = base.fork(r as u64);
                    (0..dim)
                        .map(|_| rng.uniform_range(-self.init_range, self.init_range))
                        .collect()
                }
            };
            minimize(&Energy { vqe: self }, &x0, &self.method, &self.options)
        });
        let mut runs: Vec<_> = runs.into_iter().collect::<Result<Vec<_>>>()?;
        let restart_energies: Vec<f64> = runs.iter().map(|r| r.value).collect();
        let evaluations = runs.iter().map(|r| r.evaluations).sum();
        let best_index = (0..runs.len())
            .min_by(|&a, &b| runs[a].value.total_cmp(&runs[b].value))
            .expect("at least one restart");
        let best = runs.swap_remove(best_index);
        Ok(VqeResult {
            energy: best.value,
            parameters: best.x,
            iterations: best.iterations,
            evaluations,
            converged: best.converged,
            history: best.history,
            restart_energies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::ansatz::{self, Entanglement};
    use crate::algorithms::models;

    #[test]
    fn h2_reaches_chemical_accuracy_and_beyond() {
        let h2 = models::h2_sto3g();
        let circuit = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
        let res = Vqe::new(h2.hamiltonian.clone(), circuit)
            .unwrap()
            .with_restarts(4)
            .run()
            .unwrap();
        assert!(
            (res.energy - (-1.8572750302023795)).abs() < 1e-8,
            "{}",
            res.energy
        );
        assert!(res.converged);
    }

    /// On a problem where double precision converges to the exact ground
    /// energy, single precision does too.
    #[test]
    fn single_precision_reaches_the_exact_energy() {
        let h = models::transverse_field_ising(4, 1.0, 0.8, false);
        let circuit = ansatz::efficient_su2(4, 3, Entanglement::Linear);
        let double = Vqe::new(h, circuit).unwrap().with_restarts(6).with_seed(3);
        let exact = double.exact_energy().unwrap();
        let single = double.clone().with_precision(Precision::Single);
        assert_eq!(single.precision(), Precision::Single);
        for vqe in [&double, &single] {
            let r = vqe.run().unwrap();
            assert!(
                (r.energy - exact).abs() < 1e-5,
                "{:?}: {} vs {exact}",
                vqe.precision(),
                r.energy
            );
        }
        // Parameter shift needs double precision.
        let shift = single.with_gradient(GradientMethod::ParameterShift);
        assert!(shift.run().is_err());
    }

    #[test]
    fn tfim_and_heisenberg_match_exact_diagonalisation() {
        for (h, layers) in [
            (models::transverse_field_ising(4, 1.0, 0.8, false), 3),
            (models::heisenberg_xxz(4, 1.0, 0.5, false), 4),
        ] {
            let circuit = ansatz::efficient_su2(4, layers, Entanglement::Linear);
            let vqe = Vqe::new(h, circuit).unwrap().with_restarts(6).with_seed(3);
            let exact = vqe.exact_energy().unwrap();
            let res = vqe.run().unwrap();
            assert!(res.energy >= exact - 1e-9, "variational principle violated");
            assert!(
                (res.energy - exact).abs() < 1e-5,
                "vqe {} exact {exact}",
                res.energy
            );
        }
    }

    #[test]
    fn methods_and_gradients_are_interchangeable() {
        let h2 = models::h2_sto3g();
        let circuit = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
        for (method, gradient) in [
            (Method::lbfgs(), GradientMethod::ParameterShift),
            (Method::adam(0.1), GradientMethod::Adjoint),
            (Method::nelder_mead(), GradientMethod::Adjoint),
        ] {
            let res = Vqe::new(h2.hamiltonian.clone(), circuit.clone())
                .unwrap()
                .with_method(method.clone())
                .with_gradient(gradient)
                .with_restarts(3)
                .with_options(Options {
                    max_iterations: 3000,
                    ..Default::default()
                })
                .run()
                .unwrap();
            assert!(
                (res.energy + 1.8572750302023795).abs() < 1e-5,
                "{} {}",
                method.name(),
                res.energy
            );
        }
    }

    #[test]
    fn restarts_are_reproducible() {
        let h = models::transverse_field_ising(3, 1.0, 1.0, true);
        let c = ansatz::efficient_su2(3, 2, Entanglement::Circular);
        let a = Vqe::new(h.clone(), c.clone())
            .unwrap()
            .with_restarts(3)
            .with_seed(9)
            .run()
            .unwrap();
        let b = Vqe::new(h, c)
            .unwrap()
            .with_restarts(3)
            .with_seed(9)
            .run()
            .unwrap();
        assert_eq!(a.parameters, b.parameters);
        assert_eq!(a.restart_energies, b.restart_energies);
    }
}
