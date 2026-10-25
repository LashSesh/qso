//! The geometry of parametrised states: state derivatives, the quantum
//! Fisher information, quantum natural gradient and variational
//! imaginary-time evolution.
//!
//! For `|ψ(θ)⟩ = U(θ)|ψ₀⟩` the quantum Fisher information (QFI) is
//!
//! ```text
//! F_ij = 4 Re(⟨∂_iψ|∂_jψ⟩ − ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩),
//! ```
//!
//! four times the Fubini–Study metric. QSO computes all `|∂_kψ⟩` exactly in
//! one forward pass that carries one vector per parameter (cost: parameters
//! × gates).
//!
//! * The quantum natural gradient (Stokes, Izaac, Killoran and Carleo,
//!   Quantum 4, 269 (2020)) steps along `(g + λI)⁻¹ ∇E` with the
//!   Fubini–Study metric `g = F/4`, the steepest descent in the geometry of
//!   states rather than of parameters.
//! * Variational imaginary-time evolution (McArdle et al., npj Quantum Inf.
//!   5, 75 (2019)) follows McLachlan's principle: `A θ̇ = −C` with
//!   `A = F/4` and `C_i = Re⟨∂_iψ|H|ψ⟩ = ∂_iE/2`, so `θ̇ = −2F⁺∇E` (`F⁺` the
//!   pseudo-inverse), integrated with fourth-order Runge–Kutta. When the
//!   ansatz can represent `e^{−Hτ}|ψ₀⟩/‖·‖`, it reproduces that evolution.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::gradients::prepare;
use crate::error::{QsoError, Result};
use crate::linalg::{RMatrix, eigh_real, inner};
use crate::quantum::kernels;
use crate::quantum::{Circuit, Param, PauliSum, StateVector};

/// The final state and `|∂_kψ⟩` for every parameter `k`.
pub fn state_derivatives(
    circuit: &Circuit,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<(StateVector, Vec<Vec<C64>>)> {
    let mut psi = prepare(circuit, &PauliSum::new(), theta, initial)?.into_amplitudes();
    let dim = psi.len();
    let mut derivatives: Vec<Option<Vec<C64>>> = vec![None; theta.len()];
    for inst in circuit.instructions() {
        let angles = inst.angles(theta);
        let mut fresh = Vec::new();
        for (slot, param) in inst.params.iter().enumerate() {
            if let Param::Symbol(p) = param {
                let mut mu = psi.clone();
                kernels::apply_instruction_derivative(&mut mu, inst, &angles, slot);
                fresh.push((p.index, p.scale, mu));
            }
        }
        kernels::apply_instruction(&mut psi, inst, &angles, false, 0);
        for v in derivatives.iter_mut().flatten() {
            kernels::apply_instruction(v, inst, &angles, false, 0);
        }
        for (k, scale, mu) in fresh {
            let slot = derivatives[k].get_or_insert_with(|| vec![C64::new(0.0, 0.0); dim]);
            for (d, m) in slot.iter_mut().zip(&mu) {
                *d += scale * m;
            }
        }
    }
    let derivatives = derivatives
        .into_iter()
        .map(|d| d.unwrap_or_else(|| vec![C64::new(0.0, 0.0); dim]))
        .collect();
    Ok((StateVector::from_amplitudes(psi)?, derivatives))
}

fn fisher_from(psi: &[C64], derivatives: &[Vec<C64>]) -> RMatrix {
    let p = derivatives.len();
    let overlaps: Vec<C64> = derivatives.iter().map(|d| inner(d, psi)).collect();
    let mut f = RMatrix::zeros(p, p);
    for i in 0..p {
        for j in i..p {
            let v = 4.0
                * (inner(&derivatives[i], &derivatives[j]) - overlaps[i] * overlaps[j].conj()).re;
            f[(i, j)] = v;
            f[(j, i)] = v;
        }
    }
    f
}

/// The quantum Fisher information matrix `F` at `theta`.
pub fn quantum_fisher_information(
    circuit: &Circuit,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<RMatrix> {
    let (psi, d) = state_derivatives(circuit, theta, initial)?;
    Ok(fisher_from(psi.amplitudes(), &d))
}

/// `Σ_{λ_k > cut} v_k v_kᵀ b / (λ_k + shift)` for symmetric `a` with
/// eigenpairs `(λ_k, v_k)`, `cut = rcond · max λ`.
fn solve_psd(a: &RMatrix, b: &[f64], shift: f64, rcond: f64) -> Vec<f64> {
    let (values, vectors) = eigh_real(a);
    let largest = values.iter().cloned().fold(0.0f64, f64::max);
    let cut = rcond * largest;
    let n = b.len();
    let mut x = vec![0.0; n];
    for (k, &lambda) in values.iter().enumerate() {
        if lambda + shift <= cut || lambda + shift <= 0.0 {
            continue;
        }
        let coefficient: f64 =
            (0..n).map(|i| vectors[(i, k)] * b[i]).sum::<f64>() / (lambda + shift);
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += coefficient * vectors[(i, k)];
        }
    }
    x
}

/// Energy, gradient and natural-gradient direction `(F/4 + λI)⁻¹ ∇E`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NaturalGradient {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub direction: Vec<f64>,
}

fn energy_gradient_fisher(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<(f64, Vec<f64>, RMatrix)> {
    if observable.min_qubits() > circuit.n_qubits() {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: circuit.n_qubits(),
        });
    }
    let (psi, d) = state_derivatives(circuit, theta, initial)?;
    let h_psi = observable.apply(&psi)?;
    let value = inner(psi.amplitudes(), &h_psi).re;
    let gradient = d.iter().map(|dk| 2.0 * inner(dk, &h_psi).re).collect();
    Ok((value, gradient, fisher_from(psi.amplitudes(), &d)))
}

/// The natural-gradient direction with Tikhonov `regularization` λ ≥ 0.
pub fn natural_gradient(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
    regularization: f64,
) -> Result<NaturalGradient> {
    if !(regularization >= 0.0 && regularization.is_finite()) {
        return Err(QsoError::invalid(
            "regularization",
            "must be finite and non-negative",
        ));
    }
    let (value, gradient, f) = energy_gradient_fisher(circuit, observable, theta, initial)?;
    let direction = solve_psd(&(f * 0.25), &gradient, regularization, 1e-12);
    Ok(NaturalGradient {
        value,
        gradient,
        direction,
    })
}

/// Settings of [`minimize_natural_gradient`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NaturalGradientOptions {
    pub learning_rate: f64,
    pub regularization: f64,
    pub max_iterations: usize,
    /// Stop when `‖∇E‖` falls below this.
    pub gradient_tolerance: f64,
}

impl Default for NaturalGradientOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            regularization: 1e-3,
            max_iterations: 500,
            gradient_tolerance: 1e-6,
        }
    }
}

impl NaturalGradientOptions {
    pub fn with_learning_rate(mut self, eta: f64) -> Self {
        self.learning_rate = eta;
        self
    }

    pub fn with_regularization(mut self, lambda: f64) -> Self {
        self.regularization = lambda;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_gradient_tolerance(mut self, tol: f64) -> Self {
        self.gradient_tolerance = tol;
        self
    }
}

/// Outcome of [`minimize_natural_gradient`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NaturalGradientResult {
    pub energy: f64,
    pub parameters: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// Energy before each step and at the end.
    pub history: Vec<f64>,
}

/// Minimise `⟨H⟩` by quantum natural gradient descent
/// `θ ← θ − η (F/4 + λI)⁻¹ ∇E`.
pub fn minimize_natural_gradient(
    circuit: &Circuit,
    observable: &PauliSum,
    theta0: &[f64],
    initial: Option<&StateVector>,
    options: &NaturalGradientOptions,
) -> Result<NaturalGradientResult> {
    if !(options.learning_rate > 0.0 && options.learning_rate.is_finite()) {
        return Err(QsoError::invalid(
            "learning_rate",
            "must be positive and finite",
        ));
    }
    let mut theta = theta0.to_vec();
    let mut history = Vec::new();
    let mut converged = false;
    let mut iterations = 0;
    loop {
        let step = natural_gradient(circuit, observable, &theta, initial, options.regularization)?;
        history.push(step.value);
        let norm = step.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm < options.gradient_tolerance {
            converged = true;
            break;
        }
        if iterations >= options.max_iterations {
            break;
        }
        for (t, d) in theta.iter_mut().zip(&step.direction) {
            *t -= options.learning_rate * d;
        }
        iterations += 1;
    }
    Ok(NaturalGradientResult {
        energy: *history.last().expect("at least one evaluation"),
        parameters: theta,
        iterations,
        converged,
        history,
    })
}

/// Outcome of [`imaginary_time_evolution`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ImaginaryTimeResult {
    /// Imaginary times `τ_0 = 0, …, τ_steps = time`.
    pub times: Vec<f64>,
    /// `⟨H⟩` at each time.
    pub energies: Vec<f64>,
    /// Parameters at each time.
    pub parameters: Vec<Vec<f64>>,
}

/// Variational imaginary-time evolution of `circuit(θ)|ψ₀⟩` under
/// `observable` from `theta0` for imaginary time `time`, in `steps`
/// fourth-order Runge–Kutta steps. `rcond` cuts small eigenvalues of `F`
/// in its pseudo-inverse (redundant parameters).
pub fn imaginary_time_evolution(
    circuit: &Circuit,
    observable: &PauliSum,
    theta0: &[f64],
    initial: Option<&StateVector>,
    time: f64,
    steps: usize,
    rcond: f64,
) -> Result<ImaginaryTimeResult> {
    if !(time >= 0.0 && time.is_finite()) || steps == 0 {
        return Err(QsoError::invalid(
            "time",
            "need a finite non-negative time and at least one step",
        ));
    }
    if !(0.0..1.0).contains(&rcond) {
        return Err(QsoError::invalid("rcond", "must lie in [0, 1)"));
    }
    let rate = |theta: &[f64]| -> Result<(f64, Vec<f64>)> {
        let (e, g, f) = energy_gradient_fisher(circuit, observable, theta, initial)?;
        let x = solve_psd(&f, &g, 0.0, rcond.max(1e-12));
        Ok((e, x.into_iter().map(|v| -2.0 * v).collect()))
    };
    let dt = time / steps as f64;
    let mut theta = theta0.to_vec();
    let mut out = ImaginaryTimeResult {
        times: vec![0.0],
        energies: Vec::new(),
        parameters: vec![theta.clone()],
    };
    let axpy = |x: &[f64], a: f64, y: &[f64]| -> Vec<f64> {
        x.iter().zip(y).map(|(xi, yi)| xi + a * yi).collect()
    };
    for s in 0..steps {
        let (e, k1) = rate(&theta)?;
        out.energies.push(e);
        let (_, k2) = rate(&axpy(&theta, dt / 2.0, &k1))?;
        let (_, k3) = rate(&axpy(&theta, dt / 2.0, &k2))?;
        let (_, k4) = rate(&axpy(&theta, dt, &k3))?;
        for i in 0..theta.len() {
            theta[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        out.times.push((s + 1) as f64 * dt);
        out.parameters.push(theta.clone());
    }
    out.energies.push(rate(&theta)?.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::ansatz::{Entanglement, efficient_su2, real_amplitudes};
    use crate::algorithms::models;
    use crate::rng::Rng;

    fn fidelity(c: &Circuit, a: &[f64], b: &[f64]) -> f64 {
        let x = StateVector::from_circuit(c, a).unwrap();
        let y = StateVector::from_circuit(c, b).unwrap();
        inner(x.amplitudes(), y.amplitudes()).norm_sqr()
    }

    #[test]
    fn fisher_information_matches_finite_differences_of_the_fidelity() {
        // F_ij = −2 ∂²f/∂δ_i∂δ_j with f(δ) = |⟨ψ(θ)|ψ(θ+δ)⟩|².
        let c = efficient_su2(3, 1, Entanglement::Linear);
        let mut rng = Rng::seed_from_u64(4);
        let theta: Vec<f64> = (0..c.num_parameters())
            .map(|_| rng.uniform_range(-2.0, 2.0))
            .collect();
        let f = quantum_fisher_information(&c, &theta, None).unwrap();
        let h = 1e-3;
        for i in 0..theta.len() {
            for j in 0..theta.len() {
                let at = |si: f64, sj: f64| {
                    let mut t = theta.clone();
                    t[i] += si * h;
                    t[j] += sj * h;
                    fidelity(&c, &theta, &t)
                };
                let second =
                    (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h);
                assert!(
                    (f[(i, j)] + 2.0 * second).abs() < 1e-5,
                    "F[{i},{j}] = {} vs {}",
                    f[(i, j)],
                    -2.0 * second
                );
            }
        }
        // Shared and scaled parameters: RY(2θ) on |0⟩ has F = 4.
        let mut c = Circuit::new(1);
        let p = c.new_parameter();
        c.ry(0, 2.0 * p);
        let f = quantum_fisher_information(&c, &[0.4], None).unwrap();
        assert!((f[(0, 0)] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn natural_gradient_converges_faster_than_plain_gradient_descent() {
        let h2 = models::h2_sto3g();
        let c = real_amplitudes(2, 2, Entanglement::Linear);
        let exact = h2.hamiltonian.ground_state_energy(2).unwrap();
        let theta0 = vec![0.1, -0.2, 0.3, 0.05, -0.1, 0.2];
        let opts = NaturalGradientOptions::default()
            .with_learning_rate(0.2)
            .with_gradient_tolerance(1e-6);
        let qng = minimize_natural_gradient(&c, &h2.hamiltonian, &theta0, None, &opts).unwrap();
        assert!(qng.converged);
        assert!(
            (qng.energy - exact).abs() < 1e-8,
            "{} vs {exact}",
            qng.energy
        );
        // Plain gradient descent with the same step size.
        let mut theta = theta0.clone();
        let mut steps = 0;
        loop {
            let (_, g) = crate::algorithms::gradients::expectation_and_gradient(
                &c,
                &h2.hamiltonian,
                &theta,
                None,
                crate::algorithms::GradientMethod::Adjoint,
            )
            .unwrap();
            if g.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-6 || steps >= 5000 {
                break;
            }
            for (t, gi) in theta.iter_mut().zip(&g) {
                *t -= 0.2 * gi;
            }
            steps += 1;
        }
        assert!(
            qng.iterations * 2 < steps,
            "QNG {} vs GD {steps}",
            qng.iterations
        );
    }

    #[test]
    fn imaginary_time_evolution_follows_the_exact_evolution() {
        // Real Hamiltonian, real initial state: the normalised exact
        // evolution stays real, and real amplitudes on two qubits (two
        // layers, redundant parameters) can represent every real state.
        let h: PauliSum = "Z0 + 0.6 X0 X1 - 0.4 Z1 + 0.3 X1".parse().unwrap();
        let c = real_amplitudes(2, 2, Entanglement::Linear);
        let theta0 = vec![0.3, -0.5, 0.2, 0.4, 0.1, -0.3];
        let time = 2.0;
        let res = imaginary_time_evolution(&c, &h, &theta0, None, time, 200, 1e-8).unwrap();
        let psi0 = StateVector::from_circuit(&c, &theta0).unwrap();
        let dense = h.to_dense(2).unwrap();
        for (k, &tau) in res.times.iter().enumerate().step_by(20) {
            // e^{−Hτ} = exp(−i H (−iτ)) via the spectral decomposition.
            let (values, vectors) = crate::linalg::eigh(&dense);
            let u = crate::linalg::dense::from_spectrum(&values, &vectors, |e| {
                C64::new((-e * tau).exp(), 0.0)
            });
            let v = &u * nalgebra::DVector::from_column_slice(psi0.amplitudes());
            let norm = v.norm();
            let hv = &dense * &v;
            let e_exact = (v.dotc(&hv)).re / (norm * norm);
            assert!(
                (res.energies[k] - e_exact).abs() < 1e-6,
                "τ = {tau}: {} vs {e_exact}",
                res.energies[k]
            );
        }
        // Imaginary time lowers the energy monotonically.
        for w in res.energies.windows(2) {
            assert!(w[1] <= w[0] + 1e-12);
        }
        assert!(imaginary_time_evolution(&c, &h, &theta0, None, -1.0, 10, 1e-8).is_err());
    }
}
