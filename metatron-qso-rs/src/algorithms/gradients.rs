//! Gradients of expectation values of parametrised circuits.
//!
//! For `E(θ) = ⟨ψ₀|U(θ)† H U(θ)|ψ₀⟩` with `U = U_N ⋯ U_1`:
//!
//! * [`GradientMethod::Adjoint`] — adjoint differentiation (Jones & Gacon,
//!   arXiv:2009.02823). One forward pass and one backward pass with three
//!   state vectors give *all* partial derivatives exactly, at the cost of
//!   roughly three circuit simulations, independent of the parameter count.
//! * [`GradientMethod::ParameterShift`] — the hardware-compatible shift
//!   rules: two evaluations per gate parameter for gates `exp(−iθG/2)` with
//!   `G² = I` and for phase gates, four for single-controlled rotations
//!   (generator spectrum {0, ±½}).
//! * [`GradientMethod::FiniteDifference`] — central differences (reference
//!   only).

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::linalg::inner;
use crate::quantum::kernels;
use crate::quantum::{Circuit, GateKind, MAX_QUBITS, Param, PauliSum, StateVector};

/// How gradients are computed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum GradientMethod {
    #[default]
    Adjoint,
    ParameterShift,
    FiniteDifference {
        step: f64,
    },
}

fn prepare(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<StateVector> {
    circuit.check_parameters(theta)?;
    if circuit.n_qubits() > MAX_QUBITS {
        return Err(QsoError::TooManyQubits {
            what: "state-vector simulation",
            max: MAX_QUBITS,
            requested: circuit.n_qubits(),
        });
    }
    let state = match initial {
        Some(s) => {
            if s.n_qubits() != circuit.n_qubits() {
                return Err(QsoError::DimensionMismatch {
                    expected: circuit.n_qubits(),
                    actual: s.n_qubits(),
                });
            }
            s.clone()
        }
        None => StateVector::zero(circuit.n_qubits()),
    };
    if observable.min_qubits() > circuit.n_qubits() {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: circuit.n_qubits(),
        });
    }
    Ok(state)
}

/// `⟨ψ(θ)|H|ψ(θ)⟩` where `|ψ(θ)⟩ = U(θ)|ψ₀⟩` (`|ψ₀⟩ = |0…0⟩` by default).
pub fn expectation(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<f64> {
    let mut state = prepare(circuit, observable, theta, initial)?;
    state.apply_circuit(circuit, theta)?;
    observable.expectation(&state)
}

/// Expectation value and its gradient with respect to `theta`.
pub fn expectation_and_gradient(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
    method: GradientMethod,
) -> Result<(f64, Vec<f64>)> {
    match method {
        GradientMethod::Adjoint => adjoint(circuit, observable, theta, initial),
        GradientMethod::ParameterShift => parameter_shift(circuit, observable, theta, initial),
        GradientMethod::FiniteDifference { step } => {
            let value = expectation(circuit, observable, theta, initial)?;
            let mut grad = vec![0.0; theta.len()];
            let mut shifted = theta.to_vec();
            for i in 0..theta.len() {
                shifted[i] = theta[i] + step;
                let plus = expectation(circuit, observable, &shifted, initial)?;
                shifted[i] = theta[i] - step;
                let minus = expectation(circuit, observable, &shifted, initial)?;
                shifted[i] = theta[i];
                grad[i] = (plus - minus) / (2.0 * step);
            }
            Ok((value, grad))
        }
    }
}

fn adjoint(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<(f64, Vec<f64>)> {
    let mut phi = prepare(circuit, observable, theta, initial)?;
    phi.apply_circuit(circuit, theta)?;
    let mut phi = phi.into_amplitudes();
    let mut lambda = vec![C64::new(0.0, 0.0); phi.len()];
    observable.apply_amplitudes(&phi, &mut lambda);
    let value = inner(&phi, &lambda).re;
    let mut grad = vec![0.0; theta.len()];
    let mut mu = vec![C64::new(0.0, 0.0); phi.len()];
    for inst in circuit.instructions().iter().rev() {
        let angles = inst.angles(theta);
        let inverse = inst.inverse();
        let inv_angles = inverse.angles(theta);
        // φ ← U_k† φ  (state before gate k)
        kernels::apply_instruction(&mut phi, &inverse, &inv_angles, false, 0);
        if inst.is_parametrised() {
            for (slot, param) in inst.params.iter().enumerate() {
                if let Param::Symbol(p) = param {
                    mu.copy_from_slice(&phi);
                    kernels::apply_instruction_derivative(&mut mu, inst, &angles, slot);
                    grad[p.index] += 2.0 * p.scale * inner(&lambda, &mu).re;
                }
            }
        }
        // λ ← U_k† λ
        kernels::apply_instruction(&mut lambda, &inverse, &inv_angles, false, 0);
    }
    Ok((value, grad))
}

fn parameter_shift(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<(f64, Vec<f64>)> {
    let value = expectation(circuit, observable, theta, initial)?;
    let bound = circuit.bind(theta)?;
    let mut grad = vec![0.0; theta.len()];
    let eval_with = |k: usize, slot: usize, shift: f64| -> Result<f64> {
        let mut c = bound.clone();
        let inst = &mut c.instructions_mut()[k];
        if let Param::Value(v) = inst.params[slot] {
            inst.params[slot] = Param::Value(v + shift);
        }
        expectation(&c, observable, &[], initial)
    };
    let half_pi = std::f64::consts::FRAC_PI_2;
    for (k, inst) in circuit.instructions().iter().enumerate() {
        for (slot, param) in inst.params.iter().enumerate() {
            let Param::Symbol(p) = param else { continue };
            let n_controls = inst.controls.len();
            let derivative = match (inst.kind, n_controls) {
                (
                    GateKind::RX
                    | GateKind::RY
                    | GateKind::RZ
                    | GateKind::RXX
                    | GateKind::RYY
                    | GateKind::RZZ,
                    0,
                )
                | (GateKind::U, 0)
                | (GateKind::P, _) => {
                    (eval_with(k, slot, half_pi)? - eval_with(k, slot, -half_pi)?) / 2.0
                }
                (GateKind::RX | GateKind::RY | GateKind::RZ, 1) => {
                    // Four-term rule for generator spectrum {0, ±1/2}.
                    let s2 = 2f64.sqrt();
                    let cp = (s2 + 1.0) / (4.0 * s2);
                    let cm = (s2 - 1.0) / (4.0 * s2);
                    cp * (eval_with(k, slot, half_pi)? - eval_with(k, slot, -half_pi)?)
                        - cm * (eval_with(k, slot, 3.0 * half_pi)?
                            - eval_with(k, slot, -3.0 * half_pi)?)
                }
                _ => {
                    return Err(QsoError::invalid(
                        "circuit",
                        format!(
                            "no parameter-shift rule for {}{}; use adjoint differentiation",
                            "c".repeat(n_controls),
                            inst.kind.name()
                        ),
                    ));
                }
            };
            grad[p.index] += p.scale * derivative;
        }
    }
    Ok((value, grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn test_circuit() -> Circuit {
        let mut c = Circuit::new(3);
        let p = c.new_parameters(6);
        c.h(0).ry(1, p[0]).rx(2, 2.0 * p[1] + 0.3).cx(0, 1);
        c.rzz(1, 2, p[2])
            .crx(0, 2, p[3])
            .cry(2, 1, -p[0])
            .crz(1, 0, p[4]);
        c.u(1, p[5], p[1], -0.4)
            .p(2, p[2])
            .cp(0, 1, p[4])
            .ryy(0, 2, p[3])
            .rxx(1, 0, 0.5 * p[5]);
        c.ccx(0, 1, 2).rz(0, p[1]);
        c
    }

    #[test]
    fn adjoint_parameter_shift_and_finite_differences_agree() {
        let c = test_circuit();
        let h: PauliSum = "0.7 Z0 Z1 - 0.4 X2 + 0.25 Y0 Y1 X2 + 1.1 Z1 - 0.3"
            .parse()
            .unwrap();
        let mut rng = Rng::seed_from_u64(12);
        for _ in 0..4 {
            let theta: Vec<f64> = (0..6).map(|_| rng.uniform_range(-3.0, 3.0)).collect();
            let (e_adj, g_adj) =
                expectation_and_gradient(&c, &h, &theta, None, GradientMethod::Adjoint).unwrap();
            let (e_fd, g_fd) = expectation_and_gradient(
                &c,
                &h,
                &theta,
                None,
                GradientMethod::FiniteDifference { step: 1e-5 },
            )
            .unwrap();
            assert!((e_adj - e_fd).abs() < 1e-12);
            for (a, b) in g_adj.iter().zip(&g_fd) {
                assert!((a - b).abs() < 1e-7, "adjoint {a} vs fd {b}");
            }
        }
    }

    #[test]
    fn parameter_shift_is_exact_for_supported_gates() {
        let mut c = Circuit::new(2);
        let p = c.new_parameters(5);
        c.ry(0, p[0])
            .rx(1, p[1])
            .cx(0, 1)
            .rzz(0, 1, 3.0 * p[2])
            .crx(1, 0, p[3])
            .cp(0, 1, p[4])
            .u(1, p[0], p[1], p[2]);
        let h: PauliSum = "Z0 + 0.5 X0 X1 - 0.2 Y1".parse().unwrap();
        let theta = [0.3, -1.2, 0.8, 2.1, -0.6];
        let (_, g_shift) =
            expectation_and_gradient(&c, &h, &theta, None, GradientMethod::ParameterShift).unwrap();
        let (_, g_adj) =
            expectation_and_gradient(&c, &h, &theta, None, GradientMethod::Adjoint).unwrap();
        for (a, b) in g_shift.iter().zip(&g_adj) {
            assert!((a - b).abs() < 1e-12, "shift {a} vs adjoint {b}");
        }
        // Unsupported gates are reported, not silently mis-differentiated.
        let mut c2 = Circuit::new(3);
        let q = c2.new_parameter();
        c2.controlled(GateKind::RY, &[2], &[0, 1], &[q.into()])
            .unwrap();
        assert!(
            expectation_and_gradient(&c2, &h, &[0.1], None, GradientMethod::ParameterShift)
                .is_err()
        );
        assert!(expectation_and_gradient(&c2, &h, &[0.1], None, GradientMethod::Adjoint).is_ok());
    }

    #[test]
    fn custom_initial_state_is_respected() {
        let mut c = Circuit::new(1);
        let t = c.new_parameter();
        c.ry(0, t);
        let z: PauliSum = "Z0".parse().unwrap();
        let one = StateVector::basis(1, 1).unwrap();
        // RY(θ)|1⟩ → ⟨Z⟩ = −cos θ, derivative sin θ.
        let (e, g) =
            expectation_and_gradient(&c, &z, &[0.7], Some(&one), GradientMethod::Adjoint).unwrap();
        assert!((e + 0.7f64.cos()).abs() < 1e-12);
        assert!((g[0] - 0.7f64.sin()).abs() < 1e-12);
    }
}
