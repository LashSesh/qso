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
#[cfg(test)]
use crate::linalg::inner;
use crate::parallel::{SyncPtr, map_tasks, sum_c64};
use crate::quantum::Instruction;
use crate::quantum::gates::{dmatrix_1q, dmatrix_2q, matrix_1q, matrix_2q};
use crate::quantum::kernels;
use crate::quantum::plan::BLOCK_QUBITS;
use crate::quantum::scalar::Amplitude;
use crate::quantum::{
    Circuit, GateKind, MAX_QUBITS, Param, PauliSum, Precision, StateVector, StateVector32,
};

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

pub(crate) fn prepare(
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

/// The initial state of a single-precision evaluation, checked.
fn prepare_single(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector32>,
) -> Result<StateVector32> {
    circuit.check_parameters(theta)?;
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
        None => StateVector32::zero(circuit.n_qubits())?,
    };
    if observable.min_qubits() > circuit.n_qubits() {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: circuit.n_qubits(),
        });
    }
    Ok(state)
}

/// ‖φ‖² of a single-precision state, which must be positive and finite
/// (rounding moves it away from one; a zero or NaN state is an error).
fn norm2_single(phi: &StateVector32) -> Result<f64> {
    let norm2 = phi.norm().powi(2);
    if !(norm2.is_finite() && norm2 > 0.0) {
        return Err(QsoError::invalid(
            "initial",
            "the state's norm is zero or not finite",
        ));
    }
    Ok(norm2)
}

/// An energy or gradient that left the range of single precision.
fn overflowed() -> QsoError {
    QsoError::invalid(
        "observable",
        "the energy or gradient overflows single precision; use double precision",
    )
}

/// [`expectation`] in single precision ([`StateVector32`], half the
/// memory); the sum is taken in double precision and divided by the
/// state's norm², which rounding moves away from one.
pub fn expectation_single(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector32>,
) -> Result<f64> {
    let mut state = prepare_single(circuit, observable, theta, initial)?;
    state.apply_circuit(circuit, theta)?;
    norm2_single(&state)?;
    state.expectation(observable)
}

/// [`expectation_and_gradient`] by adjoint differentiation in single
/// precision: two [`StateVector32`]s, the gradient terms summed in double
/// precision; energy and gradient divided by the state's norm² (as
/// [`StateVector32::expectation`] does). The identity term of the
/// observable is left out of the co-state `H·φ`, where its single-precision
/// rounding would swamp the gradient of a large constant, and added to the
/// energy afterwards.
pub fn expectation_and_gradient_single(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector32>,
) -> Result<(f64, Vec<f64>)> {
    let mut phi = prepare_single(circuit, observable, theta, initial)?;
    phi.apply_circuit(circuit, theta)?;
    let norm2 = norm2_single(&phi)?;
    let constant = observable.constant();
    let varying = if observable.terms().iter().any(|(_, s)| s.is_identity()) {
        std::borrow::Cow::Owned(PauliSum::from_terms(
            observable
                .terms()
                .iter()
                .filter(|(_, s)| !s.is_identity())
                .cloned(),
        ))
    } else {
        std::borrow::Cow::Borrowed(observable)
    };
    let (value, mut grad) = adjoint_with(circuit, theta, phi.into_amplitudes(), |x, y| {
        varying.apply_amplitudes(x, y)
    });
    for g in &mut grad {
        *g /= norm2;
    }
    let value = value / norm2 + constant;
    if !value.is_finite() || grad.iter().any(|g| !g.is_finite()) {
        return Err(overflowed());
    }
    Ok((value, grad))
}

/// [`expectation_and_gradient`] at the given precision; single precision
/// supports the adjoint method only (parameter shift and finite differences
/// need double precision to resolve their differences).
pub fn expectation_and_gradient_at(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    method: GradientMethod,
    precision: Precision,
) -> Result<(f64, Vec<f64>)> {
    match precision {
        Precision::Double => expectation_and_gradient(circuit, observable, theta, None, method),
        _ => match method {
            GradientMethod::Adjoint => {
                expectation_and_gradient_single(circuit, observable, theta, None)
            }
            _ => Err(QsoError::invalid(
                "method",
                "single precision supports adjoint gradients only",
            )),
        },
    }
}

/// [`expectation`] at the given precision.
pub fn expectation_at(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    precision: Precision,
) -> Result<f64> {
    match precision {
        Precision::Double => expectation(circuit, observable, theta, None),
        _ => expectation_single(circuit, observable, theta, None),
    }
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

/// Check every parameter vector before any simulation runs.
fn check_rows(circuit: &Circuit, thetas: &[Vec<f64>]) -> Result<()> {
    thetas.iter().try_for_each(|t| circuit.check_parameters(t))
}

/// Run `f(k)` for `k in 0..count`: in parallel across `k` when one
/// simulation of `n_qubits` qubits is too small to use the threads well
/// (below 16 qubits it uses at most two), one after another (each
/// simulation parallel inside) otherwise. Results do not depend on the
/// number of threads.
fn batch<T, F>(n_qubits: usize, count: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync + Send,
{
    let small = n_qubits < crate::parallel::PAR_THRESHOLD.trailing_zeros() as usize;
    let results: Vec<Result<T>> = if small {
        crate::parallel::map_tasks(count, f)
    } else {
        (0..count).map(f).collect()
    };
    results.into_iter().collect()
}

/// [`expectation`] for many parameter vectors at once.
///
/// For registers below 16 qubits the evaluations run in parallel, one per
/// thread, which is much faster than a loop when each evaluation is too
/// small to use the threads by itself (energy landscapes, grid searches,
/// batches of classifier inputs). Every parameter vector is checked before
/// the first evaluation.
pub fn expectation_batch(
    circuit: &Circuit,
    observable: &PauliSum,
    thetas: &[Vec<f64>],
    initial: Option<&StateVector>,
) -> Result<Vec<f64>> {
    check_rows(circuit, thetas)?;
    batch(circuit.n_qubits(), thetas.len(), |k| {
        expectation(circuit, observable, &thetas[k], initial)
    })
}

/// [`expectation_and_gradient`] for many parameter vectors at once (in
/// parallel as for [`expectation_batch`]).
pub fn expectation_and_gradient_batch(
    circuit: &Circuit,
    observable: &PauliSum,
    thetas: &[Vec<f64>],
    initial: Option<&StateVector>,
    method: GradientMethod,
) -> Result<Vec<(f64, Vec<f64>)>> {
    check_rows(circuit, thetas)?;
    batch(circuit.n_qubits(), thetas.len(), |k| {
        expectation_and_gradient(circuit, observable, &thetas[k], initial, method)
    })
}

fn adjoint(
    circuit: &Circuit,
    observable: &PauliSum,
    theta: &[f64],
    initial: Option<&StateVector>,
) -> Result<(f64, Vec<f64>)> {
    let mut phi = prepare(circuit, observable, theta, initial)?;
    phi.apply_circuit(circuit, theta)?;
    Ok(adjoint_with(
        circuit,
        theta,
        phi.into_amplitudes(),
        |x, y| observable.apply_amplitudes(x, y),
    ))
}

/// Adjoint differentiation of `⟨ψ|O|ψ⟩` for any Hermitian `O` given by
/// `apply(x, y)`: `y = O x`. `phi` is the final state `U(θ)|ψ₀⟩` (checked
/// by the caller).
pub(crate) fn adjoint_with<A, F>(
    circuit: &Circuit,
    theta: &[f64],
    phi: Vec<A>,
    apply: F,
) -> (f64, Vec<f64>)
where
    A: Amplitude,
    F: Fn(&[A], &mut [A]),
{
    adjoint_with_block(circuit, theta, phi, apply, BLOCK_QUBITS)
}

/// [`adjoint_with`] with blocks of `2^c` amplitudes.
fn adjoint_with_block<A, F>(
    circuit: &Circuit,
    theta: &[f64],
    mut phi: Vec<A>,
    apply: F,
    c: usize,
) -> (f64, Vec<f64>)
where
    A: Amplitude,
    F: Fn(&[A], &mut [A]),
{
    let mut lambda = vec![A::ZERO; phi.len()];
    apply(&phi, &mut lambda);
    // Summed in double precision whatever the amplitudes.
    let value = sum_c64(phi.len(), |i| phi[i].to_c64().conj() * lambda[i].to_c64()).re;
    let mut grad = vec![0.0; theta.len()];
    // Gates whose targets lie in a cache block are differentiated block by
    // block (when the register is larger than a block); the others one at
    // a time over the whole register.
    let blocked = phi.len() > 1 << c;
    // Diagonal gates act element by element, so they can be undone and
    // differentiated block by block on any qubits.
    let local = |inst: &Instruction| {
        blocked && (inst.kind.is_diagonal() || inst.targets.iter().all(|&t| t < c))
    };
    let insts = circuit.instructions();
    let mut k = insts.len();
    while k > 0 {
        if local(&insts[k - 1]) {
            let mut j = k - 1;
            while j > 0 && local(&insts[j - 1]) {
                j -= 1;
            }
            backward_blocked(&insts[j..k], theta, &mut phi, &mut lambda, c, &mut grad);
            k = j;
            continue;
        }
        k -= 1;
        let inst = &insts[k];
        let angles = inst.angles(theta);
        let inverse = inst.inverse();
        let inv_angles = inverse.angles(theta);
        // φ ← U_k† φ  (state before gate k)
        kernels::apply_instruction(&mut phi, &inverse, &inv_angles, false, 0);
        if inst.is_parametrised() {
            for (slot, param) in inst.params.iter().enumerate() {
                if let Param::Symbol(p) = param {
                    let term = kernels::derivative_element(
                        &lambda,
                        &phi,
                        inst,
                        &angles,
                        slot,
                        inst.control_mask(),
                    );
                    grad[p.index] += 2.0 * p.scale * term.re;
                }
            }
        }
        // λ ← U_k† λ
        kernels::apply_instruction(&mut lambda, &inverse, &inv_angles, false, 0);
    }
    (value, grad)
}

/// The backward pass over `run` (gates whose targets lie below qubit `c`)
/// block by block: each block of `φ` and `λ` stays in cache while every
/// gate of the run is undone and differentiated on it. The gradient terms
/// are summed over blocks in block order, so the result does not depend
/// on the number of threads.
fn backward_blocked<A: Amplitude>(
    run: &[Instruction],
    theta: &[f64],
    phi: &mut [A],
    lambda: &mut [A],
    c: usize,
    grad: &mut [f64],
) {
    struct Step<'a> {
        inst: &'a Instruction,
        angles: Vec<f64>,
        inverse: Instruction,
        inv_angles: Vec<f64>,
        /// (parameter slot, parameter index, scale) of each derivative.
        slots: Vec<(usize, usize, f64)>,
        low: usize,
        high: usize,
        /// For diagonal gates: the diagonal of the inverse and of each
        /// derivative, by local index (first target the high bit).
        diagonal: Option<(Vec<C64>, Vec<Vec<C64>>)>,
    }
    /// The diagonal of a gate's 2×2 or 4×4 matrix.
    fn diag(n_targets: usize, m: &[C64]) -> Vec<C64> {
        let d = 1 << n_targets;
        (0..d).map(|k| m[k * d + k]).collect()
    }
    let block = 1usize << c;
    let steps: Vec<Step<'_>> = run
        .iter()
        .rev()
        .map(|inst| {
            let inverse = inst.inverse();
            let controls = inst.control_mask();
            let angles = inst.angles(theta);
            let inv_angles = inverse.angles(theta);
            let slots: Vec<(usize, usize, f64)> = inst
                .params
                .iter()
                .enumerate()
                .filter_map(|(slot, p)| match p {
                    Param::Symbol(s) => Some((slot, s.index, s.scale)),
                    Param::Value(_) => None,
                })
                .collect();
            let high_target = inst.targets.iter().any(|&t| t >= c);
            let diagonal = (inst.kind.is_diagonal() && high_target).then(|| {
                let t = inst.targets.len();
                let (inv, derivs) = if t == 1 {
                    let inv = diag(1, &matrix_1q(inverse.kind, &inv_angles));
                    let d = slots
                        .iter()
                        .map(|&(slot, _, _)| diag(1, &dmatrix_1q(inst.kind, &angles, slot)))
                        .collect();
                    (inv, d)
                } else {
                    let inv = diag(2, &matrix_2q(inverse.kind, &inv_angles));
                    let d = slots
                        .iter()
                        .map(|_| diag(2, &dmatrix_2q(inst.kind, &angles)))
                        .collect();
                    (inv, d)
                };
                (inv, derivs)
            });
            Step {
                inst,
                angles,
                inv_angles,
                inverse,
                slots,
                low: controls & (block - 1),
                high: controls & !(block - 1),
                diagonal,
            }
        })
        .collect();
    let (pp, lp) = (SyncPtr::new(phi), SyncPtr::new(lambda));
    let partials: Vec<Vec<f64>> = map_tasks(phi.len() / block, |k| {
        let base = k * block;
        // SAFETY: distinct `k` address disjoint blocks of both registers.
        let (p, l) = unsafe {
            (
                std::slice::from_raw_parts_mut(pp.get().add(base), block),
                std::slice::from_raw_parts_mut(lp.get().add(base), block),
            )
        };
        let mut out = Vec::new();
        for s in &steps {
            if let Some((inv, derivs)) = &s.diagonal {
                // A diagonal gate with a target above the block: the high
                // target bits are fixed by `base`, so in this block the gate
                // is a phase (all targets high) or a diagonal gate on its
                // one low target, applied with the vector kernels.
                if base & s.high != s.high {
                    out.extend(std::iter::repeat_n(0.0, derivs.len()));
                    continue;
                }
                let targets = &s.inst.targets;
                let low_target = targets.iter().copied().find(|&t| t < c);
                // Local index of the gate (first target the high bit) with
                // the low target's bit set to `x`.
                let index = |x: usize| {
                    targets.iter().fold(0, |acc, &t| {
                        let bit = if t < c { x } else { (base >> t) & 1 };
                        (acc << 1) | bit
                    })
                };
                match low_target {
                    None => {
                        // A phase (one for Z, S, T and P where the target
                        // bit is 0, whose derivative is then zero too).
                        let (u, i) = (inv[index(0)], index(0));
                        let one = C64::new(1.0, 0.0);
                        if u != one {
                            kernels::scale_masked(p, s.low, u);
                        }
                        for d in derivs {
                            let term = if d[i] == C64::new(0.0, 0.0) {
                                0.0
                            } else {
                                (d[i] * kernels::inner_masked(l, p, s.low)).re
                            };
                            out.push(term);
                        }
                        if u != one {
                            kernels::scale_masked(l, s.low, u);
                        }
                    }
                    Some(t) => {
                        let zero = C64::new(0.0, 0.0);
                        let u = [inv[index(0)], zero, zero, inv[index(1)]].map(A::from_c64);
                        kernels::apply_1q(p, t, s.low, &u);
                        for d in derivs {
                            let dm = [d[index(0)], zero, zero, d[index(1)]];
                            out.push(kernels::element_1q_matrix(l, p, t, s.low, &dm).re);
                        }
                        kernels::apply_1q(l, t, s.low, &u);
                    }
                }
                continue;
            }
            let on = base & s.high == s.high;
            if on {
                kernels::apply_instruction_masked(p, &s.inverse, &s.inv_angles, false, 0, s.low);
            }
            for &(slot, _, _) in &s.slots {
                if on {
                    let term = kernels::derivative_element(l, p, s.inst, &s.angles, slot, s.low);
                    out.push(term.re);
                } else {
                    // The derivative vanishes where the controls are off.
                    out.push(0.0);
                }
            }
            if on {
                kernels::apply_instruction_masked(l, &s.inverse, &s.inv_angles, false, 0, s.low);
            }
        }
        out
    });
    let mut t = 0;
    for s in &steps {
        for &(_, index, scale) in &s.slots {
            let total: f64 = partials.iter().map(|v| v[t]).sum();
            grad[index] += 2.0 * scale * total;
            t += 1;
        }
    }
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
    #[test]
    fn the_blocked_backward_pass_matches_the_plain_one() {
        use super::*;
        // Blocks of 2³ on 7 qubits: gates inside and across blocks,
        // controls inside and above, gates with several parameters.
        let n = 7;
        let mut c = Circuit::new(n);
        let p = c.new_parameters(46);
        let mut k = 0;
        let mut next = || {
            k += 1;
            p[k - 1]
        };
        for layer in 0..2 {
            for q in 0..n {
                c.ry(q, next()).rz(q, next());
            }
            c.u(1, next(), next(), next());
            c.crx(6, 1, next()).crx(0, 2, next()).cry(5, 4, next());
            for q in 0..n - 1 {
                c.cx(q, q + 1);
            }
            c.rzz(0, 1, next()).rzz(2, 3, next()).rx(5, next());
            if layer == 0 {
                c.h(6).cz(6, 0);
            }
        }
        let theta: Vec<f64> = (0..46).map(|j| 0.1 + 0.37 * j as f64).collect();
        let h: PauliSum = "Z0 Z1 + 0.5 X2 Y6 - 0.3 Y3 Z4 + 0.2 X5".parse().unwrap();
        let mut psi = StateVector::zero(n);
        psi.apply_circuit(&c, &theta).unwrap();
        let phi = psi.into_amplitudes();
        let apply = |x: &[C64], y: &mut [C64]| h.apply_amplitudes(x, y);
        let (v0, g0) = adjoint_with_block(&c, &theta, phi.clone(), apply, 20);
        let (v1, g1) = adjoint_with_block(&c, &theta, phi, apply, 3);
        assert_eq!(v0, v1);
        for (a, b) in g0.iter().zip(&g1) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        assert!(g0.iter().filter(|x| x.abs() > 1e-6).count() > 20);
    }

    /// Random circuits over the gates with shift rules, with diagonal gates
    /// (controlled too) on qubits above the block, shared and scaled
    /// parameters: the blocked pass equals the plain one to 10⁻¹², and both
    /// equal the parameter-shift gradient to 10⁻¹⁰.
    #[test]
    fn random_circuits_match_the_parameter_shift_rule() {
        use super::*;
        let n = 7;
        let mut rng = Rng::seed_from_u64(1260);
        for round in 0..6 {
            let mut c = Circuit::new(n);
            let p = c.new_parameters(8);
            let q = |rng: &mut Rng| rng.below(n as u64) as usize;
            for _ in 0..40 {
                let (a, b) = loop {
                    let (a, b) = (q(&mut rng), q(&mut rng));
                    if a != b {
                        break (a, b);
                    }
                };
                let k = rng.below(8) as usize;
                // A parameter, sometimes scaled and shifted.
                let x = if rng.below(3) == 0 {
                    2.0 * p[k] - 0.3
                } else {
                    p[k]
                };
                match rng.below(13) {
                    0 => c.rx(a, x),
                    1 => c.ry(a, x),
                    2 => c.rz(a, x),
                    3 => c.p(a, x),
                    4 => c.rzz(a, b, x),
                    5 => c.rxx(a, b, x),
                    6 => c.ryy(a, b, x),
                    7 => c.crz(a, b, x),
                    8 => c.cp(a, b, x),
                    9 => c.crx(a, b, x),
                    10 => c.cry(a, b, x),
                    11 => c.cx(a, b).h(b),
                    _ => c.s(a).t(b),
                };
            }
            let theta: Vec<f64> = (0..8).map(|_| rng.uniform_range(-3.0, 3.0)).collect();
            let h: PauliSum = if round % 2 == 0 {
                "Z0 Z6 - 0.5 Z3 + 0.25 Z1 Z2 Z5 + 0.1".parse().unwrap()
            } else {
                "Z0 Z6 + 0.5 X2 Y6 - 0.3 Y3 Z4 + 0.2 X5".parse().unwrap()
            };
            let mut psi = StateVector::zero(n);
            psi.apply_circuit(&c, &theta).unwrap();
            let phi = psi.into_amplitudes();
            let apply = |x: &[C64], y: &mut [C64]| h.apply_amplitudes(x, y);
            let (v0, g0) = adjoint_with_block(&c, &theta, phi.clone(), apply, 20);
            let (v1, g1) = adjoint_with_block(&c, &theta, phi, apply, 3);
            assert_eq!(v0, v1);
            let (vs, gs) =
                expectation_and_gradient(&c, &h, &theta, None, GradientMethod::ParameterShift)
                    .unwrap();
            assert!((v0 - vs).abs() < 1e-12);
            for ((a, b), s) in g0.iter().zip(&g1).zip(&gs) {
                assert!((a - b).abs() < 1e-12, "plain {a} vs blocked {b}");
                assert!((a - s).abs() < 1e-10, "adjoint {a} vs parameter shift {s}");
            }
        }
    }

    /// Single precision against double precision on random circuits (the
    /// shift-rule gates, diagonal and general observables), on registers
    /// below and above the cache block: energies to 10⁻⁵ and gradients to
    /// 10⁻⁴ of their scale.
    #[test]
    fn single_precision_leaves_the_constant_out_of_the_co_state() {
        use super::*;
        use crate::algorithms::{ansatz, ansatz::Entanglement, models};
        use crate::quantum::PauliString;
        // A large identity term: before, its rounding in H·φ cost the
        // gradient 1.4·10⁻³ relative at an offset of 10⁴.
        let n = 10;
        let c = ansatz::efficient_su2(n, 2, Entanglement::Linear);
        let theta: Vec<f64> = (0..c.num_parameters())
            .map(|k| 0.1 + 0.37 * k as f64)
            .collect();
        let mut h = models::transverse_field_ising(n, 1.0, 0.8, false);
        h.add_term(1e4, PauliString::identity());
        let (e, g) =
            expectation_and_gradient(&c, &h, &theta, None, GradientMethod::Adjoint).unwrap();
        let (e32, g32) = expectation_and_gradient_single(&c, &h, &theta, None).unwrap();
        let scale = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err = g
            .iter()
            .zip(&g32)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-5 * scale, "{err} vs {scale}");
        assert!((e32 - e).abs() < 1e-6, "{e32} vs {e}");
    }

    #[test]
    fn single_precision_rejects_zero_nan_and_overflow() {
        use super::*;
        use crate::algorithms::{ansatz, ansatz::Entanglement};
        let c = ansatz::efficient_su2(3, 1, Entanglement::Linear);
        let theta = vec![0.3; c.num_parameters()];
        let h: PauliSum = "Z0 + X1".parse().unwrap();
        let mut zero = StateVector32::zero(3).unwrap();
        zero.amplitudes_mut().fill(Default::default());
        let mut nan = StateVector32::zero(3).unwrap();
        nan.amplitudes_mut()[1].re = f32::NAN;
        for bad in [&zero, &nan] {
            assert!(expectation_single(&c, &h, &theta, Some(bad)).is_err());
            assert!(expectation_and_gradient_single(&c, &h, &theta, Some(bad)).is_err());
        }
        let huge: PauliSum = "1e39 Z0 + X1".parse().unwrap();
        let err = expectation_and_gradient_single(&c, &huge, &theta, None).unwrap_err();
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    #[test]
    fn single_precision_matches_double_precision() {
        use super::*;
        let mut rng = Rng::seed_from_u64(1270);
        for (round, n) in [7usize, 9, 16].into_iter().enumerate() {
            let mut c = Circuit::new(n);
            let p = c.new_parameters(10);
            for _ in 0..60 {
                let (a, b) = loop {
                    let (a, b) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
                    if a != b {
                        break (a, b);
                    }
                };
                let x = p[rng.below(10) as usize];
                match rng.below(7) {
                    0 => c.rx(a, x),
                    1 => c.ry(a, x),
                    2 => c.rz(a, x),
                    3 => c.rzz(a, b, x),
                    4 => c.crx(a, b, x),
                    5 => c.cx(a, b),
                    _ => c.h(a),
                };
            }
            let theta: Vec<f64> = (0..10).map(|_| rng.uniform_range(-3.0, 3.0)).collect();
            let h: PauliSum = if round % 2 == 0 {
                "Z0 Z1 - 0.5 Z2 + 0.25 Z3 Z5 Z6 + 0.1".parse().unwrap()
            } else {
                "Z0 Z1 + 0.5 X2 Y6 - 0.3 Y3 Z4 + 0.2 X5".parse().unwrap()
            };
            let (e64, g64) =
                expectation_and_gradient(&c, &h, &theta, None, GradientMethod::Adjoint).unwrap();
            let (e32, g32) = expectation_and_gradient_single(&c, &h, &theta, None).unwrap();
            let scale = g64.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            assert!((e64 - e32).abs() < 1e-5, "{e64} vs {e32}");
            assert!((expectation_single(&c, &h, &theta, None).unwrap() - e64).abs() < 1e-5);
            for (a, b) in g64.iter().zip(&g32) {
                assert!((a - b).abs() < 1e-4 * scale, "{a} vs {b}");
            }
        }
        // Single precision has no shift rules or differences.
        let c = test_circuit();
        let h: PauliSum = "Z0".parse().unwrap();
        let theta = [0.1; 6];
        assert!(
            expectation_and_gradient_at(
                &c,
                &h,
                &theta,
                GradientMethod::ParameterShift,
                Precision::Single
            )
            .is_err()
        );
        let (a, _) =
            expectation_and_gradient_at(&c, &h, &theta, GradientMethod::Adjoint, Precision::Single)
                .unwrap();
        assert!((a - expectation_at(&c, &h, &theta, Precision::Double).unwrap()).abs() < 1e-5);
    }

    #[test]
    fn batches_match_single_evaluations() {
        use crate::algorithms::ansatz::{Entanglement, efficient_su2};
        let circuit = efficient_su2(5, 2, Entanglement::Linear);
        let h: PauliSum = "Z0 Z1 + 0.5 X2 - 0.3 Y3 Z4".parse().unwrap();
        let thetas: Vec<Vec<f64>> = (0..17)
            .map(|k| {
                (0..circuit.num_parameters())
                    .map(|j| 0.1 * (k * j) as f64 - 0.7)
                    .collect()
            })
            .collect();
        let values = expectation_batch(&circuit, &h, &thetas, None).unwrap();
        let both =
            expectation_and_gradient_batch(&circuit, &h, &thetas, None, GradientMethod::Adjoint)
                .unwrap();
        for (k, theta) in thetas.iter().enumerate() {
            let (v, g) =
                expectation_and_gradient(&circuit, &h, theta, None, GradientMethod::Adjoint)
                    .unwrap();
            assert_eq!(values[k], expectation(&circuit, &h, theta, None).unwrap());
            assert_eq!(both[k], (v, g));
        }
        // An invalid entry fails the whole batch.
        let mut bad = thetas.clone();
        bad[3].pop();
        assert!(expectation_batch(&circuit, &h, &bad, None).is_err());
    }

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

#[cfg(test)]
/// Independent references for the adjoint pass (from the review of 1.26): a
/// plain pass that copies the state for every derivative, finite
/// differences, and the derivative kernel, on random circuits over the
/// whole gate set with controls and every block size.
mod against_references {
    use super::*;
    use crate::quantum::{Parameter, PauliString};
    use crate::rng::Rng;

    fn rand_state(n: usize, rng: &mut Rng) -> Vec<C64> {
        (0..1usize << n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect()
    }

    /// Old-style reference: copy + derivative + inner, no blocking.
    fn reference(
        circuit: &Circuit,
        theta: &[f64],
        mut phi: Vec<C64>,
        h: &PauliSum,
    ) -> (f64, Vec<f64>) {
        let mut lambda = vec![C64::new(0.0, 0.0); phi.len()];
        h.apply_amplitudes(&phi, &mut lambda);
        let value = inner(&phi, &lambda).re;
        let mut grad = vec![0.0; theta.len()];
        for inst in circuit.instructions().iter().rev() {
            let angles = inst.angles(theta);
            let inverse = inst.inverse();
            let inv_angles = inverse.angles(theta);
            kernels::apply_instruction(&mut phi, &inverse, &inv_angles, false, 0);
            for (slot, param) in inst.params.iter().enumerate() {
                if let Param::Symbol(p) = param {
                    let mut mu = phi.clone();
                    kernels::apply_instruction_derivative(&mut mu, inst, &angles, slot);
                    grad[p.index] += 2.0 * p.scale * inner(&lambda, &mu).re;
                }
            }
            kernels::apply_instruction(&mut lambda, &inverse, &inv_angles, false, 0);
        }
        (value, grad)
    }

    fn distinct(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
        let mut v: Vec<usize> = Vec::new();
        while v.len() < k {
            let q = rng.below(n as u64) as usize;
            if !v.contains(&q) {
                v.push(q);
            }
        }
        v
    }

    fn random_circuit(
        n: usize,
        n_params: usize,
        n_gates: usize,
        rng: &mut Rng,
        diag_bias: bool,
    ) -> Circuit {
        use GateKind::*;
        let mut c = Circuit::new(n);
        let p = c.new_parameters(n_params);
        let kinds = [
            H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, RX, RY, RZ, P, U, Swap, RXX, RYY, RZZ,
        ];
        let diag = [Z, S, Sdg, T, Tdg, RZ, P, RZZ];
        for _ in 0..n_gates {
            let kind = if diag_bias && rng.below(2) == 0 {
                diag[rng.below(diag.len() as u64) as usize]
            } else {
                kinds[rng.below(kinds.len() as u64) as usize]
            };
            let nt = kind.num_targets().unwrap();
            let nc = match rng.below(4) {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            };
            let q = distinct(rng, n, nt + nc);
            let (targets, controls) = q.split_at(nt);
            let params: Vec<Param> = (0..kind.num_params())
                .map(|_| {
                    let k = rng.below(n_params as u64) as usize;
                    match rng.below(4) {
                        0 => Param::Value(rng.uniform_range(-2.0, 2.0)),
                        1 => Param::Symbol(Parameter {
                            index: k,
                            scale: -1.7,
                            offset: 0.4,
                        }),
                        _ => Param::Symbol(p[k]),
                    }
                })
                .collect();
            c.controlled(kind, targets, controls, &params).unwrap();
        }
        c
    }

    fn observables() -> Vec<PauliSum> {
        let mut v: Vec<PauliSum> = vec![
            "Z0 Z5 - 0.5 Z3 + 0.25 Z1 Z2 Z4 + 0.1".parse().unwrap(),
            "Z0 Z5 + 0.5 X2 Y5 - 0.3 Y3 Z4 + 0.2 X1 - 0.7 + 0.4 Z2"
                .parse()
                .unwrap(),
        ];
        // repeated strings, identity twice, zero coefficient
        let mut h = PauliSum::new();
        let z = |q| PauliString::single(q, crate::quantum::Pauli::Z);
        h.add_term(0.3, z(1))
            .add_term(-0.8, z(1))
            .add_term(0.5, PauliString::identity())
            .add_term(0.25, PauliString::identity())
            .add_term(0.0, PauliString::single(0, crate::quantum::Pauli::X))
            .add_term(1.1, z(4));
        v.push(h);
        v
    }

    #[test]
    fn blocked_pass_matches_a_plain_reference_on_random_circuits() {
        let mut rng = Rng::seed_from_u64(99);
        let mut worst = (0.0f64, 0.0f64);
        let mut count = 0;
        for n in 6..=10 {
            for round in 0..6 {
                let np = 6;
                let c = random_circuit(n, np, 60, &mut rng, round % 2 == 0);
                let theta: Vec<f64> = (0..np).map(|_| rng.uniform_range(-3.0, 3.0)).collect();
                for h in observables() {
                    let mut psi = StateVector::zero(n);
                    // random initial state to avoid symmetric zeros
                    let init = rand_state(n, &mut rng);
                    let nrm = init.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
                    psi = StateVector::from_amplitudes(init.iter().map(|a| a / nrm).collect())
                        .unwrap_or(psi);
                    psi.apply_circuit(&c, &theta).unwrap();
                    let phi = psi.into_amplitudes();
                    let (vr, gr) = reference(&c, &theta, phi.clone(), &h);
                    for cb in 0..=5 {
                        let apply = |x: &[C64], y: &mut [C64]| h.apply_amplitudes(x, y);
                        let (v, g) = adjoint_with_block(&c, &theta, phi.clone(), apply, cb);
                        assert!((v - vr).abs() < 1e-12);
                        for (k, (a, b)) in g.iter().zip(&gr).enumerate() {
                            let e = (a - b).abs();
                            if e > worst.0 {
                                worst = (e, *b);
                            }
                            assert!(
                                e < 1e-11,
                                "n {n} round {round} block {cb} param {k}: {a} vs {b}\n{:?}",
                                c.instructions()
                            );
                        }
                        count += 1;
                    }
                }
            }
        }
        eprintln!(
            "REVIEW blocked-vs-reference worst {:e} (|g| {}), cases {count}",
            worst.0, worst.1
        );
    }

    #[test]
    fn the_reference_matches_finite_differences() {
        let mut rng = Rng::seed_from_u64(7);
        let mut worst = 0.0f64;
        for n in [6, 8] {
            for _ in 0..4 {
                let np = 5;
                let c = random_circuit(n, np, 40, &mut rng, true);
                let theta: Vec<f64> = (0..np).map(|_| rng.uniform_range(-3.0, 3.0)).collect();
                for h in observables() {
                    let mut psi = StateVector::zero(n);
                    psi.apply_circuit(&c, &theta).unwrap();
                    let (_, g) = adjoint_with_block(
                        &c,
                        &theta,
                        psi.into_amplitudes(),
                        |x, y| h.apply_amplitudes(x, y),
                        2,
                    );
                    for k in 0..np {
                        let e = |s: f64| {
                            let mut t = theta.clone();
                            t[k] += s;
                            expectation(&c, &h, &t, None).unwrap()
                        };
                        let st = 1e-4;
                        let fd =
                            (8.0 * (e(st) - e(-st)) - (e(2.0 * st) - e(-2.0 * st))) / (12.0 * st);
                        worst = worst.max((fd - g[k]).abs());
                        assert!((fd - g[k]).abs() < 1e-8, "{fd} vs {}", g[k]);
                    }
                }
            }
        }
        eprintln!("REVIEW fd worst {worst:e}");
    }

    #[test]
    fn derivative_elements_of_every_gate_kind_match_copies() {
        use GateKind::*;
        let mut rng = Rng::seed_from_u64(3);
        let mut worst = 0.0f64;
        for n in [5usize, 17] {
            let phi = rand_state(n, &mut rng);
            let lam = rand_state(n, &mut rng);
            for kind in [RX, RY, RZ, P, U, RXX, RYY, RZZ] {
                for nc in 0..=2usize {
                    for _ in 0..(if n == 5 { 6 } else { 2 }) {
                        let nt = kind.num_targets().unwrap();
                        let q = distinct(&mut rng, n, nt + nc);
                        let (targets, controls) = q.split_at(nt);
                        let angles: Vec<f64> = (0..kind.num_params())
                            .map(|_| rng.uniform_range(-7.0, 7.0))
                            .collect();
                        let inst = Instruction {
                            kind,
                            targets: targets.to_vec(),
                            controls: controls.to_vec(),
                            params: angles.iter().map(|&a| Param::Value(a)).collect(),
                            matrix: None,
                        };
                        for which in 0..kind.num_params() {
                            let mut mu = phi.clone();
                            kernels::apply_instruction_derivative(&mut mu, &inst, &angles, which);
                            let r = inner(&lam, &mu);
                            let d = kernels::derivative_element(
                                &lam,
                                &phi,
                                &inst,
                                &angles,
                                which,
                                inst.control_mask(),
                            );
                            let scale = (phi.len() as f64).sqrt();
                            let e = (r - d).norm() / scale;
                            worst = worst.max(e);
                            assert!(
                                e < 1e-13,
                                "{kind:?} t {targets:?} c {controls:?} w {which}: {r} vs {d}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("REVIEW derivative_element worst rel {worst:e}");
    }
}
