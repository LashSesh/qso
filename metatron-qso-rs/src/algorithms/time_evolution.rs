//! Hamiltonian simulation: circuits for `exp(−iHt)` from product formulas
//! and from qDRIFT, with rigorous error bounds.
//!
//! For `H = Σ_γ h_γ P_γ` (Pauli strings `P_γ`, in the order of the
//! [`PauliSum`]), one step of length `τ = t/r` of
//!
//! * order 1 (Lie–Trotter) applies `exp(−iτ h_γ P_γ)` for γ = 1, …, Γ;
//! * order 2 (Strang) applies half steps for γ = 1, …, Γ and back;
//! * order 2k ≥ 4 is Suzuki's recursion
//!   `S₂ₖ(τ) = S₂ₖ₋₂(uτ)² S₂ₖ₋₂((1 − 4u)τ) S₂ₖ₋₂(uτ)²` with
//!   `u = 1/(4 − 4^{1/(2k−1)})`.
//!
//! [`trotter_error_bound`] bounds the spectral-norm distance between the
//! circuit's unitary and `exp(−iHt)`:
//!
//! * order 1: `τ²/2 · Σ_γ ‖[Σ_{γ'>γ} H_γ', H_γ]‖` (Childs, Su, Tran, Wiebe
//!   and Zhu, Phys. Rev. X 11, 011020 (2021), Proposition 15);
//! * order 2: `τ³/12 · Σ_γ ‖[B_γ, [B_γ, H_γ]]‖ + τ³/24 · Σ_γ ‖[H_γ, [H_γ, B_γ]]‖`
//!   with `B_γ = Σ_{γ'>γ} H_γ'` (Proposition 16 there);
//! * order 2k ≥ 4: the smaller of two bounds. The product of the 5^{k−1}
//!   second-order stages with times `s_i τ` (`Σ s_i = 1`) differs from
//!   `exp(−iHτ)` by at most `Σ_i` (order-2 bound at `|s_i| τ`), a
//!   commutator bound of third order in τ; and, since the formula is exact
//!   to order 2k, the Taylor remainder `2 Σ_{m>2k} (Mτ)^m/m!` with
//!   `M = Σ_i |s_i| · Σ_γ |h_γ|` bounds both series.
//!
//! The commutators are computed exactly in the Pauli algebra; their norms
//! are bounded by the sum of the absolute coefficients. Bounds per step
//! are multiplied by `r`, and no bound exceeds 2.
//!
//! In the tests (Heisenberg, Ising and random 4-qubit Hamiltonians) the
//! first- and second-order bounds exceed the actual error by factors of
//! 3.5–13. The bound for order 4 and higher is valid but loose: it falls
//! only as `r⁻²` until `Mτ` is small, while the error falls as `r⁻⁴`.
//!
//! [`qdrift_circuit`] samples `N` terms with probabilities `|h_γ|/λ`,
//! `λ = Σ|h_γ|`, each applied for time `λt/N` (Campbell, Phys. Rev. Lett.
//! 123, 070503 (2019)); the average channel is within diamond distance
//! [`qdrift_error_bound`] `= 2λ²t²/N · e^{2λt/N}` of the exact evolution.
//!
//! The identity term of `H` only contributes a global phase, which
//! circuits omit.
//!
//! ```
//! use metatron_qso::algorithms::time_evolution::{trotter_circuit, trotter_error_bound};
//! use metatron_qso::algorithms::models::heisenberg_xxz;
//!
//! let h = heisenberg_xxz(4, 1.0, 0.5, false);
//! let c = trotter_circuit(&h, 4, 1.0, 10, 2).unwrap();
//! let bound = trotter_error_bound(&h, 1.0, 10, 2).unwrap();
//! assert!(bound < 0.05 && c.len() > 0);
//! ```

use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, PauliString, PauliSum, QubitOperator};
use crate::rng::Rng;
use num_complex::Complex64 as C64;

/// Largest supported product-formula order.
pub const MAX_ORDER: usize = 10;

/// Non-identity terms of `h` in order.
fn nontrivial_terms(h: &PauliSum) -> Vec<(f64, PauliString)> {
    h.terms()
        .iter()
        .filter(|(c, s)| !s.is_identity() && *c != 0.0)
        .copied()
        .collect()
}

fn check(h: &PauliSum, t: f64, order: usize) -> Result<()> {
    if !t.is_finite() {
        return Err(QsoError::invalid("t", "time must be finite"));
    }
    if order == 0 || order > MAX_ORDER || (order > 1 && order % 2 == 1) {
        return Err(QsoError::invalid(
            "order",
            format!("order must be 1 or even and at most {MAX_ORDER}, got {order}"),
        ));
    }
    if h.terms().iter().any(|(c, _)| !c.is_finite()) {
        return Err(QsoError::invalid(
            "hamiltonian",
            "coefficients must be finite",
        ));
    }
    Ok(())
}

/// Signed times `s_i` of the second-order stages of the Suzuki formula of
/// even `order` (`Σ s_i = 1`).
fn suzuki_stages(order: usize) -> Vec<f64> {
    if order <= 2 {
        return vec![1.0];
    }
    let k = order / 2;
    let u = 1.0 / (4.0 - 4f64.powf(1.0 / (2 * k - 1) as f64));
    let inner = suzuki_stages(order - 2);
    let mut out = Vec::with_capacity(5 * inner.len());
    for s in [u, u, 1.0 - 4.0 * u, u, u] {
        out.extend(inner.iter().map(|x| s * x));
    }
    out
}

/// The rotations `(string, θ)` meaning `exp(−iθP/2)` of one product
/// formula step, adjacent equal strings merged.
fn step_rotations(terms: &[(f64, PauliString)], tau: f64, order: usize) -> Vec<(PauliString, f64)> {
    let mut out: Vec<(PauliString, f64)> = Vec::new();
    let mut push = |s: PauliString, theta: f64| match out.last_mut() {
        Some((last, angle)) if *last == s => *angle += theta,
        _ => out.push((s, theta)),
    };
    if order == 1 {
        for &(c, s) in terms {
            push(s, 2.0 * c * tau);
        }
        return out;
    }
    for stage in suzuki_stages(order) {
        let half = stage * tau;
        for &(c, s) in terms {
            push(s, c * half);
        }
        for &(c, s) in terms.iter().rev() {
            push(s, c * half);
        }
    }
    out
}

/// Circuit on `n_qubits` qubits for `exp(−iHt)` from `steps` steps of the
/// product formula of `order` (1, 2, 4, 6, …). See the module docs.
pub fn trotter_circuit(
    h: &PauliSum,
    n_qubits: usize,
    t: f64,
    steps: usize,
    order: usize,
) -> Result<Circuit> {
    check(h, t, order)?;
    if steps == 0 {
        return Err(QsoError::invalid("steps", "need at least one step"));
    }
    if h.min_qubits() > n_qubits {
        return Err(QsoError::DimensionMismatch {
            expected: n_qubits,
            actual: h.min_qubits(),
        });
    }
    let terms = nontrivial_terms(h);
    let tau = t / steps as f64;
    let one = step_rotations(&terms, tau, order);
    let mut rotations: Vec<(PauliString, f64)> = Vec::with_capacity(one.len() * steps);
    for _ in 0..steps {
        for &(s, theta) in &one {
            match rotations.last_mut() {
                Some((last, angle)) if *last == s => *angle += theta,
                _ => rotations.push((s, theta)),
            }
        }
    }
    let mut c = Circuit::new(n_qubits);
    for (s, theta) in rotations {
        if theta != 0.0 {
            c.pauli_rotation(&s, theta)?;
        }
    }
    Ok(c)
}

/// `‖[Σ_{γ'>γ} H_γ', H_γ]‖` summed over γ, and the two sums of the
/// second-order bound, for unit time.
fn commutator_sums(terms: &[(f64, PauliString)], second: bool) -> (f64, f64, f64) {
    let ops: Vec<QubitOperator> = terms
        .iter()
        .map(|&(c, s)| QubitOperator::from_terms([(C64::new(c, 0.0), s)]))
        .collect();
    let (mut first, mut bba, mut aab) = (0.0, 0.0, 0.0);
    for g in 0..terms.len() {
        // [B, A] with B the suffix sum, collected term by term.
        let mut ba = QubitOperator::new();
        for later in &ops[g + 1..] {
            for (c, s) in later.commutator(&ops[g]).terms() {
                ba.add_term(c, s);
            }
        }
        ba.simplify(0.0);
        first += ba.one_norm();
        if second && !ba.is_empty() {
            let mut b_ba = QubitOperator::new();
            for later in &ops[g + 1..] {
                for (c, s) in later.commutator(&ba).terms() {
                    b_ba.add_term(c, s);
                }
            }
            bba += b_ba.simplify(0.0).one_norm();
            // [A, [A, B]] = −[A, [B, A]].
            aab += ops[g].commutator(&ba).simplify(0.0).one_norm();
        }
    }
    (first, bba, aab)
}

/// `2 Σ_{m>p} x^m/m!`.
fn taylor_tail(x: f64, p: usize) -> f64 {
    let mut term = 1.0;
    for m in 1..=p + 1 {
        term *= x / m as f64;
    }
    let mut sum = 0.0;
    let mut m = p + 1;
    while term > 1e-18 * sum && m < p + 10_000 {
        sum += term;
        m += 1;
        term *= x / m as f64;
        if !sum.is_finite() {
            return f64::INFINITY;
        }
    }
    2.0 * sum
}

/// Upper bound on `‖U_circuit − exp(−iHt)‖` (spectral norm, ignoring the
/// global phase of the identity term) for [`trotter_circuit`] with the same
/// arguments. See the module docs; the cost grows as Γ² (order 1) or Γ³
/// (higher orders) in the number of terms Γ.
pub fn trotter_error_bound(h: &PauliSum, t: f64, steps: usize, order: usize) -> Result<f64> {
    check(h, t, order)?;
    if steps == 0 {
        return Err(QsoError::invalid("steps", "need at least one step"));
    }
    let terms = nontrivial_terms(h);
    let tau = t.abs() / steps as f64;
    let (first, bba, aab) = commutator_sums(&terms, order > 1);
    let lambda: f64 = terms.iter().map(|(c, _)| c.abs()).sum();
    let per_step = if order == 1 {
        (tau * tau / 2.0 * first).min(taylor_tail(lambda * tau, 1))
    } else {
        let c2 = bba / 12.0 + aab / 24.0;
        let stages = suzuki_stages(order);
        let cubic: f64 = stages.iter().map(|s| s.abs().powi(3)).sum::<f64>() * c2 * tau.powi(3);
        let mass: f64 = stages.iter().map(|s| s.abs()).sum::<f64>() * lambda;
        cubic.min(taylor_tail(mass * tau, order))
    };
    Ok((steps as f64 * per_step).min(2.0))
}

/// One random qDRIFT circuit for `exp(−iHt)` with `samples` rotations,
/// each `exp(−i (λt/N) sign(h_γ) P_γ)` with γ drawn with probability
/// `|h_γ|/λ`. Average over seeds for the qDRIFT channel.
pub fn qdrift_circuit(
    h: &PauliSum,
    n_qubits: usize,
    t: f64,
    samples: usize,
    seed: u64,
) -> Result<Circuit> {
    check(h, t, 1)?;
    if samples == 0 {
        return Err(QsoError::invalid("samples", "need at least one sample"));
    }
    if h.min_qubits() > n_qubits {
        return Err(QsoError::DimensionMismatch {
            expected: n_qubits,
            actual: h.min_qubits(),
        });
    }
    let terms = nontrivial_terms(h);
    let mut c = Circuit::new(n_qubits);
    let lambda: f64 = terms.iter().map(|(c, _)| c.abs()).sum();
    if terms.is_empty() || t == 0.0 {
        return Ok(c);
    }
    let mut cumulative = Vec::with_capacity(terms.len());
    let mut acc = 0.0;
    for (h, _) in &terms {
        acc += h.abs() / lambda;
        cumulative.push(acc);
    }
    let mut rng = Rng::seed_from_u64(seed);
    let angle = 2.0 * lambda * t / samples as f64;
    for _ in 0..samples {
        let u = rng.uniform();
        let k = cumulative.partition_point(|&x| x <= u).min(terms.len() - 1);
        let (h, s) = terms[k];
        c.pauli_rotation(&s, angle * h.signum())?;
    }
    Ok(c)
}

/// Campbell's bound `2λ²t²/N · e^{2λt/N}` (at most 1) on the diamond
/// distance (with the factor ½) between the qDRIFT channel with `samples`
/// rotations and `exp(−iHt)`.
pub fn qdrift_error_bound(h: &PauliSum, t: f64, samples: usize) -> Result<f64> {
    check(h, t, 1)?;
    if samples == 0 {
        return Err(QsoError::invalid("samples", "need at least one sample"));
    }
    let lambda: f64 = nontrivial_terms(h).iter().map(|(c, _)| c.abs()).sum();
    let x = lambda * t.abs();
    let n = samples as f64;
    Ok((2.0 * x * x / n * (2.0 * x / n).exp()).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::models::{heisenberg_xxz, transverse_field_ising};
    use crate::linalg::{CMatrix, expm_hermitian};
    use crate::quantum::StateVector;

    /// Unitary of a circuit, column by column.
    fn unitary(c: &Circuit) -> CMatrix {
        let n = c.n_qubits();
        let dim = 1 << n;
        let mut u = CMatrix::zeros(dim, dim);
        for col in 0..dim {
            let mut psi = StateVector::basis(n, col).unwrap();
            psi.apply_circuit(c, &[]).unwrap();
            for (row, a) in psi.amplitudes().iter().enumerate() {
                u[(row, col)] = *a;
            }
        }
        u
    }

    fn spectral_norm(m: &CMatrix) -> f64 {
        m.clone().singular_values().max()
    }

    fn exact(h: &PauliSum, n: usize, t: f64) -> CMatrix {
        // Drop the identity term, as circuits do.
        let traceless = PauliSum::from_terms(nontrivial_terms(h));
        expm_hermitian(&traceless.to_dense(n).unwrap(), t)
    }

    fn random_hamiltonian(n: usize, terms: usize, seed: u64) -> PauliSum {
        let mut rng = Rng::seed_from_u64(seed);
        let mut h = PauliSum::new();
        for _ in 0..terms {
            let factors: Vec<(usize, crate::quantum::Pauli)> = (0..n)
                .filter_map(|q| {
                    let p = [
                        crate::quantum::Pauli::I,
                        crate::quantum::Pauli::X,
                        crate::quantum::Pauli::Y,
                        crate::quantum::Pauli::Z,
                    ][rng.below(4) as usize];
                    (p != crate::quantum::Pauli::I).then_some((q, p))
                })
                .collect();
            h.add_term(
                rng.uniform_range(-1.0, 1.0),
                PauliString::new(&factors).unwrap(),
            );
        }
        h
    }

    #[test]
    fn pauli_rotations_are_exact() {
        for (k, label) in [
            "X0", "Y1", "Z2", "X0 X2", "Y0 Y1", "Z1 Z2", "X0 Y1 Z2", "Y0 Z2", "X1 Y2",
        ]
        .iter()
        .enumerate()
        {
            let h: PauliSum = label.parse().unwrap();
            let theta = 0.3 + 0.2 * k as f64;
            let mut c = Circuit::new(3);
            c.pauli_rotation(&h.terms()[0].1, theta).unwrap();
            let err = spectral_norm(&(unitary(&c) - exact(&h, 3, theta / 2.0)));
            assert!(err < 1e-12, "{label}: {err}");
        }
        let mut c = Circuit::new(2);
        let s: PauliSum = "X2".parse().unwrap();
        assert!(c.pauli_rotation(&s.terms()[0].1, 1.0).is_err());
    }

    #[test]
    fn errors_stay_below_their_bounds_and_converge_with_the_order() {
        let n = 4;
        let hamiltonians = [
            heisenberg_xxz(n, 1.0, 0.7, true),
            transverse_field_ising(n, 1.0, 0.8, false),
            random_hamiltonian(n, 12, 3),
        ];
        for h in &hamiltonians {
            let t = 1.0;
            for order in [1usize, 2, 4] {
                let mut errors = Vec::new();
                for steps in [4usize, 8, 16, 32] {
                    let c = trotter_circuit(h, n, t, steps, order).unwrap();
                    let err = spectral_norm(&(unitary(&c) - exact(h, n, t)));
                    let bound = trotter_error_bound(h, t, steps, order).unwrap();
                    assert!(
                        err <= bound * (1.0 + 1e-9) + 1e-12,
                        "order {order}, r {steps}: {err} > {bound}"
                    );
                    errors.push(err);
                }
                // Doubling the steps divides the error by 2^order.
                let ratio = errors[2] / errors[3];
                let expected = (1u32 << order) as f64;
                assert!(
                    errors[3] < 1e-11 || (ratio / expected - 1.0).abs() < 0.2,
                    "order {order}: ratio {ratio}, expected {expected}, {errors:?}"
                );
            }
        }
    }

    #[test]
    fn first_and_second_order_bounds_are_tight_for_two_terms() {
        // H = X + Z: ‖[Z, X]‖ = 2, the first-order bound is τ² and the
        // actual error approaches it for small τ.
        let h: PauliSum = "X0 + Z0".parse().unwrap();
        let t = 0.01;
        let c = trotter_circuit(&h, 1, t, 1, 1).unwrap();
        let err = spectral_norm(&(unitary(&c) - exact(&h, 1, t)));
        let bound = trotter_error_bound(&h, t, 1, 1).unwrap();
        assert!((bound - t * t).abs() < 1e-15);
        assert!(err / bound > 0.99 && err <= bound);
        let c = trotter_circuit(&h, 1, t, 1, 2).unwrap();
        let err = spectral_norm(&(unitary(&c) - exact(&h, 1, t)));
        let bound = trotter_error_bound(&h, t, 1, 2).unwrap();
        assert!(err <= bound && err / bound > 0.3, "{err} {bound}");
    }

    #[test]
    fn commuting_terms_are_exact() {
        let h: PauliSum = "Z0 Z1 + 0.5 Z1 Z2 - Z0 + 2".parse().unwrap();
        for order in [1, 2, 4] {
            assert_eq!(trotter_error_bound(&h, 3.0, 1, order).unwrap(), 0.0);
            let c = trotter_circuit(&h, 3, 3.0, 1, order).unwrap();
            assert!(spectral_norm(&(unitary(&c) - exact(&h, 3, 3.0))) < 1e-12);
        }
    }

    #[test]
    fn qdrift_channel_converges_as_one_over_n() {
        // Exact average channel: ρ ↦ Σ_γ p_γ e^{−iτ sign P} ρ e^{iτ sign P},
        // applied N times, against exp(−iHt) on |0…0⟩.
        let n = 3;
        let h: PauliSum = "0.8 X0 X1 + 0.5 Z1 + 0.3 Y1 Y2 - 0.6 Z0 Z2"
            .parse()
            .unwrap();
        let t = 0.8;
        let lambda: f64 = h.terms().iter().map(|(c, _)| c.abs()).sum();
        let mut rho0 = CMatrix::zeros(1 << n, 1 << n);
        rho0[(0, 0)] = C64::new(1.0, 0.0);
        let u = exact(&h, n, t);
        let target = &u * &rho0 * u.adjoint();
        let mut distances = Vec::new();
        for samples in [10usize, 40, 160] {
            let tau = lambda * t / samples as f64;
            let gates: Vec<(f64, CMatrix)> = h
                .terms()
                .iter()
                .map(|&(c, s)| {
                    let g = PauliSum::from_terms([(c.signum(), s)]);
                    (
                        c.abs() / lambda,
                        expm_hermitian(&g.to_dense(n).unwrap(), tau),
                    )
                })
                .collect();
            let mut rho = rho0.clone();
            for _ in 0..samples {
                let mut next = CMatrix::zeros(1 << n, 1 << n);
                for (p, g) in &gates {
                    next += (g * &rho * g.adjoint()) * C64::new(*p, 0.0);
                }
                rho = next;
            }
            let diff = &rho - &target;
            let trace_distance = 0.5
                * diff
                    .symmetric_eigenvalues()
                    .iter()
                    .map(|x| x.abs())
                    .sum::<f64>();
            let bound = qdrift_error_bound(&h, t, samples).unwrap();
            assert!(trace_distance <= bound, "{trace_distance} > {bound}");
            distances.push(trace_distance);
        }
        // Four times the samples, a quarter of the error.
        for w in distances.windows(2) {
            assert!((w[0] / w[1] - 4.0).abs() < 0.6, "{distances:?}");
        }
        // Sampled circuits: the average over seeds approaches the channel.
        let samples = 40;
        let mut avg = CMatrix::zeros(1 << n, 1 << n);
        let runs = 400;
        for seed in 0..runs {
            let c = qdrift_circuit(&h, n, t, samples, seed).unwrap();
            let v = unitary(&c);
            avg += &v * &rho0 * v.adjoint();
        }
        avg /= C64::new(runs as f64, 0.0);
        let diff = &avg - &target;
        let td = 0.5
            * diff
                .symmetric_eigenvalues()
                .iter()
                .map(|x| x.abs())
                .sum::<f64>();
        assert!(
            td < qdrift_error_bound(&h, t, samples).unwrap() + 0.1,
            "{td}"
        );
    }

    #[test]
    fn invalid_arguments() {
        let h: PauliSum = "X0 + Z0".parse().unwrap();
        assert!(trotter_circuit(&h, 1, 1.0, 0, 1).is_err());
        assert!(trotter_circuit(&h, 1, 1.0, 1, 3).is_err());
        assert!(trotter_circuit(&h, 1, f64::NAN, 1, 2).is_err());
        assert!(trotter_circuit(&h, 1, 1.0, 1, 12).is_err());
        assert!(trotter_circuit(&"X3".parse().unwrap(), 2, 1.0, 1, 1).is_err());
        assert!(qdrift_circuit(&h, 1, 1.0, 0, 1).is_err());
        assert!(qdrift_error_bound(&h, 1.0, 0).is_err());
        // Suzuki stage times sum to one.
        for order in [2, 4, 6] {
            assert!((suzuki_stages(order).iter().sum::<f64>() - 1.0).abs() < 1e-14);
        }
        assert!(trotter_error_bound(&h, 1e6, 1, 4).unwrap() <= 2.0);
    }
}
