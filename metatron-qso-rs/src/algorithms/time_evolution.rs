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
//! [`trotter_error_bounds`] bounds the spectral-norm distance between the
//! circuit's unitary and `exp(−iHt)` in four ways, and
//! [`trotter_error_bound`] returns the smallest (at most 2):
//!
//! * **Nested commutators** ([`TrotterBounds::nested`]), for every order p.
//!   One step is a product of exponentials `E_j = exp(τB_j)`, and
//!   `S'(τ) = (A + T(τ)) S(τ)` with `A = −iH` and
//!   `T(τ) = Σ_j Ad_{E_J ⋯ E_{j+1}}(B_j) − A`, so that
//!   `‖S(τ) − e^{τA}‖ ≤ ∫₀^τ ‖T(σ)‖ dσ` (the error representation of
//!   Childs, Su, Tran, Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021)).
//!   The Taylor coefficients `C_m` of `T` vanish below `m = p`; those from
//!   p up to a degree D are nested commutators of order m + 1, computed as
//!   Pauli operators (their norms bounded by the sum of absolute
//!   coefficients). Beyond D, `‖C_m‖ ≤ Σ_j ‖B_j‖ (2β_j)^m/m!` with `β_j`
//!   the norm of the exponentials after `E_j`. D grows from p + 1 until
//!   that tail is below 1 % of the rest, within a fixed budget of work;
//!   a bound out of reach is infinite. The series is built for `H/λ`
//!   (`λ = Σ|h_γ|`) and evaluated at `λτ`, so the units of H do not
//!   matter, and it is cached for repeated calls on one Hamiltonian.
//! * **Commutators** ([`TrotterBounds::commutators`]), for every order.
//!   The same coefficients `C_m` below a degree D, and beyond them the
//!   Taylor remainder of every conjugation in integral form,
//!   `‖R_j(σ)‖ ≤ σ^D Σ_{|q|=D} ‖ad_{B_J}^{q_J} ⋯ ad_{B_{j+1}}^{q_{j+1}} B_j‖/q!`
//!   (the commutator expansion of a conjugation of exponentials of
//!   Childs et al.; the conjugations are unitary). Each nested commutator
//!   of Pauli rotations is a single Pauli string or zero, so the sum runs
//!   over the strings that do not commute, computed in place: an
//!   exponential leaves the strings that commute with it alone. The error
//!   generator is also factored in the middle of the formula,
//!   `S' − AS = L T R` with L and R the halves (as Childs et al. do at
//!   fourth order), which halves the depth of the conjugations; the
//!   smaller of the two is taken. D starts at the degree of the nested
//!   bound and rises while that helps, within a work budget. At most the
//!   nested bound, and far below it for larger Hamiltonians.
//! * **Stage by stage** ([`TrotterBounds::stages`]). Order 1:
//!   `τ²/2 · Σ_γ ‖[Σ_{γ'>γ} H_γ', H_γ]‖` (Proposition 15 there); order 2:
//!   `τ³/12 · Σ_γ ‖[B_γ, [B_γ, H_γ]]‖ + τ³/24 · Σ_γ ‖[H_γ, [H_γ, B_γ]]‖`
//!   with `B_γ = Σ_{γ'>γ} H_γ'` (Proposition 16); order 2k ≥ 4: the sum
//!   of the order-2 bound over the 5^{k−1} second-order stages at their
//!   times `|s_i| τ`, of third order only.
//! * **Taylor** ([`TrotterBounds::taylor`]): a formula exact to order p
//!   differs from `exp(−iHτ)` by at most `2 Σ_{m>p} (Mτ)^m/m!` with
//!   `M = Σ_i |s_i| · Σ_γ |h_γ|`.
//!
//! Bounds per step are multiplied by `r`. In the validation suite
//! (Heisenberg, Ising and random 4-qubit Hamiltonians, 8 to 32 steps) the
//! nested bound exceeds the actual error by factors of 7–11 at order 4 and
//! 8–24 at order 6, and falls as `r^{−p}` like the error. The
//! stage-by-stage bound of these orders falls only as `r⁻²` and exceeds
//! the error 780 to 10⁹ times. At orders 1 and 2 the nested bound equals
//! the stage-by-stage one to a few percent from about 16 steps and is up
//! to eight times larger at fewer. At few steps, where `2βτ` is large, it
//! is out of reach and the others apply. Its tail ignores which terms
//! commute: for larger Hamiltonians (8 to 16 Heisenberg spins, say) the
//! budget stops the degree early, the tail dominates and the bound, still
//! valid, lies thousands of times above the error. Rounding in the
//! vanishing coefficients sets a floor of about 10⁻¹⁵ per step.
//!
//! [`TrotterBounds::average`] bounds the error averaged over states, the
//! normalised Frobenius norm `‖U − exp(−iHt)‖_F/√d` (the root mean square
//! of `‖(U − exp(−iHt))ψ‖` over Haar-random ψ): the commutator bound
//! with the root sum of squares of the Pauli coefficients of `C_m` for
//! their sum (the Frobenius norm is unitarily invariant too). For a chain
//! of n spins the coefficients' sum grows as n and their root sum of
//! squares as √n. [`trotter_steps`] finds the steps for a target error
//! from both.
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
//! use metatron_qso::algorithms::time_evolution::{
//!     trotter_circuit, trotter_error_bound, trotter_error_bounds, trotter_steps,
//! };
//! use metatron_qso::algorithms::models::heisenberg_xxz;
//!
//! let h = heisenberg_xxz(4, 1.0, 0.5, false);
//! // Second order, 20 steps: ‖U − exp(−iHt)‖ ≤ 0.0214.
//! let c = trotter_circuit(&h, 4, 1.0, 20, 2).unwrap();
//! let bound = trotter_error_bound(&h, 1.0, 20, 2).unwrap();
//! assert!((bound - 0.0214).abs() < 1e-4);
//! assert_eq!(c.n_qubits(), 4);
//! // Fourth order: 1.8·10⁻⁵ from the commutator sums, 2.0·10⁻⁵ from the
//! // nested commutators, 0.013 stage by stage.
//! let b = trotter_error_bounds(&h, 1.0, 20, 4).unwrap();
//! assert!((b.commutators - 1.77e-5).abs() < 1e-7 && (b.nested - 2.02e-5).abs() < 1e-7);
//! assert!((b.stages - 0.0131).abs() < 1e-4);
//! // Eight spins, t = 2, 32 steps: 6.5·10⁻⁴ from the commutator sums
//! // (72 from the nested commutators), 5.3·10⁻⁵ averaged over states.
//! let h8 = heisenberg_xxz(8, 1.0, 0.5, false);
//! let b = trotter_error_bounds(&h8, 2.0, 32, 4).unwrap();
//! assert!((b.commutators - 6.50e-4).abs() < 1e-5 && b.nested > 70.0);
//! assert!((b.average - 5.34e-5).abs() < 1e-6);
//! // An error of 10⁻³ takes 29 steps, or 16 on average over states.
//! let r = trotter_steps(&h8, 2.0, 1e-3, 4).unwrap();
//! assert_eq!((r.worst, r.average), (29, 16));
//! ```

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, PauliString, PauliSum, QubitOperator};
use crate::rng::Rng;
use num_complex::Complex64 as C64;

/// Largest supported product-formula order.
pub const MAX_ORDER: usize = 10;
/// Largest number of gates in a generated circuit (a rotation of a
/// weight-w string takes at most 6w gates).
pub const MAX_GATES: usize = 1 << 22;

/// Upper bound on the gates of `Circuit::pauli_rotation` for `s`.
fn rotation_cost(s: &PauliString) -> usize {
    6 * s.weight() as usize
}

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
    let stages = if order == 1 {
        1
    } else {
        2 * 5usize.pow((order / 2 - 1) as u32)
    };
    let per_pass: usize = terms.iter().map(|(_, s)| rotation_cost(s)).sum();
    if per_pass.saturating_mul(stages).saturating_mul(steps) > MAX_GATES {
        return Err(QsoError::invalid(
            "steps",
            format!("the circuit would exceed {MAX_GATES} gates"),
        ));
    }
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

/// Most Pauli strings in one coefficient of the error series of
/// [`TrotterBounds::nested`]; larger series are not computed.
pub const MAX_SERIES_TERMS: usize = 1 << 16;

/// Hasher for the two masks of a [`PauliString`] (multiply and rotate; the
/// default SipHash is several times slower here).
#[derive(Clone, Copy, Default)]
struct MaskHasher(u64);

impl Hasher for MaskHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(29) ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type StringMap = HashMap<PauliString, C64, BuildHasherDefault<MaskHasher>>;

/// The norms `‖C_m‖₁` (sum of absolute Pauli coefficients), m = 0..=degree
/// (and the operations spent),
/// of the error generator `T(τ) = Σ_m C_m τ^m` of one step with
/// exponentials `exp(τ B_j)`, `B_j = −i b_j P_j` (`rotations`: `(P_j, b_j)`
/// in the order they act), for `H = Σ terms`: `S'(τ) = (A + T(τ)) S(τ)`
/// with `A = −iH`, and
/// `T(τ) = Σ_j Ad_{E_J ⋯ E_{j+1}}(B_j) − A`, built by Horner's scheme
/// `Y_j = B_j + exp(τ ad_{B_j}) Y_{j−1}` on power series in τ. `None` when
/// a coefficient would hold more than [`MAX_SERIES_TERMS`] strings or the
/// pass would take more than `budget` operations on strings.
fn error_series(
    rotations: &[(PauliString, f64)],
    terms: &[(f64, PauliString)],
    degree: usize,
    budget: f64,
) -> Option<(Vec<f64>, f64)> {
    type Series = Vec<StringMap>;
    let mut y: Series = vec![StringMap::default(); degree + 1];
    let mut work = 0.0;
    for &(p, b) in rotations {
        // ad_B(c Q) = −i b c [P, Q] = −2i b c P Q when P and Q anticommute.
        let ad = |x: &StringMap| -> StringMap {
            let mut out = StringMap::with_capacity_and_hasher(x.len(), Default::default());
            for (q, &c) in x {
                if p.commutes_with(q) {
                    continue;
                }
                let (phase, r) = p.multiply(q);
                *out.entry(r).or_insert(C64::new(0.0, 0.0)) += C64::new(0.0, -2.0 * b) * c * phase;
            }
            out.retain(|_, c| *c != C64::new(0.0, 0.0));
            out
        };
        // exp(τ ad_B) on the series: new[d + k] += ad_B^k(old[d]) / k!.
        let mut next: Series = vec![StringMap::default(); degree + 1];
        for d in 0..=degree {
            let mut z = std::mem::take(&mut y[d]);
            let mut factorial = 1.0;
            for k in 0..=degree - d {
                if z.is_empty() {
                    break;
                }
                if k > 0 {
                    factorial *= k as f64;
                }
                let target = &mut next[d + k];
                for (q, &c) in &z {
                    *target.entry(*q).or_insert(C64::new(0.0, 0.0)) += c / factorial;
                }
                work += z.len() as f64;
                if target.len() > MAX_SERIES_TERMS || work > budget {
                    return None;
                }
                if k < degree - d {
                    z = ad(&z);
                }
            }
        }
        *next[0].entry(p).or_insert(C64::new(0.0, 0.0)) += C64::new(0.0, -b);
        y = next;
    }
    // Subtract A from the constant term.
    for &(h, p) in terms {
        *y[0].entry(p).or_insert(C64::new(0.0, 0.0)) -= C64::new(0.0, -h);
    }
    Some((
        y.iter()
            .map(|m| m.values().map(|c| c.norm()).sum())
            .collect(),
        work,
    ))
}

/// The bounds of [`trotter_error_bounds`] on `‖U_circuit − exp(−iHt)‖`;
/// `bound` is the smallest. A bound that is not available is infinite.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrotterBounds {
    /// The smallest of the others, at most 2.
    pub bound: f64,
    /// From the Taylor coefficients of the step's error generator, nested
    /// commutators of order p + 1 and higher computed as Pauli operators
    /// up to an adaptive degree, and a tail (see the module docs). Infinite
    /// when it cannot beat the other bounds even with the largest degree,
    /// or when its first series has more than [`MAX_SERIES_TERMS`] strings
    /// per coefficient or exceeds the work budget. Not capped at 2.
    pub nested: f64,
    /// The commutator bound of order 1 or 2 (Childs et al.,
    /// Propositions 15 and 16), applied stage by stage for higher orders.
    pub stages: f64,
    /// The Taylor remainder of a formula exact to order p.
    pub taylor: f64,
    /// From the same error series as `nested`, its coefficients below a
    /// degree D computed exactly and the rest bounded by the nested
    /// commutators of degree D of the conjugations, each computed from the
    /// Pauli algebra (those of commuting strings vanish): the
    /// commutator-scaling bound of Childs et al. (the remainder of their
    /// commutator expansion of a conjugation of exponentials, Sec. III)
    /// with exact leading terms. At most `nested`. Infinite when the series exceeds
    /// [`MAX_COMMUTATOR_TERMS`] strings per coefficient or the work
    /// budget. Not capped at 2.
    #[serde(default = "infinite")]
    pub commutators: f64,
    /// Bound on the error averaged over states, `‖U − exp(−iHt)‖_F/√d`
    /// (`d = 2ⁿ`; the root mean square of `‖(U − exp(−iHt))ψ‖` over
    /// Haar-random ψ): the bound of `commutators` with the root sum of
    /// squares of the Pauli coefficients for their sum, or `bound` if
    /// smaller. Typically smaller than `bound` by about the square root
    /// of the number of terms.
    #[serde(default = "infinite")]
    pub average: f64,
}

fn infinite() -> f64 {
    f64::INFINITY
}

/// Upper bound on `‖U_circuit − exp(−iHt)‖` (spectral norm, ignoring the
/// global phase of the identity term) for [`trotter_circuit`] with the same
/// arguments: the smallest of [`trotter_error_bounds`].
pub fn trotter_error_bound(h: &PauliSum, t: f64, steps: usize, order: usize) -> Result<f64> {
    Ok(trotter_error_bounds(h, t, steps, order)?.bound)
}

/// The bounds on `‖U_circuit − exp(−iHt)‖` for [`trotter_circuit`] with
/// the same arguments; see the module docs. The commutator sums cost Γ²
/// (order 1) or Γ³ (higher orders) in the number of terms Γ; the nested
/// bound costs one pass over the step's exponentials per power of τ up to
/// p + 1, on operators of up to [`MAX_SERIES_TERMS`] strings.
pub fn trotter_error_bounds(
    h: &PauliSum,
    t: f64,
    steps: usize,
    order: usize,
) -> Result<TrotterBounds> {
    check(h, t, order)?;
    if steps == 0 {
        return Err(QsoError::invalid("steps", "need at least one step"));
    }
    let terms = nontrivial_terms(h);
    // Nested commutators multiply several coefficients; beyond 1e100 they
    // could overflow, and the trivial bound 2 is all that is left anyway.
    if terms.iter().any(|(c, _)| c.abs() > 1e100) {
        return Ok(TrotterBounds {
            bound: 2.0,
            nested: f64::INFINITY,
            stages: f64::INFINITY,
            taylor: f64::INFINITY,
            commutators: f64::INFINITY,
            average: 2.0,
        });
    }
    let tau = t.abs() / steps as f64;
    let (first, bba, aab) = commutator_sums(&terms, order > 1);
    let lambda: f64 = terms.iter().map(|(c, _)| c.abs()).sum();
    let (stage_step, taylor_step) = if order == 1 {
        (tau * tau / 2.0 * first, taylor_tail(lambda * tau, 1))
    } else {
        let c2 = bba / 12.0 + aab / 24.0;
        let stages = suzuki_stages(order);
        let cubic: f64 = stages.iter().map(|s| s.abs().powi(3)).sum::<f64>() * c2 * tau.powi(3);
        let mass: f64 = stages.iter().map(|s| s.abs()).sum::<f64>() * lambda;
        (cubic, taylor_tail(mass * tau, order))
    };
    let other = stage_step.min(taylor_step).min(2.0 / steps as f64);
    let (nested_step, degree) = nested_step_bound(&terms, tau, order, other);
    let (commutator_step, average_step) = commutator_step_bounds(&terms, tau, order, degree);
    let total = |per_step: f64| {
        let b = steps as f64 * per_step;
        if b.is_nan() { f64::INFINITY } else { b }
    };
    let (nested, stages, taylor) = (total(nested_step), total(stage_step), total(taylor_step));
    // Rounding aside, the commutator bound at the nested bound's degree is
    // the smaller; the minimum makes it so exactly.
    let commutators = total(commutator_step).min(nested);
    let bound = commutators.min(stages).min(taylor).min(2.0);
    Ok(TrotterBounds {
        bound,
        nested,
        stages,
        taylor,
        commutators,
        average: total(average_step).min(bound),
    })
}

/// Most steps [`trotter_steps`] considers.
pub const MAX_STEPS: usize = 1 << 40;

/// Step counts of [`trotter_steps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrotterSteps {
    /// Steps for which [`TrotterBounds::bound`] (the worst case over
    /// states) is at most ε.
    pub worst: usize,
    /// Steps for which [`TrotterBounds::average`] (the root mean square
    /// over random states) is at most ε.
    pub average: usize,
}

/// The numbers of steps r of [`trotter_circuit`] with `order` for which
/// the bounds of [`trotter_error_bounds`] on the error of `exp(−iHt)` are
/// at most `epsilon`, worst case and averaged over states. Each is found
/// by doubling r and bisecting: the bound is at most ε at r and above ε
/// at r − 1 (the bounds fall with r, so r is the smallest). A target
/// below the rounding floor of the bounds (about 10⁻¹⁵ λt, λ = Σ|h_γ|)
/// or needing more than [`MAX_STEPS`] steps is an error.
pub fn trotter_steps(h: &PauliSum, t: f64, epsilon: f64, order: usize) -> Result<TrotterSteps> {
    check(h, t, order)?;
    if !(epsilon > 0.0 && epsilon.is_finite()) {
        return Err(QsoError::invalid(
            "epsilon",
            format!("target error must be positive and finite, got {epsilon}"),
        ));
    }
    let mut seen: HashMap<usize, TrotterBounds> = HashMap::new();
    let mut bounds = |r: usize| -> Result<TrotterBounds> {
        if let Some(b) = seen.get(&r) {
            return Ok(*b);
        }
        let b = trotter_error_bounds(h, t, r, order)?;
        seen.insert(r, b);
        Ok(b)
    };
    let mut search = |pick: fn(&TrotterBounds) -> f64, name: &str| -> Result<usize> {
        let mut hi = 1usize;
        while pick(&bounds(hi)?) > epsilon {
            if hi >= MAX_STEPS {
                return Err(QsoError::invalid(
                    "epsilon",
                    format!(
                        "no number of steps up to {MAX_STEPS} brings the {name} bound to {epsilon}"
                    ),
                ));
            }
            hi *= 2;
        }
        let mut lo = hi / 2; // above ε (or 0)
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if pick(&bounds(mid)?) <= epsilon {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(hi)
    };
    let worst = search(|b| b.bound, "worst-case")?;
    let average = search(|b| b.average, "average")?;
    Ok(TrotterSteps { worst, average })
}

/// Highest power of τ the nested bound computes exactly.
const MAX_SERIES_DEGREE: usize = 40;
/// Most operations on strings one nested bound spends on its series
/// (about a quarter of a second; series are cached across calls).
const MAX_SERIES_WORK: f64 = 4e6;

/// `‖S(τ) − exp(−iHτ)‖` for one step of the formula of `order` p, from
/// `S(τ) − e^{τA} = ∫₀^τ e^{(τ−σ)A} T(σ) S(σ) dσ`: at most
/// `∫₀^τ ‖T(σ)‖ dσ`. The coefficients `C_m` of `T` vanish below `m = p`;
/// those from p to a degree D are computed exactly, the rest bounded by
/// `‖C_m‖ ≤ Σ_j ‖B_j‖ (2β_j)^m / m!` with `β_j = Σ_{i>j} ‖B_i‖` (the
/// multinomial theorem on the nested commutators of the conjugations).
/// D starts at p + 1 and grows until that tail is below 1 % of the rest,
/// within [`MAX_SERIES_DEGREE`], [`MAX_SERIES_TERMS`] and
/// [`MAX_SERIES_WORK`]. Nothing is computed, or the degree no longer
/// raised, once even the tail beyond [`MAX_SERIES_DEGREE`] is at least
/// `other`, the best other bound of the step. Returns the bound and the
/// degree D it was reached with (0 when none was).
fn nested_step_bound(
    terms: &[(f64, PauliString)],
    tau: f64,
    order: usize,
    other: f64,
) -> (f64, usize) {
    // The series is computed for H/λ and evaluated at λτ (C_m(H) =
    // λ^{m+1} C_m(H/λ)), so that the units of H do not overflow it.
    if terms.is_empty() {
        return (0.0, 0);
    }
    let (lambda, scaled, rotations) = unit_step(terms, order);
    let tau = lambda * tau;
    // ‖B_j‖/(2β_j) and 2β_j τ of every exponential with later ones.
    let mut weights = Vec::with_capacity(rotations.len());
    let mut beta = 0.0;
    for &(_, b) in rotations.iter().rev() {
        if beta > 0.0 {
            weights.push((b.abs() / (2.0 * beta), 2.0 * beta * tau));
        }
        beta += b.abs();
    }
    let tail = |degree: usize| -> f64 {
        weights
            .iter()
            .map(|&(w, x)| w * taylor_tail(x, degree + 1) / 2.0)
            .sum()
    };
    // (NaN gives up too.)
    let reach = tail(MAX_SERIES_DEGREE);
    if reach.is_nan() || reach >= other {
        return (f64::INFINITY, 0);
    }
    let key = SeriesKey {
        order,
        terms: scaled
            .iter()
            .map(|(c, s)| (c.to_bits(), s.x_mask(), s.z_mask()))
            .collect(),
    };
    let mut degree = order + 1;
    let mut best = (f64::INFINITY, 0);
    let mut spent = 0.0;
    loop {
        let Some(norms) = cached_series(&key, &rotations, &scaled, degree, &mut spent) else {
            return best;
        };
        // Coefficients below order p vanish exactly; they are kept for the
        // rounding they hold (a floor of about 10⁻¹⁵ per unit of λt).
        let lead: f64 = (0..=degree)
            .map(|m| norms[m] * tau.powi(m as i32 + 1) / (m + 1) as f64)
            .sum();
        let rest = tail(degree);
        let value = lead + rest;
        if value < best.0 {
            best = (value, degree);
        }
        if rest <= 0.01 * lead || degree >= MAX_SERIES_DEGREE {
            return best;
        }
        // The smallest degree whose tail is small enough, unless it cannot
        // beat the other bounds.
        let mut next = degree + 1;
        while next < MAX_SERIES_DEGREE && tail(next) > 0.01 * lead {
            next += 1;
        }
        let hope = tail(next) + lead;
        if hope.is_nan() || hope >= other.min(best.0) {
            return best;
        }
        degree = next;
    }
}

/// An error series of earlier calls: the order and the terms of `H/λ`
/// (coefficient bits and masks), compared in full.
#[derive(Clone, PartialEq)]
struct SeriesKey {
    order: usize,
    terms: Vec<(u64, u64, u64)>,
}

/// A computed series of some degree, or a degree that was out of reach
/// with the budget left then.
#[derive(Clone)]
enum CachedSeries {
    Done(usize, Vec<f64>),
    OutOfReach { degree: usize, budget: f64 },
}

/// Series of the last calls: loops over step counts or times on one
/// Hamiltonian compute each series once.
static SERIES_CACHE: Mutex<Vec<(SeriesKey, Vec<CachedSeries>)>> = Mutex::new(Vec::new());
/// Hamiltonians kept in [`SERIES_CACHE`].
const SERIES_CACHE_SIZE: usize = 16;

/// The norms of the error series up to `degree`, from the cache or
/// computed within what is left of [`MAX_SERIES_WORK`] (`spent` counts
/// the work of this bound).
fn cached_series(
    key: &SeriesKey,
    rotations: &[(PauliString, f64)],
    terms: &[(f64, PauliString)],
    degree: usize,
    spent: &mut f64,
) -> Option<Vec<f64>> {
    let budget = MAX_SERIES_WORK - *spent;
    {
        let cache = SERIES_CACHE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, entries)) = cache.iter().find(|(k, _)| k == key) {
            for e in entries {
                match e {
                    // A series of a higher degree holds this one.
                    CachedSeries::Done(d, norms) if *d >= degree => {
                        return Some(norms[..=degree].to_vec());
                    }
                    CachedSeries::OutOfReach {
                        degree: d,
                        budget: b,
                    } if *d <= degree && *b >= budget => {
                        return None;
                    }
                    _ => {}
                }
            }
        }
    }
    let result = error_series(rotations, terms, degree, budget);
    let entry = match &result {
        Some((norms, work)) => {
            *spent += work;
            CachedSeries::Done(degree, norms.clone())
        }
        None => CachedSeries::OutOfReach { degree, budget },
    };
    let mut cache = SERIES_CACHE.lock().unwrap_or_else(|p| p.into_inner());
    match cache.iter().position(|(k, _)| k == key) {
        Some(i) => cache[i].1.push(entry),
        None => {
            if cache.len() >= SERIES_CACHE_SIZE {
                cache.remove(0);
            }
            cache.push((key.clone(), vec![entry]));
        }
    }
    result.map(|(norms, _)| norms)
}

/// Most Pauli strings in one coefficient of the commutator series of
/// [`TrotterBounds::commutators`]; larger series are not computed.
pub const MAX_COMMUTATOR_TERMS: usize = 1 << 19;
/// Most operations on strings (scans and updates) the commutator series of
/// one bound may take together, about two seconds (series are cached
/// across calls).
const MAX_COMMUTATOR_WORK: f64 = 5e7;

/// One coefficient of a power series in τ: per Pauli string, the exact
/// coefficient and that of the majorant.
#[derive(Clone, Default)]
struct Coefficient {
    entries: Vec<(PauliString, C64, f64)>,
    index: HashMap<PauliString, u32, BuildHasherDefault<MaskHasher>>,
    /// The entries acting on each qubit: an exponential only changes the
    /// strings that overlap it.
    by_qubit: Vec<Vec<u32>>,
}

impl Coefficient {
    fn add(&mut self, s: PauliString, exact: C64, majorant: f64) {
        match self.index.get(&s) {
            Some(&i) => {
                let e = &mut self.entries[i as usize];
                e.1 += exact;
                e.2 += majorant;
            }
            None => {
                let i = self.entries.len() as u32;
                self.index.insert(s, i);
                self.entries.push((s, exact, majorant));
                let mut support = s.x_mask() | s.z_mask();
                while support != 0 {
                    let q = support.trailing_zeros() as usize;
                    if self.by_qubit.len() <= q {
                        self.by_qubit.resize(q + 1, Vec::new());
                    }
                    self.by_qubit[q].push(i);
                    support &= support - 1;
                }
            }
        }
    }

    /// The entries whose strings anticommute with `p`, each once; returns
    /// the entries looked at.
    fn anticommuting(&self, p: &PauliString, out: &mut Vec<(PauliString, C64, f64)>) -> usize {
        let mut support = p.x_mask() | p.z_mask();
        let mut seen = 0u64;
        let mut looked = 0;
        while support != 0 {
            let q = support.trailing_zeros() as usize;
            if let Some(list) = self.by_qubit.get(q) {
                looked += list.len();
                for &i in list {
                    let e = self.entries[i as usize];
                    // Counted at the first qubit of p it acts on.
                    if (e.0.x_mask() | e.0.z_mask()) & seen == 0 && !p.commutes_with(&e.0) {
                        out.push(e);
                    }
                }
            }
            seen |= 1 << q;
            support &= support - 1;
        }
        looked
    }
}

/// Norms of the coefficients `C_m`, m = 0..=degree, of the error generator
/// of one unit step (see [`error_series`]): `one` the sum of absolute
/// Pauli coefficients (at least the spectral norm), `two` their root sum
/// of squares (the normalised Frobenius norm), and `majorant` the sum over
/// every exponential j and multi-index q with |q| = m of
/// `‖ad_{B_J}^{q_J} ⋯ ad_{B_{j+1}}^{q_{j+1}} B_j‖ / q!` — the nested
/// commutators of the conjugations, each a single Pauli string, those of
/// commuting strings zero.
#[derive(Clone)]
struct CommutatorSeries {
    one: Vec<f64>,
    two: Vec<f64>,
    majorant: Vec<f64>,
}

impl CommutatorSeries {
    fn truncated(&self, degree: usize) -> Self {
        Self {
            one: self.one[..=degree].to_vec(),
            two: self.two[..=degree].to_vec(),
            majorant: self.majorant[..=degree].to_vec(),
        }
    }

    fn degree(&self) -> usize {
        self.one.len() - 1
    }
}

/// `exp(τ ad_B)`, `B = −ibP`, on the series `y` in place: it leaves the
/// strings that commute with P alone and maps an anticommuting `Q` (with
/// `PQ = φR`) to `cos(2bτ) Q − iφ sin(2bτ) R`, whose Taylor coefficients
/// go to higher powers of τ. The majorant takes absolute values of every
/// term. `None` when a coefficient would exceed [`MAX_COMMUTATOR_TERMS`]
/// strings or `work` would exceed `budget`.
fn conjugate(
    y: &mut [Coefficient],
    p: PauliString,
    b: f64,
    hits: &mut Vec<(PauliString, C64, f64)>,
    work: &mut f64,
    budget: f64,
) -> Option<()> {
    let degree = y.len() - 1;
    let factor = C64::new(0.0, -2.0 * b);
    // From the highest degree down, so that what this exponential adds to
    // higher degrees is not conjugated again.
    for d in (0..degree).rev() {
        hits.clear();
        let looked = y[d].anticommuting(&p, hits);
        *work += looked as f64 + (hits.len() * (degree - d)) as f64;
        if *work > budget {
            return None;
        }
        for &(q, c, a) in hits.iter() {
            let (phase, r) = p.multiply(&q);
            let (mut exact, mut majorant) = (c, a);
            for k in 1..=degree - d {
                let on_r = k % 2 == 1;
                // ad_B Q = −2ib φ R and ad_B R = −2ib φ̄ Q.
                let turn = if on_r { phase } else { phase.conj() };
                exact *= factor * turn / k as f64;
                majorant *= 2.0 * b.abs() / k as f64;
                y[d + k].add(if on_r { r } else { q }, exact, majorant);
            }
        }
        if y[d + 1..]
            .iter()
            .any(|c| c.entries.len() > MAX_COMMUTATOR_TERMS)
        {
            return None;
        }
    }
    Some(())
}

/// The series of [`CommutatorSeries`] to `degree` for the error generator
/// factored at `split`: with `S = L R`, `R = E_k ⋯ E_1` the first `split`
/// exponentials and L the rest, `S' − AS = L T R` with
/// `T = Σ_{j≤k} Ad_{E_k⋯E_{j+1}}(B_j) + Σ_{j>k} Ad_{E_{k+1}^{-1}⋯E_{j−1}^{-1}}(B_j)
/// − Ad_{E_{k+1}^{-1}⋯E_J^{-1}}(A)`, and `‖S(τ) − e^{τA}‖ ≤ ∫₀^τ ‖T‖`
/// since L and R are unitary. `split` = J is the form of
/// [`error_series`]; a split in the middle halves the depth of the
/// conjugations (as Childs et al. do for their fourth-order bounds). Each
/// sum is built by Horner's scheme (`Y ← B_j + Ad_{E_j} Y` forwards over
/// R, `W ← B_j + Ad_{E_j^{-1}} W` backwards over L from `W = −A`) with
/// [`conjugate`]. `None` when out of the limits; otherwise also the
/// operations spent.
fn commutator_series(
    rotations: &[(PauliString, f64)],
    terms: &[(f64, PauliString)],
    degree: usize,
    budget: f64,
    split: usize,
) -> Option<(CommutatorSeries, f64)> {
    let mut hits: Vec<(PauliString, C64, f64)> = Vec::new();
    let mut work = 0.0;
    let mut right: Vec<Coefficient> = vec![Coefficient::default(); degree + 1];
    for &(p, b) in &rotations[..split] {
        conjugate(&mut right, p, b, &mut hits, &mut work, budget)?;
        right[0].add(p, C64::new(0.0, -b), b.abs());
    }
    let mut left: Vec<Coefficient> = vec![Coefficient::default(); degree + 1];
    for &(h, p) in terms {
        left[0].add(p, C64::new(0.0, h), h.abs());
    }
    for &(p, b) in rotations[split..].iter().rev() {
        conjugate(&mut left, p, -b, &mut hits, &mut work, budget)?;
        left[0].add(p, C64::new(0.0, -b), b.abs());
    }
    for (r, l) in right.iter_mut().zip(&left) {
        for &(s, c, a) in &l.entries {
            r.add(s, c, a);
        }
    }
    let series = CommutatorSeries {
        one: right
            .iter()
            .map(|c| c.entries.iter().map(|e| e.1.norm()).sum())
            .collect(),
        two: right
            .iter()
            .map(|c| c.entries.iter().map(|e| e.1.norm_sqr()).sum::<f64>().sqrt())
            .collect(),
        majorant: right
            .iter()
            .map(|c| c.entries.iter().map(|e| e.2).sum())
            .collect(),
    };
    Some((series, work))
}

/// Commutator series of the last calls (see [`SERIES_CACHE`]).
/// Cached series per Hamiltonian and factorisation.
type CommutatorEntries = Vec<((SeriesKey, usize), Vec<CachedCommutators>)>;

static COMMUTATOR_CACHE: Mutex<CommutatorEntries> = Mutex::new(Vec::new());

#[derive(Clone)]
enum CachedCommutators {
    Done(CommutatorSeries),
    OutOfReach { degree: usize, budget: f64 },
}

/// The commutator series to `degree` factored at `split`, from the cache
/// or computed within what is left of [`MAX_COMMUTATOR_WORK`] (`spent`
/// counts the work of this bound).
fn cached_commutators(
    key: &SeriesKey,
    rotations: &[(PauliString, f64)],
    terms: &[(f64, PauliString)],
    degree: usize,
    split: usize,
    spent: &mut f64,
) -> Option<CommutatorSeries> {
    let budget = MAX_COMMUTATOR_WORK - *spent;
    let key = (key.clone(), split);
    {
        let cache = COMMUTATOR_CACHE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, entries)) = cache.iter().find(|(k, _)| *k == key) {
            for e in entries {
                match e {
                    CachedCommutators::Done(s) if s.degree() >= degree => {
                        return Some(s.truncated(degree));
                    }
                    CachedCommutators::OutOfReach {
                        degree: d,
                        budget: b,
                    } if *d <= degree && *b >= budget => return None,
                    _ => {}
                }
            }
        }
    }
    let result = commutator_series(rotations, terms, degree, budget, split).map(|(s, work)| {
        *spent += work;
        s
    });
    let entry = match &result {
        Some(s) => CachedCommutators::Done(s.clone()),
        None => {
            *spent = MAX_COMMUTATOR_WORK;
            CachedCommutators::OutOfReach { degree, budget }
        }
    };
    let mut cache = COMMUTATOR_CACHE.lock().unwrap_or_else(|p| p.into_inner());
    match cache.iter().position(|(k, _)| *k == key) {
        Some(i) => cache[i].1.push(entry),
        None => {
            if cache.len() >= 2 * SERIES_CACHE_SIZE {
                cache.remove(0);
            }
            cache.push((key, vec![entry]));
        }
    }
    result
}

/// λ, the terms of `H/λ` and the exponentials of one unit step.
type UnitStep = (f64, Vec<(f64, PauliString)>, Vec<(PauliString, f64)>);

/// The terms of `H/λ`, `λ = Σ|h_γ|`, and the exponentials `exp(B_j)`,
/// `B_j = −i b_j P_j`, of one unit step of the formula of `order`.
fn unit_step(terms: &[(f64, PauliString)], order: usize) -> UnitStep {
    let lambda: f64 = terms.iter().map(|(c, _)| c.abs()).sum();
    let scaled: Vec<(f64, PauliString)> = terms.iter().map(|&(c, s)| (c / lambda, s)).collect();
    // Rotation exp(−iθP/2) is exp(τ B) with B = −i(θ/2)P.
    let rotations = step_rotations(&scaled, 1.0, order)
        .into_iter()
        .map(|(p, theta)| (p, theta / 2.0))
        .collect();
    (lambda, scaled, rotations)
}

/// The factorisations of the error generator tried (see
/// [`commutator_series`]): all exponentials on the right, and a split in
/// the middle.
fn splits(rotations: usize) -> Vec<usize> {
    let mut out = vec![rotations, rotations / 2];
    out.dedup();
    out
}

/// Per step of length τ: the bound of [`TrotterBounds::commutators`] on
/// `‖S(τ) − e^{−iHτ}‖` and that of [`TrotterBounds::average`] on
/// `‖S(τ) − e^{−iHτ}‖_F/√d`, infinite when the series is out of reach.
/// Each expands `T(σ)` to a degree D: the coefficients below D exactly,
/// the remainder of every conjugation by its Taylor remainder in integral
/// form, at most `σ^D Σ_{|q|=D} ‖ad^q B_j‖/q!` (the conjugations are
/// unitary). So, with `x = λτ` and the series of `H/λ`,
/// `∫₀^τ ‖T‖ ≤ Σ_{m<D} ‖C_m‖ x^{m+1}/(m+1) + M_D x^{D+1}/(D+1)`, and the
/// same with the Frobenius norm, whose unitary invariance carries the
/// integral representation over. The series is computed for both
/// factorisations of [`splits`], to the degree of the nested bound (at
/// least p + 1) and higher while the remainder is above 1 % of the rest
/// and each degree lowers the bound by 5 %; the smallest value over the
/// degrees D from p up is taken. A series out of reach at the first
/// degree is tried at p + 1 and p.
fn commutator_step_bounds(
    terms: &[(f64, PauliString)],
    tau: f64,
    order: usize,
    degree: usize,
) -> (f64, f64) {
    if terms.is_empty() || tau == 0.0 {
        return (0.0, 0.0);
    }
    let (lambda, scaled, rotations) = unit_step(terms, order);
    let x = lambda * tau;
    let key = SeriesKey {
        order,
        terms: scaled
            .iter()
            .map(|(c, s)| (c.to_bits(), s.x_mask(), s.z_mask()))
            .collect(),
    };
    // The best value of each split's series to degree `d`, its lead and
    // its rest (worst case), and the average-case value.
    let power = |m: usize| x.powi(m as i32 + 1) / (m + 1) as f64;
    let evaluate = |series: &CommutatorSeries| {
        let mut best = (f64::INFINITY, 0.0, 0.0, f64::INFINITY);
        let (mut lead_one, mut lead_two) = (0.0, 0.0);
        for d in 0..=series.degree() {
            if d >= order {
                let rest = series.majorant[d] * power(d);
                if lead_one + rest < best.0 {
                    best = (lead_one + rest, lead_one, rest, best.3);
                }
                best.3 = best.3.min(lead_two + rest);
            }
            lead_one += series.one[d] * power(d);
            lead_two += series.two[d] * power(d);
        }
        best
    };
    let (mut worst, mut average) = (f64::INFINITY, f64::INFINITY);
    for split in splits(rotations.len()) {
        // From the degree of the nested bound (at least p + 1) up while the
        // remainder is more than 1 % of the rest and each degree gains 5 %
        // (each costs about twice the last), within the limits; out of
        // reach at the first degree, p + 1 and p are tried.
        let first = degree.clamp(order + 1, MAX_SERIES_DEGREE);
        let mut tries = vec![first, order + 1, order];
        tries.dedup();
        // Each split, and each first degree tried, has the whole budget; a
        // series out of reach takes all that was left.
        let mut spent = 0.0;
        let Some((mut d, series)) = tries.iter().find_map(|&d| {
            spent = 0.0;
            cached_commutators(&key, &rotations, &scaled, d, split, &mut spent).map(|s| (d, s))
        }) else {
            continue;
        };
        let mut best = evaluate(&series);
        while d < MAX_SERIES_DEGREE && best.2 > 0.01 * best.1 && d >= first {
            d += 1;
            let Some(series) = cached_commutators(&key, &rotations, &scaled, d, split, &mut spent)
            else {
                break;
            };
            let next = evaluate(&series);
            let gain = next.0 < 0.95 * best.0;
            best = (next.0.min(best.0), next.1, next.2, next.3.min(best.3));
            if !gain {
                break;
            }
        }
        worst = worst.min(best.0);
        average = average.min(best.3);
    }
    (worst, average)
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
    let widest = terms
        .iter()
        .map(|(_, s)| rotation_cost(s))
        .max()
        .unwrap_or(0);
    if samples.saturating_mul(widest) > MAX_GATES {
        return Err(QsoError::invalid(
            "samples",
            format!("the circuit would exceed {MAX_GATES} gates"),
        ));
    }
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
        assert!(trotter_circuit(&h, 1, 1.0, MAX_GATES, 2).is_err());
        assert!(qdrift_circuit(&h, 1, 1.0, MAX_GATES, 0).is_err());
        // Overflowing angles are errors, not panics.
        let big: PauliSum = "1e308 X0 + 1e308 Z0".parse().unwrap();
        assert!(trotter_circuit(&big, 1, 10.0, 1, 1).is_err());
        assert!(qdrift_circuit(&big, 1, 1.0, 10, 0).is_err());
        assert!(qdrift_circuit(&h, 1, 1e308, 1, 0).is_err());
        // Coefficients whose commutators overflow give the trivial bound.
        let huge: PauliSum = "1e160 X0 + 1e160 Z0".parse().unwrap();
        for order in [1, 2, 4] {
            assert_eq!(trotter_error_bound(&huge, 1.0, 1, order).unwrap(), 2.0);
        }
    }

    #[test]
    fn nested_bounds_hold_and_fall_with_the_order() {
        let n = 4;
        let hs = [
            heisenberg_xxz(n, 1.0, 0.5, false),
            transverse_field_ising(n, 1.0, 0.7, false),
            random_hamiltonian(n, 8, 3),
        ];
        for h in &hs {
            let terms = nontrivial_terms(h);
            let target = exact(h, n, 1.0);
            for order in [1usize, 2, 4] {
                // The coefficients below order p vanish.
                let rotations: Vec<(PauliString, f64)> = step_rotations(&terms, 1.0, order)
                    .into_iter()
                    .map(|(p, theta)| (p, theta / 2.0))
                    .collect();
                let (norms, _) =
                    error_series(&rotations, &terms, order + 1, f64::INFINITY).unwrap();
                assert!(norms[..order].iter().all(|c| *c < 1e-12), "{norms:?}");
                assert!(norms[order] > 1e-3);
                for steps in [1usize, 4, 16] {
                    let c = trotter_circuit(h, n, 1.0, steps, order).unwrap();
                    let e = spectral_norm(&(unitary(&c) - &target));
                    let b = trotter_error_bounds(h, 1.0, steps, order).unwrap();
                    for v in [b.nested, b.stages, b.taylor, b.bound] {
                        assert!(v >= e, "order {order}, {steps} steps: {v} < {e}");
                    }
                    assert_eq!(b.bound, b.commutators.min(b.stages).min(b.taylor).min(2.0));
                }
            }
            // Order 4 falls as r⁻⁴.
            let b64 = trotter_error_bounds(h, 1.0, 64, 4).unwrap().nested;
            let b128 = trotter_error_bounds(h, 1.0, 128, 4).unwrap().nested;
            assert!((b64 / b128 / 16.0 - 1.0).abs() < 0.1, "{}", b64 / b128);
        }
        // A large series is given up within the budget, and the answer is
        // still the smallest bound.
        let big = random_hamiltonian(8, 40, 5);
        let b = trotter_error_bounds(&big, 1.0, 64, 6).unwrap();
        assert_eq!(b.bound, b.commutators.min(b.stages).min(b.taylor).min(2.0));
    }

    #[test]
    fn nested_bounds_do_not_depend_on_the_units_of_h() {
        // H·s for time t/s is the same evolution.
        let h = random_hamiltonian(3, 5, 9);
        for (order, steps) in [(4usize, 8usize), (6, 8), (8, 8)] {
            let unit = trotter_error_bounds(&h, 1.0, steps, order).unwrap().nested;
            assert!(unit.is_finite(), "order {order}");
            for scale in [1e-6, 1e6, 1e12, 1e24, 1e40] {
                // Same terms in the same order (add_term, as `h` was built).
                let mut hs = PauliSum::new();
                for &(c, p) in h.terms() {
                    hs.add_term(c * scale, p);
                }
                let b = trotter_error_bounds(&hs, 1.0 / scale, steps, order).unwrap();
                assert!(
                    (b.nested / unit - 1.0).abs() < 1e-6,
                    "order {order}, scale {scale}: {} vs {unit}",
                    b.nested
                );
            }
        }
        // A second call reuses the series.
        let h = heisenberg_xxz(6, 1.0, 0.5, false);
        let first = trotter_error_bounds(&h, 1.0, 40, 4).unwrap();
        let start = std::time::Instant::now();
        let again = trotter_error_bounds(&h, 1.0, 40, 4).unwrap();
        assert_eq!(first, again);
        assert!(
            start.elapsed().as_secs_f64() < crate::time_limit(0.05),
            "{:?}",
            start.elapsed()
        );
    }
}

#[cfg(test)]
mod commutator_tests;
