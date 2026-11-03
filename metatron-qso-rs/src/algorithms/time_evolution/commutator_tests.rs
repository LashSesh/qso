//! Validation of the commutator bounds and the average error (1.56).

use super::*;
use crate::algorithms::models::{heisenberg_xxz, transverse_field_ising};
use crate::linalg::{CMatrix, expm_hermitian};
use crate::quantum::{Pauli, StateVector};

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

fn exact(h: &PauliSum, n: usize, t: f64) -> CMatrix {
    let traceless = PauliSum::from_terms(nontrivial_terms(h));
    expm_hermitian(&traceless.to_dense(n).unwrap(), t)
}

fn power(m: &CMatrix, mut k: usize) -> CMatrix {
    let mut base = m.clone();
    let mut out = CMatrix::identity(m.nrows(), m.ncols());
    while k > 0 {
        if k & 1 == 1 {
            out = &out * &base;
        }
        k >>= 1;
        if k > 0 {
            base = &base * &base;
        }
    }
    out
}

/// Spectral and normalised Frobenius norms of `S(t/r)^r − exp(−iHt)`.
fn dense_errors(h: &PauliSum, n: usize, t: f64, steps: usize, order: usize) -> (f64, f64) {
    let step = unitary(&trotter_circuit(h, n, t / steps as f64, 1, order).unwrap());
    let d = power(&step, steps) - exact(h, n, t);
    let spectral = d.clone().singular_values().max();
    let frobenius =
        d.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() / ((1usize << n) as f64).sqrt();
    (spectral, frobenius)
}

fn random_hamiltonian(n: usize, terms: usize, seed: u64) -> PauliSum {
    let mut rng = Rng::seed_from_u64(seed);
    let mut h = PauliSum::new();
    for _ in 0..terms {
        let factors: Vec<(usize, Pauli)> = (0..n)
            .filter_map(|q| {
                let p = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][rng.below(4) as usize];
                (p != Pauli::I).then_some((q, p))
            })
            .collect();
        h.add_term(
            rng.uniform_range(-1.0, 1.0),
            PauliString::new(&factors).unwrap(),
        );
    }
    h
}

/// The Heisenberg chain of Childs et al. (their Eq. (39)),
/// `Σ_{j<n} (X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1} + h_j Z_j)`, `h_j`
/// uniform in [−1, 1], its terms in the even-odd order (bonds from an even
/// site first, each with the field on its left site) or in the X-Y-Z
/// order.
pub(super) fn childs_heisenberg(n: usize, seed: u64, even_odd: bool) -> PauliSum {
    let mut rng = Rng::seed_from_u64(seed);
    let fields: Vec<f64> = (0..n - 1).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
    let pair = |j: usize, p: Pauli| PauliString::new(&[(j, p), (j + 1, p)]).unwrap();
    let mut h = PauliSum::new();
    if even_odd {
        for parity in [0, 1] {
            for j in (parity..n - 1).step_by(2) {
                for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                    h.add_term(1.0, pair(j, p));
                }
                h.add_term(fields[j], PauliString::single(j, Pauli::Z));
            }
        }
    } else {
        for p in [Pauli::X, Pauli::Y] {
            for j in 0..n - 1 {
                h.add_term(1.0, pair(j, p));
            }
        }
        for (j, &field) in fields.iter().enumerate() {
            h.add_term(1.0, pair(j, Pauli::Z));
            h.add_term(field, PauliString::single(j, Pauli::Z));
        }
    }
    h
}

/// Smallest r with `‖S(t/r)^r − exp(−iHt)‖ ≤ ε` (dense; the error falls
/// with r here).
fn empirical_steps(h: &PauliSum, n: usize, t: f64, epsilon: f64, order: usize) -> usize {
    let mut hi = 1;
    while dense_errors(h, n, t, hi, order).0 > epsilon {
        hi *= 2;
    }
    let mut lo = hi / 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if dense_errors(h, n, t, mid, order).0 <= epsilon {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[test]
fn commutator_bounds_hold_against_dense_errors() {
    let n = 4;
    let hs = [
        heisenberg_xxz(n, 1.0, 0.5, false),
        transverse_field_ising(n, 1.0, 0.7, false),
        random_hamiltonian(n, 8, 3),
        childs_heisenberg(n, 1, true),
    ];
    for h in &hs {
        let terms = nontrivial_terms(h);
        for order in [1usize, 2, 4, 6] {
            // The series: coefficients below order p vanish, and the
            // majorant bounds the exact coefficients.
            let (_, scaled, rotations) = unit_step(&terms, order);
            let (s, _) = commutator_series(
                &rotations,
                &scaled,
                order + 2,
                f64::INFINITY,
                rotations.len() / 2,
            )
            .unwrap();
            assert!(s.one[..order].iter().all(|c| *c < 1e-12), "{:?}", s.one);
            for m in 1..=order + 2 {
                assert!(
                    s.two[m] <= s.one[m] * (1.0 + 1e-12)
                        && s.one[m] <= s.majorant[m] * (1.0 + 1e-12)
                );
            }
            for steps in [1usize, 4, 16] {
                let (e, f) = dense_errors(h, n, 1.0, steps, order);
                let b = trotter_error_bounds(h, 1.0, steps, order).unwrap();
                assert!(
                    b.commutators >= e,
                    "order {order}, {steps} steps: {} < {e}",
                    b.commutators
                );
                assert!(
                    b.average >= f,
                    "order {order}, {steps} steps: {} < {f}",
                    b.average
                );
                assert!(b.commutators <= b.nested);
                assert!(b.average <= b.bound && f <= e * (1.0 + 1e-12));
                assert_eq!(b.bound, b.commutators.min(b.stages).min(b.taylor).min(2.0));
            }
        }
    }
}

#[test]
fn average_error_is_the_error_on_random_states() {
    let n = 6;
    let h = childs_heisenberg(n, 2, true);
    let (t, steps, order) = (1.0, 16, 2);
    let step = unitary(&trotter_circuit(&h, n, t / steps as f64, 1, order).unwrap());
    let d = power(&step, steps) - exact(&h, n, t);
    let dim = 1usize << n;
    let frobenius2 = d.iter().map(|z| z.norm_sqr()).sum::<f64>() / dim as f64;
    // Haar-random states: normalised complex Gaussian vectors.
    let mut rng = Rng::seed_from_u64(7);
    let samples = 400;
    let mut values = Vec::with_capacity(samples);
    for _ in 0..samples {
        let psi: Vec<C64> = (0..dim)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let norm = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let v = nalgebra::DVector::from_iterator(dim, psi.iter().map(|z| z / norm));
        values.push((&d * v).norm_squared());
    }
    let mean = values.iter().sum::<f64>() / samples as f64;
    let sd = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples - 1) as f64).sqrt();
    let se = sd / (samples as f64).sqrt();
    assert!(
        (mean - frobenius2).abs() < 4.0 * se,
        "{mean} vs {frobenius2} ± {se}"
    );
    let b = trotter_error_bounds(&h, t, steps, order).unwrap();
    assert!(
        b.average >= frobenius2.sqrt() && b.average < b.bound,
        "{b:?}"
    );
    let worst = d.clone().singular_values().max();
    assert!(b.bound >= worst);
}

/// Published step counts of Childs et al. (their Fig. 5; mean of five
/// random instances): n, the empirical r, their bound.
const CHILDS_EVEN_ODD: [(usize, f64, f64); 2] = [(8, 86.4, f64::NAN), (10, 126.4, 645.4)];
const CHILDS_XYZ: [(usize, f64, f64); 2] = [(8, 99.2, f64::NAN), (10, 134.4, 968.066)];

#[test]
fn heisenberg_step_counts_lie_between_the_empirical_ones_and_the_published_bounds() {
    // Fourth order, t = n, ε = 10⁻³, as in Childs et al.
    for (even_odd, table) in [(true, CHILDS_EVEN_ODD), (false, CHILDS_XYZ)] {
        for (seed, &(n, empirical, published)) in table.iter().enumerate() {
            let h = childs_heisenberg(n, seed as u64 + 11, even_odd);
            let t = n as f64;
            let r = trotter_steps(&h, t, 1e-3, 4).unwrap();
            if n <= 8 {
                // This instance's own empirical count, and the published one.
                let own = empirical_steps(&h, n, t, 1e-3, 4);
                assert!(own <= r.worst, "n {n}: {own} > {r:?}");
                assert!(
                    (own as f64 / empirical - 1.0).abs() < 0.15,
                    "n {n}: {own} vs {empirical}"
                );
            } else {
                assert!(
                    r.worst as f64 > empirical && (r.worst as f64) < published,
                    "n {n}: {r:?}"
                );
            }
            assert!(r.average < r.worst, "{r:?}");
        }
    }
}

/// Step counts for the Heisenberg chains of Childs et al. against their
/// published bounds (`cargo test --release -- --ignored measure_childs
/// --nocapture`).
#[test]
#[ignore]
fn measure_childs_heisenberg() {
    let published_even_odd = [
        (10, 645.4),
        (16, 1347.8),
        (23, 2357.0),
        (32, 3897.4),
        (45, 6553.8),
        (64, 11198.2),
    ];
    let published_xyz = [
        (10, 968.066),
        (16, 2019.88),
        (23, 3505.5),
        (32, 5834.46),
        (45, 9783.66),
        (64, 16681.5),
    ];
    for (even_odd, table) in [(true, published_even_odd), (false, published_xyz)] {
        for &(n, published) in &table {
            let mut worst = Vec::new();
            let mut average = Vec::new();
            let start = std::time::Instant::now();
            for seed in 0..5 {
                let h = childs_heisenberg(n, 100 + seed, even_odd);
                let r = trotter_steps(&h, n as f64, 1e-3, 4).unwrap();
                worst.push(r.worst as f64);
                average.push(r.average as f64);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "{} n {n}: r {:.1} (published {published}), average {:.1}, {:.1} s for five",
                if even_odd { "even-odd" } else { "X-Y-Z" },
                mean(&worst),
                mean(&average),
                start.elapsed().as_secs_f64()
            );
        }
    }
}

/// Bounds against the exact errors of a Heisenberg chain of eight spins
/// (`cargo test --release -- --ignored measure_tightness --nocapture`).
#[test]
#[ignore]
fn measure_tightness() {
    let n = 8;
    for (name, h) in [
        (
            "Heisenberg (Childs et al.), even-odd",
            childs_heisenberg(n, 5, true),
        ),
        ("XXZ chain", heisenberg_xxz(n, 1.0, 0.5, false)),
    ] {
        for (order, steps) in [(2usize, 16usize), (2, 64), (4, 8), (4, 32), (6, 8)] {
            let (e, f) = dense_errors(&h, n, 2.0, steps, order);
            let start = std::time::Instant::now();
            let b = trotter_error_bounds(&h, 2.0, steps, order).unwrap();
            println!(
                "{name}, t = 2, order {order}, {steps} steps: error {e:.2e} (average {f:.2e}); \
                 nested {:.1}x, commutators {:.1}x, stages {:.1}x, average {:.1}x the average error, {:.2} s",
                b.nested / e,
                b.commutators / e,
                b.stages / e,
                b.average / f,
                start.elapsed().as_secs_f64()
            );
        }
    }
}
