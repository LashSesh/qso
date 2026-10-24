//! Quantum Fourier transform and phase estimation.
//!
//! Convention: `QFT|x⟩ = 2^{−n/2} Σ_y e^{2πi·xy/2ⁿ} |y⟩` with little-endian
//! basis indices (the matrix is the unitary DFT matrix).

use std::f64::consts::PI;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, StateVector};

/// QFT circuit on `n` qubits. Without the final swaps the output register
/// is bit-reversed.
pub fn qft(n: usize, swaps: bool) -> Circuit {
    let mut c = Circuit::new(n);
    for j in (0..n).rev() {
        c.h(j);
        for k in (0..j).rev() {
            c.cp(k, j, PI / 2f64.powi((j - k) as i32));
        }
    }
    if swaps {
        for q in 0..n / 2 {
            c.swap(q, n - 1 - q);
        }
    }
    c
}

/// Inverse QFT.
pub fn inverse_qft(n: usize, swaps: bool) -> Circuit {
    qft(n, swaps).inverse()
}

/// Result of [`estimate_phase`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseEstimate {
    /// Most likely phase `m/2ᵗ ∈ [0, 1)`.
    pub phase: f64,
    /// Probability of that outcome.
    pub probability: f64,
    /// Distribution over all `2ᵗ` outcomes `m`.
    pub distribution: Vec<f64>,
}

fn mat_mul(a: &[C64; 4], b: &[C64; 4]) -> [C64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Phase-estimation circuit with `precision` counting qubits (`0..t`, qubit
/// `k` carries weight `2^k`) and one target qubit `t` prepared in
/// `eigenstate` of the single-qubit `unitary`.
pub fn phase_estimation_circuit(
    precision: usize,
    unitary: [C64; 4],
    eigenstate: [C64; 2],
) -> Result<Circuit> {
    if precision == 0 || precision > 24 {
        return Err(QsoError::invalid(
            "precision",
            "need 1..=24 counting qubits",
        ));
    }
    let norm = (eigenstate[0].norm_sqr() + eigenstate[1].norm_sqr()).sqrt();
    if norm < 1e-12 {
        return Err(QsoError::ZeroNorm);
    }
    let (a, b) = (eigenstate[0] / norm, eigenstate[1] / norm);
    let t = precision;
    let mut c = Circuit::new(t + 1);
    // Unitary preparing the eigenstate from |0⟩: first column (a, b).
    c.unitary(vec![a, -b.conj(), b, a.conj()], &[t], &[])?;
    for k in 0..t {
        c.h(k);
    }
    let mut power = unitary;
    for k in 0..t {
        c.unitary(power.to_vec(), &[t], &[k])?;
        power = mat_mul(&power, &power);
    }
    let counting: Vec<usize> = (0..t).collect();
    c.append_mapped(&inverse_qft(t, true), &counting)?;
    Ok(c)
}

/// Estimate the eigenphase `φ` (`U|u⟩ = e^{2πiφ}|u⟩`) of a single-qubit
/// unitary with `precision` bits.
pub fn estimate_phase(
    precision: usize,
    unitary: [C64; 4],
    eigenstate: [C64; 2],
) -> Result<PhaseEstimate> {
    let c = phase_estimation_circuit(precision, unitary, eigenstate)?;
    let state = StateVector::from_circuit(&c, &[])?;
    let probs = state.probabilities();
    let t = precision;
    let mut distribution = vec![0.0; 1 << t];
    for (b, p) in probs.iter().enumerate() {
        distribution[b & ((1 << t) - 1)] += p;
    }
    let m = (0..distribution.len())
        .max_by(|&x, &y| distribution[x].total_cmp(&distribution[y]))
        .expect("non-empty");
    Ok(PhaseEstimate {
        phase: m as f64 / 2f64.powi(t as i32),
        probability: distribution[m],
        distribution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qft_equals_dft_matrix() {
        let n = 4;
        let dim = 1usize << n;
        let c = qft(n, true);
        for x in 0..dim {
            let psi = {
                let mut s = StateVector::basis(n, x).unwrap();
                s.apply_circuit(&c, &[]).unwrap();
                s
            };
            for y in 0..dim {
                let expected = C64::from_polar(
                    1.0 / (dim as f64).sqrt(),
                    2.0 * PI * (x * y) as f64 / dim as f64,
                );
                assert!(
                    (psi.amplitudes()[y] - expected).norm() < 1e-12,
                    "x={x} y={y}"
                );
            }
        }
    }

    #[test]
    fn inverse_qft_undoes_qft() {
        let n = 5;
        let mut rng = crate::rng::Rng::seed_from_u64(2);
        let psi0 = StateVector::random(n, &mut rng);
        let mut psi = psi0.clone();
        psi.apply_circuit(&qft(n, true), &[]).unwrap();
        psi.apply_circuit(&inverse_qft(n, true), &[]).unwrap();
        assert!((psi.fidelity(&psi0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exact_phases_are_recovered_with_certainty() {
        let t = 5;
        for m in [0usize, 1, 7, 13, 31] {
            let phi = m as f64 / 32.0;
            // P(2πφ) has eigenvector |1⟩ with eigenvalue e^{2πiφ}.
            let u = [
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::from_polar(1.0, 2.0 * PI * phi),
            ];
            let est = estimate_phase(t, u, [C64::new(0.0, 0.0), C64::new(1.0, 0.0)]).unwrap();
            assert!((est.phase - phi).abs() < 1e-12, "m={m}");
            assert!((est.probability - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn inexact_phase_is_best_t_bit_approximation() {
        // φ = 1/3 with 6 bits: the most likely outcome is round(64/3) = 21,
        // with probability ≥ 4/π² ≈ 0.405 (Nielsen & Chuang, eq. 5.27).
        let phi = 1.0 / 3.0;
        let u = [
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::from_polar(1.0, 2.0 * PI * phi),
        ];
        let est = estimate_phase(6, u, [C64::new(0.0, 0.0), C64::new(1.0, 0.0)]).unwrap();
        assert!((est.phase - 21.0 / 64.0).abs() < 1e-12);
        assert!(est.probability > 4.0 / (PI * PI));
        // A non-diagonal unitary: RY(θ) has eigenphases ±θ/(4π) on (1, ±i)/√2.
        let theta: f64 = 2.0 * PI * 0.25;
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let ry = [
            C64::new(c, 0.0),
            C64::new(-s, 0.0),
            C64::new(s, 0.0),
            C64::new(c, 0.0),
        ];
        let v = [C64::new(1.0, 0.0), C64::new(0.0, -1.0)];
        let est = estimate_phase(4, ry, v).unwrap();
        assert!(
            (est.phase - 0.125).abs() < 1e-12 && est.probability > 1.0 - 1e-10,
            "{est:?}"
        );
    }
}
