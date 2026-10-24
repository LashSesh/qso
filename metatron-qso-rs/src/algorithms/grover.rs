//! Grover search (Grover, STOC 1996) as an explicit gate circuit.
//!
//! With `M` marked items among `N = 2ⁿ` and `sin θ = √(M/N)`, `k` Grover
//! iterations succeed with probability `sin²((2k+1)θ)`. The simulation
//! reproduces this law exactly (see tests).

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, StateVector};

/// Grover rotation angle `θ = asin √(M/N)`.
fn angle(n_qubits: usize, n_marked: usize) -> f64 {
    ((n_marked as f64) / 2f64.powi(n_qubits as i32))
        .sqrt()
        .asin()
}

/// Number of iterations maximising the success probability.
pub fn optimal_iterations(n_qubits: usize, n_marked: usize) -> usize {
    let theta = angle(n_qubits, n_marked);
    if theta <= 0.0 {
        return 0;
    }
    (std::f64::consts::PI / (4.0 * theta) - 0.5)
        .round()
        .max(0.0) as usize
}

/// `sin²((2k+1)θ)`.
pub fn theoretical_success(n_qubits: usize, n_marked: usize, iterations: usize) -> f64 {
    let theta = angle(n_qubits, n_marked);
    ((2 * iterations + 1) as f64 * theta).sin().powi(2)
}

fn validate(n_qubits: usize, marked: &[usize]) -> Result<Vec<usize>> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QsoError::invalid("n_qubits", "need 1..=30 qubits"));
    }
    if marked.is_empty() {
        return Err(QsoError::invalid(
            "marked",
            "at least one marked item required",
        ));
    }
    let mut m = marked.to_vec();
    m.sort_unstable();
    m.dedup();
    if let Some(&bad) = m.iter().find(|&&x| x >= 1 << n_qubits) {
        return Err(QsoError::invalid(
            "marked",
            format!("item {bad} outside 0..2^{n_qubits}"),
        ));
    }
    Ok(m)
}

/// Phase oracle flipping the sign of every marked basis state.
pub fn oracle(n_qubits: usize, marked: &[usize]) -> Result<Circuit> {
    let marked = validate(n_qubits, marked)?;
    let all: Vec<usize> = (0..n_qubits).collect();
    let mut c = Circuit::new(n_qubits);
    for &w in &marked {
        let zeros: Vec<usize> = (0..n_qubits).filter(|q| (w >> q) & 1 == 0).collect();
        for &q in &zeros {
            c.x(q);
        }
        c.mcz(&all);
        for &q in &zeros {
            c.x(q);
        }
    }
    Ok(c)
}

/// Diffusion operator `2|s⟩⟨s| − I` (up to a global phase).
pub fn diffusion(n_qubits: usize) -> Circuit {
    let all: Vec<usize> = (0..n_qubits).collect();
    let mut c = Circuit::new(n_qubits);
    for q in 0..n_qubits {
        c.h(q).x(q);
    }
    c.mcz(&all);
    for q in 0..n_qubits {
        c.x(q).h(q);
    }
    c
}

/// Full Grover circuit: Hadamards, then `iterations` × (oracle, diffusion).
pub fn grover_circuit(n_qubits: usize, marked: &[usize], iterations: usize) -> Result<Circuit> {
    let o = oracle(n_qubits, marked)?;
    let d = diffusion(n_qubits);
    let mut c = Circuit::new(n_qubits);
    for q in 0..n_qubits {
        c.h(q);
    }
    for _ in 0..iterations {
        c.append(&o)?;
        c.append(&d)?;
    }
    Ok(c)
}

/// Outcome of [`grover_search`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroverResult {
    pub n_qubits: usize,
    pub marked: Vec<usize>,
    pub iterations: usize,
    /// Simulated probability of measuring a marked item.
    pub success_probability: f64,
    /// `sin²((2k+1)θ)`.
    pub theoretical_success_probability: f64,
    /// Number of gates and two-or-more-qubit gates of the circuit.
    pub gates: usize,
    pub multi_qubit_gates: usize,
}

/// Simulate Grover search (`iterations = None` → optimal count).
pub fn grover_search(
    n_qubits: usize,
    marked: &[usize],
    iterations: Option<usize>,
) -> Result<GroverResult> {
    let marked = validate(n_qubits, marked)?;
    let k = iterations.unwrap_or_else(|| optimal_iterations(n_qubits, marked.len()));
    let circuit = grover_circuit(n_qubits, &marked, k)?;
    let state = StateVector::from_circuit(&circuit, &[])?;
    let probs = state.probabilities();
    Ok(GroverResult {
        n_qubits,
        iterations: k,
        success_probability: marked.iter().map(|&w| probs[w]).sum(),
        theoretical_success_probability: theoretical_success(n_qubits, marked.len(), k),
        gates: circuit.len(),
        multi_qubit_gates: circuit.multi_qubit_gate_count(),
        marked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_follows_sine_law_for_every_iteration_count() {
        for (n, marked) in [
            (3usize, vec![5usize]),
            (5, vec![3, 17]),
            (6, vec![0, 9, 33, 62]),
        ] {
            for k in 0..8 {
                let r = grover_search(n, &marked, Some(k)).unwrap();
                assert!(
                    (r.success_probability - r.theoretical_success_probability).abs() < 1e-12,
                    "n={n} k={k}"
                );
            }
        }
    }

    #[test]
    fn optimal_iterations_give_high_success() {
        for n in [4usize, 8, 10, 12] {
            let r = grover_search(n, &[(1 << n) - 3], None).unwrap();
            assert!(
                r.success_probability > 0.94,
                "n={n}: {}",
                r.success_probability
            );
            let expected =
                (std::f64::consts::PI / 4.0 * ((1u64 << n) as f64).sqrt()).floor() as usize;
            assert!(r.iterations.abs_diff(expected) <= 1);
        }
    }

    #[test]
    fn invalid_inputs() {
        assert!(grover_search(3, &[8], None).is_err());
        assert!(grover_search(3, &[], None).is_err());
        assert_eq!(optimal_iterations(2, 4), 0);
    }
}
