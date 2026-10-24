//! Parametrised circuit templates.

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::quantum::Circuit;

/// Pattern of two-qubit entangling gates in a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Entanglement {
    /// CX(i, i+1) for i = 0 … n−2.
    Linear,
    /// Linear plus CX(n−1, 0).
    Circular,
    /// CX(i, j) for all i < j.
    Full,
}

impl Entanglement {
    /// Qubit pairs of one entangling layer.
    pub fn pairs(self, n: usize) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        match self {
            Entanglement::Linear => {}
            Entanglement::Circular => {
                if n > 2 {
                    pairs.push((n - 1, 0));
                }
            }
            Entanglement::Full => {
                pairs = (0..n)
                    .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                    .collect();
            }
        }
        pairs
    }
}

/// Single-qubit rotation axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rotation {
    RX,
    RY,
    RZ,
}

/// Hardware-efficient ansatz: `layers` repetitions of (one rotation per axis
/// in `rotations` on every qubit, then CX entanglers), followed by a final
/// rotation layer. Parameters: `n·|rotations|·(layers+1)`.
pub fn hardware_efficient(
    n: usize,
    layers: usize,
    rotations: &[Rotation],
    entanglement: Entanglement,
) -> Circuit {
    let mut c = Circuit::new(n);
    let rotation_layer = |c: &mut Circuit| {
        for q in 0..n {
            for &r in rotations {
                let p = c.new_parameter();
                match r {
                    Rotation::RX => c.rx(q, p),
                    Rotation::RY => c.ry(q, p),
                    Rotation::RZ => c.rz(q, p),
                };
            }
        }
    };
    for _ in 0..layers {
        rotation_layer(&mut c);
        for (a, b) in entanglement.pairs(n) {
            c.cx(a, b);
        }
    }
    rotation_layer(&mut c);
    c
}

/// `RY` rotations and CX entanglers (real amplitudes only) — suitable for
/// real Hamiltonians such as H₂ or MaxCut.
pub fn real_amplitudes(n: usize, layers: usize, entanglement: Entanglement) -> Circuit {
    hardware_efficient(n, layers, &[Rotation::RY], entanglement)
}

/// `RY·RZ` rotations and CX entanglers.
pub fn efficient_su2(n: usize, layers: usize, entanglement: Entanglement) -> Circuit {
    hardware_efficient(n, layers, &[Rotation::RY, Rotation::RZ], entanglement)
}

/// Explicit gate-level QAOA circuit for MaxCut with parameters
/// `[γ₁ … γ_p, β₁ … β_p]`. `exp(−iγ·w(1 − ZZ)/2)` is implemented as
/// `RZZ(−γw)` (up to a global phase) and the mixer `exp(−iβΣX)` as `RX(2β)`
/// on every qubit. [`crate::algorithms::qaoa::MaxCutQaoa`] simulates the
/// same state with fast diagonal kernels; this circuit is for export and
/// for running on other backends.
pub fn qaoa_maxcut_circuit(graph: &Graph, p: usize) -> Result<Circuit> {
    let n = graph.num_nodes();
    if n == 0 || p == 0 {
        return Err(QsoError::invalid(
            "graph/p",
            "need at least one node and one layer",
        ));
    }
    let mut c = Circuit::new(n);
    let gammas = c.new_parameters(p);
    let betas = c.new_parameters(p);
    for q in 0..n {
        c.h(q);
    }
    for l in 0..p {
        for &(u, v, w) in graph.edges() {
            c.rzz(u, v, gammas[l] * -w);
        }
        for q in 0..n {
            c.rx(q, betas[l] * 2.0);
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_counts_and_structure() {
        let c = hardware_efficient(4, 2, &[Rotation::RY, Rotation::RZ], Entanglement::Circular);
        assert_eq!(c.num_parameters(), 4 * 2 * 3);
        assert_eq!(c.multi_qubit_gate_count(), 2 * 4);
        assert_eq!(
            real_amplitudes(3, 1, Entanglement::Full).num_parameters(),
            6
        );
        assert_eq!(Entanglement::Full.pairs(4).len(), 6);
        assert_eq!(Entanglement::Circular.pairs(2), vec![(0, 1)]);
    }
}
