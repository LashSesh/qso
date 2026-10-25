//! Continuous-time quantum walks as gate circuits.
//!
//! Node `x` of the graph is the basis state `|x⟩` of `⌈log₂ N⌉` qubits
//! (little-endian, as everywhere in the crate); padding states beyond the
//! last node are left alone. Each builder returns a [`Circuit`] whose
//! unitary is `exp(−iHt)` for `H = γL` or `H = −γA`:
//!
//! * [`circulant_walk`] — exactly, for circulant graphs on `2ⁿ` nodes:
//!   the Fourier transform diagonalises `H`, so `U = QFT·D·QFT†` with a
//!   diagonal `D` built from `CX` parity chains and `RZ` rotations.
//! * [`cayley_walk`] — exactly, for Cayley graphs of `Z₂ⁿ` (hypercubes and
//!   their relatives): `H` is a sum of commuting `X`-strings.
//! * [`trotter_walk`] — approximately, for any graph: a product formula
//!   over the edges (first or second order), each edge a rotation in the
//!   two-dimensional space of its end nodes. The error falls as
//!   `steps⁻¹` or `steps⁻²`.
//! * [`diagonal_unitary`] — any diagonal unitary `Σ_y e^{iφ(y)}|y⟩⟨y|`,
//!   exactly (global phase included), by the Walsh decomposition of `φ`.
//!
//! The circuits export to OpenQASM 2 ([`to_qasm2`](crate::quantum::to_qasm2)),
//! and are limited like OpenQASM programs to
//! [`MAX_QASM_INSTRUCTIONS`] gates; graphs may have at most 2²⁰ nodes.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::quantum::StateVector;
//! use metatron_qso::walks::{circuits, ContinuousWalk, Generator};
//!
//! let g = generators::cycle(8);
//! let c = circuits::circulant_walk(&g, Generator::Laplacian, 1.0, 0.7).unwrap();
//! let state = StateVector::from_circuit(&c, &[]).unwrap();
//! let exact = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap().probabilities(0, 0.7).unwrap();
//! for (p, q) in state.probabilities().iter().zip(&exact) {
//!     assert!((p - q).abs() < 1e-10);
//! }
//! ```

use std::f64::consts::PI;

use num_complex::Complex64 as C64;

use super::Generator;
use crate::algorithms::qft::{inverse_qft, qft};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::quantum::Circuit;
use crate::quantum::qasm::MAX_QASM_INSTRUCTIONS;

/// Qubits needed for `n` nodes (at least one).
fn qubits_for(n: usize) -> usize {
    (n.max(2) - 1).ilog2() as usize + 1
}

fn check_inputs(graph: &Graph, gamma: f64, t: f64) -> Result<()> {
    if graph.num_nodes() < 2 {
        return Err(QsoError::invalid("graph", "need at least two nodes"));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(QsoError::invalid(
            "gamma",
            "hopping rate must be positive and finite",
        ));
    }
    if !t.is_finite() {
        return Err(QsoError::invalid("t", "evolution time must be finite"));
    }
    if graph.num_nodes() > 1 << 20 {
        return Err(QsoError::invalid("graph", "at most 2^20 nodes"));
    }
    Ok(())
}

/// Reject circuits with more gates than an OpenQASM program may have.
fn check_gate_count(gates: u128) -> Result<()> {
    if gates > MAX_QASM_INSTRUCTIONS as u128 {
        return Err(QsoError::invalid(
            "circuit",
            format!("the walk circuit would have {gates} gates, more than {MAX_QASM_INSTRUCTIONS}"),
        ));
    }
    Ok(())
}

/// A global phase `e^{iα}` as gates: `X·P(α)·X·P(α) = e^{iα}·I`.
fn global_phase(c: &mut Circuit, alpha: f64) {
    if alpha != 0.0 {
        c.p(0, alpha).x(0).p(0, alpha).x(0);
    }
}

/// `exp(iθ·Z_S)` for the qubits in `qubits`, by a `CX` parity chain into
/// the last qubit and `RZ(−2θ)`.
fn z_string_rotation(c: &mut Circuit, qubits: &[usize], theta: f64) {
    let (&last, rest) = qubits.split_last().expect("non-empty");
    for &q in rest {
        c.cx(q, last);
    }
    c.rz(last, -2.0 * theta);
    for &q in rest.iter().rev() {
        c.cx(q, last);
    }
}

/// Exact circuit for the diagonal unitary `Σ_y e^{i·phases[y]} |y⟩⟨y|` on
/// `log₂(phases.len())` qubits (a power of two, at least 2).
///
/// With the Walsh coefficients `a_S = 2⁻ⁿ Σ_y φ(y)(−1)^{y·S}`,
/// `e^{iφ} = Π_S exp(i a_S Z_S)`: one `RZ` and `2(|S| − 1)` `CX` per
/// non-zero coefficient, so at most `2ⁿ` rotations; the global phase
/// `a_∅` is kept.
pub fn diagonal_unitary(phases: &[f64]) -> Result<Circuit> {
    let len = phases.len();
    if len < 2 || !len.is_power_of_two() {
        return Err(QsoError::NotPowerOfTwo(len));
    }
    if phases.iter().any(|p| !p.is_finite()) {
        return Err(QsoError::invalid("phases", "phases must be finite"));
    }
    let n = len.trailing_zeros() as usize;
    // Fast Walsh–Hadamard transform.
    let mut a = phases.to_vec();
    let mut h = 1;
    while h < len {
        for block in (0..len).step_by(2 * h) {
            for i in block..block + h {
                let (x, y) = (a[i], a[i + h]);
                a[i] = x + y;
                a[i + h] = x - y;
            }
        }
        h *= 2;
    }
    a.iter_mut().for_each(|x| *x /= len as f64);
    let scale = phases.iter().fold(1.0f64, |m, p| m.max(p.abs()));
    let kept = |s: usize| a[s].abs() > 1e-15 * scale;
    let gates: u128 = 4
        + (1..len)
            .filter(|&s| kept(s))
            .map(|s| 2 * s.count_ones() as u128 - 1)
            .sum::<u128>();
    check_gate_count(gates)?;
    let mut c = Circuit::new(n);
    global_phase(&mut c, a[0]);
    for (s, &coeff) in a.iter().enumerate().skip(1) {
        if kept(s) {
            let qubits: Vec<usize> = (0..n).filter(|&q| s >> q & 1 == 1).collect();
            z_string_rotation(&mut c, &qubits, coeff);
        }
    }
    Ok(c)
}

/// The graph's jumps `(s, weight)` (`1 ≤ s ≤ N/2`) if it is circulant on a
/// power-of-two number of nodes.
fn circulant_jumps(graph: &Graph) -> Result<Vec<(usize, f64)>> {
    let n = graph.num_nodes();
    if !n.is_power_of_two() {
        return Err(QsoError::invalid(
            "graph",
            format!("circulant walk circuits need 2^k nodes, got {n}"),
        ));
    }
    let mut jumps = vec![0.0; n / 2 + 1];
    for &(v, w) in graph.neighbors(0) {
        jumps[v.min(n - v)] = w;
    }
    let not_circulant = || QsoError::invalid("graph", "graph is not circulant");
    let expected = graph.degree(0);
    for u in 0..n {
        if graph.degree(u) != expected {
            return Err(not_circulant());
        }
        for &(v, w) in graph.neighbors(u) {
            let d = (v + n - u) % n;
            if jumps[d.min(n - d)] != w {
                return Err(not_circulant());
            }
        }
    }
    Ok((1..=n / 2)
        .filter(|&s| jumps[s] != 0.0)
        .map(|s| (s, jumps[s]))
        .collect())
}

/// Exact walk circuit for a circulant graph on `2ⁿ` nodes (see the module
/// documentation): an inverse QFT, the diagonal of eigenphases
/// `e^{−iE_y t}`, and a QFT.
pub fn circulant_walk(graph: &Graph, generator: Generator, gamma: f64, t: f64) -> Result<Circuit> {
    check_inputs(graph, gamma, t)?;
    let jumps = circulant_jumps(graph)?;
    let n = graph.num_nodes();
    let degree: f64 = jumps
        .iter()
        .map(|&(s, w)| if 2 * s == n { w } else { 2.0 * w })
        .sum();
    let phases: Vec<f64> = (0..n)
        .map(|y| {
            // Adjacency eigenvalue of the Fourier mode y.
            let lambda: f64 = jumps
                .iter()
                .map(|&(s, w)| {
                    let c = (2.0 * PI * ((s * y) % n) as f64 / n as f64).cos();
                    if 2 * s == n { w * c } else { 2.0 * w * c }
                })
                .sum();
            let energy = match generator {
                Generator::Laplacian => gamma * (degree - lambda),
                Generator::Adjacency => -gamma * lambda,
            };
            -energy * t
        })
        .collect();
    let k = qubits_for(n);
    let mut c = inverse_qft(k, true);
    c.append(&diagonal_unitary(&phases)?)?;
    c.append(&qft(k, true))?;
    Ok(c)
}

/// Exact walk circuit for a Cayley graph of `Z₂ⁿ` (node `x` adjacent to
/// `x ⊕ g` for the generators `g`, as built by
/// [`cayley_z2`](crate::graph::generators::cayley_z2)): `H` is a sum of
/// commuting `X`-strings `X^g`, each a parity rotation.
pub fn cayley_walk(graph: &Graph, generator: Generator, gamma: f64, t: f64) -> Result<Circuit> {
    check_inputs(graph, gamma, t)?;
    let n = graph.num_nodes();
    if !n.is_power_of_two() {
        return Err(QsoError::invalid(
            "graph",
            format!("Cayley graphs of Z_2^d have 2^d nodes, got {n}"),
        ));
    }
    let generators: Vec<(usize, f64)> = graph.neighbors(0).to_vec();
    let not_cayley = || QsoError::invalid("graph", "graph is not a Cayley graph of Z_2^d");
    for u in 0..n {
        if graph.degree(u) != generators.len() {
            return Err(not_cayley());
        }
        for &(g, w) in &generators {
            if graph.edge_weight(u, u ^ g) != Some(w) {
                return Err(not_cayley());
            }
        }
    }
    let k = qubits_for(n);
    let mut c = Circuit::new(k);
    let degree: f64 = generators.iter().map(|&(_, w)| w).sum();
    if generator == Generator::Laplacian {
        global_phase(&mut c, -gamma * degree * t);
    }
    // −γA = −γ Σ_g w_g X^g, so U = Π_g exp(iγ w_g t X^g).
    for &(g, w) in &generators {
        let qubits: Vec<usize> = (0..k).filter(|&q| g >> q & 1 == 1).collect();
        let theta = gamma * w * t;
        if let [q] = qubits[..] {
            c.rx(q, -2.0 * theta);
            continue;
        }
        for &q in &qubits {
            c.h(q);
        }
        z_string_rotation(&mut c, &qubits, theta);
        for &q in &qubits {
            c.h(q);
        }
    }
    Ok(c)
}

/// Append `exp(−i h_e τ)` for the edge `{u, v}` of weight `w`: a rotation
/// in span{|u⟩, |v⟩}. The pivot qubit is the lowest bit in which `u` and
/// `v` differ; `CX` gates from it map the pair onto two states differing
/// only there, and a gate on the pivot controlled by all other qubits
/// rotates them.
fn edge_rotation(c: &mut Circuit, k: usize, u: usize, v: usize, matrix: [C64; 4]) -> Result<()> {
    let diff = u ^ v;
    let p = diff.trailing_zeros() as usize;
    let (a, _) = if u >> p & 1 == 0 { (u, v) } else { (v, u) };
    let others: Vec<usize> = (0..k).filter(|&q| q != p && diff >> q & 1 == 1).collect();
    let zeros: Vec<usize> = (0..k).filter(|&q| q != p && a >> q & 1 == 0).collect();
    let controls: Vec<usize> = (0..k).filter(|&q| q != p).collect();
    for &q in &others {
        c.cx(p, q);
    }
    for &q in &zeros {
        c.x(q);
    }
    c.unitary(matrix.to_vec(), &[p], &controls)?;
    for &q in &zeros {
        c.x(q);
    }
    for &q in others.iter().rev() {
        c.cx(p, q);
    }
    Ok(())
}

/// Product-formula (Trotter–Suzuki) walk circuit for any graph: `steps`
/// steps of length `τ = t/steps`, each the product over the edges of
/// `exp(−i h_e τ)` (`order = 1`) or the symmetric product with half steps
/// forward and backward (`order = 2`). Each edge term is exact; the error
/// comes from the edges not commuting and falls as `steps⁻ᵒʳᵈᵉʳ`.
pub fn trotter_walk(
    graph: &Graph,
    generator: Generator,
    gamma: f64,
    t: f64,
    steps: usize,
    order: usize,
) -> Result<Circuit> {
    check_inputs(graph, gamma, t)?;
    if steps == 0 {
        return Err(QsoError::invalid("steps", "need at least one step"));
    }
    if !(order == 1 || order == 2) {
        return Err(QsoError::invalid("order", "order must be 1 or 2"));
    }
    let k = qubits_for(graph.num_nodes());
    // Gates per edge: CX gates and X gates around one multi-controlled gate.
    let per_step: u128 = graph
        .edges()
        .iter()
        .map(|&(u, v, _)| {
            let diff = u ^ v;
            let p = diff.trailing_zeros() as usize;
            let a = if u >> p & 1 == 0 { u } else { v };
            let zeros = (0..k).filter(|&q| q != p && a >> q & 1 == 0).count();
            (2 * (diff.count_ones() as usize - 1) + 2 * zeros + 1) as u128
        })
        .sum();
    check_gate_count(per_step * steps as u128 * order as u128)?;
    let tau = t / steps as f64;
    let term = |w: f64, dt: f64| -> [C64; 4] {
        // Adjacency: exp(iγwτ X); Laplacian: e^{−iγwτ}·exp(iγwτ X).
        let theta = gamma * w * dt;
        let phase = match generator {
            Generator::Laplacian => C64::from_polar(1.0, -theta),
            Generator::Adjacency => C64::new(1.0, 0.0),
        };
        let (cos, isin) = (
            C64::new(theta.cos(), 0.0) * phase,
            C64::new(0.0, theta.sin()) * phase,
        );
        [cos, isin, isin, cos]
    };
    let edges = graph.edges();
    let mut c = Circuit::new(k);
    for _ in 0..steps {
        if order == 1 {
            for &(u, v, w) in edges {
                edge_rotation(&mut c, k, u, v, term(w, tau))?;
            }
        } else {
            for &(u, v, w) in edges.iter().chain(edges.iter().rev()) {
                edge_rotation(&mut c, k, u, v, term(w, tau / 2.0))?;
            }
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{generators, metatron};
    use crate::quantum::{StateVector, from_qasm2, to_qasm2};
    use crate::walks::ContinuousWalk;

    /// Largest deviation `‖U_circuit|x⟩ − exp(−iHt)|x⟩‖` over the nodes.
    fn deviation(graph: &Graph, c: &Circuit, generator: Generator, gamma: f64, t: f64) -> f64 {
        let walk = ContinuousWalk::new(graph, generator, gamma).unwrap();
        let u = walk.propagator(t).unwrap();
        let n = graph.num_nodes();
        (0..n)
            .map(|x| {
                let mut psi = StateVector::basis(c.n_qubits(), x).unwrap();
                psi.apply_circuit(c, &[]).unwrap();
                let amps = psi.amplitudes();
                let inside: f64 = (0..n).map(|y| (amps[y] - u[(y, x)]).norm_sqr()).sum();
                let outside: f64 = amps[n..].iter().map(|a| a.norm_sqr()).sum();
                (inside + outside).sqrt()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn diagonal_unitaries_are_exact_including_phase() {
        let mut rng = crate::rng::Rng::seed_from_u64(2);
        for n in 1..=5 {
            let phases: Vec<f64> = (0..1 << n).map(|_| rng.uniform_range(-4.0, 4.0)).collect();
            let c = diagonal_unitary(&phases).unwrap();
            for (y, &phase) in phases.iter().enumerate() {
                let mut psi = StateVector::basis(n, y).unwrap();
                psi.apply_circuit(&c, &[]).unwrap();
                let expected = C64::from_polar(1.0, phase);
                assert!((psi.amplitudes()[y] - expected).norm() < 1e-12);
            }
        }
        assert!(diagonal_unitary(&[0.0, 1.0, 2.0]).is_err());
    }

    #[test]
    fn circulant_and_cayley_circuits_are_exact() {
        let circulants = [
            generators::cycle(8),
            generators::cycle(16),
            generators::circulant(16, &[1, 3, 8]).unwrap(),
            generators::complete(8),
            Graph::from_weighted_edges(
                4,
                &[
                    (0, 1, 0.5),
                    (1, 2, 0.5),
                    (2, 3, 0.5),
                    (3, 0, 0.5),
                    (0, 2, 2.0),
                    (1, 3, 2.0),
                ],
            )
            .unwrap(),
        ];
        for g in &circulants {
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                let c = circulant_walk(g, generator, 0.9, 1.3).unwrap();
                assert!(
                    deviation(g, &c, generator, 0.9, 1.3) < 1e-10,
                    "{}",
                    g.name()
                );
            }
        }
        let cayleys = [
            generators::hypercube(4),
            generators::cayley_z2(4, &[3, 5, 6, 15]).unwrap(),
            generators::complete(8),
        ];
        for g in &cayleys {
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                let c = cayley_walk(g, generator, 0.7, 2.1).unwrap();
                assert!(
                    deviation(g, &c, generator, 0.7, 2.1) < 1e-10,
                    "{}",
                    g.name()
                );
            }
        }
        // Gate counts: hypercube walks are single-qubit rotations.
        let c = cayley_walk(&generators::hypercube(6), Generator::Adjacency, 1.0, 1.0).unwrap();
        assert_eq!(c.len(), 6);
        assert!(circulant_walk(&generators::cycle(6), Generator::Laplacian, 1.0, 1.0).is_err());
        // Size limits: a 2^16-node cycle is fine, a long product formula is not.
        let big =
            circulant_walk(&generators::cycle(1 << 16), Generator::Laplacian, 1.0, 1.0).unwrap();
        assert!(big.len() <= MAX_QASM_INSTRUCTIONS);
        assert!(
            trotter_walk(
                &generators::hypercube(14),
                Generator::Laplacian,
                1.0,
                1.0,
                18,
                2
            )
            .is_err()
        );
        assert!(circulant_walk(&generators::path(8), Generator::Laplacian, 1.0, 1.0).is_err());
        assert!(cayley_walk(&generators::cycle(8), Generator::Laplacian, 1.0, 1.0).is_err());
    }

    #[test]
    fn product_formulas_converge_at_their_order() {
        let graphs = [
            generators::petersen(),
            metatron::metatron_cube(),
            generators::path(5),
        ];
        for g in &graphs {
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                for order in [1usize, 2] {
                    let err = |steps: usize| {
                        let c = trotter_walk(g, generator, 1.0, 1.0, steps, order).unwrap();
                        deviation(g, &c, generator, 1.0, 1.0)
                    };
                    let (e1, e2) = (err(32), err(64));
                    let ratio = e1 / e2;
                    let expected = (1 << order) as f64;
                    assert!(
                        (ratio / expected - 1.0).abs() < 0.15,
                        "{} order {order}: {e1} {e2}",
                        g.name()
                    );
                }
            }
        }
        // A single edge is exact in one step.
        let k2 = generators::path(2);
        let c = trotter_walk(&k2, Generator::Laplacian, 1.0, 2.0, 1, 1).unwrap();
        assert!(deviation(&k2, &c, Generator::Laplacian, 1.0, 2.0) < 1e-12);
        assert!(trotter_walk(&k2, Generator::Laplacian, 1.0, 2.0, 0, 1).is_err());
        assert!(trotter_walk(&k2, Generator::Laplacian, 1.0, 2.0, 1, 3).is_err());
    }

    #[test]
    fn walk_circuits_export_to_qasm() {
        let g = metatron::metatron_cube();
        let c = trotter_walk(&g, Generator::Laplacian, 1.0, 0.5, 2, 2).unwrap();
        let parsed = from_qasm2(&to_qasm2(&c, &[], false).unwrap()).unwrap();
        let (mut a, mut b) = (
            StateVector::basis(4, 3).unwrap(),
            StateVector::basis(4, 3).unwrap(),
        );
        a.apply_circuit(&c, &[]).unwrap();
        b.apply_circuit(&parsed, &[]).unwrap();
        assert!((a.fidelity(&b).unwrap() - 1.0).abs() < 1e-10);
        let c = circulant_walk(&generators::cycle(16), Generator::Adjacency, 1.0, 0.5).unwrap();
        assert!(to_qasm2(&c, &[], true).is_ok());
    }
}
