use std::f64::consts::{FRAC_PI_2, PI};

use super::program::{is_clifford_circuit, is_clifford_program, run_program, sample_program};
use super::*;
use crate::graph::generators;
use crate::quantum::{Circuit, PauliSum, Program};

/// A random Clifford circuit drawn from every supported gate form.
fn random_clifford(n: usize, gates: usize, rng: &mut Rng) -> Circuit {
    let mut c = Circuit::new(n);
    let q = |rng: &mut Rng| rng.below(n as u64) as usize;
    let pair = |rng: &mut Rng| {
        let a = rng.below(n as u64) as usize;
        let mut b = rng.below(n as u64 - 1) as usize;
        if b >= a {
            b += 1;
        }
        (a, b)
    };
    let angle = |rng: &mut Rng| (rng.below(8) as f64 - 4.0) * FRAC_PI_2;
    // One-qubit circuits draw only single-qubit gates.
    let kinds = if n == 1 { 13 } else { 20 };
    for _ in 0..gates {
        match rng.below(kinds) {
            0 => {
                c.h(q(rng));
            }
            1 => {
                c.s(q(rng));
            }
            2 => {
                c.sdg(q(rng));
            }
            3 => {
                c.x(q(rng));
            }
            4 => {
                c.y(q(rng));
            }
            5 => {
                c.z(q(rng));
            }
            6 => {
                c.sx(q(rng));
            }
            7 => {
                c.sxdg(q(rng));
            }
            8 => {
                c.rx(q(rng), angle(rng));
            }
            9 => {
                c.ry(q(rng), angle(rng));
            }
            10 => {
                c.rz(q(rng), angle(rng));
            }
            11 => {
                c.p(q(rng), angle(rng));
            }
            12 => {
                c.u(q(rng), angle(rng), angle(rng), angle(rng));
            }
            13 => {
                let (a, b) = pair(rng);
                c.cx(a, b);
            }
            14 => {
                let (a, b) = pair(rng);
                c.cy(a, b);
            }
            15 => {
                let (a, b) = pair(rng);
                c.cz(a, b);
            }
            16 => {
                let (a, b) = pair(rng);
                c.swap(a, b);
            }
            17 => {
                let (a, b) = pair(rng);
                c.rxx(a, b, angle(rng));
            }
            18 => {
                let (a, b) = pair(rng);
                c.ryy(a, b, angle(rng));
            }
            _ => {
                let (a, b) = pair(rng);
                if rng.below(2) == 0 {
                    c.rzz(a, b, angle(rng));
                } else {
                    c.cp(a, b, PI);
                }
            }
        }
    }
    c
}

fn random_pauli(n: usize, rng: &mut Rng) -> Vec<(usize, Pauli)> {
    (0..n)
        .filter_map(|q| match rng.below(4) {
            1 => Some((q, Pauli::X)),
            2 => Some((q, Pauli::Y)),
            3 => Some((q, Pauli::Z)),
            _ => None,
        })
        .collect()
}

fn pauli_sum(ops: &[(usize, Pauli)]) -> PauliSum {
    let text: Vec<String> = ops.iter().map(|(q, p)| format!("{p:?}{q}")).collect();
    if text.is_empty() {
        "1".parse().unwrap()
    } else {
        text.join(" ").parse().unwrap()
    }
}

#[test]
fn random_clifford_circuits_match_the_state_vector() {
    let mut rng = Rng::seed_from_u64(1);
    for trial in 0..200 {
        let n = 1 + (trial % 8);
        let c = random_clifford(n, 4 * n + 10, &mut rng);
        assert!(is_clifford_circuit(&c, &[]));
        let tab = StabilizerState::from_circuit(&c, &[]).unwrap();
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let f = tab.to_statevector().unwrap().fidelity(&psi).unwrap();
        assert!((f - 1.0).abs() < 1e-10, "trial {trial}: fidelity {f}");
        for _ in 0..10 {
            let p = random_pauli(n, &mut rng);
            let exact = pauli_sum(&p).expectation(&psi).unwrap();
            let e = tab.expectation(&p).unwrap();
            assert!(
                (e - exact).abs() < 1e-10,
                "trial {trial}: {p:?} {e} vs {exact}"
            );
        }
    }
}

#[test]
fn sampling_is_uniform_on_the_support() {
    let mut rng = Rng::seed_from_u64(2);
    for trial in 0..60 {
        let n = 2 + (trial % 7);
        let c = random_clifford(n, 5 * n, &mut rng);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let probs = psi.probabilities();
        let support: Vec<usize> = (0..probs.len()).filter(|&b| probs[b] > 1e-12).collect();
        let sampler = StabilizerState::from_circuit(&c, &[]).unwrap().sampler();
        assert_eq!(1usize << sampler.random_bits(), support.len());
        for &b in &support {
            assert!((probs[b] - 1.0 / support.len() as f64).abs() < 1e-10);
        }
        for _ in 0..50 {
            let v = sampler.sample_words(&mut rng)[0] as usize;
            assert!(probs[v] > 1e-12, "sampled {v:b} outside the support");
        }
    }
}

#[test]
fn measurements_collapse_like_the_state_vector() {
    let mut rng = Rng::seed_from_u64(3);
    for trial in 0..100 {
        let n = 2 + (trial % 6);
        let c = random_clifford(n, 5 * n, &mut rng);
        let mut tab = StabilizerState::from_circuit(&c, &[]).unwrap();
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        // Measure a random Pauli; project the state vector onto the
        // observed eigenspace and compare.
        let p = random_pauli(n, &mut rng);
        let expected = tab.expectation(&p).unwrap();
        let outcome = tab.measure_pauli(&p, &mut rng).unwrap();
        if expected != 0.0 {
            assert_eq!(outcome, expected < 0.0, "deterministic outcome");
        }
        let op = pauli_sum(&p);
        let sign = if outcome { -1.0 } else { 1.0 };
        // (1 + sign·P)/2 |ψ⟩ via P|ψ⟩ as a matrix-free product.
        let ppsi = op.apply(&psi).unwrap();
        let amps: Vec<C64> = psi
            .amplitudes()
            .iter()
            .zip(&ppsi)
            .map(|(a, b)| (a + b * sign) * 0.5)
            .collect();
        let projected = StateVector::from_amplitudes(amps).unwrap();
        let f = tab.to_statevector().unwrap().fidelity(&projected).unwrap();
        assert!((f - 1.0).abs() < 1e-10, "trial {trial}: fidelity {f}");
        // Measuring again repeats the outcome.
        assert_eq!(tab.measure_pauli(&p, &mut rng).unwrap(), outcome);
    }
}

#[test]
fn ghz_state_on_two_thousand_qubits() {
    let n = 2000;
    let mut s = StabilizerState::zero(n).unwrap();
    s.h(0).unwrap();
    for q in 1..n {
        s.cx(q - 1, q).unwrap();
    }
    assert_eq!(s.expectation_str("Z0 Z1999").unwrap(), 1.0);
    assert_eq!(s.expectation_str("-Z17 Z1234").unwrap(), -1.0);
    assert_eq!(s.expectation_str("Z5").unwrap(), 0.0);
    let all_x: Vec<(usize, Pauli)> = (0..n).map(|q| (q, Pauli::X)).collect();
    assert_eq!(s.expectation(&all_x).unwrap(), 1.0);
    let mut rng = Rng::seed_from_u64(4);
    let counts = s.sample_counts(200, &mut rng);
    assert_eq!(counts.len(), 2);
    assert!(counts.contains_key(&"0".repeat(n)) && counts.contains_key(&"1".repeat(n)));
    // Measuring one qubit fixes all others.
    let v = s.measure(1000, &mut rng).unwrap();
    for q in [0, 999, 1999] {
        assert_eq!(s.measure(q, &mut rng).unwrap(), v);
    }
    assert_eq!(s.entanglement_entropy(&[0, 1, 2]).unwrap(), 0);
}

#[test]
fn graph_states() {
    let mut rng = Rng::seed_from_u64(5);
    for spec in [
        "petersen",
        "cycle:7",
        "fruit-of-life",
        "metatron",
        "path:5",
        "star:6",
    ] {
        let g = generators::from_spec(spec).unwrap();
        let n = g.num_nodes();
        let s = StabilizerState::graph_state(&g).unwrap();
        // The same state as H on every qubit followed by CZ on every edge.
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.h(q);
        }
        for &(u, v, _) in g.edges() {
            c.cz(u, v);
        }
        let t = StabilizerState::from_circuit(&c, &[]).unwrap();
        assert!(s.same_state(&t) && t.same_state(&s), "{spec}");
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let f = s.to_statevector().unwrap().fidelity(&psi).unwrap();
        assert!((f - 1.0).abs() < 1e-10, "{spec}");
        // Entanglement of random bipartitions: the state-vector entropy and
        // the cut-rank of the adjacency matrix over GF(2).
        for _ in 0..8 {
            let part: Vec<usize> = (0..n).filter(|_| rng.below(2) == 1).collect();
            let e = s.entanglement_entropy(&part).unwrap();
            let rest: Vec<usize> = (0..n).filter(|q| !part.contains(q)).collect();
            // A pure state has the same entropy on both sides; the smaller
            // side has the smaller reduced density matrix.
            let small = if part.len() <= rest.len() {
                &part
            } else {
                &rest
            };
            let exact = psi.entanglement_entropy(small).unwrap();
            assert!(
                (e as f64 - exact).abs() < 1e-8,
                "{spec} {part:?}: {e} vs {exact}"
            );
            let mut rows: Vec<Vec<u64>> = part
                .iter()
                .map(|&a| {
                    let mut v = vec![0u64];
                    for (j, &b) in rest.iter().enumerate() {
                        if g.has_edge(a, b) {
                            v[0] |= 1 << j;
                        }
                    }
                    v
                })
                .collect();
            assert_eq!(gf2_rank(&mut rows, rest.len()), e, "{spec} cut-rank");
        }
        // Local complementation is the local Clifford of Hein et al.
        for v in [0, n / 2, n - 1] {
            let mut lc = s.clone();
            lc.local_complement(&g, v).unwrap();
            let h = local_complementation(&g, v).unwrap();
            assert!(
                lc.same_state(&StabilizerState::graph_state(&h).unwrap()),
                "{spec} * {v}"
            );
        }
    }
    // Metatron's Cube is K13: local complementation at a node leaves the
    // star, so its graph state is a GHZ state up to local Cliffords.
    let k13 = generators::from_spec("metatron").unwrap();
    let star = local_complementation(&k13, 0).unwrap();
    assert_eq!(star.num_edges(), 12);
    assert_eq!(star.degree(0), 12);
}

#[test]
fn clifford_programs() {
    // Teleport |+i⟩ = S H|0⟩ with feed-forward.
    let src = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; bit[3] c;
        h q[0]; s q[0];
        h q[1]; cx q[1], q[2]; cx q[0], q[1]; h q[0];
        c[0] = measure q[0]; c[1] = measure q[1];
        if (c[1] == 1) x q[2]; if (c[0] == 1) z q[2];
        sdg q[2]; h q[2]; c[2] = measure q[2];";
    let p = Program::from_qasm(src).unwrap();
    assert!(is_clifford_program(&p));
    let exact = p.probabilities().unwrap();
    let shots = 4000;
    let counts = sample_program(&p, shots, 7).unwrap();
    for (record, &n) in &counts {
        // Bit 2 always reads 0: the teleported state is undone exactly.
        assert_eq!(record >> 2, 0);
        let pr = exact[record];
        let sd = (shots as f64 * pr * (1.0 - pr)).sqrt();
        assert!(
            (n as f64 - shots as f64 * pr).abs() < 5.0 * sd + 1.0,
            "{record}: {n}"
        );
    }
    // A GHZ program on 1000 qubits, measured at both ends.
    let mut src =
        String::from("OPENQASM 3.0; include \"stdgates.inc\"; qubit[1000] q; bit[2] c; h q[0];");
    for k in 1..1000 {
        src.push_str(&format!(" cx q[{}], q[{k}];", k - 1));
    }
    src.push_str(" c[0] = measure q[0]; c[1] = measure q[999];");
    let ghz = Program::from_qasm(&src).unwrap();
    let counts = sample_program(&ghz, 500, 1).unwrap();
    assert_eq!(counts.keys().copied().collect::<Vec<_>>(), vec![0b00, 0b11]);
    let mut rng = Rng::seed_from_u64(2);
    assert!(matches!(run_program(&ghz, &mut rng).unwrap(), 0b00 | 0b11));
    // A program without qubits writes no bits.
    let empty = Program::from_qasm("bit[2] c;").unwrap();
    assert_eq!(sample_program(&empty, 3, 0).unwrap().into_iter().collect::<Vec<_>>(), vec![(0, 3)]);
    // Non-Clifford programs are rejected.
    let t = Program::from_qasm("qubit q; bit c; t q; c = measure q;").unwrap();
    assert!(!is_clifford_program(&t));
    assert!(sample_program(&t, 10, 0).is_err());
}

#[test]
fn invalid_input_is_an_error() {
    assert!(StabilizerState::zero(0).is_err());
    assert!(StabilizerState::zero(MAX_STABILIZER_QUBITS + 1).is_err());
    let mut s = StabilizerState::zero(3).unwrap();
    assert!(s.h(3).is_err());
    assert!(s.cx(1, 1).is_err());
    let mut c = Circuit::new(2);
    c.t(0);
    assert!(StabilizerState::from_circuit(&c, &[]).is_err());
    let mut c = Circuit::new(2);
    c.rz(0, 0.3);
    assert!(StabilizerState::from_circuit(&c, &[]).is_err());
    let mut c = Circuit::new(2);
    c.ch(0, 1);
    assert!(StabilizerState::from_circuit(&c, &[]).is_err());
    let mut c = Circuit::new(3);
    c.ccx(0, 1, 2);
    assert!(!is_clifford_circuit(&c, &[]));
    assert!(s.expectation_str("X0 X0").is_err());
    assert!(s.expectation_str("X7").is_err());
    assert!(s.expectation_str("Q1").is_err());
    assert!(s.entanglement_entropy(&[0, 0]).is_err());
    assert!(StabilizerState::zero(21).unwrap().to_statevector().is_err());
    let weighted = Graph::from_weighted_edges(2, &[(0, 1, 0.5)]).unwrap();
    assert!(StabilizerState::graph_state(&weighted).is_err());
    assert!(local_complementation(&generators::from_spec("cycle:4").unwrap(), 4).is_err());
}

#[test]
fn identities_and_signs() {
    let mut s = StabilizerState::zero(2).unwrap();
    assert_eq!(s.stabilizers(), vec!["+Z0", "+Z1"]);
    s.x(0).unwrap();
    assert_eq!(s.expectation_str("Z0").unwrap(), -1.0);
    s.h(1).unwrap().s(1).unwrap(); // |+i⟩
    assert_eq!(s.expectation_str("Y1").unwrap(), 1.0);
    assert_eq!(s.stabilizers(), vec!["-Z0", "+Y1"]);
    let mut rng = Rng::seed_from_u64(0);
    s.reset(0, &mut rng).unwrap();
    assert_eq!(s.expectation_str("Z0").unwrap(), 1.0);
    // ⟨H⟩ of a sum.
    let h: PauliSum = "0.5*Z0 + 2*Y1 - X1".parse().unwrap();
    assert!((s.expectation_sum(&h).unwrap() - 2.5).abs() < 1e-12);
    assert!(s.measure_pauli(&[], &mut rng).is_ok_and(|v| !v));
}

#[test]
fn transposition_round_trips() {
    let mut rng = Rng::seed_from_u64(9);
    let mut a = [0u64; 64];
    for v in a.iter_mut() {
        *v = rng.next_u64();
    }
    let mut t = a;
    transpose64(&mut t);
    for (k, row) in a.iter().enumerate() {
        for (j, col) in t.iter().enumerate() {
            assert_eq!((row >> j) & 1, (col >> k) & 1);
        }
    }
    // Rows and columns of a tableau convert back and forth exactly.
    for n in [1, 63, 64, 65, 130] {
        let c = random_clifford(n, 3 * n, &mut rng);
        let s = StabilizerState::from_circuit(&c, &[]).unwrap();
        let mut t = s.clone();
        t.set_rows(&s.rows());
        assert_eq!(s, t);
    }
}
