//! JSON deserialisation of the public data types: no input may panic, and
//! every accepted value must be usable (simulated, summarised, printed).
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::graph::Graph;
use metatron_qso::quantum::{Circuit, PauliSum, StateVector};

fuzz_target!(|data: &[u8]| {
    if let Ok(c) = serde_json::from_slice::<Circuit>(data) {
        let text = serde_json::to_string(&c).expect("circuits serialise");
        assert_eq!(serde_json::from_str::<Circuit>(&text).expect("round trip"), c);
        if c.n_qubits() <= 10 && c.num_parameters() <= 1000 {
            let theta = vec![0.3; c.num_parameters()];
            let psi = StateVector::from_circuit(&c, &theta).expect("valid circuits simulate");
            assert!((psi.norm() - 1.0).abs() < 1e-9, "norm {}", psi.norm());
        }
    }
    if let Ok(s) = serde_json::from_slice::<StateVector>(data) {
        assert_eq!(s.probabilities().len(), 1 << s.n_qubits());
    }
    if let Ok(g) = serde_json::from_slice::<Graph>(data) {
        let n = g.num_nodes();
        for &(u, v, w) in g.edges() {
            assert!(u < v && v < n && w.is_finite() && w != 0.0, "edge ({u}, {v}, {w})");
        }
        assert_eq!(g.degrees().iter().sum::<usize>(), 2 * g.num_edges());
        let _ = g.summary();
    }
    if let Ok(p) = serde_json::from_slice::<PauliSum>(data) {
        let _ = p.to_string();
        if p.min_qubits() <= 10 {
            let psi = StateVector::zero(p.min_qubits());
            let _ = p.expectation(&psi);
        }
    }
});
