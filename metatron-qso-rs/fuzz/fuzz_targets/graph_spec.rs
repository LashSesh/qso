//! Graph specifications: no input may panic or exhaust memory, and every
//! accepted specification must give a well-formed graph.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::graph::generators::from_spec;

fuzz_target!(|spec: &str| {
    let Ok(g) = from_spec(spec) else {
        return;
    };
    let n = g.num_nodes();
    for &(u, v, w) in g.edges() {
        assert!(u < v && v < n && w.is_finite() && w != 0.0, "edge ({u}, {v}, {w})");
    }
    assert_eq!(g.degrees().iter().sum::<usize>(), 2 * g.num_edges());
});
