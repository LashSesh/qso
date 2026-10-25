//! Edge-list and GraphML readers: no input may panic or exhaust memory,
//! and every accepted graph must be well formed and survive a round trip
//! through both writers unchanged. The first byte selects the reader.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::graph::Graph;
use metatron_qso::graph::io::{from_edge_list, from_graphml, to_edge_list, to_graphml};

fn same(a: &Graph, b: &Graph) -> bool {
    a.num_nodes() == b.num_nodes() && a.edges() == b.edges()
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    let Ok(text) = std::str::from_utf8(rest) else {
        return;
    };
    let parsed = if mode % 2 == 0 { from_edge_list(text) } else { from_graphml(text) };
    let Ok(g) = parsed else {
        return;
    };
    let n = g.num_nodes();
    for &(u, v, w) in g.edges() {
        assert!(u < v && v < n && w.is_finite() && w != 0.0, "edge ({u}, {v}, {w})");
    }
    assert_eq!(g.degrees().iter().sum::<usize>(), 2 * g.num_edges());
    let a = from_edge_list(&to_edge_list(&g)).expect("edge list round trip");
    assert!(same(&g, &a), "edge list round trip changed the graph");
    let b = from_graphml(&to_graphml(&g)).expect("GraphML round trip");
    assert!(same(&g, &b), "GraphML round trip changed the graph");
});
