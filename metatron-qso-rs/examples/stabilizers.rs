//! Stabilizer simulation: a GHZ state on 10 000 qubits, graph states of
//! Metatron's Cube and its relatives, and entanglement from cut-ranks.
//!
//!     cargo run --release --example stabilizers

use std::time::Instant;

use metatron_qso::Result;
use metatron_qso::graph::generators;
use metatron_qso::rng::Rng;
use metatron_qso::stabilizer::{StabilizerState, local_complementation};

fn main() -> Result<()> {
    // ANCHOR: ghz
    // GHZ state on 10 000 qubits: H and a chain of CNOTs.
    let n = 10_000;
    let start = Instant::now();
    let mut ghz = StabilizerState::zero(n)?;
    ghz.h(0)?;
    for q in 1..n {
        ghz.cx(q - 1, q)?;
    }
    let mut rng = Rng::seed_from_u64(1);
    let first = ghz.measure(0, &mut rng)?;
    let last = ghz.measure(n - 1, &mut rng)?;
    assert_eq!(first, last);
    println!(
        "GHZ on {n} qubits: built and measured at both ends in {:.2} s; outcomes {} {}",
        start.elapsed().as_secs_f64(),
        u8::from(first),
        u8::from(last)
    );
    // ANCHOR_END: ghz

    // ANCHOR: graph_states
    // Graph states: entanglement across a cut equals the rank over GF(2)
    // of the adjacency matrix between the two sides.
    println!("\nentanglement of graph states (ebits):");
    for (spec, part) in [
        ("metatron", vec![0]),
        ("fruit-of-life", vec![0]),
        ("fruit-of-life", (1..7).collect()),
        ("cycle:1000", (0..500).collect()),
        ("hypercube:10", (0..512).collect()),
    ] {
        let g = generators::from_spec(spec)?;
        let s = StabilizerState::graph_state(&g)?;
        println!(
            "  {:<28} {:>4} of {:>4} qubits  {:>4}",
            g.name(),
            part.len(),
            g.num_nodes(),
            s.entanglement_entropy(&part)?
        );
    }
    // ANCHOR_END: graph_states

    // ANCHOR: local_complement
    // Metatron's Cube is K13. Local complementation at any node removes
    // every edge between the others: the star, whose graph state is a GHZ
    // state up to Hadamards on the leaves.
    let k13 = generators::from_spec("metatron")?;
    let star = local_complementation(&k13, 0)?;
    let mut state = StabilizerState::graph_state(&k13)?;
    state.local_complement(&k13, 0)?;
    assert!(state.same_state(&StabilizerState::graph_state(&star)?));
    println!(
        "\nMetatron's Cube * 0: {} edges, all at node 0 (degree {})",
        star.num_edges(),
        star.degree(0)
    );
    // ANCHOR_END: local_complement
    Ok(())
}
