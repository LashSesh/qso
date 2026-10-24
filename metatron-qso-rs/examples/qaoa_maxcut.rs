//! QAOA for MaxCut with layer-wise INTERP initialisation.
//!
//! Run with `cargo run --release --example qaoa_maxcut`.

use metatron_qso::algorithms::qaoa::optimize_layerwise;
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    let mut rng = Rng::seed_from_u64(2026);
    let graphs = [
        generators::cycle(10),
        generators::hypercube(3),
        generators::petersen(),
        generators::random_regular(14, 3, &mut rng)?,
        metatron::fruit_of_life(),
    ];
    let config = QaoaConfig {
        restarts: 6,
        ..Default::default()
    };
    for g in &graphs {
        println!(
            "{} ({} nodes, {} edges)",
            g.name(),
            g.num_nodes(),
            g.num_edges()
        );
        for r in optimize_layerwise(g, 4, &config)? {
            println!(
                "  p={}  ratio {:.6}  P(optimal cut) {:.4}  most likely {} (cut {} of {})",
                r.p,
                r.approximation_ratio.unwrap_or(f64::NAN),
                r.optimal_probability,
                bitstring(r.most_likely, g.num_nodes()),
                r.most_likely_cut,
                r.max_cut
            );
        }
    }
    Ok(())
}
