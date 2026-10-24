//! Quantum walks on the Metatron family of graphs.
//!
//! Run with `cargo run --release --example metatron_walks`.

use metatron_qso::graph::metatron;
use metatron_qso::prelude::*;
use metatron_qso::walks::{classical_hitting_times, total_variation};

fn main() -> Result<()> {
    let cube = metatron::metatron_cube();
    let fol = metatron::fruit_of_life();

    println!("== Geometry ==");
    for (graph, label) in [(&cube, "Metatron's Cube"), (&fol, "Fruit of Life")] {
        let group = graph.automorphism_group();
        println!(
            "{label:16} nodes {:2}  edges {:2}  |Aut| = {}  orbits {:?}",
            graph.num_nodes(),
            graph.num_edges(),
            group.order().expect("fits in u128"),
            group.orbits().iter().map(Vec::len).collect::<Vec<_>>()
        );
    }
    println!("line lengths of Metatron's Cube (length × count):");
    for (length, count) in metatron::line_lengths() {
        println!("  {length:.4} × {count}");
    }

    println!("\n== Coherent walk from the centre ==");
    for (graph, label) in [(&cube, "Metatron's Cube"), (&fol, "Fruit of Life")] {
        let walk = ContinuousWalk::new(graph, Generator::Laplacian, 1.0)?;
        let limit = walk.limiting_distribution(0)?;
        let mix = walk.mixing_time(0, 0.05, 200.0, 2000)?;
        println!(
            "{label:16} P̄(centre) = {:.6}  P̄(inner) = {:.6}  P̄(outer) = {:.6}  mixing time (ε=0.05) = {:?}",
            limit[0], limit[1], limit[7], mix
        );
    }
    println!(
        "(K13 closed form: P̄(centre) = 145/169 = {:.6})",
        145.0 / 169.0
    );

    println!("\n== Dephasing turns the walk classical ==");
    let uniform = vec![1.0 / 13.0; 13];
    for rate in [0.0, 0.5, 5.0] {
        let open = DephasingWalk::new(&fol, Generator::Laplacian, 1.0, rate)?;
        let p = open.populations(0, 20.0)?;
        println!(
            "dephasing rate {rate:>4}: TV distance to uniform at t = 20: {:.4}",
            total_variation(&p, &uniform)
        );
    }

    println!("\n== Spatial search for an outer node ==");
    for (graph, label) in [(&cube, "Metatron's Cube"), (&fol, "Fruit of Life")] {
        let gamma_c = SpatialSearch::critical_gamma(graph)?;
        let t_max = std::f64::consts::PI * 13f64.sqrt();
        let r = SpatialSearch::optimize(graph, &[9], Generator::Laplacian, t_max, 2000)?;
        let classical = classical_hitting_times(graph, 9)?;
        let mean_classical = classical.iter().sum::<f64>() / 12.0;
        println!(
            "{label:16} γ_c = {gamma_c:.4}  γ* = {:.4}  t* = {:.3}  P(success) = {:.4}   classical mean hitting time {:.2} steps",
            r.gamma, r.time, r.success_probability, mean_classical
        );
    }
    Ok(())
}
