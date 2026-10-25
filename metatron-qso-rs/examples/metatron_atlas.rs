//! An atlas of the Metatron family: the 13-node graphs formed by the lines
//! of Metatron's Cube up to a given length, from the Fruit-of-Life contact
//! graph (length 1) to the complete graph K₁₃ (all 78 lines), analysed
//! with the 1.3 tools.
//!
//! For each graph: its symmetry, how far a walk from the centre reduces
//! (equitable partition), whether the Laplacian spectrum is integral,
//! state transfer from the centre to an outer node, and three ways to
//! search for an outer node — continuous-time (Childs–Goldstone), coined
//! (Shenvi–Kempe–Whaley) and Szegedy's hitting time — next to the
//! classical hitting time.
//!
//! Run with `cargo run --release --example metatron_atlas`.

use metatron_qso::graph::metatron;
use metatron_qso::prelude::*;
use metatron_qso::walks::{
    Coin, CoinedWalk, Quotient, StateTransfer, SzegedyWalk, classical_hitting_times,
};

fn main() -> Result<()> {
    let lengths: Vec<f64> = metatron::line_lengths().iter().map(|&(l, _)| l).collect();
    println!(
        "State transfer: adjacency generator, γ = 1, centre → outer node 7. Searches mark node 7:\n\
         continuous time with the Laplacian (rate and time optimised, t ≤ π√13), coined with the\n\
         Grover coin (≤ 12 steps); hitting times count steps of the walks.\n"
    );
    println!(
        "| max length | edges | degrees | |Aut| | cells from centre | integral spectrum | PST centre→outer | best transfer (t ≤ 20) | CTQW search | coined search | Szegedy / classical hitting |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for &length in &lengths {
        let g = metatron::distance_graph(length + 1e-9);
        let degrees = g.degrees();
        let (dmin, dmax) = (degrees.iter().min().unwrap(), degrees.iter().max().unwrap());
        let order = g.automorphism_group().order().expect("small group");
        let cells = Quotient::with_singletons(&g, &[0])?.num_cells();
        let integral = g
            .laplacian_spectrum()
            .iter()
            .all(|l| (l - l.round()).abs() < 1e-9);
        // Node 7 is an outer circle of the Fruit of Life.
        let transfer = StateTransfer::new(&g, Generator::Adjacency, 1.0, 0, 7)?;
        let pst = transfer
            .perfect_transfer_time()
            .map_or("none".to_string(), |t| format!("t = {t:.4}"));
        let best = transfer.best_transfer(20.0, 4000)?;
        let search = SpatialSearch::optimize(
            &g,
            &[7],
            Generator::Laplacian,
            std::f64::consts::PI * 13f64.sqrt(),
            1000,
        )?;
        let coined = CoinedWalk::new(&g, Coin::Grover)?
            .with_marked(&[7])?
            .search(12)?;
        let quantum_hitting = SzegedyWalk::new(&g, &[7])?.hitting_time(10_000);
        let h = classical_hitting_times(&g, 7)?;
        let total: f64 = degrees.iter().sum::<usize>() as f64;
        let classical: f64 = (0..13).map(|u| degrees[u] as f64 / total * h[u]).sum();
        println!(
            "| {length:.4} | {} | {dmin}–{dmax} | {order} | {cells} | {} | {pst} | {:.4} at {:.2} | {:.4} at {:.2} | {:.4} at {} | {} / {:.2} |",
            g.num_edges(),
            if integral { "yes" } else { "no" },
            best.fidelity,
            best.time,
            search.success_probability,
            search.time,
            coined.success_probability,
            coined.steps,
            quantum_hitting.map_or("–".to_string(), |t| t.to_string()),
            classical,
        );
    }
    println!(
        "\nA walk from the centre reduces to three cells (centre, inner ring, outer ring) on every\n\
         graph of the family except K13, where the twelve other nodes form a single cell."
    );
    Ok(())
}
