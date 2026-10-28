//! QAOA for MaxCut at depth 10 on 28 qubits, end to end on one machine.
//!
//! QAOA angles of random 3-regular graphs concentrate: the optimum for one
//! graph is close to the optimum for another of a different size. So the
//! angles are optimised layer by layer (INTERP) on a small graph, carried
//! over to the large one, and refined there with L-BFGS and exact
//! gradients in single precision (half the memory of double precision).
//!
//! Run with `cargo run --release --example qaoa_large` (28 qubits: about
//! 4.7 GB and, on a 4-core VM, a minute per gradient). Options:
//! `--nodes N` (28), `--depth P` (10), `--steps K` (L-BFGS iterations, 30),
//! `--from M` (nodes of the small graph, 16), `--double`, `--seed S` (1).

use std::time::Instant;

use metatron_qso::algorithms::qaoa::optimize_layerwise;
use metatron_qso::optimize::Options;
use metatron_qso::prelude::*;
use metatron_qso::quantum::Precision;

/// Peak resident memory of this process in GB (Linux), if known.
fn peak_gb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / (1u64 << 20) as f64)
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().expect("a number"))
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let (nodes, depth, steps, from) = (
        arg("--nodes", 28),
        arg("--depth", 10),
        arg("--steps", 30),
        arg("--from", 16),
    );
    let seed = arg("--seed", 1) as u64;
    let precision = if std::env::args().any(|a| a == "--double") {
        Precision::Double
    } else {
        Precision::Single
    };

    // 1. Angles, layer by layer on a small random 3-regular graph.
    let start = Instant::now();
    let small = generators::random_regular(from, 3, &mut Rng::seed_from_u64(seed))?;
    let config = QaoaConfig {
        restarts: 2,
        ..Default::default()
    };
    let layers = optimize_layerwise(&small, depth, &config)?;
    let best = layers.last().expect("depth ≥ 1");
    let angles: Vec<f64> = best.gammas.iter().chain(&best.betas).copied().collect();
    println!(
        "{from} nodes, p = 1…{depth} layer by layer: ratio {:.5} at p = {depth} ({:.1} s)",
        best.approximation_ratio.unwrap_or(f64::NAN),
        start.elapsed().as_secs_f64()
    );

    // 2. The large graph: the cut of every one of its 2ⁿ strings, and
    //    the maximum cut by enumeration.
    let start = Instant::now();
    let graph = generators::random_regular(nodes, 3, &mut Rng::seed_from_u64(seed))?;
    let qaoa = MaxCutQaoa::new(&graph, depth)?.with_precision(precision);
    println!(
        "{nodes} nodes, {} edges: maximum cut {} ({:.1} s)",
        graph.num_edges(),
        qaoa.max_cut(),
        start.elapsed().as_secs_f64()
    );

    // 3. The carried-over angles, then L-BFGS from them.
    let start = Instant::now();
    let carried = qaoa.analyze(&angles)?;
    println!(
        "carried over: ratio {:.5} ({:.1} s)",
        carried.approximation_ratio.unwrap_or(f64::NAN),
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let refine = QaoaConfig {
        restarts: 1,
        initial: Some(angles),
        options: Options {
            max_iterations: steps,
            ..Default::default()
        },
        ..Default::default()
    };
    let r = qaoa.optimize(&refine)?;
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "refined ({} precision): ratio {:.5}, ⟨C⟩ = {:.4} of {}, P(optimal cut) {:.2e}, \
         {} iterations, {} evaluations, {:.0} s ({:.1} s per evaluation), converged: {}",
        precision.name(),
        r.approximation_ratio.unwrap_or(f64::NAN),
        r.expectation,
        r.max_cut,
        r.optimal_probability,
        r.iterations,
        r.evaluations,
        seconds,
        seconds / r.evaluations.max(1) as f64,
        r.converged
    );
    if let Some(gb) = peak_gb() {
        println!("peak memory {gb:.2} GB");
    }
    Ok(())
}
