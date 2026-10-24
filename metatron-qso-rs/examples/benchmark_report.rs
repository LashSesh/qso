//! Prints the benchmark and validation table used in `docs/BENCHMARKS.md`.
//!
//! Run with `cargo run --release --example benchmark_report`. Timings are
//! medians of repeated runs on the current machine.

use std::time::Instant;

use metatron_qso::algorithms::gradients::{GradientMethod, expectation_and_gradient};
use metatron_qso::algorithms::qaoa::optimize_layerwise;
use metatron_qso::prelude::*;

fn median_seconds(repeats: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut t: Vec<f64> = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn fmt_time(s: f64) -> String {
    if s < 1e-3 {
        format!("{:.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{s:.2} s")
    }
}

fn main() -> Result<()> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("Machine: {threads} hardware threads; build: release\n");

    println!("## Statevector throughput (layer = H + RZ on every qubit, CX chain)\n");
    println!("| qubits | gates/layer | time/layer | ns per gate·amplitude |");
    println!("|---:|---:|---:|---:|");
    for n in [10usize, 14, 18, 20, 22, 24] {
        let mut layer = Circuit::new(n);
        for q in 0..n {
            layer.h(q).rz(q, 0.3);
        }
        for q in 0..n - 1 {
            layer.cx(q, q + 1);
        }
        let mut psi = StateVector::zero(n);
        let repeats = match n {
            0..=14 => 101,
            15..=20 => 15,
            _ => 3,
        };
        let t = median_seconds(repeats, || psi.apply_circuit(&layer, &[]).unwrap());
        let per = t / (layer.len() as f64 * (1u64 << n) as f64) * 1e9;
        println!("| {n} | {} | {} | {per:.3} |", layer.len(), fmt_time(t));
    }

    println!("\n## Cost per gate type (22 qubits, 5 passes over all qubits)\n");
    println!("| gate | time per pass | ns per gate·amplitude |");
    println!("|---|---:|---:|");
    let n = 22;
    for kind in ["H", "RY", "RZ", "CX"] {
        let mut c = Circuit::new(n);
        for _ in 0..5 {
            for q in 0..n {
                match kind {
                    "H" => c.h(q),
                    "RY" => c.ry(q, 0.3),
                    "RZ" => c.rz(q, 0.3),
                    _ => c.cx(q, (q + 1) % n),
                };
            }
        }
        let mut psi = StateVector::zero(n);
        let t = median_seconds(5, || psi.apply_circuit(&c, &[]).unwrap());
        let per = t / (c.len() as f64 * (1u64 << n) as f64) * 1e9;
        println!("| {kind} | {} | {per:.3} |", fmt_time(t / 5.0));
    }

    println!("\n## Gradients (TFIM, 10 qubits, EfficientSU2 with 4 layers = 100 parameters)\n");
    let h = models::transverse_field_ising(10, 1.0, 0.7, false);
    let circuit = ansatz::efficient_su2(10, 4, Entanglement::Linear);
    let theta: Vec<f64> = (0..circuit.num_parameters())
        .map(|k| 0.01 * k as f64)
        .collect();
    println!("| method | time per full gradient |");
    println!("|---|---:|");
    for (name, m) in [
        ("adjoint", GradientMethod::Adjoint),
        ("parameter shift", GradientMethod::ParameterShift),
        (
            "finite differences",
            GradientMethod::FiniteDifference { step: 1e-6 },
        ),
    ] {
        let t = median_seconds(21, || {
            expectation_and_gradient(&circuit, &h, &theta, None, m).unwrap();
        });
        println!("| {name} | {} |", fmt_time(t));
    }

    println!("\n## QAOA value + gradient (p = 3, random 3-regular graphs)\n");
    println!("| nodes | time |");
    println!("|---:|---:|");
    for n in [10usize, 14, 18, 20] {
        let g = generators::random_regular(n, 3, &mut Rng::seed_from_u64(1))?;
        let qaoa = MaxCutQaoa::new(&g, 3)?;
        let x = [0.4, 0.6, 0.8, 0.5, 0.3, 0.1];
        let repeats = if n <= 14 { 51 } else { 7 };
        let t = median_seconds(repeats, || {
            qaoa.expectation_and_gradient(&x).unwrap();
        });
        println!("| {n} | {} |", fmt_time(t));
    }

    println!("\n## Quantum walks\n");
    println!("| case | time |");
    println!("|---|---:|");
    let cube = ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0)?;
    let t = median_seconds(101, || {
        cube.probabilities(0, 1.3).unwrap();
    });
    println!(
        "| Metatron's Cube, P(t) (dense spectral) | {} |",
        fmt_time(t)
    );
    let torus = generators::grid_2d(100, 100, true);
    let walk = ContinuousWalk::new(&torus, Generator::Laplacian, 1.0)?;
    let t = median_seconds(5, || {
        walk.probabilities(0, 5.0).unwrap();
    });
    println!(
        "| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | {} |",
        fmt_time(t)
    );
    let t = median_seconds(5, || {
        metatron::metatron_cube().automorphism_group();
    });
    println!(
        "| automorphism group of Metatron's Cube (13!) | {} |",
        fmt_time(t)
    );

    println!("\n## Validated results (reproduced by this run)\n");
    println!("| quantity | QSO | reference |");
    println!("|---|---:|---:|");
    let h2 = models::h2_sto3g();
    let vqe = Vqe::new(
        h2.hamiltonian.clone(),
        ansatz::real_amplitudes(2, 1, Entanglement::Linear),
    )?
    .with_restarts(4)
    .run()?;
    println!(
        "| H₂/STO-3G total energy, VQE (Ha) | {:.6} | −1.137306 (FCI) |",
        vqe.energy + h2.nuclear_repulsion
    );
    let ring = MaxCutQaoa::new(&generators::cycle(10), 1)?.optimize(&QaoaConfig::default())?;
    println!(
        "| QAOA p=1 ratio, ring C₁₀ | {:.6} | 0.75 (Farhi et al.) |",
        ring.approximation_ratio.unwrap()
    );
    let q3 = MaxCutQaoa::new(&generators::hypercube(3), 1)?.optimize(&QaoaConfig::default())?;
    println!(
        "| QAOA p=1 ratio, cube graph Q₃ | {:.6} | 0.6925 (3-regular, triangle-free) |",
        q3.approximation_ratio.unwrap()
    );
    let layered = optimize_layerwise(&metatron::metatron_cube(), 3, &QaoaConfig::default())?;
    println!(
        "| QAOA p=3 ratio, Metatron's Cube (K₁₃, C_max = 42) | {:.6} | — |",
        layered[2].approximation_ratio.unwrap()
    );
    let limit = cube.limiting_distribution(0)?;
    println!(
        "| CTQW limiting P(centre), Metatron's Cube | {:.6} | 145/169 = {:.6} |",
        limit[0],
        145.0 / 169.0
    );
    let search = SpatialSearch::new(
        &metatron::metatron_cube(),
        &[3],
        Generator::Laplacian,
        1.0 / 13.0,
    )?
    .success_probability(std::f64::consts::PI * 13f64.sqrt() / 2.0);
    println!("| spatial search on K₁₃, P at t = π√13/2 | {search:.10} | 1 |");
    let g = grover::grover_search(10, &[123], None)?;
    println!(
        "| Grover, 10 qubits, {} iterations | {:.6} | {:.6} = sin²((2k+1)θ) |",
        g.iterations, g.success_probability, g.theoretical_success_probability
    );
    println!(
        "| \\|Aut\\| Metatron's Cube / Fruit of Life / dodecahedron | {} / {} / {} | 13! / 12 / 120 |",
        metatron::metatron_cube()
            .automorphism_group()
            .order()
            .unwrap(),
        metatron::fruit_of_life()
            .automorphism_group()
            .order()
            .unwrap(),
        generators::platonic(generators::PlatonicSolid::Dodecahedron)
            .automorphism_group()
            .order()
            .unwrap()
    );
    Ok(())
}
