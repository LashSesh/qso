//! Prints the benchmark and validation table used in `docs/BENCHMARKS.md`.
//!
//! Run with `cargo run --release --example benchmark_report`. Timings are
//! medians of repeated runs on the current machine. With `--json PATH` the
//! timings are also written as JSON (`{"metrics": {name: seconds}, ...}`)
//! for the weekly benchmark tracking (`benchmarks/track.py`).

use std::time::Instant;

use metatron_qso::algorithms::gradients::{
    GradientMethod, expectation, expectation_and_gradient, expectation_and_gradient_batch,
    expectation_batch,
};
use metatron_qso::algorithms::qaoa::optimize_layerwise;
use metatron_qso::noise;
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
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1).cloned());
    let mut metrics = serde_json::Map::new();
    let mut record = |name: String, seconds: f64| {
        metrics.insert(name, serde_json::json!(seconds));
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("Machine: {threads} hardware threads; build: release\n");

    println!("## Statevector throughput (layer = H + RZ on every qubit, CX chain)\n");
    println!("| qubits | gates/layer | time/layer | ns per gate·amplitude |");
    println!("|---:|---:|---:|---:|");
    for n in [10usize, 14, 16, 18, 20, 22, 24] {
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
        record(format!("statevector_layer/qubits={n}"), t);
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
        record(format!("gate_pass/{kind}/qubits={n}"), t / 5.0);
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
        record(format!("gradient_tfim10/{}", name.replace(' ', "_")), t);
        println!("| {name} | {} |", fmt_time(t));
    }

    println!("\n## Batched evaluation (same circuit and Hamiltonian, 256 parameter vectors)\n");
    println!("| quantity | one call per vector | one batch call | speed-up |");
    println!("|---|---:|---:|---:|");
    let thetas: Vec<Vec<f64>> = (0..256)
        .map(|k| {
            (0..circuit.num_parameters())
                .map(|j| ((k * 31 + j * 7) % 97) as f64 * 0.06 - 3.0)
                .collect()
        })
        .collect();
    let loop_values = median_seconds(5, || {
        for t in &thetas {
            expectation(&circuit, &h, t, None).unwrap();
        }
    });
    let batch_values = median_seconds(5, || {
        expectation_batch(&circuit, &h, &thetas, None).unwrap();
    });
    let loop_grads = median_seconds(3, || {
        for t in &thetas {
            expectation_and_gradient(&circuit, &h, t, None, GradientMethod::Adjoint).unwrap();
        }
    });
    let batch_grads = median_seconds(3, || {
        expectation_and_gradient_batch(&circuit, &h, &thetas, None, GradientMethod::Adjoint)
            .unwrap();
    });
    record("batch256_tfim10/values_loop".into(), loop_values);
    record("batch256_tfim10/values_batch".into(), batch_values);
    record("batch256_tfim10/gradients_loop".into(), loop_grads);
    record("batch256_tfim10/gradients_batch".into(), batch_grads);
    for (name, a, b) in [
        ("expectation values", loop_values, batch_values),
        ("values + adjoint gradients", loop_grads, batch_grads),
    ] {
        println!(
            "| {name} | {} | {} | {:.1}× |",
            fmt_time(a),
            fmt_time(b),
            a / b
        );
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
        record(format!("qaoa_value_gradient_p3/nodes={n}"), t);
        println!("| {n} | {} |", fmt_time(t));
    }

    println!("\n## Quantum walks\n");
    println!("| case | time |");
    println!("|---|---:|");
    let cube = ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0)?;
    let t = median_seconds(101, || {
        cube.probabilities(0, 1.3).unwrap();
    });
    record("walk/metatron_dense".into(), t);
    println!(
        "| Metatron's Cube, P(t) (dense spectral) | {} |",
        fmt_time(t)
    );
    let torus = generators::grid_2d(100, 100, true);
    let walk = ContinuousWalk::new(&torus, Generator::Laplacian, 1.0)?;
    let t = median_seconds(5, || {
        walk.probabilities(0, 5.0).unwrap();
    });
    record("walk/torus_100x100_krylov".into(), t);
    println!(
        "| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | {} |",
        fmt_time(t)
    );
    let t = median_seconds(5, || {
        metatron::metatron_cube().automorphism_group();
    });
    record("automorphism_group/metatron".into(), t);
    println!(
        "| automorphism group of Metatron's Cube (13!) | {} |",
        fmt_time(t)
    );

    println!(
        "\n## Noisy circuits (device: T1 = 100, T2 = 80, gate durations, depolarising errors)\n"
    );
    println!(
        "Layers of RY on every qubit and a brick of CX, 4 layers; one trajectory against the noiseless circuit.\n"
    );
    println!("| qubits | gates | noiseless | one trajectory | ratio |");
    println!("|---:|---:|---:|---:|---:|");
    let device = noise::DeviceModel::new("benchmark")
        .with_default_qubit(noise::QubitProperties::new().with_t1_t2(100.0, 80.0))
        .with_rule(
            noise::GateRule::any()
                .with_num_qubits(2)
                .with_duration(0.3)
                .with_depolarizing(0.01),
        )
        .with_rule(
            noise::GateRule::any()
                .with_duration(0.035)
                .with_depolarizing(0.001),
        );
    let layered = |n: usize| {
        let mut c = Circuit::new(n);
        for l in 0..4 {
            for q in 0..n {
                c.ry(q, 0.3 + 0.1 * (q + l) as f64);
            }
            for q in (l % 2..n - 1).step_by(2) {
                c.cx(q, q + 1);
            }
        }
        c
    };
    for n in [12usize, 16, 20] {
        let c = layered(n);
        let ideal = median_seconds(5, || {
            StateVector::from_circuit(&c, &[]).unwrap();
        });
        let mut k = 0;
        let noisy = median_seconds(5, || {
            k += 1;
            noise::TrajectorySimulator::new(device.clone(), 1, k)
                .unwrap()
                .trajectory_state(&c, &[], 0)
                .unwrap();
        });
        record(format!("trajectory/qubits={n}"), noisy);
        println!(
            "| {n} | {} | {} | {} | {:.1} |",
            c.len(),
            fmt_time(ideal),
            fmt_time(noisy),
            noisy / ideal
        );
    }
    // Throughput: many trajectories (compiled once, parallel over
    // trajectories) against as many noiseless runs.
    println!("\n| 200 runs, 16 qubits | time | per run |");
    println!("|---|---:|---:|");
    let c = layered(16);
    let ideal = median_seconds(3, || {
        for _ in 0..200 {
            StateVector::from_circuit(&c, &[]).unwrap();
        }
    });
    let z: PauliSum = "Z0".parse().unwrap();
    let noisy = median_seconds(3, || {
        noise::TrajectorySimulator::new(device.clone(), 200, 1)
            .unwrap()
            .expectation(&c, &[], &z)
            .unwrap();
    });
    record("trajectories200/qubits=16".into(), noisy);
    println!(
        "| noiseless | {} | {} |",
        fmt_time(ideal),
        fmt_time(ideal / 200.0)
    );
    println!(
        "| trajectories ({:.1}× the noiseless time) | {} | {} |",
        noisy / ideal,
        fmt_time(noisy),
        fmt_time(noisy / 200.0)
    );
    println!("\n| density matrix, same circuit | time |");
    println!("|---|---:|");
    for n in [8usize, 10] {
        let c = layered(n);
        let t = median_seconds(3, || {
            let mut rho = noise::DensityMatrix::zero(n).unwrap();
            rho.apply_device(&c, &[], &device).unwrap();
        });
        record(format!("density_matrix_device/qubits={n}"), t);
        println!("| {n} qubits | {} |", fmt_time(t));
    }

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
    if let Some(path) = json_path {
        let report = serde_json::json!({
            "commit": std::env::var("QSO_COMMIT").or_else(|_| std::env::var("GITHUB_SHA")).ok(),
            "threads": threads,
            "metrics": metrics,
        });
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&report).expect("serialisable") + "\n",
        )
        .map_err(|e| QsoError::invalid("--json", e.to_string()))?;
        eprintln!("wrote {path}");
    }
    Ok(())
}
