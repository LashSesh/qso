//! Computes every number of the QSO technical note and writes it as JSON.
//!
//! ```text
//! cargo run --release --example technical_note -- --out note.json
//! cargo run --release --example technical_note -- --quick     # smoke test
//! ```
//!
//! Each quantity is reported next to its independent reference (closed
//! form, published value or a second method). Timings are medians of
//! repeated runs after a warm-up and depend on the machine.

use std::time::Instant;

use metatron_qso::algorithms::gradients::{GradientMethod, expectation_and_gradient};
use metatron_qso::algorithms::grover::{self as grover_mod, optimal_iterations};
use metatron_qso::algorithms::qaoa::optimize_layerwise;
use metatron_qso::algorithms::{GraphClassifier, classifier::VariationalClassifier};
use metatron_qso::photonics::{self, BosonSampler};
use metatron_qso::prelude::*;
use metatron_qso::walks::classical_hitting_times;
use serde_json::{Value, json};

/// Problem sizes: full run for the note, reduced run for CI smoke tests.
#[derive(Clone, Copy)]
struct Scale {
    quick: bool,
}

impl Scale {
    fn pick<T>(self, full: T, quick: T) -> T {
        if self.quick { quick } else { full }
    }
}

/// A section of the report.
type Section = fn(Scale) -> Result<Value>;
/// A labelled dataset generator.
type Generator2d = fn(usize, &mut Rng) -> Dataset;

fn median_seconds(repeats: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut t: Vec<f64> = (0..repeats.max(1))
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scale = Scale {
        quick: args.iter().any(|a| a == "--quick"),
    };
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1).cloned());
    let started = Instant::now();
    let mut report = serde_json::Map::new();
    let sections: [(&str, Section); 8] = [
        ("grover", grover),
        ("walks", walks),
        ("qaoa", qaoa),
        ("vqe", vqe),
        ("vqc", vqc),
        ("graph_classifier", graph_classifier),
        ("boson_sampling", boson_sampling),
        ("performance", performance),
    ];
    report.insert("environment".into(), environment(scale));
    for (name, section) in sections {
        let t = Instant::now();
        let value = section(scale)?;
        eprintln!("{name:>18}: {:.1} s", t.elapsed().as_secs_f64());
        report.insert(name.into(), value);
    }
    report.insert(
        "total_seconds".into(),
        json!(started.elapsed().as_secs_f64()),
    );
    let text = serde_json::to_string_pretty(&Value::Object(report)).expect("serialisable");
    match out {
        Some(path) => std::fs::write(&path, text + "\n")
            .map_err(|e| QsoError::invalid("out", format!("cannot write {path}: {e}")))?,
        None => println!("{text}"),
    }
    Ok(())
}

fn environment(scale: Scale) -> Value {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    json!({
        "qso_version": env!("CARGO_PKG_VERSION"),
        "commit": std::env::var("QSO_COMMIT").or_else(|_| std::env::var("GITHUB_SHA")).ok(),
        "machine": std::env::var("QSO_MACHINE").ok(),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "threads": threads,
        "date": civil_date(unix),
        "quick": scale.quick,
    })
}

/// `YYYY-MM-DD` from Unix seconds (Howard Hinnant's days-from-civil inverse).
fn civil_date(unix: u64) -> String {
    let z = (unix / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

// ---------------------------------------------------------------------------
// Grover

fn grover(scale: Scale) -> Result<Value> {
    let mut single = Vec::new();
    for n in 2..=scale.pick(16, 10) {
        let size = 1usize << n;
        let r = grover_mod::grover_search(n, &[size / 3], None)?;
        // Classical search of an unsorted list needs (N + 1)/2 queries on
        // average to find one marked item; Grover needs k oracle calls.
        let classical = (size as f64 + 1.0) / 2.0;
        single.push(json!({
            "n_qubits": n,
            "items": size,
            "iterations": r.iterations,
            "success": r.success_probability,
            "theory": r.theoretical_success_probability,
            "abs_error": (r.success_probability - r.theoretical_success_probability).abs(),
            "classical_expected_queries": classical,
            "query_ratio": classical / r.iterations.max(1) as f64,
        }));
    }
    let n = 10;
    let mut multi = Vec::new();
    for m in [1usize, 2, 4, 8, 16, 32] {
        let marked: Vec<usize> = (0..m).map(|i| (i * 997 + 13) % (1 << n)).collect();
        let r = grover_mod::grover_search(n, &marked, None)?;
        multi.push(json!({
            "n_qubits": n,
            "marked": m,
            "iterations": r.iterations,
            "optimal_iterations": optimal_iterations(n, m),
            "success": r.success_probability,
            "theory": r.theoretical_success_probability,
            "abs_error": (r.success_probability - r.theoretical_success_probability).abs(),
        }));
    }
    let t = median_seconds(scale.pick(5, 1), || {
        grover_mod::grover_search(12, &[1234], None).unwrap();
    });
    let max_err = single
        .iter()
        .chain(&multi)
        .map(|v| v["abs_error"].as_f64().unwrap())
        .fold(0.0, f64::max);
    Ok(json!({
        "single": single,
        "multi": multi,
        "max_abs_error": max_err,
        "seconds_n12": t,
    }))
}

// ---------------------------------------------------------------------------
// Quantum walks

fn walks(scale: Scale) -> Result<Value> {
    let cube = metatron::metatron_cube();
    let walk = ContinuousWalk::new(&cube, Generator::Laplacian, 1.0)?;
    let limit = walk.limiting_distribution(0)?;
    // Mixing time on the grid t_max·k/steps against the closed form: from
    // node 0 of K13 the time-averaged distance to the limit is
    // (24/169)·|sin(13T)/(13T)|.
    let (t_max, steps) = (5.0, scale.pick(5000, 500));
    let mut mixing = Vec::new();
    for eps in [0.05, 0.02, 0.01] {
        let reference = (1..=steps)
            .rev()
            .map(|k| t_max * k as f64 / steps as f64)
            .take_while(|&t| 24.0 / 169.0 * ((13.0 * t).sin() / (13.0 * t)).abs() <= eps)
            .last();
        mixing.push(json!({
            "epsilon": eps,
            "mixing_time": walk.mixing_time(0, eps, t_max, steps)?,
            "closed_form": reference,
        }));
    }
    let fruit = metatron::fruit_of_life();
    let fruit_walk = ContinuousWalk::new(&fruit, Generator::Laplacian, 1.0)?;
    let fruit_limit = fruit_walk.limiting_distribution(0)?;
    let fruit_mixing = fruit_walk.mixing_time(0, 0.05, 200.0, scale.pick(4000, 400))?;

    // Spatial search on Metatron's Cube = K13.
    let t_star = std::f64::consts::PI * 13f64.sqrt() / 2.0;
    let exact = SpatialSearch::new(&cube, &[3], Generator::Laplacian, 1.0 / 13.0)?;
    let optimised = SpatialSearch::optimize(
        &cube,
        &[3],
        Generator::Laplacian,
        std::f64::consts::PI * 13f64.sqrt(),
        2000,
    )?;
    let classical = classical_hitting_times(&cube, 3)?;
    let classical_mean = classical
        .iter()
        .enumerate()
        .filter(|&(v, _)| v != 3)
        .map(|(_, &h)| h)
        .sum::<f64>()
        / 12.0;

    // Scaling on complete graphs: quantum time π√N/2 vs classical N − 1.
    let mut scaling = Vec::new();
    for n_nodes in scale.pick(vec![13usize, 64, 256, 1024], vec![13, 64]) {
        let g = generators::complete(n_nodes);
        let s = SpatialSearch::new(&g, &[0], Generator::Laplacian, 1.0 / n_nodes as f64)?;
        let t = std::f64::consts::PI * (n_nodes as f64).sqrt() / 2.0;
        let h = classical_hitting_times(&g, 0)?;
        scaling.push(json!({
            "nodes": n_nodes,
            "quantum_time": t,
            "success_at_quantum_time": s.success_probability(t),
            "classical_mean_hitting_time": h[1],
            "classical_reference": (n_nodes - 1) as f64,
            "ratio": h[1] / t,
        }));
    }

    // Fruit of Life contact graph: search for every vertex orbit.
    let mut fruit_search = Vec::new();
    for v in [0usize, 1, 7] {
        let r = SpatialSearch::optimize(&fruit, &[v], Generator::Laplacian, 40.0, 4000)?;
        let h = classical_hitting_times(&fruit, v)?;
        let mean = h.iter().sum::<f64>() / 12.0;
        fruit_search.push(json!({
            "marked": v,
            "gamma": r.gamma,
            "time": r.time,
            "success": r.success_probability,
            "expected_time": r.expected_time,
            "classical_mean_hitting_time": mean,
        }));
    }

    let mut groups = Vec::new();
    for (name, g, reference) in [
        ("Metatron's Cube", metatron::metatron_cube(), "13! (S13)"),
        ("Fruit of Life", metatron::fruit_of_life(), "12 (D6)"),
        (
            "dodecahedron",
            generators::platonic(generators::PlatonicSolid::Dodecahedron),
            "120 (Ih)",
        ),
        ("Petersen", generators::petersen(), "120 (S5)"),
        ("Heawood", generators::heawood(), "336"),
    ] {
        groups.push(json!({
            "graph": name,
            "order": g.automorphism_group().order(),
            "reference": reference,
        }));
    }
    Ok(json!({
        "metatron_limiting_centre": limit[0],
        "metatron_limiting_centre_reference": 145.0 / 169.0,
        "metatron_mixing_times": mixing,
        "fruit_limiting_centre": fruit_limit[0],
        "fruit_mixing_time_eps005": fruit_mixing,
        "search_exact": {
            "gamma": 1.0 / 13.0,
            "time": t_star,
            "success": exact.success_probability(t_star),
        },
        "search_optimised": {
            "gamma": optimised.gamma,
            "time": optimised.time,
            "success": optimised.success_probability,
        },
        "classical_mean_hitting_time": classical_mean,
        "complete_graph_scaling": scaling,
        "fruit_of_life_search": fruit_search,
        "automorphism_groups": groups,
    }))
}

// ---------------------------------------------------------------------------
// QAOA

fn ratio(r: &QaoaResult) -> f64 {
    r.approximation_ratio.unwrap_or(f64::NAN)
}

fn qaoa(scale: Scale) -> Result<Value> {
    let config = QaoaConfig::default();
    let mut ring = Vec::new();
    for p in 1..=scale.pick(4, 2) {
        let r = MaxCutQaoa::new(&generators::cycle(10), p)?.optimize(&config)?;
        let reference = (2.0 * p as f64 + 1.0) / (2.0 * p as f64 + 2.0);
        ring.push(json!({"p": p, "ratio": ratio(&r), "reference": reference,
                         "abs_error": (ratio(&r) - reference).abs()}));
    }
    let q3 = MaxCutQaoa::new(&generators::hypercube(3), 1)?.optimize(&config)?;
    let heawood = MaxCutQaoa::new(&generators::heawood(), 2)?.optimize(&QaoaConfig {
        restarts: 16,
        ..Default::default()
    })?;
    let metatron = optimize_layerwise(&metatron::metatron_cube(), scale.pick(3, 2), &config)?;
    let petersen = optimize_layerwise(&generators::petersen(), scale.pick(4, 2), &config)?;
    let layer_rows = |rs: &[QaoaResult]| -> Vec<Value> {
        rs.iter()
            .map(|r| json!({"p": r.p, "ratio": ratio(r), "optimal_probability": r.optimal_probability}))
            .collect()
    };

    // Random 3-regular graphs: mean ratio over instances, layer-wise INTERP.
    let (instances, nodes, p_max) = scale.pick((10usize, 14usize, 4usize), (3, 10, 2));
    let mut rng = Rng::seed_from_u64(2026);
    let mut per_p: Vec<Vec<f64>> = vec![Vec::new(); p_max];
    let mut per_p_opt: Vec<Vec<f64>> = vec![Vec::new(); p_max];
    for _ in 0..instances {
        let g = generators::random_regular(nodes, 3, &mut rng)?;
        for (k, r) in optimize_layerwise(&g, p_max, &config)?.iter().enumerate() {
            per_p[k].push(ratio(r));
            per_p_opt[k].push(r.optimal_probability);
        }
    }
    let random_regular: Vec<Value> = (0..p_max)
        .map(|k| {
            let (m, s) = mean_std(&per_p[k]);
            let (mo, so) = mean_std(&per_p_opt[k]);
            json!({"p": k + 1, "ratio_mean": m, "ratio_std": s,
                   "optimal_probability_mean": mo, "optimal_probability_std": so})
        })
        .collect();
    Ok(json!({
        "ring_c10": ring,
        "cube_q3_p1": {"ratio": ratio(&q3), "reference": 0.692450},
        "heawood_p2": {"ratio": ratio(&heawood), "reference": 0.7559},
        "metatron_cube": layer_rows(&metatron),
        "metatron_max_cut": metatron[0].max_cut,
        "petersen": layer_rows(&petersen),
        "random_regular": {"nodes": nodes, "instances": instances, "layers": random_regular},
    }))
}

// ---------------------------------------------------------------------------
// VQE

fn vqe(scale: Scale) -> Result<Value> {
    let h2 = models::h2_sto3g();
    let r = Vqe::new(
        h2.hamiltonian.clone(),
        ansatz::real_amplitudes(2, 1, Entanglement::Linear),
    )?
    .with_restarts(4)
    .run()?;
    let h2_exact = h2.hamiltonian.ground_state_energy(2)? + h2.nuclear_repulsion;
    let h2_vqe = r.energy + h2.nuclear_repulsion;

    let n = scale.pick(8, 4);
    let models: Vec<(&str, PauliSum)> = vec![
        (
            "TFIM (J = 1, h = 1)",
            models::transverse_field_ising(n, 1.0, 1.0, false),
        ),
        (
            "XXZ (J = 1, Δ = 0.5)",
            models::heisenberg_xxz(n, 1.0, 0.5, false),
        ),
    ];
    let mut rows = Vec::new();
    for (name, h) in &models {
        let exact = h.ground_state_energy(n)?;
        for (ansatz_name, rotations) in [
            ("RealAmplitudes (RY)", vec![ansatz::Rotation::RY]),
            (
                "EfficientSU2 (RY, RZ)",
                vec![ansatz::Rotation::RY, ansatz::Rotation::RZ],
            ),
        ] {
            for layers in 1..=scale.pick(4, 2) {
                let circuit =
                    ansatz::hardware_efficient(n, layers, &rotations, Entanglement::Linear);
                let parameters = circuit.num_parameters();
                let start = Instant::now();
                let res = Vqe::new(h.clone(), circuit)?
                    .with_restarts(scale.pick(4, 1))
                    .with_seed(7)
                    .run()?;
                rows.push(json!({
                    "model": name,
                    "ansatz": ansatz_name,
                    "layers": layers,
                    "parameters": parameters,
                    "energy": res.energy,
                    "exact": exact,
                    "abs_error": res.energy - exact,
                    "relative_error": (res.energy - exact) / exact.abs(),
                    "seconds": start.elapsed().as_secs_f64(),
                }));
            }
        }
    }
    Ok(json!({
        "h2": {"vqe": h2_vqe, "exact": h2_exact, "fci_reference": -1.137306,
               "abs_error": (h2_vqe - h2_exact).abs()},
        "qubits": n,
        "spin_models": rows,
    }))
}

// ---------------------------------------------------------------------------
// Variational classifier

type Dataset = (Vec<Vec<f64>>, Vec<bool>);

fn blobs(n: usize, rng: &mut Rng) -> Dataset {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..n {
        let y = i % 2 == 0;
        let c = if y { 0.8 } else { -0.8 };
        xs.push(vec![c + 0.35 * rng.normal(), c + 0.35 * rng.normal()]);
        ys.push(y);
    }
    (xs, ys)
}

fn circle(n: usize, rng: &mut Rng) -> Dataset {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    while xs.len() < n {
        let x = [rng.uniform_range(-1.0, 1.0), rng.uniform_range(-1.0, 1.0)];
        let r2 = x[0] * x[0] + x[1] * x[1];
        if (r2 - 0.45).abs() < 0.08 {
            continue;
        }
        xs.push(vec![2.0 * x[0], 2.0 * x[1]]);
        ys.push(r2 < 0.45);
    }
    (xs, ys)
}

fn moons(n: usize, rng: &mut Rng) -> Dataset {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..n {
        let y = i % 2 == 0;
        let t = rng.uniform_range(0.0, std::f64::consts::PI);
        let (x0, x1) = if y {
            (t.cos(), t.sin())
        } else {
            (1.0 - t.cos(), 0.5 - t.sin())
        };
        xs.push(vec![
            1.6 * (x0 - 0.5) + 0.1 * rng.normal(),
            1.6 * (x1 - 0.25) + 0.1 * rng.normal(),
        ]);
        ys.push(y);
    }
    (xs, ys)
}

/// Classical baseline: logistic regression on `features(x)` by gradient
/// descent on the mean cross-entropy.
fn logistic_accuracy(
    train: &Dataset,
    test: &Dataset,
    features: impl Fn(&[f64]) -> Vec<f64>,
) -> (f64, f64) {
    let phi: Vec<Vec<f64>> = train.0.iter().map(|x| features(x)).collect();
    let d = phi[0].len();
    let mut w = vec![0.0; d + 1];
    for _ in 0..5000 {
        let mut g = vec![0.0; d + 1];
        for (f, &y) in phi.iter().zip(&train.1) {
            let z = w[d] + f.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
            let e = 1.0 / (1.0 + (-z).exp()) - if y { 1.0 } else { 0.0 };
            g.iter_mut().zip(f).for_each(|(gi, fi)| *gi += e * fi);
            g[d] += e;
        }
        w.iter_mut()
            .zip(&g)
            .for_each(|(wi, gi)| *wi -= 0.5 * gi / phi.len() as f64);
    }
    let acc = |data: &Dataset| {
        let correct = data
            .0
            .iter()
            .zip(&data.1)
            .filter(|(x, y)| {
                let f = features(x);
                let z = w[d] + f.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
                (z >= 0.0) == **y
            })
            .count();
        correct as f64 / data.0.len() as f64
    };
    (acc(train), acc(test))
}

fn vqc(scale: Scale) -> Result<Value> {
    let seeds = scale.pick(5u64, 1);
    let samples = scale.pick(80, 30);
    let generators_: [(&str, Generator2d); 3] =
        [("blobs", blobs), ("circle", circle), ("moons", moons)];
    let mut rows = Vec::new();
    for (name, generate) in generators_ {
        let (mut q_train, mut q_test) = (Vec::new(), Vec::new());
        let (mut lin_test, mut quad_test) = (Vec::new(), Vec::new());
        let mut seconds = Vec::new();
        for seed in 0..seeds {
            let mut rng = Rng::seed_from_u64(100 + seed);
            let train = generate(samples, &mut rng);
            let test = generate(samples, &mut rng);
            let mut model = VariationalClassifier::new(2, 2, 4)?;
            let start = Instant::now();
            model.fit(
                &train.0,
                &train.1,
                &metatron_qso::optimize::Options {
                    max_iterations: scale.pick(400, 100),
                    ..Default::default()
                },
                seed,
            )?;
            seconds.push(start.elapsed().as_secs_f64());
            q_train.push(model.accuracy(&train.0, &train.1)?);
            q_test.push(model.accuracy(&test.0, &test.1)?);
            lin_test.push(logistic_accuracy(&train, &test, |x| x.to_vec()).1);
            quad_test.push(
                logistic_accuracy(&train, &test, |x| {
                    vec![x[0], x[1], x[0] * x[0], x[1] * x[1], x[0] * x[1]]
                })
                .1,
            );
        }
        let ms = |v: &[f64]| {
            let (m, s) = mean_std(v);
            json!({"mean": m, "std": s})
        };
        rows.push(json!({
            "dataset": name,
            "samples_per_split": samples,
            "seeds": seeds,
            "vqc_train": ms(&q_train),
            "vqc_test": ms(&q_test),
            "logistic_linear_test": ms(&lin_test),
            "logistic_quadratic_test": ms(&quad_test),
            "vqc_train_seconds": ms(&seconds),
        }));
    }
    Ok(json!({"model": "2 qubits, 4 re-uploading layers, 16 parameters", "datasets": rows}))
}

// ---------------------------------------------------------------------------
// Graph classifier

fn has_square(g: &Graph) -> bool {
    // A 4-cycle exists iff two vertices share at least two neighbours.
    let n = g.num_nodes();
    (0..n).any(|u| {
        (u + 1..n).any(|v| {
            g.neighbors(u)
                .iter()
                .filter(|&&(w, _)| g.has_edge(w, v))
                .count()
                >= 2
        })
    })
}

fn cubic_dataset(
    nodes: usize,
    per_class: usize,
    rng: &mut Rng,
    label: impl Fn(&Graph) -> Option<bool>,
) -> Result<(Vec<Graph>, Vec<bool>)> {
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    while pos.len() < per_class || neg.len() < per_class {
        let g = generators::random_regular(nodes, 3, rng)?;
        match label(&g) {
            Some(true) if pos.len() < per_class => pos.push(g),
            Some(false) if neg.len() < per_class => neg.push(g),
            _ => {}
        }
    }
    let labels = [vec![true; per_class], vec![false; per_class]].concat();
    Ok(([pos, neg].concat(), labels))
}

/// Number of distinct graphs up to isomorphism, estimated by adjacency
/// spectra (cospectral pairs are rare at these sizes).
fn distinct_spectra(graphs: &[Graph]) -> usize {
    let mut keys: Vec<Vec<i64>> = graphs
        .iter()
        .map(|g| {
            g.adjacency_spectrum()
                .iter()
                .map(|x| (x * 1e6).round() as i64)
                .collect()
        })
        .collect();
    keys.sort();
    keys.dedup();
    keys.len()
}

fn graph_task(
    scale: Scale,
    name: &str,
    nodes: usize,
    label: impl Fn(&Graph) -> Option<bool> + Copy,
    oracle: &str,
) -> Result<Value> {
    let per_class = scale.pick(20, 6);
    let seeds = scale.pick(5u64, 1);
    let mut rows = Vec::new();
    let mut distinct = Vec::new();
    for layers in [1usize, 2] {
        let (mut train_acc, mut test_acc, mut seconds) = (Vec::new(), Vec::new(), Vec::new());
        for seed in 0..seeds {
            let mut rng = Rng::seed_from_u64(500 + seed);
            let (train, y_train) = cubic_dataset(nodes, per_class, &mut rng, label)?;
            let (test, y_test) = cubic_dataset(nodes, per_class, &mut rng, label)?;
            if layers == 1 {
                distinct.push(json!({
                    "train_positive": distinct_spectra(&train[..per_class]),
                    "train_negative": distinct_spectra(&train[per_class..]),
                }));
            }
            let mut model = GraphClassifier::new(layers)?;
            let start = Instant::now();
            let report = model.fit(
                &train,
                &y_train,
                &metatron_qso::optimize::Options {
                    max_iterations: scale.pick(200, 50),
                    ..Default::default()
                },
                scale.pick(3, 1),
                seed,
            )?;
            seconds.push(start.elapsed().as_secs_f64());
            train_acc.push(report.accuracy);
            test_acc.push(model.accuracy(&test, &y_test)?);
        }
        let ms = |v: &[f64]| {
            let (m, s) = mean_std(v);
            json!({"mean": m, "std": s})
        };
        rows.push(json!({
            "layers": layers,
            "train": ms(&train_acc),
            "test": ms(&test_acc),
            "seconds": ms(&seconds),
        }));
    }
    Ok(json!({
        "task": name,
        "nodes": nodes,
        "graphs_per_class_and_split": per_class,
        "seeds": seeds,
        "models": rows,
        "degree_baseline_test": 0.5,
        "oracle": oracle,
        "distinct_graphs": distinct,
    }))
}

fn graph_classifier(scale: Scale) -> Result<Value> {
    let triangles = graph_task(
        scale,
        "contains a triangle (random cubic graphs)",
        scale.pick(12, 8),
        |g| Some(g.triangle_count() > 0),
        "triangle count > 0",
    )?;
    let squares = graph_task(
        scale,
        "girth 4 vs girth ≥ 5 (triangle-free random cubic graphs)",
        scale.pick(14, 10),
        |g| {
            if g.triangle_count() > 0 {
                None
            } else {
                Some(has_square(g))
            }
        },
        "4-cycle present",
    )?;
    Ok(json!({"triangles": triangles, "squares": squares}))
}

// ---------------------------------------------------------------------------
// Boson sampling

fn naive_permanent(a: &[C64], n: usize) -> C64 {
    fn rec(a: &[C64], n: usize, row: usize, used: &mut [bool]) -> C64 {
        if row == n {
            return C64::new(1.0, 0.0);
        }
        let mut total = C64::new(0.0, 0.0);
        for c in 0..n {
            if !used[c] {
                used[c] = true;
                total += a[row * n + c] * rec(a, n, row + 1, used);
                used[c] = false;
            }
        }
        total
    }
    rec(a, n, 0, &mut vec![false; n])
}

fn boson_sampling(scale: Scale) -> Result<Value> {
    // Permanent against the definition.
    let mut rng = Rng::seed_from_u64(77);
    let mut max_rel = 0.0f64;
    for n in 1..=8 {
        let a: Vec<C64> = (0..n * n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let m = metatron_qso::linalg::CMatrix::from_row_slice(n, n, &a);
        let p = photonics::permanent(&m)?;
        let q = naive_permanent(&a, n);
        max_rel = max_rel.max((p - q).norm() / q.norm());
    }

    // Hong–Ou–Mandel.
    let hom = BosonSampler::new(
        photonics::beam_splitter(std::f64::consts::FRAC_PI_4, 0.0),
        &[1, 1],
    )?;
    let (p_q, p_c) = (
        hom.probability(&[1, 1])?,
        hom.probability_distinguishable(&[1, 1])?,
    );

    // Full bunching: P_boson / P_distinguishable = n!.
    let mut bunching = Vec::new();
    let mut factorial = 1.0;
    for n in 1..=6usize {
        factorial *= n as f64;
        let u = photonics::haar_unitary(8, &mut rng)?;
        let mut input = vec![0; 8];
        input[..n].iter_mut().for_each(|s| *s = 1);
        let s = BosonSampler::new(u, &input)?;
        let mut out = vec![0; 8];
        out[5] = n;
        bunching.push(json!({
            "photons": n,
            "ratio": s.probability(&out)? / s.probability_distinguishable(&out)?,
            "reference": factorial,
        }));
    }

    // Exact sampler against the exact distribution.
    let (m, n, shots) = (8usize, 4usize, scale.pick(200_000usize, 20_000));
    let u = photonics::haar_unitary(m, &mut Rng::seed_from_u64(78))?;
    let sampler = BosonSampler::new(u, &[1, 1, 1, 1, 0, 0, 0, 0])?;
    let dist = sampler.distribution()?;
    let dist_c = sampler.distribution_distinguishable()?;
    let samples = sampler.sample(shots, &mut Rng::seed_from_u64(79))?;
    let index: std::collections::HashMap<&Vec<usize>, usize> = dist
        .patterns
        .iter()
        .enumerate()
        .map(|(k, p)| (p, k))
        .collect();
    let mut counts = vec![0usize; dist.patterns.len()];
    for s in &samples {
        counts[index[s]] += 1;
    }
    let tvd_samples = 0.5
        * counts
            .iter()
            .zip(&dist.probabilities)
            .map(|(&c, &p)| (c as f64 / shots as f64 - p).abs())
            .sum::<f64>();
    let (mut chi2, mut dof) = (0.0, 0usize);
    for (&c, &p) in counts.iter().zip(&dist.probabilities) {
        let e = p * shots as f64;
        if e >= 5.0 {
            chi2 += (c as f64 - e).powi(2) / e;
            dof += 1;
        }
    }
    // Expected distance of an exact sampler's histogram from the exact
    // distribution: E|X/N − p| ≈ √(2p(1 − p)/(πN)) per cell.
    let tvd_expected = 0.5
        * dist
            .probabilities
            .iter()
            .map(|&p| (2.0 * p * (1.0 - p) / (std::f64::consts::PI * shots as f64)).sqrt())
            .sum::<f64>();
    let tvd_q_c = 0.5
        * dist
            .probabilities
            .iter()
            .zip(&dist_c.probabilities)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();

    // Distance between bosonic and distinguishable statistics, m = n².
    let mut distance = Vec::new();
    for n in 2..=scale.pick(4usize, 3) {
        let m = n * n;
        let mut tvds = Vec::new();
        for k in 0..scale.pick(10u64, 2) {
            let u = photonics::haar_unitary(m, &mut Rng::seed_from_u64(1000 + k))?;
            let mut input = vec![0; m];
            input[..n].iter_mut().for_each(|s| *s = 1);
            let s = BosonSampler::new(u, &input)?;
            let (a, b) = (s.distribution()?, s.distribution_distinguishable()?);
            tvds.push(
                0.5 * a
                    .probabilities
                    .iter()
                    .zip(&b.probabilities)
                    .map(|(x, y)| (x - y).abs())
                    .sum::<f64>(),
            );
        }
        let (mean, std) = mean_std(&tvds);
        distance.push(
            json!({"photons": n, "modes": m, "tvd_mean": mean, "tvd_std": std,
                             "patterns": photonics::count_patterns(n, m) as f64}),
        );
    }

    // Timings.
    let mut permanent_times = Vec::new();
    for n in scale.pick(vec![10usize, 14, 18, 20, 22, 24, 26], vec![10, 14, 18]) {
        let a = metatron_qso::linalg::CMatrix::from_fn(n, n, |_, _| {
            C64::new(rng.normal(), rng.normal())
        });
        let repeats = if n <= 20 { 5 } else { 1 };
        let t = median_seconds(repeats, || {
            photonics::permanent(&a).unwrap();
        });
        permanent_times.push(json!({"n": n, "seconds": t}));
    }
    let mut sample_times = Vec::new();
    for n in scale.pick(vec![4usize, 6, 8, 10, 12, 14], vec![4, 6]) {
        let m = n * n;
        let u = photonics::haar_unitary(m, &mut rng)?;
        let mut input = vec![0; m];
        input[..n].iter_mut().for_each(|s| *s = 1);
        let s = BosonSampler::new(u, &input)?;
        let shots = if n <= 10 { 200 } else { 20 };
        let start = Instant::now();
        s.sample(shots, &mut Rng::seed_from_u64(5))?;
        sample_times.push(json!({"photons": n, "modes": m,
                                 "seconds_per_sample": start.elapsed().as_secs_f64() / shots as f64}));
    }
    Ok(json!({
        "permanent_max_relative_error_n_le_8": max_rel,
        "hom": {"p_coincidence_boson": p_q, "p_coincidence_distinguishable": p_c,
                "visibility": 1.0 - p_q / p_c},
        "full_bunching": bunching,
        "sampler_validation": {
            "modes": m, "photons": n, "patterns": dist.patterns.len(), "shots": shots,
            "tvd_empirical_vs_exact": tvd_samples,
            "tvd_expected_for_exact_sampler": tvd_expected,
            "chi2": chi2, "chi2_cells": dof,
            "tvd_boson_vs_distinguishable": tvd_q_c,
            "normalisation_error": (dist.probabilities.iter().sum::<f64>() - 1.0).abs(),
        },
        "boson_vs_distinguishable": distance,
        "permanent_seconds": permanent_times,
        "sampling_seconds": sample_times,
    }))
}

// ---------------------------------------------------------------------------
// Simulator performance

fn performance(scale: Scale) -> Result<Value> {
    let mut layers = Vec::new();
    for n in scale.pick(vec![10usize, 14, 18, 20, 22, 24], vec![10, 14]) {
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
        layers.push(json!({
            "qubits": n, "gates": layer.len(), "seconds": t,
            "ns_per_gate_amplitude": t / (layer.len() as f64 * (1u64 << n) as f64) * 1e9,
        }));
    }
    let h = models::transverse_field_ising(10, 1.0, 0.7, false);
    let circuit = ansatz::efficient_su2(10, 4, Entanglement::Linear);
    let theta: Vec<f64> = (0..circuit.num_parameters())
        .map(|k| 0.01 * k as f64)
        .collect();
    let mut gradients = Vec::new();
    for (name, m) in [
        ("adjoint", GradientMethod::Adjoint),
        ("parameter shift", GradientMethod::ParameterShift),
        (
            "finite differences",
            GradientMethod::FiniteDifference { step: 1e-6 },
        ),
    ] {
        let t = median_seconds(scale.pick(21, 3), || {
            expectation_and_gradient(&circuit, &h, &theta, None, m).unwrap();
        });
        gradients.push(json!({"method": name, "seconds": t}));
    }
    Ok(json!({
        "statevector_layers": layers,
        "gradients_tfim10_100_parameters": gradients,
    }))
}
