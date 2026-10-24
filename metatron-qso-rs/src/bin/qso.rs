//! `qso` — command-line front end for Metatron QSO.
//!
//! Every command prints a human-readable summary, or JSON with `--json`.

use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};

use metatron_qso::algorithms::ansatz::{self, Entanglement};
use metatron_qso::algorithms::qaoa::{MaxCutQaoa, QaoaConfig, optimize_layerwise};
use metatron_qso::algorithms::{grover, models, vqe::Vqe};
use metatron_qso::backend::{Backend, DensityMatrixBackend, StatevectorBackend};
use metatron_qso::graph::{Graph, generators, io};
use metatron_qso::noise::NoiseModel;
use metatron_qso::optimize::{Method, Options};
use metatron_qso::quantum::{PauliSum, bitstring, from_qasm2, to_qasm2};
use metatron_qso::walks::{
    Coin, CoinedWalk, ContinuousWalk, Generator, Quotient, QuotientWalk, SpatialSearch,
    StateTransfer, SzegedyWalk, circuits, classical_hitting_times,
};
use metatron_qso::{QsoError, Result};

#[derive(Parser)]
#[command(
    name = "qso",
    version,
    about = "Validated quantum simulation: circuits, walks, VQE, QAOA"
)]
struct Cli {
    /// Emit machine-readable JSON instead of text.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum GeneratorArg {
    Laplacian,
    Adjacency,
}

impl From<GeneratorArg> for Generator {
    fn from(g: GeneratorArg) -> Self {
        match g {
            GeneratorArg::Laplacian => Generator::Laplacian,
            GeneratorArg::Adjacency => Generator::Adjacency,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Structure, spectrum and symmetry group of a graph.
    ///
    /// Every GRAPH argument is a specification (metatron, cycle:8,
    /// johnson:7,3, …) or a file: an edge list, or GraphML (.graphml,
    /// .xml); `file:PATH` forces a file.
    Graph {
        /// Graph specification, e.g. metatron, fruit-of-life, cycle:8, platonic:dodecahedron.
        #[arg(default_value = "metatron")]
        spec: String,
        /// Write the graph as an edge list (or GraphML for .graphml/.xml).
        #[arg(long)]
        save: Option<std::path::PathBuf>,
    },
    /// Decide whether two graphs are isomorphic (canonical labelling).
    Iso { first: String, second: String },
    /// Reduce a walk by symmetry: the coarsest equitable partition in which
    /// the start node is a cell of its own, and the walk on it.
    Reduce {
        #[arg(long, default_value = "metatron")]
        graph: String,
        #[arg(long, default_value_t = 0)]
        start: usize,
        #[arg(long, default_value_t = 1.0)]
        time: f64,
        #[arg(long, default_value_t = 1.0)]
        gamma: f64,
        #[arg(long, value_enum, default_value_t = GeneratorArg::Laplacian)]
        generator: GeneratorArg,
    },
    /// Perfect state transfer between two nodes, or the best transfer.
    Transfer {
        #[arg(long, default_value = "path:3")]
        graph: String,
        #[arg(long, default_value_t = 0)]
        from: usize,
        /// Target node (default: the last node).
        #[arg(long)]
        to: Option<usize>,
        #[arg(long, default_value_t = 1.0)]
        gamma: f64,
        #[arg(long, value_enum, default_value_t = GeneratorArg::Adjacency)]
        generator: GeneratorArg,
        /// Window for the best transfer when there is no PST.
        #[arg(long, default_value_t = 50.0)]
        t_max: f64,
    },
    /// Coined discrete-time search (Shenvi–Kempe–Whaley).
    Coined {
        #[arg(long, default_value = "hypercube:10")]
        graph: String,
        #[arg(long, value_delimiter = ',', default_value = "0")]
        marked: Vec<usize>,
        /// grover, fourier or hadamard.
        #[arg(long, default_value = "grover")]
        coin: String,
        /// Search window in steps (default: ⌈π√N⌉).
        #[arg(long)]
        steps: Option<usize>,
    },
    /// Szegedy quantum hitting time versus the classical hitting time.
    Hitting {
        #[arg(long, default_value = "complete:64")]
        graph: String,
        #[arg(long, value_delimiter = ',', default_value = "0")]
        marked: Vec<usize>,
        #[arg(long, default_value_t = 100_000)]
        max_steps: usize,
    },
    /// Export a walk exp(−iHt) as an OpenQASM 2.0 circuit.
    WalkCircuit {
        #[arg(long, default_value = "cycle:8")]
        graph: String,
        #[arg(long, default_value_t = 1.0)]
        time: f64,
        #[arg(long, default_value_t = 1.0)]
        gamma: f64,
        #[arg(long, value_enum, default_value_t = GeneratorArg::Laplacian)]
        generator: GeneratorArg,
        /// exact (circulant or Z_2^d Cayley graphs) or trotter.
        #[arg(long, default_value = "exact")]
        method: String,
        /// Product-formula steps.
        #[arg(long, default_value_t = 10)]
        steps: usize,
        /// Product-formula order (1 or 2).
        #[arg(long, default_value_t = 2)]
        order: usize,
    },
    /// Continuous-time quantum walk probabilities.
    Walk {
        #[arg(long, default_value = "metatron")]
        graph: String,
        #[arg(long, default_value_t = 0)]
        start: usize,
        #[arg(long, default_value_t = 1.0)]
        time: f64,
        #[arg(long, default_value_t = 1.0)]
        gamma: f64,
        #[arg(long, value_enum, default_value_t = GeneratorArg::Laplacian)]
        generator: GeneratorArg,
    },
    /// Quantum-walk spatial search (Childs–Goldstone).
    Search {
        #[arg(long, default_value = "metatron")]
        graph: String,
        /// Comma-separated marked nodes.
        #[arg(long, value_delimiter = ',', default_value = "0")]
        marked: Vec<usize>,
        /// Hopping rate (default: optimised around the critical rate).
        #[arg(long)]
        gamma: Option<f64>,
        #[arg(long, value_enum, default_value_t = GeneratorArg::Laplacian)]
        generator: GeneratorArg,
        /// Search window (default: π√N).
        #[arg(long)]
        t_max: Option<f64>,
    },
    /// QAOA for MaxCut.
    Qaoa {
        #[arg(long, default_value = "metatron")]
        graph: String,
        #[arg(long, default_value_t = 1)]
        p: usize,
        #[arg(long, default_value_t = 8)]
        restarts: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Optimise depths 1..=p with INTERP initialisation.
        #[arg(long)]
        layerwise: bool,
    },
    /// Variational quantum eigensolver.
    Vqe {
        /// h2, tfim:N,J,H or heisenberg:N,J,DELTA.
        #[arg(long, default_value = "h2")]
        model: String,
        /// Arbitrary Pauli Hamiltonian, e.g. "0.5 Z0 Z1 - X0" (overrides --model).
        #[arg(long)]
        hamiltonian: Option<String>,
        /// Register size for --hamiltonian (default: smallest that fits).
        #[arg(long)]
        qubits: Option<usize>,
        #[arg(long, default_value_t = 2)]
        layers: usize,
        /// lbfgs, adam, nelder-mead, spsa, gradient-descent.
        #[arg(long, default_value = "lbfgs")]
        method: String,
        #[arg(long, default_value_t = 4)]
        restarts: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Grover search.
    Grover {
        #[arg(long, default_value_t = 8)]
        qubits: usize,
        #[arg(long, value_delimiter = ',', default_value = "5")]
        marked: Vec<usize>,
        #[arg(long)]
        iterations: Option<usize>,
    },
    /// Simulate an OpenQASM 2.0 file and sample measurement counts.
    Qasm {
        file: std::path::PathBuf,
        #[arg(long, default_value_t = 1024)]
        shots: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Depolarising noise "p1,p2" (density-matrix backend).
        #[arg(long, value_delimiter = ',')]
        noise: Option<Vec<f64>>,
    },
}

fn read_graph_file(path: &std::path::Path) -> Result<Graph> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| QsoError::Parse(format!("cannot read {}: {e}", path.display())))?;
    let lower = path.to_string_lossy().to_ascii_lowercase();
    let g = if lower.ends_with(".graphml") || lower.ends_with(".xml") {
        io::from_graphml(&text)?
    } else {
        io::from_edge_list(&text)?
    };
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
    Ok(g.with_name(name.unwrap_or_default()))
}

/// A graph from a specification string or a file.
fn graph(spec: &str) -> Result<Graph> {
    if let Some(path) = spec.strip_prefix("file:") {
        return read_graph_file(std::path::Path::new(path));
    }
    match generators::from_spec(spec) {
        Err(e) if std::path::Path::new(spec).is_file() => {
            read_graph_file(std::path::Path::new(spec)).map_err(|_| e)
        }
        result => result,
    }
}

fn vector(v: &[f64]) -> String {
    // Print tiny values (e.g. −1e−16 from a zero eigenvalue) as exact zeros.
    v.iter()
        .map(|&x| format!("{:.6}", if x.abs() < 5e-13 { 0.0 } else { x }))
        .collect::<Vec<_>>()
        .join(" ")
}

fn model_hamiltonian(model: &str) -> Result<(PauliSum, usize, f64)> {
    let (kind, args) = model.split_once(':').unwrap_or((model, ""));
    let nums: Vec<f64> = args
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| QsoError::Parse(format!("bad model '{model}'")))
        })
        .collect::<Result<_>>()?;
    match (kind, nums.as_slice()) {
        ("h2", []) => {
            let h2 = models::h2_sto3g();
            Ok((h2.hamiltonian, 2, h2.nuclear_repulsion))
        }
        ("tfim", [n, j, h]) => Ok((
            models::transverse_field_ising(*n as usize, *j, *h, false),
            *n as usize,
            0.0,
        )),
        ("heisenberg", [n, j, d]) => Ok((
            models::heisenberg_xxz(*n as usize, *j, *d, false),
            *n as usize,
            0.0,
        )),
        _ => Err(QsoError::Parse(format!(
            "unknown model '{model}' (use h2, tfim:N,J,H or heisenberg:N,J,DELTA)"
        ))),
    }
}

fn run(cli: &Cli) -> Result<(Value, String)> {
    match &cli.command {
        Command::Graph { spec, save } => {
            let g = graph(spec)?;
            if let Some(path) = save {
                let lower = path.to_string_lossy().to_ascii_lowercase();
                let text = if lower.ends_with(".graphml") || lower.ends_with(".xml") {
                    io::to_graphml(&g)
                } else {
                    io::to_edge_list(&g)
                };
                std::fs::write(path, text).map_err(|e| {
                    QsoError::Parse(format!("cannot write {}: {e}", path.display()))
                })?;
            }
            let summary = g.summary();
            let group = g.automorphism_group();
            let lap = g.laplacian_spectrum();
            let order = group.order().map(|o| o.to_string());
            let srg = g.strongly_regular_parameters();
            let mut text = format!(
                "{name}\n  nodes {n}, edges {e}, degree {dmin}..{dmax}, connected {c}, bipartite {b}\n  diameter {diam:?}, triangles {t}, algebraic connectivity {ac:.6}\n  |Aut| = {ord} (log10 {log:.3}), orbits {orbits}\n  Laplacian spectrum: {spec}",
                name = summary.name,
                n = summary.nodes,
                e = summary.edges,
                dmin = summary.min_degree,
                dmax = summary.max_degree,
                c = summary.connected,
                b = summary.bipartite,
                diam = summary.diameter,
                t = summary.triangles,
                ac = summary.algebraic_connectivity,
                ord = order.clone().unwrap_or_else(|| "> u128".into()),
                log = group.log10_order(),
                orbits = group.orbits().len(),
                spec = vector(&lap),
            );
            if let Some((n, k, l, m)) = srg {
                text.push_str(&format!("\n  strongly regular ({n}, {k}, {l}, {m})"));
            }
            Ok((
                json!({"summary": summary, "automorphism_order": order, "log10_order": group.log10_order(),
                       "orbits": group.orbits(), "laplacian_spectrum": lap, "strongly_regular": srg}),
                text,
            ))
        }
        Command::Iso { first, second } => {
            let (g, h) = (graph(first)?, graph(second)?);
            let iso = g.isomorphism(&h);
            let text = match &iso {
                Some(map) => format!("{} ≅ {}\n  mapping {:?}", g.name(), h.name(), map),
                None => format!("{} and {} are not isomorphic", g.name(), h.name()),
            };
            Ok((json!({"isomorphic": iso.is_some(), "mapping": iso}), text))
        }
        Command::Reduce {
            graph: spec,
            start,
            time,
            gamma,
            generator,
        } => {
            let g = graph(spec)?;
            let q = Quotient::with_singletons(&g, &[*start])?;
            let cell = q.cell_of().expect("explicit graph")[*start];
            let walk = QuotientWalk::new(&q, (*generator).into(), *gamma)?;
            let p = walk.cell_probabilities(cell, *time)?;
            let limit = walk.limiting_distribution(cell)?;
            let text = format!(
                "{}: {} nodes reduce to {} cells (start in cell {cell})\n  sizes {:?}\n  reduced spectrum {}\n  P(cell, t={time}) {}\n  limit {}",
                g.name(),
                g.num_nodes(),
                q.num_cells(),
                q.sizes(),
                vector(walk.spectrum()),
                vector(&p),
                vector(&limit)
            );
            Ok((
                json!({"cells": q.num_cells(), "sizes": q.sizes(), "cell_of": q.cell_of(),
                       "start_cell": cell, "spectrum": walk.spectrum(), "probabilities": p,
                       "limiting_distribution": limit}),
                text,
            ))
        }
        Command::Transfer {
            graph: spec,
            from,
            to,
            gamma,
            generator,
            t_max,
        } => {
            let g = graph(spec)?;
            let to = to.unwrap_or(g.num_nodes().saturating_sub(1));
            let t = StateTransfer::new(&g, (*generator).into(), *gamma, *from, to)?;
            let best = t.best_transfer(*t_max, 4000)?;
            let text = match t.perfect_transfer_time() {
                Some(tau) => format!(
                    "{}: perfect state transfer {from} → {to} at t = {tau:.10}",
                    g.name()
                ),
                None => format!(
                    "{}: no perfect state transfer {from} → {to} (strongly cospectral: {}); best fidelity {:.6} at t = {:.6} within [0, {t_max}]",
                    g.name(),
                    t.strongly_cospectral(),
                    best.fidelity,
                    best.time
                ),
            };
            Ok((
                json!({"perfect_transfer_time": t.perfect_transfer_time(),
                       "strongly_cospectral": t.strongly_cospectral(), "support": t.support(),
                       "best": best}),
                text,
            ))
        }
        Command::Coined {
            graph: spec,
            marked,
            coin,
            steps,
        } => {
            let g = graph(spec)?;
            let coin = match coin.to_ascii_lowercase().as_str() {
                "grover" => Coin::Grover,
                "fourier" => Coin::Fourier,
                "hadamard" => Coin::Hadamard,
                other => {
                    return Err(QsoError::invalid(
                        "coin",
                        format!("unknown coin '{other}' (grover, fourier or hadamard)"),
                    ));
                }
            };
            let steps = steps
                .unwrap_or((std::f64::consts::PI * (g.num_nodes() as f64).sqrt()).ceil() as usize);
            let r = CoinedWalk::new(&g, coin)?
                .with_marked(marked)?
                .search(steps)?;
            let text = format!(
                "coined search on {} for {:?}: success {:.6} after {} steps (N = {})",
                g.name(),
                marked,
                r.success_probability,
                r.steps,
                g.num_nodes()
            );
            Ok((
                json!({"steps": r.steps, "success_probability": r.success_probability}),
                text,
            ))
        }
        Command::Hitting {
            graph: spec,
            marked,
            max_steps,
        } => {
            let g = graph(spec)?;
            let w = SzegedyWalk::new(&g, marked)?;
            let quantum = w.hitting_time(*max_steps);
            // Classical hitting time from the stationary distribution, for
            // a single marked node.
            let classical = match marked.as_slice() {
                [m] => {
                    let h = classical_hitting_times(&g, *m)?;
                    let total: f64 = (0..g.num_nodes()).map(|u| g.weighted_degree(u)).sum();
                    Some(
                        (0..g.num_nodes())
                            .map(|u| g.weighted_degree(u) / total * h[u])
                            .sum::<f64>(),
                    )
                }
                _ => None,
            };
            let text = format!(
                "{}: Szegedy quantum hitting time {}, classical hitting time {}",
                g.name(),
                quantum.map_or("not reached".to_string(), |t| t.to_string()),
                classical.map_or("(single marked node only)".to_string(), |c| format!(
                    "{c:.4}"
                ))
            );
            Ok((json!({"quantum": quantum, "classical": classical}), text))
        }
        Command::WalkCircuit {
            graph: spec,
            time,
            gamma,
            generator,
            method,
            steps,
            order,
        } => {
            let g = graph(spec)?;
            let generator: Generator = (*generator).into();
            let c = match method.as_str() {
                "exact" => circuits::circulant_walk(&g, generator, *gamma, *time)
                    .or_else(|_| circuits::cayley_walk(&g, generator, *gamma, *time))
                    .map_err(|_| {
                        QsoError::invalid(
                            "method",
                            "exact circuits need a circulant graph on 2^n nodes or a Cayley graph of Z_2^n; use --method trotter",
                        )
                    })?,
                "trotter" => circuits::trotter_walk(&g, generator, *gamma, *time, *steps, *order)?,
                other => {
                    return Err(QsoError::invalid(
                        "method",
                        format!("unknown method '{other}' (exact or trotter)"),
                    ));
                }
            };
            let qasm = to_qasm2(&c, &[], false)?;
            Ok((
                json!({"qubits": c.n_qubits(), "gates": c.len(), "qasm": qasm}),
                qasm,
            ))
        }
        Command::Walk {
            graph: spec,
            start,
            time,
            gamma,
            generator,
        } => {
            let g = graph(spec)?;
            let walk = ContinuousWalk::new(&g, (*generator).into(), *gamma)?;
            let p = walk.probabilities(*start, *time)?;
            let limit = walk.limiting_distribution(*start).ok();
            let mut text = format!(
                "P(node, t={time}) from node {start} on {}:\n  {}",
                g.name(),
                vector(&p)
            );
            if let Some(l) = &limit {
                text.push_str(&format!("\nlimiting distribution:\n  {}", vector(l)));
            }
            Ok((
                json!({"graph": g.name(), "time": time, "probabilities": p, "limiting_distribution": limit}),
                text,
            ))
        }
        Command::Search {
            graph: spec,
            marked,
            gamma,
            generator,
            t_max,
        } => {
            let g = graph(spec)?;
            let t_max = t_max.unwrap_or(std::f64::consts::PI * (g.num_nodes() as f64).sqrt());
            let r = match gamma {
                Some(x) => SpatialSearch::new(&g, marked, (*generator).into(), *x)?
                    .optimize_time(t_max, 2000),
                None => SpatialSearch::optimize(&g, marked, (*generator).into(), t_max, 1000)?,
            };
            let text = format!(
                "search on {} for {:?}: gamma {:.6}, best time {:.6}, success {:.6} (expected time with repetition {:.4})",
                g.name(),
                marked,
                r.gamma,
                r.time,
                r.success_probability,
                r.expected_time
            );
            Ok((serde_json::to_value(&r).expect("serialisable"), text))
        }
        Command::Qaoa {
            graph: spec,
            p,
            restarts,
            seed,
            layerwise,
        } => {
            let g = graph(spec)?;
            let config = QaoaConfig {
                restarts: *restarts,
                seed: *seed,
                ..Default::default()
            };
            let results = if *layerwise {
                optimize_layerwise(&g, *p, &config)?
            } else {
                vec![MaxCutQaoa::new(&g, *p)?.optimize(&config)?]
            };
            let n = g.num_nodes();
            let text = results
                .iter()
                .map(|r| {
                    format!(
                        "p={}  <C>={:.6}  C_max={}  ratio={:.6}  P(optimal)={:.4}  most likely {} (cut {})",
                        r.p,
                        r.expectation,
                        r.max_cut,
                        r.approximation_ratio.unwrap_or(f64::NAN),
                        r.optimal_probability,
                        bitstring(r.most_likely, n),
                        r.most_likely_cut
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok((serde_json::to_value(&results).expect("serialisable"), text))
        }
        Command::Vqe {
            model,
            hamiltonian,
            qubits,
            layers,
            method,
            restarts,
            seed,
        } => {
            let (h, n, offset) = match hamiltonian {
                Some(expr) => {
                    let h: PauliSum = expr.parse()?;
                    let n = qubits.unwrap_or(h.min_qubits().max(1));
                    (h, n, 0.0)
                }
                None => model_hamiltonian(model)?,
            };
            let circuit = ansatz::efficient_su2(n, *layers, Entanglement::Linear);
            let vqe = Vqe::new(h, circuit)?
                .with_method(Method::from_name(method)?)
                .with_options(Options {
                    max_iterations: 5000,
                    ..Default::default()
                })
                .with_restarts(*restarts)
                .with_seed(*seed);
            let exact = vqe.exact_energy()?;
            let r = vqe.run()?;
            let text = format!(
                "VQE ({method}, {n} qubits, {layers} layers): E = {:.10}  exact = {:.10}  error = {:.3e}{}",
                r.energy,
                exact,
                r.energy - exact,
                if offset != 0.0 {
                    format!(
                        "  total (with nuclear repulsion) = {:.10}",
                        r.energy + offset
                    )
                } else {
                    String::new()
                }
            );
            Ok((
                json!({"result": r, "exact_energy": exact, "constant_offset": offset}),
                text,
            ))
        }
        Command::Grover {
            qubits,
            marked,
            iterations,
        } => {
            let r = grover::grover_search(*qubits, marked, *iterations)?;
            let text = format!(
                "Grover on {} qubits, {} marked, {} iterations: success {:.6} (theory {:.6}), {} gates",
                r.n_qubits,
                r.marked.len(),
                r.iterations,
                r.success_probability,
                r.theoretical_success_probability,
                r.gates
            );
            Ok((serde_json::to_value(&r).expect("serialisable"), text))
        }
        Command::Qasm {
            file,
            shots,
            seed,
            noise,
        } => {
            let src = std::fs::read_to_string(file)
                .map_err(|e| QsoError::Parse(format!("cannot read {}: {e}", file.display())))?;
            let circuit = from_qasm2(&src)?;
            let counts = match noise {
                Some(p) if p.len() == 2 => DensityMatrixBackend::new(NoiseModel::depolarizing(
                    p[0], p[1],
                )?)
                .run(&circuit, &[], *shots, *seed)?,
                Some(_) => {
                    return Err(QsoError::invalid(
                        "noise",
                        "expected two probabilities p1,p2",
                    ));
                }
                None => StatevectorBackend.run(&circuit, &[], *shots, *seed)?,
            };
            let n = circuit.n_qubits();
            let labelled: std::collections::BTreeMap<String, usize> =
                counts.iter().map(|(&b, &c)| (bitstring(b, n), c)).collect();
            let mut sorted: Vec<_> = labelled.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let text = format!(
                "{} qubits, {} gates, depth {}\n{}",
                n,
                circuit.len(),
                circuit.depth(),
                sorted
                    .iter()
                    .take(16)
                    .map(|(b, c)| format!("  {b}  {c}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            Ok((
                json!({"qubits": n, "gates": circuit.len(), "depth": circuit.depth(), "counts": labelled}),
                text,
            ))
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok((value, text)) => {
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).expect("valid JSON")
                );
            } else {
                println!("{text}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
