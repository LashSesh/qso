//! `qso` — command-line front end for Metatron QSO.
//!
//! Every command prints a human-readable summary, or JSON with `--json`.

use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};

use metatron_qso::algorithms::ansatz::{self, Entanglement};
use metatron_qso::algorithms::qaoa::{MaxCutQaoa, QaoaConfig, optimize_layerwise};
use metatron_qso::algorithms::time_evolution::{trotter_circuit, trotter_error_bound};
use metatron_qso::algorithms::{grover, models, vqe::Vqe};
use metatron_qso::backend::{Backend, DensityMatrixBackend, DeviceBackend, StatevectorBackend};
use metatron_qso::chemistry::adapt::{AdaptOptions, adapt_vqe_molecule};
use metatron_qso::chemistry::ansatz as chem_ansatz;
use metatron_qso::chemistry::{
    AoIntegrals, Basis, Mapping, MolecularIntegrals, Molecule, QubitEncoding, ScfOptions, rhf, uhf,
};
use metatron_qso::estimation::{Estimate, EstimatorOptions, ShotAllocation, estimate};
use metatron_qso::graph::{Graph, generators, io};
use metatron_qso::mitigation::ReadoutCalibration;
use metatron_qso::mitigation::pec::pec_expectation;
use metatron_qso::mitigation::zne::{Extrapolation, Folding, zne};
use metatron_qso::noise::{DeviceModel, NoiseModel};
use metatron_qso::optimize::{Method, Options};
use metatron_qso::qec;
use metatron_qso::quantum::{
    Circuit, PauliSum, Program, bitstring, from_qasm2, record_bitstring, to_qasm2,
};
use metatron_qso::rng::Rng;
use metatron_qso::stabilizer::StabilizerState;
use metatron_qso::stabilizer::program::{is_clifford_program, sample_program};
use metatron_qso::walks::{
    Coin, CoinedWalk, ContinuousWalk, Generator, PrettyGoodTransfer, Quotient, QuotientWalk,
    SpatialSearch, StateTransfer, SzegedyWalk, circuits, classical_hitting_times,
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

#[derive(Clone, Copy, ValueEnum)]
enum QasmVersion {
    #[value(name = "qasm2")]
    Qasm2,
    #[value(name = "qasm3")]
    Qasm3,
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
    /// Perfect state transfer between two nodes, or the best transfer;
    /// pretty good state transfer decided exactly on paths and cycles.
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
    /// Simulate an OpenQASM 2 or 3 program and sample its classical bits
    /// (all qubits if it has none). Mid-circuit measurements, resets and
    /// conditions are simulated exactly; with --noise or --device the
    /// program must be a circuit (OpenQASM 2 measurements are then replaced
    /// by measuring every qubit at the end).
    Qasm {
        file: std::path::PathBuf,
        /// Print the program as OpenQASM 2 or 3 instead of running it.
        #[arg(long, value_enum, conflicts_with_all = ["noise", "device"])]
        export: Option<QasmVersion>,
        #[arg(long, default_value_t = 1024)]
        shots: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Depolarising noise "p1,p2" (density-matrix backend).
        #[arg(long, value_delimiter = ',')]
        noise: Option<Vec<f64>>,
        /// Device description (JSON): gate errors, T1/T2, readout errors.
        #[arg(long, conflicts_with = "noise")]
        device: Option<std::path::PathBuf>,
        /// Simulate the device with this many quantum trajectories instead
        /// of density matrices (any register size).
        #[arg(long, requires = "device")]
        trajectories: Option<usize>,
        /// Run on the stabilizer simulator (Clifford programs only; chosen
        /// automatically for Clifford programs too large for a state vector).
        #[arg(long, conflicts_with_all = ["noise", "device"])]
        stabilizer: bool,
        /// Run on a matrix-product state (any program; exact up to the
        /// reported fidelity bound).
        #[arg(long, conflicts_with_all = ["noise", "device", "stabilizer"])]
        mps: bool,
        /// Largest bond dimension of the matrix-product state.
        #[arg(long, default_value_t = 256, requires = "mps")]
        max_bond: usize,
        /// Run on a single-precision state vector (half the memory, one
        /// more qubit; programs without classical bits).
        #[arg(long, conflicts_with_all = ["noise", "device", "stabilizer", "mps"])]
        single: bool,
    },
    /// Estimate ⟨H⟩ of an OpenQASM circuit from shots, with standard
    /// errors and optional error mitigation.
    Estimate {
        file: std::path::PathBuf,
        /// Observable, e.g. "Z0 Z1 + 0.5 X2".
        #[arg(long)]
        observable: String,
        /// Total shots per estimate.
        #[arg(long, default_value_t = 10000)]
        shots: usize,
        #[arg(long, value_enum, default_value_t = AllocationArg::Weighted)]
        allocation: AllocationArg,
        /// Device description (JSON); without it the circuit runs noiselessly.
        #[arg(long)]
        device: Option<std::path::PathBuf>,
        /// Simulate the device with quantum trajectories (at most this many
        /// per run) instead of density matrices.
        #[arg(long, requires = "device")]
        trajectories: Option<usize>,
        /// Mitigate readout errors with a tensored calibration.
        #[arg(long, requires = "device")]
        mitigate_readout: bool,
        /// Zero-noise extrapolation from these scale factors, e.g. 1,3,5.
        #[arg(long, value_delimiter = ',')]
        zne: Option<Vec<f64>>,
        #[arg(long, value_enum, default_value_t = ExtrapolationArg::Richardson)]
        extrapolation: ExtrapolationArg,
        /// Probabilistic error cancellation with this many samples (Pauli
        /// noise only; each sample is evaluated exactly, so --shots,
        /// --trajectories and readout errors do not apply to it).
        #[arg(long, requires = "device")]
        pec: Option<usize>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Hartree–Fock of a molecule from its geometry (STO-3G or 6-31G,
    /// H to Ne), optionally full CI in an active space and an FCIDUMP file.
    Scf {
        /// Atoms in ångström, e.g. "H 0 0 0; H 0 0 0.735".
        #[arg(long)]
        atoms: String,
        #[arg(long, default_value = "sto-3g")]
        basis: String,
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        charge: i32,
        /// Spin multiplicity 2S + 1.
        #[arg(long, default_value_t = 1)]
        multiplicity: u32,
        /// Unrestricted Hartree–Fock (open shells).
        #[arg(long)]
        unrestricted: bool,
        /// Doubly occupied orbitals folded into the core energy (with
        /// --fci or --fcidump).
        #[arg(long, default_value_t = 0)]
        frozen: usize,
        /// Orbitals kept after the frozen ones (default: all; with --fci or
        /// --fcidump).
        #[arg(long)]
        active: Option<usize>,
        /// Most self-consistent-field iterations.
        #[arg(long, default_value_t = 200)]
        max_iterations: usize,
        /// Full CI in the frozen-core active space.
        #[arg(long)]
        fci: bool,
        /// Write the active-space Hamiltonian as an FCIDUMP file.
        #[arg(long)]
        fcidump: Option<std::path::PathBuf>,
    },
    /// Molecular Hamiltonian from an FCIDUMP file: qubit Hamiltonian,
    /// Hartree–Fock and full-CI energies, and a variational solution.
    Molecule {
        file: std::path::PathBuf,
        #[arg(long, value_enum, default_value_t = MappingArg::JordanWigner)]
        mapping: MappingArg,
        /// Remove two qubits by symmetry (parity mapping only).
        #[arg(long)]
        two_qubit_reduction: bool,
        /// Variational method: none, uccsd, upccgsd (with --k layers) or adapt.
        #[arg(long, value_enum, default_value_t = ChemMethodArg::Uccsd)]
        method: ChemMethodArg,
        #[arg(long, default_value_t = 1)]
        k: usize,
        /// Number of full-CI roots to report.
        #[arg(long, default_value_t = 1)]
        roots: usize,
    },
    /// Trotter–Suzuki circuit for exp(−iHt) with its error bound.
    Trotter {
        /// Hamiltonian, e.g. "Z0 Z1 + 0.5 X0 + 0.5 X1".
        #[arg(long)]
        hamiltonian: String,
        /// Register size (default: smallest that fits).
        #[arg(long)]
        qubits: Option<usize>,
        #[arg(long, default_value_t = 1.0)]
        time: f64,
        #[arg(long, default_value_t = 10)]
        steps: usize,
        /// 1, 2, 4, 6, …
        #[arg(long, default_value_t = 2)]
        order: usize,
        /// Write the circuit as OpenQASM 2.0.
        #[arg(long)]
        output: Option<std::path::PathBuf>,
    },
    /// Memory experiment of a repetition, rotated surface, triangular
    /// colour or bivariate bicycle code under circuit-level depolarizing
    /// noise: sampled with Pauli frames and decoded by union-find,
    /// minimum-weight matching or BP+OSD (colour and bicycle codes need
    /// BP+OSD).
    Qec {
        #[arg(long, value_enum, default_value_t = CodeArg::Surface)]
        code: CodeArg,
        #[arg(long, value_enum, default_value_t = DecoderArg::UnionFind)]
        decoder: DecoderArg,
        /// Basis of the memory (not for the repetition code): z protects
        /// |0⟩_L, x |+⟩_L.
        #[arg(long, value_enum, default_value_t = BasisArg::Z)]
        basis: BasisArg,
        /// Code distances (odd for the surface and colour codes), e.g.
        /// 3,5,7; for bivariate bicycle codes their qubit counts (72, 90,
        /// 108, 144, 288).
        #[arg(long, value_delimiter = ',', default_value = "3,5")]
        distance: Vec<usize>,
        /// Rounds of syndrome extraction (default: the distance).
        #[arg(long)]
        rounds: Option<usize>,
        /// Noise probabilities, e.g. 0.001,0.01.
        #[arg(long, value_delimiter = ',', default_value = "0.001")]
        p: Vec<f64>,
        #[arg(long, default_value_t = 100_000)]
        shots: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Write the circuit (first distance and probability) in Stim's
        /// text format.
        #[arg(long)]
        stim: Option<std::path::PathBuf>,
        /// Decode a circuit in Stim's text format instead of a generated
        /// memory experiment (code, distance, rounds and p are ignored).
        #[arg(long, conflicts_with = "stim")]
        circuit: Option<std::path::PathBuf>,
        /// Order of the CNOTs of the colour and bicycle codes: greedy,
        /// bicycle (the depth-8 cycle of Bravyi et al.), or six directions
        /// for the colour code, e.g. 0,1,2,3,4,5 (not with --circuit).
        #[arg(long, default_value = "greedy")]
        schedule: String,
        /// Also bound the circuit distance: the fewest faults that flip a
        /// logical observable undetected.
        #[arg(long)]
        circuit_distance: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum BasisArg {
    Z,
    X,
}

#[derive(Clone, Copy, ValueEnum)]
enum DecoderArg {
    UnionFind,
    Matching,
    CorrelatedMatching,
    Bposd,
}

#[derive(Clone, Copy, ValueEnum)]
enum CodeArg {
    Surface,
    Repetition,
    Color,
    Bb,
}

#[derive(Clone, Copy, ValueEnum)]
enum MappingArg {
    JordanWigner,
    Parity,
    BravyiKitaev,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum ChemMethodArg {
    None,
    Uccsd,
    Upccgsd,
    Adapt,
}

#[derive(Clone, Copy, ValueEnum)]
enum AllocationArg {
    Uniform,
    Weighted,
    Adaptive,
}

#[derive(Clone, Copy, ValueEnum)]
enum ExtrapolationArg {
    Linear,
    Richardson,
    Exponential,
}

fn read_device(path: &std::path::Path) -> Result<DeviceModel> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| QsoError::Parse(format!("cannot read {}: {e}", path.display())))?;
    DeviceModel::from_json(&text)
}

/// The backend for an optional device file (density matrices, or
/// trajectories when given).
fn device_backend(
    device: Option<&DeviceModel>,
    trajectories: Option<usize>,
) -> Result<Box<dyn Backend>> {
    Ok(match (device, trajectories) {
        (Some(d), Some(t)) => Box::new(DeviceBackend::trajectories(d.clone(), t)?),
        (Some(d), None) => Box::new(DeviceBackend::density_matrix(d.clone())?),
        (None, _) => Box::new(StatevectorBackend),
    })
}

fn estimate_json(e: &Estimate) -> Value {
    let (lo, hi) = e.confidence_interval(0.95);
    json!({"value": e.value, "std_error": e.std_error, "samples": e.samples, "ci95": [lo, hi]})
}

fn estimate_text(e: &Estimate) -> String {
    let (lo, hi) = e.confidence_interval(0.95);
    format!(
        "{:.6} ± {:.6} (95% CI [{lo:.6}, {hi:.6}], {} samples)",
        e.value, e.std_error, e.samples
    )
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
        // A file that is not a specification reports its own parse errors.
        Err(_) if std::path::Path::new(spec).is_file() => {
            read_graph_file(std::path::Path::new(spec))
        }
        result => result,
    }
}

/// A circuit from an OpenQASM 2 or 3 program: its gates if it has only
/// gates; OpenQASM 2 programs keep their reading with measurements dropped.
fn circuit_from_qasm(src: &str) -> Result<Circuit> {
    let program = Program::from_qasm(src)?;
    if program.is_unitary() {
        return program.to_circuit();
    }
    from_qasm2(src).map_err(|_| {
        QsoError::invalid(
            "program",
            "this needs a circuit: an OpenQASM 3 program without measurements, resets or \
             conditions, or an OpenQASM 2 program without resets or conditions",
        )
    })
}

fn read_text(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path)
        .map_err(|e| QsoError::Parse(format!("cannot read {}: {e}", path.display())))
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
            // Exact PGST decision where the graph family allows it; perfect
            // transfer implies it on every graph.
            let exact = PrettyGoodTransfer::from_graph(&g, (*generator).into(), *gamma, *from, to);
            // Paths and cycles are recognised before any arithmetic, so an
            // error on one means the decision itself failed.
            let family = g.is_connected()
                && g.is_unweighted()
                && g.degrees().iter().all(|&d| d <= 2)
                && g.num_edges() + 1 >= g.num_nodes();
            let pretty_good = exact
                .as_ref()
                .ok()
                .map(|p| p.pretty_good())
                .or(t.perfect_transfer_time().map(|_| true));
            let pgst_line = match (pretty_good, &exact) {
                (Some(true), _) => "\npretty good state transfer: yes".to_string(),
                (Some(false), _) => "\npretty good state transfer: no (exact)".to_string(),
                (None, Err(e)) if family => {
                    format!("\npretty good state transfer: undecided ({e})")
                }
                (None, _) => {
                    "\npretty good state transfer: undecided (exact only for paths and cycles)"
                        .to_string()
                }
            };
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
                       "pretty_good": pretty_good, "best": best}),
                text + &pgst_line,
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
            export,
            shots,
            seed,
            noise,
            device,
            trajectories,
            stabilizer,
            mps,
            max_bond,
            single,
        } => {
            let src = read_text(file)?;
            let program = Program::from_qasm(&src)?;
            if let Some(version) = export {
                let text = match version {
                    QasmVersion::Qasm2 => program.to_qasm2()?,
                    QasmVersion::Qasm3 => program.to_qasm3()?,
                };
                return Ok((json!({ "qasm": text }), text.trim_end().to_string()));
            }
            if noise.is_none() && device.is_none() {
                // Exact simulation of the program; one without classical
                // bits is measured on every qubit.
                if *single && program.n_bits() != 0 {
                    return Err(QsoError::invalid(
                        "single",
                        "single precision samples programs without classical bits (every qubit is measured at the end)",
                    ));
                }
                let tableau = *stabilizer
                    || (!*mps
                        && !*single
                        && program.n_qubits() > metatron_qso::quantum::MAX_QUBITS
                        && is_clifford_program(&program));
                let mut bound = None;
                let labelled: std::collections::BTreeMap<String, usize> = if *mps {
                    let options = metatron_qso::mps::MpsOptions {
                        max_bond: *max_bond,
                        ..Default::default()
                    };
                    if program.n_bits() == 0 {
                        let mut state = metatron_qso::mps::Mps::from_circuit(
                            &program.to_circuit()?,
                            &[],
                            options,
                        )?;
                        bound = state.fidelity_lower_bound();
                        state.sample_counts(*shots, &mut Rng::seed_from_u64(*seed))?
                    } else {
                        let (counts, b) = metatron_qso::mps::program::sample_program(
                            &program, *shots, *seed, options,
                        )?;
                        bound = b;
                        counts
                            .iter()
                            .map(|(&r, &c)| (record_bitstring(r, program.n_bits()), c))
                            .collect()
                    }
                } else if *single {
                    let state = metatron_qso::quantum::StateVector32::from_circuit(
                        &program.to_circuit()?,
                        &[],
                    )?;
                    state
                        .sample(*shots, &mut Rng::seed_from_u64(*seed))
                        .iter()
                        .map(|(&b, &c)| {
                            (metatron_qso::quantum::bitstring(b, program.n_qubits()), c)
                        })
                        .collect()
                } else if tableau {
                    if program.n_bits() == 0 {
                        let state = StabilizerState::from_circuit(&program.to_circuit()?, &[])?;
                        state.sample_counts(*shots, &mut Rng::seed_from_u64(*seed))?
                    } else {
                        sample_program(&program, *shots, *seed)?
                            .iter()
                            .map(|(&r, &c)| (record_bitstring(r, program.n_bits()), c))
                            .collect()
                    }
                } else {
                    let (run, width) = if program.n_bits() == 0 {
                        let c = program.to_circuit()?;
                        (Program::from_circuit(&c, &[], true)?, program.n_qubits())
                    } else {
                        (program.clone(), program.n_bits())
                    };
                    run.sample(*shots, *seed)?
                        .iter()
                        .map(|(&r, &c)| (record_bitstring(r, width), c))
                        .collect()
                };
                let backend = if *mps {
                    "mps"
                } else if *single {
                    "statevector-single"
                } else if tableau {
                    "stabilizer"
                } else {
                    "program"
                };
                let mut sorted: Vec<_> = labelled.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1));
                let text = format!(
                    "{} qubits, {} bits, {} operations, backend {backend}{}\n{}",
                    program.n_qubits(),
                    program.n_bits(),
                    program.len(),
                    bound.map_or(String::new(), |b| format!(", fidelity ≥ {b:.12}")),
                    sorted
                        .iter()
                        .take(16)
                        .map(|(b, c)| format!("  {b}  {c}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                return Ok((
                    json!({"qubits": program.n_qubits(), "bits": program.n_bits(),
                           "operations": program.len(), "backend": backend, "counts": labelled,
                           "fidelity_lower_bound": bound}),
                    text,
                ));
            }
            let circuit = circuit_from_qasm(&src)?;
            let device = device.as_deref().map(read_device).transpose()?;
            // Here a device or depolarising noise is given.
            let (backend_name, counts) = match (&device, noise.as_deref()) {
                (Some(_), _) => {
                    let backend = device_backend(device.as_ref(), *trajectories)?;
                    let counts = backend.run(&circuit, &[], *shots, *seed)?;
                    (backend.name().to_string(), counts)
                }
                (None, Some(&[p1, p2])) => {
                    let backend = DensityMatrixBackend::new(NoiseModel::depolarizing(p1, p2)?);
                    let counts = backend.run(&circuit, &[], *shots, *seed)?;
                    ("density-matrix".to_string(), counts)
                }
                _ => {
                    return Err(QsoError::invalid(
                        "noise",
                        "expected two probabilities p1,p2",
                    ));
                }
            };
            let n = circuit.n_qubits();
            let labelled: std::collections::BTreeMap<String, usize> =
                counts.iter().map(|(&b, &c)| (bitstring(b, n), c)).collect();
            let mut sorted: Vec<_> = labelled.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let text = format!(
                "{} qubits, {} gates, depth {}, backend {backend_name}\n{}",
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
                json!({"qubits": n, "gates": circuit.len(), "depth": circuit.depth(), "backend": backend_name, "counts": labelled}),
                text,
            ))
        }
        Command::Scf {
            atoms,
            basis,
            charge,
            multiplicity,
            unrestricted,
            frozen,
            active,
            fci,
            fcidump,
            max_iterations,
        } => {
            if (*frozen > 0 || active.is_some()) && !*fci && fcidump.is_none() {
                return Err(QsoError::invalid(
                    "frozen",
                    "--frozen and --active choose the space of --fci or --fcidump",
                ));
            }
            let mol = Molecule::from_atoms(atoms, *charge, *multiplicity)?;
            if !*unrestricted {
                let (na, nb) = mol.n_alpha_beta();
                if na != nb {
                    return Err(QsoError::invalid(
                        "multiplicity",
                        "an open shell needs --unrestricted",
                    ));
                }
            }
            let ints = AoIntegrals::compute(&mol, Basis::from_name(basis)?)?;
            let options = ScfOptions {
                max_iterations: *max_iterations,
                ..ScfOptions::default()
            };
            let hf = if *unrestricted {
                uhf(&mol, &ints, &options)?
            } else {
                rhf(&mol, &ints, &options)?
            };
            let (na, nb) = mol.n_alpha_beta();
            let mut lines = vec![
                format!(
                    "{} basis functions, {na} + {nb} electrons",
                    ints.n_functions()
                ),
                format!(
                    "{}   {:.10}  ({} iterations, <S²> = {:.6})",
                    if *unrestricted { "UHF" } else { "RHF" },
                    hf.energy,
                    hf.iterations,
                    hf.s_squared()
                ),
            ];
            let mut fci_energy = Value::Null;
            if *fci || fcidump.is_some() {
                let mo = hf.molecular_integrals(&ints, *frozen, *active)?;
                if *fci {
                    let e = mo.fci_energies(1)?[0];
                    lines.push(format!(
                        "full CI ({} orbitals, {} + {} electrons)   {e:.10}  (correlation {:.10})",
                        mo.n_orbitals(),
                        mo.n_alpha(),
                        mo.n_beta(),
                        e - hf.energy
                    ));
                    fci_energy = json!(e);
                }
                if let Some(path) = fcidump {
                    std::fs::write(path, mo.to_fcidump()).map_err(|e| {
                        QsoError::invalid("fcidump", format!("{}: {e}", path.display()))
                    })?;
                    lines.push(format!("wrote {}", path.display()));
                }
            }
            Ok((
                json!({
                    "basis_functions": ints.n_functions(),
                    "electrons": [na, nb],
                    "restricted": !unrestricted,
                    "energy": hf.energy,
                    "nuclear_repulsion": hf.nuclear_repulsion,
                    "iterations": hf.iterations,
                    "s_squared": hf.s_squared(),
                    "orbital_energies": hf.orbital_energies,
                    "fci": fci_energy,
                }),
                lines.join("\n"),
            ))
        }
        Command::Molecule {
            file,
            mapping,
            two_qubit_reduction,
            method,
            k,
            roots,
        } => {
            let text = std::fs::read_to_string(file)
                .map_err(|e| QsoError::invalid("file", format!("{}: {e}", file.display())))?;
            let mol = MolecularIntegrals::from_fcidump(&text)?;
            let mapping = match mapping {
                MappingArg::JordanWigner => Mapping::JordanWigner,
                MappingArg::Parity => Mapping::Parity,
                MappingArg::BravyiKitaev => Mapping::BravyiKitaev,
            };
            if *two_qubit_reduction && mapping != Mapping::Parity {
                return Err(QsoError::invalid(
                    "two-qubit-reduction",
                    "requires --mapping parity",
                ));
            }
            let encoding = QubitEncoding {
                mapping,
                two_qubit_reduction: *two_qubit_reduction,
            };
            let h = mol.qubit_hamiltonian(encoding)?;
            let n = encoding.n_qubits(mol.n_spin_orbitals());
            let hf = mol.hartree_fock_energy();
            let fci = mol.fci_energies(*roots)?;
            let mut lines = vec![
                format!(
                    "{} orbitals, {} + {} electrons; {} qubits ({}{}), {} Pauli terms",
                    mol.n_orbitals(),
                    mol.n_alpha(),
                    mol.n_beta(),
                    n,
                    mapping.name(),
                    if *two_qubit_reduction {
                        ", two-qubit reduction"
                    } else {
                        ""
                    },
                    h.num_terms()
                ),
                format!("Hartree–Fock   {hf:.10}"),
                format!(
                    "full CI        {}",
                    fci.iter()
                        .map(|e| format!("{e:.10}"))
                        .collect::<Vec<_>>()
                        .join("  ")
                ),
            ];
            let mut variational = Value::Null;
            if *method != ChemMethodArg::None {
                if n > 20 {
                    return Err(QsoError::invalid(
                        "method",
                        format!("{n} qubits are too many for a variational run here"),
                    ));
                }
                let (label, energy, parameters, extra) = match method {
                    ChemMethodArg::Adapt => {
                        let r =
                            adapt_vqe_molecule(&mol, encoding, false, &AdaptOptions::default())?;
                        (
                            "ADAPT-VQE".to_string(),
                            r.energy,
                            r.parameters.len(),
                            json!({"operators": r.operators, "converged": r.converged}),
                        )
                    }
                    _ => {
                        let (label, ansatz) = if *method == ChemMethodArg::Uccsd {
                            ("UCCSD".to_string(), chem_ansatz::uccsd(&mol, encoding)?)
                        } else {
                            (
                                format!("{k}-UpCCGSD"),
                                chem_ansatz::k_upccgsd(&mol, *k, encoding)?,
                            )
                        };
                        let p = ansatz.num_parameters();
                        let r = Vqe::new(h.clone(), ansatz)?
                            .with_initial_parameters(vec![0.0; p])?
                            .run()?;
                        (label, r.energy, p, json!({"converged": r.converged}))
                    }
                };
                lines.push(format!(
                    "{label:<14} {energy:.10}  ({parameters} parameter{}, error vs full CI {:.2e})",
                    if parameters == 1 { "" } else { "s" },
                    energy - fci[0]
                ));
                variational = json!({"method": label, "energy": energy, "parameters": parameters, "details": extra});
            }
            Ok((
                json!({
                    "orbitals": mol.n_orbitals(),
                    "electrons": [mol.n_alpha(), mol.n_beta()],
                    "qubits": n,
                    "mapping": mapping.name(),
                    "two_qubit_reduction": two_qubit_reduction,
                    "pauli_terms": h.num_terms(),
                    "hartree_fock": hf,
                    "fci": fci,
                    "variational": variational,
                }),
                lines.join("\n"),
            ))
        }
        Command::Qec {
            code,
            decoder,
            basis,
            distance,
            rounds,
            p,
            shots,
            seed,
            stim,
            circuit: from_file,
            schedule,
            circuit_distance,
        } => {
            let kind = match decoder {
                DecoderArg::UnionFind => qec::DecoderChoice::UnionFind,
                DecoderArg::Matching => qec::DecoderChoice::Matching,
                DecoderArg::CorrelatedMatching => qec::DecoderChoice::CorrelatedMatching,
                DecoderArg::Bposd => qec::DecoderChoice::BpOsd,
            };
            let decoder_name = match decoder {
                DecoderArg::UnionFind => "union-find",
                DecoderArg::Matching => "matching",
                DecoderArg::CorrelatedMatching => "correlated matching",
                DecoderArg::Bposd => "BP+OSD",
            };
            let css_schedule = match schedule.as_str() {
                "greedy" => qec::CssSchedule::Greedy,
                "bicycle" => qec::CssSchedule::Bicycle,
                list => {
                    let order: Vec<u8> = list
                        .split(',')
                        .map(|x| x.trim().parse::<u8>())
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|_| {
                            QsoError::invalid(
                                "schedule",
                                format!("{list:?}: use greedy, bicycle or six directions"),
                            )
                        })?;
                    let order: [u8; 6] = order.try_into().map_err(|_| {
                        QsoError::invalid(
                            "schedule",
                            "a schedule by directions lists six directions",
                        )
                    })?;
                    qec::CssSchedule::Directions(order)
                }
            };
            if css_schedule != qec::CssSchedule::Greedy
                && (from_file.is_some() || !matches!(code, CodeArg::Color | CodeArg::Bb))
            {
                return Err(QsoError::invalid(
                    "schedule",
                    "schedules are for the colour and bicycle codes",
                ));
            }
            // Bounds on the circuit distance, as JSON and text.
            let bound = |c: &qec::DetectorCircuit| -> Result<(serde_json::Value, String)> {
                if !*circuit_distance {
                    return Ok((serde_json::Value::Null, String::new()));
                }
                let r = qec::circuit_distance(
                    &qec::detector_error_model(c, false)?,
                    &qec::DistanceOptions::default(),
                )?;
                let text = match (r.distance(), r.upper_bound()) {
                    (Some(d), _) => format!("; circuit distance {d}"),
                    (None, Some(u)) => format!("; circuit distance {} to {u}", r.lower_bound),
                    (None, None) if r.lower_bound == usize::MAX => {
                        "; no undetectable logical error".to_string()
                    }
                    (None, None) => format!("; circuit distance at least {}", r.lower_bound),
                };
                let lower = (r.lower_bound != usize::MAX).then_some(r.lower_bound);
                Ok((
                    json!({"lower_bound": lower, "upper_bound": r.upper_bound()}),
                    text,
                ))
            };
            let memory_basis = match basis {
                BasisArg::Z => qec::MemoryBasis::Z,
                BasisArg::X => qec::MemoryBasis::X,
            };
            if let Some(path) = from_file {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    QsoError::invalid("circuit", format!("{}: {e}", path.display()))
                })?;
                let circuit = qec::DetectorCircuit::from_stim(&text)?;
                let rate = qec::logical_error_rate_using(&circuit, *shots, *seed, kind)?;
                let (distance_json, distance_text) = bound(&circuit)?;
                let line = format!(
                    "{}: {} qubits, {} detectors, {} observables; logical error rate {:.3e} ± {:.1e} ({} of {} shots, {decoder_name}){distance_text}",
                    path.display(),
                    circuit.n_qubits(),
                    circuit.n_detectors(),
                    circuit.n_observables(),
                    rate.rate,
                    rate.standard_error,
                    rate.errors,
                    rate.shots
                );
                return Ok((
                    json!({
                        "circuit": path.display().to_string(),
                        "qubits": circuit.n_qubits(),
                        "detectors": circuit.n_detectors(),
                        "observables": circuit.n_observables(),
                        "decoder": kind,
                        "logical_error_rate": rate,
                        "circuit_distance": distance_json,
                    }),
                    line,
                ));
            }
            let mut rows = Vec::new();
            let mut lines = Vec::new();
            for &prob in p {
                for &d in distance {
                    let noise = qec::CircuitNoise::uniform(prob);
                    // The bicycle codes are named by their qubits.
                    let (bicycle, code_distance) = match code {
                        CodeArg::Bb => {
                            let (c, cd) = qec::CssCode::bivariate_bicycle_code(d)?;
                            (Some(c), cd)
                        }
                        _ => (None, d),
                    };
                    let r = rounds.unwrap_or(code_distance);
                    let circuit = match code {
                        CodeArg::Surface => {
                            qec::surface_code_memory_with(d, r, &noise, memory_basis)?
                        }
                        CodeArg::Repetition => qec::repetition_code_memory(d, r, &noise)?,
                        CodeArg::Color => qec::css_code_memory_with(
                            &qec::CssCode::color_code(d)?,
                            r,
                            &noise,
                            memory_basis,
                            css_schedule,
                        )?,
                        CodeArg::Bb => qec::css_code_memory_with(
                            bicycle.as_ref().expect("built above"),
                            r,
                            &noise,
                            memory_basis,
                            css_schedule,
                        )?,
                    };
                    let label = match &bicycle {
                        Some(c) => format!("[[{}, {}, {code_distance}]]", c.n(), c.k()),
                        None => format!("d = {d}"),
                    };
                    if let Some(path) = stim.as_ref().filter(|_| rows.is_empty()) {
                        std::fs::write(path, circuit.to_stim()?).map_err(|e| {
                            QsoError::invalid("stim", format!("{}: {e}", path.display()))
                        })?;
                    }
                    let rate = qec::logical_error_rate_using(&circuit, *shots, *seed, kind)?;
                    let (distance_json, distance_text) = bound(&circuit)?;
                    lines.push(format!(
                        "p = {prob}, {label}, {r} rounds: {} qubits, {} detectors; logical error rate {:.3e} ± {:.1e} ({} of {} shots, {}){distance_text}",
                        circuit.n_qubits(),
                        circuit.n_detectors(),
                        rate.rate,
                        rate.standard_error,
                        rate.errors,
                        rate.shots,
                        decoder_name
                    ));
                    rows.push(json!({
                        "p": prob,
                        "distance": code_distance,
                        "code_qubits": bicycle.as_ref().map(|c| c.n()),
                        "rounds": r,
                        "qubits": circuit.n_qubits(),
                        "detectors": circuit.n_detectors(),
                        "decoder": kind,
                        "schedule": css_schedule,
                        "logical_error_rate": rate,
                        "circuit_distance": distance_json,
                    }));
                }
            }
            Ok((json!(rows), lines.join("\n")))
        }
        Command::Trotter {
            hamiltonian,
            qubits,
            time,
            steps,
            order,
            output,
        } => {
            let h: PauliSum = hamiltonian.parse()?;
            let n = qubits.unwrap_or(h.min_qubits().max(1));
            let c = trotter_circuit(&h, n, *time, *steps, *order)?;
            let bound = trotter_error_bound(&h, *time, *steps, *order)?;
            // The actual error when the unitary is small enough to form.
            let actual = if n <= 10 {
                let traceless = PauliSum::from_terms(
                    h.terms().iter().filter(|(_, s)| !s.is_identity()).copied(),
                );
                let exact = metatron_qso::linalg::expm_hermitian(&traceless.to_dense(n)?, *time);
                let mut worst: f64 = 0.0;
                for col in 0..1usize << n {
                    let mut psi = metatron_qso::quantum::StateVector::basis(n, col)?;
                    psi.apply_circuit(&c, &[])?;
                    let diff: f64 = psi
                        .amplitudes()
                        .iter()
                        .enumerate()
                        .map(|(row, a)| (a - exact[(row, col)]).norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    worst = worst.max(diff);
                }
                Some(worst)
            } else {
                None
            };
            if let Some(path) = output {
                std::fs::write(path, to_qasm2(&c, &[], false)?)
                    .map_err(|e| QsoError::invalid("output", format!("{}: {e}", path.display())))?;
            }
            let text = format!(
                "order {order}, {steps} steps, t = {time}: {} gates, depth {}, error bound {bound:.3e}{}",
                c.len(),
                c.depth(),
                actual.map_or(String::new(), |a| format!(
                    ", largest column error {a:.3e} (a lower bound on ‖U − exp(−iHt)‖)"
                ))
            );
            Ok((
                json!({"qubits": n, "gates": c.len(), "depth": c.depth(), "error_bound": bound, "column_error": actual}),
                text,
            ))
        }
        Command::Estimate {
            file,
            observable,
            shots,
            allocation,
            device,
            trajectories,
            mitigate_readout,
            zne: scales,
            extrapolation,
            pec,
            seed,
        } => {
            let circuit = circuit_from_qasm(&read_text(file)?)?;
            let h: PauliSum = observable.parse()?;
            let device = device.as_deref().map(read_device).transpose()?;
            let backend = device_backend(device.as_ref(), *trajectories)?;
            let n = circuit.n_qubits();
            // PEC first: it fails fast when the noise is not Pauli noise.
            let pec_result = match pec {
                Some(samples) => {
                    let d = device.as_ref().expect("clap requires --device");
                    Some(pec_expectation(&circuit, &[], d, &h, *samples, *seed)?)
                }
                None => None,
            };
            let mut options = EstimatorOptions::new(*shots)
                .with_allocation(match allocation {
                    AllocationArg::Uniform => ShotAllocation::Uniform,
                    AllocationArg::Weighted => ShotAllocation::Weighted,
                    AllocationArg::Adaptive => ShotAllocation::Adaptive { pilot: 0.1 },
                })
                .with_seed(*seed);
            if *mitigate_readout {
                let cal =
                    ReadoutCalibration::tensored(backend.as_ref(), n, *shots, *seed ^ 0x5eed)?;
                options = options.with_readout_mitigation(cal);
            }
            let raw = estimate(backend.as_ref(), &circuit, &[], &h, &options)?.estimate;
            let mut value = json!({
                "qubits": n,
                "backend": backend.name(),
                "shots": shots,
                "seed": seed,
                "allocation": format!("{:?}", options.allocation),
                "readout_mitigated": mitigate_readout,
                "calibration_shots": if *mitigate_readout { 2 * shots } else { 0 },
                "estimate": estimate_json(&raw),
            });
            let label = if *mitigate_readout {
                "<H> (readout mitigated)"
            } else {
                "<H>"
            };
            let mut text = format!("{label} = {}", estimate_text(&raw));
            if n <= 24 {
                let exact = h.expectation(&metatron_qso::quantum::StateVector::from_circuit(
                    &circuit,
                    &[],
                )?)?;
                value["noiseless"] = json!(exact);
                text.push_str(&format!("\nnoiseless <H> = {exact:.6}"));
            }
            if let Some(scales) = scales {
                let method = match extrapolation {
                    ExtrapolationArg::Linear => Extrapolation::Linear,
                    ExtrapolationArg::Richardson => Extrapolation::Richardson,
                    ExtrapolationArg::Exponential => Extrapolation::Exponential { asymptote: 0.0 },
                };
                let mut k = 0;
                let r = zne(&circuit, scales, Folding::Global, method, |folded| {
                    k += 1;
                    let opts = options.clone().with_seed(seed.wrapping_add(k));
                    Ok(estimate(backend.as_ref(), folded, &[], &h, &opts)?.estimate)
                })?;
                value["zne"] = json!({
                    "estimate": estimate_json(&r.estimate),
                    "scales": r.scales,
                    "values": r.values.iter().map(estimate_json).collect::<Vec<_>>(),
                });
                text.push_str(&format!(
                    "\nZNE ({:?}) <H> = {}",
                    r.scales,
                    estimate_text(&r.estimate)
                ));
            }
            if let Some(r) = pec_result {
                value["pec"] = json!({
                    "estimate": estimate_json(&r.estimate),
                    "overhead": r.overhead,
                    "noisy_operations": r.noisy_operations,
                });
                text.push_str(&format!(
                    "\nPEC (overhead {:.3}) <H> = {}",
                    r.overhead,
                    estimate_text(&r.estimate)
                ));
            }
            Ok((value, text))
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
