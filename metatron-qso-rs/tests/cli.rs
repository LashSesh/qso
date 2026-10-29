//! The command-line tool, run as a program: every subcommand on a small
//! input, with its JSON output checked against known values. Running the
//! binary also puts the CLI's lines into the coverage report.

#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qso"))
        .args(args)
        .current_dir(root())
        .output()
        .expect("qso runs")
}

/// `qso --json ARGS`, which must succeed.
fn qso(args: &[&str]) -> Value {
    let mut all = vec!["--json"];
    all.extend_from_slice(args);
    let out = run(&all);
    assert!(
        out.status.success(),
        "qso {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("JSON output")
}

/// `qso ARGS` in text mode, which must succeed.
fn qso_text(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "qso {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// `qso ARGS`, which must fail; returns its error output.
fn qso_fails(args: &[&str]) -> String {
    let out = run(args);
    assert!(!out.status.success(), "qso {args:?} succeeded");
    String::from_utf8(out.stderr).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}

/// A scratch file for this test (removed when the test ends).
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let file = format!("qso-cli-{}-{name}", std::process::id());
        Scratch(std::env::temp_dir().join(file))
    }

    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn graphs_symmetry_and_isomorphism() {
    let g = qso(&["graph", "metatron"]);
    assert_eq!(g["automorphism_order"], "6227020800");
    assert_eq!(g["orbits"].as_array().unwrap().len(), 1);
    let lap: Vec<f64> = serde_json::from_value(g["laplacian_spectrum"].clone()).unwrap();
    close(lap[0], 0.0, 1e-9);
    assert!(lap[1..].iter().all(|&x| (x - 13.0).abs() < 1e-9));

    let saved = Scratch::new("j52.graphml");
    qso(&["graph", "johnson:5,2", "--save", saved.path()]);
    let iso = qso(&["iso", saved.path(), "kneser:5,2"]);
    // J(5,2) is the complement of the Petersen graph K(5,2).
    assert_eq!(iso["isomorphic"], false);
    let iso = qso(&["iso", saved.path(), "johnson:5,2"]);
    assert_eq!(iso["isomorphic"], true);
    assert_eq!(iso["mapping"].as_array().unwrap().len(), 10);
    assert!(qso_text(&["graph", "petersen"]).contains("120"));
    assert!(qso_fails(&["graph", "nonsense:3"]).contains("nonsense"));
}

#[test]
fn walks_reduction_and_transfer() {
    let w = qso(&["walk", "--graph", "metatron", "--time", "1"]);
    let limit: Vec<f64> = serde_json::from_value(w["limiting_distribution"].clone()).unwrap();
    close(limit[0], 145.0 / 169.0, 1e-12);
    let p: Vec<f64> = serde_json::from_value(w["probabilities"].clone()).unwrap();
    close(p.iter().sum(), 1.0, 1e-12);

    // The hypercube Q6 from a vertex reduces to the 7 distance classes.
    let r = qso(&["reduce", "--graph", "hypercube:6"]);
    assert_eq!(r["cells"], 7);

    // P3 has perfect state transfer at t = π/√2.
    let t = qso(&["transfer", "--graph", "path:3"]);
    close(
        f(&t["perfect_transfer_time"]),
        std::f64::consts::PI / 2f64.sqrt(),
        1e-6,
    );
    assert_eq!(t["pretty_good"], true);

    let s = qso(&["search", "--graph", "metatron", "--marked", "3"]);
    close(f(&s["success_probability"]), 1.0, 1e-9);
    close(f(&s["gamma"]), 1.0 / 13.0, 1e-6);

    // Deterministic: the value of the library (SKW search on Q5).
    let c = qso(&["coined", "--graph", "hypercube:5"]);
    close(f(&c["success_probability"]), 0.4137588512, 1e-9);
    assert_eq!(c["steps"], 6);
    let h = qso(&["hitting", "--graph", "cycle:16"]);
    assert!(f(&h["quantum"]) < f(&h["classical"]));

    // Two bosons: the Shrikhande and rook's graph look alike without
    // interaction and differ with it.
    let free = qso(&["particles", "--compare", "rook:4x4"]);
    assert_eq!(free["configurations"], 136);
    assert_eq!(free["compare"]["distinguished"], false);
    let d: Vec<f64> = serde_json::from_value(free["densities"].clone()).unwrap();
    close(d.iter().sum(), 2.0, 1e-12);
    let hub = qso(&["particles", "--on-site", "1", "--compare", "rook:4x4"]);
    assert_eq!(hub["compare"]["distinguished"], true);
    let fermions = qso(&[
        "particles",
        "--graph",
        "path:2",
        "--statistics",
        "fermions",
        "--neighbour",
        "0.5",
    ]);
    assert_eq!(fermions["configurations"], 1);
    assert!(qso_fails(&["particles", "--statistics", "anyons"]).contains("statistics"));
    // Different sizes; isomorphic graphs at a large time stay below the
    // tolerance; more bosons than nodes start wrapped around.
    let sizes = qso(&["particles", "--graph", "cycle:5", "--compare", "cycle:6"]);
    assert_eq!(sizes["compare"]["distinguished"], true);
    let late = qso(&[
        "particles",
        "--graph",
        "rook:4x4",
        "--compare",
        "hamming:2,4",
        "--on-site",
        "1",
        "--time",
        "1e6",
    ]);
    assert_eq!(late["compare"]["distinguished"], false);
    let wrapped = qso(&["particles", "--graph", "path:2", "--particles", "3"]);
    assert_eq!(wrapped["start"], json!([0, 1, 0]));
    assert!(
        qso_text(&["particles", "--graph", "path:3", "--particles", "1"]).contains("1 boson on")
    );
}

#[test]
fn algorithms() {
    // QAOA p = 1 on an even ring reaches the ratio 3/4.
    let q = qso(&["qaoa", "--graph", "cycle:6", "--p", "1"]);
    close(f(&q[0]["approximation_ratio"]), 0.75, 1e-6);

    let v = qso(&["vqe", "--model", "h2"]);
    close(f(&v["result"]["energy"]), f(&v["exact_energy"]), 1e-6);
    assert_eq!(v["precision"], "double");
    // The same in single precision, and QAOA by layers.
    let v = qso(&[
        "vqe",
        "--model",
        "tfim:4,1,0.7",
        "--layers",
        "3",
        "--single",
    ]);
    close(f(&v["result"]["energy"]), f(&v["exact_energy"]), 1e-5);
    assert_eq!(v["precision"], "single");
    let q = qso(&[
        "qaoa",
        "--graph",
        "cycle:6",
        "--p",
        "3",
        "--layerwise",
        "--single",
    ]);
    close(f(&q[0]["approximation_ratio"]), 0.75, 1e-5);
    close(f(&q[2]["approximation_ratio"]), 1.0, 1e-5);

    let g = qso(&["grover", "--qubits", "4", "--marked", "3"]);
    close(
        f(&g["success_probability"]),
        f(&g["theoretical_success_probability"]),
        1e-12,
    );
    assert_eq!(g["iterations"], 3);
}

#[test]
fn programs_on_every_backend() {
    let walk = Scratch::new("walk.qasm");
    let c = qso(&["walk-circuit", "--graph", "cycle:8", "--time", "0.7"]);
    std::fs::write(&walk.0, c["qasm"].as_str().unwrap()).unwrap();
    assert_eq!(c["qubits"], 3);

    let total = |r: &Value| -> u64 {
        r["counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|c| c.as_u64().unwrap())
            .sum()
    };
    let sv = qso(&["qasm", walk.path(), "--shots", "400", "--seed", "3"]);
    assert_eq!(total(&sv), 400);
    let mps = qso(&[
        "qasm",
        walk.path(),
        "--shots",
        "400",
        "--seed",
        "3",
        "--mps",
        "--max-bond",
        "8",
    ]);
    assert_eq!(mps["backend"], "mps");
    close(f(&mps["fidelity_lower_bound"]), 1.0, 1e-12);
    let single = qso(&["qasm", walk.path(), "--shots", "400", "--single"]);
    assert_eq!(single["backend"], "statevector-single");
    let noisy = qso(&[
        "qasm",
        walk.path(),
        "--shots",
        "200",
        "--device",
        "examples/devices/toy_device.json",
        "--trajectories",
        "20",
    ]);
    assert_eq!(total(&noisy), 200);

    let teleport = "tests/data/openqasm/v3/teleport.qasm";
    let r = qso(&["qasm", teleport, "--shots", "300"]);
    assert_eq!(r["bits"], 3);
    assert_eq!(total(&r), 300);
    let clifford = qso(&[
        "qasm",
        "tests/data/openqasm/v2/qec.qasm",
        "--stabilizer",
        "--shots",
        "100",
    ]);
    assert_eq!(clifford["backend"], "stabilizer");
    let exported = qso_text(&[
        "qasm",
        "tests/data/openqasm/v2/qec.qasm",
        "--export",
        "qasm3",
    ]);
    assert!(exported.starts_with("OPENQASM 3"));
    assert!(qso_fails(&["qasm", teleport, "--single"]).contains("single"));

    let e = qso(&[
        "estimate",
        walk.path(),
        "--observable",
        "Z0 Z1",
        "--shots",
        "4000",
    ]);
    let est = &e["estimate"];
    assert!((f(&est["value"]) - f(&e["noiseless"])).abs() < 5.0 * f(&est["std_error"]));
    let mitigated = qso(&[
        "estimate",
        walk.path(),
        "--observable",
        "Z0 Z1",
        "--device",
        "examples/devices/toy_device.json",
        "--shots",
        "4000",
        "--mitigate-readout",
        "--zne",
        "1,3",
    ]);
    assert_eq!(mitigated["readout_mitigated"], true);
}

#[test]
fn chemistry_and_dynamics() {
    // RHF of H2 in STO-3G at 1.4 bohr; PySCF 2.14 gives −1.116714325 Ha.
    let h2 = qso(&["scf", "--atoms", "H 0 0 0; H 0 0 0.7408481"]);
    close(f(&h2["energy"]), -1.116714325, 1e-8);
    assert_eq!(h2["method"], "RHF");

    let fcidump = Scratch::new("lih.fcidump");
    let lih = qso(&[
        "scf",
        "--atoms",
        "Li 0 0 0; H 0 0 1.595",
        "--frozen",
        "1",
        "--active",
        "3",
        "--fci",
        "--fcidump",
        fcidump.path(),
    ]);
    assert!(f(&lih["fci"]) < f(&lih["energy"]));
    let m = qso(&["molecule", fcidump.path(), "--method", "none"]);
    close(f(&m["fci"][0]), f(&lih["fci"]), 1e-8);

    let o2 = qso(&[
        "scf",
        "--atoms",
        "O 0 0 0; O 0 0 1.2075",
        "--multiplicity",
        "3",
        "--restricted-open",
    ]);
    assert_eq!(o2["method"], "ROHF");
    let hcl = qso(&[
        "scf",
        "--atoms",
        "H 0 0 0; Cl 0 0 1.2746",
        "--basis",
        "6-31g*",
        "--cartesian",
    ]);
    assert_eq!(hcl["harmonics"], "cartesian");

    let h = qso(&["molecule", "examples/molecules/h2_sto3g.fcidump"]);
    close(f(&h["variational"]["energy"]), -1.1373060357533997, 1e-8);
    close(f(&h["fci"][0]), -1.1373060357533997, 1e-8);

    let out = Scratch::new("trotter.qasm");
    let t = qso(&[
        "trotter",
        "--hamiltonian",
        "Z0 Z1 + 0.5 X0 + 0.5 X1",
        "--steps",
        "20",
        "--order",
        "4",
        "--output",
        out.path(),
    ]);
    assert!(f(&t["column_error"]) <= f(&t["error_bound"]));
    // Order 4 at 20 steps: the nested bound is the smallest.
    assert_eq!(f(&t["bounds"]["nested"]), f(&t["error_bound"]));
    assert!(f(&t["bounds"]["stages"]) > 10.0 * f(&t["error_bound"]));
    assert!(
        std::fs::read_to_string(&out.0)
            .unwrap()
            .starts_with("OPENQASM 2.0")
    );
}

#[test]
fn spin_dynamics() {
    let root = env!("CARGO_MANIFEST_DIR");
    let ch = format!("{root}/examples/spins/ch.json");
    let inept = format!("{root}/examples/spins/inept.json");
    let r = qso(&["spins", &ch, "--observe", "13C", "--sequence", &inept]);
    let lines = r["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 2);
    // INEPT: antiphase lines of about γ_H/γ_C ± 1 times the direct ones.
    let direct = qso(&["spins", &ch, "--observe", "13C"]);
    let im = |v: &serde_json::Value| f(&v["amplitude"][1]);
    let d = im(&direct["lines"][0]).abs();
    assert!((im(&lines[0]).abs() / d - 4.976).abs() < 0.01);
    assert!((im(&lines[1]).abs() / d - 2.976).abs() < 0.01);
    let para = format!("{root}/examples/spins/parahydrogen.json");
    let p = qso(&["spins", &para, "--initial", "pasadena:0,1", "--pulse", "45"]);
    let lines = p["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 4);
    for l in lines {
        close(im(l).abs(), 0.125, 4e-4);
    }
    // A sequence that ends at another field is observed there.
    let jump = Scratch::new("jump.json");
    std::fs::write(
        jump.path(),
        r#"[{"field": 4.7}, {"pulse": {"target": {"isotope": "13C"}, "angle": 90.0, "phase": 0.0}}]"#,
    )
    .unwrap();
    let at = qso(&["spins", &ch, "--observe", "13C", "--sequence", jump.path()]);
    assert_eq!(f(&at["field"]), 4.7);
    let freq: Vec<f64> = at["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| f(&l["frequency"]))
        .collect();
    assert_eq!(freq.len(), 2);
    // ¹³C at 60 ppm and 4.7 T: −γB·60·10⁻⁶/2π, split by J = 140 Hz.
    let nu = -6.728_284e7 * 4.7 * 60e-6 / (2.0 * std::f64::consts::PI);
    assert!(((freq[0] + freq[1]) / 2.0 - nu).abs() < 1e-6, "{freq:?}");
    assert!((freq[1] - freq[0] - 140.0).abs() < 1e-6, "{freq:?}");
    // Redfield relaxation of formate: T1 of ¹³C from its dipolar coupling
    // to the proton (Solomon) and its shielding anisotropy.
    let formate = format!("{root}/examples/spins/formate.json");
    let r = qso(&["spins", &formate, "--observe", "13C", "--rates"]);
    let t1 = f(&r["relaxation_times"][0]["t1"]);
    // Without relaxation there is no time.
    let none = qso(&["spins", &ch, "--observe", "13C", "--rates"]);
    assert!(none["relaxation_times"][0]["t1"].is_null());
    let (gc, gh, tau, field) = (6.728_284e7, 26.752_212_8e7, 2e-11, 9.4);
    let j = |w: f64| tau / (1.0 + w * w * tau * tau);
    let b = 1.000_000_000_55e-7 * gc * gh * 1.054_571_817e-34 / 1.10e-10f64.powi(3);
    let (wc, wh) = (gc * field * (1.0 + 171e-6), gh * field * (1.0 + 8.4e-6));
    let dipolar = b * b / 10.0 * (j(wh - wc) + 3.0 * j(wc) + 6.0 * j(wh + wc));
    let w0 = gc * field;
    let csa = w0 * w0 / 5.0 * 9950e-12 * j(wc);
    assert!(
        (t1 * (dipolar + csa) - 1.0).abs() < 1e-8,
        "{t1} vs {}",
        1.0 / (dipolar + csa)
    );
}

#[test]
fn error_correction() {
    let stim = Scratch::new("surface.stim");
    let r = qso(&[
        "qec",
        "--distance",
        "3,5",
        "--p",
        "0.001",
        "--shots",
        "2000",
        "--stim",
        stim.path(),
    ]);
    assert_eq!(r.as_array().unwrap().len(), 2);
    assert_eq!(r[0]["decoder"], "union-find");
    let m = qso(&[
        "qec",
        "--circuit",
        stim.path(),
        "--shots",
        "2000",
        "--decoder",
        "matching",
    ]);
    assert!(f(&m["logical_error_rate"]["rate"]) < 0.05);
    let c = qso(&[
        "qec",
        "--circuit",
        "tests/data/stim/surface_d5_memory_z.stim",
        "--shots",
        "500",
        "--decoder",
        "correlated-matching",
        "--circuit-distance",
    ]);
    assert_eq!(c["circuit_distance"]["lower_bound"], 5);
    assert_eq!(c["circuit_distance"]["upper_bound"], 5);
    let colour = qso(&[
        "qec",
        "--code",
        "color",
        "--distance",
        "3",
        "--p",
        "0.002",
        "--shots",
        "500",
        "--decoder",
        "bposd",
    ]);
    assert_eq!(colour[0]["decoder"], "bp-osd");
    assert!(
        qso_fails(&[
            "qec",
            "--circuit",
            stim.path(),
            "--schedule",
            "bicycle",
            "--shots",
            "10"
        ])
        .contains("schedule")
    );
}
