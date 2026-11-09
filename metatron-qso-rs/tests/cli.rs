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
fn walks_with_site_energies() {
    // The detuned dimer H = [[3, −1], [−1, 0]]: Rabi's formula gives 4/13
    // on node 1 at t = π/√13, and the walker spends 2/13 of the time there.
    let t = (std::f64::consts::PI / 13f64.sqrt()).to_string();
    for (generator, energies) in [
        ("adjacency", "3,0"),
        ("laplacian", "3,0"),
        ("adjacency", "-3,0"),
    ] {
        let w = qso(&[
            "walk",
            "--graph",
            "path:2",
            "--generator",
            generator,
            "--energies",
            energies,
            "--time",
            &t,
        ]);
        let p: Vec<f64> = serde_json::from_value(w["probabilities"].clone()).unwrap();
        close(p[1], 4.0 / 13.0, 1e-12);
        let limit: Vec<f64> = serde_json::from_value(w["limiting_distribution"].clone()).unwrap();
        close(limit[0], 11.0 / 13.0, 1e-12);
        close(limit[1], 2.0 / 13.0, 1e-12);
        let e: Vec<f64> = serde_json::from_value(w["site_energies"].clone()).unwrap();
        assert_eq!(e.len(), 2);
    }
    let plain = qso(&["walk", "--graph", "path:2"]);
    assert_eq!(plain["site_energies"], Value::Null);
    assert!(
        qso_text(&["walk", "--graph", "cycle:4", "--energies", "0.5,0,0,0"])
            .contains("with site energies")
    );
    assert!(qso_fails(&["walk", "--graph", "cycle:4", "--energies", "1,2"]).contains('4'));
}

#[test]
fn transport_to_a_trap() {
    // The detuned dimer (energies 3, 0; loss 0.05 on both nodes; trap κ = 1
    // on node 1; dephasing 1): η = 32000/38881 and τ = 139260/38881, with
    // either generator (they differ by a constant on the diagonal).
    let reference = [
        "transport",
        "--graph",
        "path:2",
        "--start",
        "0",
        "--trap",
        "1",
        "--kappa",
        "1",
        "--loss",
        "0.05",
        "--dephasing",
        "1",
        "--energies",
        "3,0",
    ];
    for generator in ["adjacency", "laplacian"] {
        let mut args = reference.to_vec();
        args.extend(["--generator", generator]);
        let r = qso(&args);
        close(f(&r["efficiency"]), 0.823024099174, 1e-10);
        close(f(&r["arrival_time"]), 3.581698001595, 1e-10);
        close(f(&r["efficiency"]), 32000.0 / 38881.0, 1e-13);
        close(f(&r["lost"]), 6881.0 / 38881.0, 1e-13);
        close(f(&r["dark"]), 0.0, 1e-13);
        assert!(f(&r["error"]) <= 1e-12);
        assert_eq!(r["dimension"], 2);
        assert_eq!(r["generator"], generator);
        assert_eq!(r["traps"], json!([[1, 1.0]]));
        assert_eq!(r["losses"], json!([0.05, 0.05]));
        assert_eq!(r["dephasing"], json!([1.0, 1.0]));
        assert_eq!(r["scan"], Value::Null);
        assert_eq!(r["best"], Value::Null);
        assert!(!r["solves"].as_array().unwrap().is_empty());
    }
    let text = qso_text(&reference);
    assert!(text.contains("transport from node 0 to trap 1 on"));
    assert!(text.contains("efficiency 0.823024 (lost 0.176976, never absorbed 0)"));
    assert!(text.contains("mean arrival time 3.581698"));
    assert!(text.contains("bright dimension 2"));

    // The scan over a rate added to every node reproduces the exact
    // rationals; the best rate is 39/20 with η* = 800/963.
    let dimer = [
        "transport",
        "--trap",
        "1",
        "--loss",
        "0.05",
        "--energies",
        "3,0",
    ];
    let mut args = dimer.to_vec();
    args.extend(["--scan", "0,1,3,10,100", "--best-rate", "0.01,100"]);
    let r = qso(&args);
    let scan = r["scan"].as_array().unwrap();
    let want = [
        (0.0, 224000.0 / 290427.0, 9200540.0 / 2032989.0),
        (1.0, 32000.0 / 38881.0, 139260.0 / 38881.0),
        (3.0, 288000.0 / 348689.0, 100414940.0 / 28243809.0),
        (10.0, 7072000.0 / 9575681.0, 11335156860.0 / 2116225501.0),
        (
            100.0,
            64672000.0 / 235515281.0,
            7074070324860.0 / 475976382901.0,
        ),
    ];
    assert_eq!(scan.len(), want.len());
    for (s, (rate, eta, tau)) in scan.iter().zip(want) {
        assert_eq!(f(&s["rate"]), rate);
        close(f(&s["efficiency"]), eta, 1e-12);
        close(f(&s["arrival_time"]), tau, 1e-10 * tau);
    }
    close(f(&r["efficiency"]), 224000.0 / 290427.0, 1e-12);
    let best = &r["best"];
    close(f(&best["rate"]), 1.95, 1e-5);
    close(f(&best["efficiency"]), 800.0 / 963.0, 1e-12);
    assert_eq!(best["interior"], true);
    let mut args = dimer.to_vec();
    args.extend(["--best-rate", "0.01,100"]);
    assert!(qso_text(&args).contains("best rate 1.950000: efficiency 0.830737"));
    // With its own dephasing 1 the best rate added is 0.95.
    args.extend(["--dephasing", "1"]);
    let r = qso(&args);
    close(f(&r["best"]["rate"]), 0.95, 1e-5);
    assert!(qso_text(&args).contains("best added rate 0.950000: efficiency 0.830737"));
    close(f(&r["best"]["efficiency"]), 800.0 / 963.0, 1e-12);

    // K5 without loss from node 0 to node 1: a quarter arrives, the rest
    // stays dark; two traps with their own rates are reported per trap.
    let k5 = qso(&["transport", "--graph", "complete:5", "--trap", "1"]);
    close(f(&k5["efficiency"]), 0.25, 1e-12);
    close(f(&k5["dark"]), 0.75, 1e-12);
    close(f(&k5["arrival_time"]), 9.0 / 4.0, 1e-9);
    let two = qso(&[
        "transport",
        "--graph",
        "cycle:6",
        "--trap",
        "2,4",
        "--kappa",
        "1,0.5",
        "--loss",
        "0.01",
    ]);
    let trapped: Vec<f64> = serde_json::from_value(two["trapped"].clone()).unwrap();
    assert_eq!(trapped.len(), 2);
    close(trapped.iter().sum(), f(&two["efficiency"]), 1e-12);
    close(
        f(&two["efficiency"]) + f(&two["lost"]) + f(&two["dark"]),
        1.0,
        1e-12,
    );
    assert_eq!(two["traps"], json!([[2, 1.0], [4, 0.5]]));

    assert!(qso_fails(&["transport", "--graph", "path:3"]).contains("--trap"));
    assert!(qso_fails(&["transport", "--trap", "1", "--kappa", "1,2"]).contains("kappa"));
    assert!(qso_fails(&["transport", "--trap", "1", "--loss", "0.1,0.1,0.1"]).contains("loss"));
    assert!(qso_fails(&["transport", "--trap", "1", "--best-rate", "1"]).contains("best-rate"));
    assert!(qso_fails(&["transport", "--trap", "1", "--best-rate", "2,1"]).contains("range"));
    assert!(qso_fails(&["transport", "--trap", "2"]).contains('2'));
    // The dark tolerance is relative to the largest centred entry of H,
    // not to the hopping: with energies 100, 0 and hopping 10⁻⁹ the
    // coupling counts as dark.
    let help = qso_text(&["transport", "--help"]);
    assert!(help.contains("after removing its mean"), "{help}");
    assert!(!help.contains("relative to the hopping"), "{help}");
    let detuned = qso(&[
        "transport",
        "--graph",
        "path:2",
        "--trap",
        "1",
        "--energies",
        "100,0",
        "--gamma",
        "1e-9",
    ]);
    close(f(&detuned["dark"]), 1.0, 1e-12);
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
    // PySCF 2.14: water in STO-3G with a frozen core.
    let water = qso(&[
        "scf",
        "--atoms",
        "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        "--frozen",
        "1",
        "--mp2",
        "--ccsd-t",
    ]);
    close(f(&water["mp2"]["correlation"]), -0.03544594119987926, 1e-8);
    close(f(&water["ccsd"]["correlation"]), -0.04936022397391921, 1e-8);
    close(f(&water["ccsd_t"]["triples"]), -6.748571270771238e-05, 1e-8);
    let oh = qso(&[
        "scf",
        "--atoms",
        "O 0 0 0; H 0 0 0.97",
        "--multiplicity",
        "2",
        "--basis",
        "6-31g",
        "--restricted-open",
        "--ccsd",
    ]);
    close(f(&oh["ccsd"]["correlation"]), -0.10014818757154079, 1e-8);
    assert_eq!(oh["ccsd"]["spin_orbitals"], true);
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
        "--epsilon",
        "1e-6",
    ]);
    assert!(f(&t["column_error"]) <= f(&t["error_bound"]));
    // Order 4 at 20 steps: the commutator bound is the smallest, at most
    // the nested one.
    assert_eq!(f(&t["bounds"]["commutators"]), f(&t["error_bound"]));
    assert!(f(&t["bounds"]["commutators"]) <= f(&t["bounds"]["nested"]));
    assert!(f(&t["bounds"]["average"]) <= f(&t["error_bound"]));
    assert!(f(&t["bounds"]["stages"]) > 10.0 * f(&t["error_bound"]));
    let needed = &t["steps_needed"];
    assert!(needed["average"].as_u64().unwrap() <= needed["worst"].as_u64().unwrap());
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
fn radical_pairs() {
    let root = env!("CARGO_MANIFEST_DIR");
    let pair = format!("{root}/examples/spins/flavin_tryptophan.json");
    // Flavin–tryptophan at zero field from the singlet, k_S = k_T = 2·10⁷ s⁻¹.
    let r = qso(&[
        "radical-pair",
        &pair,
        "--rates",
        "2e7,2e7,0",
        "--fields",
        "0",
    ]);
    let y = &r["yields"][0];
    close(f(&y["singlet"]), 0.344_285_671_543_554, 1e-12);
    close(
        f(&y["singlet"]) + f(&y["triplet"]) + f(&y["escape"]),
        1.0,
        1e-12,
    );
    assert_eq!(f(&y["field"]), 0.0);
    assert_eq!(r["electrons"].as_array().unwrap().len(), 2);
    assert_eq!(r["nuclei"].as_array().unwrap().len(), 6);
    assert_eq!(r["nuclei"][0]["isotope"], "14N");
    assert_eq!(r["rates"], json!([2e7, 2e7, 0.0]));
    assert_eq!(r["recombination"], "haberkorn");
    assert_eq!(r["birth"], "singlet");
    assert_eq!(y["cidnp"].as_array().unwrap().len(), 6);
    // A sweep of three fields with the CIDNP of each.
    let text = qso_text(&[
        "radical-pair",
        &pair,
        "--rates",
        "2e7,2e7,0",
        "--fields",
        "0:0.002:3",
        "--cidnp",
    ]);
    let rows: Vec<&str> = text
        .lines()
        .skip_while(|l| !l.contains("B (mT)"))
        .skip(1)
        .take_while(|l| l.starts_with("  "))
        .collect();
    assert_eq!(rows.len(), 3, "{text}");
    for (row, b) in rows.iter().zip(["0.000000", "1.000000", "2.000000"]) {
        assert_eq!(row.split_whitespace().next(), Some(b), "{text}");
    }
    assert!(
        text.contains("(half the change of Φ_S from 0 to 2.000000 mT)"),
        "{text}"
    );
    let cidnp: Vec<&str> = text
        .lines()
        .skip_while(|l| !l.starts_with("CIDNP"))
        .collect();
    assert_eq!(cidnp.len(), 1 + 3 * (1 + 6), "{text}");
    // The same sweep in JSON, from T₀ with Jones, Maeda and Hore's term.
    let s = qso(&[
        "radical-pair",
        &pair,
        "--rates",
        "2e7,1e7,1e6",
        "--fields",
        "0:0.002:3",
        "--birth",
        "t0",
        "--recombination",
        "jones-maeda-hore",
    ]);
    assert_eq!(s["birth"], "triplet-zero");
    assert_eq!(s["recombination"], "jones-maeda-hore");
    // The JSON's names of the birth are read back as the same birth.
    let back = qso(&[
        "radical-pair",
        &pair,
        "--rates",
        "2e7,1e7,1e6",
        "--fields",
        "0:0.002:3",
        "--birth",
        s["birth"].as_str().unwrap(),
        "--recombination",
        "jones-maeda-hore",
    ]);
    assert_eq!(back, s);
    // The sweep's extremes and half-effect field, relative to its end.
    let sweep = &s["sweep"];
    close(f(&sweep["half_effect_reference"]), 0.002, 1e-18);
    let ys: Vec<f64> = s["yields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|y| f(&y["singlet"]))
        .collect();
    let lowest = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(f(&sweep["singlet_lowest"]["singlet"]), lowest);
    assert_eq!(f(&sweep["singlet_highest"]["singlet"]), highest);
    let half = f(&sweep["half_effect_field"]);
    assert!(half > 0.0 && half < 0.002, "{sweep}");
    // A single field has no sweep; the reasons for no half-effect field.
    assert!(r["sweep"].is_null());
    for (fields, why) in [
        ("0:0:3", "equals its value at 0 T"),
        (
            "0.001:0.002:3",
            "the sweep starts at 1.000000 mT, not at 0 T",
        ),
        ("0,0.002", "a sweep of three fields or more is needed"),
    ] {
        let text = qso_text(&[
            "radical-pair",
            &pair,
            "--rates",
            "2e7,2e7,0",
            "--fields",
            fields,
        ]);
        assert!(text.contains(why), "{text}");
        assert!(text.contains("half-effect field B½: none"), "{text}");
    }
    let yields = s["yields"].as_array().unwrap();
    assert_eq!(yields.len(), 3);
    for (y, b) in yields.iter().zip([0.0, 0.001, 0.002]) {
        close(f(&y["field"]), b, 1e-18);
        close(
            f(&y["singlet"]) + f(&y["triplet"]) + f(&y["escape"]),
            1.0,
            1e-12,
        );
    }
    // A system with one electron is not a radical pair.
    let one = Scratch::new("one_electron.json");
    std::fs::write(
        one.path(),
        r#"{"spins": [{"isotope": "e", "shift": 0.0}, {"isotope": "1H", "shift": 0.0}],
            "couplings": [[0, 1, 28024951.42420715]], "field": 0.0, "frame": "laboratory"}"#,
    )
    .unwrap();
    let err = qso_fails(&["radical-pair", one.path(), "--rates", "1e7,1e7,0"]);
    assert!(err.contains("exactly two electrons"), "{err}");
    // The rates have no default: leaving them out is a usage error.
    let out = run(&["radical-pair", &pair]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--rates"));
    // Two rates are not three.
    let err = qso_fails(&["radical-pair", &pair, "--rates", "1e7,1e7"]);
    assert!(err.contains("three rates"), "{err}");
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
    // The same shots in windows of three rounds committing two.
    let w = qso(&[
        "qec",
        "--circuit",
        stim.path(),
        "--shots",
        "2000",
        "--decoder",
        "matching",
        "--window",
        "3",
        "--commit",
        "2",
    ]);
    assert!(f(&w["logical_error_rate"]["rate"]) < 0.05);
    assert_eq!(w["decoder"]["window-matching"]["window"], 3);
    // In parallel windows committing two rounds with a buffer of one.
    let p = qso(&[
        "qec",
        "--circuit",
        stim.path(),
        "--shots",
        "2000",
        "--decoder",
        "matching",
        "--parallel",
        "--commit",
        "2",
        "--buffer",
        "1",
    ]);
    assert!(f(&p["logical_error_rate"]["rate"]) < 0.05);
    assert_eq!(p["decoder"]["parallel-window-matching"]["buffer"], 1);
    // Beyond 10 000 rounds the rounds are one repeated block.
    let long = qso(&[
        "qec",
        "--code",
        "repetition",
        "--distance",
        "3",
        "--rounds",
        "20000",
        "--p",
        "0.001",
        "--shots",
        "64",
        "--decoder",
        "matching",
        "--parallel",
        "--commit",
        "3",
    ]);
    assert_eq!(long[0]["rounds"], 20000);
    assert_eq!(long[0]["detectors"], 2 * 20001);
    assert!(qso_fails(&["qec", "--commit", "2", "--shots", "10"]).contains("commit"));
    assert!(
        qso_fails(&[
            "qec",
            "--decoder",
            "union-find",
            "--window",
            "3",
            "--shots",
            "10"
        ])
        .contains("window")
    );
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
    // The same on the orbits of the circuit's symmetries; the flag needs
    // the distance.
    let mut symmetric = vec![
        "qec",
        "--circuit",
        "tests/data/stim/surface_d5_memory_z.stim",
        "--shots",
        "100",
        "--circuit-distance",
        "--symmetry",
    ];
    let c = qso(&symmetric);
    assert_eq!(c["circuit_distance"]["lower_bound"], 5);
    assert_eq!(c["circuit_distance"]["upper_bound"], 5);
    symmetric.retain(|a| *a != "--circuit-distance");
    assert!(qso_fails(&symmetric).contains("circuit-distance"));
    // No time: the bounds before any step (the shortest graph-like cycle
    // above, two below), and the text says so.
    let args = [
        "qec",
        "--circuit",
        "tests/data/stim/surface_d5_memory_z.stim",
        "--shots",
        "100",
        "--circuit-distance",
        "--max-seconds",
        "0",
    ];
    let c = qso(&args);
    assert_eq!(c["circuit_distance"]["lower_bound"], 2);
    assert_eq!(c["circuit_distance"]["upper_bound"], 5);
    assert!(qso_text(&args).contains("circuit distance 2 to 5"));
    assert!(
        qso_fails(&[
            "qec",
            "--circuit",
            stim.path(),
            "--shots",
            "10",
            "--max-seconds",
            "5"
        ])
        .contains("circuit-distance")
    );
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

#[test]
fn heralded_erasure() {
    // Nine tenths of every probability as heralded erasures: 53 herald
    // detectors join the 8 of the repetition code, and decoding
    // conditions on the heralds that fired.
    let stim = Scratch::new("erasure.stim");
    let args = [
        "qec",
        "--code",
        "repetition",
        "--distance",
        "3",
        "--p",
        "0.01",
        "--erasure-share",
        "0.9",
        "--shots",
        "2000",
    ];
    let mut with_file = args.to_vec();
    with_file.extend(["--stim", stim.path()]);
    let e = qso(&with_file);
    assert_eq!(e[0]["erasure_share"], 0.9);
    assert_eq!(e[0]["herald_detectors"], 53);
    assert_eq!(e[0]["detectors"], 61);
    assert!(f(&e[0]["logical_error_rate"]["rate"]) < 0.05);
    let text = qso_text(&args);
    assert!(text.contains("3 rounds, erasure share 0.9:"), "{text}");
    assert!(text.contains("union-find with heralds"), "{text}");
    // Without the option: no heralds, and the text as before.
    let mut without = args.to_vec();
    without.drain(7..9);
    let plain = qso(&without);
    assert_eq!(plain[0]["erasure_share"], 0.0);
    assert_eq!(plain[0]["herald_detectors"], 0);
    assert_eq!(plain[0]["detectors"], 8);
    assert!(!qso_text(&without).contains("erasure"));
    // The written circuit holds Stim's HERALDED_ERASE and decodes as it
    // is, by each decoder that uses heralds.
    assert!(
        std::fs::read_to_string(stim.path())
            .unwrap()
            .contains("HERALDED_ERASE(")
    );
    for decoder in ["union-find", "matching", "bposd"] {
        let c = qso(&[
            "qec",
            "--circuit",
            stim.path(),
            "--shots",
            "2000",
            "--decoder",
            decoder,
        ]);
        assert_eq!(c["erasure_share"], Value::Null);
        assert_eq!(c["herald_detectors"], 53);
        assert_eq!(c["detectors"], 61);
        assert!(f(&c["logical_error_rate"]["rate"]) < 0.05);
    }
    // The share belongs to generated experiments: clap refuses it with a
    // file.
    let out = run(&[
        "qec",
        "--circuit",
        stim.path(),
        "--erasure-share",
        "0.5",
        "--shots",
        "10",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--erasure-share"));
    // The circuit distance does not take heralded operations yet, from a
    // file or generated.
    let refusal = "the circuit distance does not take heralded operations yet";
    let err = qso_fails(&[
        "qec",
        "--circuit",
        stim.path(),
        "--shots",
        "10",
        "--circuit-distance",
    ]);
    assert!(err.contains(refusal), "{err}");
    let mut distance = args.to_vec();
    distance.push("--circuit-distance");
    assert!(qso_fails(&distance).contains(refusal));
    // Windows and correlated matching do not use heralds yet.
    for extra in [
        &["--decoder", "matching", "--window", "3"][..],
        &["--decoder", "bposd", "--parallel", "--commit", "2"][..],
        &["--decoder", "correlated-matching"][..],
    ] {
        let mut refused = args.to_vec();
        refused.extend_from_slice(extra);
        let err = qso_fails(&refused);
        assert!(err.contains("does not use heralds yet"), "{err}");
        assert!(!err.contains('{'), "{err}");
    }
    // A share outside [0, 1] is refused.
    let mut wrong = args.to_vec();
    wrong[8] = "1.5";
    assert!(qso_fails(&wrong).contains("erasure_share"));
    // Beyond 10 000 rounds the rounds are one repeated block, heralds
    // included: 15 a round and 8 more, as the 53 = 15 · 3 + 8 of three
    // rounds above.
    let long = qso(&[
        "qec",
        "--code",
        "repetition",
        "--distance",
        "3",
        "--rounds",
        "20000",
        "--p",
        "0.001",
        "--erasure-share",
        "0.5",
        "--shots",
        "16",
        "--decoder",
        "matching",
    ]);
    assert_eq!(long[0]["rounds"], 20000);
    assert_eq!(long[0]["detectors"], 2 * 20001 + 300008);
    assert_eq!(long[0]["herald_detectors"], 300008);
}

#[test]
fn pauli_propagation() {
    // GHZ on 70 qubits, beyond every state-vector and density-matrix
    // simulator: ⟨Z0 Z69⟩ = 1, ⟨X0 X69⟩ = 0 and ⟨X⊗70⟩ = 1.
    let n = 70;
    let mut src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\nh q[0];\n");
    for i in 0..n - 1 {
        src.push_str(&format!("cx q[{i}],q[{}];\n", i + 1));
    }
    let ghz = Scratch::new("ghz70.qasm");
    std::fs::write(&ghz.0, &src).unwrap();
    let all_x: String = (0..n)
        .map(|q| format!("X{q}"))
        .collect::<Vec<_>>()
        .join(" ");

    let zz = qso(&[
        "propagate",
        ghz.path(),
        "--observable",
        "Z0 Z69",
        "--operator",
    ]);
    assert_eq!(zz["qubits"], n);
    assert_eq!(f(&zz["value"]), 1.0);
    assert_eq!(f(&zz["truncation_bound"]), 0.0);
    assert_eq!(zz["terms"], 1);
    assert_eq!(zz["dropped_terms"], 0);
    // C† Z0 Z69 C of the CX chain is Z1 … Z69 (H on qubit 0 leaves it).
    let z_tail: String = (1..n)
        .map(|q| format!("Z{q}"))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(zz["operator"], json!([[1.0, z_tail]]));
    let xx = qso(&["propagate", ghz.path(), "--observable", "X0 X69"]);
    assert_eq!(f(&xx["value"]), 0.0);
    assert_eq!(xx["operator"], Value::Null);
    let x = qso(&["propagate", ghz.path(), "--observable", &all_x]);
    assert_eq!(f(&x["value"]), 1.0);
    // From |0…01⟩ (qubit 0 on the right) the state is (|0…0⟩ − |1…1⟩)/√2.
    let one = format!("{}1", "0".repeat(n - 1));
    let flipped = qso(&[
        "propagate",
        ghz.path(),
        "--observable",
        &all_x,
        "--state",
        &one,
    ]);
    assert_eq!(f(&flipped["value"]), -1.0);
    let text = qso_text(&["propagate", ghz.path(), "--observable", "Z0 Z69"]);
    assert!(text.starts_with("<O> = 1.000000000000"), "{text}");

    // Truncation: RY(0.7) twice on qubit 0, ⟨Z0⟩ = cos 1.4 exactly. Dropping
    // the string sin 0.7·X0 between the two rotations leaves cos² 0.7; the
    // reported bound must cover the error.
    let ry = Scratch::new("ry.qasm");
    std::fs::write(
        &ry.0,
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nry(0.7) q[0];\nry(0.7) q[0];\ncx q[0],q[1];\n",
    )
    .unwrap();
    let exact = qso(&["propagate", ry.path(), "--observable", "Z0"]);
    close(f(&exact["value"]), 1.4f64.cos(), 1e-15);
    assert_eq!(f(&exact["truncation_bound"]), 0.0);
    let truncated = qso(&[
        "propagate",
        ry.path(),
        "--observable",
        "Z0",
        "--min-coefficient",
        "0.7",
    ]);
    let bound = f(&truncated["truncation_bound"]);
    assert!(bound > 0.0);
    assert!(truncated["dropped_terms"].as_u64().unwrap() >= 1);
    assert!((f(&truncated["value"]) - 1.4f64.cos()).abs() <= bound + 1e-15);
    let weight = qso(&[
        "propagate",
        ry.path(),
        "--observable",
        "Z1",
        "--max-weight",
        "1",
    ]);
    assert!(weight["dropped_terms"].as_u64().unwrap() >= 1);
    assert!((f(&weight["value"]) - 1.4f64.cos()).abs() <= f(&weight["truncation_bound"]) + 1e-15);

    // Device noise: a three-qubit GHZ on the example device, against its
    // density matrix (readout errors are not applied by either).
    let ghz3 = Scratch::new("ghz3.qasm");
    let src3 = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n";
    std::fs::write(&ghz3.0, src3).unwrap();
    let device = "examples/devices/toy_device.json";
    let noisy = qso(&[
        "propagate",
        ghz3.path(),
        "--observable",
        "Z0 Z1 + Z1 Z2 + X0 X1 X2",
        "--device",
        device,
    ]);
    let circuit = metatron_qso::quantum::from_qasm2(src3).unwrap();
    let model =
        metatron_qso::noise::DeviceModel::from_json(&std::fs::read_to_string(device).unwrap())
            .unwrap();
    let mut rho = metatron_qso::noise::DensityMatrix::zero(3).unwrap();
    rho.apply_device(&circuit, &[], &model).unwrap();
    let h: metatron_qso::quantum::PauliSum = "Z0 Z1 + Z1 Z2 + X0 X1 X2".parse().unwrap();
    let reference = rho.expectation(&h).unwrap();
    assert!(reference < 2.99);
    close(f(&noisy["value"]), reference, 1e-12);
    assert!(noisy["layers"].as_u64().unwrap() > 1);

    assert!(
        qso_fails(&["propagate", ghz.path(), "--observable", "Z70"]).contains("70"),
        "a qubit beyond the circuit"
    );
    assert!(
        qso_fails(&[
            "propagate",
            ghz3.path(),
            "--observable",
            "Z0",
            "--state",
            "01"
        ])
        .contains("3")
    );
    assert!(
        qso_fails(&[
            "propagate",
            ghz3.path(),
            "--observable",
            "Z0",
            "--state",
            "0+1"
        ])
        .contains("state")
    );
    assert!(
        qso_fails(&[
            "propagate",
            ghz3.path(),
            "--observable",
            "Z0",
            "--max-terms",
            "0"
        ])
        .contains("max_terms")
    );

    // Review regression: a final measurement is left out (as the Python
    // `propagate` does), a measurement followed by a gate on its qubit
    // dephases it and is refused instead of ignored (it printed 1, the
    // exact value is 0), and so are resets.
    let header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n";
    let fin = Scratch::new("final_measure.qasm");
    std::fs::write(
        &fin.0,
        format!("{header}h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"),
    )
    .unwrap();
    assert_eq!(
        f(&qso(&["propagate", fin.path(), "--observable", "Z0 Z1"])["value"]),
        1.0
    );
    let mid = Scratch::new("mid_measure.qasm");
    std::fs::write(
        &mid.0,
        format!("{header}h q[0];\nmeasure q[0] -> c[0];\nh q[0];\n"),
    )
    .unwrap();
    assert!(qso_fails(&["propagate", mid.path(), "--observable", "Z0"]).contains("measured"));
    let reset = Scratch::new("reset.qasm");
    std::fs::write(&reset.0, format!("{header}x q[0];\nreset q[0];\n")).unwrap();
    assert!(qso_fails(&["propagate", reset.path(), "--observable", "Z0"]).contains("reset"));
}
