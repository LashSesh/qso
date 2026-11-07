//! Tests of the dense transport solver ([`DephasingWalk`]): exact
//! rationals, closed forms, dark states, a Liouville-space brute force,
//! time traces, FMO against QuTiP, scans and refusals.

use std::time::Instant;

use num_complex::Complex64 as C64;

use super::dense::{Bright, DenseModel, Start};
use super::*;
use crate::graph::{Graph, generators, metatron};
use crate::linalg::dense::expm;
use crate::linalg::{CMatrix, RMatrix};
use crate::rng::Rng;
use crate::walks::{DephasingWalk, Generator, ProductOpenWalk};

fn opts() -> TransportOptions {
    TransportOptions::default()
}

/// `H = −A` (`Generator::Adjacency`, γ = 1) with the dephasing `rate`.
fn walk(g: &Graph, rate: f64) -> DephasingWalk {
    DephasingWalk::new(g, Generator::Adjacency, 1.0, rate).unwrap()
}

/// Rates `value` at the listed nodes, 0 elsewhere.
fn at(n: usize, entries: &[(usize, f64)]) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for &(j, x) in entries {
        v[j] = x;
    }
    v
}

fn assert_abs(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.16e}, want {want:.16e} (|Δ| {:.1e} > {tol:.0e})",
        (got - want).abs()
    );
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        ((got - want) / want).abs() <= tol,
        "{what}: got {got:.16e}, want {want:.16e} (rel {:.1e} > {tol:.0e})",
        ((got - want) / want).abs()
    );
}

/// η to 1e-12 and τ to 1e-9 relative (`None` for `tau = None`).
fn check(r: &TransportResult, eta: f64, tau: Option<f64>, what: &str) {
    assert_abs(r.efficiency, eta, 1e-12, &format!("{what}: η"));
    match tau {
        Some(t) => assert_rel(
            r.arrival_time.expect("an arrival time"),
            t,
            1e-9,
            &format!("{what}: τ"),
        ),
        None => assert!(r.arrival_time.is_none(), "{what}: τ {:?}", r.arrival_time),
    }
    assert!(r.error <= 1e-12, "{what}: error {:e}", r.error);
}

/// The detuned dimer: path(2), energies [3, 0], trap (1, 1), loss 1/20 on
/// both sites, dephasing `rate`.
fn dimer(generator: Generator, rate: f64) -> DephasingWalk {
    DephasingWalk::new(&generators::path(2), generator, 1.0, rate)
        .unwrap()
        .with_site_energies(&[3.0, 0.0])
        .unwrap()
        .with_traps(&[(1, 1.0)])
        .unwrap()
        .with_losses(&[0.05, 0.05])
        .unwrap()
}

/// SymPy's exact dimer values: (γ, η, τ, lost).
const DIMER: [(f64, f64, f64, f64); 5] = [
    (
        0.0,
        224000.0 / 290427.0,
        9200540.0 / 2032989.0,
        66427.0 / 290427.0,
    ),
    (1.0, 32000.0 / 38881.0, 139260.0 / 38881.0, 6881.0 / 38881.0),
    (
        3.0,
        288000.0 / 348689.0,
        100414940.0 / 28243809.0,
        60689.0 / 348689.0,
    ),
    (
        10.0,
        7072000.0 / 9575681.0,
        11335156860.0 / 2116225501.0,
        2503681.0 / 9575681.0,
    ),
    (
        100.0,
        64672000.0 / 235515281.0,
        7074070324860.0 / 475976382901.0,
        170843281.0 / 235515281.0,
    ),
];

/// The rounded values of the roadmap.
const DIMER_ROUNDED: [(f64, f64); 5] = [
    (0.771278, 4.5256),
    (0.823024, 3.5817),
    (0.825951, 3.5553),
    (0.738538, 5.3563),
    (0.274598, 14.8622),
];

fn check_dimer(r: &TransportResult, k: usize, what: &str) {
    let (_, eta, tau, lost) = DIMER[k];
    assert_abs(r.efficiency, eta, 1e-13, &format!("{what}: η"));
    assert_abs(r.lost, lost, 1e-13, &format!("{what}: lost"));
    assert_rel(r.arrival_time.unwrap(), tau, 1e-11, &format!("{what}: τ"));
    assert!(r.error <= 1e-13, "{what}: error {:e}", r.error);
    assert_eq!(r.dark, 0.0, "{what}");
    assert_eq!(r.dimension, 2, "{what}");
    let (eta6, tau4) = DIMER_ROUNDED[k];
    assert_eq!(
        format!("{:.6}", r.efficiency),
        format!("{eta6:.6}"),
        "{what}"
    );
    assert_eq!(
        format!("{:.4}", r.arrival_time.unwrap()),
        format!("{tau4:.4}"),
        "{what}"
    );
}

#[test]
fn the_detuned_dimer_matches_the_exact_rationals() {
    for generator in [Generator::Adjacency, Generator::Laplacian] {
        for (k, &(rate, ..)) in DIMER.iter().enumerate() {
            let r = dimer(generator, rate).transport(0, &opts()).unwrap();
            check_dimer(&r, k, &format!("{generator:?}, γ = {rate}"));
            assert_eq!(r.added_dephasing, 0.0);
            assert_eq!(r.trapped.len(), 1);
            assert_eq!(r.trapped[0], r.efficiency);
            let names: Vec<&str> = r.solves.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(names, ["schur", "X", "Y"]);
            assert!(r.solves.iter().all(|s| s.backward_error <= 1e-13));
            assert!(r.condition.is_finite() && r.condition >= 1.0);
            assert!(r.slowest_decay.unwrap() > 0.0);
        }
    }
}

#[test]
fn one_schur_form_serves_the_dimer_scan() {
    let rates: Vec<f64> = DIMER.iter().map(|d| d.0).collect();
    for generator in [Generator::Adjacency, Generator::Laplacian] {
        let scan = dimer(generator, 0.0)
            .dephasing_scan(0, &rates, &opts())
            .unwrap();
        assert_eq!(scan.len(), rates.len());
        for (k, r) in scan.iter().enumerate() {
            check_dimer(r, k, &format!("scan {generator:?}, γ = {}", rates[k]));
            assert_eq!(r.added_dephasing, rates[k]);
        }
    }
    // Input order is kept, and an empty scan is empty.
    let w = dimer(Generator::Adjacency, 0.0);
    let back = w.dephasing_scan(0, &[100.0, 0.0, 1.0], &opts()).unwrap();
    check_dimer(&back[0], 4, "reversed");
    check_dimer(&back[1], 0, "reversed");
    check_dimer(&back[2], 1, "reversed");
    assert!(w.dephasing_scan(0, &[], &opts()).unwrap().is_empty());
}

#[test]
fn the_best_rate_of_the_dimer_is_39_over_20() {
    let best = dimer(Generator::Adjacency, 0.0)
        .optimal_dephasing(0, 0.01, 100.0, &opts())
        .unwrap();
    assert_rel(best.rate, 39.0 / 20.0, 1e-6, "rate");
    assert_abs(best.result.efficiency, 800.0 / 963.0, 1e-13, "η*");
    // τ at the rate found. The efficiency is flat at its maximum, so the
    // rate is known to about √(ε/η″) ~ 1e-7 relative only, and τ moves
    // with it at dτ/dγ = 0.0142 (numpy, central difference): τ* agrees
    // with 3320/963 to 1e-9 relative plus that slope times the rate error.
    let tau = best.result.arrival_time.unwrap();
    let slack = 1e-9 * 3320.0 / 963.0 + 0.015 * (best.rate - 1.95).abs();
    assert!(
        (tau - 3320.0 / 963.0).abs() <= slack,
        "τ*: {tau:.16e} (rate {:.12})",
        best.rate
    );
    let fresh = dimer(Generator::Adjacency, best.rate)
        .transport(0, &opts())
        .unwrap();
    assert_rel(
        tau,
        fresh.arrival_time.unwrap(),
        1e-11,
        "τ* against a fresh walk",
    );
    assert!(best.interior);
    assert_eq!(best.result.added_dephasing, best.rate);
    assert!(best.evaluations > 17, "{}", best.evaluations);
}

/// The optimum on a flat efficiency (no loss, every node dephased: η = 1
/// at every rate) is the low end, and a best rate at an end of the range
/// is not interior.
#[test]
fn the_optimum_at_a_flat_efficiency_and_at_an_end_of_the_range() {
    // C₈ from node 1: η = ½ without dephasing (a dark state), 1 with any.
    let c8 = walk(&generators::cycle(8), 0.0)
        .with_traps(&[(0, 1.0)])
        .unwrap();
    assert_abs(
        c8.transport(1, &opts()).unwrap().efficiency,
        0.5,
        1e-12,
        "C8",
    );
    let flat = c8.optimal_dephasing(1, 0.1, 10.0, &opts()).unwrap();
    assert_eq!(flat.rate, 0.1);
    assert!(!flat.interior);
    assert_eq!(flat.evaluations, 14);
    assert_abs(flat.result.efficiency, 1.0, 1e-12, "C8 flat");
    assert_eq!(flat.result.added_dephasing, 0.1);
    // The dimer's efficiency falls beyond 39/20: on [10, 100] the best rate
    // is the low end.
    let end = dimer(Generator::Adjacency, 0.0)
        .optimal_dephasing(0, 10.0, 100.0, &opts())
        .unwrap();
    assert!(!end.interior);
    assert_rel(end.rate, 10.0, 1e-6, "dimer on [10, 100]");
    assert_abs(end.result.efficiency, DIMER[3].1, 1e-9, "dimer η(10)");
}

#[test]
fn complete_graphs_without_loss() {
    for (k, n) in [5usize, 16, 64].into_iter().enumerate() {
        let nf = n as f64;
        let base = walk(&generators::complete(n), 0.0)
            .with_traps(&[(1, 1.0)])
            .unwrap();
        // No dephasing: a rank-2 closure.
        let r = base.transport(0, &opts()).unwrap();
        let tau = (nf * nf - 2.0 * nf + 3.0) / (2.0 * (nf - 1.0));
        check(&r, 1.0 / (nf - 1.0), Some(tau), &format!("K{n}"));
        assert_abs(r.dark, (nf - 2.0) / (nf - 1.0), 1e-12, "dark");
        assert_eq!(r.dimension, 2);
        // γ = 0.8 at node 2: rank 3.
        let one = base
            .clone()
            .with_node_dephasing(&at(n, &[(2, 0.8)]))
            .unwrap()
            .transport(0, &opts())
            .unwrap();
        let tau_one = [229.0 / 42.0, 30967.0 / 1960.0, 2320471.0 / 36456.0][k];
        check(&one, 1.0 / (nf - 2.0), Some(tau_one), &format!("K{n} one"));
        assert_eq!(one.dimension, 3);
        // γ = 1 everywhere: every node bright.
        let all = base
            .clone()
            .with_node_dephasing(&vec![1.0; n])
            .unwrap()
            .transport(0, &opts())
            .unwrap();
        check(&all, 1.0, Some(nf * (nf + 2.0) / 4.0), &format!("K{n} all"));
        assert_eq!(all.dimension, n);
        assert_eq!(all.dark, 0.0);
        // At the trap dephasing changes nothing; at the start it frees
        // everything.
        for (node, eta) in [(1, 1.0 / (nf - 1.0)), (0, 1.0)] {
            let r = base
                .clone()
                .with_node_dephasing(&at(n, &[(node, 0.7)]))
                .unwrap()
                .transport(0, &opts())
                .unwrap();
            assert_abs(r.efficiency, eta, 1e-12, &format!("K{n}, 0.7 at {node}"));
        }
    }
}

/// The graph "m6e": t–u1, t–u2, j1–u1, j2–u2 (+1), j1–u2, j2–u1 (−1),
/// s–j1, s–j2 (+1), nodes (t, u1, u2, j1, j2, s) = 0..6.
fn m6e() -> Graph {
    Graph::from_weighted_edges(
        6,
        &[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (3, 1, 1.0),
            (4, 2, 1.0),
            (3, 2, -1.0),
            (4, 1, -1.0),
            (5, 3, 1.0),
            (5, 4, 1.0),
        ],
    )
    .unwrap()
}

/// The signed cycle C₄ with weights 1, 1, 1, −1.
fn signed_c4() -> Graph {
    Graph::from_weighted_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, -1.0)]).unwrap()
}

#[test]
fn dark_states_on_cycles_cubes_stars_and_signed_graphs() {
    let o = opts();
    // C₈, trap 0.
    let c8 = walk(&generators::cycle(8), 0.0)
        .with_traps(&[(0, 1.0)])
        .unwrap();
    let r = c8.transport(1, &o).unwrap();
    check(&r, 0.5, Some(4.0), "C8 from 1");
    assert_abs(r.dark, 0.5, 1e-12, "C8 dark");
    check(&c8.transport(4, &o).unwrap(), 1.0, Some(4.5), "C8 from 4");
    let deph = |j: usize, g: f64| c8.clone().with_node_dephasing(&at(8, &[(j, g)])).unwrap();
    let r = deph(2, 0.5).transport(1, &o).unwrap();
    check(&r, 0.75, Some(11.4805555556), "C8, 0.5 at 2");
    assert_abs(r.dark, 0.25, 1e-12, "C8, 0.5 at 2: dark");
    check(
        &deph(4, 0.5).transport(1, &o).unwrap(),
        0.5,
        Some(3.73),
        "C8, 0.5 at 4",
    );
    check(
        &deph(2, 0.6).transport(1, &o).unwrap(),
        0.75,
        Some(10.5608045336),
        "C8, 0.6 at 2",
    );
    check(
        &deph(2, 0.6).transport(4, &o).unwrap(),
        1.0,
        Some(6.6881270903),
        "C8, 0.6 at 2, from 4",
    );
    // Q₃, trap 0.
    let q3 = walk(&generators::hypercube(3), 0.0)
        .with_traps(&[(0, 1.0)])
        .unwrap();
    for (s, eta, tau) in [
        (1, 1.0 / 3.0, 23.0 / 6.0),
        (3, 1.0 / 3.0, 47.0 / 12.0),
        (7, 1.0, 89.0 / 36.0),
    ] {
        check(
            &q3.transport(s, &o).unwrap(),
            eta,
            Some(tau),
            &format!("Q3 from {s}"),
        );
    }
    // Q₃ with uniform loss: a shift, not a seed (dimension 4, not 8).
    let r = walk(&generators::hypercube(3), 0.0)
        .with_traps(&[(7, 1.0)])
        .unwrap()
        .with_losses(&[0.05; 8])
        .unwrap()
        .transport(1, &o)
        .unwrap();
    check(&r, 0.2788468798732, Some(3.2680774212), "Q3 with loss");
    assert_eq!(r.dimension, 4);
    assert_eq!(r.dark, 0.0);
    assert_abs(r.efficiency + r.lost, 1.0, 1e-12, "Q3 with loss: η + lost");
    // Star K₁,₅ from leaf 1 to leaf 2.
    let star = walk(&generators::star(6), 0.0)
        .with_traps(&[(2, 1.0)])
        .unwrap();
    let r = star.transport(1, &o).unwrap();
    check(&r, 0.25, Some(41.0 / 8.0), "star");
    assert_abs(r.dark, 0.75, 1e-12, "star dark");
    let r = star
        .clone()
        .with_node_dephasing(&at(6, &[(0, 0.7)]))
        .unwrap()
        .transport(1, &o)
        .unwrap();
    check(&r, 0.25, Some(883.0 / 160.0), "star, 0.7 at the centre");
    // Signed C₄, trap 0.
    let c4 = walk(&signed_c4(), 0.0).with_traps(&[(0, 1.0)]).unwrap();
    let r = c4.transport(2, &o).unwrap();
    check(&r, 0.0, None, "signed C4 from 2");
    assert_abs(r.dark, 1.0, 1e-12, "signed C4 dark");
    let r = c4
        .clone()
        .with_node_dephasing(&at(4, &[(1, 0.5)]))
        .unwrap()
        .transport(2, &o)
        .unwrap();
    check(&r, 1.0, Some(377.0 / 20.0), "signed C4, 0.5 at 1");
    for g in [0.5, 0.8] {
        let r = c4
            .clone()
            .with_node_dephasing(&at(4, &[(2, g)]))
            .unwrap()
            .transport(1, &o)
            .unwrap();
        check(&r, 0.5, Some(1.25), &format!("signed C4, {g} at 2"));
        assert_abs(r.dark, 0.5, 1e-12, "signed C4 dark");
    }
    // m6e: loss at u1, u2; dephasing 0.8 at j1, j2 and 0.5 at u1, u2.
    let r = walk(&m6e(), 0.0)
        .with_traps(&[(0, 1.0)])
        .unwrap()
        .with_losses(&at(6, &[(1, 0.1), (2, 0.1)]))
        .unwrap()
        .with_node_dephasing(&at(6, &[(3, 0.8), (4, 0.8), (1, 0.5), (2, 0.5)]))
        .unwrap()
        .transport(5, &o)
        .unwrap();
    check(&r, 0.6257822277847, Some(19.48890801001), "m6e");
    assert_abs(r.dark, 0.0, 1e-12, "m6e dark");
}

/// A mixed start with coherences between bright and dark states on a graph
/// with two pairs of twins (each pair carries a dark state); dephasing on
/// one twin makes its pair bright. References: the full Liouvillian in
/// numpy, its complex Schur form sorted to put the non-decaying
/// eigenvalues first, and the decaying block solved for `X` and `Y`.
#[test]
fn mixed_starts_on_twins_match_the_sorted_liouvillian() {
    let g = Graph::from_weighted_edges(
        7,
        &[
            (0, 1, 1.0),
            (1, 2, 0.7),
            (3, 1, 0.8),
            (4, 1, 0.8),
            (5, 2, -0.6),
            (6, 2, -0.6),
            (3, 4, 0.3),
        ],
    )
    .unwrap();
    let raw = [
        C64::new(0.1, 0.0),
        C64::new(0.2, 0.1),
        C64::new(0.0, 0.2),
        C64::new(0.5, 0.0),
        C64::new(-0.3, 0.1),
        C64::new(0.6, -0.2),
        C64::new(0.2, 0.3),
    ];
    let size = raw.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let psi: Vec<C64> = raw.iter().map(|z| z / size).collect();
    let mixed = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2];
    let rho = CMatrix::from_fn(7, 7, |a, b| {
        let d = if a == b { 0.3 * mixed[a] } else { 0.0 };
        psi[a] * psi[b].conj() * 0.7 + d
    });
    let base = walk(&g, 0.0)
        .with_site_energies(&[0.2, -0.1, 0.4, 0.3, 0.3, -0.5, -0.5])
        .unwrap()
        .with_traps(&[(0, 1.0)])
        .unwrap();
    let cases = [
        (
            [0.0, 0.4, 0.0, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            0.7181119122677182,
            11.753022388557223,
            0.07545951630370924,
            0.20642857142857257,
        ),
        (
            [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            0.45755448808044874,
            11.187867293874374,
            0.05887408334812298,
            0.48357142857142826,
        ),
        (
            [0.0, 0.4, 0.0, 0.9, 0.0, 0.0, 0.0],
            [0.02, 0.02, 0.07, 0.02, 0.02, 0.02, 0.02],
            0.578269322325448,
            10.010372968922319,
            0.4217306776745534,
            0.0,
        ),
    ];
    for (k, (gam, loss, eta, tau, lost, dark)) in cases.into_iter().enumerate() {
        let w = base
            .clone()
            .with_node_dephasing(&gam)
            .unwrap()
            .with_losses(&loss)
            .unwrap();
        let r = w.transport_from(&rho, &opts()).unwrap();
        check(&r, eta, Some(tau), &format!("twins {k}"));
        assert_abs(r.lost, lost, 1e-12, &format!("twins {k}: lost"));
        assert_abs(r.dark, dark, 1e-12, &format!("twins {k}: dark"));
    }
}

#[test]
fn dark_states_span_the_complement() {
    for (name, g, trap, start, dark) in [
        ("K16", generators::complete(16), 1, 0, 14.0 / 15.0),
        ("C8", generators::cycle(8), 0, 1, 0.5),
    ] {
        let n = g.num_nodes();
        let w = walk(&g, 0.0).with_traps(&[(trap, 1.0)]).unwrap();
        let v = w.dark_states(&opts()).unwrap();
        let r = w.transport(start, &opts()).unwrap();
        assert_eq!(v.nrows(), n, "{name}");
        assert_eq!(v.ncols(), n - r.dimension, "{name}");
        let gram = v.transpose() * &v - RMatrix::identity(v.ncols(), v.ncols());
        assert!(gram.amax() <= 1e-12, "{name}: {:e}", gram.amax());
        let weight: f64 = v.row(start).iter().map(|x| x * x).sum();
        assert_abs(weight, dark, 1e-12, &format!("{name}: ‖Vᵀe_s‖²"));
        assert_abs(r.dark, dark, 1e-12, &format!("{name}: dark"));
        assert!(v.row(trap).amax() <= 1e-12, "{name}: Vᵀe_trap");
    }
    // Every node bright: no dark states.
    let w = walk(&generators::cycle(8), 1.0)
        .with_traps(&[(0, 1.0)])
        .unwrap();
    assert_eq!(w.dark_states(&opts()).unwrap().ncols(), 0);
}

/// The Liouvillian `n² × n²` on row-major `vec ρ` (`ρ_ab` at `a·n + b`):
/// `−i(H_eff⊗I − I⊗H̄_eff) + Σ_j γ_j(E_jj⊗E_jj − ½E_jj⊗I − ½I⊗E_jj)`.
fn liouvillian(h: &RMatrix, traps: &[(usize, f64)], losses: &[f64], gam: &[f64]) -> CMatrix {
    let n = h.nrows();
    let mut heff = h.map(|x| C64::new(x, 0.0));
    for (j, &g) in losses.iter().enumerate() {
        heff[(j, j)] -= C64::new(0.0, 0.5 * g);
    }
    for &(t, k) in traps {
        heff[(t, t)] -= C64::new(0.0, k);
    }
    let m = n * n;
    let i = C64::new(0.0, 1.0);
    let mut l = CMatrix::zeros(m, m);
    for a in 0..n {
        for b in 0..n {
            let row = a * n + b;
            for c in 0..n {
                l[(row, c * n + b)] -= i * heff[(a, c)];
                l[(row, a * n + c)] += i * heff[(b, c)].conj();
            }
            if a == b {
                l[(row, row)] += C64::new(gam[a], 0.0);
            }
            l[(row, row)] -= C64::new(0.5 * (gam[a] + gam[b]), 0.0);
        }
    }
    l
}

/// `(η, τ, lost, trapped)` by the test-side Liouvillian and LU.
fn brute_force(
    h: &RMatrix,
    traps: &[(usize, f64)],
    losses: &[f64],
    gam: &[f64],
    rho0: &CMatrix,
) -> (f64, f64, f64, Vec<f64>) {
    let n = h.nrows();
    let lu = liouvillian(h, traps, losses, gam).lu();
    let v = nalgebra::DVector::from_fn(n * n, |k, _| -rho0[(k / n, k % n)]);
    let x = lu.solve(&v).unwrap();
    let y = lu.solve(&(-&x)).unwrap();
    let trapped: Vec<f64> = traps
        .iter()
        .map(|&(t, k)| 2.0 * k * x[t * n + t].re)
        .collect();
    let eta: f64 = trapped.iter().sum();
    let eta_tau: f64 = traps.iter().map(|&(t, k)| 2.0 * k * y[t * n + t].re).sum();
    let lost: f64 = (0..n).map(|j| losses[j] * x[j * n + j].re).sum();
    (eta, eta_tau / eta, lost, trapped)
}

/// A random walk of the brute-force tests: signed weights on a path plus
/// random edges, energies in (−1, 1), one or two traps, loss at least 0.01
/// everywhere, dephasing in (0, 2) on about 60 % of the nodes.
fn random_walk(rng: &mut Rng, n: usize) -> (DephasingWalk, RMatrix) {
    let mut edges = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if v == u + 1 || rng.uniform() < 0.4 {
                edges.push((u, v, rng.sign() * rng.uniform_range(0.3, 1.5)));
            }
        }
    }
    let g = Graph::from_weighted_edges(n, &edges).unwrap();
    let energies: Vec<f64> = (0..n).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
    let losses: Vec<f64> = (0..n).map(|_| rng.uniform_range(0.01, 0.1)).collect();
    let gam: Vec<f64> = (0..n)
        .map(|_| {
            if rng.uniform() < 0.6 {
                rng.uniform_range(0.0, 2.0)
            } else {
                0.0
            }
        })
        .collect();
    let first = rng.below(n as u64) as usize;
    let mut traps = vec![(first, rng.uniform_range(0.2, 1.5))];
    if n > 2 && rng.uniform() < 0.5 {
        let second = (first + 1 + rng.below(n as u64 - 1) as usize) % n;
        traps.push((second, rng.uniform_range(0.2, 1.5)));
    }
    let w = walk(&g, 0.0)
        .with_site_energies(&energies)
        .unwrap()
        .with_traps(&traps)
        .unwrap()
        .with_losses(&losses)
        .unwrap()
        .with_node_dephasing(&gam)
        .unwrap();
    let mut h = -g.adjacency_matrix();
    for (j, e) in energies.iter().enumerate() {
        h[(j, j)] += e;
    }
    (w, h)
}

/// A random density matrix `AA†/tr(AA†)`.
fn random_state(rng: &mut Rng, n: usize) -> CMatrix {
    let a = CMatrix::from_fn(n, n, |_, _| C64::new(rng.normal(), rng.normal()));
    let p = &a * a.adjoint();
    let tr: C64 = (0..n).map(|j| p[(j, j)]).sum();
    p / tr
}

#[test]
fn liouville_space_brute_force() {
    let mut rng = Rng::seed_from_u64(163);
    for case in 0..20 {
        let n = 2 + case % 5;
        let (w, h) = random_walk(&mut rng, n);
        let (traps, losses, gam) = (w.traps(), w.losses(), w.node_dephasing());
        let mut starts: Vec<(String, CMatrix, Option<usize>)> = (0..n)
            .map(|s| {
                let mut rho = CMatrix::zeros(n, n);
                rho[(s, s)] = C64::new(1.0, 0.0);
                (format!("node {s}"), rho, Some(s))
            })
            .collect();
        starts.push(("random state".into(), random_state(&mut rng, n), None));
        for (name, rho0, node) in starts {
            let what = format!("case {case} (n = {n}), {name}");
            let (eta, tau, lost, trapped) = brute_force(&h, traps, losses, gam, &rho0);
            let r = match node {
                Some(s) => w.transport(s, &opts()).unwrap(),
                None => w.transport_from(&rho0, &opts()).unwrap(),
            };
            assert_abs(r.efficiency, eta, 1e-12, &format!("{what}: η"));
            assert_rel(r.arrival_time.unwrap(), tau, 1e-11, &format!("{what}: τ"));
            assert_abs(r.lost, lost, 1e-12, &format!("{what}: lost"));
            for (k, (a, b)) in r.trapped.iter().zip(&trapped).enumerate() {
                assert_abs(*a, *b, 1e-12, &format!("{what}: trap {k}"));
            }
            assert_eq!(r.dark, 0.0, "{what}");
            assert!(r.error <= 1e-12, "{what}: error {:e}", r.error);
        }
    }
}

#[test]
fn the_time_integral_of_the_propagated_sink_population() {
    let w = walk(&generators::path(4), 0.0)
        .with_tolerance(1e-12)
        .with_site_energies(&[0.3, -0.2, 0.1, 0.0])
        .unwrap()
        .with_traps(&[(3, 0.7)])
        .unwrap()
        .with_losses(&[0.03; 4])
        .unwrap()
        .with_node_dephasing(&[0.0, 0.4, 0.4, 0.0])
        .unwrap();
    let r = w.transport(0, &opts()).unwrap();
    check(&r, 0.8698512709990, Some(4.4244755000), "P4");
    assert_abs(r.lost, 0.130148729001, 1e-12, "P4 lost");
    // Trace every 0.05 to T = 1000, where Tr ρ < 1e-12.
    let h = 0.05;
    let steps = 20_000;
    let times: Vec<f64> = (0..=steps).map(|k| k as f64 * h).collect();
    let trace = w.transport_trace(0, &times).unwrap();
    assert_eq!(trace.len(), times.len());
    for s in &trace {
        let tr: f64 = s.populations.iter().sum();
        assert_abs(
            s.trapped[0] + s.lost + tr,
            1.0,
            1e-9,
            &format!("t = {}", s.time),
        );
    }
    let last = trace.last().unwrap();
    let remaining: f64 = last.populations.iter().sum();
    assert!(remaining < 1e-12, "Tr ρ(T) = {remaining:e}");
    let p_t = last.trapped[0];
    assert_abs(p_t, r.efficiency, 1e-9, "p_sink(T)");
    assert_abs(last.lost, r.lost, 1e-9, "p_loss(T)");
    // Simpson's rule for ∫₀ᵀ p_sink.
    let integral = h / 3.0
        * trace
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let w = if k == 0 || k == steps {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                w * s.trapped[0]
            })
            .sum::<f64>();
    let t_end = last.time;
    let tau = (t_end * p_t - integral) / p_t;
    assert_rel(tau, r.arrival_time.unwrap(), 1e-7, "τ from the trace");
}

/// A trace from a partly dark start: the sinks fill to `η`, the dark
/// weight stays on the graph, and sinks, loss and `Tr ρ` add up to 1 at
/// every time (C₈ from node 1, dephasing at node 2: η = ¾, dark ¼).
#[test]
fn the_trace_of_a_partly_dark_start_keeps_the_dark_weight() {
    let w = walk(&generators::cycle(8), 0.0)
        .with_traps(&[(0, 1.0)])
        .unwrap()
        .with_node_dephasing(&at(8, &[(2, 0.5)]))
        .unwrap();
    let r = w.transport(1, &opts()).unwrap();
    check(&r, 0.75, Some(11.4805555556), "C8");
    assert_abs(r.dark, 0.25, 1e-12, "C8 dark");
    let times: Vec<f64> = (0..=40).map(|k| 50.0 * k as f64).collect();
    let trace = w.transport_trace(1, &times).unwrap();
    for s in &trace {
        let tr: f64 = s.populations.iter().sum();
        assert_abs(
            s.trapped[0] + s.lost + tr,
            1.0,
            1e-9,
            &format!("t = {}", s.time),
        );
        assert_eq!(s.lost, 0.0);
    }
    let last = trace.last().unwrap();
    assert_abs(last.trapped[0], r.efficiency, 1e-8, "p_sink(2000)");
    let remaining: f64 = last.populations.iter().sum();
    assert_abs(remaining, r.dark, 1e-8, "Tr ρ(2000)");
}

#[test]
fn evolution_with_traps_matches_the_liouville_exponential() {
    let mut rng = Rng::seed_from_u64(1630);
    for case in 0..6 {
        let n = 2 + case % 4;
        let (w, h) = random_walk(&mut rng, n);
        let w = w.with_tolerance(1e-12);
        let rho0 = random_state(&mut rng, n);
        let t = 1.3;
        let l = liouvillian(&h, w.traps(), w.losses(), w.node_dephasing());
        let prop = expm(&(l * C64::new(t, 0.0)));
        let v = nalgebra::DVector::from_fn(n * n, |k, _| rho0[(k / n, k % n)]);
        let want = prop * v;
        let got = w.evolve(&rho0, t).unwrap();
        let mut worst = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                worst = worst.max((got[(a, b)] - want[a * n + b]).norm());
            }
        }
        assert!(worst <= 1e-9, "case {case}: {worst:e}");
        // The trace states the same ρ.
        let trace = w.transport_trace(0, &[0.0, t]).unwrap();
        let mut start = CMatrix::zeros(n, n);
        start[(0, 0)] = C64::new(1.0, 0.0);
        let rho = w.evolve(&start, t).unwrap();
        for j in 0..n {
            assert_abs(trace[1].populations[j], rho[(j, j)].re, 1e-9, "trace");
        }
        assert_eq!(trace[0].populations[0], 1.0);
        assert_eq!(trace[0].lost, 0.0);
    }
}

#[test]
fn node_dephasing_equals_the_uniform_rate() {
    let g = metatron::fruit_of_life();
    let uniform = DephasingWalk::new(&g, Generator::Laplacian, 1.0, 0.5).unwrap();
    let nodes = DephasingWalk::new(&g, Generator::Laplacian, 1.0, 0.0)
        .unwrap()
        .with_node_dephasing(&[0.5; 13])
        .unwrap();
    // All-zero builders leave the walk unchanged as well.
    let zeros = uniform
        .clone()
        .with_site_energies(&[0.0; 13])
        .unwrap()
        .with_losses(&[0.0; 13])
        .unwrap()
        .with_traps(&[])
        .unwrap();
    let mut rho0 = CMatrix::zeros(13, 13);
    rho0[(7, 7)] = C64::new(1.0, 0.0);
    let a = uniform.evolve(&rho0, 1.7).unwrap();
    assert_eq!(a, nodes.evolve(&rho0, 1.7).unwrap());
    assert_eq!(a, zeros.evolve(&rho0, 1.7).unwrap());
    assert_eq!(nodes.node_dephasing(), &[0.5; 13]);
}

/// Adolphs and Renger's FMO Hamiltonian of C. tepidum (cm⁻¹), as in
/// Sarovar et al., Nat. Phys. 6, 462 (2010), SI.
const FMO_CM: [[f64; 7]; 7] = [
    [12410.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
    [-87.7, 12530.0, 30.8, 8.2, 0.7, 11.8, 4.3],
    [5.5, 30.8, 12210.0, -53.5, -2.2, -9.6, 6.0],
    [-5.9, 8.2, -53.5, 12320.0, -70.7, -17.0, -63.3],
    [6.7, 0.7, -2.2, -70.7, 12480.0, 81.1, -1.3],
    [-13.7, 11.8, -9.6, -17.0, 81.1, 12630.0, 39.7],
    [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440.0],
];

/// 1 cm⁻¹ in rad/ps: `2πc` with `c` in cm/ps.
const CM: f64 = 2.0 * std::f64::consts::PI * 2.99792458e-2;

/// FMO in ps: energies relative to the mean, couplings `V_ij` as weights
/// `−V_ij` of `−A`, trap at BChl 3 (κ = 1/ps), loss 1/ns, dephasing
/// `rate`.
fn fmo(rate: f64) -> DephasingWalk {
    let mean = (0..7).map(|j| FMO_CM[j][j]).sum::<f64>() / 7.0;
    let mut edges = Vec::new();
    for (i, row) in FMO_CM.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().skip(i + 1) {
            edges.push((i, j, -v * CM));
        }
    }
    let g = Graph::from_weighted_edges(7, &edges).unwrap();
    let energies: Vec<f64> = (0..7).map(|j| (FMO_CM[j][j] - mean) * CM).collect();
    walk(&g, rate)
        .with_site_energies(&energies)
        .unwrap()
        .with_traps(&[(2, 1.0)])
        .unwrap()
        .with_losses(&[1e-3; 7])
        .unwrap()
}

#[test]
fn fmo_matches_qutip() {
    let cases = [
        (0.0, 0.9468496984369823, 47.0869692949756),
        (1.0, 0.9818827535623038, 18.147121788722004),
        (10.0, 0.9937129913795011, 6.291855323156555),
        (300.0 * CM, 0.9940061133634202, 5.99744121130125),
        (100.0, 0.9930966074315369, 6.9078977658981655),
    ];
    for (rate, eta, tau) in cases {
        let r = fmo(rate).transport(0, &opts()).unwrap();
        assert_abs(r.efficiency, eta, 1e-11, &format!("FMO γ = {rate}: η"));
        assert_rel(
            r.arrival_time.unwrap(),
            tau,
            1e-10,
            &format!("FMO γ = {rate}: τ"),
        );
    }
    let best = fmo(0.0).optimal_dephasing(0, 1.0, 1000.0, &opts()).unwrap();
    assert_rel(best.rate, 26.0181, 1e-3, "FMO best rate");
    assert_abs(best.result.efficiency, 0.9943859289, 1e-9, "FMO η*");
    assert!(best.interior);
}

/// FMO with its raw site energies (about 2300 rad/ps on every site) gives
/// the results of the centred energies: a common offset leaves the
/// Liouvillian unchanged, and the solver must not lose digits to it (it
/// lost 2·10⁻¹¹ in η and 6·10⁻¹⁰ in τ before `H′_B` was centred).
#[test]
fn a_common_energy_offset_costs_no_digits() {
    let raw = |rate: f64| {
        let mut edges = Vec::new();
        for (i, row) in FMO_CM.iter().enumerate() {
            for (j, &v) in row.iter().enumerate().skip(i + 1) {
                edges.push((i, j, -v * CM));
            }
        }
        let g = Graph::from_weighted_edges(7, &edges).unwrap();
        let energies: Vec<f64> = (0..7).map(|j| FMO_CM[j][j] * CM).collect();
        walk(&g, rate)
            .with_site_energies(&energies)
            .unwrap()
            .with_traps(&[(2, 1.0)])
            .unwrap()
            .with_losses(&[1e-3; 7])
            .unwrap()
    };
    for rate in [0.0, 1.0, 10.0] {
        let want = fmo(rate).transport(0, &opts()).unwrap();
        let got = raw(rate).transport(0, &opts()).unwrap();
        let what = format!("raw FMO, γ = {rate}");
        assert_abs(got.efficiency, want.efficiency, 1e-12, &what);
        assert_rel(
            got.arrival_time.unwrap(),
            want.arrival_time.unwrap(),
            1e-10,
            &what,
        );
        assert!(got.error <= 1e-12, "{what}: error {:e}", got.error);
    }
    // Only node 0 dephased: the closure by Gram–Schmidt on the centred `H`.
    let partial = |w: DephasingWalk| {
        w.with_losses(&[0.0; 7])
            .unwrap()
            .with_node_dephasing(&at(7, &[(0, 1.0)]))
            .unwrap()
            .transport(0, &opts())
            .unwrap()
    };
    let want = partial(fmo(0.0));
    let got = partial(raw(0.0));
    assert_abs(
        got.efficiency,
        want.efficiency,
        1e-12,
        "raw FMO, γ at node 0",
    );
    assert_rel(
        got.arrival_time.unwrap(),
        want.arrival_time.unwrap(),
        1e-10,
        "raw FMO, γ at node 0",
    );
    // A dark state and a general basis: C₈ from node 1 at energy 10⁴ on
    // every node, with and without dephasing at node 2.
    for (deph, eta, tau) in [(0.0, 0.5, 4.0), (0.5, 0.75, 11.4805555556)] {
        let r = walk(&generators::cycle(8), 0.0)
            .with_site_energies(&[1e4; 8])
            .unwrap()
            .with_traps(&[(0, 1.0)])
            .unwrap()
            .with_node_dephasing(&at(8, &[(2, deph)]))
            .unwrap()
            .transport(1, &opts())
            .unwrap();
        check(
            &r,
            eta,
            Some(tau),
            &format!("C8 at energy 1e4, γ {deph} at node 2"),
        );
        assert_abs(r.dark, 1.0 - eta, 1e-12, "C8 at energy 1e4: dark");
    }
}

/// The population GMRES stops at `MAX_TRANSPORT_WORK` with the documented
/// error.
#[test]
fn the_work_limit_stops_the_population_solves() {
    let n = 16;
    let model = DenseModel {
        h: -generators::complete(n).adjacency_matrix(),
        traps: vec![(1, 1.0)],
        losses: vec![0.0; n],
        dephasing: vec![1.0; n],
    };
    let bright = Bright::new(&model, false, &opts()).unwrap();
    assert!(bright.solve(&Start::Node(0), 0.0, true, &opts()).is_ok());
    match bright
        .with_spent(MAX_TRANSPORT_WORK)
        .solve(&Start::Node(0), 0.0, true, &opts())
    {
        Err(QsoError::InvalidArgument { name, reason }) => {
            assert_eq!(name, "graph");
            assert!(reason.contains("operations after"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn scan_matches_fresh_walks() {
    let mut rng = Rng::seed_from_u64(77);
    let n = 12;
    let mut edges = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if v == u + 1 || rng.uniform() < 0.3 {
                edges.push((u, v, rng.sign() * rng.uniform_range(0.3, 1.5)));
            }
        }
    }
    let g = Graph::from_weighted_edges(n, &edges).unwrap();
    let energies: Vec<f64> = (0..n).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
    let own: Vec<f64> = (0..n)
        .map(|j| {
            if j % 2 == 0 {
                rng.uniform_range(0.1, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let base = walk(&g, 0.0)
        .with_site_energies(&energies)
        .unwrap()
        .with_traps(&[(n - 1, 1.0)])
        .unwrap()
        .with_losses(&[0.02; 12])
        .unwrap();
    let w = base.clone().with_node_dephasing(&own).unwrap();
    let rates = [0.0, 0.1, 1.0, 10.0];
    let scan = w.dephasing_scan(0, &rates, &opts()).unwrap();
    for (r, &e) in scan.iter().zip(&rates) {
        let shifted: Vec<f64> = own.iter().map(|g| g + e).collect();
        let fresh = base
            .clone()
            .with_node_dephasing(&shifted)
            .unwrap()
            .transport(0, &opts())
            .unwrap();
        assert_abs(
            r.efficiency,
            fresh.efficiency,
            1e-12,
            &format!("e = {e}: η"),
        );
        assert_rel(
            r.arrival_time.unwrap(),
            fresh.arrival_time.unwrap(),
            1e-10,
            &format!("e = {e}: τ"),
        );
        assert_eq!(r.added_dephasing, e);
        assert_eq!(fresh.added_dephasing, 0.0);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn results_do_not_depend_on_the_thread_count() {
    for n in [64, 130] {
        let w = walk(&generators::cycle(n), 1.0)
            .with_losses(&vec![0.01; n])
            .unwrap()
            .with_traps(&[(n / 2, 1.0)])
            .unwrap();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| w.transport(0, &opts()).unwrap());
        let default = w.transport(0, &opts()).unwrap();
        assert_eq!(one.efficiency, default.efficiency, "C{n}");
        assert_eq!(one.arrival_time, default.arrival_time, "C{n}");
        assert_eq!(one, default, "C{n}");
    }
}

fn assert_invalid<T: std::fmt::Debug>(r: crate::error::Result<T>, name: &str) {
    match r {
        Err(QsoError::InvalidArgument { name: got, .. }) => assert_eq!(got, name),
        other => panic!("expected an invalid `{name}`, got {other:?}"),
    }
}

fn assert_mismatch<T: std::fmt::Debug>(r: crate::error::Result<T>) {
    assert!(
        matches!(r, Err(QsoError::DimensionMismatch { .. })),
        "{r:?}"
    );
}

#[test]
fn refuses_bad_input() {
    let g = generators::path(3);
    let plain = walk(&g, 0.0);
    let w = plain.clone().with_traps(&[(2, 1.0)]).unwrap();
    let o = opts();
    // No trap.
    assert_invalid(plain.transport(0, &o), "traps");
    assert_invalid(plain.dark_states(&o), "traps");
    assert_invalid(plain.dephasing_scan(0, &[1.0], &o), "traps");
    // Nodes out of range.
    assert!(matches!(
        w.transport(3, &o),
        Err(QsoError::NodeOutOfRange {
            node: 3,
            n_nodes: 3
        })
    ));
    assert!(matches!(
        plain.clone().with_traps(&[(5, 1.0)]),
        Err(QsoError::NodeOutOfRange {
            node: 5,
            n_nodes: 3
        })
    ));
    assert!(matches!(
        w.transport_trace(4, &[1.0]),
        Err(QsoError::NodeOutOfRange { .. })
    ));
    // Traps: twice, bad rates.
    assert_invalid(plain.clone().with_traps(&[(1, 1.0), (1, 2.0)]), "traps");
    for k in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_invalid(plain.clone().with_traps(&[(1, k)]), "traps");
    }
    // Lengths and values.
    assert_mismatch(plain.clone().with_site_energies(&[0.0; 2]));
    assert_mismatch(plain.clone().with_losses(&[0.0; 4]));
    assert_mismatch(plain.clone().with_node_dephasing(&[]));
    assert_invalid(
        plain.clone().with_site_energies(&[0.0, f64::NAN, 0.0]),
        "energies",
    );
    for bad in [-0.1, f64::NAN, f64::INFINITY] {
        assert_invalid(plain.clone().with_losses(&[0.0, bad, 0.0]), "losses");
        assert_invalid(
            plain.clone().with_node_dephasing(&[bad, 0.0, 0.0]),
            "dephasing",
        );
    }
    // Options.
    let with = |f: &dyn Fn(&mut TransportOptions)| {
        let mut o = opts();
        f(&mut o);
        o
    };
    for t in [0.0, -1e-10, 0.1, f64::NAN] {
        assert_invalid(
            w.transport(0, &with(&|o| o.dark_tolerance = t)),
            "dark_tolerance",
        );
    }
    for t in [1e-17, 1e-5, f64::NAN] {
        assert_invalid(w.transport(0, &with(&|o| o.tolerance = t)), "tolerance");
    }
    assert_invalid(
        w.transport(0, &with(&|o| o.max_iterations = 0)),
        "max_iterations",
    );
    // The start ρ₀.
    assert_mismatch(w.transport_from(&CMatrix::identity(2, 2), &o));
    let mut rho = CMatrix::zeros(3, 3);
    rho[(0, 0)] = C64::new(1.0, 0.0);
    rho[(0, 1)] = C64::new(0.1, 0.0);
    assert_invalid(w.transport_from(&rho, &o), "rho0");
    let half = CMatrix::identity(3, 3) * C64::new(0.5, 0.0);
    assert_invalid(w.transport_from(&half, &o), "rho0");
    let mut nan = CMatrix::zeros(3, 3);
    nan[(0, 0)] = C64::new(1.0, 0.0);
    nan[(1, 2)] = C64::new(f64::NAN, 0.0);
    assert_invalid(w.transport_from(&nan, &o), "rho0");
    // Scans, ranges and times.
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert_invalid(w.dephasing_scan(0, &[0.0, bad], &o), "rates");
    }
    for (low, high) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, f64::INFINITY),
        (f64::NAN, 1.0),
    ] {
        assert_invalid(w.optimal_dephasing(0, low, high, &o), "range");
    }
    for times in [
        vec![-1.0],
        vec![1.0, 0.5],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert_invalid(w.transport_trace(0, &times), "times");
    }
    // Sizes: 2048 nodes for transport, 512 for evolution.
    assert_invalid(
        DephasingWalk::new(&generators::path(2049), Generator::Adjacency, 1.0, 0.0),
        "graph",
    );
    let big = walk(&generators::path(513), 0.0)
        .with_traps(&[(512, 1.0)])
        .unwrap();
    let mut rho = CMatrix::zeros(513, 513);
    rho[(0, 0)] = C64::new(1.0, 0.0);
    assert_invalid(big.evolve(&rho, 1.0), "graph");
    assert_invalid(big.populations(0, 1.0), "graph");
    assert_invalid(big.transport_trace(0, &[1.0]), "graph");
    // GMRES out of iterations.
    let k16 = walk(&generators::complete(16), 1.0)
        .with_traps(&[(1, 1.0)])
        .unwrap();
    match k16.transport(0, &with(&|o| o.max_iterations = 1)) {
        Err(QsoError::NoConvergence { method, .. }) => {
            assert_eq!(method, "GMRES (transport populations)")
        }
        other => panic!("{other:?}"),
    }
    // A bright state that decays below rounding.
    let weak = walk(
        &Graph::from_weighted_edges(2, &[(0, 1, 1e-9)]).unwrap(),
        0.0,
    )
    .with_traps(&[(0, 1.0)])
    .unwrap();
    assert_invalid(weak.transport(1, &o), "dark_tolerance");
    // Products refuse factors that lose population.
    let factor = walk(&generators::path(2), 0.5);
    assert!(ProductOpenWalk::new(vec![(factor.clone(), 2)]).is_ok());
    let energies = factor.clone().with_site_energies(&[0.1, 0.2]).unwrap();
    assert!(ProductOpenWalk::new(vec![(energies, 2)]).is_ok());
    let trapped = factor.clone().with_traps(&[(1, 1.0)]).unwrap();
    assert_invalid(ProductOpenWalk::new(vec![(trapped, 1)]), "factors");
    let lossy = factor.with_losses(&[0.0, 0.1]).unwrap();
    assert_invalid(ProductOpenWalk::new(vec![(lossy, 1)]), "factors");
}

/// Times, iterations and accuracy of the dense path, run with
/// `cargo test --release -p metatron-qso-rs --lib -- --ignored measure_transport --nocapture`
/// (sizes from `QSO_TRANSPORT_SIZES`, default `256,512,1024`).
#[test]
#[ignore]
fn measure_transport() {
    let sizes: Vec<usize> = std::env::var("QSO_TRANSPORT_SIZES")
        .unwrap_or_else(|_| "256,512,1024".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let o = opts();
    println!("threads: {}", crate::parallel::threads());
    for n in sizes {
        let mut rng = Rng::seed_from_u64(n as u64);
        let mut edges = Vec::new();
        for u in 0..n {
            for v in u + 1..n {
                if v == u + 1 || rng.uniform() < 4.0 / n as f64 {
                    edges.push((u, v, rng.sign() * rng.uniform_range(0.3, 1.5)));
                }
            }
        }
        let g = Graph::from_weighted_edges(n, &edges).unwrap();
        let mut h = -g.adjacency_matrix();
        let energies: Vec<f64> = (0..n).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
        for (j, e) in energies.iter().enumerate() {
            h[(j, j)] += e;
        }
        let traps = vec![(n - 1, 1.0)];
        let losses = vec![0.01; n];
        for (label, gam) in [
            (
                "γ on every node",
                (0..n)
                    .map(|_| rng.uniform_range(0.1, 1.0))
                    .collect::<Vec<f64>>(),
            ),
            (
                "γ on half the nodes",
                (0..n)
                    .map(|j| {
                        if j % 2 == 0 {
                            rng.uniform_range(0.1, 1.0)
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
        ] {
            let model = DenseModel {
                h: h.clone(),
                traps: traps.clone(),
                losses: losses.clone(),
                dephasing: gam,
            };
            let t0 = Instant::now();
            let bright = Bright::new(&model, false, &o).unwrap();
            let setup = t0.elapsed().as_secs_f64();
            let report = |name: &str, b: &Bright| {
                let t0 = Instant::now();
                let r = b.solve(&Start::Node(0), 0.0, true, &o).unwrap();
                let its: Vec<String> = r
                    .solves
                    .iter()
                    .map(|s| {
                        format!(
                            "{} {} its, β {:.1e}, cond {:.1e}",
                            s.name, s.iterations, s.backward_error, s.condition
                        )
                    })
                    .collect();
                println!(
                    "  n = {n}, {label}, {name}: r = {}, η {:.12}, τ {:?}, error {:.1e}, solves {:.2} s [{}]",
                    r.dimension,
                    r.efficiency,
                    r.arrival_time,
                    r.error,
                    t0.elapsed().as_secs_f64(),
                    its.join("; ")
                );
            };
            println!("n = {n}, {label}: closure and Schur form {setup:.2} s");
            report("classical preconditioner", &bright);
            if bright.dim() == n && label == "γ on every node" {
                report("no preconditioner", &bright.without_classical());
            }
        }
    }
}
