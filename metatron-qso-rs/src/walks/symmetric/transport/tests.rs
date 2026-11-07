//! Tests of transport on orbitals and hypercubes: site terms against the
//! density matrix, the dense transport solver, closed forms on complete
//! graphs (SymPy) and the hypercube against its column chain and QuTiP.

use std::time::Instant;

use crate::error::QsoError;
use crate::graph::{Graph, generators};
use crate::walks::symmetric::{
    Dephasing, HypercubeOpenWalk, ProductOpenWalk, SymmetricOpenWalk, coordinate_permutations,
};
use crate::walks::transport::{TransportOptions, TransportResult};
use crate::walks::{DephasingWalk, Generator};

fn opts() -> TransportOptions {
    TransportOptions::default()
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

/// `η` to `eta_tol` (absolute), `τ` to `tau_tol` (relative) and the
/// error estimate below `max(eta_tol, 1e-13)`.
fn check(r: &TransportResult, eta: f64, tau: f64, eta_tol: f64, tau_tol: f64, what: &str) {
    assert_abs(r.efficiency, eta, eta_tol, &format!("{what}: η"));
    assert_rel(
        r.arrival_time.expect("an arrival time"),
        tau,
        tau_tol,
        &format!("{what}: τ"),
    );
    assert!(r.error <= eta_tol.max(1e-13), "{what}: error {:e}", r.error);
}

/// Rates `value` at the listed nodes, 0 elsewhere.
fn at(n: usize, entries: &[(usize, f64)]) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for &(j, x) in entries {
        v[j] = x;
    }
    v
}

/// `H = −A` with dephasing given by `with_node_dephasing` only.
fn orbital_walk(g: &Graph, symmetries: &[Vec<usize>]) -> SymmetricOpenWalk {
    SymmetricOpenWalk::new(g, Generator::Adjacency, 1.0, &[], symmetries).unwrap()
}

fn dense_walk(g: &Graph) -> DephasingWalk {
    DephasingWalk::new(g, Generator::Adjacency, 1.0, 0.0).unwrap()
}

/// The Petersen graph as Kneser(5, 2): nodes the pairs of `0..5` in
/// lexicographic order, adjacent when disjoint; and the permutations of
/// its nodes induced by permutations of `0..5`.
fn kneser() -> (Graph, Vec<(usize, usize)>) {
    let pairs: Vec<(usize, usize)> = (0..5)
        .flat_map(|a| (a + 1..5).map(move |b| (a, b)))
        .collect();
    let mut edges = Vec::new();
    for (i, p) in pairs.iter().enumerate() {
        for (j, q) in pairs.iter().enumerate().skip(i + 1) {
            if p.0 != q.0 && p.0 != q.1 && p.1 != q.0 && p.1 != q.1 {
                edges.push((i, j));
            }
        }
    }
    (Graph::from_edges(10, &edges).unwrap(), pairs)
}

fn induced(pairs: &[(usize, usize)], sigma: &[usize]) -> Vec<usize> {
    pairs
        .iter()
        .map(|&(a, b)| {
            let (x, y) = (sigma[a].min(sigma[b]), sigma[a].max(sigma[b]));
            pairs.iter().position(|&p| p == (x, y)).unwrap()
        })
        .collect()
}

/// The graph, the symmetries, the start, the traps and the losses,
/// dephasing and energies per node.
type PetersenCase = (
    Graph,
    Vec<Vec<usize>>,
    usize,
    Vec<(usize, f64)>,
    [Vec<f64>; 3],
);

/// Petersen with the start (0,1), traps (2,3), (2,4), (3,4) at κ 0.7, and
/// loss 0.05, dephasing 0.3 and energy 0.2 on the nodes meeting {0, 1} in
/// one point; the generators fix {0, 1} as a set.
fn petersen_case() -> PetersenCase {
    let (g, pairs) = kneser();
    let symmetries = vec![
        induced(&pairs, &[0, 1, 3, 2, 4]),
        induced(&pairs, &[0, 1, 3, 4, 2]),
        induced(&pairs, &[1, 0, 2, 3, 4]),
    ];
    let index = |p: (usize, usize)| pairs.iter().position(|&q| q == p).unwrap();
    let traps = vec![
        (index((2, 3)), 0.7),
        (index((2, 4)), 0.7),
        (index((3, 4)), 0.7),
    ];
    let one: Vec<bool> = pairs.iter().map(|&(a, b)| (a < 2) != (b < 2)).collect();
    let value = |x: f64| -> Vec<f64> { one.iter().map(|&o| if o { x } else { 0.0 }).collect() };
    (
        g,
        symmetries,
        index((0, 1)),
        traps,
        [value(0.05), value(0.3), value(0.2)],
    )
}

/// Hamming weights of the nodes of `Q_d`.
fn weights(d: usize) -> Vec<usize> {
    (0..1usize << d).map(|u| u.count_ones() as usize).collect()
}

#[test]
fn site_terms_on_orbitals_match_the_density_matrix() {
    let t = 1.1;
    // Petersen: energies, traps, losses and per-node dephasing on orbits,
    // plus a partition dephasing term of `new`.
    let (g, symmetries, start, traps, [loss, deph, eps]) = petersen_case();
    // Q₃ by weight, the trap at the antipode.
    let w = weights(3);
    let by = |f: &dyn Fn(usize) -> f64| -> Vec<f64> { w.iter().map(|&k| f(k)).collect() };
    let cases = [
        (g, symmetries, start, traps, loss, deph, eps),
        (
            generators::hypercube(3),
            coordinate_permutations(3).unwrap(),
            0,
            vec![(7, 0.6)],
            by(&|k| 0.02 + 0.03 * k as f64),
            by(&|k| [0.4, 0.1, 0.0, 0.7][k]),
            by(&|k| 0.3 * k as f64 - 0.25 * (k * k) as f64),
        ),
    ];
    for (g, symmetries, start, traps, loss, deph, eps) in cases {
        let n = g.num_nodes();
        let partition = 0.2;
        let orbital = SymmetricOpenWalk::new(
            &g,
            Generator::Adjacency,
            0.9,
            &[Dephasing::sites(n, partition)],
            &symmetries,
        )
        .unwrap()
        .with_site_energies(&eps)
        .unwrap()
        .with_traps(&traps)
        .unwrap()
        .with_losses(&loss)
        .unwrap()
        .with_node_dephasing(&deph)
        .unwrap();
        let p = orbital
            .populations(
                &orbital
                    .evolve(&orbital.localized(start).unwrap(), t)
                    .unwrap(),
            )
            .unwrap();
        let total: Vec<f64> = deph.iter().map(|g| g + partition).collect();
        let dense = DephasingWalk::new(&g, Generator::Adjacency, 0.9, 0.0)
            .unwrap()
            .with_tolerance(1e-12)
            .with_site_energies(&eps)
            .unwrap()
            .with_traps(&traps)
            .unwrap()
            .with_losses(&loss)
            .unwrap()
            .with_node_dephasing(&total)
            .unwrap();
        let q = dense.populations(start, t).unwrap();
        for (u, (a, b)) in p.iter().zip(&q).enumerate() {
            assert_abs(*a, *b, 1e-9, &format!("{n} nodes, node {u}"));
        }
        // Population has left: the trap and the losses work.
        assert!(p.iter().sum::<f64>() < 0.99);
    }
}

#[test]
fn builders_without_terms_are_bit_identical() {
    let g = generators::hypercube(4);
    let symmetries = coordinate_permutations(4).unwrap();
    let dephasing = [Dephasing::sites(16, 0.3)];
    for generator in [Generator::Laplacian, Generator::Adjacency] {
        let plain = SymmetricOpenWalk::new(&g, generator, 0.8, &dephasing, &symmetries).unwrap();
        let built = plain
            .clone()
            .with_site_energies(&[0.0; 16])
            .unwrap()
            .with_losses(&[0.0; 16])
            .unwrap()
            .with_node_dephasing(&[0.0; 16])
            .unwrap()
            .with_traps(&[])
            .unwrap();
        let x0 = plain.localized(0).unwrap();
        assert_eq!(
            plain.evolve(&x0, 1.7).unwrap(),
            built.evolve(&x0, 1.7).unwrap()
        );
        // The order of the builders changes no bit either.
        let energies: Vec<f64> = weights(4).iter().map(|&k| 0.1 * k as f64).collect();
        let losses: Vec<f64> = weights(4).iter().map(|&k| 0.01 * k as f64).collect();
        let one = plain
            .clone()
            .with_site_energies(&energies)
            .unwrap()
            .with_losses(&losses)
            .unwrap()
            .with_traps(&[(15, 1.0)])
            .unwrap();
        let other = plain
            .clone()
            .with_traps(&[(15, 1.0)])
            .unwrap()
            .with_losses(&losses)
            .unwrap()
            .with_site_energies(&energies)
            .unwrap();
        assert_eq!(
            one.evolve(&x0, 1.7).unwrap(),
            other.evolve(&x0, 1.7).unwrap()
        );
        // Replacing terms by zeros restores the walk.
        let cleared = one
            .with_site_energies(&[0.0; 16])
            .unwrap()
            .with_losses(&[0.0; 16])
            .unwrap()
            .with_traps(&[])
            .unwrap();
        assert_eq!(
            plain.evolve(&x0, 1.7).unwrap(),
            cleared.evolve(&x0, 1.7).unwrap()
        );
        assert!(cleared.traps().is_empty());
    }
    // The hypercube's builders too.
    let plain = HypercubeOpenWalk::new(6, 0.7, 0.2, 0.1).unwrap();
    let built = plain
        .clone()
        .with_site_energies(&[0.0; 7])
        .unwrap()
        .with_losses(&[0.0; 7])
        .unwrap()
        .with_node_dephasing(&[0.0; 7])
        .unwrap()
        .with_traps(&[])
        .unwrap();
    let x0 = plain.localized();
    assert_eq!(
        plain.evolve(&x0, 1.3).unwrap(),
        built.evolve(&x0, 1.3).unwrap()
    );
}

#[test]
fn orbital_transport_matches_the_dense_transport() {
    let o = opts();
    // The star K₁,₅ (centre 0), start leaf 1, trap leaf 2: 17 orbitals.
    let star = generators::star(6);
    let symmetries = vec![vec![0, 1, 2, 4, 3, 5], vec![0, 1, 2, 4, 5, 3]];
    let walk = orbital_walk(&star, &symmetries)
        .with_traps(&[(2, 1.0)])
        .unwrap();
    assert_eq!(walk.num_orbitals(), 17);
    let r = walk.transport(1, &o).unwrap();
    check(&r, 0.25, 41.0 / 8.0, 1e-11, 1e-10, "star");
    assert_abs(r.dark, 0.75, 1e-11, "star: dark");
    assert_eq!(r.dimension, 17);
    assert_eq!(r.slowest_decay, None);
    let names: Vec<&str> = r.solves.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["adjoint", "forward", "adjoint time"]);
    let r = walk
        .clone()
        .with_site_energies(&[0.3, -0.2, 0.5, 0.1, 0.1, 0.1])
        .unwrap()
        .transport(1, &o)
        .unwrap();
    check(&r, 1.0, 126.3373611111, 1e-11, 1e-10, "star with energies");
    // m6e (20 orbitals), as the dense path.
    let m6e = Graph::from_weighted_edges(
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
    .unwrap();
    let loss = at(6, &[(1, 0.1), (2, 0.1)]);
    let deph = at(6, &[(3, 0.8), (4, 0.8), (1, 0.5), (2, 0.5)]);
    let walk = orbital_walk(&m6e, &[vec![0, 2, 1, 4, 3, 5]])
        .with_traps(&[(0, 1.0)])
        .unwrap()
        .with_losses(&loss)
        .unwrap()
        .with_node_dephasing(&deph)
        .unwrap();
    assert_eq!(walk.num_orbitals(), 20);
    let r = walk.transport(5, &o).unwrap();
    check(&r, 0.6257822277847, 19.48890801001, 1e-11, 1e-10, "m6e");
    assert_abs(r.dark, 0.0, 1e-11, "m6e: dark");
    let dense = dense_walk(&m6e)
        .with_traps(&[(0, 1.0)])
        .unwrap()
        .with_losses(&loss)
        .unwrap()
        .with_node_dephasing(&deph)
        .unwrap()
        .transport(5, &o)
        .unwrap();
    assert_abs(r.efficiency, dense.efficiency, 1e-11, "m6e against dense");
    assert_abs(r.lost, dense.lost, 1e-11, "m6e lost against dense");
    // C₈ with dephasing 0.6 at nodes 2 and 6, start 4, trap 0, under the
    // reflection fixing 0 and 4: 34 orbitals.
    let walk = orbital_walk(&generators::cycle(8), &[vec![0, 7, 6, 5, 4, 3, 2, 1]])
        .with_traps(&[(0, 1.0)])
        .unwrap()
        .with_node_dephasing(&at(8, &[(2, 0.6), (6, 0.6)]))
        .unwrap();
    assert_eq!(walk.num_orbitals(), 34);
    check(
        &walk.transport(4, &o).unwrap(),
        1.0,
        7.18065326633,
        1e-11,
        1e-10,
        "C8",
    );
    // Petersen: an orbit of three traps, and loss, dephasing and energy on
    // an orbit of six nodes; 15 orbitals.
    let (g, symmetries, start, traps, [loss, deph, eps]) = petersen_case();
    let walk = orbital_walk(&g, &symmetries)
        .with_traps(&traps)
        .unwrap()
        .with_losses(&loss)
        .unwrap()
        .with_node_dephasing(&deph)
        .unwrap()
        .with_site_energies(&eps)
        .unwrap();
    assert_eq!(walk.num_orbitals(), 15);
    let r = walk.transport(start, &o).unwrap();
    check(&r, 0.9225966512935, 3.0896360117, 1e-11, 1e-10, "Petersen");
    assert_eq!(r.trapped.len(), 3);
    for x in &r.trapped {
        assert_abs(*x, r.efficiency / 3.0, 1e-11, "Petersen: each trap");
    }
    assert_abs(r.lost, 1.0 - r.efficiency, 1e-11, "Petersen: lost");
    let names: Vec<&str> = r.solves.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        ["adjoint", "adjoint loss", "forward", "adjoint time"]
    );
    for s in &r.solves {
        assert!(s.backward_error <= 1e-13 && s.condition >= 1.0, "{s:?}");
    }
}

/// `K_N` with `H = −A`, start 0 and trap 1 (κ 1), reduced by the
/// stabiliser of the points `0..fixed`: generators `(fixed, fixed + 1)` and
/// the cycle `fixed → … → N − 1`.
fn complete(n: usize, fixed: usize) -> SymmetricOpenWalk {
    let swap: Vec<usize> = (0..n)
        .map(|u| match u {
            _ if u == fixed => fixed + 1,
            _ if u == fixed + 1 => fixed,
            _ => u,
        })
        .collect();
    let cycle: Vec<usize> = (0..n)
        .map(|u| {
            if u < fixed {
                u
            } else if u + 1 < n {
                u + 1
            } else {
                fixed
            }
        })
        .collect();
    orbital_walk(&generators::complete(n), &[swap, cycle])
}

/// `τ` with dephasing `γ` at node 2 of `K_N` (SymPy, Liouville space on the
/// three bright states).
fn tau_one(n: f64, g: f64) -> f64 {
    let num =
        2.0 * n.powi(3) * g * g + 4.0 * n.powi(3) * g - 10.0 * n * n * g * g - 16.0 * n * n * g
            + 4.0 * n * n
            + 19.0 * n * g * g
            + 28.0 * n * g
            - 8.0 * n
            - 12.0 * g * g
            - 20.0 * g
            + 4.0;
    num / (2.0 * g * (n - 2.0) * (n - 1.0) * (g + 2.0))
}

/// The four `K_N` cases of SymPy on the ten orbitals, with start `s = 0`,
/// trap `t = 1` and the rest `R`: `(κ, [ε, Γ, γ] on s, t, R)`.
fn sympy_classes(case: usize) -> (f64, [[f64; 3]; 3]) {
    match case {
        // "uniform": κ 1, Γ 1/20 and γ 1 everywhere.
        0 => (1.0, [[0.0; 3], [0.05; 3], [1.0; 3]]),
        // "classes".
        1 => (1.5, [[0.5, -1.0, 0.25], [0.1, 0.02, 0.05], [0.5, 2.0, 1.0]]),
        // No loss, γ = 1 on the rest only.
        _ => (1.0, [[0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]]),
    }
}

fn by_class(n: usize, v: [f64; 3]) -> Vec<f64> {
    (0..n).map(|u| v[u.min(2)]).collect()
}

fn sympy_walk(n: usize, case: usize) -> SymmetricOpenWalk {
    let (kappa, [eps, loss, deph]) = sympy_classes(case);
    complete(n, 2)
        .with_traps(&[(1, kappa)])
        .unwrap()
        .with_site_energies(&by_class(n, eps))
        .unwrap()
        .with_losses(&by_class(n, loss))
        .unwrap()
        .with_node_dephasing(&by_class(n, deph))
        .unwrap()
}

#[test]
fn complete_graphs_on_ten_and_seventeen_orbitals() {
    let o = opts();
    for n in [5usize, 64] {
        let nf = n as f64;
        let base = complete(n, 2).with_traps(&[(1, 1.0)]).unwrap();
        assert_eq!(base.num_orbitals(), 10);
        let r = base.transport(0, &o).unwrap();
        let tau = (nf * nf - 2.0 * nf + 3.0) / (2.0 * (nf - 1.0));
        check(
            &r,
            1.0 / (nf - 1.0),
            tau,
            1e-13,
            1e-10,
            &format!("K{n} none"),
        );
        assert_abs(r.dark, (nf - 2.0) / (nf - 1.0), 1e-13, "dark");
        // γ = 0.8 at node 2: the stabiliser of three points, 17 orbitals.
        let one = complete(n, 3)
            .with_traps(&[(1, 1.0)])
            .unwrap()
            .with_node_dephasing(&at(n, &[(2, 0.8)]))
            .unwrap();
        assert_eq!(one.num_orbitals(), 17);
        let r = one.transport(0, &o).unwrap();
        check(
            &r,
            1.0 / (nf - 2.0),
            tau_one(nf, 0.8),
            1e-13,
            1e-10,
            &format!("K{n} one"),
        );
        // γ = 1 everywhere.
        let all = base.clone().with_node_dephasing(&vec![1.0; n]).unwrap();
        let r = all.transport(0, &o).unwrap();
        check(
            &r,
            1.0,
            nf * (nf + 2.0) / 4.0,
            1e-13,
            1e-10,
            &format!("K{n} all"),
        );
        assert_eq!(r.dark, 0.0);
    }
    assert_rel(tau_one(5.0, 0.8), 229.0 / 42.0, 1e-14, "τ one at N = 5");
    assert_rel(
        tau_one(64.0, 0.8),
        2320471.0 / 36456.0,
        1e-14,
        "τ one at N = 64",
    );
    // The SymPy systems.
    let cases = [
        (0, 64, 0.018612019193312436, 19.602546924521368),
        (1, 4, 0.711833182052909, 4.086183444011),
        (1, 64, 0.012345371425390496, 16.080226555683083),
    ];
    for (case, n, eta, tau) in cases {
        let r = sympy_walk(n, case).transport(0, &o).unwrap();
        check(
            &r,
            eta,
            tau,
            1e-14,
            1e-11,
            &format!("SymPy case {case}, K{n}"),
        );
        assert_eq!(r.dark, 0.0);
        assert_abs(r.efficiency + r.lost, 1.0, 1e-13, "η + lost");
    }
    for n in [5usize, 64] {
        let nf = n as f64;
        let eta_formula = 672000.0 / (8400.0 * nf * nf + 16000.0 * nf + 675301.0);
        let tau_formula = 20.0 * (176400.0 * nf * nf + 320000.0 * nf + 137483.0)
            / (21.0 * (8400.0 * nf * nf + 16000.0 * nf + 675301.0));
        let r = sympy_walk(n, 0).transport(0, &o).unwrap();
        check(
            &r,
            eta_formula,
            tau_formula,
            1e-14,
            1e-11,
            &format!("uniform K{n}"),
        );
        let tau_rest = (4.0 * nf.powi(3) + 14.0 * nf * nf - 23.0 * nf - 6.0) / (12.0 * nf);
        let r = sympy_walk(n, 2).transport(0, &o).unwrap();
        check(&r, 1.0, tau_rest, 1e-14, 1e-11, &format!("rest K{n}"));
    }
}

/// The column chain of `Q_d`: the path of the `d + 1` symmetric states with
/// couplings `−√((k + 1)(d − k))`.
fn chain(d: usize) -> Graph {
    let edges: Vec<(usize, usize, f64)> = (0..d)
        .map(|k| (k, k + 1, (((k + 1) * (d - k)) as f64).sqrt()))
        .collect();
    Graph::from_weighted_edges(d + 1, &edges).unwrap()
}

#[test]
fn hypercube_with_a_trap_at_the_antipode() {
    let o = opts();
    // ε_k = 0.3k − 0.1k², Γ_k = 0.02 + 0.01k, κ = 1 at weight d (QuTiP on
    // the full cube).
    let refs = [
        (3, 0.922920860248438, 2.262737985267),
        (4, 0.898638996150815, 2.513636360296),
        (5, 0.868600640550131, 2.765796748651),
    ];
    for (d, eta, tau) in refs {
        let eps: Vec<f64> = (0..=d)
            .map(|k| 0.3 * k as f64 - 0.1 * (k * k) as f64)
            .collect();
        let loss: Vec<f64> = (0..=d).map(|k| 0.02 + 0.01 * k as f64).collect();
        let n = 1usize << d;
        let w = weights(d);
        let node = |v: &[f64]| -> Vec<f64> { w.iter().map(|&k| v[k]).collect() };
        let hyper = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0)
            .unwrap()
            .with_site_energies(&eps)
            .unwrap()
            .with_losses(&loss)
            .unwrap()
            .with_traps(&[(d, 1.0)])
            .unwrap();
        let g = generators::hypercube(d);
        let symmetric = orbital_walk(&g, &coordinate_permutations(d).unwrap())
            .with_site_energies(&node(&eps))
            .unwrap()
            .with_losses(&node(&loss))
            .unwrap()
            .with_traps(&[(n - 1, 1.0)])
            .unwrap();
        let dense = dense_walk(&g)
            .with_site_energies(&node(&eps))
            .unwrap()
            .with_losses(&node(&loss))
            .unwrap()
            .with_traps(&[(n - 1, 1.0)])
            .unwrap();
        let column = dense_walk(&chain(d))
            .with_site_energies(&eps)
            .unwrap()
            .with_losses(&loss)
            .unwrap()
            .with_traps(&[(d, 1.0)])
            .unwrap();
        let results = [
            ("hypercube", hyper.transport(&o).unwrap()),
            ("orbitals", symmetric.transport(0, &o).unwrap()),
            ("dense", dense.transport(0, &o).unwrap()),
            ("chain", column.transport(0, &o).unwrap()),
        ];
        for (what, r) in &results {
            check(r, eta, tau, 1e-12, 1e-11, &format!("Q{d} {what}"));
            assert_abs(r.lost, 1.0 - eta, 1e-12, &format!("Q{d} {what}: lost"));
        }
        assert_eq!(results[0].1.trapped.len(), 1);
        assert_abs(results[0].1.trapped[0], eta, 1e-12, "trapped");
    }
    // Q₄ with loss 0.05 and dephasing 0.5 everywhere, as a site rate and as
    // per-weight rates; the chain misses the dephased coherences.
    let a = HypercubeOpenWalk::new(4, 1.0, 0.5, 0.0)
        .unwrap()
        .with_losses(&[0.05; 5])
        .unwrap()
        .with_traps(&[(4, 1.0)])
        .unwrap();
    let b = HypercubeOpenWalk::new(4, 1.0, 0.0, 0.0)
        .unwrap()
        .with_losses(&[0.05; 5])
        .unwrap()
        .with_node_dephasing(&[0.5; 5])
        .unwrap()
        .with_traps(&[(4, 1.0)])
        .unwrap();
    for (what, w) in [("site rate", a), ("per weight", b)] {
        let r = w.transport(&o).unwrap();
        check(
            &r,
            0.7103501369857,
            4.4093444882,
            1e-11,
            1e-11,
            &format!("Q4 {what}"),
        );
    }
    let column = dense_walk(&chain(4))
        .with_losses(&[0.05; 5])
        .unwrap()
        .with_node_dephasing(&[0.5; 5])
        .unwrap()
        .with_traps(&[(4, 1.0)])
        .unwrap()
        .transport(0, &o)
        .unwrap();
    assert_abs(column.efficiency, 0.8496613708, 1e-10, "Q4 chain");
    // Q₁₀ (286 orbitals): coherent, then loss 0.05 and site dephasing.
    let coherent = HypercubeOpenWalk::new(10, 1.0, 0.0, 0.0)
        .unwrap()
        .with_traps(&[(10, 1.0)])
        .unwrap();
    let r = coherent.transport(&o).unwrap();
    check(&r, 1.0, 5.8380146133, 1e-11, 1e-8, "Q10 coherent");
    assert_eq!(r.dimension, 286);
    let table = [
        (0.0, 0.831038466686, 3.118369259),
        (0.1, 0.690879244942, 2.600291555),
        (1.0, 0.234277029669, 2.905175645),
        (10.0, 0.011425968764, 25.57861533),
    ];
    let lossy = |rate: f64| {
        HypercubeOpenWalk::new(10, 1.0, rate, 0.0)
            .unwrap()
            .with_losses(&[0.05; 11])
            .unwrap()
            .with_traps(&[(10, 1.0)])
            .unwrap()
    };
    let rates: Vec<f64> = table.iter().map(|t| t.0).collect();
    let scan = lossy(0.0).dephasing_scan(&rates, &o).unwrap();
    for (&(rate, eta, tau), s) in table.iter().zip(&scan) {
        let r = lossy(rate).transport(&o).unwrap();
        check(&r, eta, tau, 1e-11, 1e-8, &format!("Q10, γ = {rate}"));
        check(s, eta, tau, 1e-11, 1e-8, &format!("Q10 scan, γ = {rate}"));
        assert_eq!(s.added_dephasing, rate);
        assert_eq!(r.dark, 0.0);
    }
}

#[test]
fn the_forward_solve_is_refined_without_loss_everywhere() {
    // Q₃₀ (5456 orbitals) with losses on odd weights only, so that the
    // adjoint is not refined, and a trap at weight 15, whose share is read
    // off the coefficient of an orbital of C(30, 15) nodes times √C(30, 15)
    // ≈ 1.2·10⁴. Without refinement of the forward solve the backward-error
    // stop left that share wrong by 1.2·10⁻¹⁰ (τ by 4·10⁻¹¹ relative). The
    // reference adds a loss of 10⁻¹⁵ on even weights, which refines both
    // solves and moves the results by about 10⁻¹⁴.
    let d = 30;
    let walk = |even: f64| {
        HypercubeOpenWalk::new(d, 1.0, 0.1, 0.05)
            .unwrap()
            .with_site_energies(&(0..=d).map(|k| 0.2 * (k as f64).sin()).collect::<Vec<_>>())
            .unwrap()
            .with_losses(
                &(0..=d)
                    .map(|k| if k % 2 == 0 { even } else { 0.03 })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .with_node_dephasing(&(0..=d).map(|k| 0.01 * k as f64).collect::<Vec<_>>())
            .unwrap()
            .with_traps(&[(d, 1.0), (d / 2, 0.1)])
            .unwrap()
    };
    let r = walk(0.0).transport(&opts()).unwrap();
    let reference = walk(1e-15).transport(&opts()).unwrap();
    assert_eq!(r.solves[2].name, "forward");
    for (a, b) in r.trapped.iter().zip(&reference.trapped) {
        assert_abs(*a, *b, 1e-12, "trapped share");
    }
    assert_rel(
        r.arrival_time.unwrap(),
        reference.arrival_time.unwrap(),
        1e-11,
        "τ",
    );
    assert_abs(r.efficiency, reference.efficiency, 1e-12, "η");
    assert!(r.error <= 1e-11, "error {:e}", r.error);
}

#[test]
fn optimal_dephasing_on_orbitals_matches_the_dense_optimum() {
    // Q₃ with loss 0.05, the start detuned by 3 and the trap at the
    // antipode: dephasing helps up to a rate near 1, and the best uniform
    // rate on the 20 orbitals equals the dense path's.
    let o = opts();
    let hyper = HypercubeOpenWalk::new(3, 1.0, 0.0, 0.0)
        .unwrap()
        .with_losses(&[0.05; 4])
        .unwrap()
        .with_site_energies(&[3.0, 0.0, 0.0, 0.0])
        .unwrap()
        .with_traps(&[(3, 1.0)])
        .unwrap();
    let w = weights(3);
    let dense = dense_walk(&generators::hypercube(3))
        .with_losses(&[0.05; 8])
        .unwrap()
        .with_site_energies(
            &w.iter()
                .map(|&k| [3.0, 0.0, 0.0, 0.0][k])
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .with_traps(&[(7, 1.0)])
        .unwrap();
    let a = hyper.optimal_dephasing(0.01, 100.0, &o).unwrap();
    let b = dense.optimal_dephasing(0, 0.01, 100.0, &o).unwrap();
    assert!(a.interior && b.interior);
    assert_rel(a.rate, b.rate, 1e-6, "rate");
    assert_abs(a.result.efficiency, b.result.efficiency, 1e-12, "η*");
    // The rates agree to the width of the golden section; at the same rate
    // the arrival times agree.
    let same = &dense.dephasing_scan(0, &[a.rate], &o).unwrap()[0];
    assert_rel(
        a.result.arrival_time.unwrap(),
        same.arrival_time.unwrap(),
        1e-10,
        "τ*",
    );
    assert_eq!(a.result.added_dephasing, a.rate);
}

#[test]
fn orbital_traces_match_the_dense_trace() {
    let times = [0.0, 0.4, 1.3, 1.3, 5.0];
    let d = 3;
    let w = weights(d);
    let loss_w = [0.01, 0.05, 0.02, 0.03];
    let eps_w = [0.0, 0.3, -0.2, 0.1];
    let node = |v: &[f64]| -> Vec<f64> { w.iter().map(|&k| v[k]).collect() };
    let g = generators::hypercube(d);
    let symmetric = orbital_walk(&g, &coordinate_permutations(d).unwrap())
        .with_losses(&node(&loss_w))
        .unwrap()
        .with_site_energies(&node(&eps_w))
        .unwrap()
        .with_node_dephasing(&[0.4; 8])
        .unwrap()
        .with_traps(&[(7, 0.8)])
        .unwrap();
    let hyper = HypercubeOpenWalk::new(d, 1.0, 0.4, 0.0)
        .unwrap()
        .with_losses(&loss_w)
        .unwrap()
        .with_site_energies(&eps_w)
        .unwrap()
        .with_traps(&[(d, 0.8)])
        .unwrap();
    let dense = dense_walk(&g)
        .with_tolerance(1e-12)
        .with_losses(&node(&loss_w))
        .unwrap()
        .with_site_energies(&node(&eps_w))
        .unwrap()
        .with_node_dephasing(&[0.4; 8])
        .unwrap()
        .with_traps(&[(7, 0.8)])
        .unwrap();
    let a = symmetric.transport_trace(0, &times).unwrap();
    let b = hyper.transport_trace(&times).unwrap();
    let c = dense.transport_trace(0, &times).unwrap();
    for ((x, y), z) in a.iter().zip(&b).zip(&c) {
        assert_eq!((x.time, y.time), (z.time, z.time));
        assert_abs(x.trapped[0], z.trapped[0], 1e-9, "orbital sink");
        assert_abs(y.trapped[0], z.trapped[0], 1e-9, "hypercube sink");
        assert_abs(x.lost, z.lost, 1e-9, "orbital loss");
        assert_abs(y.lost, z.lost, 1e-9, "hypercube loss");
        for (u, (p, q)) in x.populations.iter().zip(&z.populations).enumerate() {
            assert_abs(*p, *q, 1e-9, &format!("population of node {u}"));
        }
        for (k, p) in y.populations.iter().enumerate() {
            let sum: f64 = (0..8)
                .filter(|&u| w[u] == k)
                .map(|u| z.populations[u])
                .sum();
            assert_abs(*p, sum, 1e-9, &format!("population of weight {k}"));
        }
        let total = x.populations.iter().sum::<f64>() + x.trapped[0] + x.lost;
        assert_abs(total, 1.0, 1e-12, "conservation");
    }
    assert_eq!(a[0].trapped, vec![0.0]);
    assert_eq!(a[2], a[3]);
    // Without traps the trace still runs (no sinks).
    let free = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0).unwrap();
    let s = free.transport_trace(&[1.0]).unwrap();
    assert!(s[0].trapped.is_empty() && s[0].lost == 0.0);
    assert_abs(s[0].populations.iter().sum(), 1.0, 1e-12, "free walk");
}

#[cfg(feature = "parallel")]
#[test]
fn orbital_results_do_not_depend_on_the_thread_count() {
    let w = HypercubeOpenWalk::new(12, 1.0, 0.3, 0.0)
        .unwrap()
        .with_losses(&[0.05; 13])
        .unwrap()
        .with_traps(&[(12, 1.0)])
        .unwrap();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| w.transport(&opts()).unwrap());
    assert_eq!(one, w.transport(&opts()).unwrap());
}

fn assert_invalid<T: std::fmt::Debug>(r: crate::error::Result<T>, name: &str) {
    match r {
        Err(QsoError::InvalidArgument { name: got, .. }) => assert_eq!(got, name),
        other => panic!("expected an invalid `{name}`, got {other:?}"),
    }
}

#[test]
fn refusals() {
    let o = opts();
    let star = generators::star(6);
    let symmetries = vec![vec![0, 1, 2, 4, 3, 5], vec![0, 1, 2, 4, 5, 3]];
    let walk = orbital_walk(&star, &symmetries);
    // Not invariant: energies, losses and dephasing differ on {3, 4, 5}.
    assert_invalid(
        walk.clone()
            .with_site_energies(&[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        "energies",
    );
    assert_invalid(
        walk.clone().with_losses(&[0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        "losses",
    );
    assert_invalid(
        walk.clone()
            .with_node_dephasing(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
        "dephasing",
    );
    // A trap orbit with unequal rates, and one trap of an orbit.
    assert_invalid(
        walk.clone().with_traps(&[(3, 1.0), (4, 1.0), (5, 0.5)]),
        "traps",
    );
    assert_invalid(walk.clone().with_traps(&[(3, 1.0)]), "traps");
    assert!(
        walk.clone()
            .with_traps(&[(3, 1.0), (4, 1.0), (5, 1.0)])
            .is_ok()
    );
    // Bad values and lengths.
    assert_invalid(walk.clone().with_traps(&[(2, 0.0)]), "traps");
    assert_invalid(walk.clone().with_traps(&[(2, 1.0), (2, 1.0)]), "traps");
    assert!(matches!(
        walk.clone().with_traps(&[(6, 1.0)]),
        Err(QsoError::NodeOutOfRange { node: 6, .. })
    ));
    assert!(matches!(
        walk.clone().with_losses(&[0.0; 5]),
        Err(QsoError::DimensionMismatch { .. })
    ));
    assert_invalid(walk.clone().with_losses(&[-0.1; 6]), "losses");
    assert_invalid(walk.clone().with_site_energies(&[f64::NAN; 6]), "energies");
    // No trap; a start that is not fixed; bad options and ranges.
    assert_invalid(walk.transport(1, &o), "traps");
    let trapped = walk.clone().with_traps(&[(2, 1.0)]).unwrap();
    assert_invalid(trapped.transport(3, &o), "node");
    assert!(matches!(
        trapped.transport(9, &o),
        Err(QsoError::NodeOutOfRange { node: 9, .. })
    ));
    let bad = TransportOptions {
        tolerance: 1e-3,
        ..Default::default()
    };
    assert_invalid(trapped.transport(1, &bad), "tolerance");
    assert_invalid(trapped.dephasing_scan(1, &[-1.0], &o), "rates");
    assert_invalid(trapped.optimal_dephasing(1, 1.0, 0.5, &o), "range");
    assert_invalid(trapped.transport_trace(1, &[1.0, 0.5]), "times");
    // The hypercube.
    let cube = HypercubeOpenWalk::new(4, 1.0, 0.0, 0.0).unwrap();
    assert_invalid(cube.clone().with_traps(&[(5, 1.0)]), "traps");
    assert_invalid(cube.clone().with_traps(&[(4, 1.0), (4, 2.0)]), "traps");
    assert!(matches!(
        cube.clone().with_site_energies(&[0.0; 4]),
        Err(QsoError::DimensionMismatch { .. })
    ));
    assert_invalid(
        cube.clone().with_node_dephasing(&[f64::NAN; 5]),
        "dephasing",
    );
    assert_invalid(cube.transport(&o), "traps");
    // Q₆₂ needs more band entries than allowed: refused before the band
    // storage is allocated.
    let big = HypercubeOpenWalk::new(62, 1.0, 0.0, 0.0)
        .unwrap()
        .with_traps(&[(62, 1.0)])
        .unwrap();
    match big.transport(&o) {
        Err(
            e @ QsoError::InvalidArgument {
                name: "symmetries", ..
            },
        ) => {
            let text = e.to_string();
            assert!(
                text.contains("needs 138290880 band entries (limit 134217728)"),
                "{text}"
            );
        }
        other => panic!("expected a refusal of the band entries, got {other:?}"),
    }
    // A product walk cannot hold a trapped or lossy factor.
    let edge = DephasingWalk::new(&generators::path(2), Generator::Laplacian, 1.0, 0.1).unwrap();
    assert_invalid(
        ProductOpenWalk::new(vec![(edge.clone().with_traps(&[(1, 1.0)]).unwrap(), 3)]),
        "factors",
    );
    assert_invalid(
        ProductOpenWalk::new(vec![(edge.clone().with_losses(&[0.0, 0.1]).unwrap(), 3)]),
        "factors",
    );
    assert!(ProductOpenWalk::new(vec![(edge.with_site_energies(&[0.2, 0.0]).unwrap(), 3)]).is_ok());
}

#[test]
fn a_dark_start_needs_no_forward_solve() {
    // The signed C₄ (weights 1, 1, 1, −1), trap 0, start 2 under the
    // reflection fixing 0 and 2: the start is dark.
    let g = Graph::from_weighted_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, -1.0)])
        .unwrap();
    // u ↦ −u mod 4 maps the edge (3, 0) of weight −1 to (1, 0) of weight
    // 1: not an automorphism, so this reduction uses the trivial group.
    let walk = orbital_walk(&g, &[]).with_traps(&[(0, 1.0)]).unwrap();
    let r = walk.transport(2, &opts()).unwrap();
    assert_eq!((r.efficiency, r.dark, r.lost), (0.0, 1.0, 0.0));
    assert!(r.arrival_time.is_none());
    assert_eq!(r.solves.len(), 1);
    // Dephasing at node 1 frees it: η 1, τ 377/20.
    let r = walk
        .with_node_dephasing(&at(4, &[(1, 0.5)]))
        .unwrap()
        .transport(2, &opts())
        .unwrap();
    check(&r, 1.0, 377.0 / 20.0, 1e-11, 1e-10, "signed C4");
}

/// Wall time, orbitals, bandwidth, iterations, condition and error of the
/// orbital transport (release mode: `cargo test --release -p
/// metatron-qso-rs --lib measure_orbital_transport -- --ignored
/// --nocapture`).
#[test]
#[ignore]
fn measure_orbital_transport() {
    let o = opts();
    for d in [10usize, 20, 30, 40] {
        for rate in [0.0, 0.1, 1.0, 10.0] {
            let walk = HypercubeOpenWalk::new(d, 1.0, rate, 0.0)
                .unwrap()
                .with_losses(&vec![0.05; d + 1])
                .unwrap()
                .with_traps(&[(d, 1.0)])
                .unwrap();
            let start = Instant::now();
            let r = walk.transport(&o).unwrap();
            let its: Vec<usize> = r.solves.iter().map(|s| s.iterations).collect();
            println!(
                "Q{d}, loss 0.05, γ {rate}: {} orbitals, {:.2} s, η {:.12} τ {:?}, its {its:?}, \
                 condition {:.1e}, error {:.1e}",
                r.dimension,
                start.elapsed().as_secs_f64(),
                r.efficiency,
                r.arrival_time,
                r.condition,
                r.error
            );
        }
    }
    for d in [10usize, 20, 40] {
        let walk = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0)
            .unwrap()
            .with_traps(&[(d, 1.0)])
            .unwrap();
        let start = Instant::now();
        let r = walk.transport(&o).unwrap();
        let its: Vec<usize> = r.solves.iter().map(|s| s.iterations).collect();
        println!(
            "Q{d} coherent: {:.2} s, η {:.12} τ {:?}, its {its:?}, error {:.1e}",
            start.elapsed().as_secs_f64(),
            r.efficiency,
            r.arrival_time,
            r.error
        );
    }
    // K₂₀₄₈ on ten orbitals (K₈₁₉₂ is in the weekly validation tests).
    let n = 2048;
    let start = Instant::now();
    let walk = complete(n, 2).with_traps(&[(1, 1.0)]).unwrap();
    let built = start.elapsed().as_secs_f64();
    let r = walk.transport(0, &o).unwrap();
    println!(
        "K{n} on 10 orbitals: new {built:.2} s, transport {:.3} s, η {:.3e} τ {:?}",
        start.elapsed().as_secs_f64() - built,
        r.efficiency,
        r.arrival_time
    );
}

#[test]
fn a_tiny_loss_on_every_node_is_a_uniform_shift() {
    // With the same small loss Γ on every node the trap-dark states decay
    // at Γ alone. Starting the forward solve from the start itself gave
    // τ = −10148 on K₂₅₆ at Γ = 10⁻¹² (error 10⁻⁹); from the bright part
    // of the start, as on the dense path, τ is right.
    let o = opts();
    for (n, gamma) in [(16usize, 1e-12), (256, 1e-12), (256, 1e-9)] {
        let orbital = complete(n, 2)
            .with_traps(&[(1, 1.0)])
            .unwrap()
            .with_losses(&vec![gamma; n])
            .unwrap()
            .transport(0, &o)
            .unwrap();
        let dense = dense_walk(&generators::complete(n))
            .with_traps(&[(1, 1.0)])
            .unwrap()
            .with_losses(&vec![gamma; n])
            .unwrap()
            .transport(0, &o)
            .unwrap();
        let what = format!("K{n}, Γ = {gamma:e}");
        assert_abs(orbital.efficiency, dense.efficiency, 1e-12, &what);
        assert_abs(orbital.lost, dense.lost, 1e-12, &what);
        assert_eq!(orbital.dark, 0.0);
        assert!(orbital.error <= 1e-11, "{what}: error {:e}", orbital.error);
        assert_rel(
            orbital.arrival_time.expect("τ"),
            dense.arrival_time.expect("τ"),
            1e-8,
            &format!("{what}: τ"),
        );
    }
    // Q₁₆ against its column chain.
    let d = 16;
    let gamma = 1e-12;
    let cube = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0)
        .unwrap()
        .with_losses(&vec![gamma; d + 1])
        .unwrap()
        .with_traps(&[(d, 1.0)])
        .unwrap()
        .transport(&o)
        .unwrap();
    let chain = dense_walk(&chain(d))
        .with_losses(&vec![gamma; d + 1])
        .unwrap()
        .with_traps(&[(d, 1.0)])
        .unwrap()
        .transport(0, &o)
        .unwrap();
    assert_abs(cube.efficiency, chain.efficiency, 1e-12, "Q16: η");
    assert_rel(
        cube.arrival_time.expect("τ"),
        chain.arrival_time.expect("τ"),
        1e-8,
        "Q16: τ",
    );
}

#[test]
fn tau_needs_the_goal_oriented_check() {
    // K₁₆, trap 1, loss 10⁻⁶ on nodes 2…15 only: a bright mode decays at
    // about the loss, and the forward read-out ⟨A, X⟩ gave τ = 7.5668
    // against the exact 14.0999807 (60-digit eigen-solve) with η's error
    // 10⁻⁷. The arrival time is now either right or withheld.
    let n = 16;
    let losses: Vec<f64> = (0..n).map(|u| if u >= 2 { 1e-6 } else { 0.0 }).collect();
    let r = complete(n, 2)
        .with_traps(&[(1, 1.0)])
        .unwrap()
        .with_losses(&losses)
        .unwrap()
        .transport(0, &opts())
        .unwrap();
    if let Some(tau) = r.arrival_time {
        assert_rel(tau, 14.0999807, 1e-3, "K16: τ");
    }
}

#[test]
fn a_dark_weight_that_may_be_slow_decay_counts_as_error() {
    // C₆₄ with site energies 0.7·cos(2πu/64), trap at 32 and the
    // reflection fixing 0: the symmetric sector is a path ending at the
    // trap, so η = 1 (τ = 1.56·10¹⁹), but some of its states decay below
    // rounding. The orbitals reported η = 0.7487 with dark 0.25 and error
    // 1.9·10⁻⁴ (the dense path refuses); now the dark weight counts in
    // `error` and τ is withheld.
    let n = 64;
    let g = generators::cycle(n);
    let reflection: Vec<usize> = (0..n).map(|u| (n - u) % n).collect();
    let energies: Vec<f64> = (0..n)
        .map(|u| 0.7 * (2.0 * std::f64::consts::PI * u as f64 / n as f64).cos())
        .collect();
    let walk = SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &[], &[reflection])
        .unwrap()
        .with_site_energies(&energies)
        .unwrap()
        .with_traps(&[(32, 1.0)])
        .unwrap();
    let r = walk.transport(0, &opts()).unwrap();
    assert!(r.dark > 0.1, "dark {}", r.dark);
    assert!(
        (r.efficiency - 1.0).abs() <= r.error,
        "η {} error {:e}",
        r.efficiency,
        r.error
    );
    assert_eq!(r.arrival_time, None);
}

#[test]
fn rates_must_vanish_on_whole_orbits() {
    // A star 0–1, 0–2 with the swap of 1 and 2: a trap of 5·10⁻¹³ at node
    // 2 alone passed the absolute tolerance and made η = 0 with error 0.
    let g = Graph::from_edges(3, &[(0, 1), (0, 2)]).unwrap();
    let walk = orbital_walk(&g, &[vec![0, 2, 1]])
        .with_node_dephasing(&[1.0; 3])
        .unwrap();
    assert_invalid(walk.clone().with_traps(&[(2, 5e-13)]), "traps");
    assert_invalid(walk.clone().with_traps(&[(1, 5e-13)]), "traps");
    assert_invalid(walk.clone().with_losses(&[0.0, 0.0, 1e-14]), "losses");
    assert_invalid(
        walk.clone().with_node_dephasing(&[1.0, 1e-14, 0.0]),
        "dephasing",
    );
    // Equal rates and energies within rounding pass.
    assert!(walk.clone().with_traps(&[(1, 0.1), (2, 0.1)]).is_ok());
    assert!(walk.with_site_energies(&[0.0, 6e-17, -1.8e-16]).is_ok());
}

#[test]
fn the_refinement_costs_at_most_the_solve() {
    // Coherent Q₂₀ with the trap at the antipode: the correction of the
    // forward solve took about 100 iterations more than the solve (119
    // against 19 for the adjoint) for no gain.
    let r = HypercubeOpenWalk::new(20, 1.0, 0.0, 0.0)
        .unwrap()
        .with_traps(&[(20, 1.0)])
        .unwrap()
        .transport(&opts())
        .unwrap();
    let forward = r.solves.iter().find(|s| s.name == "forward").unwrap();
    let adjoint = r.solves.iter().find(|s| s.name == "adjoint").unwrap();
    assert!(
        forward.iterations <= 4 * adjoint.iterations,
        "forward {} against adjoint {}",
        forward.iterations,
        adjoint.iterations
    );
    assert_abs(r.efficiency, 1.0, 1e-12, "Q20: η");
}
