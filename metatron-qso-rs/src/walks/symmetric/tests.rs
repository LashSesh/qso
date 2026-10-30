use super::*;
use crate::graph::automorphism::automorphism_group;
use crate::graph::generators;
use crate::walks::driven::{DrivenWalk, MagnusOptions, QuotientDrivenWalk, QuotientTerm, Term};
use crate::walks::{Coin, CoinedWalk, Quotient, QuotientWalk};

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

fn stabilizer_of_zero(g: &Graph) -> Vec<Vec<usize>> {
    g.stabilizer(&[0]).unwrap().generators().to_vec()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn orbital_counts() {
    // The stabiliser of a node of Q_d: C(d + 3, 3) orbitals on pairs.
    for d in 1..=6 {
        let g = generators::hypercube(d);
        let o = Orbitals::new(1 << d, &stabilizer_of_zero(&g)).unwrap();
        assert_eq!(o.len() as f64, binomial(d + 3, 3), "Q{d}");
        assert_eq!(o.of(0, 0), 0);
        assert_eq!((0..o.len()).map(|k| o.size(k)).sum::<usize>(), 1 << (2 * d));
    }
    // Trivial group: every pair; full group of K_n: diagonal and the rest.
    assert_eq!(Orbitals::new(5, &[]).unwrap().len(), 25);
    let k6 = automorphism_group(&generators::complete(6));
    let o = Orbitals::new(6, k6.generators()).unwrap();
    assert_eq!(o.len(), 2);
    assert_eq!((o.size(0), o.size(1)), (6, 30));
    // Representatives are the first pairs of their orbitals.
    for k in 0..o.len() {
        let (a, b) = o.representative(k);
        assert_eq!(o.of(a, b), k);
    }
}

#[test]
fn coordinate_permutations_generate_the_stabiliser() {
    for d in 1..=6 {
        let g = generators::hypercube(d);
        let gens = coordinate_permutations(d);
        assert!(gens.iter().all(|p| is_automorphism(&g, p) && p[0] == 0));
        let a = Orbitals::new(1 << d, &gens).unwrap();
        let b = Orbitals::new(1 << d, &stabilizer_of_zero(&g)).unwrap();
        assert_eq!(a.len(), b.len(), "Q{d}");
        for u in 0..1 << d {
            for v in 0..1 << d {
                assert_eq!(a.of(u, v), b.of(u, v));
            }
        }
    }
}

#[test]
fn stabiliser_orders() {
    let q4 = generators::hypercube(4);
    assert_eq!(q4.stabilizer(&[0]).unwrap().order(), Some(24));
    assert_eq!(q4.stabilizer(&[0, 15]).unwrap().order(), Some(24));
    assert_eq!(q4.stabilizer(&[0, 1]).unwrap().order(), Some(6));
    let p = generators::petersen();
    assert_eq!(p.stabilizer(&[0]).unwrap().order(), Some(12));
    assert_eq!(p.stabilizer(&[]).unwrap().order(), Some(120));
    for g in p.stabilizer(&[0, 1]).unwrap().generators() {
        assert!(is_automorphism(&p, g));
        assert_eq!((g[0], g[1]), (0, 1));
    }
    assert!(p.stabilizer(&[10]).is_err());
}

#[test]
fn coefficients_and_operators_round_trip() {
    let g = generators::petersen();
    let o = Orbitals::new(10, &stabilizer_of_zero(&g)).unwrap();
    let x: Vec<C64> = (0..o.len()).map(|k| C64::new(k as f64, 0.5)).collect();
    let m = o.operator(&x).unwrap();
    assert_eq!(o.coefficients(&m, 1e-12).unwrap(), x);
    let mut broken = m.clone();
    broken[(1, 2)] += C64::new(1.0, 0.0);
    assert!(o.coefficients(&broken, 1e-12).is_err());
    assert!(o.operator(&x[1..]).is_err());
}

#[test]
fn site_dephasing_matches_the_full_density_matrix() {
    for (g, t) in [
        (generators::petersen(), 2.3),
        (generators::hypercube(6), 1.7),
    ] {
        let n = g.num_nodes();
        for generator in [Generator::Laplacian, Generator::Adjacency] {
            let full = DephasingWalk::new(&g, generator, 0.8, 0.6).unwrap();
            let reduced = SymmetricOpenWalk::new(
                &g,
                generator,
                0.8,
                &[Dephasing::sites(n, 0.6)],
                &stabilizer_of_zero(&g),
            )
            .unwrap();
            assert!(reduced.num_orbitals() * 6 < n * n);
            let x = reduced.evolve(&reduced.localized(0).unwrap(), t).unwrap();
            let p = reduced.populations(&x).unwrap();
            let err = max_diff(&p, &full.populations(0, t).unwrap());
            assert!(err < 1e-9, "n = {n}: {err:e}");
            // The whole state, not only its diagonal.
            let rho = full
                .evolve(
                    &reduced
                        .orbitals()
                        .operator(&reduced.localized(0).unwrap())
                        .unwrap(),
                    t,
                )
                .unwrap();
            let y = reduced.orbitals().coefficients(&rho, 1e-8).unwrap();
            let err = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            assert!(err < 1e-9, "n = {n}: {err:e}");
        }
    }
}

#[test]
fn without_dephasing_q12_matches_the_quotient_walk() {
    let d = 12;
    let g = generators::hypercube(d);
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        1.0,
        &[],
        &coordinate_permutations(d),
    )
    .unwrap();
    assert_eq!(walk.num_orbitals(), 455);
    let quotient =
        QuotientWalk::new(&Quotient::hypercube(d).unwrap(), Generator::Laplacian, 1.0).unwrap();
    let t = 0.9;
    let p = walk
        .populations(&walk.evolve(&walk.localized(0).unwrap(), t).unwrap())
        .unwrap();
    let cells = quotient.cell_probabilities(0, t).unwrap();
    for (u, &pu) in p.iter().enumerate() {
        let k = (u as u32).count_ones() as usize;
        let expected = cells[k] / binomial(d, k);
        assert!(
            (pu - expected).abs() < 1e-11,
            "node {u}: {pu} vs {expected}"
        );
    }
}

/// `⟨Z⟩` of one qubit under `H = −γX` with dephasing `κ` from `|0⟩`.
fn bloch_z(gamma: f64, kappa: f64, t: f64) -> f64 {
    let w = (4.0 * gamma * gamma - kappa * kappa / 4.0).sqrt();
    (-kappa * t / 2.0).exp() * ((w * t).cos() + kappa / (2.0 * w) * (w * t).sin())
}

#[test]
fn per_coordinate_dephasing_on_q12_is_binomial() {
    let (d, gamma, kappa, t) = (12, 0.7, 0.9, 1.3);
    let n = 1 << d;
    let g = generators::hypercube(d);
    let dephasing: Vec<Dephasing> = (0..d).map(|k| Dephasing::coordinate(n, k, kappa)).collect();
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        gamma,
        &dephasing,
        &coordinate_permutations(d),
    )
    .unwrap();
    let p = walk
        .populations(&walk.evolve(&walk.localized(0).unwrap(), t).unwrap())
        .unwrap();
    let flip = (1.0 - bloch_z(gamma, kappa, t)) / 2.0;
    for (u, &pu) in p.iter().enumerate() {
        let k = (u as u32).count_ones() as i32;
        let expected = flip.powi(k) * (1.0 - flip).powi(d as i32 - k);
        assert!(
            (pu - expected).abs() < 1e-12,
            "node {u}: {pu} vs {expected}"
        );
    }
}

#[test]
fn product_walks_at_a_thousand_coordinates() {
    let (gamma, kappa, t) = (0.7, 0.9, 1.3);
    let edge =
        DephasingWalk::new(&generators::complete(2), Generator::Laplacian, gamma, kappa).unwrap();
    let product = ProductOpenWalk::new(vec![(edge, 1000)]).unwrap();
    let marginals = product.marginals(&[0], t).unwrap();
    let flip = (1.0 - bloch_z(gamma, kappa, t)) / 2.0;
    assert!((marginals[0][1] - flip).abs() < 1e-10);
    let dist = product.count_distribution(&marginals, &[1]).unwrap();
    assert_eq!(dist.len(), 1001);
    assert!((dist.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let mean: f64 = dist.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
    assert!((mean - 1000.0 * flip).abs() < 1e-7);
    // The binomial law, through its log at the mode.
    let q = marginals[0][1];
    let k = (1000.0 * q) as usize;
    let log = (1..=1000).map(|i| (i as f64).ln()).sum::<f64>()
        - (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
        - (1..=1000 - k).map(|i| (i as f64).ln()).sum::<f64>()
        + k as f64 * q.ln()
        + (1000 - k) as f64 * (1.0 - q).ln();
    assert!((dist[k] - log.exp()).abs() < 1e-12 * 1e3);
}

#[test]
fn product_walks_match_the_trivial_group() {
    // P₃ □ K₂ with site dephasing on each factor at different rates.
    let (path, edge) = (generators::path(3), generators::complete(2));
    let g = path.cartesian_product(&edge).unwrap();
    let n = g.num_nodes();
    // Node (i, j) of the product is i·2 + j.
    let first: Vec<usize> = (0..n).map(|u| u / 2).collect();
    let second: Vec<usize> = (0..n).map(|u| u % 2).collect();
    for u in 0..n {
        for &(v, _) in g.neighbors(u) {
            assert!(first[u] == first[v] || second[u] == second[v]);
        }
    }
    let (gamma, t) = (0.6, 1.9);
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        gamma,
        &[
            Dephasing {
                rate: 0.4,
                block: first,
            },
            Dephasing {
                rate: 1.1,
                block: second,
            },
        ],
        &[],
    )
    .unwrap();
    assert_eq!(walk.num_orbitals(), n * n);
    let p = walk
        .populations(&walk.evolve(&walk.localized(0).unwrap(), t).unwrap())
        .unwrap();
    let product = ProductOpenWalk::new(vec![
        (
            DephasingWalk::new(&path, Generator::Laplacian, gamma, 0.4).unwrap(),
            1,
        ),
        (
            DephasingWalk::new(&edge, Generator::Laplacian, gamma, 1.1).unwrap(),
            1,
        ),
    ])
    .unwrap();
    let m = product.marginals(&[0, 0], t).unwrap();
    for (u, &pu) in p.iter().enumerate() {
        let expected = m[0][u / 2] * m[1][u % 2];
        assert!((pu - expected).abs() < 1e-9, "node {u}: {pu} vs {expected}");
    }
}

#[test]
fn coined_walks_match_the_full_density_matrix() {
    for (g, position, coin) in [
        (generators::hypercube(4), 0.2, 0.0),
        (generators::petersen(), 0.3, 0.0),
        (generators::complete(5), 0.25, 0.0),
    ] {
        let walk = CoinedWalk::new(&g, Coin::Grover).unwrap();
        let full = DecoherentCoinedWalk::new(walk, coin, position).unwrap();
        let reduced = SymmetricCoinedWalk::new(&full, &stabilizer_of_zero(&g)).unwrap();
        let m = full.walk().num_arcs();
        assert!(reduced.num_orbitals() < m * m / 4);
        let steps = 9;
        let x = reduced
            .evolve(&reduced.localized(0).unwrap(), steps)
            .unwrap();
        let p = reduced.node_probabilities(&x).unwrap();
        let err = max_diff(&p, &full.probabilities(0, steps).unwrap());
        assert!(err < 1e-12, "{err:e}");
    }
}

#[test]
fn coin_dephasing_needs_symmetries_that_keep_the_labels() {
    let g = generators::hypercube(4);
    let walk = CoinedWalk::new(&g, Coin::Grover).unwrap();
    let full = DecoherentCoinedWalk::new(walk, 0.3, 0.0).unwrap();
    assert!(SymmetricCoinedWalk::new(&full, &stabilizer_of_zero(&g)).is_err());
    // Without symmetries every pair is its own orbital and the walk agrees.
    let reduced = SymmetricCoinedWalk::new(&full, &[]).unwrap();
    let p = reduced
        .node_probabilities(&reduced.evolve(&reduced.localized(0).unwrap(), 6).unwrap())
        .unwrap();
    assert!(max_diff(&p, &full.probabilities(0, 6).unwrap()) < 1e-12);
}

#[test]
fn rejects_what_is_not_symmetric() {
    let g = generators::petersen();
    let sym = stabilizer_of_zero(&g);
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        1.0,
        &[Dephasing::sites(10, 0.1)],
        &sym,
    )
    .unwrap();
    // Node 1 is not fixed by the stabiliser of node 0.
    assert!(walk.localized(1).is_err());
    assert!(walk.localized(10).is_err());
    // A permutation that is not an automorphism.
    let mut swap: Vec<usize> = (0..10).collect();
    swap.swap(0, 1);
    swap.swap(2, 3);
    assert!(!is_automorphism(&g, &swap));
    assert!(SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &[], &[swap]).is_err());
    assert!(SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &[], &[vec![0; 10]]).is_err());
    // Dephasing of one coordinate of Q₄ is not invariant under the stabiliser.
    let q4 = generators::hypercube(4);
    let r = SymmetricOpenWalk::new(
        &q4,
        Generator::Laplacian,
        1.0,
        &[Dephasing::coordinate(16, 0, 0.5)],
        &stabilizer_of_zero(&q4),
    );
    assert!(r.is_err());
    assert!(SymmetricOpenWalk::new(&q4, Generator::Laplacian, 0.0, &[], &[]).is_err());
    assert!(walk.evolve(&walk.uniform(), -1.0).is_err());
    assert!(walk.evolve(&walk.uniform()[1..], 1.0).is_err());
}

#[test]
fn stationary_states() {
    let g = generators::petersen();
    let sym = automorphism_group(&g).generators().to_vec();
    // Without dephasing the uniform superposition is an eigenstate.
    let walk = SymmetricOpenWalk::new(&g, Generator::Adjacency, 1.0, &[], &sym).unwrap();
    assert_eq!(walk.num_orbitals(), 3);
    let x = walk.evolve(&walk.uniform(), 5.0).unwrap();
    for (a, b) in x.iter().zip(&walk.uniform()) {
        assert!((a - b).norm() < 1e-13);
    }
    // With dephasing its coherences decay to the maximally mixed state,
    // which stays.
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Adjacency,
        1.0,
        &[Dephasing::sites(10, 0.4)],
        &sym,
    )
    .unwrap();
    let mixed = walk
        .orbitals()
        .coefficients(&(CMatrix::identity(10, 10) * C64::new(0.1, 0.0)), 0.0);
    let mixed = mixed.unwrap();
    let x = walk.evolve(&mixed, 5.0).unwrap();
    assert!(x.iter().zip(&mixed).all(|(a, b)| (a - b).norm() < 1e-13));
    let x = walk.evolve(&walk.uniform(), 80.0).unwrap();
    assert!(x.iter().zip(&mixed).all(|(a, b)| (a - b).norm() < 1e-9));
}

#[test]
fn driven_quotient_walks_match_the_full_walk() {
    let g = generators::hypercube(8);
    let n = g.num_nodes();
    let q = Quotient::with_singletons(&g, &[0]).unwrap();
    assert_eq!(q.num_cells(), 9);
    let reduced = QuotientDrivenWalk::new(
        &q,
        vec![
            QuotientTerm::Graph(Generator::Laplacian),
            QuotientTerm::Cells(vec![0]),
            QuotientTerm::Uniform,
            QuotientTerm::Potential((0..9).map(|k| 0.1 * k as f64).collect()),
        ],
    )
    .unwrap();
    let potential: Vec<f64> = (0..n)
        .map(|u| 0.1 * (u as u32).count_ones() as f64)
        .collect();
    let full = DrivenWalk::new(
        n,
        vec![
            Term::graph(&g, Generator::Laplacian),
            Term::projector(n, &[0]).unwrap(),
            Term::Uniform,
            Term::potential(&potential).unwrap(),
        ],
    )
    .unwrap();
    let c = |t: f64| {
        Ok(vec![
            0.3 + 0.2 * t,
            -2.0 * t.sin(),
            -1.0 + 0.5 * t,
            (0.7 * t).cos(),
        ])
    };
    let opts = MagnusOptions {
        fixed_step: Some(0.02),
        tol: 1e-12,
        ..MagnusOptions::default()
    };
    let small = reduced
        .evolve(&reduced.uniform_state(), 0.0, 3.0, c, &opts)
        .unwrap();
    let psi0 = vec![C64::new(1.0 / (n as f64).sqrt(), 0.0); n];
    let large = full.evolve(&psi0, 0.0, 3.0, c, &opts).unwrap();
    let lifted = q.lift(&small.state).unwrap();
    let err = lifted
        .iter()
        .zip(&large.state)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "{err:e}");
    let p = reduced.cell_probabilities(&small.state);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    // Errors.
    assert!(QuotientDrivenWalk::new(&q, vec![QuotientTerm::Cells(vec![9])]).is_err());
    assert!(QuotientDrivenWalk::new(&q, vec![QuotientTerm::Potential(vec![0.0; 3])]).is_err());
    assert!(QuotientDrivenWalk::new(&q, vec![]).is_err());
    assert!(
        Quotient::hypercube(3)
            .unwrap()
            .lift(&[C64::new(1.0, 0.0); 4])
            .is_err()
    );
}

#[test]
fn adiabatic_search_on_a_hypercube_of_a_billion_nodes() {
    // Interpolating search on Q₃₀ (2³⁰ nodes, 31 cells): from the uniform
    // superposition under H(s) = −(1 − s)|s⟩⟨s| − s|0⟩⟨0| with a slow
    // linear schedule the walker ends at the marked node's cell.
    let q = Quotient::hypercube(30).unwrap();
    let walk = QuotientDrivenWalk::new(
        &q,
        vec![QuotientTerm::Uniform, QuotientTerm::Cells(vec![0])],
    )
    .unwrap();
    let n = (1u64 << 30) as f64;
    let (eps, rn) = (0.1, (n - 1.0).sqrt());
    let total = n / (eps * rn) * rn.atan();
    let s = |t: f64| 0.5 * (1.0 + ((2.0 * eps * rn * t / n - rn.atan()).tan()) / rn);
    let r = walk
        .evolve(
            &walk.uniform_state(),
            0.0,
            total,
            |t| Ok(vec![-(1.0 - s(t)), -s(t)]),
            &MagnusOptions::default(),
        )
        .unwrap();
    let p = walk.cell_probabilities(&r.state);
    assert!(p[0] > 0.98, "{}", p[0]);
}
