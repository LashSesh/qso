use super::*;
use crate::graph::automorphism::automorphism_group;
use crate::graph::generators;
use crate::rng::Rng;
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
        let gens = coordinate_permutations(d).unwrap();
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
        &coordinate_permutations(d).unwrap(),
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
        &coordinate_permutations(d).unwrap(),
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
    let p = reduced.cell_probabilities(&small.state).unwrap();
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
    let p = walk.cell_probabilities(&r.state).unwrap();
    assert!(p[0] > 0.98, "{}", p[0]);
}

/// The coordinate every arc of `Q_d` flips.
fn direction_labels(walk: &CoinedWalk) -> Vec<usize> {
    walk.arcs()
        .iter()
        .map(|&(u, v)| (u ^ v).trailing_zeros() as usize)
        .collect()
}

#[test]
fn coin_dephasing_with_direction_labels_reduces_on_hypercubes() {
    let g = generators::hypercube(4);
    let walk = CoinedWalk::new(&g, Coin::Grover).unwrap();
    let labels = direction_labels(&walk);
    let full = DecoherentCoinedWalk::new(walk, 0.3, 0.2)
        .unwrap()
        .with_coin_labels(labels)
        .unwrap();
    for symmetries in [stabilizer_of_zero(&g), coordinate_permutations(4).unwrap()] {
        let reduced = SymmetricCoinedWalk::new(&full, &symmetries).unwrap();
        assert!(reduced.num_orbitals() * 8 < 64 * 64);
        let x = reduced.evolve(&reduced.localized(0).unwrap(), 9).unwrap();
        let p = reduced.node_probabilities(&x).unwrap();
        assert!(max_diff(&p, &full.probabilities(0, 9).unwrap()) < 1e-12);
    }
}

#[test]
fn full_coin_dephasing_is_the_classical_chain_on_arcs() {
    // With the coin measured every step the arc populations follow the
    // Markov chain P(a → b) = |U_ba|², here on Q₇ (896 arcs).
    let d = 7;
    let g = generators::hypercube(d);
    let walk = CoinedWalk::new(&g, Coin::Grover).unwrap();
    let m = walk.num_arcs();
    let mut chain = vec![vec![0.0; m]; m];
    let mut e = vec![C64::new(0.0, 0.0); m];
    for a in 0..m {
        e[a] = C64::new(1.0, 0.0);
        for (b, z) in walk.step(&e).unwrap().into_iter().enumerate() {
            chain[a][b] = z.norm_sqr();
        }
        e[a] = C64::new(0.0, 0.0);
    }
    let labels = direction_labels(&walk);
    let tails: Vec<usize> = walk.arcs().iter().map(|a| a.0).collect();
    let open = DecoherentCoinedWalk::new(walk, 1.0, 0.0)
        .unwrap()
        .with_coin_labels(labels)
        .unwrap();
    let reduced = SymmetricCoinedWalk::new(&open, &coordinate_permutations(d).unwrap()).unwrap();
    // The first step starts from a pure coin state; from then on the
    // coherences between different coin states are gone.
    let mut x = reduced.evolve(&reduced.localized(0).unwrap(), 1).unwrap();
    let o = reduced.orbitals();
    let mut p: Vec<f64> = (0..m).map(|a| x[o.of(a, a)].re).collect();
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for step in 2..=12 {
        x = reduced.evolve(&x, 1).unwrap();
        let mut next = vec![0.0; m];
        for a in 0..m {
            for b in 0..m {
                next[b] += p[a] * chain[a][b];
            }
        }
        p = next;
        let mut nodes = vec![0.0; 1 << d];
        for a in 0..m {
            nodes[tails[a]] += p[a];
        }
        let err = max_diff(&reduced.node_probabilities(&x).unwrap(), &nodes);
        assert!(err < 1e-12, "step {step}: {err:e}");
    }
}

#[test]
fn orbitals_match_the_orbit_closure() {
    // Random permutations of up to 9 points: orbitals by breadth-first
    // closure of every pair under the generators.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |m: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as usize
    };
    for _ in 0..200 {
        let n = 1 + next(9);
        let gens: Vec<Vec<usize>> = (0..next(3))
            .map(|_| {
                let mut p: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    p.swap(i, next(i + 1));
                }
                p
            })
            .collect();
        let o = Orbitals::new(n, &gens).unwrap();
        let mut label = vec![usize::MAX; n * n];
        let mut count = 0;
        for start in 0..n * n {
            if label[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            label[start] = count;
            while let Some(p) = stack.pop() {
                for g in &gens {
                    let q = g[p / n] * n + g[p % n];
                    if label[q] == usize::MAX {
                        label[q] = count;
                        stack.push(q);
                    }
                }
            }
            count += 1;
        }
        assert_eq!(o.len(), count);
        for (p, &l) in label.iter().enumerate() {
            // Both number orbitals by their first pair.
            assert_eq!(o.of(p / n, p % n), l);
        }
    }
}

#[test]
fn complex_coins_match_the_full_density_matrix() {
    // A coin invariant under exchanging the two arcs, on C₈ with the
    // reflection u ↦ −u, compared coefficient by coefficient.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let coin = CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(s, 0.0),
            C64::new(0.0, s),
            C64::new(0.0, s),
            C64::new(s, 0.0),
        ],
    );
    let g = generators::cycle(8);
    let reflection: Vec<usize> = (0..8).map(|u| (8 - u) % 8).collect();
    let full =
        DecoherentCoinedWalk::new(CoinedWalk::new(&g, Coin::Unitary(coin)).unwrap(), 0.0, 0.15)
            .unwrap();
    let reduced = SymmetricCoinedWalk::new(&full, &[reflection]).unwrap();
    let x0 = reduced.localized(0).unwrap();
    let o = reduced.orbitals();
    let mut rho = o.operator(&x0).unwrap();
    let mut complex = false;
    for steps in 1..=7 {
        rho = full.step(&rho).unwrap();
        let x = reduced.evolve(&x0, steps).unwrap();
        let y = o.coefficients(&rho, 1e-12).unwrap();
        let err = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-14, "step {steps}: {err:e}");
        complex |= x.iter().any(|z| z.im.abs() > 1e-3);
    }
    // The coin makes the coefficients complex, so a missing conjugate
    // would show.
    assert!(complex);
}

#[test]
fn coins_that_are_not_invariant_are_rejected() {
    let g = generators::petersen();
    let sym = stabilizer_of_zero(&g);
    let fourier =
        DecoherentCoinedWalk::new(CoinedWalk::new(&g, Coin::Fourier).unwrap(), 0.0, 0.0).unwrap();
    assert!(SymmetricCoinedWalk::new(&fourier, &sym).is_err());
    let marked = |node: usize| {
        let walk = CoinedWalk::new(&g, Coin::Grover)
            .unwrap()
            .with_marked(&[node])
            .unwrap();
        DecoherentCoinedWalk::new(walk, 0.0, 0.2).unwrap()
    };
    assert!(SymmetricCoinedWalk::new(&marked(1), &sym).is_err());
    let full = marked(0);
    let reduced = SymmetricCoinedWalk::new(&full, &sym).unwrap();
    let p = reduced
        .node_probabilities(&reduced.evolve(&reduced.localized(0).unwrap(), 8).unwrap())
        .unwrap();
    assert!(max_diff(&p, &full.probabilities(0, 8).unwrap()) < 1e-12);
    let mut nan = reduced.localized(0).unwrap();
    nan[0] = C64::new(f64::NAN, 0.0);
    assert!(reduced.evolve(&nan, 1).is_err());
}

#[test]
fn limits_and_invalid_input() {
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
    // The work guard.
    assert!(walk.evolve(&walk.localized(0).unwrap(), 1e6).is_err());
    let o = walk.orbitals();
    let m = o.operator(&walk.uniform()).unwrap();
    assert!(o.coefficients(&m, f64::NAN).is_err());
    assert!(o.coefficients(&m, -1.0).is_err());
    assert!(coordinate_permutations(21).is_err());
    assert_eq!(Dephasing::coordinate(4, 70, 1.0).block, vec![0; 4]);
    // Marginals that are not populations.
    let edge =
        DephasingWalk::new(&generators::complete(2), Generator::Laplacian, 1.0, 0.0).unwrap();
    let product = ProductOpenWalk::new(vec![(edge, 3)]).unwrap();
    assert!(product.count_distribution(&[vec![0.0, 2.0]], &[1]).is_err());
    assert!(product.count_distribution(&[vec![0.5]], &[0]).is_err());
    assert!(product.count_distribution(&[], &[]).is_err());
}

#[test]
#[cfg_attr(target_arch = "wasm32", ignore = "a panic aborts under WASI")]
fn of_panics_outside_the_points() {
    let o = Orbitals::new(3, &[]).unwrap();
    assert!(std::panic::catch_unwind(|| o.of(0, 3)).is_err());
}

#[test]
fn driven_adjacency_walks_on_quotients() {
    let g = generators::petersen();
    let q = Quotient::with_singletons(&g, &[0]).unwrap();
    let reduced = QuotientDrivenWalk::new(
        &q,
        vec![
            QuotientTerm::Graph(Generator::Adjacency),
            QuotientTerm::Cells(vec![0, 0]),
        ],
    )
    .unwrap();
    let full = DrivenWalk::new(
        10,
        vec![
            Term::graph(&g, Generator::Adjacency),
            Term::projector(10, &[0]).unwrap(),
        ],
    )
    .unwrap();
    let c = |t: f64| Ok(vec![1.0 + 0.3 * t, (2.0 * t).cos()]);
    let opts = MagnusOptions {
        fixed_step: Some(0.01),
        ..MagnusOptions::default()
    };
    let mut c0 = vec![C64::new(0.0, 0.0); q.num_cells()];
    c0[q.cell_of().unwrap()[0]] = C64::new(1.0, 0.0);
    let small = reduced.evolve(&c0, 0.0, 2.5, c, &opts).unwrap();
    let mut psi0 = vec![C64::new(0.0, 0.0); 10];
    psi0[0] = C64::new(1.0, 0.0);
    let large = full.evolve(&psi0, 0.0, 2.5, c, &opts).unwrap();
    let lifted = q.lift(&small.state).unwrap();
    let err = lifted
        .iter()
        .zip(&large.state)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "{err:e}");
    assert!(reduced.cell_probabilities(&small.state[1..]).is_err());
}

/// The orbitals from point stabilisers equal those of the union–find on
/// all pairs: the same numbering, representatives, sizes and labels, with
/// all Schreier generators and with random elements only.
fn assert_same_orbitals(n: usize, gens: &[Vec<usize>], what: &str) {
    let pairs = Orbitals::new(n, gens).unwrap();
    assert!(matches!(pairs.labels, Labels::Pairs(_)));
    for exhaustive in [usize::MAX, 0] {
        for seed in [0, 7] {
            let (s, reps, sizes) =
                Suborbits::with_exhaustive_work(n, gens, seed, exhaustive).unwrap();
            assert_eq!(reps.len(), pairs.len(), "{what}");
            for (o, (&(a, b), &size)) in reps.iter().zip(&sizes).enumerate() {
                assert_eq!((a as usize, b as usize), pairs.representative(o), "{what}");
                assert_eq!(size as usize, pairs.size(o), "{what}");
            }
            for u in 0..n {
                for v in 0..n {
                    assert_eq!(s.of(u, v), pairs.of(u, v), "{what}: ({u}, {v})");
                }
            }
        }
    }
}

/// Generators of the stabiliser of `0…0` in the automorphisms of `H(d, q)`
/// (node `x` has digits `x` in base `q`): two that permute the coordinates,
/// two that permute the nonzero symbols of the first.
fn hamming_stabiliser(d: usize, q: usize) -> Vec<Vec<usize>> {
    let n = q.pow(d as u32);
    let digits = |x: usize| -> Vec<usize> { (0..d).map(|k| x / q.pow(k as u32) % q).collect() };
    let number = |v: &[usize]| -> usize { v.iter().rev().fold(0, |acc, &c| acc * q + c) };
    let apply = |f: &dyn Fn(Vec<usize>) -> Vec<usize>| -> Vec<usize> {
        (0..n).map(|x| number(&f(digits(x)))).collect()
    };
    let mut gens = Vec::new();
    if d > 1 {
        gens.push(apply(&|mut v| {
            v.swap(0, 1);
            v
        }));
        gens.push(apply(&|v| (0..d).map(|k| v[(k + d - 1) % d]).collect()));
    }
    if q > 2 {
        gens.push(apply(&|mut v| {
            v[0] = match v[0] {
                1 => 2,
                2 => 1,
                c => c,
            };
            v
        }));
        gens.push(apply(&|mut v| {
            if v[0] > 0 {
                v[0] = v[0] % (q - 1) + 1;
            }
            v
        }));
    }
    gens
}

#[test]
fn orbitals_from_stabilisers_equal_those_of_all_pairs() {
    // Fixed points, transitive and intransitive groups, the trivial group.
    let petersen = generators::petersen();
    assert_same_orbitals(
        10,
        &stabilizer_of_zero(&petersen),
        "Petersen, stabiliser of 0",
    );
    assert_same_orbitals(10, petersen.automorphism_group().generators(), "Petersen");
    for d in 3..=6 {
        assert_same_orbitals(1 << d, &coordinate_permutations(d).unwrap(), "Q_d");
    }
    let q4 = generators::hypercube(4);
    assert_same_orbitals(16, q4.automorphism_group().generators(), "Q4");
    let j = generators::johnson(7, 3).unwrap();
    assert_same_orbitals(35, j.automorphism_group().generators(), "J(7, 3)");
    assert_same_orbitals(35, &stabilizer_of_zero(&j), "J(7, 3), stabiliser of 0");
    let paley = generators::paley(13).unwrap();
    assert_same_orbitals(13, paley.automorphism_group().generators(), "Paley(13)");
    let c = generators::cycle(12);
    assert_same_orbitals(12, c.automorphism_group().generators(), "C12");
    assert_same_orbitals(64, &hamming_stabiliser(3, 4), "H(3, 4), stabiliser of 0");
    assert_same_orbitals(7, &[], "trivial");
    // Groups generated by random cycles of a few points, most points fixed.
    let mut rng = Rng::seed_from_u64(48);
    for trial in 0..60 {
        let n = 2 + rng.below(40) as usize;
        let gens: Vec<Vec<usize>> = (0..1 + rng.below(3))
            .map(|_| {
                let mut p: Vec<usize> = (0..n).collect();
                let mut points: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut points);
                let len = 2 + rng.below(5.min(n as u64 - 1)) as usize;
                for w in 0..len {
                    p[points[w]] = points[(w + 1) % len];
                }
                p
            })
            .collect();
        assert_same_orbitals(n, &gens, &format!("random {trial}"));
    }
}

#[test]
fn large_hypercubes_follow_the_walk_on_counts() {
    // Q₁₄ (16 384 nodes, beyond the union–find on pairs) with site and
    // coordinate dephasing: the C(17, 3) = 680 orbitals of the coordinate
    // permutations, and the distance distribution of HypercubeOpenWalk.
    let d = 14;
    let n = 1 << d;
    let g = generators::hypercube(d);
    let mut dephasing = vec![Dephasing::sites(n, 0.3)];
    dephasing.extend((0..d).map(|k| Dephasing::coordinate(n, k, 0.2)));
    let symmetries = coordinate_permutations(d).unwrap();
    let walk =
        SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &dephasing, &symmetries).unwrap();
    assert!(matches!(walk.orbitals().labels, Labels::Suborbits(_)));
    assert_eq!(walk.num_orbitals() as f64, binomial(d + 3, 3));
    let reference = HypercubeOpenWalk::new(d, 1.0, 0.3, 0.2).unwrap();
    for t in [0.3, 1.1] {
        let p = walk
            .populations(&walk.evolve(&walk.localized(0).unwrap(), t).unwrap())
            .unwrap();
        let mut by_distance = vec![0.0; d + 1];
        for (u, x) in p.iter().enumerate() {
            by_distance[u.count_ones() as usize] += x;
        }
        let expected = reference
            .distance_distribution(&reference.evolve(&reference.localized(), t).unwrap())
            .unwrap();
        assert!(max_diff(&by_distance, &expected) < 1e-12, "t = {t}");
    }
    // Dephasing that the symmetries do not keep is refused.
    let uneven = vec![Dephasing::coordinate(n, 0, 0.2)];
    assert!(SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &uneven, &symmetries).is_err());
    // Terms count with their multiplicity: coordinate 0 twice and the others
    // once is refused; every coordinate as two halves is the same walk.
    let mut doubled = dephasing.clone();
    doubled.push(Dephasing::coordinate(n, 0, 0.2));
    assert!(SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &doubled, &symmetries).is_err());
    let mut halves = vec![Dephasing::sites(n, 0.3)];
    halves.extend((0..2 * d).map(|k| Dephasing::coordinate(n, k % d, 0.1)));
    let split =
        SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &halves, &symmetries).unwrap();
    let x = walk.evolve(&walk.localized(0).unwrap(), 0.7).unwrap();
    let y = split.evolve(&split.localized(0).unwrap(), 0.7).unwrap();
    assert!(x.iter().zip(&y).all(|(a, b)| (a - b).norm() < 1e-14));
    // Dense operators stop at 8192 points.
    let o = walk.orbitals();
    assert!(o.operator(&walk.uniform()).is_err());
}

#[test]
fn hamming_graphs_factorise_over_their_coordinates() {
    // H(4, 10): 10 000 nodes. From 0…0 with the same dephasing on every
    // coordinate, the walk is the product of walks on K₁₀ with site
    // dephasing, one per coordinate. The factor's reference is the walk on
    // K₁₀'s orbitals by union–find on its 100 pairs (the RK4 steps of
    // DephasingWalk keep only about 10⁻¹¹).
    let (d, q): (usize, usize) = (4, 10);
    let n = q.pow(d as u32);
    let g = generators::hamming(d, q).unwrap();
    let rate = 0.4;
    let dephasing: Vec<Dephasing> = (0..d)
        .map(|k| Dephasing {
            rate,
            block: (0..n).map(|x| x / q.pow(k as u32) % q).collect(),
        })
        .collect();
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        1.0,
        &dephasing,
        &hamming_stabiliser(d, q),
    )
    .unwrap();
    assert!(matches!(walk.orbitals().labels, Labels::Suborbits(_)));
    let kq = generators::complete(q);
    let factor = SymmetricOpenWalk::new(
        &kq,
        Generator::Laplacian,
        1.0,
        &[Dephasing::sites(q, rate)],
        &stabilizer_of_zero(&kq),
    )
    .unwrap();
    for t in [0.2, 0.9] {
        let p = walk
            .populations(&walk.evolve(&walk.localized(0).unwrap(), t).unwrap())
            .unwrap();
        let f = factor
            .populations(&factor.evolve(&factor.localized(0).unwrap(), t).unwrap())
            .unwrap();
        let mut worst = 0.0f64;
        for (x, &px) in p.iter().enumerate() {
            let expected: f64 = (0..d).map(|k| f[x / q.pow(k as u32) % q]).product();
            worst = worst.max((px - expected).abs());
        }
        assert!(worst < 1e-13, "t = {t}: {worst:e}");
    }
}

/// The point permutation `sigma` of `0..m` acting on the `k`-subsets of a
/// Johnson graph, whose nodes are the subsets in colex order (increasing
/// bit masks).
fn on_subsets(m: usize, k: usize, sigma: &[usize]) -> Vec<usize> {
    let sets: Vec<u64> = (0u64..1 << m)
        .filter(|s| s.count_ones() as usize == k)
        .collect();
    sets.iter()
        .map(|&s| {
            let image = (0..m)
                .filter(|&i| s >> i & 1 == 1)
                .fold(0u64, |acc, i| acc | 1 << sigma[i]);
            sets.binary_search(&image).unwrap()
        })
        .collect()
}

/// Generators of the stabiliser of node 0 (`{0, …, k−1}`) of `J(m, k)`:
/// the permutations of the subset and of the rest.
fn johnson_stabiliser(m: usize, k: usize) -> Vec<Vec<usize>> {
    let cycle = |lo: usize, hi: usize| -> Vec<usize> {
        (0..m)
            .map(|i| {
                if i >= lo && i < hi {
                    lo + (i - lo + 1) % (hi - lo)
                } else {
                    i
                }
            })
            .collect()
    };
    let swap = |a: usize, b: usize| -> Vec<usize> {
        (0..m)
            .map(|i| {
                if i == a {
                    b
                } else if i == b {
                    a
                } else {
                    i
                }
            })
            .collect()
    };
    [swap(0, 1), cycle(0, k), swap(k, k + 1), cycle(k, m)]
        .iter()
        .map(|sigma| on_subsets(m, k, sigma))
        .collect()
}

/// Time and size of open walks on large symmetric graphs, run with
/// `cargo test --release -- --ignored measure_large_open_walks --nocapture`.
#[test]
#[ignore]
fn measure_large_open_walks() {
    type Case = (String, Graph, Vec<Dephasing>, Vec<Vec<usize>>);
    let mut cases: Vec<Case> = Vec::new();
    for d in [14, 16, 18] {
        let n = 1 << d;
        let mut deph = vec![Dephasing::sites(n, 0.3)];
        deph.extend((0..d).map(|k| Dephasing::coordinate(n, k, 0.2)));
        cases.push((
            format!("Q{d}"),
            generators::hypercube(d),
            deph,
            coordinate_permutations(d).unwrap(),
        ));
    }
    for (d, q) in [(4usize, 10usize), (7, 5)] {
        let n = q.pow(d as u32);
        let deph = (0..d)
            .map(|k| Dephasing {
                rate: 0.4,
                block: (0..n).map(|x| x / q.pow(k as u32) % q).collect(),
            })
            .collect();
        cases.push((
            format!("H({d}, {q})"),
            generators::hamming(d, q).unwrap(),
            deph,
            hamming_stabiliser(d, q),
        ));
    }
    for (m, k) in [(24usize, 4usize), (30, 4)] {
        let g = generators::johnson(m, k).unwrap();
        let n = g.num_nodes();
        cases.push((
            format!("J({m}, {k})"),
            g,
            vec![Dephasing::sites(n, 0.3)],
            johnson_stabiliser(m, k),
        ));
    }
    for (name, g, deph, sym) in cases {
        let clock = std::time::Instant::now();
        let walk = SymmetricOpenWalk::new(&g, Generator::Laplacian, 1.0, &deph, &sym).unwrap();
        let build = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let x = walk.evolve(&walk.localized(0).unwrap(), 1.0).unwrap();
        let evolve = clock.elapsed().as_secs_f64();
        let p = walk.populations(&x).unwrap();
        // Hypercubes: C(d + 3, 3) orbitals, and the distance distribution
        // of HypercubeOpenWalk.
        if let Some(d) = name.strip_prefix('Q').and_then(|d| d.parse::<usize>().ok()) {
            assert_eq!(walk.num_orbitals() as f64, binomial(d + 3, 3));
            let reference = HypercubeOpenWalk::new(d, 1.0, 0.3, 0.2).unwrap();
            let expected = reference
                .distance_distribution(&reference.evolve(&reference.localized(), 1.0).unwrap())
                .unwrap();
            let mut by_distance = vec![0.0; d + 1];
            for (u, x) in p.iter().enumerate() {
                by_distance[u.count_ones() as usize] += x;
            }
            let worst = max_diff(&by_distance, &expected);
            eprintln!("{name}: distance distribution to {worst:.1e} of HypercubeOpenWalk");
            assert!(worst < 1e-12);
        }
        eprintln!(
            "{name}: {} nodes, {} orbitals, built in {build:.2} s, evolved to t = 1 in \
             {evolve:.2} s, total population {:.15}",
            g.num_nodes(),
            walk.num_orbitals(),
            p.iter().sum::<f64>()
        );
    }
}

#[test]
fn johnson_graphs_beyond_the_pairs() {
    // J(16, 4) has 1820 nodes: the stabiliser's orbitals by both methods,
    // and its walk on them equal to the one on the pairs' orbitals.
    let g = generators::johnson(16, 4).unwrap();
    let n = g.num_nodes();
    let sym = johnson_stabiliser(16, 4);
    assert_same_orbitals(n, &sym, "J(16, 4)");
    // J(24, 4), 10 626 nodes: the walk conserves the population, and the
    // population of each node depends only on its distance from the start.
    let g = generators::johnson(24, 4).unwrap();
    let n = g.num_nodes();
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        1.0,
        &[Dephasing::sites(n, 0.3)],
        &johnson_stabiliser(24, 4),
    )
    .unwrap();
    assert!(matches!(walk.orbitals().labels, Labels::Suborbits(_)));
    // Pairs of 4-sets up to the stabiliser of {0, 1, 2, 3}: the intersections
    // of each with it and with each other.
    let p = walk
        .populations(&walk.evolve(&walk.localized(0).unwrap(), 0.8).unwrap())
        .unwrap();
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let sets: Vec<u64> = (0u64..1 << 24).filter(|s| s.count_ones() == 4).collect();
    let mut by_overlap = [f64::NAN; 5];
    for (x, &px) in p.iter().enumerate() {
        let overlap = (sets[x] & 0b1111).count_ones() as usize;
        if by_overlap[overlap].is_nan() {
            by_overlap[overlap] = px;
        }
        assert!((px - by_overlap[overlap]).abs() < 1e-15, "{x}");
    }
    // Without dephasing the walk stays on the distance partition: the
    // quotient walk of J(24, 4) gives the population of each cell.
    let coherent = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        1.0,
        &[],
        &johnson_stabiliser(24, 4),
    )
    .unwrap();
    let quotient = QuotientWalk::new(
        &Quotient::johnson(24, 4).unwrap(),
        Generator::Laplacian,
        1.0,
    )
    .unwrap();
    for t in [0.4, 1.3] {
        let p = coherent
            .populations(&coherent.evolve(&coherent.localized(0).unwrap(), t).unwrap())
            .unwrap();
        let mut cells = [0.0; 5];
        for (x, &px) in p.iter().enumerate() {
            cells[4 - (sets[x] & 0b1111).count_ones() as usize] += px;
        }
        let expected = quotient.cell_probabilities(0, t).unwrap();
        assert!(max_diff(&cells, &expected) < 1e-12, "t = {t}");
    }
}

#[test]
fn many_generators_and_identities_reach_every_orbital() {
    // Product replacement must start from every generator: ten identities
    // before the coordinate permutations of Q₅, and the eleven adjacent
    // transpositions of the coordinates of Q₁₂ (more generators than
    // slots), by Schreier generators and by random elements alone.
    let mut gens: Vec<Vec<usize>> = vec![(0..32).collect(); 10];
    gens.extend(coordinate_permutations(5).unwrap());
    assert_same_orbitals(32, &gens, "Q5 after ten identities");
    let d = 12;
    let swap = |k: usize| -> Vec<usize> {
        (0..1usize << d)
            .map(|u| {
                let (a, b) = ((u >> k) & 1, (u >> (k + 1)) & 1);
                u & !(0b11 << k) | (a << (k + 1)) | (b << k)
            })
            .collect()
    };
    let transpositions: Vec<Vec<usize>> = (0..d - 1).map(swap).collect();
    let pairs = Orbitals::new(1 << d, &transpositions).unwrap();
    assert_eq!(pairs.len(), 455);
    let (s, _, _) = Suborbits::with_exhaustive_work(1 << d, &transpositions, 0, 0).unwrap();
    for u in (0..1 << d).step_by(7) {
        for v in 0..1 << d {
            assert_eq!(s.of(u, v), pairs.of(u, v), "({u}, {v})");
        }
    }
    // Huge point sets without generators are refused before they are built.
    #[cfg(target_pointer_width = "64")]
    assert!(Orbitals::new(1 << 40, &[]).is_err());
    assert!(Orbitals::new(1 << 20, &[]).is_err());
}
