use super::*;
use crate::graph::generators;
use crate::walks::interacting::{HypercubeParticles, InteractingWalk};

const ALL: [Statistics; 3] = [
    Statistics::Bosons,
    Statistics::Fermions,
    Statistics::Distinguishable,
];

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

/// The reduced walk from `x0`, lifted, against the full walk from the
/// lifted `x0`.
fn agrees_with_the_full_walk(walk: &SymmetricInteractingWalk, full: &InteractingWalk, x0: &[C64]) {
    for t in [0.3, 1.7] {
        let reduced = walk.lift(&walk.evolve(x0, t).unwrap()).unwrap();
        let unreduced = full.evolve(&walk.lift(x0).unwrap(), t).unwrap();
        let err = reduced
            .iter()
            .zip(&unreduced)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "{:?}: {err}", walk.statistics());
        let x = walk.evolve(x0, t).unwrap();
        assert!(close(
            &walk.densities(&x).unwrap(),
            &full.densities(&unreduced).unwrap(),
            1e-10
        ));
        let pairs = walk.pair_correlations(&x).unwrap();
        let full_pairs = full.pair_correlations(&unreduced).unwrap();
        assert!((pairs - full_pairs).abs().max() < 1e-10);
    }
}

#[test]
fn reduced_walks_equal_the_full_walk() {
    let graphs = [
        ("petersen", generators::petersen()),
        ("rook 4×4", generators::rook(4, 4).unwrap()),
        ("Q4", generators::hypercube(4)),
        ("C8", generators::cycle(8)),
    ];
    for (name, g) in &graphs {
        for statistics in ALL {
            for particles in [2, 3] {
                for generator in [Generator::Adjacency, Generator::Laplacian] {
                    let start: Vec<usize> = match statistics {
                        Statistics::Fermions => (0..particles).collect(),
                        _ => vec![0; particles],
                    };
                    let walk =
                        SymmetricInteractingWalk::from_start(g, generator, 0.8, statistics, &start)
                            .unwrap()
                            .with_interaction(0.9, -0.4)
                            .unwrap();
                    let full = InteractingWalk::new(g, generator, 0.8, statistics, particles)
                        .unwrap()
                        .with_interaction(0.9, -0.4)
                        .unwrap();
                    assert_eq!(walk.num_configurations(), full.dimension(), "{name}");
                    // The start is an orbit of its own and the lift is the
                    // basis state.
                    let x0 = walk.state_of(&start).unwrap();
                    let lifted = walk.lift(&x0).unwrap();
                    let basis = full.basis_state(&start).unwrap();
                    assert!(
                        lifted
                            .iter()
                            .zip(&basis)
                            .all(|(a, b)| (a - b).norm() < 1e-15)
                    );
                    agrees_with_the_full_walk(&walk, &full, &x0);
                }
            }
        }
    }
}

#[test]
fn whole_automorphism_groups_and_fermion_signs() {
    // Under the whole group of the graph (or the rotations of C₇), every
    // orbit state that survives evolves as in the full walk; fermionic
    // configurations that a symmetry maps to minus themselves carry no
    // state.
    let c7 = generators::cycle(7);
    let rotation: Vec<usize> = (0..7).map(|u| (u + 1) % 7).collect();
    // (graph, its rotations only instead of the whole group)
    for (g, rotations) in [
        (generators::petersen(), None),
        (c7.clone(), None),
        (generators::hypercube(3), None),
        (c7, Some(vec![rotation])),
    ] {
        let whole = rotations.is_none();
        let group = g.automorphism_group();
        let generators = rotations.unwrap_or_else(|| group.generators().to_vec());
        for statistics in ALL {
            let walk = SymmetricInteractingWalk::new(
                &g,
                Generator::Adjacency,
                1.1,
                statistics,
                2,
                &generators,
            );
            if statistics == Statistics::Fermions && whole {
                // These graphs are distance-transitive: a symmetry swaps the
                // ends of every pair, so no state of two fermions is
                // invariant under the whole group.
                assert!(walk.is_err());
                let orbits =
                    ConfigurationOrbits::new(g.num_nodes(), statistics, 2, &generators).unwrap();
                assert!(orbits.is_empty());
                continue;
            }
            let walk = walk.unwrap().with_interaction(0.7, 0.3).unwrap();
            let full = InteractingWalk::new(&g, Generator::Adjacency, 1.1, statistics, 2)
                .unwrap()
                .with_interaction(0.7, 0.3)
                .unwrap();
            let orbits = walk.orbits();
            let live: usize = (0..orbits.len()).map(|o| orbits.size(o)).sum();
            assert!(live <= full.dimension());
            for o in 0..walk.num_orbits() {
                let mut x0 = vec![C64::new(0.0, 0.0); walk.num_orbits()];
                x0[o] = C64::new(1.0, 0.0);
                agrees_with_the_full_walk(&walk, &full, &x0);
            }
            if statistics == Statistics::Fermions {
                // The rotations of C₇ fix no pair: every orbit lives, with
                // the signs of the pairs that wrap around.
                assert_eq!((walk.num_orbits(), live), (3, full.dimension()));
            }
        }
    }
    // The reflection of C₄ through nodes 0 and 2 swaps 1 and 3: two
    // fermions on {1, 3} are mapped to minus themselves.
    let c4 = generators::cycle(4);
    let reflection = vec![0, 3, 2, 1];
    let walk = SymmetricInteractingWalk::new(
        &c4,
        Generator::Adjacency,
        1.0,
        Statistics::Fermions,
        2,
        &[reflection],
    )
    .unwrap();
    assert!(walk.orbits().locate(&[1, 3]).unwrap().is_none());
    assert!(walk.state_of(&[1, 3]).is_err());
    assert_eq!(
        walk.probability_of(&vec![C64::new(1.0, 0.0); walk.num_orbits()], &[1, 3])
            .unwrap(),
        0.0
    );
    let (o, sign) = walk.orbits().locate(&[0, 1]).unwrap().unwrap();
    let (o2, sign2) = walk.orbits().locate(&[0, 3]).unwrap().unwrap();
    assert_eq!(o, o2);
    // (0, 1) ↦ (0, 3) keeps the order: the same sign.
    assert_eq!(sign, sign2);
}

#[test]
fn random_graphs_with_automorphisms() {
    // Random weighted graphs closed under a random permutation σ of cycle
    // type (3)(3)(2)(1) on nine nodes, reduced by ⟨σ⟩ and by their whole
    // automorphism group, against the full walk.
    let mut rng = crate::rng::Rng::seed_from_u64(43);
    let n = 9;
    for _ in 0..3 {
        let mut nodes: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut nodes);
        let mut sigma = vec![0; n];
        for cycle in [&nodes[0..3], &nodes[3..6], &nodes[6..8], &nodes[8..9]] {
            for (k, &u) in cycle.iter().enumerate() {
                sigma[u] = cycle[(k + 1) % cycle.len()];
            }
        }
        let mut edges: std::collections::BTreeMap<(usize, usize), f64> = Default::default();
        for u in 0..n {
            for v in u + 1..n {
                if edges.contains_key(&(u, v)) || rng.uniform() > 0.4 {
                    continue;
                }
                // The edge's orbit under σ (whose order divides 6), with
                // one weight.
                let w = rng.uniform_range(0.3, 1.5);
                let (mut a, mut b) = (u, v);
                for _ in 0..6 {
                    edges.insert((a.min(b), a.max(b)), w);
                    (a, b) = (sigma[a], sigma[b]);
                }
            }
        }
        let list: Vec<(usize, usize, f64)> = edges.iter().map(|(&(u, v), &w)| (u, v, w)).collect();
        let g = Graph::from_weighted_edges(n, &list).unwrap();
        assert!(is_automorphism(&g, &sigma));
        let whole = g.automorphism_group();
        for statistics in ALL {
            for particles in [2, 3] {
                let full =
                    InteractingWalk::new(&g, Generator::Laplacian, 0.9, statistics, particles)
                        .unwrap()
                        .with_interaction(1.1, -0.5)
                        .unwrap();
                for group in [vec![sigma.clone()], whole.generators().to_vec()] {
                    let Ok(walk) = SymmetricInteractingWalk::new(
                        &g,
                        Generator::Laplacian,
                        0.9,
                        statistics,
                        particles,
                        &group,
                    ) else {
                        // Only fermions can be left without a state.
                        assert_eq!(statistics, Statistics::Fermions);
                        continue;
                    };
                    let walk = walk.with_interaction(1.1, -0.5).unwrap();
                    for o in 0..walk.num_orbits().min(5) {
                        let mut x0 = vec![C64::new(0.0, 0.0); walk.num_orbits()];
                        x0[o] = C64::new(1.0, 0.0);
                        agrees_with_the_full_walk(&walk, &full, &x0);
                    }
                }
            }
        }
    }
}

#[test]
fn hypercube_particles_on_q8_equal_the_full_walk() {
    // Two particles on Q₈ (up to 65 536 configurations) with U and V:
    // weights, pair distances and the antipode from the full walk of 1.29.
    let d = 8;
    let g = generators::hypercube(d);
    for statistics in ALL {
        let full = InteractingWalk::new(&g, Generator::Adjacency, 0.7, statistics, 2)
            .unwrap()
            .with_interaction(1.3, -0.6)
            .unwrap();
        let cube = HypercubeParticles::new(d, statistics, 2, Generator::Adjacency, 0.7)
            .unwrap()
            .with_interaction(1.3, -0.6)
            .unwrap();
        // Both at node 0, or fermions in the invariant state of 0 and e_k.
        let (psi0, x0) = if statistics == Statistics::Fermions {
            let mut psi = vec![C64::new(0.0, 0.0); full.dimension()];
            for k in 0..d {
                for (a, b) in psi.iter_mut().zip(full.basis_state(&[0, 1 << k]).unwrap()) {
                    *a += b / (d as f64).sqrt();
                }
            }
            (psi, cube.state_of(&[d - 1, 0, 1, 0]).unwrap())
        } else {
            (
                full.basis_state(&[0, 0]).unwrap(),
                cube.localized().unwrap(),
            )
        };
        let t = 0.9;
        let psi = full.evolve(&psi0, t).unwrap();
        let x = cube.evolve(&x0, t).unwrap();
        let mut weights = vec![0.0; d + 1];
        let mut distances = vec![0.0; d + 1];
        let mut antipode = 0.0;
        for (k, a) in psi.iter().enumerate() {
            let p = a.norm_sqr();
            let c = full.configuration(k).unwrap();
            weights[c[0].count_ones() as usize] += p;
            weights[c[1].count_ones() as usize] += p;
            distances[(c[0] ^ c[1]).count_ones() as usize] += p;
            if c[0] == (1 << d) - 1 && c[1] == (1 << d) - 1 {
                antipode += p;
            }
        }
        assert!(
            close(&cube.weights(&x).unwrap(), &weights, 1e-10),
            "{statistics:?}"
        );
        assert!(close(&cube.pair_distances(&x).unwrap(), &distances, 1e-10));
        assert!((cube.antipode_probability(&x).unwrap() - antipode).abs() < 1e-12);
    }
}

#[test]
fn hypercube_particles_equal_the_generic_reduction() {
    for d in [3, 4] {
        let g = generators::hypercube(d);
        for (statistics, particles) in [
            (Statistics::Bosons, 2),
            (Statistics::Bosons, 3),
            (Statistics::Distinguishable, 2),
            (Statistics::Distinguishable, 3),
            (Statistics::Fermions, 2),
            (Statistics::Fermions, 3),
            (Statistics::Fermions, 4),
        ] {
            for generator in [Generator::Adjacency, Generator::Laplacian] {
                let cube = HypercubeParticles::new(d, statistics, particles, generator, 0.7)
                    .unwrap()
                    .with_interaction(1.3, -0.6)
                    .unwrap();
                // The coordinate permutations: the stabiliser of node 0.
                let stabiliser = g.stabilizer(&[0]).unwrap();
                let walk = SymmetricInteractingWalk::new(
                    &g,
                    generator,
                    0.7,
                    statistics,
                    particles,
                    stabiliser.generators(),
                )
                .unwrap()
                .with_interaction(1.3, -0.6)
                .unwrap();
                // The same states: fermions start at nodes that no odd
                // permutation of the particles maps to a coordinate
                // permutation of themselves (in increasing order, where
                // both walks give them a positive amplitude).
                let (x0, c0) = if statistics == Statistics::Fermions {
                    let sites = &[0, 1, 6, 7][..particles];
                    let mut counts = vec![0; 1 << particles];
                    for k in 0..d {
                        counts[(0..particles).fold(0, |p, j| p | (sites[j] >> k & 1) << j)] += 1;
                    }
                    (
                        walk.state_of(sites).unwrap(),
                        cube.state_of(&counts).unwrap(),
                    )
                } else {
                    (
                        walk.state_of(&vec![0; particles]).unwrap(),
                        cube.localized().unwrap(),
                    )
                };
                assert_eq!(
                    cube.num_states(),
                    walk.num_orbits(),
                    "Q{d} {statistics:?} {particles}"
                );
                for t in [0.4, 1.9] {
                    let x = walk.evolve(&x0, t).unwrap();
                    let y = cube.evolve(&c0, t).unwrap();
                    // Weights and pair distances from the lifted state.
                    let lifted = walk.lift(&x).unwrap();
                    let full =
                        InteractingWalk::new(&g, generator, 0.7, statistics, particles).unwrap();
                    let mut weights = vec![0.0; d + 1];
                    let mut distances = vec![0.0; d + 1];
                    let pairs = (particles * (particles - 1) / 2) as f64;
                    for (k, a) in lifted.iter().enumerate() {
                        let p = a.norm_sqr();
                        let c = full.configuration(k).unwrap();
                        for &s in c {
                            weights[s.count_ones() as usize] += p;
                        }
                        for i in 0..particles {
                            for j in i + 1..particles {
                                distances[(c[i] ^ c[j]).count_ones() as usize] += p / pairs;
                            }
                        }
                    }
                    assert!(
                        close(&cube.weights(&y).unwrap(), &weights, 1e-10),
                        "Q{d} {statistics:?} {particles} {t}"
                    );
                    assert!(close(&cube.pair_distances(&y).unwrap(), &distances, 1e-10));
                    let antipode = vec![(1 << d) - 1; particles];
                    let expected = if statistics == Statistics::Fermions {
                        0.0
                    } else {
                        walk.probability_of(&x, &antipode).unwrap()
                    };
                    assert!((cube.antipode_probability(&y).unwrap() - expected).abs() < 1e-10);
                }
            }
        }
    }
}

#[test]
fn free_particles_on_hypercubes_follow_the_single_walker() {
    // Without interaction every particle walks on its own: a coordinate
    // flips with probability q = sin²(γt), so the weights are N·Bin(d, q)
    // and two particles differ in a coordinate with probability 2q(1 − q).
    let binomial = |n: usize, p: f64| -> Vec<f64> {
        (0..=n)
            .map(|k| {
                let c = (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
                c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
            })
            .collect()
    };
    for (d, particles) in [(30, 2), (40, 2), (12, 3), (6, 4)] {
        for statistics in [Statistics::Bosons, Statistics::Distinguishable] {
            let cube = HypercubeParticles::new(d, statistics, particles, Generator::Adjacency, 0.9)
                .unwrap();
            // Up to perfect transfer at γt = π/2, where every particle sits
            // at the antipode: the equally spaced spectrum once misled the
            // step control of the Krylov propagation (6391 states for Q₄₀).
            for t in [0.2, 1.1, 1.5, std::f64::consts::FRAC_PI_2 / 0.9] {
                let x = cube.evolve(&cube.localized().unwrap(), t).unwrap();
                let q = (0.9 * t).sin().powi(2);
                let weights: Vec<f64> = binomial(d, q)
                    .iter()
                    .map(|w| w * particles as f64)
                    .collect();
                assert!(
                    close(&cube.weights(&x).unwrap(), &weights, 1e-10),
                    "{d} {particles}"
                );
                assert!(close(
                    &cube.pair_distances(&x).unwrap(),
                    &binomial(d, 2.0 * q * (1.0 - q)),
                    1e-10
                ));
                let antipode = q.powi((d * particles) as i32);
                assert!((cube.antipode_probability(&x).unwrap() - antipode).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn free_fermions_on_hypercubes_are_a_determinant() {
    // Two free fermions from the invariant state of nodes 0 and e_k: the
    // Slater determinant of |0⟩ and |φ⟩ = d^{−1/2} Σ_k |e_k⟩, whose density
    // is |⟨x|U|0⟩|² + |⟨x|U|φ⟩|² with U = ⊗(cos γt + i sin γt X). At weight
    // w, ⟨x|U|φ⟩ = d^{−1/2}(is)^{w−1}c^{d−w−1}(wc² − (d − w)s²).
    for d in [5, 30] {
        for generator in [Generator::Adjacency, Generator::Laplacian] {
            let cube = HypercubeParticles::new(d, Statistics::Fermions, 2, generator, 0.9).unwrap();
            let mut counts = vec![0; 4];
            counts[0] = d - 1;
            counts[2] = 1;
            let x0 = cube.state_of(&counts).unwrap();
            for t in [0.3, 1.1, std::f64::consts::FRAC_PI_2 / 0.9] {
                let (c, s) = ((0.9 * t).cos(), (0.9 * t).sin());
                let expected: Vec<f64> = (0..=d)
                    .map(|w| {
                        let binom =
                            (0..w).fold(1.0, |acc, i| acc * (d - i) as f64 / (i + 1) as f64);
                        let first = if w >= 1 {
                            w as f64 * c.powi((d - w + 1) as i32) * s.powi(w as i32 - 1)
                        } else {
                            0.0
                        };
                        let second = if w < d {
                            (d - w) as f64 * s.powi(w as i32 + 1) * c.powi((d - w - 1) as i32)
                        } else {
                            0.0
                        };
                        binom
                            * (c.powi(2 * (d - w) as i32) * s.powi(2 * w as i32)
                                + (first - second).powi(2) / d as f64)
                    })
                    .collect();
                let x = cube.evolve(&x0, t).unwrap();
                assert!(
                    close(&cube.weights(&x).unwrap(), &expected, 1e-10),
                    "Q{d} {generator:?} {t}"
                );
            }
        }
    }
}

#[test]
fn small_hypercubes_and_documented_sizes() {
    // Three fermions on Q₂ keep one state ({0, 1, 3} and {0, 2, 3}), four
    // none; three do not fit on Q₁.
    let cube =
        HypercubeParticles::new(2, Statistics::Fermions, 3, Generator::Adjacency, 1.0).unwrap();
    assert_eq!(cube.num_states(), 1);
    let square = generators::hypercube(2);
    let walk = SymmetricInteractingWalk::new(
        &square,
        Generator::Adjacency,
        1.0,
        Statistics::Fermions,
        3,
        square.stabilizer(&[0]).unwrap().generators(),
    )
    .unwrap();
    assert_eq!(walk.num_orbits(), 1);
    assert!(
        HypercubeParticles::new(2, Statistics::Fermions, 4, Generator::Adjacency, 1.0).is_err()
    );
    assert!(
        HypercubeParticles::new(1, Statistics::Fermions, 3, Generator::Adjacency, 1.0).is_err()
    );
    // The sizes quoted in the documentation (the generic reduction of Q₈
    // and Q₉ has as many orbits; see hypercube_particles_equal_the_generic_reduction).
    for (d, particles, states) in [
        (8, 3, 1324),
        (9, 3, 2284),
        (20, 3, 156_240),
        (100, 2, 89_726),
    ] {
        let cube =
            HypercubeParticles::new(d, Statistics::Bosons, particles, Generator::Adjacency, 1.0)
                .unwrap();
        assert_eq!(cube.num_states(), states, "Q{d}, {particles} bosons");
    }
}

#[test]
fn empty_walks_and_large_inputs_are_refused() {
    // Walks without an invariant state: four fermions on K₄ and three on
    // the star with five leaves, under all their automorphisms.
    for (g, particles) in [(generators::complete(4), 4), (generators::star(6), 3)] {
        let group = g.automorphism_group();
        let walk = SymmetricInteractingWalk::new(
            &g,
            Generator::Adjacency,
            1.0,
            Statistics::Fermions,
            particles,
            group.generators(),
        );
        assert!(walk.is_err(), "{particles} fermions");
    }
    // Fermions cannot start on one site.
    let ring = generators::cycle(6);
    assert!(
        SymmetricInteractingWalk::from_start(
            &ring,
            Generator::Adjacency,
            1.0,
            Statistics::Fermions,
            &[0, 0],
        )
        .is_err()
    );
    // 2²² orbits of 23 distinguishable particles on two nodes: their
    // representatives would take 772 MB, refused before they are stored.
    let pair = generators::path(2);
    assert!(
        SymmetricInteractingWalk::new(
            &pair,
            Generator::Adjacency,
            1.0,
            Statistics::Distinguishable,
            23,
            &[vec![1, 0]],
        )
        .is_err()
    );
    // Hamiltonians beyond 2²⁶ entries, refused before the compositions
    // are enumerated.
    for (d, particles) in [(12, 4), (36, 3)] {
        assert!(
            HypercubeParticles::new(
                d,
                Statistics::Distinguishable,
                particles,
                Generator::Adjacency,
                1.0
            )
            .is_err()
        );
    }
    // Counts that do not add up to d, however large.
    let cube =
        HypercubeParticles::new(4, Statistics::Bosons, 1, Generator::Adjacency, 1.0).unwrap();
    assert!(cube.state_of(&[usize::MAX, 5]).is_err());
    assert!(cube.state_of(&[usize::MAX - 65_532, 65_536]).is_err());
    assert!(cube.state_of(&[3, 1]).is_ok());
}

#[test]
fn bound_pairs_on_a_ring() {
    // Two bosons on a ring of L sites with hopping J and on-site U: the
    // pair of total momentum zero is bound at E = ±√(U² + 16J²) (Winkler
    // et al. 2006); the ring's dihedral symmetry keeps that sector.
    let l = 101;
    let g = generators::cycle(l);
    let group = g.automorphism_group();
    for u in [3.0, -2.0, 7.5] {
        let walk = SymmetricInteractingWalk::new(
            &g,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            2,
            group.generators(),
        )
        .unwrap()
        .with_interaction(u, 0.0)
        .unwrap();
        assert_eq!(walk.num_orbits(), 51);
        let bound = u.signum() * (u * u + 16.0).sqrt();
        let spectrum = walk.spectrum().unwrap();
        let nearest = spectrum
            .iter()
            .map(|e| (e - bound).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 1e-12, "U = {u}: {nearest}");
        // The rest is the scattering band of two free particles, [−4J, 4J].
        assert!(spectrum.iter().filter(|e| e.abs() > 4.0 + 1e-9).count() == 1);
    }
}

#[test]
fn invalid_input() {
    let g = generators::petersen();
    let not_automorphism: Vec<usize> = vec![1, 0, 2, 3, 4, 5, 6, 7, 8, 9];
    assert!(
        SymmetricInteractingWalk::new(
            &g,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            2,
            &[not_automorphism]
        )
        .is_err()
    );
    assert!(
        SymmetricInteractingWalk::new(
            &g,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            2,
            &[vec![0; 10]]
        )
        .is_err()
    );
    assert!(
        SymmetricInteractingWalk::new(&g, Generator::Adjacency, 0.0, Statistics::Bosons, 2, &[])
            .is_err()
    );
    assert!(
        SymmetricInteractingWalk::new(&g, Generator::Adjacency, 1.0, Statistics::Fermions, 11, &[])
            .is_err()
    );
    let walk = SymmetricInteractingWalk::from_start(
        &g,
        Generator::Adjacency,
        1.0,
        Statistics::Bosons,
        &[0, 0],
    )
    .unwrap();
    assert!(walk.state_of(&[0, 10]).is_err());
    assert!(walk.state_of(&[0]).is_err());
    assert!(walk.evolve(&[C64::new(1.0, 0.0)], 1.0).is_err());
    assert!(walk.clone().with_interaction(f64::NAN, 0.0).is_err());
    // Too many configurations for their orbits.
    assert!(ConfigurationOrbits::new(1 << 20, Statistics::Bosons, 2, &[]).is_err());
    // Hypercubes.
    assert!(HypercubeParticles::new(0, Statistics::Bosons, 2, Generator::Adjacency, 1.0).is_err());
    assert!(HypercubeParticles::new(4, Statistics::Bosons, 5, Generator::Adjacency, 1.0).is_err());
    assert!(HypercubeParticles::new(40, Statistics::Bosons, 4, Generator::Adjacency, 1.0).is_err());
    let fermions =
        HypercubeParticles::new(5, Statistics::Fermions, 2, Generator::Adjacency, 1.0).unwrap();
    assert!(fermions.localized().is_err());
    assert!(fermions.state_of(&[5, 0, 0, 1]).is_err());
}
