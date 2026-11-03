//! Walkers on coordinate graphs against the walks of 1.29 and 1.43 and
//! closed forms.

use super::*;
use crate::walks::{InteractingWalk, SymmetricInteractingWalk};

/// Every configuration's probability after `t` against the full walk of
/// 1.29, and the orbit count against 1.43.
fn compare(
    graph: CoordinateGraph,
    generator: Generator,
    statistics: Statistics,
    start: &[u64],
    u: f64,
    v: f64,
) {
    let g = graph.graph().unwrap();
    let starts: Vec<usize> = start.iter().map(|&s| s as usize).collect();
    let ours = CoordinateInteractingWalk::from_start(graph, generator, 0.7, statistics, start)
        .unwrap()
        .with_interaction(u, v)
        .unwrap();
    let theirs =
        SymmetricInteractingWalk::from_start(&g, generator, 0.7, statistics, &starts).unwrap();
    // On J(2k, k) 1.43 also uses the complement, so its orbits are
    // coarser; on K(2k, k), a perfect matching, it counts the orbits of
    // every configuration, ours only those the start reaches.
    match graph {
        CoordinateGraph::Kneser { n, k } if n == 2 * k => {}
        CoordinateGraph::Johnson { n, k } if n == 2 * k => assert!(
            ours.num_orbits() >= theirs.num_orbits(),
            "{graph:?} {statistics:?} {start:?}"
        ),
        _ => assert_eq!(
            ours.num_orbits(),
            theirs.num_orbits(),
            "{graph:?} {statistics:?} {start:?}"
        ),
    }
    let full = InteractingWalk::new(&g, generator, 0.7, statistics, start.len())
        .unwrap()
        .with_interaction(u, v)
        .unwrap();
    let t = 1.3;
    let psi = full.evolve(&full.basis_state(&starts).unwrap(), t).unwrap();
    let x = ours.evolve(&ours.state_of(start).unwrap(), t).unwrap();
    let mut worst = 0.0f64;
    for (i, amplitude) in psi.iter().enumerate() {
        let sites: Vec<u64> = full
            .configuration(i)
            .unwrap()
            .iter()
            .map(|&s| s as u64)
            .collect();
        let p = ours.probability_of(&x, &sites).unwrap();
        worst = worst.max((p - amplitude.norm_sqr()).abs());
    }
    assert!(
        worst < 1e-12,
        "{graph:?} {generator:?} {statistics:?} {start:?}: {worst:e}"
    );
    // Densities per orbit of nodes add up to the particles.
    let total: f64 = ours
        .node_densities(&x)
        .unwrap()
        .iter()
        .map(|&(_, size, d)| size * d)
        .sum();
    assert!((total - start.len() as f64).abs() < 1e-10, "{total}");
}

#[test]
fn walkers_equal_the_full_walk_on_every_family() {
    use Statistics::*;
    let q4 = CoordinateGraph::Hamming { d: 4, q: 2 };
    let h33 = CoordinateGraph::Hamming { d: 3, q: 3 };
    let k5 = CoordinateGraph::Hamming { d: 1, q: 5 };
    let j62 = CoordinateGraph::Johnson { n: 6, k: 2 };
    let kn72 = CoordinateGraph::Kneser { n: 7, k: 2 };
    // The exceptions: the complement of J(4, 2) is not in the group,
    // K(4, 2) is a perfect matching, K_2 = H(1, 2) a single edge.
    let j42 = CoordinateGraph::Johnson { n: 4, k: 2 };
    let kn42 = CoordinateGraph::Kneser { n: 4, k: 2 };
    let k2 = CoordinateGraph::Hamming { d: 1, q: 2 };
    for (graph, start) in [
        (q4, vec![0, 0, 0]),
        (q4, vec![0, 3, 5]),
        (h33, vec![0, 4]),
        (h33, vec![2, 2, 13]),
        (k5, vec![1, 1, 3]),
        (j62, vec![0, 7, 7]),
        (kn72, vec![0, 3, 9]),
        (j42, vec![0, 0, 5]),
        (j42, vec![1, 2, 3]),
        (kn42, vec![0, 0, 1]),
        (kn42, vec![0, 2, 5]),
        (k2, vec![0, 0, 1]),
        (k2, vec![0, 1]),
    ] {
        for statistics in [Bosons, Fermions, Distinguishable] {
            let mut distinct = start.clone();
            distinct.sort_unstable();
            distinct.dedup();
            if statistics == Fermions && distinct.len() < start.len() {
                continue;
            }
            for generator in [Generator::Adjacency, Generator::Laplacian] {
                compare(graph, generator, statistics, &start, 0.9, -0.4);
            }
        }
    }
}

#[test]
fn a_perfect_matching_keeps_its_components() {
    // K(2k, k) pairs each subset with its complement: the start reaches
    // only configurations on its pairs, and the lower bound on the orbits
    // (from all configurations) must not refuse it.
    let graph = CoordinateGraph::Kneser { n: 40, k: 20 };
    let walk = CoordinateInteractingWalk::from_start(
        graph,
        Generator::Adjacency,
        1.0,
        Statistics::Bosons,
        &[0, 0],
    )
    .unwrap();
    assert_eq!(walk.num_orbits(), 3);
    let x = walk.evolve(&walk.state_of(&[0, 0]).unwrap(), 0.4).unwrap();
    let partner = graph.num_nodes() - 1;
    let total: f64 = [[0, 0], [0, partner], [partner, partner]]
        .iter()
        .map(|c| walk.probability_of(&x, c).unwrap())
        .sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
    // Off the start's pair: probability zero, and no state to start from.
    assert_eq!(walk.probability_of(&x, &[0, 1]).unwrap(), 0.0);
    assert!(walk.state_of(&[0, 1]).is_err());
    assert!(walk.locate(&[0, 1]).is_err());
}

/// `⟨y|e^{iγtA}|x⟩` on `H(d, q)`: per place `e^{−iγt}(δ + (e^{iγtq} − 1)/q)`.
fn amplitude(d: usize, q: usize, gamma: f64, t: f64, x: u64, y: u64) -> C64 {
    let phase = C64::from_polar(1.0, -gamma * t);
    let off = (C64::from_polar(1.0, gamma * t * q as f64) - 1.0) / q as f64;
    let (mut a, mut b) = (x, y);
    let mut out = C64::new(1.0, 0.0);
    for _ in 0..d {
        let same = a % q as u64 == b % q as u64;
        out *= phase * (off + if same { 1.0 } else { 0.0 });
        a /= q as u64;
        b /= q as u64;
    }
    out
}

/// Permanent or determinant of a 3 × 3 matrix.
fn combine(m: &[[C64; 3]; 3], fermions: bool) -> C64 {
    let perms = [
        ([0, 1, 2], 1.0),
        ([1, 2, 0], 1.0),
        ([2, 0, 1], 1.0),
        ([0, 2, 1], -1.0),
        ([2, 1, 0], -1.0),
        ([1, 0, 2], -1.0),
    ];
    perms
        .iter()
        .map(|(p, s)| {
            let sign = if fermions { *s } else { 1.0 };
            m[0][p[0]] * m[1][p[1]] * m[2][p[2]] * sign
        })
        .sum()
}

/// `n!`.
fn factorial(n: usize) -> f64 {
    (2..=n).map(|i| i as f64).product()
}

/// The product of the factorials of the multiplicities of `sites`.
fn multiplicities(sites: &[u64]) -> f64 {
    let mut sorted = sites.to_vec();
    sorted.sort_unstable();
    let mut out = 1.0;
    let mut run = 1;
    for k in 1..=sorted.len() {
        if k < sorted.len() && sorted[k] == sorted[k - 1] {
            run += 1;
        } else {
            out *= factorial(run);
            run = 1;
        }
    }
    out
}

#[test]
fn free_walkers_on_a_million_nodes_follow_their_permanents() {
    // Without interaction a configuration's amplitude is the permanent
    // (bosons, over the root of the multiplicities' factorials) or the
    // determinant (fermions) of the walker's closed-form amplitudes: three
    // bosons from one node of Q20 (2^20 nodes, the 156 240 states of
    // 1.43's HypercubeParticles) and of H(8, 4) (65 536 nodes), three
    // fermions from neighbouring nodes of Q10 and of H(6, 4).
    for (d, q, start, fermions) in [
        (20usize, 2usize, [0u64, 0, 0], false),
        (10, 2, [0, 1, 2], true),
        (8, 4, [0, 0, 0], false),
        (6, 4, [0, 1, 2], true),
    ] {
        let graph = CoordinateGraph::Hamming { d, q };
        let statistics = if fermions {
            Statistics::Fermions
        } else {
            Statistics::Bosons
        };
        let walk = CoordinateInteractingWalk::from_start(
            graph,
            Generator::Adjacency,
            1.0,
            statistics,
            &start,
        )
        .unwrap();
        let t = 0.37;
        let x = walk.evolve(&walk.state_of(&start).unwrap(), t).unwrap();
        let norm: f64 = x.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
        let n = graph.num_nodes();
        for target in [
            [0u64, 1, 6],
            [3, 5, 9],
            [2, 100, n - 1],
            [7, 11, 13],
            [5, 5, 12],
        ] {
            if fermions && target[0] == target[1] {
                continue;
            }
            let mut m = [[C64::new(0.0, 0.0); 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    m[a][b] = amplitude(d, q, 1.0, t, start[a], target[b]);
                }
            }
            let want = combine(&m, fermions).norm_sqr()
                / (multiplicities(&start) * multiplicities(&target));
            let got = walk.probability_of(&x, &target).unwrap();
            assert!(
                (got - want).abs() < 1e-11 * want.max(1e-12),
                "H({d},{q}) {statistics:?} {target:?}: {got:e} vs {want:e}"
            );
        }
        if (d, q) == (20, 2) {
            let cube =
                crate::walks::HypercubeParticles::new(20, statistics, 3, Generator::Adjacency, 1.0)
                    .unwrap();
            assert_eq!(walk.num_orbits(), cube.num_states());
        }
    }
}

#[test]
fn inputs_are_checked() {
    let g = CoordinateGraph::Hamming { d: 3, q: 2 };
    let walk = |start: &[u64], s| {
        CoordinateInteractingWalk::from_start(g, Generator::Adjacency, 1.0, s, start)
    };
    assert!(walk(&[], Statistics::Bosons).is_err());
    assert!(walk(&[8], Statistics::Bosons).is_err());
    assert!(walk(&[1, 1], Statistics::Fermions).is_err());
    assert!(walk(&[0; 7], Statistics::Bosons).is_err());
    assert!(CoordinateGraph::Hamming { d: 70, q: 2 }.word(0).is_err());
    assert!(CoordinateGraph::Johnson { n: 5, k: 5 }.word(0).is_err());
    // Words and nodes are inverse.
    for graph in [
        CoordinateGraph::Hamming { d: 4, q: 3 },
        CoordinateGraph::Johnson { n: 9, k: 4 },
        CoordinateGraph::Kneser { n: 9, k: 3 },
    ] {
        for node in [0, 1, 17, graph.num_nodes() - 1] {
            assert_eq!(graph.node(&graph.word(node).unwrap()).unwrap(), node);
        }
    }
    // Too many orbits are refused before the search.
    let big = CoordinateInteractingWalk::from_start(
        CoordinateGraph::Hamming { d: 40, q: 2 },
        Generator::Adjacency,
        1.0,
        Statistics::Distinguishable,
        &[0, 1, 2, 3, 4, 5],
    );
    assert!(big.is_err());
}

/// Orbits and times on large coordinate graphs, for the documentation
/// (`cargo test --release -- --ignored measure_coordinates --nocapture`).
#[test]
#[ignore]
fn measure_coordinates() {
    use Statistics::*;
    for (graph, statistics, start, u) in [
        (
            CoordinateGraph::Hamming { d: 20, q: 2 },
            Bosons,
            vec![0u64, 0, 0],
            1.0,
        ),
        (
            CoordinateGraph::Hamming { d: 12, q: 2 },
            Fermions,
            vec![0, 1, 2],
            0.0,
        ),
        (
            CoordinateGraph::Hamming { d: 8, q: 4 },
            Fermions,
            vec![0, 1, 2],
            0.0,
        ),
        (
            CoordinateGraph::Hamming { d: 6, q: 2 },
            Bosons,
            vec![0, 0, 0, 0, 0],
            1.0,
        ),
        (
            CoordinateGraph::Johnson { n: 40, k: 3 },
            Bosons,
            vec![0, 0, 1],
            1.0,
        ),
        (
            CoordinateGraph::Kneser { n: 30, k: 3 },
            Distinguishable,
            vec![0, 1],
            0.5,
        ),
    ] {
        let clock = std::time::Instant::now();
        let walk = CoordinateInteractingWalk::from_start(
            graph,
            Generator::Adjacency,
            1.0,
            statistics,
            &start,
        )
        .unwrap()
        .with_interaction(u, 0.0)
        .unwrap();
        let built = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let x = walk.evolve(&walk.state_of(&start).unwrap(), 1.0).unwrap();
        let evolved = clock.elapsed().as_secs_f64();
        let back = walk.probability_of(&x, &start).unwrap();
        eprintln!(
            "{graph:?}, {} {statistics:?} from {start:?}, U = {u}: {} nodes, {:.2e} configurations, \
             {} orbits ({built:.1} s), evolution to t = 1 {evolved:.1} s, return probability {back:.4e}",
            start.len(),
            graph.num_nodes(),
            walk.log_num_configurations().exp(),
            walk.num_orbits()
        );
    }
}
