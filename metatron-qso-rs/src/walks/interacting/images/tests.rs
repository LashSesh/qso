//! Walkers by minimal images against the walks of 1.29, 1.43 and 1.53,
//! Burnside's lemma and closed forms.

use super::*;
use crate::graph::generators;
use crate::walks::{
    CoordinateGraph, CoordinateInteractingWalk, InteractingWalk, SymmetricInteractingWalk,
};
use std::collections::HashSet;

const STATISTICS: [Statistics; 3] = [
    Statistics::Bosons,
    Statistics::Fermions,
    Statistics::Distinguishable,
];

/// Every configuration's probability after `t` against the full walk of
/// 1.29 (from the start alone, or from its orbit state under `group`),
/// and the orbit count against 1.43.
fn compare(
    graph: &Graph,
    statistics: Statistics,
    start: &[usize],
    group: Option<&AutomorphismGroup>,
) {
    let (u, v) = (0.4, 0.25);
    for generator in [Generator::Adjacency, Generator::Laplacian] {
        let ours = match group {
            None => GroupInteractingWalk::from_start(graph, generator, 0.7, statistics, start),
            Some(g) => GroupInteractingWalk::new(graph, generator, 0.7, statistics, start, g),
        }
        .unwrap()
        .with_interaction(u, v)
        .unwrap();
        let theirs = match group {
            None => SymmetricInteractingWalk::from_start(graph, generator, 0.7, statistics, start),
            Some(g) => SymmetricInteractingWalk::new(
                graph,
                generator,
                0.7,
                statistics,
                start.len(),
                g.generators(),
            ),
        }
        .unwrap()
        .with_interaction(u, v)
        .unwrap();
        assert_eq!(
            ours.num_orbits(),
            theirs.num_orbits(),
            "{} {statistics:?} {start:?}",
            graph.name()
        );
        let t = 1.3;
        let x = ours.evolve(&ours.state_of(start).unwrap(), t).unwrap();
        let y = theirs.evolve(&theirs.state_of(start).unwrap(), t).unwrap();
        let full = InteractingWalk::new(graph, generator, 0.7, statistics, start.len())
            .unwrap()
            .with_interaction(u, v)
            .unwrap();
        let mut worst = 0.0f64;
        for i in 0..full.dimension() {
            let sites = full.configuration(i).unwrap().to_vec();
            let p = ours.probability_of(&x, &sites).unwrap();
            let q = theirs.probability_of(&y, &sites).unwrap();
            worst = worst.max((p - q).abs());
        }
        if group.is_none() {
            let psi = full.evolve(&full.basis_state(start).unwrap(), t).unwrap();
            for (i, a) in psi.iter().enumerate() {
                let sites = full.configuration(i).unwrap().to_vec();
                let p = ours.probability_of(&x, &sites).unwrap();
                worst = worst.max((p - a.norm_sqr()).abs());
            }
        }
        assert!(
            worst < 1e-12,
            "{} {generator:?} {statistics:?} {start:?}: {worst:e}",
            graph.name()
        );
        let total: f64 = ours.densities(&x).unwrap().iter().sum();
        assert!((total - start.len() as f64).abs() < 1e-10, "{total}");
    }
}

fn graphs() -> Vec<Graph> {
    vec![
        generators::petersen(),
        generators::paley(13).unwrap(),
        generators::circulant(12, &[1, 4]).unwrap(),
        generators::rook(3, 4).unwrap(),
        generators::shrikhande(),
        generators::hypercube(3),
    ]
}

#[test]
fn walkers_equal_the_full_walk_from_one_configuration() {
    for g in graphs() {
        for statistics in STATISTICS {
            for start in [
                vec![0, 0],
                vec![0, 1],
                vec![0, 0, 0],
                vec![0, 1, 5],
                vec![2, 2, 7],
            ] {
                if statistics == Statistics::Fermions && start.windows(2).any(|w| w[0] == w[1]) {
                    continue;
                }
                // Three particles on the smaller graphs only (the full walk
                // of 1.29 is the reference).
                if start.len() == 3
                    && (g.num_nodes() > 13
                        || (statistics == Statistics::Distinguishable && g.num_nodes() > 10))
                {
                    continue;
                }
                compare(&g, statistics, &start, None);
            }
        }
    }
}

#[test]
fn orbit_states_of_the_whole_group_equal_1_43() {
    for g in graphs() {
        let group = g.automorphism_group();
        for statistics in STATISTICS {
            for start in [vec![0, 1], vec![0, 0, 3], vec![0, 1, 2]] {
                if statistics == Statistics::Fermions && start.windows(2).any(|w| w[0] == w[1]) {
                    continue;
                }
                if SymmetricInteractingWalk::new(
                    &g,
                    Generator::Adjacency,
                    0.7,
                    statistics,
                    start.len(),
                    group.generators(),
                )
                .and_then(|w| w.state_of(&start))
                .is_err()
                {
                    // No invariant state holds the start.
                    assert!(
                        GroupInteractingWalk::new(
                            &g,
                            Generator::Adjacency,
                            0.7,
                            statistics,
                            &start,
                            &group
                        )
                        .is_err()
                    );
                    continue;
                }
                compare(&g, statistics, &start, Some(&group));
            }
        }
    }
}

/// Every element of the group generated by `gens`.
fn elements(n: usize, gens: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let identity: Vec<usize> = (0..n).collect();
    let mut seen: HashSet<Vec<usize>> = HashSet::from([identity.clone()]);
    let mut queue = vec![identity];
    let mut head = 0;
    while head < queue.len() {
        let g = queue[head].clone();
        head += 1;
        for s in gens {
            let h: Vec<usize> = g.iter().map(|&x| s[x]).collect();
            if seen.insert(h.clone()) {
                queue.push(h);
            }
        }
    }
    queue
}

/// The orbits of `k` particles carrying an invariant state, by the cycle
/// index: bosons `[x^k] Π 1/(1 − x^ℓ)`, fermions `[x^k] Π (1 + (−1)^{ℓ+1} x^ℓ)`
/// (the trace of the group on the exterior power), distinguishable
/// particles `fix(g)^k`, averaged over the group.
fn burnside(n: usize, group: &[Vec<usize>], k: usize, statistics: Statistics) -> f64 {
    let mut total = 0.0;
    for g in group {
        let mut seen = vec![false; n];
        let mut cycles = Vec::new();
        for s in 0..n {
            if seen[s] {
                continue;
            }
            let (mut x, mut len) = (s, 0);
            while !seen[x] {
                seen[x] = true;
                x = g[x];
                len += 1;
            }
            cycles.push(len);
        }
        let mut poly = vec![0.0; k + 1];
        poly[0] = 1.0;
        for &l in &cycles {
            let mut next = poly.clone();
            match statistics {
                Statistics::Bosons => {
                    // × 1/(1 − x^l): add shifted copies.
                    for d in l..=k {
                        next[d] += next[d - l];
                    }
                }
                Statistics::Fermions => {
                    let sign = if l % 2 == 1 { 1.0 } else { -1.0 };
                    for d in (l..=k).rev() {
                        next[d] += sign * poly[d - l];
                    }
                }
                Statistics::Distinguishable => {}
            }
            poly = next;
        }
        total += match statistics {
            Statistics::Distinguishable => {
                (cycles.iter().filter(|&&l| l == 1).count() as f64).powi(k as i32)
            }
            _ => poly[k],
        };
    }
    total / group.len() as f64
}

#[test]
fn orbit_counts_follow_burnside() {
    for g in [
        generators::petersen(),
        generators::paley(17).unwrap(),
        generators::circulant(15, &[1, 3]).unwrap(),
        generators::shrikhande(),
    ] {
        let n = g.num_nodes();
        let group = g.automorphism_group();
        let all = elements(n, group.generators());
        assert_eq!(all.len() as u128, group.order().unwrap());
        for statistics in [
            Statistics::Bosons,
            Statistics::Distinguishable,
            Statistics::Fermions,
        ] {
            for k in [2usize, 3] {
                let start: Vec<usize> = (0..k).collect();
                let walk = GroupInteractingWalk::new(
                    &g,
                    Generator::Adjacency,
                    1.0,
                    statistics,
                    &start,
                    &group,
                );
                let want = burnside(n, &all, k, statistics).round() as usize;
                match walk {
                    Ok(w) => assert_eq!(w.num_orbits(), want, "{} {statistics:?} {k}", g.name()),
                    Err(_) => assert_eq!(statistics, Statistics::Fermions),
                }
            }
        }
    }
}

#[test]
fn minimal_images_are_invariants_of_the_orbits() {
    let g = generators::paley(29).unwrap();
    let group = g.automorphism_group();
    let all = elements(29, group.generators());
    let images = Images::new(29, group.generators(), (all.len() as f64).ln());
    let mut rng = Rng::seed_from_u64(158);
    for _ in 0..200 {
        let k = 2 + rng.below(3) as usize;
        let mut sites: Vec<usize> = (0..k).map(|_| rng.below(29) as usize).collect();
        sites.sort_unstable();
        let form = images.form(&sites, Statistics::Bosons).unwrap().unwrap();
        // The image lies in the orbit and is its smallest element; the
        // stabiliser has the order the group's elements give.
        let mut orbit: Vec<Vec<usize>> = all
            .iter()
            .map(|h| {
                let mut c: Vec<usize> = sites.iter().map(|&s| h[s]).collect();
                c.sort_unstable();
                c
            })
            .collect();
        let fixing = orbit.iter().filter(|c| **c == sites).count();
        orbit.sort();
        orbit.dedup();
        let key: Vec<usize> = form.key.iter().map(|&x| x as usize).collect();
        assert_eq!(key, orbit[0], "{sites:?}");
        assert!(
            (form.log_stabiliser - (fixing as f64).ln()).abs() < 1e-9,
            "{sites:?}"
        );
        let h = &all[rng.below(all.len() as u64) as usize];
        let mut moved: Vec<usize> = sites.iter().map(|&s| h[s]).collect();
        moved.sort_unstable();
        assert_eq!(
            images
                .form(&moved, Statistics::Bosons)
                .unwrap()
                .unwrap()
                .key,
            form.key
        );
    }
}

#[test]
fn coordinate_graphs_agree_with_their_canonical_forms() {
    for (graph, start) in [
        (CoordinateGraph::Hamming { d: 3, q: 3 }, vec![0u64, 0, 4]),
        (CoordinateGraph::Johnson { n: 7, k: 2 }, vec![0, 3]),
        (CoordinateGraph::Kneser { n: 7, k: 2 }, vec![0, 1, 1]),
    ] {
        let g = graph.graph().unwrap();
        let sites: Vec<usize> = start.iter().map(|&s| s as usize).collect();
        let ours = GroupInteractingWalk::from_start(
            &g,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            &sites,
        )
        .unwrap();
        let theirs = CoordinateInteractingWalk::from_start(
            graph,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            &start,
        )
        .unwrap();
        assert_eq!(ours.num_orbits(), theirs.num_orbits(), "{graph:?}");
    }
}

/// `⟨y|e^{iγtA}|x⟩` on the Paley graph of `q`, from its three spectral
/// projectors (eigenvalues `(q − 1)/2` and `(−1 ± √q)/2`).
fn paley_amplitude(q: usize, square: &[bool], t: f64, x: usize, y: usize) -> C64 {
    let n = q as f64;
    let k = (n - 1.0) / 2.0;
    let (theta, tau) = ((-1.0 + n.sqrt()) / 2.0, (-1.0 - n.sqrt()) / 2.0);
    let (a, i) = if x == y {
        (0.0, 1.0)
    } else if square[(y + q - x) % q] {
        (1.0, 0.0)
    } else {
        (0.0, 0.0)
    };
    let e0 = 1.0 / n;
    let e1 = (a - k / n - tau * (i - 1.0 / n)) / (theta - tau);
    let e2 = (i - 1.0 / n) - e1;
    C64::from_polar(e0, t * k)
        + C64::from_polar(1.0, t * theta) * e1
        + C64::from_polar(1.0, t * tau) * e2
}

fn det3(m: &[[C64; 3]; 3], fermions: bool) -> C64 {
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

#[test]
fn free_walkers_on_a_paley_graph_follow_their_permanents() {
    // Three bosons from one node of Paley(101) (1.8·10⁵ configurations)
    // under the stabiliser of the node, and three fermions on the orbit
    // state of {0, 1, 2} under the whole group (x ↦ ax + b, a a square):
    // amplitudes from the permanents and determinants of the closed-form
    // single-particle amplitudes, summed over the orbit with its signs.
    // ({0, 1, 2} carries no invariant state: x ↦ 2 − x swaps 0 and 2.)
    let squares = |q: usize| {
        let mut square = vec![false; q];
        for x in 1..q {
            square[x * x % q] = true;
        }
        square
    };
    let t = 0.31;
    let q = 101;
    let g = generators::paley(q).unwrap();
    let square = squares(q);
    let bosons = GroupInteractingWalk::from_start(
        &g,
        Generator::Adjacency,
        1.0,
        Statistics::Bosons,
        &[0, 0, 0],
    )
    .unwrap();
    let x = bosons
        .evolve(&bosons.state_of(&[0, 0, 0]).unwrap(), t)
        .unwrap();
    for target in [[0usize, 1, 2], [3, 3, 17], [5, 50, 100], [7, 7, 7]] {
        // Every particle starts at node 0: the rows are equal.
        let row = target.map(|y| paley_amplitude(q, &square, t, 0, y));
        let m = [row; 3];
        let mult: f64 = {
            let mut c = [0usize; 101];
            for &s in &target {
                c[s] += 1;
            }
            c.iter()
                .map(|&m| (1..=m).product::<usize>() as f64)
                .product()
        };
        let want = det3(&m, false).norm_sqr() / (6.0 * mult);
        let got = bosons.probability_of(&x, &target).unwrap();
        assert!(
            // The propagation is accurate to about 10⁻¹⁵ in each amplitude.
            (got - want).abs() < 1e-10 * want + 1e-15,
            "{target:?}: {got:e} vs {want:e}"
        );
    }
    let q = 401;
    let g = generators::paley(q).unwrap();
    let square = squares(q);
    let group = g.automorphism_group();
    assert_eq!(group.order(), Some((q * (q - 1) / 2) as u128));
    let fermions = GroupInteractingWalk::new(
        &g,
        Generator::Adjacency,
        1.0,
        Statistics::Fermions,
        &[0, 1, 3],
        &group,
    )
    .unwrap();
    let x = fermions
        .evolve(&fermions.state_of(&[0, 1, 3]).unwrap(), t)
        .unwrap();
    // The orbit of {0, 1, 3} with its signs (one sign per configuration:
    // no symmetry maps it to minus itself), from the affine maps.
    let mut orbit: HashMap<[usize; 3], f64> = HashMap::new();
    for a in (1..q).filter(|&a| square[a]) {
        for b in 0..q {
            let mut c = [0usize, 1, 3].map(|s| (a * s + b) % q);
            let mut sign = 1.0;
            for i in 0..3 {
                for j in i + 1..3 {
                    if c[i] > c[j] {
                        sign = -sign;
                    }
                }
            }
            c.sort_unstable();
            if let Some(old) = orbit.insert(c, sign) {
                assert_eq!(old, sign);
            }
        }
    }
    let size = orbit.len() as f64;
    for target in [[0usize, 1, 2], [3, 9, 17], [5, 100, 400]] {
        let mut amp = C64::new(0.0, 0.0);
        for (c, &sign) in &orbit {
            let mut m = [[C64::new(0.0, 0.0); 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    m[a][b] = paley_amplitude(q, &square, t, c[a], target[b]);
                }
            }
            amp += det3(&m, true) * sign;
        }
        let want = amp.norm_sqr() / size;
        let got = fermions.probability_of(&x, &target).unwrap();
        assert!(
            (got - want).abs() < 1e-11,
            "{target:?}: {got:e} vs {want:e}"
        );
    }
}

#[test]
fn inputs_are_checked() {
    let g = generators::petersen();
    let stat = Statistics::Bosons;
    assert!(GroupInteractingWalk::from_start(&g, Generator::Adjacency, 0.0, stat, &[0]).is_err());
    assert!(GroupInteractingWalk::from_start(&g, Generator::Adjacency, 1.0, stat, &[]).is_err());
    assert!(GroupInteractingWalk::from_start(&g, Generator::Adjacency, 1.0, stat, &[10]).is_err());
    assert!(
        GroupInteractingWalk::from_start(&g, Generator::Adjacency, 1.0, stat, &[0; 7]).is_err()
    );
    assert!(
        GroupInteractingWalk::from_start(
            &g,
            Generator::Adjacency,
            1.0,
            Statistics::Fermions,
            &[1, 1]
        )
        .is_err()
    );
    // A group of another graph.
    let other = generators::cycle(10).automorphism_group();
    assert!(GroupInteractingWalk::new(&g, Generator::Adjacency, 1.0, stat, &[0], &other).is_err());
    let walk =
        GroupInteractingWalk::from_start(&g, Generator::Adjacency, 1.0, stat, &[0, 0]).unwrap();
    assert!(walk.state_of(&[0]).is_err());
    assert!(walk.evolve(&[C64::new(1.0, 0.0)], 1.0).is_err() || walk.num_orbits() == 1);
}

/// Orbits and times on large symmetric graphs (`cargo test --release --
/// --ignored measure_images --nocapture`).
#[test]
#[ignore]
fn measure_images() {
    let run = |name: &str, g: &Graph, statistics: Statistics, start: &[usize], whole: bool| {
        let t0 = std::time::Instant::now();
        let group = if whole {
            g.automorphism_group()
        } else {
            let mut fixed = start.to_vec();
            fixed.sort_unstable();
            fixed.dedup();
            g.stabilizer(&fixed).unwrap()
        };
        let t_group = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let w = match GroupInteractingWalk::new(
            g,
            Generator::Adjacency,
            1.0,
            statistics,
            start,
            &group,
        ) {
            Ok(w) => w.with_interaction(1.0, 0.0).unwrap(),
            Err(e) => {
                println!(
                    "{name} {statistics:?} {start:?}: {e} after {:.1} s",
                    t1.elapsed().as_secs_f64()
                );
                return;
            }
        };
        let t_build = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let x = w.evolve(&w.state_of(start).unwrap(), 1.0).unwrap();
        let t_evolve = t2.elapsed().as_secs_f64();
        let norm: f64 = x.iter().map(|a| a.norm_sqr()).sum();
        println!(
            "{name} {statistics:?} {start:?} ({}): {:.2e} configurations, |G| = {:.2e}, {} orbits; group {t_group:.2} s, orbits {t_build:.1} s, evolve {t_evolve:.2} s, norm-1 {:.1e}",
            if whole { "whole group" } else { "stabiliser" },
            w.log_num_configurations().exp(),
            w.log_group_order().exp(),
            w.num_orbits(),
            norm - 1.0
        );
    };
    let g = generators::paley(401).unwrap();
    run("Paley(401)", &g, Statistics::Bosons, &[0, 0, 0], false);
    for q in [401usize, 809, 1601, 3209] {
        let g = generators::paley(q).unwrap();
        run(
            &format!("Paley({q})"),
            &g,
            Statistics::Bosons,
            &[0, 0, 0],
            true,
        );
        run(
            &format!("Paley({q})"),
            &g,
            Statistics::Fermions,
            &[0, 1, 3],
            true,
        );
    }
    let g = generators::paley(101).unwrap();
    run("Paley(101)", &g, Statistics::Fermions, &[0, 1, 3, 7], true);
    let g = generators::circulant(2000, &[1, 7, 40]).unwrap();
    run(
        "circulant(2000; 1, 7, 40)",
        &g,
        Statistics::Bosons,
        &[0, 0, 0],
        true,
    );
}
