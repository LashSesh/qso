use super::*;
use crate::graph::generators;
use crate::photonics::permanent::permanent;
use crate::rng::Rng;
use crate::walks::ContinuousWalk;

const ALL: [Statistics; 3] = [
    Statistics::Bosons,
    Statistics::Fermions,
    Statistics::Distinguishable,
];

/// A small weighted graph with a node of degree three and a pendant node.
fn kite() -> Graph {
    Graph::from_weighted_edges(
        5,
        &[
            (0, 1, 1.0),
            (1, 2, 0.7),
            (2, 0, 1.3),
            (2, 3, 0.4),
            (3, 4, 1.1),
        ],
    )
    .unwrap()
}

fn walk(g: &Graph, generator: Generator, s: Statistics, n: usize) -> InteractingWalk {
    InteractingWalk::new(g, generator, 0.8, s, n).unwrap()
}

#[test]
fn configurations_are_ranked_in_order() {
    let g = generators::cycle(6);
    for (s, p, expected) in [
        (Statistics::Bosons, 3, 56),
        (Statistics::Fermions, 3, 20),
        (Statistics::Distinguishable, 3, 216),
        (Statistics::Bosons, 1, 6),
        (Statistics::Fermions, 6, 1),
    ] {
        let w = walk(&g, Generator::Adjacency, s, p);
        assert_eq!(w.dimension(), expected, "{s:?} {p}");
        let mut seen = std::collections::HashSet::new();
        for k in 0..w.dimension() {
            let c = w.configuration(k).unwrap().to_vec();
            match s {
                Statistics::Bosons => assert!(c.windows(2).all(|x| x[0] <= x[1])),
                Statistics::Fermions => assert!(c.windows(2).all(|x| x[0] < x[1])),
                Statistics::Distinguishable => {}
            }
            assert_eq!(w.index_of(&c).unwrap(), k);
            assert!(seen.insert(c));
        }
    }
    // Any order of the sites names the same configuration.
    let w = walk(&g, Generator::Adjacency, Statistics::Bosons, 3);
    assert_eq!(w.index_of(&[4, 1, 4]), w.index_of(&[1, 4, 4]));
    let d = walk(&g, Generator::Adjacency, Statistics::Distinguishable, 2);
    assert_ne!(d.index_of(&[0, 1]).unwrap(), d.index_of(&[1, 0]).unwrap());
}

#[test]
fn one_particle_is_the_continuous_walk() {
    let g = kite();
    for generator in [Generator::Adjacency, Generator::Laplacian] {
        let single = ContinuousWalk::new(&g, generator, 0.8)
            .unwrap()
            .propagator(1.7)
            .unwrap();
        for s in ALL {
            let u = walk(&g, generator, s, 1).propagator(1.7).unwrap();
            assert!((u - &single).norm() < 1e-12, "{s:?} {generator:?}");
        }
    }
}

/// `⟨a|H|b⟩` built independently on the full tensor-product space of
/// distinguishable particles, then restricted to the symmetric
/// (bosons) or antisymmetric (fermions) subspace.
fn reference_hamiltonian(
    g: &Graph,
    generator: Generator,
    s: Statistics,
    p: usize,
    u: f64,
    v: f64,
    walk: &InteractingWalk,
) -> RMatrix {
    let n = g.num_nodes();
    let h = match generator {
        Generator::Adjacency => g.adjacency_matrix() * -0.8,
        Generator::Laplacian => g.laplacian_matrix() * 0.8,
    };
    let full = n.pow(p as u32);
    let digits =
        |x: usize| -> Vec<usize> { (0..p).map(|k| x / n.pow((p - 1 - k) as u32) % n).collect() };
    let number = |d: &[usize]| d.iter().fold(0, |r, &s| r * n + s);
    let mut hf = RMatrix::zeros(full, full);
    for x in 0..full {
        let d = digits(x);
        for k in 0..p {
            for i in 0..n {
                let mut e = d.clone();
                e[k] = i;
                hf[(number(&e), x)] += h[(i, d[k])];
            }
            for q in k + 1..p {
                if d[k] == d[q] {
                    hf[(x, x)] += u;
                }
                if g.has_edge(d[k], d[q]) {
                    hf[(x, x)] += v;
                }
            }
        }
    }
    if s == Statistics::Distinguishable {
        return hf;
    }
    // Symmetrised basis vectors: Σ_σ (±1)^σ |c_σ⟩, normalised.
    let perms = permutations(p);
    let dim = walk.dimension();
    let mut basis = RMatrix::zeros(full, dim);
    for a in 0..dim {
        let c = walk.configuration(a).unwrap();
        for (perm, sign) in &perms {
            let e: Vec<usize> = perm.iter().map(|&k| c[k]).collect();
            let w = if s == Statistics::Fermions {
                *sign
            } else {
                1.0
            };
            basis[(number(&e), a)] += w;
        }
        let norm = basis.column(a).norm();
        basis.column_mut(a).unscale_mut(norm);
    }
    basis.transpose() * hf * basis
}

fn permutations(p: usize) -> Vec<(Vec<usize>, f64)> {
    if p == 0 {
        return vec![(Vec::new(), 1.0)];
    }
    let mut out = Vec::new();
    for (rest, sign) in permutations(p - 1) {
        // Insert p − 1 at every position; moving it left past j elements
        // changes the sign j times.
        for pos in 0..=rest.len() {
            let mut q = rest.clone();
            q.insert(pos, p - 1);
            let flips = rest.len() - pos;
            out.push((q, if flips % 2 == 0 { sign } else { -sign }));
        }
    }
    out
}

/// `K₅` with weights of both signs: hops jump over up to three fermions.
fn signed_complete() -> Graph {
    let mut edges = Vec::new();
    for u in 0..5 {
        for v in u + 1..5 {
            let w = [0.9, -1.2, 0.5, -0.7, 1.4][(u * 3 + v) % 5];
            edges.push((u, v, w));
        }
    }
    Graph::from_weighted_edges(5, &edges).unwrap()
}

#[test]
fn hamiltonian_matches_the_tensor_product_space() {
    for (g, sizes) in [(kite(), vec![2, 3]), (signed_complete(), vec![4])] {
        hamiltonian_matches_on(&g, &sizes);
    }
}

fn hamiltonian_matches_on(g: &Graph, sizes: &[usize]) {
    let g = g.clone();
    for generator in [Generator::Adjacency, Generator::Laplacian] {
        for s in ALL {
            for &p in sizes {
                let (u, v) = (1.9, -0.6);
                let w = walk(&g, generator, s, p).with_interaction(u, v).unwrap();
                let reference = reference_hamiltonian(&g, generator, s, p, u, v, &w);
                let ours = w.hamiltonian().to_dense();
                assert!(
                    (&ours - &reference).abs().max() < 1e-12,
                    "{s:?} {generator:?} {p}: {}",
                    (&ours - &reference).abs().max()
                );
            }
        }
    }
}

#[test]
fn free_amplitudes_are_permanents_and_determinants() {
    let g = kite();
    let t = 1.3;
    let single = ContinuousWalk::new(&g, Generator::Adjacency, 0.8)
        .unwrap()
        .propagator(t)
        .unwrap();
    let sub = |rows: &[usize], cols: &[usize]| {
        CMatrix::from_fn(rows.len(), cols.len(), |r, c| single[(rows[r], cols[c])])
    };
    let factorials = |c: &[usize]| -> f64 {
        let mut f = 1.0;
        let mut run = 1;
        for k in 1..=c.len() {
            if k < c.len() && c[k] == c[k - 1] {
                run += 1;
            } else {
                f *= (1..=run).product::<usize>() as f64;
                run = 1;
            }
        }
        f
    };
    for s in ALL {
        let w = walk(&g, Generator::Adjacency, s, 3);
        let u = w.propagator(t).unwrap();
        for b in 0..w.dimension() {
            let input = w.configuration(b).unwrap();
            for a in 0..w.dimension() {
                let output = w.configuration(a).unwrap();
                let m = sub(output, input);
                let expected = match s {
                    Statistics::Bosons => {
                        permanent(&m).unwrap() / (factorials(output) * factorials(input)).sqrt()
                    }
                    Statistics::Fermions => m.determinant(),
                    Statistics::Distinguishable => (0..3).map(|k| m[(k, k)]).product(),
                };
                assert!((u[(a, b)] - expected).norm() < 1e-12, "{s:?}");
            }
        }
    }
}

#[test]
fn krylov_propagation_matches_permanents() {
    // Three bosons on 25 nodes: 2925 configurations, beyond the dense
    // limit, so the walk propagates by Krylov.
    let g = generators::grid_2d(5, 5, true);
    let w = walk(&g, Generator::Adjacency, Statistics::Bosons, 3);
    assert!(w.dimension() > DENSE_LIMIT && w.spectrum().is_none());
    let t = 0.9;
    let single = ContinuousWalk::new(&g, Generator::Adjacency, 0.8)
        .unwrap()
        .propagator(t)
        .unwrap();
    let input = [0, 0, 12];
    let psi = w.evolve(&w.basis_state(&input).unwrap(), t).unwrap();
    let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
    for output in [[0, 0, 12], [1, 5, 12], [3, 3, 3], [0, 6, 18]] {
        let m = CMatrix::from_fn(3, 3, |r, c| single[(output[r], input[c])]);
        let mut f: f64 = 2.0; // the input has a site with two bosons
        if output[0] == output[1] && output[1] == output[2] {
            f *= 6.0;
        } else if output[0] == output[1] || output[1] == output[2] {
            f *= 2.0;
        }
        let expected = permanent(&m).unwrap() / f.sqrt();
        let got = psi[w.index_of(&output).unwrap()];
        assert!((got - expected).norm() < 1e-9, "{output:?}");
    }
    assert!(w.propagator(t).is_err());
}

/// `exp(−iHt)ψ` by many small Taylor steps, independent of the Krylov
/// code.
fn taylor_evolve(h: &CsrMatrix, psi: &[C64], t: f64) -> Vec<C64> {
    let steps = 200;
    let dt = t / steps as f64;
    let mut psi = psi.to_vec();
    let mut term = vec![C64::new(0.0, 0.0); psi.len()];
    let mut next = term.clone();
    for _ in 0..steps {
        term.copy_from_slice(&psi);
        for k in 1..=20 {
            h.matvec(&term, &mut next);
            let f = C64::new(0.0, -dt / k as f64);
            for (a, b) in term.iter_mut().zip(&next) {
                *a = b * f;
            }
            for (p, a) in psi.iter_mut().zip(&term) {
                *p += a;
            }
        }
    }
    psi
}

#[test]
fn krylov_propagation_with_interaction() {
    // Beyond the dense limit: five fermions on 15 nodes (3003
    // configurations) and three distinguishable particles on 14 (2744).
    let g15 = generators::circulant(15, &[1, 3]).unwrap();
    let g14 = generators::grid_2d(2, 7, false);
    for (g, s, p, start) in [
        (g15, Statistics::Fermions, 5, vec![0, 3, 4, 9, 14]),
        (g14, Statistics::Distinguishable, 3, vec![0, 0, 7]),
    ] {
        let w = walk(&g, Generator::Laplacian, s, p)
            .with_interaction(2.5, -0.8)
            .unwrap();
        assert!(w.dimension() > DENSE_LIMIT && w.spectrum().is_none());
        let psi0 = w.basis_state(&start).unwrap();
        let ours = w.evolve(&psi0, 0.6).unwrap();
        let reference = taylor_evolve(w.hamiltonian(), &psi0, 0.6);
        let err = ours
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "{s:?}: {err:e}");
    }
}

#[test]
fn hubbard_dimer_has_its_closed_form_spectrum() {
    // Two particles on two sites, hopping −1, on-site interaction U:
    // symmetric (bosons, spin singlet) U and (U ± √(U² + 16))/2, and the
    // antisymmetric state at 0, shifted by V when both sites are full.
    let g = generators::path(2);
    let (u, v) = (3.0, 0.7);
    let make = |s| {
        InteractingWalk::new(&g, Generator::Adjacency, 1.0, s, 2)
            .unwrap()
            .with_interaction(u, v)
            .unwrap()
    };
    let r = (u * u + 16.0f64).sqrt();
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
    let bosons = make(Statistics::Bosons);
    let mut expected = [(u - r) / 2.0, u, (u + r) / 2.0];
    expected.sort_by(f64::total_cmp);
    // V shifts only |1,1⟩, which mixes with the others; without V:
    let free_v = InteractingWalk::new(&g, Generator::Adjacency, 1.0, Statistics::Bosons, 2)
        .unwrap()
        .with_interaction(u, 0.0)
        .unwrap();
    assert!(close(free_v.spectrum().unwrap(), &expected));
    assert_eq!(bosons.dimension(), 3);
    assert!(close(make(Statistics::Fermions).spectrum().unwrap(), &[v]));
    let mut both = vec![
        v,
        (u + v - (u - v).hypot(4.0)) / 2.0,
        u,
        (u + v + (u - v).hypot(4.0)) / 2.0,
    ];
    both.sort_by(f64::total_cmp);
    // With V: the bosonic block is [[U, −√2, 0], [−√2, V, −√2], [0, −√2, U]],
    // eigenvalues U and (U + V ± √((U − V)² + 16))/2; distinguishable
    // particles add the antisymmetric state at V.
    assert!(close(
        make(Statistics::Distinguishable).spectrum().unwrap(),
        &both
    ));
    let mut sym = both.clone();
    sym.remove(sym.iter().position(|&x| (x - v).abs() < 1e-12).unwrap());
    assert!(close(bosons.spectrum().unwrap(), &sym));
}

#[test]
fn two_particle_interference_on_an_edge() {
    // Hong–Ou–Mandel: one particle on each end of an edge, a quarter period
    // later two bosons are never found apart, two distinguishable particles
    // half of the time.
    let g = generators::path(2);
    let t = std::f64::consts::FRAC_PI_4;
    let p = |s| {
        let w = InteractingWalk::new(&g, Generator::Adjacency, 1.0, s, 2).unwrap();
        let probs = w.probabilities(&[0, 1], t).unwrap();
        (probs[w.index_of(&[0, 1]).unwrap()], w)
    };
    let (bosons, w) = p(Statistics::Bosons);
    assert!(bosons < 1e-15);
    let psi = w.evolve(&w.basis_state(&[0, 1]).unwrap(), t).unwrap();
    let corr = w.pair_correlations(&psi).unwrap();
    assert!(corr[(0, 1)].abs() < 1e-15 && (corr[(0, 0)] - 2.0).abs() < 1e-12);
    let (dist, w) = p(Statistics::Distinguishable);
    assert!((dist - 0.25).abs() < 1e-12);
    assert!((dist + p_of(&w, &[1, 0], t) - 0.5).abs() < 1e-12);
    assert!((p(Statistics::Fermions).0 - 1.0).abs() < 1e-12);
}

fn p_of(w: &InteractingWalk, sites: &[usize], t: f64) -> f64 {
    w.probabilities(&[0, 1], t).unwrap()[w.index_of(sites).unwrap()]
}

#[test]
fn densities_and_correlations_sum_to_the_particles() {
    let g = generators::petersen();
    for s in ALL {
        let w = walk(&g, Generator::Laplacian, s, 3)
            .with_interaction(2.0, 0.5)
            .unwrap();
        let psi = w.evolve(&w.basis_state(&[0, 4, 7]).unwrap(), 2.3).unwrap();
        let d = w.densities(&psi).unwrap();
        assert!((d.iter().sum::<f64>() - 3.0).abs() < 1e-12);
        let c = w.pair_correlations(&psi).unwrap();
        assert!((c.sum() - 9.0).abs() < 1e-11);
        for (i, di) in d.iter().enumerate() {
            assert!((c.row(i).sum() - 3.0 * di).abs() < 1e-11);
            // ⟨n_i²⟩ = ⟨n_i⟩ for fermions.
            if s == Statistics::Fermions {
                assert!((c[(i, i)] - di).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn certificates_are_graph_invariants() {
    let g = kite();
    let mut rng = Rng::seed_from_u64(7);
    let mut perm: Vec<usize> = (0..5).collect();
    for s in ALL {
        let a = walk(&g, Generator::Adjacency, s, 2)
            .with_interaction(1.5, 0.3)
            .unwrap()
            .certificate(&[0.7, 2.0])
            .unwrap();
        for _ in 0..3 {
            rng.shuffle(&mut perm);
            let h = g.relabeled(&perm).unwrap();
            let b = walk(&h, Generator::Adjacency, s, 2)
                .with_interaction(1.5, 0.3)
                .unwrap()
                .certificate(&[0.7, 2.0])
                .unwrap();
            assert!(certificate_distance(&a, &b) < 1e-12, "{s:?}");
        }
    }
    assert_eq!(certificate_distance(&[1.0], &[1.0, 2.0]), f64::INFINITY);
    assert!(certificate_distance(&[f64::NAN, 1.0], &[0.0, 1.0]).is_nan());
}

#[test]
fn certificate_tolerance_grows_with_the_time() {
    // Rounding grows linearly with t: at t = 10⁶ the certificates of a
    // relabelled 4 × 4 rook's graph differ by about 10⁻⁹, far below the
    // tolerance, which still separates the Shrikhande graph.
    let rook = generators::rook(4, 4).unwrap();
    let mut perm: Vec<usize> = (0..16).collect();
    Rng::seed_from_u64(3).shuffle(&mut perm);
    let relabelled = rook.relabeled(&perm).unwrap();
    let make = |g: &Graph| {
        walk(g, Generator::Adjacency, Statistics::Bosons, 2)
            .with_interaction(1.0, 0.0)
            .unwrap()
    };
    for t in [1.0, 1e3, 1e6] {
        let w = make(&rook);
        let tolerance = w.certificate_tolerance(&[t]).unwrap();
        let a = w.certificate(&[t]).unwrap();
        let same = certificate_distance(&a, &make(&relabelled).certificate(&[t]).unwrap());
        assert!(
            same < 1e-3 * tolerance,
            "t = {t}: {same:e} vs {tolerance:e}"
        );
        let other = make(&generators::shrikhande()).certificate(&[t]).unwrap();
        assert!(
            certificate_distance(&a, &other) > 10.0 * tolerance,
            "t = {t}"
        );
    }
    let w = make(&rook);
    assert!(w.certificate_tolerance(&[]).is_err());
    assert!(w.certificate_tolerance(&[f64::INFINITY]).is_err());
    assert!(w.certificate(&[f64::NAN]).is_err());
}

#[test]
fn interaction_tells_the_shrikhande_and_rook_graphs_apart() {
    let (a, b) = (generators::shrikhande(), generators::rook(4, 4).unwrap());
    assert!(!a.is_isomorphic(&b));
    let times = [1.0];
    let cert = |g: &Graph, s, u| {
        walk(g, Generator::Adjacency, s, 2)
            .with_interaction(u, 0.0)
            .unwrap()
            .certificate(&times)
            .unwrap()
    };
    for s in ALL {
        // Without interaction: the same certificate (Gamble et al.).
        assert!(certificate_distance(&cert(&a, s, 0.0), &cert(&b, s, 0.0)) < 1e-10);
    }
    // On-site interaction separates bosons and distinguishable particles;
    // fermions never share a site.
    for s in [Statistics::Bosons, Statistics::Distinguishable] {
        assert!(certificate_distance(&cert(&a, s, 1.0), &cert(&b, s, 1.0)) > 1e-6);
    }
}

#[test]
fn invalid_walks_are_rejected() {
    let g = generators::cycle(4);
    let new = |s, p| InteractingWalk::new(&g, Generator::Adjacency, 1.0, s, p);
    assert!(new(Statistics::Fermions, 5).is_err());
    assert!(new(Statistics::Bosons, 0).is_err());
    assert!(InteractingWalk::new(&g, Generator::Adjacency, 0.0, Statistics::Bosons, 1).is_err());
    assert!(
        InteractingWalk::new(
            &Graph::new(0),
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            1
        )
        .is_err()
    );
    // 4¹² configurations of 12 distinguishable particles are too many.
    assert!(new(Statistics::Distinguishable, 12).is_err());
    let big = generators::cycle(200);
    assert!(InteractingWalk::new(&big, Generator::Adjacency, 1.0, Statistics::Bosons, 4).is_err());
    let w = new(Statistics::Fermions, 2).unwrap();
    assert!(w.index_of(&[1, 1]).is_err());
    assert!(w.index_of(&[1]).is_err());
    assert!(w.index_of(&[1, 4]).is_err());
    assert!(w.configuration(6).is_err());
    assert!(w.evolve(&[C64::new(1.0, 0.0)], 1.0).is_err());
    assert!(
        w.evolve(&w.basis_state(&[0, 1]).unwrap(), f64::NAN)
            .is_err()
    );
    assert!(w.clone().with_interaction(f64::INFINITY, 0.0).is_err());
    assert!(w.certificate(&[]).is_err());
    // Too many values in a certificate, stored sites, or nonzeros.
    let k = InteractingWalk::new(
        &generators::complete(64),
        Generator::Adjacency,
        1.0,
        Statistics::Fermions,
        2,
    )
    .unwrap();
    assert!(k.certificate(&vec![1.0; 100]).is_err());
    let edge = generators::path(2);
    assert!(
        InteractingWalk::new(
            &edge,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            100_000
        )
        .is_err()
    );
    assert!(
        InteractingWalk::new(
            &edge,
            Generator::Adjacency,
            1.0,
            Statistics::Bosons,
            usize::MAX
        )
        .is_err()
    );
    let point = Graph::new(1);
    assert!(
        InteractingWalk::new(
            &point,
            Generator::Adjacency,
            1.0,
            Statistics::Distinguishable,
            1 << 26
        )
        .is_err()
    );
    let one = InteractingWalk::new(&point, Generator::Adjacency, 1.0, Statistics::Bosons, 1000)
        .unwrap()
        .with_interaction(2.0, 0.0)
        .unwrap();
    // 1000 bosons on one site: U · 1000 · 999 / 2.
    assert_eq!(one.spectrum().unwrap(), &[999_000.0]);
    let dense = generators::complete(1500);
    assert!(
        InteractingWalk::new(&dense, Generator::Adjacency, 1.0, Statistics::Fermions, 2).is_err()
    );
}

#[test]
fn many_sites_are_enumerated_quickly() {
    // One walker on 2¹⁶ nodes and two fermions on 2000: the configurations
    // come from a successor function, not one search per rank.
    let start = std::time::Instant::now();
    let w = walk(
        &generators::cycle(1 << 16),
        Generator::Adjacency,
        Statistics::Bosons,
        1,
    );
    assert_eq!(w.index_of(&[65_535]).unwrap(), 65_535);
    let f = walk(
        &generators::cycle(2000),
        Generator::Adjacency,
        Statistics::Fermions,
        2,
    );
    assert_eq!(f.dimension(), 1999 * 1000);
    assert_eq!(f.index_of(&[1998, 1999]).unwrap(), f.dimension() - 1);
    assert!(start.elapsed().as_secs_f64() < 30.0);
}

#[test]
fn certificates_from_orbit_representatives() {
    // Two interacting bosons on the strongly regular graphs of 1.29: one
    // start per orbit of the automorphisms gives the whole certificate,
    // with dense and with Krylov rows.
    let times = [0.7, 2.0];
    for g in [
        generators::shrikhande(),
        generators::rook(4, 4).unwrap(),
        generators::chang(1).unwrap(),
        generators::petersen(),
    ] {
        let group = g.automorphism_group();
        for statistics in ALL {
            let w = InteractingWalk::new(&g, Generator::Adjacency, 1.0, statistics, 2)
                .unwrap()
                .with_interaction(1.0, 0.3)
                .unwrap();
            let full = w.certificate(&times).unwrap();
            let tolerance = w.certificate_tolerance(&times).unwrap();
            for krylov in [false, true] {
                let reduced = w
                    .orbit_certificate(&times, group.generators(), krylov)
                    .unwrap();
                let distance = certificate_distance(&full, &reduced);
                assert!(distance < tolerance, "{statistics:?} {krylov}: {distance}");
            }
        }
    }
    // Weighted graphs with the Laplacian: the weighted degree is summed in
    // the order of the neighbours, so it differs between the nodes of one
    // orbit in the last bit, which must not reject the symmetry.
    let mut edges = Vec::new();
    for (jump, weight) in [(1, 1.0), (2, 0.45), (4, -0.3)] {
        for u in 0..8 {
            let v = (u + jump) % 8;
            if jump < 4 || u < v {
                edges.push((u, v, weight));
            }
        }
    }
    let weighted = Graph::from_weighted_edges(8, &edges).unwrap();
    let group = weighted.automorphism_group();
    assert!(!group.generators().is_empty());
    for statistics in ALL {
        let w = InteractingWalk::new(&weighted, Generator::Laplacian, 1.0, statistics, 2)
            .unwrap()
            .with_interaction(0.8, -0.2)
            .unwrap();
        let full = w.certificate(&times).unwrap();
        let reduced = w.certificate_by_orbits(&times, group.generators()).unwrap();
        assert!(certificate_distance(&full, &reduced) < w.certificate_tolerance(&times).unwrap());
    }
    // Beyond the dense limit: two bosons on Q₆ (2080 configurations) and a
    // relabelled copy agree, where the dense certificate is refused.
    let q6 = generators::hypercube(6);
    let mut rng = Rng::seed_from_u64(9);
    let mut relabel: Vec<usize> = (0..64).collect();
    for i in (1..64).rev() {
        relabel.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let edges: Vec<(usize, usize, f64)> = q6
        .edges()
        .iter()
        .map(|&(u, v, w)| (relabel[u], relabel[v], w))
        .collect();
    let copy = Graph::from_weighted_edges(64, &edges).unwrap();
    let certificate = |g: &Graph| {
        let w = InteractingWalk::new(g, Generator::Adjacency, 1.0, Statistics::Bosons, 2)
            .unwrap()
            .with_interaction(1.0, 0.0)
            .unwrap();
        assert!(w.certificate(&[1.0]).is_err());
        let c = w
            .certificate_by_orbits(&[1.0], g.automorphism_group().generators())
            .unwrap();
        (c, w.certificate_tolerance(&[1.0]).unwrap())
    };
    let (a, tolerance) = certificate(&q6);
    let (b, _) = certificate(&copy);
    assert_eq!(a.len(), 2080 * 2080);
    assert!(certificate_distance(&a, &b) < tolerance);
    // Each row is a probability distribution (Krylov steps keep the norm
    // to about 10⁻¹¹).
    let total: f64 = a.iter().sum();
    assert!((total - 2080.0).abs() < 1e-7, "{total}");
    // A permutation that is no automorphism is refused.
    let w = InteractingWalk::new(&q6, Generator::Adjacency, 1.0, Statistics::Bosons, 2).unwrap();
    let mut swap: Vec<usize> = (0..64).collect();
    swap.swap(0, 3);
    assert!(w.certificate_by_orbits(&[1.0], &[swap]).is_err());
}
