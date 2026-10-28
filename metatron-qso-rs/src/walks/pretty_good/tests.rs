use super::*;
use crate::error::QsoError;
use crate::graph::generators;
use crate::walks::{ContinuousWalk, StateTransfer};

fn is_prime(n: usize) -> bool {
    n >= 2
        && (2..)
            .take_while(|d| d * d <= n)
            .all(|d| !n.is_multiple_of(d))
}

#[test]
fn cyclotomic_polynomials_and_powers() {
    assert_eq!(cyclotomic_polynomial(1).unwrap(), vec![-1, 1]);
    assert_eq!(cyclotomic_polynomial(6).unwrap(), vec![1, -1, 1]);
    assert_eq!(cyclotomic_polynomial(12).unwrap(), vec![1, 0, -1, 0, 1]);
    // Φ_105 is the first with a coefficient −2.
    assert!(cyclotomic_polynomial(105).unwrap().contains(&-2));
    // Reduced powers agree numerically with ζ^k.
    for n in [5usize, 12, 30, 105, 210] {
        let f = Cyclotomic::new(n).unwrap();
        let z = C64::from_polar(1.0, std::f64::consts::TAU / n as f64);
        for k in 0..n {
            let v: C64 = f.powers[k]
                .iter()
                .enumerate()
                .map(|(i, &c)| z.powi(i as i32) * c as f64)
                .sum();
            assert!((v - z.powi(k as i32)).norm() < 1e-9, "{n} {k}");
        }
    }
}

#[test]
fn integer_kernels_are_relations() {
    let cols = vec![vec![2, 0, 1], vec![4, 0, 2], vec![0, 3, 0], vec![1, 1, 1]];
    let (kernel, rank) = integer_kernel(&cols).unwrap();
    assert_eq!(rank, 3);
    assert_eq!(kernel.len(), 1);
    for l in &kernel {
        for row in 0..3 {
            let sum: i128 = l.iter().zip(&cols).map(|(x, c)| x * c[row] as i128).sum();
            assert_eq!(sum, 0);
        }
    }
    // A primitive generator: (2, −1, 0, 0) up to sign.
    assert_eq!(kernel[0][2], 0);
    assert_eq!(kernel[0][0].abs(), 2);
}

#[test]
fn ends_of_paths_follow_godsil_kirkland_severini_smith() {
    // Adjacency walk on P_n, end to end: PGST iff n + 1 is a prime, twice
    // a prime or a power of two (Phys. Rev. Lett. 109, 050502 (2012)).
    for n in 2..=200usize {
        let m = n + 1;
        let want = is_prime(m) || (m % 2 == 0 && is_prime(m / 2)) || m.is_power_of_two();
        let t = PrettyGoodTransfer::path(n, Generator::Adjacency, 1.0, 0, n - 1).unwrap();
        assert!(t.strongly_cospectral(), "P{n}");
        assert_eq!(t.pretty_good(), want, "P{n}");
        // PST only for P2 and P3.
        assert_eq!(t.perfect_time().is_some(), n <= 3, "P{n}");
        if let Some(l) = t.obstruction() {
            // The relation holds and has odd parity.
            let levels = &t.levels;
            let sum: f64 = l
                .iter()
                .zip(&levels[1..])
                .map(|(&x, lv)| x as f64 * (lv.energy - levels[0].energy))
                .sum();
            assert!(sum.abs() < 1e-8, "P{n}: {sum}");
            let parity: i64 = l
                .iter()
                .zip(&levels[1..])
                .map(|(&x, lv)| x * i64::from(lv.sign != levels[0].sign))
                .sum();
            assert_eq!(parity.rem_euclid(2), 1, "P{n}");
        }
    }
}

#[test]
fn laplacian_paths_have_pgst_exactly_at_powers_of_two() {
    // Heisenberg (XXX) chains: PGST between the ends iff n is a power of
    // two (Banchi, Coutinho, Godsil and Severini 2017).
    for n in 2..=130usize {
        let t = PrettyGoodTransfer::path(n, Generator::Laplacian, 1.0, 0, n - 1).unwrap();
        assert!(t.strongly_cospectral(), "P{n}");
        assert_eq!(t.pretty_good(), n.is_power_of_two(), "P{n}");
        assert_eq!(t.perfect_time().is_some(), n == 2, "P{n}");
    }
}

#[test]
fn cycles_have_antipodal_pgst_exactly_at_powers_of_two() {
    // Pal, Electron. J. Combin. 24 (2017): C_n has PGST between antipodal
    // vertices iff n is a power of two; PST only for C_4.
    for n in (4..=128usize).step_by(2) {
        for generator in [Generator::Adjacency, Generator::Laplacian] {
            let t = PrettyGoodTransfer::cycle(n, generator, 1.0, 0, n / 2).unwrap();
            assert!(t.strongly_cospectral(), "C{n}");
            assert_eq!(t.pretty_good(), n.is_power_of_two(), "C{n} {generator:?}");
            assert_eq!(t.perfect_time().is_some(), n == 4, "C{n}");
        }
    }
    // Non-antipodal pairs are not strongly cospectral.
    assert!(
        !PrettyGoodTransfer::cycle(8, Generator::Adjacency, 1.0, 0, 3)
            .unwrap()
            .strongly_cospectral()
    );
}

#[test]
fn exact_decisions_agree_with_the_numerical_pst_analysis() {
    // Hypercubes, circulants and paths: strong cospectrality and PST
    // (time included) agree with StateTransfer, whose PST answers are
    // confirmed by evaluating the amplitude.
    let mut cases: Vec<(PrettyGoodTransfer, Graph, Generator, f64, usize, usize)> = Vec::new();
    for d in 1..=6usize {
        let units: Vec<Vec<i64>> = (0..d)
            .map(|i| (0..d).map(|j| i64::from(i == j)).collect())
            .collect();
        let (u, v) = (vec![0usize; d], vec![1usize; d]);
        let exact =
            PrettyGoodTransfer::cayley(&[2; 6][..d], &units, Generator::Adjacency, 0.7, &u, &v)
                .unwrap();
        cases.push((
            exact,
            generators::hypercube(d),
            Generator::Adjacency,
            0.7,
            0,
            (1 << d) - 1,
        ));
    }
    for (n, jumps) in [
        (8usize, vec![1usize, 2]),
        (10, vec![1, 3]),
        (12, vec![1, 5]),
        (16, vec![1, 2, 4]),
        (9, vec![1, 3]),
    ] {
        let mut set: Vec<Vec<i64>> = Vec::new();
        for &j in &jumps {
            set.push(vec![j as i64]);
            if 2 * j != n {
                set.push(vec![-(j as i64)]);
            }
        }
        let g = generators::circulant(n, &jumps).unwrap();
        for v in 1..n {
            let exact =
                PrettyGoodTransfer::cayley(&[n], &set, Generator::Laplacian, 1.3, &[0], &[v])
                    .unwrap();
            cases.push((exact, g.clone(), Generator::Laplacian, 1.3, 0, v));
        }
    }
    for n in 2..=12usize {
        for v in 1..n {
            for generator in [Generator::Adjacency, Generator::Laplacian] {
                let exact = PrettyGoodTransfer::path(n, generator, 1.0, 0, v).unwrap();
                cases.push((exact, generators::path(n), generator, 1.0, 0, v));
            }
        }
    }
    // Products of cyclic groups: tori, Z_3 × Z_5, and the Hamming graph
    // H(2, 3) (two coordinates mod 3, one changed per edge).
    for (moduli, set) in [
        (
            vec![4usize, 6],
            vec![vec![1i64, 0], vec![-1, 0], vec![0, 1], vec![0, -1]],
        ),
        (
            vec![3, 5],
            vec![
                vec![1, 0],
                vec![-1, 0],
                vec![0, 1],
                vec![0, -1],
                vec![1, 1],
                vec![-1, -1],
            ],
        ),
        (
            vec![3, 3],
            vec![vec![1, 0], vec![2, 0], vec![0, 1], vec![0, 2]],
        ),
        (
            vec![2, 4, 2],
            vec![
                vec![1, 0, 0],
                vec![0, 1, 0],
                vec![0, 3, 0],
                vec![0, 0, 1],
                vec![1, 2, 1],
            ],
        ),
    ] {
        let size: usize = moduli.iter().product();
        let coords = |mut x: usize| -> Vec<usize> {
            moduli
                .iter()
                .map(|&m| {
                    let c = x % m;
                    x /= m;
                    c
                })
                .collect()
        };
        let index = |c: &[i64]| -> usize {
            c.iter()
                .zip(&moduli)
                .rev()
                .fold(0, |acc, (&x, &m)| acc * m + x.rem_euclid(m as i64) as usize)
        };
        let mut edges = Vec::new();
        for a in 0..size {
            let ca = coords(a);
            for s in &set {
                let sum: Vec<i64> = ca.iter().zip(s).map(|(&x, &y)| x as i64 + y).collect();
                let b = index(&sum);
                if a < b {
                    edges.push((a, b));
                }
            }
        }
        let g = Graph::from_edges(size, &edges).unwrap();
        for v in 1..size {
            for generator in [Generator::Adjacency, Generator::Laplacian] {
                let exact = PrettyGoodTransfer::cayley(
                    &moduli,
                    &set,
                    generator,
                    0.9,
                    &coords(0),
                    &coords(v),
                )
                .unwrap();
                cases.push((exact, g.clone(), generator, 0.9, 0, v));
            }
        }
    }
    let mut pst = 0;
    for (exact, graph, generator, gamma, u, v) in &cases {
        let numeric = StateTransfer::new(graph, *generator, *gamma, *u, *v).unwrap();
        let label = format!("{} {u}→{v}", graph.name());
        assert_eq!(
            exact.strongly_cospectral(),
            numeric.strongly_cospectral(),
            "{label}"
        );
        match (exact.perfect_time(), numeric.perfect_transfer_time()) {
            (Some(a), Some(b)) => {
                assert!((a - b).abs() < 1e-9 * b, "{label}: {a} vs {b}");
                pst += 1;
            }
            (None, None) => {}
            (a, b) => panic!("{label}: {a:?} vs {b:?}"),
        }
        if exact.perfect_time().is_some() {
            assert!(exact.pretty_good(), "{label}");
        }
        // The fidelity is the walk's, for every pair.
        for t in [0.3, 1.7, 4.1] {
            assert!(
                (exact.fidelity(t) - numeric.fidelity(t)).abs() < 1e-10,
                "{label} t = {t}: {} vs {}",
                exact.fidelity(t),
                numeric.fidelity(t)
            );
        }
    }
    assert!(pst >= 8, "{pst}");
}

#[test]
fn pretty_good_times_reach_their_fidelity_in_the_walk() {
    // P4 (n + 1 = 5 prime) has PGST without PST: a time with fidelity above
    // 0.999 within the window, confirmed by propagating the walk.
    let exact = PrettyGoodTransfer::path(4, Generator::Adjacency, 1.0, 0, 3).unwrap();
    assert!(exact.pretty_good() && exact.perfect_time().is_none());
    let best = exact.best_transfer(2000.0, 200_000).unwrap();
    assert!(best.fidelity > 0.999, "{best:?}");
    let walk = ContinuousWalk::new(&generators::path(4), Generator::Adjacency, 1.0).unwrap();
    let p = walk.probabilities(0, best.time).unwrap();
    assert!((p[3] - best.fidelity).abs() < 1e-9);
    // P8 (n + 1 = 9) has none: the fidelity stays bounded away from 1.
    let none = PrettyGoodTransfer::path(8, Generator::Adjacency, 1.0, 0, 7).unwrap();
    assert!(!none.pretty_good());
    assert!(none.best_transfer(2000.0, 200_000).unwrap().fidelity < 0.99);
}

#[test]
fn graphs_are_recognised_and_inputs_checked() {
    // A path with shuffled labels.
    let perm = [3usize, 0, 4, 1, 2];
    let edges: Vec<(usize, usize)> = (0..4).map(|i| (perm[i], perm[i + 1])).collect();
    let g = Graph::from_edges(5, &edges).unwrap();
    let t =
        PrettyGoodTransfer::from_graph(&g, Generator::Adjacency, 1.0, perm[0], perm[4]).unwrap();
    assert!(t.pretty_good()); // n + 1 = 6 = 2·3
    let c = PrettyGoodTransfer::from_graph(&generators::cycle(8), Generator::Laplacian, 1.0, 2, 6)
        .unwrap();
    assert!(c.pretty_good());
    assert!(
        PrettyGoodTransfer::from_graph(&generators::hypercube(3), Generator::Adjacency, 1.0, 0, 7)
            .is_err()
    );
    assert!(PrettyGoodTransfer::path(1, Generator::Adjacency, 1.0, 0, 0).is_err());
    assert!(PrettyGoodTransfer::path(4, Generator::Adjacency, 1.0, 1, 1).is_err());
    assert!(PrettyGoodTransfer::path(4, Generator::Adjacency, 0.0, 0, 3).is_err());
    assert!(
        PrettyGoodTransfer::cayley(&[6], &[vec![1]], Generator::Adjacency, 1.0, &[0], &[3])
            .is_err()
    );
    assert!(
        PrettyGoodTransfer::cayley(&[6], &[vec![0]], Generator::Adjacency, 1.0, &[0], &[3])
            .is_err()
    );
    assert!(
        PrettyGoodTransfer::cayley(
            &[6],
            &[vec![1], vec![5]],
            Generator::Adjacency,
            1.0,
            &[0],
            &[7]
        )
        .is_err()
    );
    // Not strongly cospectral: no PGST, rank 0, the true fidelity (the
    // walk's) and its window search.
    let t = PrettyGoodTransfer::path(5, Generator::Adjacency, 1.0, 0, 1).unwrap();
    assert!(!t.strongly_cospectral() && !t.pretty_good() && t.rank() == 0);
    let walk = ContinuousWalk::new(&generators::path(5), Generator::Adjacency, 1.0).unwrap();
    assert!((t.fidelity(1.0) - walk.probabilities(0, 1.0).unwrap()[1]).abs() < 1e-12);
    let best = t.best_transfer(10.0, 1000).unwrap();
    assert!((best.fidelity - walk.probabilities(0, best.time).unwrap()[1]).abs() < 1e-12);
    // Repeated or missing connection elements, hopping rates out of range,
    // sizes beyond the field, and errors that name the argument.
    let bad = |r: Result<PrettyGoodTransfer>, name: &str| match r {
        Err(QsoError::InvalidArgument { name: n, .. }) => assert_eq!(n, name),
        other => panic!("{other:?}"),
    };
    bad(
        PrettyGoodTransfer::cayley(
            &[6],
            &[vec![1], vec![5], vec![7], vec![-1]],
            Generator::Adjacency,
            1.0,
            &[0],
            &[3],
        ),
        "connection",
    );
    bad(
        PrettyGoodTransfer::cayley(&[6], &[], Generator::Adjacency, 1.0, &[0], &[3]),
        "connection",
    );
    bad(
        PrettyGoodTransfer::cayley(
            &[6],
            &[vec![1], vec![5]],
            Generator::Adjacency,
            1.0,
            &[0],
            &[9],
        ),
        "v",
    );
    bad(
        PrettyGoodTransfer::path(4, Generator::Adjacency, 1e-200, 0, 3),
        "gamma",
    );
    bad(
        PrettyGoodTransfer::path(4, Generator::Adjacency, 1e200, 0, 3),
        "gamma",
    );
    bad(
        PrettyGoodTransfer::path(1_000_000, Generator::Adjacency, 1.0, 0, 3),
        "n",
    );
    bad(
        PrettyGoodTransfer::cycle(10_000, Generator::Adjacency, 1.0, 0, 3),
        "n",
    );
    bad(
        PrettyGoodTransfer::cayley(
            &[9973, 2],
            &[vec![1, 0], vec![-1, 0]],
            Generator::Adjacency,
            1.0,
            &[0, 0],
            &[1, 0],
        ),
        "moduli",
    );
    // A group whose relation search would not fit in memory is refused
    // before it starts (17 000 distinct eigenvalues).
    bad(
        PrettyGoodTransfer::cayley(
            &[1024, 64],
            &[vec![1, 0], vec![-1, 0], vec![0, 1], vec![0, -1]],
            Generator::Adjacency,
            1.0,
            &[0, 0],
            &[512, 0],
        ),
        "graph",
    );
    // Odd cycles use Q(ζ_n) itself: C_5001 fits.
    let odd = PrettyGoodTransfer::cycle(5001, Generator::Adjacency, 1.0, 0, 1).unwrap();
    assert!(!odd.strongly_cospectral());
}

#[test]
fn perfect_transfer_times_are_exact_at_extreme_rates() {
    // C_4 antipodal: PST at π/(2γ), from the integer data alone.
    for gamma in [1e-90, 1e-3, 1.0, 1e90] {
        let t = PrettyGoodTransfer::cycle(4, Generator::Adjacency, gamma, 0, 2).unwrap();
        let expected = std::f64::consts::FRAC_PI_2 / gamma;
        let got = t.perfect_time().unwrap();
        assert!((got - expected).abs() <= 1e-12 * expected, "{gamma}: {got}");
    }
}
