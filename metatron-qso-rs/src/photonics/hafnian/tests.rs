use super::*;
use crate::Rng;
use crate::photonics::permanent;

fn random_symmetric(n: usize, seed: u64) -> CMatrix {
    let mut rng = Rng::seed_from_u64(seed);
    let a = CMatrix::from_fn(n, n, |_, _| C64::new(rng.normal(), rng.normal()));
    &a + a.transpose()
}

/// Sum over (loop) matchings by recursion on the first vertex.
fn brute(a: &CMatrix, loops: bool) -> C64 {
    fn rec(a: &CMatrix, rest: &[usize], loops: bool) -> C64 {
        let Some((&i, others)) = rest.split_first() else {
            return ONE;
        };
        let mut s = ZERO;
        if loops {
            s += a[(i, i)] * rec(a, others, loops);
        }
        for (k, &j) in others.iter().enumerate() {
            let mut left: Vec<usize> = others.to_vec();
            left.remove(k);
            s += a[(i, j)] * rec(a, &left, loops);
        }
        s
    }
    let all: Vec<usize> = (0..a.nrows()).collect();
    rec(a, &all, loops)
}

fn close(a: C64, b: C64, tol: f64) -> bool {
    (a - b).norm() <= tol * b.norm().max(1.0)
}

#[test]
fn hafnians_match_brute_force() {
    for (n, seed) in [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)] {
        let a = random_symmetric(n, seed);
        assert!(close(hafnian(&a).unwrap(), brute(&a, false), 1e-11), "{n}");
        assert!(
            close(loop_hafnian(&a).unwrap(), brute(&a, true), 1e-11),
            "{n}"
        );
    }
    for n in [1, 3, 5, 7] {
        let a = random_symmetric(n, 10 + n as u64);
        assert_eq!(hafnian(&a).unwrap(), ZERO);
        assert!(
            close(loop_hafnian(&a).unwrap(), brute(&a, true), 1e-11),
            "{n}"
        );
    }
    assert_eq!(hafnian(&CMatrix::zeros(0, 0)).unwrap(), ONE);
    assert_eq!(loop_hafnian(&CMatrix::zeros(0, 0)).unwrap(), ONE);
}

#[test]
fn closed_forms() {
    // All ones: (2n − 1)!! perfect matchings; with loops, the telephone
    // numbers (matchings of K_{2n} with loops = involutions of 2n points).
    let double_factorial = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0];
    let involutions = [1.0, 2.0, 10.0, 76.0, 764.0, 9496.0, 140152.0, 2390480.0];
    for k in 1..8 {
        let ones = CMatrix::from_element(2 * k, 2 * k, ONE);
        // Relative errors reach 1.3·10⁻¹² at 14 × 14 (cancellation in the
        // alternating sum).
        assert!(close(
            hafnian(&ones).unwrap(),
            C64::new(double_factorial[k], 0.0),
            1e-11
        ));
        assert!(close(
            loop_hafnian(&ones).unwrap(),
            C64::new(involutions[k], 0.0),
            1e-11
        ));
    }
    // A bipartite graph: haf([[0, B], [Bᵀ, 0]]) = perm(B).
    let mut rng = Rng::seed_from_u64(7);
    for k in [2, 4, 6, 8] {
        let b = CMatrix::from_fn(k, k, |_, _| C64::new(rng.normal(), rng.normal()));
        let mut a = CMatrix::zeros(2 * k, 2 * k);
        a.view_mut((0, k), (k, k)).copy_from(&b);
        a.view_mut((k, 0), (k, k)).copy_from(&b.transpose());
        assert!(
            close(hafnian(&a).unwrap(), permanent(&b).unwrap(), 1e-11),
            "{k}"
        );
    }
    // A diagonal matrix: only loops.
    let d = CMatrix::from_diagonal(&nalgebra::DVector::from_fn(6, |i, _| {
        C64::new(i as f64 + 1.0, 0.5)
    }));
    let product = (0..6).fold(ONE, |p, i| p * d[(i, i)]);
    assert!(close(loop_hafnian(&d).unwrap(), product, 1e-12));
    assert_eq!(hafnian(&d).unwrap(), ZERO);
}

#[test]
fn hafnians_count_perfect_matchings_of_graphs() {
    // The 3-cube has 9 perfect matchings, the Petersen graph 6, K_{4,4} 24.
    for (g, count) in [
        (crate::graph::generators::hypercube(3), 9.0),
        (crate::graph::generators::petersen(), 6.0),
        (crate::graph::generators::complete_bipartite(4, 4), 24.0),
    ] {
        let a = g.adjacency_matrix().map(|v| C64::new(v, 0.0));
        assert!(close(hafnian(&a).unwrap(), C64::new(count, 0.0), 1e-12));
    }
}

/// Definition of the torontonian, with determinants of submatrices.
fn torontonian_by_definition(o: &CMatrix) -> C64 {
    let n = o.nrows() / 2;
    let mut s = ZERO;
    for z in 0u64..(1 << n) {
        let modes: Vec<usize> = (0..n).filter(|&i| (z >> i) & 1 == 1).collect();
        let idx: Vec<usize> = modes
            .iter()
            .copied()
            .chain(modes.iter().map(|&i| i + n))
            .collect();
        let k = idx.len();
        let m = CMatrix::identity(k, k) - CMatrix::from_fn(k, k, |r, c| o[(idx[r], idx[c])]);
        let det = if k == 0 { ONE } else { m.determinant() };
        let sign = if (n - modes.len()) % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        s += det.sqrt().inv() * sign;
    }
    s
}

#[test]
fn torontonians_follow_their_definition_and_limits() {
    // Positive semidefinite O of small norm, as for physical states.
    let mut rng = Rng::seed_from_u64(3);
    for n in [1, 2, 3, 4] {
        let a = CMatrix::from_fn(2 * n, 2 * n, |_, _| {
            C64::new(0.1 * rng.normal(), 0.1 * rng.normal())
        });
        let o = &a * a.adjoint() * C64::new(0.5, 0.0);
        assert!(
            close(
                torontonian(&o).unwrap(),
                torontonian_by_definition(&o),
                1e-12
            ),
            "{n}"
        );
    }
    // O = 0 (vacuum): every subset has det 1, so Tor = Σ (−1)^{N−|Z|} = 0
    // for N ≥ 1 (no clicks are certain, all clicks impossible).
    assert!(torontonian(&CMatrix::zeros(4, 4)).unwrap().norm() < 1e-15);
    assert_eq!(torontonian(&CMatrix::zeros(0, 0)).unwrap(), ONE);
    assert!(torontonian(&CMatrix::zeros(3, 3)).is_err());
}

#[test]
fn invalid_inputs_are_rejected_and_results_are_deterministic() {
    let mut a = random_symmetric(4, 1);
    a[(0, 1)] += C64::new(1.0, 0.0);
    assert!(hafnian(&a).is_err());
    assert!(loop_hafnian(&CMatrix::zeros(2, 3)).is_err());
    assert!(hafnian(&CMatrix::zeros(MAX_HAFNIAN_DIM + 2, MAX_HAFNIAN_DIM + 2)).is_err());
    let mut nan = random_symmetric(4, 2);
    nan[(1, 1)] = C64::new(f64::NAN, 0.0);
    assert!(loop_hafnian(&nan).is_err());
    let a = random_symmetric(20, 9);
    let h1 = hafnian(&a).unwrap();
    let h2 = hafnian(&a).unwrap();
    assert_eq!(h1, h2);
}

#[test]
fn repeated_hafnians_match_the_expanded_matrix() {
    let mut rng = Rng::seed_from_u64(31);
    for rpt in [
        vec![2],
        vec![1, 1],
        vec![2, 1, 1],
        vec![3, 2],
        vec![2, 2, 2],
        vec![4, 0, 2],
        vec![1, 3, 1],
        vec![0, 0],
    ] {
        let m = rpt.len();
        let a = random_symmetric(m, rng.next_u64());
        let mu: Vec<C64> = (0..m)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let idx: Vec<usize> = (0..m)
            .flat_map(|i| std::iter::repeat_n(i, rpt[i]))
            .collect();
        let k = idx.len();
        let expanded = CMatrix::from_fn(k, k, |r, c| a[(idx[r], idx[c])]);
        let mut with_loops = expanded.clone();
        for r in 0..k {
            with_loops[(r, r)] = mu[idx[r]];
        }
        assert!(
            close(
                hafnian_repeated(&a, &rpt).unwrap(),
                brute(&expanded, false),
                1e-12
            ),
            "{rpt:?}"
        );
        assert!(
            close(
                loop_hafnian_repeated(&a, &mu, &rpt).unwrap(),
                brute(&with_loops, true),
                1e-12
            ),
            "{rpt:?}"
        );
    }
    let a = random_symmetric(2, 1);
    assert!(hafnian_repeated(&a, &[1]).is_err());
    assert!(loop_hafnian_repeated(&a, &[ONE], &[1, 1]).is_err());
    assert!(hafnian_repeated(&a, &[usize::MAX / 2, 3]).is_err());
}

#[test]
fn repeated_hafnians_neither_overflow_nor_hide_their_limits() {
    let fact = |n: usize| (1..=n).map(|i| i as f64).product::<f64>();
    // haf of J_{2k} with off-diagonal blocks c: c^k k!; c = 10⁶, k = 20
    // overflowed the powers of Kan's formula.
    let c = 1e6;
    let a = CMatrix::from_row_slice(2, 2, &[ZERO, C64::new(c, 0.0), C64::new(c, 0.0), ZERO]);
    let want = c.powi(20) * fact(20);
    let got = hafnian_repeated(&a, &[20, 20]).unwrap();
    assert!((got.re - want).abs() < 1e-10 * want, "{got} vs {want}");
    // [[1]] repeated 40 times: 39!! (the measured error is about 10⁻¹¹).
    let one = CMatrix::from_element(1, 1, C64::new(1.0, 0.0));
    let want: f64 = (1..40).step_by(2).map(|i| i as f64).product();
    let got = hafnian_repeated(&one, &[40]).unwrap();
    assert!((got.re - want).abs() < 1e-9 * want, "{got} vs {want}");
    // At most MAX_HAFNIAN_DIM rows.
    assert!(hafnian_repeated(&one, &[42]).is_err());
    // A wrong diagonal length is reported as such.
    match loop_hafnian_repeated(&one, &[ONE, ONE], &[2]) {
        Err(QsoError::DimensionMismatch { expected, actual }) => {
            assert_eq!((expected, actual), (1, 2))
        }
        other => panic!("{other:?}"),
    }
    // Symmetry is checked relative to the entries.
    let tiny = CMatrix::from_row_slice(2, 2, &[ZERO, C64::new(1e-13, 0.0), ZERO, ZERO]);
    assert!(hafnian(&tiny).is_err());
}
