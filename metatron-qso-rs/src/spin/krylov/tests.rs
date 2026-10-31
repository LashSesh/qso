use super::*;
use crate::rng::Rng;

/// A sparse dissipative generator: random couplings `−iH` (about `k`
/// per row) plus a negative diagonal of rates.
fn generator(n: usize, k: usize, scale: f64, seed: u64) -> SparseBlock {
    let mut rng = Rng::seed_from_u64(seed);
    let mut triplets = Vec::new();
    for r in 0..n {
        triplets.push((
            r as u32,
            r as u32,
            C64::new(-rng.uniform(), scale * rng.normal()),
        ));
        for _ in 0..k {
            let c = rng.below(n as u64) as u32;
            let v = C64::new(rng.normal(), rng.normal()) * scale * 0.3;
            // Anti-Hermitian couplings, as −i[H, ·] gives.
            triplets.push((r as u32, c, v));
            triplets.push((c, r as u32, -v.conj()));
        }
    }
    SparseBlock::from_triplets(n, triplets)
}

fn random_vector(n: usize, seed: u64) -> Vec<C64> {
    let mut rng = Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| C64::new(rng.normal(), rng.normal()))
        .collect()
}

fn dense_expv(a: &SparseBlock, v: &[C64], t: f64) -> Vec<C64> {
    let p = expm(&(a.to_dense() * C64::new(t, 0.0)));
    (0..v.len())
        .map(|r| (0..v.len()).map(|c| p[(r, c)] * v[c]).sum())
        .collect()
}

fn distance(x: &[C64], y: &[C64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

#[test]
fn krylov_propagation_matches_the_dense_exponential() {
    // From one step to hundreds: ‖A‖t from 0.5 to about 3000.
    for (n, scale, t, seed) in [
        (40, 1.0, 0.5, 1u64),
        (150, 10.0, 2.0, 2),
        (300, 50.0, 3.0, 3),
        (120, 0.01, 50.0, 4),
    ] {
        let a = generator(n, 3, scale, seed);
        let v = random_vector(n, seed + 10);
        let norm = norm2(&v);
        let want = dense_expv(&a, &v, t);
        let got = expv(&a, &v, t, 1e-12 * norm, 0.0).unwrap();
        let err = distance(&got, &want) / norm;
        assert!(err < 1e-11, "n = {n}, ‖A‖t = {}: {err:e}", a.norm_inf() * t);
    }
}

#[test]
fn breakdown_and_trivial_cases_are_exact() {
    // An invariant subspace of dimension two: the Krylov space breaks down.
    let n = 50;
    let mut triplets = vec![(0u32, 1u32, C64::new(0.0, 3.0)), (1, 0, C64::new(0.0, 3.0))];
    triplets.push((0, 0, C64::new(-0.5, 0.0)));
    for r in 2..n as u32 {
        triplets.push((r, r, C64::new(-1.0, r as f64)));
        if r + 1 < n as u32 {
            triplets.push((r, r + 1, C64::new(0.7, 0.0)));
        }
    }
    let a = SparseBlock::from_triplets(n, triplets);
    let mut v = vec![C64::new(0.0, 0.0); n];
    v[0] = C64::new(1.0, 0.0);
    let want = dense_expv(&a, &v, 4.0);
    let got = expv(&a, &v, 4.0, 1e-12, 0.0).unwrap();
    assert!(distance(&got, &want) < 1e-13, "{}", distance(&got, &want));
    // Zero time, zero vector and a stationary vector.
    assert_eq!(expv(&a, &v, 0.0, 1e-12, 0.0).unwrap(), v);
    let zero = vec![C64::new(0.0, 0.0); n];
    assert_eq!(expv(&a, &zero, 1.0, 1e-12, 0.0).unwrap(), zero);
    let still = SparseBlock::from_triplets(3, vec![(1, 1, C64::new(-1.0, 0.0))]);
    let x = vec![C64::new(2.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, -1.0)];
    assert_eq!(expv(&still, &x, 1e3, 1e-12, 0.0).unwrap(), x);
}

#[test]
fn the_estimate_bounds_the_work() {
    // The steps the estimate assumes are at least those taken: the work
    // of a run, timed by its products, stays below the estimate.
    let a = generator(400, 4, 20.0, 7);
    let v = random_vector(400, 8);
    let t = 5.0;
    let norm = a.norm_inf();
    let work = krylov_work(a.n, a.nnz(), norm, t);
    let one = krylov_work(a.n, a.nnz(), norm, 0.0);
    // At least the steps of a run at the same tolerance as the relaxation.
    let start = std::time::Instant::now();
    expv(&a, &v, t, KRYLOV_TOLERANCE * norm2(&v), 0.0).unwrap();
    let elapsed = start.elapsed();
    assert!(work / one > norm * t / 10.0, "{}", work / one);
    assert!(elapsed.as_secs_f64() < 60.0);
}

#[test]
fn sparse_blocks_from_columns_and_accumulators() {
    let a = generator(30, 2, 1.0, 11);
    let dense = a.to_dense();
    let columns: Vec<Vec<(u32, C64)>> = (0..30)
        .map(|c| {
            (0..30)
                .filter(|&r| dense[(r, c)] != C64::new(0.0, 0.0))
                .map(|r| (r as u32, dense[(r, c)]))
                .collect()
        })
        .collect();
    let b = SparseBlock::from_columns(30, &columns);
    assert_eq!(b.to_dense(), dense);
    assert_eq!(b.nnz(), a.nnz());
    let row_sums: f64 = (0..30)
        .map(|r| (0..30).map(|c| dense[(r, c)].norm()).sum::<f64>())
        .fold(0.0, f64::max);
    assert!((a.norm_inf() - row_sums).abs() < 1e-12 * row_sums);
    let mut acc = Accumulator::new(10);
    acc.add(7, C64::new(1.0, 0.0));
    acc.add(2, C64::new(0.5, 0.0));
    acc.add(7, C64::new(-1.0, 0.0));
    assert_eq!(acc.take(), vec![(2, C64::new(0.5, 0.0))]);
    assert!(acc.take().is_empty());
    let x = random_vector(30, 3);
    let mut y = vec![C64::new(0.0, 0.0); 30];
    a.apply(&x, &mut y);
    for r in 0..30 {
        let want: C64 = (0..30).map(|c| dense[(r, c)] * x[c]).sum();
        assert!((y[r] - want).norm() < 1e-13);
    }
}

#[test]
fn orbits_reduce_an_invariant_generator() {
    // Elements i ↦ i + 10 (mod 30) and the reflection i ↦ 29 − i: a
    // generator symmetrised over the group they generate, and an
    // invariant vector, propagate on the orbits as in full.
    let n = 30;
    let shift: Vec<u32> = (0..n as u32).map(|i| (i + 10) % 30).collect();
    let mirror: Vec<u32> = (0..n as u32).map(|i| 29 - i).collect();
    let images = vec![shift.clone(), mirror.clone()];
    let orbits = Orbits::new(n, &images);
    // ⟨i ↦ i+10, i ↦ 29−i⟩ has order 6 and acts freely on 30 points.
    assert_eq!(orbits.len(), 5);
    assert!(orbits.sizes.iter().all(|&s| s == 6));
    let base = generator(n, 2, 2.0, 5).to_dense();
    // All six group elements: x ↦ ±x + 10k.
    let group: Vec<Vec<u32>> = (0..3)
        .flat_map(|k| {
            [
                (0..n as u32).map(move |i| (i + 10 * k) % 30).collect(),
                (0..n as u32).map(move |i| (29 - i + 10 * k) % 30).collect(),
            ]
        })
        .collect();
    let mut sym = CMatrix::zeros(n, n);
    for g in &group {
        for r in 0..n {
            for c in 0..n {
                sym[(g[r] as usize, g[c] as usize)] += base[(r, c)] / 6.0;
            }
        }
    }
    let mut triplets = Vec::new();
    for r in 0..n {
        for c in 0..n {
            triplets.push((r as u32, c as u32, sym[(r, c)]));
        }
    }
    let a = SparseBlock::from_triplets(n, triplets);
    assert!(Orbits::invariant(&a, &images, 1e-12));
    assert!(!Orbits::invariant(&generator(n, 2, 2.0, 5), &images, 1e-12));
    let y0 = random_vector(orbits.len(), 6);
    let x0: Vec<C64> = orbits.of.iter().map(|&o| y0[o as usize]).collect();
    assert!(orbits.asymmetry(&x0) < 1e-15);
    let full = dense_expv(&a, &x0, 0.7);
    let reduced = expv(&orbits.reduce(&a), &orbits.average(&x0), 0.7, 1e-13, 0.0).unwrap();
    let lifted: Vec<C64> = orbits.of.iter().map(|&o| reduced[o as usize]).collect();
    assert!(distance(&full, &lifted) < 1e-11 * norm2(&x0));
}
