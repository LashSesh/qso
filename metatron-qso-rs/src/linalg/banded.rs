//! Banded LU factorisation after a reverse Cuthill–McKee ordering
//! (crate-internal).
//!
//! A sparse matrix whose pattern is a graph of small bandwidth after
//! reordering (the orbital generators of the symmetric open walks, for
//! instance) is factorised exactly in `O(n·k_l·(k_l + k_u))` operations and
//! `n·(2k_l + k_u + 1)` entries instead of `O(n³)` and `n²`:
//!
//! * [`reverse_cuthill_mckee`] orders the nodes so that neighbours get
//!   nearby positions (Cuthill and McKee, Proc. 24th ACM National Conference
//!   (1969); reversed as George proposed), deterministically: every tie goes
//!   to the smaller index.
//! * [`BandedLu`] is LAPACK's `zgbtf2` (unblocked LU with partial pivoting in
//!   band storage, the factor `U` widened by `k_l` rows for the fill of the
//!   row interchanges) with solves for `A` and for `A†`. Its limits are
//!   checked before the band storage is allocated.
//!
//! Everything is sequential, so the results do not depend on the thread
//! count.

use num_complex::Complex64 as C64;

use crate::error::{QsoError, Result};

const ZERO: C64 = C64::new(0.0, 0.0);

/// The number of levels of a breadth-first search from `start` and the
/// nodes of its last level (`mark` is all false on entry and on return).
fn levels(adjacency: &[Vec<u32>], start: usize, mark: &mut [bool]) -> (usize, Vec<u32>) {
    mark[start] = true;
    let mut touched = vec![start as u32];
    let mut level = vec![start as u32];
    let mut depth = 1;
    loop {
        let mut next = Vec::new();
        for &u in &level {
            for &w in &adjacency[u as usize] {
                if !mark[w as usize] {
                    mark[w as usize] = true;
                    touched.push(w);
                    next.push(w);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        depth += 1;
        level = next;
    }
    for u in touched {
        mark[u as usize] = false;
    }
    (depth, level)
}

/// The reverse Cuthill–McKee order of the graph with symmetric adjacency
/// lists `adjacency` (node `u`'s neighbours, without `u`): `order[k]` is the
/// old index of the node at new position `k`.
///
/// Each connected component, taken in the order of its smallest index,
/// starts at a pseudo-peripheral node: from that smallest index, the
/// breadth-first search is repeated from the node of smallest degree in its
/// last level (ties by index) as long as the number of levels grows. The
/// Cuthill–McKee order then visits the unplaced neighbours of each placed
/// node by increasing (degree, index), and each component's order is
/// reversed.
pub(crate) fn reverse_cuthill_mckee(adjacency: &[Vec<u32>]) -> Vec<u32> {
    let n = adjacency.len();
    let degree = |u: u32| adjacency[u as usize].len();
    let mut placed = vec![false; n];
    let mut mark = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    for root in 0..n {
        if placed[root] {
            continue;
        }
        let mut start = root;
        let (mut depth, mut last) = levels(adjacency, start, &mut mark);
        loop {
            let candidate = last
                .iter()
                .copied()
                .min_by_key(|&u| (degree(u), u))
                .expect("a level is never empty") as usize;
            let (d, l) = levels(adjacency, candidate, &mut mark);
            if d <= depth {
                break;
            }
            (start, depth, last) = (candidate, d, l);
        }
        let first = order.len();
        order.push(start as u32);
        placed[start] = true;
        let mut head = first;
        while head < order.len() {
            let u = order[head] as usize;
            head += 1;
            let mut next: Vec<u32> = adjacency[u]
                .iter()
                .copied()
                .filter(|&w| !placed[w as usize])
                .collect();
            next.sort_unstable_by_key(|&w| (degree(w), w));
            for w in next {
                // A repeated neighbour is placed once.
                if !placed[w as usize] {
                    placed[w as usize] = true;
                    order.push(w);
                }
            }
        }
        order[first..].reverse();
    }
    order
}

/// The LU factorisation with partial pivoting of a band matrix after a
/// symmetric reordering, `P_o A P_oᵀ = P₀L₀P₁L₁ ⋯ U` in LAPACK's `zgbtf2`
/// layout (`P_o` the reordering, `P_j` the interchange of rows `j` and
/// `ipiv[j]`, `L_j` unit lower with the multipliers of column `j`).
#[derive(Clone, Debug)]
pub(crate) struct BandedLu {
    n: usize,
    /// Sub- and superdiagonals of the reordered matrix.
    kl: usize,
    ku: usize,
    /// Rows of the band storage, `2k_l + k_u + 1`: `U` has `k_l + k_u`
    /// superdiagonals.
    ld: usize,
    /// Entry `(i, j)` of the reordered matrix (and later of `L` and `U`) at
    /// `j·ld + k_l + k_u + i − j`.
    ab: Vec<C64>,
    /// Row `j` was interchanged with row `ipiv[j] ≥ j` at step `j`.
    ipiv: Vec<u32>,
    /// `order[k]`: the original index at position `k`.
    order: Vec<u32>,
}

impl BandedLu {
    /// Factorise the `n × n` matrix whose row `r` (original indices) `row(r,
    /// entries)` pushes as `(column, value)` pairs (repeated columns add up;
    /// `entries` is cleared before every call, and every row is asked for
    /// twice: once for the bandwidth, once for the values), reordered by
    /// `order` (a permutation of `0..n`, such as [`reverse_cuthill_mckee`]'s).
    ///
    /// Before the band storage is allocated it refuses more than
    /// `max_entries` band entries (`TooManyQubits`) and more than `max_work`
    /// estimated operations, `8·n·k_l·(k_l + k_u + 1)` (an invalid
    /// argument); then it refuses an exactly singular matrix (an invalid
    /// argument). The errors are phrased for the one caller, the
    /// preconditioner of the orbital transport solves.
    pub(crate) fn new(
        n: usize,
        order: Vec<u32>,
        row: impl Fn(usize, &mut Vec<(u32, C64)>),
        max_entries: usize,
        max_work: f64,
    ) -> Result<Self> {
        if order.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: order.len(),
            });
        }
        let mut rank = vec![u32::MAX; n];
        for (k, &o) in order.iter().enumerate() {
            debug_assert_eq!(rank[o as usize], u32::MAX, "{o} repeats in the order");
            rank[o as usize] = k as u32;
        }
        let mut entries = Vec::new();
        let (mut kl, mut ku) = (0usize, 0usize);
        for r in 0..n {
            entries.clear();
            row(r, &mut entries);
            let i = rank[r] as usize;
            for &(c, _) in &entries {
                let j = rank[c as usize] as usize;
                if i > j {
                    kl = kl.max(i - j);
                } else {
                    ku = ku.max(j - i);
                }
            }
        }
        let ld = kl.saturating_mul(2).saturating_add(ku).saturating_add(1);
        let size = n
            .checked_mul(ld)
            .filter(|&size| size <= max_entries)
            .ok_or(QsoError::TooManyQubits {
                what: "transport preconditioner (band entries)",
                max: max_entries,
                requested: n.saturating_mul(ld),
            })?;
        let work = 8.0 * n as f64 * kl as f64 * (kl + ku + 1) as f64;
        if work > max_work {
            return Err(QsoError::invalid(
                "symmetries",
                format!(
                    "the transport preconditioner needs about {work:.1e} operations (limit \
                     {max_work:.1e}); use more symmetries"
                ),
            ));
        }
        let kv = kl + ku;
        let mut ab = vec![ZERO; size];
        for r in 0..n {
            entries.clear();
            row(r, &mut entries);
            let i = rank[r] as usize;
            for &(c, v) in &entries {
                let j = rank[c as usize] as usize;
                ab[j * ld + kv + i - j] += v;
            }
        }
        let mut ipiv = vec![0u32; n];
        // The last column that the interchanges so far have reached.
        let mut ju = 0;
        for (j, pivot) in ipiv.iter_mut().enumerate() {
            let km = kl.min(n - 1 - j);
            let col = j * ld + kv;
            // The pivot: the largest modulus in rows j..=j + km (the first
            // of equals).
            let mut jp = 0;
            let mut best = ab[col].norm();
            for i in 1..=km {
                let a = ab[col + i].norm();
                if a > best {
                    best = a;
                    jp = i;
                }
            }
            *pivot = (j + jp) as u32;
            if ab[col + jp] == ZERO {
                return Err(QsoError::invalid(
                    "symmetries",
                    "the transport preconditioner is singular",
                ));
            }
            ju = ju.max((j + ku + jp).min(n - 1));
            if jp != 0 {
                for c in j..=ju {
                    // Rows j and j + jp of column c.
                    let at = c * ld + kv + j - c;
                    ab.swap(at, at + jp);
                }
            }
            if km > 0 {
                let inverse = C64::new(1.0, 0.0) / ab[col];
                for l in &mut ab[col + 1..=col + km] {
                    *l *= inverse;
                }
                // The rank-1 update of rows j + 1..=j + km, columns
                // j + 1..=ju.
                for c in j + 1..=ju {
                    let at = c * ld + kv + j - c;
                    let (lower, upper) = ab.split_at_mut(at + 1);
                    let f = lower[at];
                    if f != ZERO {
                        let multipliers = &lower[col + 1..=col + km];
                        for (a, l) in upper[..km].iter_mut().zip(multipliers) {
                            *a -= l * f;
                        }
                    }
                }
            }
        }
        Ok(BandedLu {
            n,
            kl,
            ku,
            ld,
            ab,
            ipiv,
            order,
        })
    }

    /// `(k_l, k_u)`: sub- and superdiagonals of the reordered matrix.
    pub(crate) fn bandwidth(&self) -> (usize, usize) {
        (self.kl, self.ku)
    }

    /// Complex entries of the band storage, `n·(2k_l + k_u + 1)`.
    pub(crate) fn entries(&self) -> usize {
        self.ab.len()
    }

    /// `x ← A⁻¹x`: the interchanges and `L` forwards, `U` backwards.
    pub(crate) fn solve(&self, x: &mut [C64]) {
        let (n, kv) = (self.n, self.kl + self.ku);
        assert_eq!(x.len(), n, "a vector of the matrix's dimension");
        let mut b: Vec<C64> = self.order.iter().map(|&o| x[o as usize]).collect();
        for j in 0..n {
            let p = self.ipiv[j] as usize;
            if p != j {
                b.swap(j, p);
            }
            let km = self.kl.min(n - 1 - j);
            let col = j * self.ld + kv;
            let bj = b[j];
            for (bi, l) in b[j + 1..=j + km]
                .iter_mut()
                .zip(&self.ab[col + 1..=col + km])
            {
                *bi -= l * bj;
            }
        }
        for j in (0..n).rev() {
            let col = j * self.ld + kv;
            b[j] /= self.ab[col];
            let bj = b[j];
            let up = kv.min(j);
            // U(j − i, j) at col − i for i = 1..=up.
            for (bi, u) in b[j - up..j].iter_mut().zip(&self.ab[col - up..col]) {
                *bi -= u * bj;
            }
        }
        for (k, &o) in self.order.iter().enumerate() {
            x[o as usize] = b[k];
        }
    }

    /// `x ← A^{−†}x`: `U†` forwards, then `L†` and the interchanges
    /// backwards.
    pub(crate) fn solve_adjoint(&self, x: &mut [C64]) {
        let (n, kv) = (self.n, self.kl + self.ku);
        assert_eq!(x.len(), n, "a vector of the matrix's dimension");
        let mut b: Vec<C64> = self.order.iter().map(|&o| x[o as usize]).collect();
        for j in 0..n {
            let col = j * self.ld + kv;
            let up = kv.min(j);
            let mut acc = b[j];
            for (bi, u) in b[j - up..j].iter().zip(&self.ab[col - up..col]) {
                acc -= u.conj() * bi;
            }
            b[j] = acc / self.ab[col].conj();
        }
        for j in (0..n).rev() {
            let km = self.kl.min(n - 1 - j);
            let col = j * self.ld + kv;
            let mut acc = b[j];
            for (bi, l) in b[j + 1..=j + km].iter().zip(&self.ab[col + 1..=col + km]) {
                acc -= l.conj() * bi;
            }
            b[j] = acc;
            let p = self.ipiv[j] as usize;
            if p != j {
                b.swap(j, p);
            }
        }
        for (k, &o) in self.order.iter().enumerate() {
            x[o as usize] = b[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CMatrix;
    use crate::rng::Rng;
    use nalgebra::DVector;

    /// Largest `|position(u) − position(v)|` over the edges.
    fn bandwidth(adjacency: &[Vec<u32>], order: &[u32]) -> usize {
        let mut position = vec![0usize; order.len()];
        for (k, &o) in order.iter().enumerate() {
            position[o as usize] = k;
        }
        let mut width = 0;
        for (u, nb) in adjacency.iter().enumerate() {
            for &v in nb {
                width = width.max(position[u].abs_diff(position[v as usize]));
            }
        }
        width
    }

    fn is_permutation(order: &[u32]) -> bool {
        let mut sorted = order.to_vec();
        sorted.sort_unstable();
        sorted.iter().enumerate().all(|(k, &o)| o as usize == k)
    }

    /// The symmetrised pattern of `rows` without the diagonal.
    fn pattern(rows: &[Vec<(u32, C64)>]) -> Vec<Vec<u32>> {
        let mut adjacency = vec![Vec::new(); rows.len()];
        for (r, row) in rows.iter().enumerate() {
            for &(c, _) in row {
                if c as usize != r {
                    adjacency[r].push(c);
                    adjacency[c as usize].push(r as u32);
                }
            }
        }
        for a in &mut adjacency {
            a.sort_unstable();
            a.dedup();
        }
        adjacency
    }

    fn dense(rows: &[Vec<(u32, C64)>]) -> CMatrix {
        let n = rows.len();
        let mut a = CMatrix::zeros(n, n);
        for (r, row) in rows.iter().enumerate() {
            for &(c, v) in row {
                a[(r, c as usize)] += v;
            }
        }
        a
    }

    fn factorise(rows: &[Vec<(u32, C64)>], order: Vec<u32>) -> BandedLu {
        BandedLu::new(
            rows.len(),
            order,
            |r, e| e.extend(&rows[r]),
            usize::MAX,
            f64::INFINITY,
        )
        .unwrap()
    }

    /// `solve` and `solve_adjoint` against nalgebra's dense LU.
    fn check_solves(lu: &BandedLu, a: &CMatrix, seed: u64) {
        let n = a.nrows();
        let mut rng = Rng::seed_from_u64(seed);
        let b: Vec<C64> = (0..n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let dense = a.clone().lu();
        let want = dense.solve(&DVector::from_vec(b.clone())).unwrap();
        let mut x = b.clone();
        lu.solve(&mut x);
        let e = (DVector::from_vec(x) - &want).norm() / want.norm();
        assert!(e <= 1e-12, "solve: {e:e}");
        let want = a
            .adjoint()
            .lu()
            .solve(&DVector::from_vec(b.clone()))
            .unwrap();
        let mut y = b;
        lu.solve_adjoint(&mut y);
        let e = (DVector::from_vec(y) - &want).norm() / want.norm();
        assert!(e <= 1e-12, "solve_adjoint: {e:e}");
    }

    /// A random complex band matrix (`kl = ku = w`) whose diagonal does not
    /// dominate, so that rows are interchanged.
    fn random_band(n: usize, w: usize, rng: &mut Rng) -> Vec<Vec<(u32, C64)>> {
        (0..n)
            .map(|i| {
                (i.saturating_sub(w)..(i + w + 1).min(n))
                    .map(|j| (j as u32, C64::new(rng.normal(), rng.normal())))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn banded_solves_match_a_dense_lu() {
        let mut rng = Rng::seed_from_u64(21);
        for n in [50, 200] {
            let rows = random_band(n, 7, &mut rng);
            let a = dense(&rows);
            let natural = factorise(&rows, (0..n as u32).collect());
            assert_eq!(natural.bandwidth(), (7, 7));
            assert_eq!(natural.entries(), n * (3 * 7 + 1));
            assert!(
                natural
                    .ipiv
                    .iter()
                    .enumerate()
                    .any(|(j, &p)| p as usize != j)
            );
            check_solves(&natural, &a, 1);
            // The same matrix with its nodes relabelled at random: the
            // reverse Cuthill–McKee order finds a band again.
            let mut relabel: Vec<u32> = (0..n as u32).collect();
            rng.shuffle(&mut relabel);
            let mut scrambled = vec![Vec::new(); n];
            for (i, row) in rows.iter().enumerate() {
                scrambled[relabel[i] as usize] =
                    row.iter().map(|&(j, v)| (relabel[j as usize], v)).collect();
            }
            let order = reverse_cuthill_mckee(&pattern(&scrambled));
            assert!(is_permutation(&order));
            let lu = factorise(&scrambled, order);
            let (kl, ku) = lu.bandwidth();
            assert!(kl == ku && kl <= 21, "{kl} {ku}");
            check_solves(&lu, &dense(&scrambled), 2);
        }
    }

    /// The orbitals `(n₀₁, n₁₀, n₁₁)` of the hypercube `Q_d` under the
    /// permutations of the coordinates, in the order of
    /// `walks::symmetric::HypercubeOpenWalk` (lexicographic), and its
    /// generator: a trap of rate `kappa` at the antipode, loss and site
    /// dephasing at every node, site energies `ε·weight`.
    fn hypercube(
        d: usize,
        kappa: f64,
        loss: f64,
        dephasing: f64,
        eps: f64,
    ) -> Vec<Vec<(u32, C64)>> {
        let count = |d: usize| (d + 3) * (d + 2) * (d + 1) / 6;
        let index = |n01: usize, n10: usize, n11: usize| {
            let m = d - n01;
            count(d) - count(m) + n10 * (m + 1) - n10 * n10.saturating_sub(1) / 2 + n11
        };
        let i = C64::new(0.0, 1.0);
        let k = |w: usize| (if w == d { kappa } else { 0.0 }) + 0.5 * loss;
        let mut rows = Vec::with_capacity(count(d));
        for n01 in 0..=d {
            for n10 in 0..=d - n01 {
                for n11 in 0..=d - n01 - n10 {
                    let n = [d - n01 - n10 - n11, n01, n10, n11];
                    let (wa, wb) = (n10 + n11, n01 + n11);
                    let decay = k(wa) + k(wb) + if n01 + n10 > 0 { dephasing } else { 0.0 };
                    let mut row = vec![(
                        index(n01, n10, n11) as u32,
                        C64::new(-decay, -eps * (wa as f64 - wb as f64)),
                    )];
                    for (flip, sign) in [([2, 3, 0, 1], -i), ([1, 0, 3, 2], i)] {
                        for c in 0..4 {
                            if n[c] > 0 {
                                let mut m = n;
                                m[c] -= 1;
                                m[flip[c]] += 1;
                                let w = (n[c] as f64 * (n[flip[c]] + 1) as f64).sqrt();
                                row.push((index(m[1], m[2], m[3]) as u32, -sign * w));
                            }
                        }
                    }
                    rows.push(row);
                }
            }
        }
        rows
    }

    #[test]
    fn hypercube_orbitals_have_the_measured_bandwidths() {
        for (d, orbitals, width) in [(10, 286, 41), (20, 1771, 131)] {
            let rows = hypercube(d, 1.0, 0.05, 0.7, 0.3);
            assert_eq!(rows.len(), orbitals);
            let adjacency = pattern(&rows);
            let order = reverse_cuthill_mckee(&adjacency);
            assert!(is_permutation(&order));
            assert_eq!(bandwidth(&adjacency, &order), width, "d = {d}");
            if d == 10 {
                let lu = factorise(&rows, order);
                assert_eq!(lu.bandwidth(), (width, width));
                assert_eq!(lu.entries(), orbitals * (3 * width + 1));
            }
        }
        // The generator less a small shift, as the transport
        // preconditioner factorises it.
        let mut rows = hypercube(6, 1.0, 0.05, 0.7, 0.3);
        for (r, row) in rows.iter_mut().enumerate() {
            row.push((r as u32, C64::new(-0.01, 0.0)));
        }
        let lu = factorise(&rows, reverse_cuthill_mckee(&pattern(&rows)));
        check_solves(&lu, &dense(&rows), 3);
    }

    #[test]
    fn orders_are_permutations_of_every_component() {
        // A path relabelled at random, two isolated nodes and a triangle
        // with a repeated neighbour.
        let n = 40;
        let mut rng = Rng::seed_from_u64(4);
        let mut label: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut label);
        let mut adjacency = vec![Vec::new(); n + 5];
        for k in 0..n - 1 {
            let (u, v) = (label[k], label[k + 1]);
            adjacency[u as usize].push(v);
            adjacency[v as usize].push(u);
        }
        for (u, v) in [
            (n + 2, n + 3),
            (n + 3, n + 4),
            (n + 4, n + 2),
            (n + 2, n + 3),
        ] {
            adjacency[u].push(v as u32);
            adjacency[v].push(u as u32);
        }
        let order = reverse_cuthill_mckee(&adjacency);
        assert!(is_permutation(&order));
        assert_eq!(bandwidth(&adjacency, &order), 2);
        // The path alone has bandwidth 1 and starts at an end.
        let path = adjacency[..n].to_vec();
        let order = reverse_cuthill_mckee(&path);
        assert_eq!(bandwidth(&path, &order), 1);
        assert_eq!(path[order[0] as usize].len(), 1);
        assert!(reverse_cuthill_mckee(&[]).is_empty());
    }

    #[test]
    fn limits_are_checked_before_the_band_is_allocated() {
        // Two million nodes, each coupled to the nodes 100 000 positions
        // away: 6·10¹¹ band entries (9.6 TB) are refused before any is
        // allocated.
        let n = 2_000_000;
        let far = |r: usize, e: &mut Vec<(u32, C64)>| {
            e.push((r as u32, C64::new(1.0, 0.0)));
            for c in [r.checked_sub(100_000), Some(r + 100_000).filter(|&c| c < n)] {
                e.extend(c.map(|c| (c as u32, C64::new(0.5, 0.0))));
            }
        };
        let order: Vec<u32> = (0..n as u32).collect();
        match BandedLu::new(n, order, far, 1 << 27, f64::INFINITY) {
            Err(QsoError::TooManyQubits { max, requested, .. }) => {
                assert_eq!(max, 1 << 27);
                assert!(requested > 1 << 27);
            }
            other => panic!("{:?}", other.map(|lu| lu.entries())),
        }
        // The work limit.
        let rows = random_band(100, 3, &mut Rng::seed_from_u64(5));
        let work = 8.0 * 100.0 * 3.0 * 7.0;
        let new = |max_work: f64| {
            BandedLu::new(
                100,
                (0..100).collect(),
                |r, e| e.extend(&rows[r]),
                usize::MAX,
                max_work,
            )
        };
        assert!(new(work).is_ok());
        assert!(matches!(
            new(work * 0.99),
            Err(QsoError::InvalidArgument {
                name: "symmetries",
                ..
            })
        ));
        // An exactly singular matrix: column 5 is zero, and stays zero
        // through the interchanges and updates.
        let mut rows = random_band(20, 2, &mut Rng::seed_from_u64(6));
        for row in &mut rows {
            row.retain(|&(c, _)| c != 5);
        }
        let singular = BandedLu::new(
            20,
            (0..20).collect(),
            |r, e| e.extend(&rows[r]),
            usize::MAX,
            f64::INFINITY,
        );
        assert!(matches!(
            singular,
            Err(QsoError::InvalidArgument {
                name: "symmetries",
                ..
            })
        ));
    }
}
