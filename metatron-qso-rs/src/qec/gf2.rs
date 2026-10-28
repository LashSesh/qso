//! Linear algebra over GF(2) on bit-packed vectors: a fully reduced
//! basis that grows one vector at a time and can record which inserted
//! vectors make up each of its members.

/// Words of a vector of `len` bits.
pub(crate) fn words(len: usize) -> usize {
    len.div_ceil(64)
}

pub(crate) fn get(v: &[u64], i: usize) -> bool {
    v[i / 64] >> (i % 64) & 1 == 1
}

pub(crate) fn flip(v: &mut [u64], i: usize) {
    v[i / 64] ^= 1 << (i % 64);
}

pub(crate) fn xor_into(target: &mut [u64], source: &[u64]) {
    for (t, s) in target.iter_mut().zip(source) {
        *t ^= s;
    }
}

/// Parity of the overlap of two vectors.
pub(crate) fn dot(a: &[u64], b: &[u64]) -> bool {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x & y).count_ones())
        .sum::<u32>()
        % 2
        == 1
}

/// Indices of the set bits, ascending.
pub(crate) fn ones(v: &[u64]) -> impl Iterator<Item = usize> + '_ {
    v.iter().enumerate().flat_map(|(w, &word)| {
        let mut word = word;
        std::iter::from_fn(move || {
            if word == 0 {
                return None;
            }
            let b = word.trailing_zeros() as usize;
            word &= word - 1;
            Some(64 * w + b)
        })
    })
}

/// The vector of `len` bits with the given ones (listed twice cancels).
pub(crate) fn from_ones(len: usize, ones: &[usize]) -> Vec<u64> {
    let mut v = vec![0; words(len)];
    for &i in ones {
        flip(&mut v, i);
    }
    v
}

/// A basis in reduced echelon form: every member has a pivot bit that is
/// zero in every other member. With `track`, member `k` also records the
/// inserted vectors (by the order they were accepted) whose sum it is.
#[derive(Clone, Debug)]
pub(crate) struct Basis {
    words: usize,
    /// Members, `words` words each.
    vectors: Vec<u64>,
    pivots: Vec<usize>,
    /// Member whose pivot is bit `i`, or `usize::MAX`.
    pivot_of: Vec<usize>,
    /// Words of a combination; 0 without tracking.
    combo_words: usize,
    combos: Vec<u64>,
    scratch: Vec<usize>,
}

impl Basis {
    /// An empty basis of vectors of `len` bits; `track` records
    /// combinations of up to `max_rank` accepted vectors.
    pub(crate) fn new(len: usize, track: bool, max_rank: usize) -> Self {
        Basis {
            words: words(len),
            vectors: Vec::new(),
            pivots: Vec::new(),
            pivot_of: vec![usize::MAX; len],
            combo_words: if track { words(max_rank.max(1)) } else { 0 },
            combos: Vec::new(),
            scratch: Vec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        for &p in &self.pivots {
            self.pivot_of[p] = usize::MAX;
        }
        self.vectors.clear();
        self.pivots.clear();
        self.combos.clear();
    }

    /// The pivot (lowest bit) of member `k`.
    pub(crate) fn pivot(&self, k: usize) -> usize {
        self.pivots[k]
    }

    pub(crate) fn rank(&self) -> usize {
        self.pivots.len()
    }

    pub(crate) fn combo_words(&self) -> usize {
        self.combo_words
    }

    /// The member whose pivot is bit `i`, if any.
    pub(crate) fn pivot_member(&self, i: usize) -> Option<usize> {
        let k = self.pivot_of[i];
        (k != usize::MAX).then_some(k)
    }

    /// The recorded combination of member `k`.
    pub(crate) fn combo(&self, k: usize) -> &[u64] {
        &self.combos[k * self.combo_words..(k + 1) * self.combo_words]
    }

    /// Subtract the members whose pivots are set in `v`, adding their
    /// combinations to `combo` when given: afterwards `v` has no pivot bit
    /// set, and is zero exactly when it was in the span.
    pub(crate) fn reduce(&mut self, v: &mut [u64], mut combo: Option<&mut [u64]>) {
        // Subtracting a member clears its pivot and sets no other pivot,
        // so the members to subtract are those whose pivots `v` has now.
        let mut used = std::mem::take(&mut self.scratch);
        used.clear();
        used.extend(ones(v).filter_map(|i| self.pivot_member(i)));
        let w = self.words;
        for &k in &used {
            xor_into(v, &self.vectors[k * w..(k + 1) * w]);
            if let Some(c) = combo.as_deref_mut() {
                let cw = self.combo_words;
                xor_into(c, &self.combos[k * cw..(k + 1) * cw]);
            }
        }
        self.scratch = used;
    }

    /// Insert `v` (of `words` words); returns its member index if it was
    /// independent of the basis.
    pub(crate) fn insert(&mut self, v: &[u64]) -> Option<usize> {
        let k = self.rank();
        let mut v = v.to_vec();
        let mut combo = vec![0u64; self.combo_words];
        if self.combo_words > 0 {
            assert!(k < 64 * self.combo_words, "more members than tracked");
            flip(&mut combo, k);
        }
        self.reduce(&mut v, (self.combo_words > 0).then_some(&mut combo[..]));
        let p = ones(&v).next()?;
        // Clear the new pivot from the other members.
        let (w, cw) = (self.words, self.combo_words);
        for j in 0..k {
            if get(&self.vectors[j * w..(j + 1) * w], p) {
                xor_into(&mut self.vectors[j * w..(j + 1) * w], &v);
                xor_into(&mut self.combos[j * cw..(j + 1) * cw], &combo);
            }
        }
        self.vectors.extend_from_slice(&v);
        self.combos.extend_from_slice(&combo);
        self.pivots.push(p);
        self.pivot_of[p] = k;
        Some(k)
    }

    /// Whether `v` is in the span.
    #[cfg(test)]
    pub(crate) fn contains(&mut self, v: &[u64]) -> bool {
        let mut v = v.to_vec();
        self.reduce(&mut v, None);
        v.iter().all(|&x| x == 0)
    }
}

/// A basis of the null space `{x : H x = 0}` of the matrix with rows
/// `rows` (each of `len` bits).
pub(crate) fn kernel(rows: &[Vec<u64>], len: usize) -> Vec<Vec<u64>> {
    // Reduce the columns: column j of H, augmented by the unit vector e_j;
    // a dependent column gives a kernel vector from its combination.
    let m = rows.len();
    let mut basis = Basis::new(m, true, len);
    let mut chosen: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    for j in 0..len {
        let col: Vec<u64> = {
            let mut c = vec![0; words(m)];
            for (i, r) in rows.iter().enumerate() {
                if get(r, j) {
                    flip(&mut c, i);
                }
            }
            c
        };
        let mut v = col.clone();
        let mut combo = vec![0; basis.combo_words()];
        basis.reduce(&mut v, Some(&mut combo));
        if v.iter().all(|&x| x == 0) {
            // Column j is the sum of the chosen columns in `combo`.
            let mut x = vec![0; words(len)];
            flip(&mut x, j);
            for k in ones(&combo) {
                flip(&mut x, chosen[k]);
            }
            out.push(x);
        } else {
            basis.insert(&col);
            chosen.push(j);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn kernels_and_spans_of_random_matrices() {
        let mut rng = Rng::seed_from_u64(5);
        for _ in 0..200 {
            let m = 1 + rng.below(9) as usize;
            let n = 1 + rng.below(130) as usize;
            let rows: Vec<Vec<u64>> = (0..m)
                .map(|_| {
                    let ones: Vec<usize> = (0..n).filter(|_| rng.below(3) == 0).collect();
                    from_ones(n, &ones)
                })
                .collect();
            let mut b = Basis::new(n, true, m);
            for r in &rows {
                b.insert(r);
            }
            let k = kernel(&rows, n);
            // Rank–nullity, and every kernel vector is orthogonal to every
            // row.
            assert_eq!(b.rank() + k.len(), n);
            for x in &k {
                assert!(rows.iter().all(|r| !dot(r, x)));
            }
            // Members are the recorded sums of rows, and each row is in
            // the span.
            let accepted: Vec<&Vec<u64>> = {
                let mut c = Basis::new(n, false, 0);
                rows.iter().filter(|r| c.insert(r).is_some()).collect()
            };
            for k in 0..b.rank() {
                let mut s = vec![0; words(n)];
                for i in ones(b.combo(k)) {
                    xor_into(&mut s, accepted[i]);
                }
                assert_eq!(s, b.vectors[k * b.words..(k + 1) * b.words]);
            }
            assert!(rows.iter().all(|r| b.contains(r)));
        }
    }
}
