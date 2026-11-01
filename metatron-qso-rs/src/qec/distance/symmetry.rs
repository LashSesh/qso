//! Symmetries of detector error models, so that the exhaustive search of
//! [`exact_circuit_distance`](super::exact_circuit_distance) starts from
//! one error of each orbit.
//!
//! A permutation `σ` of the detectors is a symmetry of a model when
//!
//! * it maps the detectors of every error onto the detectors of an error,
//!   which permutes the errors (`π`) and maps the sets of errors that fire
//!   no detector onto such sets, and
//! * `π` maps the undetectable sets that flip no observable onto such
//!   sets, so that logical errors map onto logical errors of the same
//!   weight. This holds exactly when every observable read through `π`
//!   (error `k` flips it when `π(k)` does) is a sum of rows of the
//!   detectors and observables over GF(2): those sums are exactly what
//!   vanishes on every set that fires nothing and flips nothing. It is
//!   checked by elimination.
//!
//! Symmetries come from the automorphisms of the graph that joins every
//! error to the detectors it flips (detectors and errors kept apart),
//! found by QSO's automorphism search; those that fail the second check
//! are dropped, and the rest generate the group the search uses.

use std::collections::HashMap;
use std::time::Instant;

use super::Error;
use crate::graph::Graph;
use crate::graph::automorphism::coloured_group;
use crate::qec::gf2;

/// Most detectors for which symmetries are checked (the elimination costs
/// about `n² · m / 64` word operations for `n` detectors and `m` errors).
pub const MAX_SYMMETRY_DETECTORS: usize = 1 << 12;

/// Checks of detector permutations against one model.
pub(super) struct Checker<'a> {
    n: usize,
    errors: &'a [Error],
    /// The first error with each set of detectors, and how many have it.
    by_set: HashMap<&'a [usize], (u32, usize)>,
    twins: bool,
    /// The rows of the detectors and observables, reduced (on first use).
    rows: Option<gf2::Basis>,
}

impl<'a> Checker<'a> {
    pub(super) fn new(n: usize, errors: &'a [Error]) -> Self {
        let mut by_set: HashMap<&[usize], (u32, usize)> = HashMap::new();
        for (k, e) in errors.iter().enumerate() {
            by_set
                .entry(e.detectors.as_slice())
                .or_insert((k as u32, 0))
                .1 += 1;
        }
        Checker {
            n,
            twins: by_set.len() < errors.len(),
            errors,
            by_set,
            rows: None,
        }
    }

    /// Whether two of the distinct errors flip the same detectors (and so
    /// different observables: together a logical error of weight two).
    pub(super) fn twins(&self) -> bool {
        self.twins
    }

    /// The permutation of the errors that `sigma` (detector `d ↦ sigma[d]`)
    /// induces, or why it is not a symmetry. With twins only the detectors
    /// are checked, and no permutation is returned.
    pub(super) fn check(&mut self, sigma: &[usize]) -> Result<Option<Vec<u32>>, String> {
        let n = self.n;
        if sigma.len() != n {
            return Err(format!("has {} entries for {n} detectors", sigma.len()));
        }
        let mut hit = vec![false; n];
        for &d in sigma {
            if d >= n || std::mem::replace(&mut hit[d], true) {
                return Err(format!("is not a permutation of the {n} detectors"));
            }
        }
        let mut pi = Vec::with_capacity(self.errors.len());
        let mut image = Vec::new();
        for e in self.errors {
            image.clear();
            image.extend(e.detectors.iter().map(|&d| sigma[d]));
            image.sort_unstable();
            let count = self.by_set[e.detectors.as_slice()].1;
            match self.by_set.get(image.as_slice()) {
                Some(&(k, c)) if c == count => pi.push(k),
                _ => {
                    return Err(format!(
                        "maps the detectors {:?} of error {} to {image:?}, which {}",
                        e.detectors,
                        e.index,
                        if self.by_set.contains_key(image.as_slice()) {
                            "a different number of errors flip"
                        } else {
                            "no error flips"
                        }
                    ));
                }
            }
        }
        if self.twins {
            return Ok(None);
        }
        let m = self.errors.len();
        let bits = self.errors.iter().fold(0u64, |b, e| b | e.observables);
        let errors = self.errors;
        let rows = self.rows.get_or_insert_with(|| {
            let mut rows = vec![vec![0u64; gf2::words(m)]; n + 64];
            for (k, e) in errors.iter().enumerate() {
                for &d in &e.detectors {
                    gf2::flip(&mut rows[d], k);
                }
                for b in (0..64).filter(|b| e.observables >> b & 1 == 1) {
                    gf2::flip(&mut rows[n + b], k);
                }
            }
            let mut basis = gf2::Basis::new(m, false, 0);
            for r in &rows {
                basis.insert(r);
            }
            basis
        });
        for b in (0..64).filter(|b| bits >> b & 1 == 1) {
            let mut read = vec![0u64; gf2::words(m)];
            for (k, &p) in pi.iter().enumerate() {
                if errors[p as usize].observables >> b & 1 == 1 {
                    gf2::flip(&mut read, k);
                }
            }
            if !rows.contains(&read) {
                return Err(format!(
                    "maps a set of errors that fires no detector and flips no observable \
                     onto one that flips observable {b}"
                ));
            }
        }
        Ok(Some(pi))
    }

    /// The symmetries found: automorphisms of the graph of detectors and
    /// errors that pass [`Checker::check`], as detector and error
    /// permutations (none with twins; `None` once past `deadline`).
    pub(super) fn find(
        &mut self,
        deadline: Option<Instant>,
    ) -> Option<Vec<(Vec<usize>, Vec<u32>)>> {
        if self.twins {
            return Some(Vec::new());
        }
        let n = self.n;
        // Detectors that no error flips play no part (and would only add
        // permutations among themselves).
        let mut used = vec![usize::MAX; n];
        let mut detectors = Vec::new();
        for e in self.errors {
            for &d in &e.detectors {
                if used[d] == usize::MAX {
                    used[d] = detectors.len();
                    detectors.push(d);
                }
            }
        }
        let u = detectors.len();
        let m = self.errors.len();
        let edges: Vec<(usize, usize)> = self
            .errors
            .iter()
            .enumerate()
            .flat_map(|(k, e)| e.detectors.iter().map(move |&d| (d, k)))
            .map(|(d, k)| (used[d], u + k))
            .collect();
        let g = Graph::from_edges(u + m, &edges).expect("a simple bipartite graph");
        let colours: Vec<u32> = (0..u + m).map(|v| u32::from(v >= u)).collect();
        let group = coloured_group(&g, &colours, deadline)?;
        let mut out = Vec::new();
        for p in group.generators() {
            let mut sigma: Vec<usize> = (0..n).collect();
            for (c, &d) in detectors.iter().enumerate() {
                sigma[d] = detectors[p[c]];
            }
            if let Ok(Some(pi)) = self.check(&sigma)
                && pi.iter().enumerate().any(|(k, &q)| q as usize != k)
            {
                out.push((sigma, pi));
            }
        }
        Some(out)
    }
}

/// The search's order of `m` errors under the group that the error
/// permutations `perms` generate: each orbit as a block of consecutive
/// positions (in index order), the blocks ordered by their first error,
/// and the position where each block starts.
pub(super) fn orbit_order(m: usize, perms: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut orbit = vec![usize::MAX; m];
    let mut count = 0;
    for k in 0..m {
        if orbit[k] != usize::MAX {
            continue;
        }
        orbit[k] = count;
        let mut stack = vec![k];
        while let Some(x) = stack.pop() {
            for p in perms {
                let y = p[x] as usize;
                if orbit[y] == usize::MAX {
                    orbit[y] = count;
                    stack.push(y);
                }
            }
        }
        count += 1;
    }
    let mut order: Vec<u32> = (0..m as u32).collect();
    order.sort_by_key(|&k| (orbit[k as usize], k));
    let starts = (0..m)
        .filter(|&i| i == 0 || orbit[order[i] as usize] != orbit[order[i - 1] as usize])
        .map(|i| i as u32)
        .collect();
    (order, starts)
}
