//! Coined quantum walks with decoherence, on density matrices.
//!
//! Every step applies the unitary step of a [`CoinedWalk`] (coin, then
//! flip-flop shift), `ρ ↦ UρU†`, and then dephasing in the arc basis
//! `|u → v⟩` (position `u`, coin `v`):
//!
//! * position dephasing: with probability `p_x` the position is measured
//!   and the outcome forgotten, `ρ_ab ↦ (1 − p_x)·ρ_ab` for arcs at
//!   different nodes;
//! * coin dephasing: with probability `p_c` the coin register is measured,
//!   the index of the arc among those leaving its node (in order of
//!   increasing neighbour), `ρ_ab ↦ (1 − p_c)·ρ_ab` for arcs with
//!   different indices.
//!
//! Both act at once: `ρ_ab ↦ f_x·f_c·ρ_ab`. Each is a mixture of the
//! identity and a projective measurement, so the step is a channel for all
//! `p_c, p_x ∈ [0, 1]`. Different arcs at the same node always have
//! different indices.
//!
//! With full coin dephasing the node probabilities follow a classical
//! Markov chain on the arcs: from an arc arriving at a node, the walker
//! leaves along arc `j` with probability `|C_jk|²`. For the Fourier coin,
//! `|C_jk|² = 1/d`, so this is the simple random walk; the tests check
//! it against matrix powers of `D⁻¹A`.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::{Coin, CoinedWalk};
//! use metatron_qso::walks::decoherent::DecoherentCoinedWalk;
//!
//! let walk = CoinedWalk::new(&generators::cycle(10), Coin::Hadamard).unwrap();
//! let open = DecoherentCoinedWalk::new(walk, 0.1, 0.0).unwrap();
//! let p = open.probabilities(0, 20).unwrap();
//! assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! ```

use num_complex::Complex64 as C64;

use super::CoinedWalk;
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;

/// Most arcs (twice the edges) of a walk on density matrices: 2048² complex
/// entries are 64 MiB.
pub const MAX_DECOHERENT_ARCS: usize = 2048;

/// A coined walk with coin and position dephasing.
#[derive(Clone, Debug)]
pub struct DecoherentCoinedWalk {
    walk: CoinedWalk,
    /// Node of every arc.
    tails: Vec<usize>,
    /// Index of every arc among the arcs leaving its node (the coin state).
    labels: Vec<usize>,
    coin_dephasing: f64,
    position_dephasing: f64,
}

fn check_probability(name: &'static str, p: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&p) {
        return Err(QsoError::invalid(name, format!("{p} is not a probability")));
    }
    Ok(())
}

impl DecoherentCoinedWalk {
    /// `walk` with coin dephasing `coin` and position dephasing `position`
    /// (probabilities per step).
    pub fn new(walk: CoinedWalk, coin: f64, position: f64) -> Result<Self> {
        check_probability("coin", coin)?;
        check_probability("position", position)?;
        if walk.num_arcs() > MAX_DECOHERENT_ARCS {
            return Err(QsoError::TooManyQubits {
                what: "arcs of a coined walk on density matrices",
                max: MAX_DECOHERENT_ARCS,
                requested: walk.num_arcs(),
            });
        }
        let tails: Vec<usize> = walk.arcs().into_iter().map(|(u, _)| u).collect();
        let labels = (0..tails.len())
            .map(|a| a - tails.iter().position(|&u| u == tails[a]).expect("own node"))
            .collect();
        Ok(Self {
            walk,
            tails,
            labels,
            coin_dephasing: coin,
            position_dephasing: position,
        })
    }

    /// The same walk with the coin register's states given by `labels`,
    /// one per arc (in the order of [`CoinedWalk::arcs`]) and different
    /// for the arcs leaving each node: coin dephasing then measures the
    /// label. The default is the index of the arc at its node; on a
    /// hypercube the coordinate an arc flips makes the coin dephasing
    /// invariant under permutations of the coordinates.
    pub fn with_coin_labels(mut self, labels: Vec<usize>) -> Result<Self> {
        if labels.len() != self.tails.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.tails.len(),
                actual: labels.len(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        for (&u, &l) in self.tails.iter().zip(&labels) {
            if !seen.insert((u, l)) {
                return Err(QsoError::invalid(
                    "labels",
                    format!("two arcs leaving node {u} have the label {l}"),
                ));
            }
        }
        self.labels = labels;
        Ok(self)
    }

    /// The coin register's state of every arc.
    pub fn coin_labels(&self) -> &[usize] {
        &self.labels
    }

    /// The underlying coherent walk.
    pub fn walk(&self) -> &CoinedWalk {
        &self.walk
    }

    pub fn coin_dephasing(&self) -> f64 {
        self.coin_dephasing
    }

    pub fn position_dephasing(&self) -> f64 {
        self.position_dephasing
    }

    /// `|ψ⟩⟨ψ|`.
    pub fn pure(&self, psi: &[C64]) -> Result<CMatrix> {
        let m = self.walk.num_arcs();
        if psi.len() != m {
            return Err(QsoError::DimensionMismatch {
                expected: m,
                actual: psi.len(),
            });
        }
        Ok(CMatrix::from_fn(m, m, |a, b| psi[a] * psi[b].conj()))
    }

    /// The walker at `node` with the maximally mixed coin.
    pub fn localized_mixed(&self, node: usize) -> Result<CMatrix> {
        let n = self.walk.num_nodes();
        if node >= n {
            return Err(QsoError::NodeOutOfRange { node, n_nodes: n });
        }
        let m = self.walk.num_arcs();
        let arcs: Vec<usize> = (0..m).filter(|&a| self.tails[a] == node).collect();
        if arcs.is_empty() {
            return Err(QsoError::invalid(
                "node",
                format!("node {node} has no edges"),
            ));
        }
        let w = 1.0 / arcs.len() as f64;
        let mut rho = CMatrix::zeros(m, m);
        for a in arcs {
            rho[(a, a)] = C64::new(w, 0.0);
        }
        Ok(rho)
    }

    fn check(&self, rho: &CMatrix) -> Result<()> {
        let m = self.walk.num_arcs();
        if rho.nrows() != m || rho.ncols() != m {
            return Err(QsoError::DimensionMismatch {
                expected: m,
                actual: rho.nrows(),
            });
        }
        Ok(())
    }

    /// One step: `ρ ↦ UρU†`, then dephasing.
    pub fn step(&self, rho: &CMatrix) -> Result<CMatrix> {
        self.check(rho)?;
        let m = self.walk.num_arcs();
        // W = Uρ column by column; then UρU† = (U W†)†.
        let mut w = CMatrix::zeros(m, m);
        for c in 0..m {
            let col: Vec<C64> = rho.column(c).iter().copied().collect();
            let out = self.walk.step(&col)?;
            w.column_mut(c).copy_from_slice(&out);
        }
        let mut out = CMatrix::zeros(m, m);
        for r in 0..m {
            let row: Vec<C64> = w.row(r).iter().map(|z| z.conj()).collect();
            let v = self.walk.step(&row)?;
            // v = column r of U·W†, so (UρU†)[r, c] = conj(v[c]).
            for (c, z) in v.into_iter().enumerate() {
                out[(r, c)] = z.conj();
            }
        }
        let (keep_coin, keep_position) = (1.0 - self.coin_dephasing, 1.0 - self.position_dephasing);
        if keep_coin != 1.0 || keep_position != 1.0 {
            for c in 0..m {
                for r in 0..m {
                    let mut f = 1.0;
                    if self.tails[r] != self.tails[c] {
                        f *= keep_position;
                    }
                    if self.labels[r] != self.labels[c] {
                        f *= keep_coin;
                    }
                    out[(r, c)] *= f;
                }
            }
        }
        Ok(out)
    }

    /// `steps` steps from `rho0`.
    pub fn evolve(&self, rho0: &CMatrix, steps: usize) -> Result<CMatrix> {
        self.check(rho0)?;
        let mut rho = rho0.clone();
        for _ in 0..steps {
            rho = self.step(&rho)?;
        }
        Ok(rho)
    }

    /// Probability of each node: the weight of the arcs leaving it.
    pub fn node_probabilities(&self, rho: &CMatrix) -> Result<Vec<f64>> {
        self.check(rho)?;
        let mut p = vec![0.0; self.walk.num_nodes()];
        for (a, &u) in self.tails.iter().enumerate() {
            p[u] += rho[(a, a)].re;
        }
        Ok(p)
    }

    /// Node probabilities after `steps` steps from `start` with the uniform
    /// coin state (as [`CoinedWalk::probabilities`]).
    pub fn probabilities(&self, start: usize, steps: usize) -> Result<Vec<f64>> {
        let rho0 = self.pure(&self.walk.localized_state(start, None)?)?;
        self.node_probabilities(&self.evolve(&rho0, steps)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, generators};
    use crate::walks::Coin;

    fn simple_random_walk(g: &Graph, start: usize, steps: usize) -> Vec<f64> {
        let n = g.num_nodes();
        let mut p = vec![0.0; n];
        p[start] = 1.0;
        for _ in 0..steps {
            let mut next = vec![0.0; n];
            for (u, &pu) in p.iter().enumerate() {
                let d = g.degree(u) as f64;
                for &(v, _) in g.neighbors(u) {
                    next[v] += pu / d;
                }
            }
            p = next;
        }
        p
    }

    #[test]
    fn without_dephasing_it_is_the_coherent_walk() {
        let g = generators::petersen();
        let walk = CoinedWalk::new(&g, Coin::Grover).unwrap();
        let open = DecoherentCoinedWalk::new(walk.clone(), 0.0, 0.0).unwrap();
        for steps in [0, 1, 7] {
            let a = open.probabilities(2, steps).unwrap();
            let b = walk.probabilities(2, steps).unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        }
    }

    #[test]
    fn full_coin_dephasing_gives_the_classical_walk() {
        // Fourier coin: |C_jk|² = 1/d, so the walk is the simple random walk,
        // whatever the position dephasing.
        for g in [
            generators::petersen(),
            generators::hypercube(3),
            generators::cycle(7),
        ] {
            let walk = CoinedWalk::new(&g, Coin::Fourier).unwrap();
            for position in [0.0, 0.3, 1.0] {
                let open = DecoherentCoinedWalk::new(walk.clone(), 1.0, position).unwrap();
                let rho0 = open.localized_mixed(0).unwrap();
                for steps in [1, 2, 5, 12] {
                    let p = open
                        .node_probabilities(&open.evolve(&rho0, steps).unwrap())
                        .unwrap();
                    let q = simple_random_walk(&g, 0, steps);
                    assert!(
                        p.iter().zip(&q).all(|(x, y)| (x - y).abs() < 1e-12),
                        "{steps}: {p:?} vs {q:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn full_coin_dephasing_of_the_grover_walk_is_a_chain_on_arcs() {
        // From an arriving arc, Grover sends the walker back with
        // probability (1 − 2/d)² and to each other neighbour with (2/d)².
        let g = generators::complete(5);
        let walk = CoinedWalk::new(&g, Coin::Grover).unwrap();
        let open = DecoherentCoinedWalk::new(walk.clone(), 1.0, 0.0).unwrap();
        let arcs = walk.arcs();
        let m = arcs.len();
        let mut p = vec![0.0; m];
        let rho0 = open.localized_mixed(0).unwrap();
        for a in 0..m {
            p[a] = rho0[(a, a)].re;
        }
        let mut rho = rho0;
        for _ in 0..6 {
            // Arc (u → v) at u: the coin sends it to (u → w), the shift to
            // (w → u).
            let mut next = vec![0.0; m];
            for (a, &(u, v)) in arcs.iter().enumerate() {
                let d = g.degree(u) as f64;
                for &(x, w) in &arcs {
                    if x != u {
                        continue;
                    }
                    let amp = if w == v { 2.0 / d - 1.0 } else { 2.0 / d };
                    let back = arcs.iter().position(|&(s, t)| s == w && t == u).unwrap();
                    next[back] += p[a] * amp * amp;
                }
            }
            p = next;
            rho = open.step(&rho).unwrap();
            for a in 0..m {
                assert!((rho[(a, a)].re - p[a]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn steps_preserve_trace_and_hermiticity() {
        let g = generators::hypercube(3);
        let walk = CoinedWalk::new(&g, Coin::Grover)
            .unwrap()
            .with_marked(&[5])
            .unwrap();
        let open = DecoherentCoinedWalk::new(walk, 0.2, 0.4).unwrap();
        let rho = open.evolve(&open.localized_mixed(1).unwrap(), 9).unwrap();
        assert!((rho.trace().re - 1.0).abs() < 1e-12);
        assert!((&rho - rho.adjoint()).norm() < 1e-12);
        let eig = crate::linalg::eigh(&rho).0;
        assert!(eig.iter().all(|&l| l > -1e-12));
    }

    fn dense_step(walk: &CoinedWalk) -> CMatrix {
        let m = walk.num_arcs();
        let mut u = CMatrix::zeros(m, m);
        for c in 0..m {
            let mut e = vec![C64::new(0.0, 0.0); m];
            e[c] = C64::new(1.0, 0.0);
            u.column_mut(c).copy_from_slice(&walk.step(&e).unwrap());
        }
        u
    }

    fn random_density(m: usize, seed: u64) -> CMatrix {
        let mut rng = crate::Rng::seed_from_u64(seed);
        let a = CMatrix::from_fn(m, m, |_, _| C64::new(rng.normal(), rng.normal()));
        let rho = &a * a.adjoint();
        let t = rho.trace();
        rho / t
    }

    #[test]
    fn a_step_is_the_unitary_step_with_complex_coins() {
        // The Fourier coin of degree ≥ 3 is complex: a transposed step would
        // show here.
        for g in [generators::complete(4), generators::petersen()] {
            let walk = CoinedWalk::new(&g, Coin::Fourier).unwrap();
            let u = dense_step(&walk);
            let open = DecoherentCoinedWalk::new(walk.clone(), 0.0, 0.0).unwrap();
            let rho = random_density(walk.num_arcs(), 3);
            let want = &u * &rho * u.adjoint();
            assert!((open.step(&rho).unwrap() - want).norm() < 1e-12);
            let a = open.probabilities(0, 5).unwrap();
            let b = walk.probabilities(0, 5).unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        }
    }

    #[test]
    fn every_dephasing_strength_keeps_states_positive() {
        let cases = [
            (generators::cycle(10), Coin::Hadamard, 0.1, 0.0),
            (generators::complete(5), Coin::Grover, 1.0, 0.0),
            (generators::complete(13), Coin::Fourier, 1.0, 0.0),
            (generators::petersen(), Coin::Grover, 0.5, 0.0),
            (generators::petersen(), Coin::Fourier, 0.3, 0.9),
        ];
        for (g, coin, pc, px) in cases {
            let walk = CoinedWalk::new(&g, coin).unwrap();
            let open = DecoherentCoinedWalk::new(walk.clone(), pc, px).unwrap();
            let mut rho = random_density(walk.num_arcs(), 7);
            for _ in 0..6 {
                rho = open.step(&rho).unwrap();
                let eig = crate::linalg::eigh(&rho).0;
                assert!(eig.iter().all(|&l| l > -1e-12), "{pc} {px}: {}", eig[0]);
                assert!((rho.trace().re - 1.0).abs() < 1e-12);
            }
            let pure = open.probabilities(0, 8).unwrap();
            assert!(pure.iter().all(|&p| p > -1e-12));
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let walk = CoinedWalk::new(&generators::cycle(4), Coin::Grover).unwrap();
        assert!(DecoherentCoinedWalk::new(walk.clone(), 1.5, 0.0).is_err());
        assert!(DecoherentCoinedWalk::new(walk.clone(), 0.0, f64::NAN).is_err());
        let open = DecoherentCoinedWalk::new(walk, 0.5, 0.5).unwrap();
        assert!(open.localized_mixed(4).is_err());
        assert!(open.step(&CMatrix::zeros(3, 3)).is_err());
        assert!(open.pure(&[C64::new(1.0, 0.0)]).is_err());
        assert!(open.clone().with_coin_labels(vec![0; 8]).is_err());
        assert!(open.clone().with_coin_labels(vec![0, 1]).is_err());
        let relabelled = open.with_coin_labels(vec![1, 0, 1, 0, 1, 0, 1, 0]).unwrap();
        assert_eq!(relabelled.coin_labels()[0], 1);
        let big = CoinedWalk::new(&generators::complete(64), Coin::Grover).unwrap();
        assert!(DecoherentCoinedWalk::new(big, 0.0, 0.0).is_err());
    }
}
