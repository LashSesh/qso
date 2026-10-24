//! Discrete-time quantum walks.
//!
//! * [`CoinedWalk`] — the coined walk on a graph. The walker lives on the
//!   arcs `u → v` (position `u`, coin pointing to `v`); one step applies a
//!   coin to the arcs leaving every node and then the *flip-flop* shift
//!   `|u → v⟩ ↦ |v → u⟩`. Coins: Grover diffusion, discrete Fourier,
//!   Hadamard or any unitary; marked nodes get the coin `−I`, which turns
//!   the walk into the search of Shenvi, Kempe & Whaley (2003).
//! * [`LineWalk`] — the walk on the integer line with the *moving* shift
//!   (`|x, L⟩ → |x−1, L⟩`, `|x, R⟩ → |x+1, R⟩`), the standard model for
//!   ballistic spreading (Konno 2002).
//! * [`SzegedyWalk`] — Szegedy's (2004) quantisation of a reversible
//!   Markov chain and its quantum hitting time.
//!
//! Edge weights are ignored by the coined walk (arcs follow the edges) and
//! define the Markov chain `P_uv = w_uv / d_u` of the Szegedy walk.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::{Coin, CoinedWalk};
//!
//! // Search for a node of the 2¹⁰-node hypercube (SKW).
//! let walk = CoinedWalk::new(&generators::hypercube(10), Coin::Grover).unwrap()
//!     .with_marked(&[123]).unwrap();
//! let best = walk.search(100).unwrap();
//! assert!(best.success_probability > 0.4 && best.steps < 50);
//! ```

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::CMatrix;

const ZERO: C64 = C64::new(0.0, 0.0);

fn check_unitary(u: &CMatrix, name: &'static str) -> Result<()> {
    let d = u.nrows();
    if u.ncols() != d || d == 0 {
        return Err(QsoError::invalid(
            name,
            "coin must be a non-empty square matrix",
        ));
    }
    if u.iter().any(|x| !x.is_finite())
        || (u.adjoint() * u - CMatrix::identity(d, d)).norm() > 1e-10
    {
        return Err(QsoError::invalid(name, "coin must be unitary"));
    }
    Ok(())
}

/// Coin operator of a [`CoinedWalk`], applied to the arcs leaving a node
/// in order of increasing neighbour index.
#[derive(Clone, Debug, PartialEq)]
pub enum Coin {
    /// Grover diffusion `(2/d)·J − I`: invariant under relabelling the
    /// arcs, so it works on any graph.
    Grover,
    /// Discrete Fourier transform `F_jk = ω^{jk}/√d`, `ω = e^{2πi/d}`.
    Fourier,
    /// The Hadamard matrix (nodes of degree 2 only).
    Hadamard,
    /// The same unitary at every node (the graph must be regular of the
    /// matching degree).
    Unitary(CMatrix),
}

/// Outcome of a discrete-time search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscreteSearchResult {
    /// Number of steps with the highest success probability in the window
    /// (the earliest, if several are equal).
    pub steps: usize,
    /// Probability of measuring a marked node after `steps` steps.
    pub success_probability: f64,
    /// Success probability after `0, 1, …, max_steps` steps.
    pub history: Vec<f64>,
}

/// Coined quantum walk with the flip-flop shift (see the module
/// documentation).
#[derive(Clone, Debug)]
pub struct CoinedWalk {
    /// Arcs leaving node `u`: `offsets[u]..offsets[u + 1]`.
    offsets: Vec<usize>,
    /// Head of every arc.
    heads: Vec<usize>,
    /// Index of the reversed arc.
    reverse: Vec<usize>,
    coin: Coin,
    /// Fourier coins by degree, built on demand at construction.
    fourier: Vec<Option<CMatrix>>,
    marked: Vec<bool>,
}

impl CoinedWalk {
    pub fn new(graph: &Graph, coin: Coin) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 {
            return Err(QsoError::invalid("graph", "graph has no nodes"));
        }
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut heads = Vec::with_capacity(2 * graph.num_edges());
        for u in 0..n {
            heads.extend(graph.neighbors(u).iter().map(|&(v, _)| v));
            offsets.push(heads.len());
        }
        let reverse = (0..n)
            .flat_map(|u| graph.neighbors(u).iter().map(move |&(v, _)| (u, v)))
            .map(|(u, v)| {
                let pos = graph
                    .neighbors(v)
                    .binary_search_by(|&(z, _)| z.cmp(&u))
                    .expect("undirected graph");
                offsets[v] + pos
            })
            .collect();
        let max_degree = (0..n).map(|u| graph.degree(u)).max().unwrap_or(0);
        let mut fourier = vec![None; max_degree + 1];
        match &coin {
            Coin::Grover => {}
            Coin::Fourier => {
                for u in 0..n {
                    let d = graph.degree(u);
                    if fourier[d].is_none() {
                        let w = 2.0 * std::f64::consts::PI / d.max(1) as f64;
                        let s = 1.0 / (d as f64).sqrt();
                        fourier[d] = Some(CMatrix::from_fn(d, d, |j, k| {
                            C64::from_polar(s, w * ((j * k) % d.max(1)) as f64)
                        }));
                    }
                }
            }
            Coin::Hadamard => {
                if let Some(u) = (0..n).find(|&u| graph.degree(u) != 2) {
                    return Err(QsoError::invalid(
                        "coin",
                        format!(
                            "the Hadamard coin needs degree 2, node {u} has {}",
                            graph.degree(u)
                        ),
                    ));
                }
            }
            Coin::Unitary(m) => {
                check_unitary(m, "coin")?;
                if let Some(u) = (0..n).find(|&u| graph.degree(u) != m.nrows()) {
                    return Err(QsoError::invalid(
                        "coin",
                        format!(
                            "a {0}×{0} coin needs a {0}-regular graph, node {u} has degree {1}",
                            m.nrows(),
                            graph.degree(u)
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            offsets,
            heads,
            reverse,
            coin,
            fourier,
            marked: vec![false; n],
        })
    }

    /// The same walk with coin `−I` at the `marked` nodes.
    pub fn with_marked(mut self, marked: &[usize]) -> Result<Self> {
        let n = self.num_nodes();
        self.marked = vec![false; n];
        for &m in marked {
            if m >= n {
                return Err(QsoError::NodeOutOfRange {
                    node: m,
                    n_nodes: n,
                });
            }
            self.marked[m] = true;
        }
        Ok(self)
    }

    pub fn num_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Dimension of the walk: twice the number of edges.
    pub fn num_arcs(&self) -> usize {
        self.heads.len()
    }

    /// The arcs `(tail, head)` in the order of the state vector.
    pub fn arcs(&self) -> Vec<(usize, usize)> {
        (0..self.num_nodes())
            .flat_map(|u| {
                self.heads[self.offsets[u]..self.offsets[u + 1]]
                    .iter()
                    .map(move |&v| (u, v))
            })
            .collect()
    }

    /// Walker at `node` with coin state `coin` (one amplitude per arc
    /// leaving the node, normalised), by default the uniform superposition.
    pub fn localized_state(&self, node: usize, coin: Option<&[C64]>) -> Result<Vec<C64>> {
        let n = self.num_nodes();
        if node >= n {
            return Err(QsoError::NodeOutOfRange { node, n_nodes: n });
        }
        let arcs = self.offsets[node]..self.offsets[node + 1];
        if arcs.is_empty() {
            return Err(QsoError::invalid(
                "node",
                "an isolated node has no coin states",
            ));
        }
        let mut psi = vec![ZERO; self.num_arcs()];
        match coin {
            None => {
                let a = C64::new(1.0 / (arcs.len() as f64).sqrt(), 0.0);
                psi[arcs].iter_mut().for_each(|x| *x = a);
            }
            Some(c) => {
                if c.len() != arcs.len() {
                    return Err(QsoError::DimensionMismatch {
                        expected: arcs.len(),
                        actual: c.len(),
                    });
                }
                let norm: f64 = c.iter().map(|x| x.norm_sqr()).sum();
                if !norm.is_finite() || (norm - 1.0).abs() >= 1e-10 {
                    return Err(QsoError::invalid("coin", "coin state must be normalised"));
                }
                psi[arcs].copy_from_slice(c);
            }
        }
        Ok(psi)
    }

    /// Uniform superposition over all arcs (the start state of a search).
    pub fn uniform_state(&self) -> Vec<C64> {
        let a = C64::new(1.0 / (self.num_arcs().max(1) as f64).sqrt(), 0.0);
        vec![a; self.num_arcs()]
    }

    fn check_state(&self, psi: &[C64]) -> Result<()> {
        if psi.len() != self.num_arcs() {
            return Err(QsoError::DimensionMismatch {
                expected: self.num_arcs(),
                actual: psi.len(),
            });
        }
        Ok(())
    }

    /// Apply the coin of node `u` to its block of arcs.
    fn apply_coin(&self, u: usize, block: &mut [C64], scratch: &mut Vec<C64>) {
        let d = block.len();
        if self.marked[u] {
            block.iter_mut().for_each(|x| *x = -*x);
            return;
        }
        let dense = match &self.coin {
            Coin::Grover => {
                let mean = block.iter().sum::<C64>() * (2.0 / d as f64);
                block.iter_mut().for_each(|x| *x = mean - *x);
                return;
            }
            Coin::Hadamard => {
                let s = std::f64::consts::FRAC_1_SQRT_2;
                let (a, b) = (block[0], block[1]);
                block[0] = (a + b) * s;
                block[1] = (a - b) * s;
                return;
            }
            Coin::Fourier => self.fourier[d].as_ref().expect("built for every degree"),
            Coin::Unitary(m) => m,
        };
        scratch.clear();
        scratch.extend((0..d).map(|r| (0..d).map(|c| dense[(r, c)] * block[c]).sum::<C64>()));
        block.copy_from_slice(scratch);
    }

    /// One step: coin, then flip-flop shift.
    pub fn step(&self, psi: &[C64]) -> Result<Vec<C64>> {
        self.check_state(psi)?;
        let mut coined = psi.to_vec();
        let mut scratch = Vec::new();
        for u in 0..self.num_nodes() {
            let block = &mut coined[self.offsets[u]..self.offsets[u + 1]];
            if !block.is_empty() {
                self.apply_coin(u, block, &mut scratch);
            }
        }
        let mut out = vec![ZERO; psi.len()];
        for (a, &x) in coined.iter().enumerate() {
            out[self.reverse[a]] = x;
        }
        Ok(out)
    }

    /// `steps` steps from `psi0`.
    pub fn evolve(&self, psi0: &[C64], steps: usize) -> Result<Vec<C64>> {
        self.check_state(psi0)?;
        let mut psi = psi0.to_vec();
        for _ in 0..steps {
            psi = self.step(&psi)?;
        }
        Ok(psi)
    }

    /// Probability of each node: the total weight of the arcs leaving it.
    pub fn node_probabilities(&self, psi: &[C64]) -> Result<Vec<f64>> {
        self.check_state(psi)?;
        Ok((0..self.num_nodes())
            .map(|u| {
                psi[self.offsets[u]..self.offsets[u + 1]]
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum()
            })
            .collect())
    }

    /// Node probabilities after `steps` steps from `start` with the uniform
    /// coin state.
    pub fn probabilities(&self, start: usize, steps: usize) -> Result<Vec<f64>> {
        let psi = self.evolve(&self.localized_state(start, None)?, steps)?;
        self.node_probabilities(&psi)
    }

    /// Search from the uniform superposition: the success probability (of
    /// measuring a marked node) after `0, …, max_steps` steps, and the step
    /// where it is highest. The success probability oscillates; choose
    /// `max_steps` to cover the first peak only (about `(π/2)·√(N/2)` steps
    /// on the hypercube) unless later, slightly higher peaks are wanted.
    pub fn search(&self, max_steps: usize) -> Result<DiscreteSearchResult> {
        if !self.marked.contains(&true) {
            return Err(QsoError::invalid(
                "marked",
                "mark nodes with with_marked first",
            ));
        }
        let success = |psi: &[C64]| -> f64 {
            (0..self.num_nodes())
                .filter(|&u| self.marked[u])
                .map(|u| {
                    psi[self.offsets[u]..self.offsets[u + 1]]
                        .iter()
                        .map(|x| x.norm_sqr())
                        .sum::<f64>()
                })
                .sum()
        };
        let mut psi = self.uniform_state();
        let mut history = vec![success(&psi)];
        for _ in 0..max_steps {
            psi = self.step(&psi)?;
            history.push(success(&psi));
        }
        let (steps, &success_probability) =
            history
                .iter()
                .enumerate()
                .fold(
                    (0, &history[0]),
                    |best, (k, p)| if *p > *best.1 { (k, p) } else { best },
                );
        Ok(DiscreteSearchResult {
            steps,
            success_probability,
            history,
        })
    }
}

/// Coined walk on the integer line with the moving shift (see the module
/// documentation). Coin state 0 (`L`) moves left, 1 (`R`) moves right.
#[derive(Clone, Debug, PartialEq)]
pub struct LineWalk {
    coin: CMatrix,
}

impl LineWalk {
    /// Walk with a 2×2 unitary coin.
    pub fn new(coin: &CMatrix) -> Result<Self> {
        check_unitary(coin, "coin")?;
        if coin.nrows() != 2 {
            return Err(QsoError::invalid("coin", "the line walk needs a 2×2 coin"));
        }
        Ok(Self { coin: coin.clone() })
    }

    /// The Hadamard walk.
    pub fn hadamard() -> Self {
        let s = C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        Self {
            coin: CMatrix::from_row_slice(2, 2, &[s, s, s, -s]),
        }
    }

    /// Position distribution after `steps` steps from the origin with coin
    /// state `coin_state`: entry `k` is the probability of position
    /// `k − steps`. The Hadamard walk from `(|L⟩ + i|R⟩)/√2` spreads
    /// symmetrically with standard deviation `→ √(1 − 1/√2)·steps`.
    pub fn distribution(&self, steps: usize, coin_state: [C64; 2]) -> Result<Vec<f64>> {
        if steps > 1 << 20 {
            return Err(QsoError::invalid("steps", "at most 2^20 steps"));
        }
        let norm = coin_state[0].norm_sqr() + coin_state[1].norm_sqr();
        if !norm.is_finite() || (norm - 1.0).abs() >= 1e-10 {
            return Err(QsoError::invalid(
                "coin_state",
                "coin state must be normalised",
            ));
        }
        let width = 2 * steps + 1;
        let mut left = vec![ZERO; width];
        let mut right = vec![ZERO; width];
        left[steps] = coin_state[0];
        right[steps] = coin_state[1];
        let c = &self.coin;
        for _ in 0..steps {
            let mut new_left = vec![ZERO; width];
            let mut new_right = vec![ZERO; width];
            for x in 0..width {
                let (l, r) = (left[x], right[x]);
                if l == ZERO && r == ZERO {
                    continue;
                }
                let (cl, cr) = (c[(0, 0)] * l + c[(0, 1)] * r, c[(1, 0)] * l + c[(1, 1)] * r);
                if x > 0 {
                    new_left[x - 1] += cl;
                }
                if x + 1 < width {
                    new_right[x + 1] += cr;
                }
            }
            left = new_left;
            right = new_right;
        }
        Ok(left
            .iter()
            .zip(&right)
            .map(|(l, r)| l.norm_sqr() + r.norm_sqr())
            .collect())
    }
}

/// Szegedy walk of the random walk `P_uv = w_uv / d_u` on a graph, with
/// marked nodes made absorbing (see the module documentation).
///
/// The walk acts on the arcs `(u, v)` with `P_uv > 0`, plus `(m, m)` for
/// every marked node. With `|ψ_u⟩ = Σ_v √P'_uv |u, v⟩` for the absorbing
/// chain `P'` (`P'_mm = 1`), one step is `W' = (S·R)²`, where
/// `R = 2Σ_u|ψ_u⟩⟨ψ_u| − I` and `S` swaps `|u, v⟩ ↔ |v, u⟩`. The walk starts
/// in `|π⟩ = Σ_u √π_u |ψ_u⟩` of the original chain (`π_u ∝ d_u`), and the
/// quantum hitting time is the smallest `T` with
/// `(1/(T+1))·Σ_{t≤T} ‖W'^t|π⟩ − |π⟩‖² ≥ 1 − π(M)` (Szegedy 2004). It grows
/// as the square root of the classical hitting time.
#[derive(Clone, Debug)]
pub struct SzegedyWalk {
    /// Arcs leaving node `u`: `offsets[u]..offsets[u + 1]`.
    offsets: Vec<usize>,
    /// `√P'_uv` on every arc.
    amplitudes: Vec<f64>,
    swap: Vec<usize>,
    start: Vec<f64>,
    marked_weight: f64,
}

impl SzegedyWalk {
    pub fn new(graph: &Graph, marked: &[usize]) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 || graph.edges().iter().any(|&(_, _, w)| w <= 0.0) {
            return Err(QsoError::invalid(
                "graph",
                "need a non-empty graph with positive weights",
            ));
        }
        if let Some(u) = (0..n).find(|&u| graph.degree(u) == 0) {
            return Err(QsoError::invalid("graph", format!("node {u} is isolated")));
        }
        if marked.is_empty() {
            return Err(QsoError::invalid(
                "marked",
                "at least one marked node required",
            ));
        }
        let mut is_marked = vec![false; n];
        for &m in marked {
            if m >= n {
                return Err(QsoError::NodeOutOfRange {
                    node: m,
                    n_nodes: n,
                });
            }
            is_marked[m] = true;
        }
        // Arcs from u: its neighbours in order, then (u, u) if u is marked.
        let mut offsets = vec![0];
        let mut amplitudes = Vec::new();
        let mut start = Vec::new();
        let total: f64 = (0..n).map(|u| graph.weighted_degree(u)).sum();
        for (u, &absorbing) in is_marked.iter().enumerate() {
            let d = graph.weighted_degree(u);
            for &(_, w) in graph.neighbors(u) {
                amplitudes.push(if absorbing { 0.0 } else { (w / d).sqrt() });
                start.push((w / total).sqrt());
            }
            if absorbing {
                amplitudes.push(1.0);
                start.push(0.0);
            }
            offsets.push(amplitudes.len());
        }
        let swap = (0..n)
            .flat_map(|u| {
                let own = graph.neighbors(u).iter().map(move |&(v, _)| (u, v));
                let own_loop = is_marked[u].then_some((u, u));
                own.chain(own_loop)
            })
            .map(|(u, v)| {
                if u == v {
                    offsets[u + 1] - 1
                } else {
                    let pos = graph
                        .neighbors(v)
                        .binary_search_by(|&(z, _)| z.cmp(&u))
                        .expect("undirected graph");
                    offsets[v] + pos
                }
            })
            .collect();
        let marked_weight = (0..n)
            .filter(|&u| is_marked[u])
            .map(|u| graph.weighted_degree(u))
            .sum::<f64>()
            / total;
        Ok(Self {
            offsets,
            amplitudes,
            swap,
            start,
            marked_weight,
        })
    }

    /// Dimension of the walk (number of arcs).
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    /// The start state `|π⟩` in the arc basis.
    pub fn stationary_state(&self) -> &[f64] {
        &self.start
    }

    /// Stationary probability `π(M)` of the marked set.
    pub fn marked_weight(&self) -> f64 {
        self.marked_weight
    }

    fn half_step(&self, psi: &[C64]) -> Vec<C64> {
        let mut reflected = vec![ZERO; psi.len()];
        for u in 0..self.offsets.len() - 1 {
            let arcs = self.offsets[u]..self.offsets[u + 1];
            let overlap: C64 = arcs.clone().map(|a| psi[a] * self.amplitudes[a]).sum();
            for a in arcs {
                reflected[a] = overlap * (2.0 * self.amplitudes[a]) - psi[a];
            }
        }
        let mut out = vec![ZERO; psi.len()];
        for (a, &x) in reflected.iter().enumerate() {
            out[self.swap[a]] = x;
        }
        out
    }

    /// One step `W' = (S·R)²`.
    pub fn step(&self, psi: &[C64]) -> Result<Vec<C64>> {
        if psi.len() != self.dimension() {
            return Err(QsoError::DimensionMismatch {
                expected: self.dimension(),
                actual: psi.len(),
            });
        }
        Ok(self.half_step(&self.half_step(psi)))
    }

    /// Deviations `‖W'^t|π⟩ − |π⟩‖²` for `t = 0, 1, …`, until `stop`
    /// returns `true` or `steps` steps are done.
    fn deviations_until(&self, steps: usize, mut stop: impl FnMut(usize, f64) -> bool) -> Vec<f64> {
        let start: Vec<C64> = self.start.iter().map(|&x| C64::new(x, 0.0)).collect();
        let mut psi = start.clone();
        let mut out = Vec::new();
        for t in 0..=steps {
            if t > 0 {
                psi = self.half_step(&self.half_step(&psi));
            }
            let d = psi
                .iter()
                .zip(&start)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum();
            out.push(d);
            if stop(t, d) {
                break;
            }
        }
        out
    }

    /// `‖W'^t|π⟩ − |π⟩‖²` for `t = 0, …, steps`.
    pub fn deviations(&self, steps: usize) -> Vec<f64> {
        self.deviations_until(steps, |_, _| false)
    }

    /// Quantum hitting time (see the type documentation), if reached within
    /// `max_steps` steps.
    pub fn hitting_time(&self, max_steps: usize) -> Option<usize> {
        let mut total = 0.0;
        let mut hit = None;
        self.deviations_until(max_steps, |t, d| {
            total += d;
            if total / (t + 1) as f64 >= 1.0 - self.marked_weight {
                hit = Some(t);
            }
            hit.is_some()
        });
        hit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;
    use crate::walks::classical_hitting_times;

    fn unitary_check(walk: &CoinedWalk, seed: u64) {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        let mut psi: Vec<C64> = (0..walk.num_arcs())
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let norm = psi.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        psi.iter_mut().for_each(|x| *x /= norm);
        let out = walk.evolve(&psi, 7).unwrap();
        assert!((out.iter().map(|x| x.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn coined_walks_are_unitary() {
        let g = generators::petersen();
        for coin in [Coin::Grover, Coin::Fourier] {
            let walk = CoinedWalk::new(&g, coin).unwrap();
            assert_eq!(walk.num_arcs(), 30);
            unitary_check(&walk, 1);
            unitary_check(&walk.with_marked(&[3]).unwrap(), 2);
        }
        let mut rng = crate::rng::Rng::seed_from_u64(4);
        let irregular = generators::erdos_renyi(15, 0.4, &mut rng).unwrap();
        unitary_check(&CoinedWalk::new(&irregular, Coin::Fourier).unwrap(), 3);
        unitary_check(
            &CoinedWalk::new(&generators::cycle(9), Coin::Hadamard).unwrap(),
            4,
        );
        let x = CMatrix::from_row_slice(
            3,
            3,
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0].map(|v| C64::new(v, 0.0)),
        );
        unitary_check(&CoinedWalk::new(&g, Coin::Unitary(x.clone())).unwrap(), 5);
        // Coins that do not fit the graph are rejected.
        assert!(CoinedWalk::new(&generators::hypercube(3), Coin::Hadamard).is_err());
        assert!(CoinedWalk::new(&generators::cycle(5), Coin::Unitary(x)).is_err());
        let not_unitary = CMatrix::from_element(2, 2, C64::new(1.0, 0.0));
        assert!(CoinedWalk::new(&generators::cycle(5), Coin::Unitary(not_unitary)).is_err());
        // The arc order follows the adjacency lists.
        let p = CoinedWalk::new(&generators::path(3), Coin::Grover).unwrap();
        assert_eq!(p.arcs(), vec![(0, 1), (1, 0), (1, 2), (2, 1)]);
    }

    #[test]
    fn grover_walk_on_complete_graph_keeps_symmetry() {
        // From the uniform coin state at node 0 of K_n, all other nodes
        // stay equally likely.
        let walk = CoinedWalk::new(&generators::complete(7), Coin::Grover).unwrap();
        for steps in [1, 2, 5] {
            let p = walk.probabilities(0, steps).unwrap();
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(p[1..].iter().all(|&x| (x - p[1]).abs() < 1e-12));
        }
        // One step from the uniform coin state moves the walker off node 0.
        assert!(walk.probabilities(0, 1).unwrap()[0] < 1e-12);
    }

    #[test]
    fn hadamard_walk_spreads_ballistically() {
        let walk = LineWalk::hadamard();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let steps = 1000;
        let p = walk
            .distribution(steps, [C64::new(s, 0.0), C64::new(0.0, s)])
            .unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        let x = |k: usize| k as f64 - steps as f64;
        let mean: f64 = p.iter().enumerate().map(|(k, q)| q * x(k)).sum();
        let var: f64 = p
            .iter()
            .enumerate()
            .map(|(k, q)| q * x(k).powi(2))
            .sum::<f64>()
            - mean * mean;
        assert!(mean.abs() < 1e-9);
        // Konno: σ/t → √(1 − 1/√2) ≈ 0.5412 (the classical walk has √t).
        assert!((var.sqrt() / steps as f64 - (1.0 - s).sqrt()).abs() < 1e-5);
        // Odd positions after an even number of steps are never occupied.
        assert!(p.iter().skip(1).step_by(2).all(|&q| q == 0.0));
        // From |L⟩ alone the distribution leans to the left.
        let left = walk
            .distribution(100, [C64::new(1.0, 0.0), C64::new(0.0, 0.0)])
            .unwrap();
        let mean_left: f64 = left
            .iter()
            .enumerate()
            .map(|(k, q)| q * (k as f64 - 100.0))
            .sum();
        assert!(mean_left < -10.0);
        assert!(
            walk.distribution(3, [C64::new(1.0, 0.0), C64::new(1.0, 0.0)])
                .is_err()
        );
    }

    #[test]
    fn skw_search_finds_a_hypercube_node_in_sqrt_n_steps() {
        // Peak near (π/2)·√(N/2) steps with success probability → 1/2
        // (numpy cross-check: n = 10 peaks at step 38 with 0.4350).
        let walk = CoinedWalk::new(&generators::hypercube(10), Coin::Grover)
            .unwrap()
            .with_marked(&[0])
            .unwrap();
        let r = walk.search(100).unwrap();
        assert_eq!(r.steps, 38);
        assert!((r.success_probability - 0.43500643358).abs() < 1e-9);
        assert!((r.history[0] - 1.0 / 1024.0).abs() < 1e-15);
        // The marked node does not matter on a vertex-transitive graph.
        let other = CoinedWalk::new(&generators::hypercube(10), Coin::Grover)
            .unwrap()
            .with_marked(&[777])
            .unwrap()
            .search(100)
            .unwrap();
        assert_eq!(other.steps, 38);
        assert!(
            CoinedWalk::new(&generators::cycle(4), Coin::Grover)
                .unwrap()
                .search(5)
                .is_err()
        );
    }

    #[test]
    fn szegedy_hitting_time_is_quadratically_faster() {
        // Numpy cross-check: complete graphs K_16, K_64, K_256 need 3, 6, 11
        // steps; cycles C_n exactly n/4.
        for (n, expected) in [(16, 3), (64, 6), (256, 11)] {
            let w = SzegedyWalk::new(&generators::complete(n), &[0]).unwrap();
            assert_eq!(w.hitting_time(10 * n), Some(expected), "K{n}");
            assert!((w.marked_weight() - 1.0 / n as f64).abs() < 1e-15);
        }
        let mut ratios = Vec::new();
        for n in [16usize, 32, 64, 128] {
            let g = generators::cycle(n);
            let quantum = SzegedyWalk::new(&g, &[0])
                .unwrap()
                .hitting_time(10 * n)
                .unwrap();
            assert_eq!(quantum, n / 4, "C{n}");
            let h = classical_hitting_times(&g, 0).unwrap();
            let classical: f64 = h.iter().sum::<f64>() / n as f64;
            ratios.push(quantum as f64 / classical.sqrt());
        }
        // Quantum / √classical is constant along the family.
        assert!(
            ratios.iter().all(|r| (r - ratios[0]).abs() < 0.02),
            "{ratios:?}"
        );
        // Norm is preserved.
        let w = SzegedyWalk::new(&generators::petersen(), &[2, 7]).unwrap();
        let start: Vec<C64> = w
            .stationary_state()
            .iter()
            .map(|&x| C64::new(x, 0.0))
            .collect();
        assert!((start.iter().map(|x| x.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-12);
        let after = w.step(&start).unwrap();
        assert!((after.iter().map(|x| x.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(SzegedyWalk::new(&generators::petersen(), &[]).is_err());
        assert!(SzegedyWalk::new(&Graph::new(3), &[0]).is_err());
    }
}
