//! Quantum walks with site dephasing, traps, losses and site energies.
//!
//! The walker's density matrix obeys the Lindblad equation
//!
//! ```text
//! dρ/dt = −i(H_eff ρ − ρ H_eff†) + Σ_j γ_j (|j⟩⟨j| ρ |j⟩⟨j| − ½{|j⟩⟨j|, ρ})
//! H_eff = H − i Σ_t κ_t |t⟩⟨t| − (i/2) Σ_j Γ_j |j⟩⟨j|,   H = γ_hop·L or −γ_hop·A, plus diag(ε)
//! ```
//!
//! with dephasing rates `γ_j` (a uniform `rate` by default), traps `κ_t`
//! (the sink fills at `2κ_t ρ_tt`), losses `Γ_j` (population lost at
//! `Γ_j ρ_jj`) and site energies `ε_j`. Entry `ρ_ab` (`a ≠ b`) decays at
//! `k_a + k_b + (γ_a + γ_b)/2` with `k = κ + Γ/2` and rotates at
//! `−i(ε_a − ε_b)`; a population `ρ_aa` decays at `2κ_a + Γ_a`. For a
//! uniform rate `γ` without traps and losses this is
//! `dρ/dt = −i[H, ρ] − γ (ρ − diag ρ)`: for `γ = 0` the coherent walk, for
//! `γ ≫ γ_hop` the classical master equation `dp/dt = −(2γ_hop²/γ)·L·p`
//! (quantum Zeno regime). On a connected graph without traps and losses the
//! unique steady state is the maximally mixed state, i.e. the uniform
//! distribution.
//!
//! Evolution ([`DephasingWalk::evolve`], [`DephasingWalk::populations`],
//! [`DephasingWalk::transport_trace`]) uses the classical Runge–Kutta
//! method with step doubling and Richardson extrapolation; the local error
//! per step is kept below the requested tolerance. It costs about `n³`
//! operations per step and is limited to 512 nodes. The efficiency and
//! arrival time at the traps ([`DephasingWalk::transport`]) come from
//! linear solves without time steps, up to 2048 nodes (see
//! [`super::transport`]).

use num_complex::Complex64 as C64;

use super::transport::dense::{Bright, Closure, DenseModel, Start};
use super::transport::{
    Budget, DephasingOptimum, TransportOptions, TransportResult, TransportState, check_energies,
    check_rates, check_scan_rates, check_times, check_traps, optimise,
};
use super::{DENSE_LIMIT, Generator};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{CMatrix, RMatrix};

/// Most nodes of density-matrix evolution (RK4 costs `∝ n⁴t` on dense
/// graphs: the complete graph on 128 nodes took 130 s at `t = 1`).
const MAX_EVOLUTION_NODES: usize = 512;

/// Open quantum walk with site dephasing, traps, losses and site energies.
#[derive(Clone, Debug)]
pub struct DephasingWalk {
    n: usize,
    /// The graph part of `H`.
    h: CMatrix,
    tol: f64,
    energies: Vec<f64>,
    traps: Vec<(usize, f64)>,
    losses: Vec<f64>,
    dephasing: Vec<f64>,
}

impl DephasingWalk {
    /// Walk on `graph` (1 to 2048 nodes) with hopping rate `gamma` and the
    /// dephasing rate `rate` on every node. Evolution needs at most 512
    /// nodes; transport works up to 2048.
    pub fn new(graph: &Graph, generator: Generator, gamma: f64, rate: f64) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 || n > DENSE_LIMIT {
            return Err(QsoError::invalid(
                "graph",
                format!("need 1..={DENSE_LIMIT} nodes"),
            ));
        }
        if !(gamma.is_finite() && gamma > 0.0) || !(rate.is_finite() && rate >= 0.0) {
            return Err(QsoError::invalid(
                "gamma/rate",
                "need gamma > 0 and rate ≥ 0",
            ));
        }
        let h_real = match generator {
            Generator::Laplacian => graph.laplacian_matrix() * gamma,
            Generator::Adjacency => graph.adjacency_matrix() * -gamma,
        };
        Ok(Self {
            n,
            h: h_real.map(|x| C64::new(x, 0.0)),
            tol: 1e-11,
            energies: vec![0.0; n],
            traps: Vec::new(),
            losses: vec![0.0; n],
            dephasing: vec![rate; n],
        })
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Set the per-step error tolerance (default 1e-11).
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-15);
        self
    }

    /// Site energies `ε_j` added to the diagonal of `H` (not scaled by the
    /// hopping rate); `n` finite values, replacing earlier ones.
    pub fn with_site_energies(mut self, energies: &[f64]) -> Result<Self> {
        check_energies(energies, self.n)?;
        self.energies = energies.to_vec();
        Ok(self)
    }

    /// Traps `(node, κ)`: the sink of node `t` fills at `2κ ρ_tt`. Nodes
    /// distinct, `κ > 0`; replaces earlier traps (`&[]` removes them).
    pub fn with_traps(mut self, traps: &[(usize, f64)]) -> Result<Self> {
        check_traps(traps, self.n)?;
        self.traps = traps.to_vec();
        Ok(self)
    }

    /// Loss rates `Γ_j ≥ 0`, one per node: population leaves node `j` at
    /// `Γ_j ρ_jj`. Replaces earlier losses.
    pub fn with_losses(mut self, rates: &[f64]) -> Result<Self> {
        check_rates("losses", rates, self.n)?;
        self.losses = rates.to_vec();
        Ok(self)
    }

    /// Dephasing rates `γ_j ≥ 0`, one per node, replacing the uniform
    /// `rate` of [`Self::new`] (equal rates give the same walk bit for bit).
    pub fn with_node_dephasing(mut self, rates: &[f64]) -> Result<Self> {
        check_rates("dephasing", rates, self.n)?;
        self.dephasing = rates.to_vec();
        Ok(self)
    }

    /// The site energies (zeros unless set).
    pub fn site_energies(&self) -> &[f64] {
        &self.energies
    }

    /// The traps `(node, κ)` in the order given.
    pub fn traps(&self) -> &[(usize, f64)] {
        &self.traps
    }

    /// The loss rates per node.
    pub fn losses(&self) -> &[f64] {
        &self.losses
    }

    /// The dephasing rates per node.
    pub fn node_dephasing(&self) -> &[f64] {
        &self.dephasing
    }

    /// `k_j = κ_j + Γ_j/2`, the decay of amplitude on node `j`.
    fn decay(&self) -> Vec<f64> {
        let mut k: Vec<f64> = self.losses.iter().map(|g| 0.5 * g).collect();
        for &(t, kappa) in &self.traps {
            k[t] += kappa;
        }
        k
    }

    fn derivative(&self, rho: &CMatrix, decay: &[f64]) -> CMatrix {
        let i = C64::new(0.0, 1.0);
        let comm = &self.h * rho - rho * &self.h;
        let mut d = comm * -i;
        let n = self.n;
        if self.energies.iter().any(|&e| e != 0.0) {
            for c in 0..n {
                for r in 0..n {
                    d[(r, c)] -= rho[(r, c)] * C64::new(0.0, self.energies[r] - self.energies[c]);
                }
            }
        }
        if decay.iter().any(|&k| k != 0.0) {
            for c in 0..n {
                for r in 0..n {
                    d[(r, c)] -= rho[(r, c)] * (decay[r] + decay[c]);
                }
            }
        }
        if self.dephasing.iter().any(|&g| g > 0.0) {
            for r in 0..n {
                for c in 0..n {
                    if r != c {
                        d[(r, c)] -= rho[(r, c)] * (0.5 * (self.dephasing[r] + self.dephasing[c]));
                    }
                }
            }
        }
        d
    }

    /// Derivatives of the sinks (`2κ_t ρ_tt` per trap) and of the loss
    /// (`Σ_j Γ_j ρ_jj`), when `sinks` is set.
    fn sink_derivative(&self, rho: &CMatrix, sinks: bool) -> Vec<f64> {
        if !sinks {
            return Vec::new();
        }
        let mut out: Vec<f64> = self
            .traps
            .iter()
            .map(|&(t, kappa)| 2.0 * kappa * rho[(t, t)].re)
            .collect();
        out.push(
            self.losses
                .iter()
                .enumerate()
                .map(|(j, g)| g * rho[(j, j)].re)
                .sum(),
        );
        out
    }

    /// One RK4 step of `(ρ, s)`, the sinks `s` only when `sinks`.
    fn rk4(
        &self,
        rho: &CMatrix,
        s: &[f64],
        dt: f64,
        decay: &[f64],
        sinks: bool,
    ) -> (CMatrix, Vec<f64>) {
        let k1 = self.derivative(rho, decay);
        let s1 = self.sink_derivative(rho, sinks);
        let r2 = rho + &k1 * C64::new(dt / 2.0, 0.0);
        let k2 = self.derivative(&r2, decay);
        let s2 = self.sink_derivative(&r2, sinks);
        let r3 = rho + &k2 * C64::new(dt / 2.0, 0.0);
        let k3 = self.derivative(&r3, decay);
        let s3 = self.sink_derivative(&r3, sinks);
        let r4 = rho + &k3 * C64::new(dt, 0.0);
        let k4 = self.derivative(&r4, decay);
        let s4 = self.sink_derivative(&r4, sinks);
        let next = rho
            + (k1 + k2 * C64::new(2.0, 0.0) + k3 * C64::new(2.0, 0.0) + k4)
                * C64::new(dt / 6.0, 0.0);
        let snext = (0..s1.len())
            .map(|k| s[k] + (s1[k] + 2.0 * s2[k] + 2.0 * s3[k] + s4[k]) * (dt / 6.0))
            .collect();
        (next, snext)
    }

    /// The step scale `max|h|·n + max|ε| + max γ + 2·max k`.
    fn scale(&self, decay: &[f64]) -> f64 {
        let fold = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        self.h.iter().map(|z| z.norm()).fold(0.0, f64::max) * self.n as f64
            + fold(&self.energies)
            + fold(&self.dephasing)
            + 2.0 * fold(decay)
    }

    /// Refuse evolution beyond 512 nodes.
    fn check_evolution(&self) -> Result<()> {
        if self.n > MAX_EVOLUTION_NODES {
            return Err(QsoError::invalid(
                "graph",
                "density-matrix evolution needs at most 512 nodes; transport works up to 2048",
            ));
        }
        Ok(())
    }

    /// Integrate `(ρ, s)` from `*now` to `t` with step doubling, starting
    /// with the step `*dt` and leaving the next step there.
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        rho: &mut CMatrix,
        s: &mut Vec<f64>,
        now: &mut f64,
        t: f64,
        dt: &mut f64,
        guard: &mut usize,
        decay: &[f64],
        sinks: bool,
    ) -> Result<()> {
        while *now < t {
            *guard += 1;
            if *guard > 10_000_000 {
                return Err(QsoError::NoConvergence {
                    method: "dephasing RK4",
                    iterations: *guard,
                    residual: t - *now,
                });
            }
            let step = dt.min(t - *now);
            let (full, sfull) = self.rk4(rho, s, step, decay, sinks);
            let (mid, smid) = self.rk4(rho, s, step / 2.0, decay, sinks);
            let (half, shalf) = self.rk4(&mid, &smid, step / 2.0, decay, sinks);
            let diff = &half - &full;
            let sdiff: Vec<f64> = shalf.iter().zip(&sfull).map(|(a, b)| a - b).collect();
            let err = diff
                .iter()
                .map(|z| z.norm())
                .chain(sdiff.iter().map(|x| x.abs()))
                .fold(0.0, f64::max)
                / 15.0;
            if err <= self.tol || step < 1e-14 {
                *rho = half + diff * C64::new(1.0 / 15.0, 0.0);
                *s = shalf
                    .iter()
                    .zip(&sdiff)
                    .map(|(a, d)| a + d * (1.0 / 15.0))
                    .collect();
                *now += step;
                let grow = if err > 0.0 {
                    0.9 * (self.tol / err).powf(0.2)
                } else {
                    2.0
                };
                *dt = step * grow.clamp(0.2, 2.0);
            } else {
                *dt = step * (0.9 * (self.tol / err).powf(0.2)).clamp(0.1, 0.9);
            }
        }
        Ok(())
    }

    /// Evolve the density matrix `rho0` for time `t ≥ 0` (at most 512
    /// nodes).
    pub fn evolve(&self, rho0: &CMatrix, t: f64) -> Result<CMatrix> {
        self.check_evolution()?;
        if rho0.nrows() != self.n || rho0.ncols() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: rho0.nrows(),
            });
        }
        if !(t >= 0.0 && t.is_finite()) {
            return Err(QsoError::invalid(
                "t",
                "time must be finite and non-negative",
            ));
        }
        let decay = self.decay();
        let scale = self.scale(&decay);
        let mut dt = (0.1 / scale.max(1e-12)).min(t.max(1e-12));
        let mut rho = rho0.clone();
        let mut s = Vec::new();
        let mut now = 0.0;
        let mut guard = 0usize;
        self.integrate(
            &mut rho, &mut s, &mut now, t, &mut dt, &mut guard, &decay, false,
        )?;
        Ok(rho)
    }

    /// Node populations at time `t` for a walker started at `start` (at
    /// most 512 nodes).
    pub fn populations(&self, start: usize, t: f64) -> Result<Vec<f64>> {
        self.check_evolution()?;
        self.check_start(start)?;
        let mut rho0 = CMatrix::zeros(self.n, self.n);
        rho0[(start, start)] = C64::new(1.0, 0.0);
        let rho = self.evolve(&rho0, t)?;
        Ok((0..self.n).map(|j| rho[(j, j)].re).collect())
    }

    /// Propagated populations, sink populations (per trap, in the order of
    /// [`Self::traps`]) and the population lost at each of `times` (finite,
    /// `≥ 0`, non-decreasing), for a walker started at `start`: RK4 with
    /// step doubling on `ρ`, the sinks and the loss together. At most 512
    /// nodes.
    pub fn transport_trace(&self, start: usize, times: &[f64]) -> Result<Vec<TransportState>> {
        self.check_evolution()?;
        self.check_start(start)?;
        check_times(times)?;
        let decay = self.decay();
        let scale = self.scale(&decay);
        let mut dt = 0.1 / scale.max(1e-12);
        let mut rho = CMatrix::zeros(self.n, self.n);
        rho[(start, start)] = C64::new(1.0, 0.0);
        let mut s = vec![0.0; self.traps.len() + 1];
        let mut now = 0.0;
        let mut guard = 0usize;
        let mut out = Vec::with_capacity(times.len());
        for &t in times {
            self.integrate(
                &mut rho, &mut s, &mut now, t, &mut dt, &mut guard, &decay, true,
            )?;
            out.push(TransportState {
                time: t,
                populations: (0..self.n).map(|j| rho[(j, j)].re).collect(),
                trapped: s[..self.traps.len()].to_vec(),
                lost: s[self.traps.len()],
            });
        }
        Ok(out)
    }

    fn check_start(&self, start: usize) -> Result<()> {
        if start >= self.n {
            return Err(QsoError::NodeOutOfRange {
                node: start,
                n_nodes: self.n,
            });
        }
        Ok(())
    }

    /// The model of the dense transport solver, after checking the options
    /// and that there is a trap.
    fn dense_model(&self, options: &TransportOptions) -> Result<DenseModel> {
        options.validate()?;
        if self.traps.is_empty() {
            return Err(QsoError::invalid(
                "traps",
                "transport needs at least one trap (with_traps)",
            ));
        }
        let mut h = RMatrix::from_fn(self.n, self.n, |i, j| self.h[(i, j)].re);
        for (j, e) in self.energies.iter().enumerate() {
            h[(j, j)] += e;
        }
        Ok(DenseModel {
            h,
            traps: self.traps.clone(),
            losses: self.losses.clone(),
            dephasing: self.dephasing.clone(),
        })
    }

    /// Efficiency, mean arrival time, losses and dark weight of a walker
    /// started at `start` (see [`super::transport`]), by linear solves on
    /// the bright states; needs a trap.
    pub fn transport(&self, start: usize, options: &TransportOptions) -> Result<TransportResult> {
        let model = self.dense_model(options)?;
        self.check_start(start)?;
        Bright::new(&model, false, options)?.solve(&Start::Node(start), 0.0, true, options)
    }

    /// [`Self::transport`] from the density matrix `rho0`, Hermitian with
    /// trace 1 (both to `1e-10·max(1, max|ρ₀|)`); positivity is the
    /// caller's.
    pub fn transport_from(
        &self,
        rho0: &CMatrix,
        options: &TransportOptions,
    ) -> Result<TransportResult> {
        let model = self.dense_model(options)?;
        if rho0.nrows() != self.n || rho0.ncols() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: if rho0.nrows() != self.n {
                    rho0.nrows()
                } else {
                    rho0.ncols()
                },
            });
        }
        let largest = rho0.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let tol = 1e-10 * largest.max(1.0);
        let finite = rho0.iter().all(|z| z.re.is_finite() && z.im.is_finite());
        let hermitian = (0..self.n)
            .all(|i| (0..self.n).all(|j| (rho0[(i, j)] - rho0[(j, i)].conj()).norm() <= tol));
        let trace: C64 = (0..self.n).map(|j| rho0[(j, j)]).sum();
        if !(finite && hermitian && (trace - C64::new(1.0, 0.0)).norm() <= tol) {
            return Err(QsoError::invalid(
                "rho0",
                "the start must be Hermitian with trace 1",
            ));
        }
        Bright::new(&model, false, options)?.solve(&Start::Density(rho0), 0.0, true, options)
    }

    /// One result per rate `e ≥ 0` (in input order), with `e` added to
    /// every node's dephasing. The whole scan needs at most two closures and
    /// Schur forms: one for `e = 0` and one shared by every `e > 0` (a
    /// uniform rate shifts the effective Hamiltonian of the bright states by
    /// `−(ie/2)I`).
    pub fn dephasing_scan(
        &self,
        start: usize,
        rates: &[f64],
        options: &TransportOptions,
    ) -> Result<Vec<TransportResult>> {
        let model = self.dense_model(options)?;
        self.check_start(start)?;
        check_scan_rates(rates)?;
        let own_everywhere = self.dephasing.iter().all(|&g| g > 0.0);
        let mut budget = Budget::default();
        let mut plus: Option<Bright> = None;
        let mut zero: Option<Bright> = None;
        let mut out = Vec::with_capacity(rates.len());
        for &e in rates {
            let bright = if e > 0.0 || own_everywhere {
                if plus.is_none() {
                    plus = Some(Bright::new(&model, true, options)?);
                }
                plus.as_ref()
            } else {
                if zero.is_none() {
                    zero = Some(Bright::new(&model, false, options)?);
                }
                zero.as_ref()
            };
            let bright = bright.expect("built above");
            out.push(bright.solve_within(&Start::Node(start), e, true, options, &mut budget)?);
        }
        Ok(out)
    }

    /// The uniform rate in `[low, high]` (`0 < low < high`, finite) which,
    /// added to every node's dephasing, gives the highest efficiency: a
    /// log-spaced grid of at least 13 rates, then golden section on `ln e`,
    /// all on one closure and Schur form. Only positive rates are searched;
    /// with dark states the efficiency can jump at `e = 0`.
    pub fn optimal_dephasing(
        &self,
        start: usize,
        low: f64,
        high: f64,
        options: &TransportOptions,
    ) -> Result<DephasingOptimum> {
        let model = self.dense_model(options)?;
        self.check_start(start)?;
        if !(low.is_finite() && high.is_finite() && low > 0.0 && low < high) {
            return Err(QsoError::invalid(
                "range",
                "need 0 < low < high, both finite",
            ));
        }
        let bright = Bright::new(&model, true, options)?;
        let s = Start::Node(start);
        // One work limit for the whole optimisation.
        let budget = std::cell::RefCell::new(Budget::default());
        optimise(
            low,
            high,
            |e| {
                let mut b = budget.borrow_mut();
                Ok(bright
                    .solve_within(&s, e, false, options, &mut b)?
                    .efficiency)
            },
            |e| bright.solve_within(&s, e, true, options, &mut budget.borrow_mut()),
        )
    }

    /// Orthonormal basis (`n × (n − r)`) of the states that never reach a
    /// trap or a node with more than the smallest loss: the complement of
    /// the bright space of [`Self::transport`]. Needs a trap.
    pub fn dark_states(&self, options: &TransportOptions) -> Result<RMatrix> {
        let model = self.dense_model(options)?;
        Ok(Closure::new(&model, false, options.dark_tolerance).complement(self.n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{generators, metatron};
    use crate::walks::{ContinuousWalk, classical_walk_probabilities};

    #[test]
    fn zero_dephasing_reproduces_coherent_walk() {
        let g = metatron::fruit_of_life();
        let open = DephasingWalk::new(&g, Generator::Laplacian, 1.0, 0.0).unwrap();
        let closed = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        let a = open.populations(0, 2.7).unwrap();
        let b = closed.probabilities(0, 2.7).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn dephasing_preserves_trace_and_hermiticity_and_mixes_to_uniform() {
        let g = metatron::fruit_of_life();
        let walk = DephasingWalk::new(&g, Generator::Laplacian, 1.0, 0.5).unwrap();
        let mut rho0 = CMatrix::zeros(13, 13);
        rho0[(7, 7)] = C64::new(1.0, 0.0);
        let rho = walk.evolve(&rho0, 3.0).unwrap();
        let trace: C64 = (0..13).map(|j| rho[(j, j)]).sum();
        assert!((trace - C64::new(1.0, 0.0)).norm() < 1e-9);
        assert!((&rho - rho.adjoint()).norm() < 1e-9);
        let p = walk.populations(7, 400.0).unwrap();
        assert!(p.iter().all(|&x| (x - 1.0 / 13.0).abs() < 1e-4), "{p:?}");
    }

    #[test]
    fn strong_dephasing_approaches_classical_walk() {
        // Zeno regime: populations follow dp/dt = −(2γ²/κ)Lp.
        let g = generators::cycle(6);
        let (gamma, kappa) = (1.0, 80.0);
        let walk = DephasingWalk::new(&g, Generator::Laplacian, gamma, kappa)
            .unwrap()
            .with_tolerance(1e-9);
        let t = 30.0;
        let q = walk.populations(0, t).unwrap();
        let c = classical_walk_probabilities(&g, 0, t, 2.0 * gamma * gamma / kappa).unwrap();
        let tv = crate::walks::total_variation(&q, &c);
        assert!(tv < 0.02, "tv = {tv}");
    }
}
