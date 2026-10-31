//! Slow noise, averaged exactly.
//!
//! A frequency that wanders slowly (charge noise, 1/f flux noise, a
//! fluctuating field) dephases a qubit in ways no Markovian channel
//! describes: echoes refocus part of it, and noise from a few fluctuators
//! leaves plateaus and beats that a Gaussian model misses. Trajectories
//! ([`TrajectorySimulator`](super::TrajectorySimulator)) sample such noise
//! and carry a statistical error; here it is averaged exactly.
//!
//! * [`MarkovProcess`]: a classical process with finitely many states, a
//!   frequency shift `ν_x` in each, switching rates and a distribution at
//!   time 0. Random telegraph noise, Ornstein–Uhlenbeck noise as the limit
//!   of a birth–death chain (the Ehrenfest model of `K` fluctuators),
//!   quasi-static noise on the nodes of a Gauss rule, and products of
//!   independent processes.
//! * Pure dephasing under ideal π pulses: the coherence
//!   `⟨exp(−i ∫ f ν dt)⟩ = 1ᵀ ∏ exp(Δ_k (W − i f_k diag ν)) p₀`, with
//!   `f = ±1` flipping at each pulse ([`MarkovProcess::dephasing`]);
//!   independent fluctuators multiply ([`Fluctuators`], 1/f noise from
//!   switching rates spread evenly in their logarithm).
//! * [`StochasticLiouville`]: Kubo's stochastic Liouville equation (J.
//!   Math. Phys. 4, 174 (1963)), one density matrix per joint state of the
//!   processes, `dρ_x/dt = −i[H + V_x, ρ_x] + D(ρ_x) + Σ_y W_xy ρ_y`,
//!   propagated by Krylov steps.
//! * Circuits: [`DensityMatrix::apply_device`] averages the telegraph
//!   noise of a device exactly (one density matrix per joint state of its
//!   telegraph processes) and its quasi-static detunings and over-rotation
//!   spreads by Gauss quadrature, with as many nodes as an a-priori bound
//!   on the error needs.
//!
//! ```
//! use metatron_qso::noise::slow::MarkovProcess;
//!
//! // Free induction under random telegraph noise (±1, flipping at rate
//! // 0.2): the closed form e^{−γt}(cos μt + γ sin(μt)/μ), μ = √(1 − γ²).
//! let process = MarkovProcess::telegraph(1.0, 0.2).unwrap();
//! let t = 3.0;
//! let coherence = process.dephasing(&[], t).unwrap();
//! let mu = (1.0f64 - 0.04).sqrt();
//! let expected = (-0.2 * t).exp() * ((mu * t).cos() + 0.2 * (mu * t).sin() / mu);
//! assert!((coherence.re - expected).abs() < 1e-12 && coherence.im.abs() < 1e-12);
//! ```

use num_complex::Complex64 as C64;

use super::DensityMatrix;
use super::device::{Op, Program, Spread};
use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, RMatrix, eigh_real};
use crate::parallel::map_tasks;
use crate::quantum::{Instruction, Param, kernels};
use crate::spin::krylov::{SparseBlock, expv, krylov_work};

/// Most states of a [`MarkovProcess`], and most joint states of the
/// processes of a [`StochasticLiouville`] equation.
pub const MAX_PROCESS_STATES: usize = 1 << 20;

/// Most nodes of a Gauss rule ([`MarkovProcess::quasi_static`], and per
/// quasi-static parameter of a circuit).
pub const MAX_QUADRATURE_NODES: usize = 1024;

/// Most fluctuators of [`Fluctuators`].
pub const MAX_FLUCTUATORS: usize = 1 << 20;

/// Most switching rates of a [`MarkovProcess`] (24 bytes each).
pub const MAX_PROCESS_RATES: usize = 1 << 24;

/// Most operations (estimated) of one propagation of a process by Krylov
/// steps.
pub const MAX_PROCESS_WORK: f64 = 1e11;

/// Most coefficients `n·d²` of a [`StochasticLiouville`] equation (its
/// Krylov steps hold 34 vectors of them, 16 bytes each: 1.1 GB).
pub const MAX_LIOUVILLE_COEFFICIENTS: usize = 1 << 21;

/// Most operations (estimated) of one propagation of a
/// [`StochasticLiouville`] equation.
pub const MAX_LIOUVILLE_WORK: f64 = 1e12;

/// Most entries of the generator of a [`StochasticLiouville`] equation
/// (an upper bound is checked before it is built; 20 bytes each).
pub const MAX_LIOUVILLE_ENTRIES: usize = 1 << 25;

/// Most density-matrix entries a circuit with slow noise holds at once
/// (16 bytes each, 1 GiB): the joint states and the sum of every group of
/// quadrature points in flight, and the start. Slow noise on density
/// matrices therefore reaches 12 qubits.
pub const MAX_SLOW_ENTRIES: usize = 1 << 26;

/// Most operations (passes over density-matrix entries: two per gate,
/// `4^k` per channel on `k` qubits, a few per phase step; summed over the
/// joint states and quadrature points) of a circuit with slow noise.
pub const MAX_SLOW_WORK: f64 = 2e11;

/// Truncation error of the quadrature over the quasi-static parameters of
/// a circuit, bounded a priori for every entry of the density matrix (the
/// rounding of the rules adds up to about 10⁻¹³ for the largest).
pub const QUADRATURE_ERROR: f64 = 1e-13;

/// Absolute error of a Krylov step in the 2-norm (states and probability
/// vectors have norm at most 1). The rounding of a stiff generator adds
/// about `10⁻¹⁶ ‖W‖ t`.
const KRYLOV_TOL: f64 = 1e-13;

const ZERO: C64 = C64::new(0.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

fn invalid(what: &'static str, message: impl Into<String>) -> QsoError {
    QsoError::invalid(what, message)
}

/// A classical Markov process with finitely many states: a frequency shift
/// `values[x]` (angular frequency) in state `x`, switching from state `x`
/// to state `y` at a rate, and a distribution `p₀` at time 0. Its
/// generator `W` has `W[y][x]` the rate from `x` to `y` and
/// `W[x][x] = −Σ_y W[y][x]`, so that probabilities evolve as
/// `dp/dt = W p`.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovProcess {
    values: Vec<f64>,
    /// `(from, to, rate)`, positive rates between different states.
    rates: Vec<(usize, usize, f64)>,
    initial: Vec<f64>,
    kind: Kind,
}

/// What a process is known to be, for its closed forms.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    General,
    /// The Ehrenfest chain of `fluctuators` telegraph processes of
    /// `amplitude` flipping at `rate`, started stationary.
    Ehrenfest {
        amplitude: f64,
        rate: f64,
        fluctuators: usize,
    },
}

impl MarkovProcess {
    /// The process with `values` (one per state), switching `rates`
    /// `(from, to, rate)` and `initial` probabilities (adding up to 1).
    pub fn new(
        values: Vec<f64>,
        rates: Vec<(usize, usize, f64)>,
        initial: Vec<f64>,
    ) -> Result<Self> {
        let n = values.len();
        if !(1..=MAX_PROCESS_STATES).contains(&n) {
            return Err(invalid(
                "values",
                format!("a process has 1 to {MAX_PROCESS_STATES} states"),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("values", "the values must be finite"));
        }
        if rates.len() > MAX_PROCESS_RATES {
            return Err(invalid(
                "rates",
                format!("at most {MAX_PROCESS_RATES} rates"),
            ));
        }
        for &(from, to, rate) in &rates {
            if from >= n || to >= n || from == to {
                return Err(invalid(
                    "rates",
                    format!("a rate from {from} to {to} between {n} states"),
                ));
            }
            if !(rate.is_finite() && rate >= 0.0) {
                return Err(invalid("rates", "rates must be finite and non-negative"));
            }
        }
        if initial.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: initial.len(),
            });
        }
        if initial.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
            return Err(invalid(
                "initial",
                "probabilities must be finite and non-negative",
            ));
        }
        let total: f64 = initial.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(invalid("initial", "the probabilities must add up to 1"));
        }
        Ok(MarkovProcess {
            values,
            rates: rates.into_iter().filter(|r| r.2 > 0.0).collect(),
            initial: initial.iter().map(|p| p / total).collect(),
            kind: Kind::General,
        })
    }

    /// Random telegraph noise: `±amplitude`, flipping at `rate` in either
    /// direction and starting in either state with probability ½ (the
    /// stationary distribution), as [`Telegraph`](super::Telegraph) on a
    /// device. Its correlation is `amplitude² e^{−2·rate·|τ|}`.
    pub fn telegraph(amplitude: f64, rate: f64) -> Result<Self> {
        if !(amplitude.is_finite() && rate.is_finite() && rate >= 0.0) {
            return Err(invalid(
                "telegraph",
                "a finite amplitude and a finite, non-negative rate",
            ));
        }
        Self::new(
            vec![amplitude, -amplitude],
            vec![(0, 1, rate), (1, 0, rate)],
            vec![0.5, 0.5],
        )
    }

    /// Ornstein–Uhlenbeck noise of standard deviation `sigma` and
    /// correlation time `correlation_time`, `⟨ν(0)ν(τ)⟩ = σ² e^{−|τ|/τ_c}`,
    /// as the Ehrenfest chain of `K = levels − 1` telegraph fluctuators of
    /// amplitude `σ/√K` and rate `1/(2τ_c)`: level `m` has the value
    /// `(2m − K)σ/√K`, climbs at rate `(K − m)/(2τ_c)`, falls at rate
    /// `m/(2τ_c)` and starts binomially distributed (stationary). The
    /// correlation is exactly exponential at every `K`; the noise becomes
    /// Gaussian as the levels grow, and its dephasing approaches Kubo's
    /// form with an error of order `1/K`.
    pub fn ornstein_uhlenbeck(sigma: f64, correlation_time: f64, levels: usize) -> Result<Self> {
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(invalid("sigma", "must be finite and non-negative"));
        }
        if !(correlation_time.is_finite() && correlation_time > 0.0) {
            return Err(invalid("correlation_time", "must be finite and positive"));
        }
        if !(2..=MAX_PROCESS_STATES).contains(&levels) {
            return Err(invalid(
                "levels",
                format!("2 to {MAX_PROCESS_STATES} levels"),
            ));
        }
        let k = levels - 1;
        let kf = k as f64;
        let rate = 0.5 / correlation_time;
        let step = sigma / kf.sqrt();
        let values = (0..levels).map(|m| (2.0 * m as f64 - kf) * step).collect();
        let mut rates = Vec::with_capacity(2 * k);
        for m in 0..k {
            rates.push((m, m + 1, (k - m) as f64 * rate));
            rates.push((m + 1, m, (m + 1) as f64 * rate));
        }
        // Binomial(K, ½) in logarithms (2^{−K} underflows for K > 1074).
        let mut log_c = 0.0f64;
        let mut initial = Vec::with_capacity(levels);
        for m in 0..levels {
            initial.push((log_c - kf * std::f64::consts::LN_2).exp());
            if m < k {
                log_c += ((k - m) as f64).ln() - ((m + 1) as f64).ln();
            }
        }
        let total: f64 = initial.iter().sum();
        for p in &mut initial {
            *p /= total;
        }
        let mut process = Self::new(values, rates, initial)?;
        process.kind = Kind::Ehrenfest {
            amplitude: step,
            rate,
            fluctuators: k,
        };
        Ok(process)
    }

    /// Quasi-static noise: a constant drawn from `spread`, on the `nodes`
    /// points of its Gauss rule (Hermite for a normal, Legendre for a
    /// uniform distribution) with their weights as probabilities. The rule
    /// averages functions whose derivatives grow at most like
    /// `|f^{(2m)}| ≤ (width · T)^{2m}` (a phase `ν t` with `t ≤ T`) to
    /// within `m!/(2m)! (σT)^{2m}` (normal) or
    /// `2^{2m}(m!)⁴/((2m + 1)((2m)!)³) (hT)^{2m}` (uniform).
    pub fn quasi_static(spread: Spread, nodes: usize) -> Result<Self> {
        check_spread(&spread)?;
        if !(1..=MAX_QUADRATURE_NODES).contains(&nodes) {
            return Err(invalid(
                "nodes",
                format!("1 to {MAX_QUADRATURE_NODES} nodes"),
            ));
        }
        let (values, weights) = gauss_rule(&spread, nodes);
        Self::new(values, Vec::new(), weights)
    }

    /// Two independent processes together: their values add, and state
    /// `(i, j)` is numbered `i·n₂ + j`.
    pub fn product(&self, other: &MarkovProcess) -> Result<Self> {
        let (n1, n2) = (self.num_states(), other.num_states());
        let n = n1
            .checked_mul(n2)
            .filter(|&n| n <= MAX_PROCESS_STATES)
            .ok_or_else(|| {
                invalid(
                    "processes",
                    format!("more than {MAX_PROCESS_STATES} joint states"),
                )
            })?;
        let count = self
            .rates
            .len()
            .checked_mul(n2)
            .zip(other.rates.len().checked_mul(n1))
            .and_then(|(a, b)| a.checked_add(b))
            .filter(|&c| c <= MAX_PROCESS_RATES)
            .ok_or_else(|| {
                invalid(
                    "processes",
                    format!("more than {MAX_PROCESS_RATES} joint rates"),
                )
            })?;
        let mut values = Vec::with_capacity(n);
        let mut initial = Vec::with_capacity(n);
        for i in 0..n1 {
            for j in 0..n2 {
                values.push(self.values[i] + other.values[j]);
                initial.push(self.initial[i] * other.initial[j]);
            }
        }
        let mut rates = Vec::with_capacity(count);
        for &(from, to, rate) in &self.rates {
            for j in 0..n2 {
                rates.push((from * n2 + j, to * n2 + j, rate));
            }
        }
        for &(from, to, rate) in &other.rates {
            for i in 0..n1 {
                rates.push((i * n2 + from, i * n2 + to, rate));
            }
        }
        let total: f64 = initial.iter().sum();
        let initial = initial.iter().map(|p| p / total).collect();
        Self::new(values, rates, initial)
    }

    pub fn num_states(&self) -> usize {
        self.values.len()
    }

    /// The value (frequency shift) of every state.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The switching rates `(from, to, rate)`.
    pub fn rates(&self) -> &[(usize, usize, f64)] {
        &self.rates
    }

    /// The distribution at time 0.
    pub fn initial(&self) -> &[f64] {
        &self.initial
    }

    /// The mean value `⟨ν(t)⟩`.
    pub fn mean(&self, t: f64) -> Result<f64> {
        check_time(t)?;
        if let Kind::Ehrenfest { .. } = self.kind {
            return Ok(0.0);
        }
        let p = self.propagate(&self.initial_vector(), t, 0.0)?;
        Ok(self.values.iter().zip(&p).map(|(v, x)| v * x.re).sum())
    }

    /// `⟨ν(0) ν(τ)⟩` for the distribution at time 0 (for a stationary one,
    /// the correlation function).
    pub fn correlation(&self, tau: f64) -> Result<f64> {
        check_time(tau)?;
        if let Kind::Ehrenfest {
            amplitude,
            rate,
            fluctuators,
        } = self.kind
        {
            return Ok(fluctuators as f64 * amplitude * amplitude * (-2.0 * rate * tau).exp());
        }
        let start: Vec<C64> = self
            .values
            .iter()
            .zip(&self.initial)
            .map(|(v, p)| C64::new(v * p, 0.0))
            .collect();
        let p = self.propagate(&start, tau, 0.0)?;
        Ok(self.values.iter().zip(&p).map(|(v, x)| v * x.re).sum())
    }

    /// The coherence of a qubit whose frequency is shifted by the process,
    /// `⟨exp(−i ∫₀ᵗ f(s) ν(s) ds)⟩`, with `f = +1` flipping its sign at
    /// every time in `pulses` (ideal π pulses, ascending, within `[0, t]`).
    /// Free induction for no pulses, a Hahn echo for one pulse at `t/2`.
    pub fn dephasing(&self, pulses: &[f64], t: f64) -> Result<C64> {
        let periods = periods(pulses, t)?;
        if self.rates.is_empty() {
            // Static noise: each value turns by its own phase.
            let turn: f64 = periods.iter().map(|(duration, sign)| duration * sign).sum();
            return Ok(self
                .values
                .iter()
                .zip(&self.initial)
                .map(|(v, p)| C64::from_polar(*p, -v * turn))
                .sum());
        }
        if let Some((amplitude, shift, rate)) = self.symmetric_telegraph() {
            return Ok(telegraph_dephasing(amplitude, shift, rate, &periods));
        }
        if let Kind::Ehrenfest {
            amplitude,
            rate,
            fluctuators,
        } = self.kind
        {
            // The chain is K independent fluctuators: the K-th power of one
            // closed-form factor.
            let one = telegraph_dephasing(amplitude, 0.0, rate, &periods);
            return Ok(one.powu(fluctuators as u32));
        }
        // One generator per sign of f, built once.
        let (plus, minus) = (self.generator(1.0), self.generator(-1.0));
        check_krylov(&plus, t)?;
        let mut p = self.initial_vector();
        for (duration, sign) in periods {
            if duration > 0.0 {
                let block = if sign > 0.0 { &plus } else { &minus };
                p = expv(block, &p, duration, KRYLOV_TOL, 0.0)?;
            }
        }
        Ok(p.iter().sum())
    }

    /// `(amplitude, shift, rate)` if the process is a telegraph process
    /// with equal rates both ways and values `shift ± amplitude`, started
    /// in its stationary distribution.
    fn symmetric_telegraph(&self) -> Option<(f64, f64, f64)> {
        if self.num_states() != 2 || self.initial != [0.5, 0.5] {
            return None;
        }
        let (mut up, mut down) = (0.0, 0.0);
        for &(from, _, rate) in &self.rates {
            if from == 0 {
                up += rate;
            } else {
                down += rate;
            }
        }
        (up == down).then(|| {
            (
                0.5 * (self.values[0] - self.values[1]),
                0.5 * (self.values[0] + self.values[1]),
                up,
            )
        })
    }

    fn initial_vector(&self) -> Vec<C64> {
        self.initial.iter().map(|&p| C64::new(p, 0.0)).collect()
    }

    /// `W − i·sign·diag ν` as a sparse matrix.
    fn generator(&self, sign: f64) -> SparseBlock {
        let n = self.num_states();
        let mut triplets: Vec<(u32, u32, C64)> = Vec::with_capacity(2 * self.rates.len() + n);
        for (x, &v) in self.values.iter().enumerate() {
            if sign != 0.0 && v != 0.0 {
                triplets.push((x as u32, x as u32, C64::new(0.0, -sign * v)));
            }
        }
        for &(from, to, rate) in &self.rates {
            triplets.push((to as u32, from as u32, C64::new(rate, 0.0)));
            triplets.push((from as u32, from as u32, C64::new(-rate, 0.0)));
        }
        SparseBlock::from_triplets(n, triplets)
    }

    /// `exp(t (W − i·sign·diag ν)) p`.
    fn propagate(&self, p: &[C64], t: f64, sign: f64) -> Result<Vec<C64>> {
        if t == 0.0 {
            return Ok(p.to_vec());
        }
        let block = self.generator(sign);
        check_krylov(&block, t)?;
        expv(&block, p, t, KRYLOV_TOL, 0.0)
    }
}

/// Refuse a Krylov propagation of a process beyond [`MAX_PROCESS_WORK`].
fn check_krylov(block: &SparseBlock, t: f64) -> Result<()> {
    let work = krylov_work(block.n, block.nnz(), block.norm_inf(), t);
    if work.is_nan() || work > MAX_PROCESS_WORK {
        return Err(invalid(
            "process",
            format!(
                "Krylov steps over ‖W‖t = {:.1e} would take about {work:.1e} operations \
                 (limit {MAX_PROCESS_WORK:.0e})",
                block.norm_inf() * t
            ),
        ));
    }
    Ok(())
}

/// The coherence under a symmetric telegraph process (values `shift ±
/// amplitude`, started stationary) over the periods, by the closed form
/// of each.
fn telegraph_dephasing(amplitude: f64, shift: f64, rate: f64, periods: &[(f64, f64)]) -> C64 {
    let mut p = [C64::new(0.5, 0.0), C64::new(0.5, 0.0)];
    for &(duration, sign) in periods {
        // W − i f diag(shift ± a): the propagator's sign is −f.
        let m = telegraph_propagator(amplitude, -sign * shift, rate, -sign, duration);
        p = [
            m[0][0] * p[0] + m[0][1] * p[1],
            m[1][0] * p[0] + m[1][1] * p[1],
        ];
    }
    p[0] + p[1]
}

/// The periods between `0`, the pulses and `t`, with the sign of `f`.
fn periods(pulses: &[f64], t: f64) -> Result<Vec<(f64, f64)>> {
    check_time(t)?;
    let mut out = Vec::with_capacity(pulses.len() + 1);
    let (mut last, mut sign) = (0.0, 1.0);
    for &p in pulses {
        if !(p.is_finite() && p >= last && p <= t) {
            return Err(invalid(
                "pulses",
                "pulse times must be finite, ascending and within [0, t]",
            ));
        }
        out.push((p - last, sign));
        (last, sign) = (p, -sign);
    }
    out.push((t - last, sign));
    Ok(out)
}

fn check_time(t: f64) -> Result<()> {
    if !(t.is_finite() && t >= 0.0) {
        return Err(invalid("t", "times must be finite and non-negative"));
    }
    Ok(())
}

fn check_spread(spread: &Spread) -> Result<()> {
    let w = width(spread);
    if !(w.is_finite() && w >= 0.0) {
        return Err(invalid(
            "spread",
            "the width must be finite and non-negative",
        ));
    }
    Ok(())
}

/// The standard deviation of a normal spread, the half-width of a uniform
/// one.
fn width(spread: &Spread) -> f64 {
    match *spread {
        Spread::Gaussian { sigma } => sigma,
        Spread::Uniform { half_width } => half_width,
    }
}

/// `exp(τ (W + i·s·diag(a + shift, −a + shift)))` of a telegraph process
/// (values `shift ± a`, flipping at `rate` both ways), for the sign `s`
/// of the phase (`−1` for the coherence `e^{−iφ}`, `±1` or 0 for the
/// elements of a density matrix), in closed form:
/// `e^{i·shift·τ}[C·I + S·((i s a τ, γτ), (γτ, −i s a τ))]` with
/// `μ² = (γτ)² − (saτ)²`, `C = e^{−γτ} cosh μ` and `S = e^{−γτ} sinh μ / μ`.
/// `μ` is real or imaginary and is found in real arithmetic, scaled, as
/// `√((γτ − |x|)(γτ + |x|))`; `C` and `S` are written with `e^{μ−γτ}` and
/// `e^{−μ−γτ}`, `μ − γτ = −x²/(μ + γτ)` without cancellation, so that
/// neither fast switching nor long or tiny times overflow or lose the
/// result.
fn telegraph_propagator(a: f64, shift: f64, rate: f64, s: f64, tau: f64) -> [[C64; 2]; 2] {
    // Here `shift` carries the sign already: the common phase is e^{i shift τ}.
    let g = rate * tau;
    let x = s * a * tau;
    let ax = x.abs();
    let scale = g.max(ax);
    let (product, mu) = if scale > 0.0 {
        let p = ((g - ax) / scale) * ((g + ax) / scale);
        (p, scale * p.abs().sqrt())
    } else {
        (0.0, 0.0)
    };
    let decay = (-g).exp();
    let (c, sh) = if product >= 0.0 {
        // μ real, 0 ≤ μ ≤ γτ: e^{μ−γτ} with μ − γτ = −x²/(μ + γτ).
        let up = if mu + g > 0.0 {
            -ax * (ax / (mu + g))
        } else {
            0.0
        };
        let (eu, ed) = (up.exp(), (-mu - g).exp());
        let sh = if mu > 1e-3 {
            (eu - ed) / (2.0 * mu)
        } else {
            let m2 = mu * mu;
            decay * (1.0 + m2 / 6.0 + m2 * m2 / 120.0 + m2 * m2 * m2 / 5040.0)
        };
        (0.5 * (eu + ed), sh)
    } else {
        // μ = i w: C = e^{−γτ} cos w, S = e^{−γτ} sin(w)/w.
        let w = mu;
        let sinc = if w > 1e-3 {
            w.sin() / w
        } else {
            let w2 = w * w;
            1.0 - w2 / 6.0 + w2 * w2 / 120.0 - w2 * w2 * w2 / 5040.0
        };
        (decay * w.cos(), decay * sinc)
    };
    let (c, sh) = (C64::new(c, 0.0), C64::new(sh, 0.0));
    let phase = C64::from_polar(1.0, shift * tau);
    [
        [phase * (c + sh * I * x), phase * sh * g],
        [phase * sh * g, phase * (c - sh * I * x)],
    ]
}

/// Nodes and weights (adding up to 1) of the Gauss rule with `m` nodes
/// for the distribution of `spread`, by Golub and Welsch: the eigenvalues
/// of the Jacobi matrix of the orthogonal polynomials (probabilists'
/// Hermite, `β_k = √k`; Legendre, `β_k = k/√(4k² − 1)`) and the squared
/// first components of its eigenvectors, made symmetric.
fn gauss_rule(spread: &Spread, m: usize) -> (Vec<f64>, Vec<f64>) {
    let beta = |k: usize| -> f64 {
        let k = k as f64;
        match spread {
            Spread::Gaussian { .. } => k.sqrt(),
            Spread::Uniform { .. } => k / (4.0 * k * k - 1.0).sqrt(),
        }
    };
    let mut jacobi = RMatrix::zeros(m, m);
    for k in 1..m {
        jacobi[(k - 1, k)] = beta(k);
        jacobi[(k, k - 1)] = beta(k);
    }
    let (values, vectors) = eigh_real(&jacobi);
    let scale = width(spread);
    let mut nodes = Vec::with_capacity(m);
    let mut weights = Vec::with_capacity(m);
    for i in 0..m {
        let j = m - 1 - i;
        nodes.push(0.5 * (values[i] - values[j]) * scale);
        weights.push(0.5 * (vectors[(0, i)].powi(2) + vectors[(0, j)].powi(2)));
    }
    let total: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }
    (nodes, weights)
}

/// `ln n!`.
fn ln_factorial(n: usize) -> f64 {
    (2..=n).map(|k| (k as f64).ln()).sum()
}

/// Fewest nodes of the Gauss rule of `spread` whose error is below `eps`
/// for functions with `|f^{(2m)}| ≤ (width · reach)^{2m}`, by the error
/// terms of the Hermite and Legendre rules (Abramowitz and Stegun 25.4.30
/// and 25.4.46, normalised to probabilities, for real and imaginary parts:
/// a factor √2). `None` beyond [`MAX_QUADRATURE_NODES`].
fn nodes_needed(spread: &Spread, reach: f64, eps: f64) -> Option<usize> {
    let c = width(spread) * reach;
    if c == 0.0 {
        return Some(1);
    }
    let target = eps.ln() - 0.5 * std::f64::consts::LN_2;
    (1..=MAX_QUADRATURE_NODES).find(|&m| {
        let mf = m as f64;
        let bound = match spread {
            Spread::Gaussian { .. } => ln_factorial(m) - ln_factorial(2 * m) + 2.0 * mf * c.ln(),
            Spread::Uniform { .. } => {
                2.0 * mf * std::f64::consts::LN_2 + 4.0 * ln_factorial(m)
                    - (2.0 * mf + 1.0).ln()
                    - 3.0 * ln_factorial(2 * m)
                    + 2.0 * mf * c.ln()
            }
        };
        bound <= target
    })
}

/// Independent processes whose values add up: their dephasing is the
/// product of their factors, for any number of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Fluctuators {
    processes: Vec<MarkovProcess>,
}

impl Fluctuators {
    /// The sum of independent `processes` (at least one).
    pub fn new(processes: Vec<MarkovProcess>) -> Result<Self> {
        if !(1..=MAX_FLUCTUATORS).contains(&processes.len()) {
            return Err(invalid(
                "processes",
                format!("1 to {MAX_FLUCTUATORS} processes"),
            ));
        }
        Ok(Fluctuators { processes })
    }

    /// 1/f noise: `count` telegraph fluctuators of `amplitude`, their rates
    /// spread evenly in their logarithm from `lowest` to `highest`. Their
    /// correlations `amplitude² e^{−2γ_k|τ|}` add up to a spectrum that falls
    /// as `1/ω` between `2·lowest` and `2·highest` (Paladino, Galperin,
    /// Falci and Altshuler, Rev. Mod. Phys. 86, 361 (2014)); few strong
    /// fluctuators leave the non-Gaussian features a Gaussian model misses.
    pub fn one_over_f(amplitude: f64, lowest: f64, highest: f64, count: usize) -> Result<Self> {
        if !(lowest.is_finite() && highest.is_finite() && lowest > 0.0 && highest >= lowest) {
            return Err(invalid("rates", "need 0 < lowest ≤ highest, both finite"));
        }
        if !(1..=MAX_FLUCTUATORS).contains(&count) {
            return Err(invalid(
                "count",
                format!("1 to {MAX_FLUCTUATORS} fluctuators"),
            ));
        }
        let processes = (0..count)
            .map(|k| {
                let x = if count == 1 {
                    0.5
                } else {
                    k as f64 / (count - 1) as f64
                };
                MarkovProcess::telegraph(amplitude, lowest * (highest / lowest).powf(x))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(processes)
    }

    pub fn processes(&self) -> &[MarkovProcess] {
        &self.processes
    }

    /// The coherence under all of them, the product of their
    /// [`dephasing`](MarkovProcess::dephasing) factors.
    pub fn dephasing(&self, pulses: &[f64], t: f64) -> Result<C64> {
        let mut out = C64::new(1.0, 0.0);
        for p in &self.processes {
            out *= p.dephasing(pulses, t)?;
        }
        Ok(out)
    }

    /// `⟨ν(0)ν(τ)⟩` of the sum: `Σ_k (C_k(τ) − m_k(0)m_k(τ)) + M(0)M(τ)`
    /// with `m_k` the means and `M` their sum.
    pub fn correlation(&self, tau: f64) -> Result<f64> {
        let (mut sum, mut m0, mut mt) = (0.0, 0.0, 0.0);
        for p in &self.processes {
            let (a, b) = (p.mean(0.0)?, p.mean(tau)?);
            sum += p.correlation(tau)? - a * b;
            m0 += a;
            mt += b;
        }
        Ok(sum + m0 * mt)
    }

    /// The joint process of all of them (at most [`MAX_PROCESS_STATES`]
    /// states), for a [`StochasticLiouville`] equation.
    pub fn combined(&self) -> Result<MarkovProcess> {
        let mut out = self.processes[0].clone();
        for p in &self.processes[1..] {
            out = out.product(p)?;
        }
        Ok(out)
    }
}

/// Whether `m` is a Hermitian `d × d` matrix (to rounding).
fn check_hermitian(what: &'static str, m: &CMatrix, d: usize) -> Result<()> {
    if m.nrows() != d || m.ncols() != d {
        return Err(QsoError::DimensionMismatch {
            expected: d,
            actual: m.nrows(),
        });
    }
    let scale = m.iter().fold(1.0f64, |a, z| a.max(z.norm()));
    if m.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
        return Err(invalid(what, "entries must be finite"));
    }
    for r in 0..d {
        for c in 0..=r {
            if (m[(r, c)] - m[(c, r)].conj()).norm() > 1e-12 * scale {
                return Err(invalid(what, "must be Hermitian"));
            }
        }
    }
    Ok(())
}

/// Kubo's stochastic Liouville equation for a system of dimension `d`
/// under classical noise: independent [`MarkovProcess`]es `k` whose values
/// `ν_k` couple through Hermitian operators `B_k`, so that in the joint
/// state `x` the Hamiltonian is `H + Σ_k ν_k(x_k) B_k`, with Lindblad
/// operators `L_j` (rates included) besides. One density matrix per joint
/// state evolves as
/// `dρ_x/dt = −i[H_x, ρ_x] + Σ_j (L_j ρ_x L_j† − ½{L_j†L_j, ρ_x}) + Σ_y W_xy ρ_y`
/// and the state of the system is `ρ = Σ_x ρ_x` (Kubo, J. Math. Phys. 4,
/// 174 (1963)). Joint states are numbered with the first process varying
/// slowest.
#[derive(Clone, Debug)]
pub struct StochasticLiouville {
    d: usize,
    processes: Vec<(MarkovProcess, CMatrix)>,
    dissipators: Vec<CMatrix>,
    joint: usize,
    /// Per process: the number of joint states between two of its states.
    strides: Vec<usize>,
}

/// The density matrices of a [`StochasticLiouville`] equation, one per
/// joint state of its processes: they add up to the state of the system,
/// and their traces are the probabilities of the joint states.
#[derive(Clone, Debug, PartialEq)]
pub struct JointState {
    d: usize,
    /// `ρ_x[r, c]` at `x·d² + r + c·d`.
    data: Vec<C64>,
}

impl JointState {
    /// The joint state with `ρ_x = blocks[x]` (square, of one dimension,
    /// finite).
    pub fn from_blocks(blocks: &[CMatrix]) -> Result<Self> {
        let Some(first) = blocks.first() else {
            return Err(invalid("blocks", "need at least one joint state"));
        };
        let d = first.nrows();
        if d == 0 {
            return Err(invalid("blocks", "need at least one level"));
        }
        let mut data = Vec::with_capacity(blocks.len() * d * d);
        for b in blocks {
            if b.nrows() != d || b.ncols() != d {
                return Err(QsoError::DimensionMismatch {
                    expected: d,
                    actual: b.nrows(),
                });
            }
            if b.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
                return Err(invalid("blocks", "entries must be finite"));
            }
            // Column-major, as `ρ_x[r, c]` at `r + c·d`.
            data.extend(b.iter().copied());
        }
        Ok(JointState { d, data })
    }

    /// Dimension of the system.
    pub fn dimension(&self) -> usize {
        self.d
    }

    pub fn num_joint_states(&self) -> usize {
        self.data.len() / (self.d * self.d)
    }

    /// `ρ_x`, the part of the state in joint state `x`.
    ///
    /// # Panics
    /// If `x` is not below [`num_joint_states`](Self::num_joint_states).
    pub fn state(&self, x: usize) -> CMatrix {
        let dd = self.d * self.d;
        let block = &self.data[x * dd..(x + 1) * dd];
        CMatrix::from_fn(self.d, self.d, |r, c| block[r + c * self.d])
    }

    /// The state of the system, `Σ_x ρ_x`.
    pub fn average(&self) -> CMatrix {
        let dd = self.d * self.d;
        let mut sum = vec![ZERO; dd];
        for block in self.data.chunks(dd) {
            for (s, z) in sum.iter_mut().zip(block) {
                *s += z;
            }
        }
        CMatrix::from_fn(self.d, self.d, |r, c| sum[r + c * self.d])
    }

    /// `Tr ρ_x` for every joint state.
    pub fn probabilities(&self) -> Vec<f64> {
        let dd = self.d * self.d;
        self.data
            .chunks(dd)
            .map(|b| (0..self.d).map(|r| b[r + r * self.d].re).sum())
            .collect()
    }
}

impl StochasticLiouville {
    /// The equation for a system of dimension `dimension` and independent
    /// `processes`, each with the Hermitian operator it couples through.
    pub fn new(dimension: usize, processes: Vec<(MarkovProcess, CMatrix)>) -> Result<Self> {
        if dimension == 0 {
            return Err(invalid("dimension", "need at least one level"));
        }
        let mut joint = 1usize;
        for (process, coupling) in &processes {
            check_hermitian("coupling", coupling, dimension)?;
            joint = joint
                .checked_mul(process.num_states())
                .filter(|&j| j <= MAX_PROCESS_STATES)
                .ok_or_else(|| {
                    invalid(
                        "processes",
                        format!("more than {MAX_PROCESS_STATES} joint states"),
                    )
                })?;
        }
        if joint
            .checked_mul(dimension * dimension)
            .is_none_or(|c| c > MAX_LIOUVILLE_COEFFICIENTS)
        {
            return Err(invalid(
                "dimension",
                format!(
                    "{joint} joint states of a {dimension}-level system exceed \
                     {MAX_LIOUVILLE_COEFFICIENTS} coefficients"
                ),
            ));
        }
        let mut strides = vec![1usize; processes.len()];
        for k in (0..processes.len().saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * processes[k + 1].0.num_states();
        }
        Ok(StochasticLiouville {
            d: dimension,
            processes,
            dissipators: Vec::new(),
            joint,
            strides,
        })
    }

    /// The same equation with Lindblad operators `operators` (rates
    /// included: `L = √γ A`).
    pub fn with_dissipators(mut self, operators: Vec<CMatrix>) -> Result<Self> {
        for l in &operators {
            if l.nrows() != self.d || l.ncols() != self.d {
                return Err(QsoError::DimensionMismatch {
                    expected: self.d,
                    actual: l.nrows(),
                });
            }
            if l.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
                return Err(invalid("dissipators", "entries must be finite"));
            }
        }
        self.dissipators = operators;
        Ok(self)
    }

    pub fn dimension(&self) -> usize {
        self.d
    }

    pub fn num_joint_states(&self) -> usize {
        self.joint
    }

    /// The state of process `k` in joint state `x`.
    fn digit(&self, x: usize, k: usize) -> usize {
        (x / self.strides[k]) % self.processes[k].0.num_states()
    }

    /// `ρ_x = p₀(x) ρ`: the system in `rho`, the processes in their
    /// distributions at time 0, independent of each other.
    pub fn start(&self, rho: &CMatrix) -> Result<JointState> {
        if rho.nrows() != self.d || rho.ncols() != self.d {
            return Err(QsoError::DimensionMismatch {
                expected: self.d,
                actual: rho.nrows(),
            });
        }
        if rho.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
            return Err(invalid("rho", "entries must be finite"));
        }
        let dd = self.d * self.d;
        let mut data = vec![ZERO; self.joint * dd];
        for x in 0..self.joint {
            let p: f64 = (0..self.processes.len())
                .map(|k| self.processes[k].0.initial[self.digit(x, k)])
                .product();
            for c in 0..self.d {
                for r in 0..self.d {
                    data[x * dd + r + c * self.d] = rho[(r, c)] * p;
                }
            }
        }
        Ok(JointState { d: self.d, data })
    }

    fn check_state(&self, state: &JointState) -> Result<()> {
        if state.d != self.d || state.data.len() != self.joint * self.d * self.d {
            return Err(QsoError::DimensionMismatch {
                expected: self.joint * self.d * self.d,
                actual: state.data.len(),
            });
        }
        Ok(())
    }

    /// `state` after a time `t` under the Hamiltonian `hamiltonian`
    /// (Hermitian, `d × d`), the noise and the dissipators.
    pub fn evolve(&self, state: &JointState, hamiltonian: &CMatrix, t: f64) -> Result<JointState> {
        self.check_state(state)?;
        check_hermitian("hamiltonian", hamiltonian, self.d)?;
        check_time(t)?;
        if t == 0.0 {
            return Ok(state.clone());
        }
        let block = self.generator(hamiltonian)?;
        let work = krylov_work(block.n, block.nnz(), block.norm_inf(), t);
        if work.is_nan() || work > MAX_LIOUVILLE_WORK {
            return Err(invalid(
                "t",
                format!(
                    "Krylov steps would take about {work:.1e} operations (limit \
                     {MAX_LIOUVILLE_WORK:.0e})"
                ),
            ));
        }
        let data = expv(&block, &state.data, t, KRYLOV_TOL, 0.0)?;
        Ok(JointState { d: self.d, data })
    }

    /// `state` after an ideal pulse `unitary`: `ρ_x ↦ U ρ_x U†` in every
    /// joint state.
    pub fn pulse(&self, state: &JointState, unitary: &CMatrix) -> Result<JointState> {
        self.check_state(state)?;
        if unitary.nrows() != self.d || unitary.ncols() != self.d {
            return Err(QsoError::DimensionMismatch {
                expected: self.d,
                actual: unitary.nrows(),
            });
        }
        if unitary
            .iter()
            .any(|z| !(z.re.is_finite() && z.im.is_finite()))
        {
            return Err(invalid("unitary", "entries must be finite"));
        }
        let product = unitary.adjoint() * unitary;
        let off = (0..self.d)
            .flat_map(|r| (0..self.d).map(move |c| (r, c)))
            .map(|(r, c)| (product[(r, c)] - if r == c { 1.0 } else { 0.0 }).norm())
            .fold(0.0, f64::max);
        if off.is_nan() || off > 1e-10 {
            return Err(invalid("unitary", "must be unitary"));
        }
        let mut out = state.clone();
        let dd = self.d * self.d;
        let adjoint = unitary.adjoint();
        for (x, block) in out.data.chunks_mut(dd).enumerate() {
            let rho = state.state(x);
            let turned = unitary * rho * &adjoint;
            for c in 0..self.d {
                for r in 0..self.d {
                    block[r + c * self.d] = turned[(r, c)];
                }
            }
        }
        Ok(out)
    }

    /// The generator of the joint density matrices for `hamiltonian`, as a
    /// sparse matrix on `x·d² + r + c·d`.
    fn generator(&self, hamiltonian: &CMatrix) -> Result<SparseBlock> {
        let d = self.d;
        let dd = d * d;
        // The dissipator, the same in every joint state.
        let mut dissipator: Vec<(usize, usize, C64)> = Vec::new();
        for l in &self.dissipators {
            let ldl = l.adjoint() * l;
            for r2 in 0..d {
                for r in 0..d {
                    let a = l[(r2, r)];
                    if a == ZERO {
                        continue;
                    }
                    for c2 in 0..d {
                        for c in 0..d {
                            let b = l[(c2, c)].conj();
                            if b != ZERO {
                                dissipator.push((r2 + c2 * d, r + c * d, a * b));
                            }
                        }
                    }
                }
            }
            for r2 in 0..d {
                for r in 0..d {
                    let v = ldl[(r2, r)] * -0.5;
                    if v != ZERO {
                        for c in 0..d {
                            dissipator.push((r2 + c * d, r + c * d, v));
                            // −½ ρ L†L: ((r', c'), (r, c)) with (L†L)[c][c'].
                            dissipator.push((c + r2 * d, c + r * d, ldl[(r, r2)] * -0.5));
                        }
                    }
                }
            }
        }
        // Entries: the commutator and the dissipator in every joint state,
        // and two per element for every switch out of it.
        let nonzero = |m: &CMatrix| m.iter().filter(|z| **z != ZERO).count();
        let mut per_block = 2 * d * nonzero(hamiltonian) + dissipator.len();
        let mut switching = 0usize;
        for (process, coupling) in &self.processes {
            per_block += 2 * d * nonzero(coupling);
            switching = switching.saturating_add(
                (2 * dd * process.rates.len()).saturating_mul(self.joint / process.num_states()),
            );
        }
        let bound = per_block
            .saturating_mul(self.joint)
            .saturating_add(switching);
        if bound > MAX_LIOUVILLE_ENTRIES || self.joint * dd > u32::MAX as usize {
            return Err(invalid(
                "dimension",
                format!(
                    "the generator may have {bound} entries, more than {MAX_LIOUVILLE_ENTRIES}"
                ),
            ));
        }
        let mut triplets: Vec<(u32, u32, C64)> = Vec::with_capacity(bound);
        let mut hx = hamiltonian.clone();
        for x in 0..self.joint {
            hx.copy_from(hamiltonian);
            for (k, (process, coupling)) in self.processes.iter().enumerate() {
                let v = process.values[self.digit(x, k)];
                if v != 0.0 {
                    hx += coupling * C64::new(v, 0.0);
                }
            }
            let base = x * dd;
            let push = |t: &mut Vec<(u32, u32, C64)>, row: usize, col: usize, v: C64| {
                t.push(((base + row) as u32, (base + col) as u32, v));
            };
            for r2 in 0..d {
                for r in 0..d {
                    let v = hx[(r2, r)];
                    if v == ZERO {
                        continue;
                    }
                    for c in 0..d {
                        // −i H ρ and +i ρ H (with H[r][r2] for the column).
                        push(&mut triplets, r2 + c * d, r + c * d, -I * v);
                        push(&mut triplets, c + r * d, c + r2 * d, I * v);
                    }
                }
            }
            for &(row, col, v) in &dissipator {
                push(&mut triplets, row, col, v);
            }
            // Switching: ρ_y gains W[y][x] ρ_x, ρ_x loses as much.
            for (k, (process, _)) in self.processes.iter().enumerate() {
                let own = self.digit(x, k);
                for &(from, to, rate) in &process.rates {
                    if from != own {
                        continue;
                    }
                    let y = x - from * self.strides[k] + to * self.strides[k];
                    for e in 0..dd {
                        triplets.push((
                            (y * dd + e) as u32,
                            (base + e) as u32,
                            C64::new(rate, 0.0),
                        ));
                        triplets.push(((base + e) as u32, (base + e) as u32, C64::new(-rate, 0.0)));
                    }
                }
            }
        }
        Ok(SparseBlock::from_triplets(self.joint * dd, triplets))
    }
}

/// A quasi-static parameter of a circuit: its distribution and the nodes
/// of its Gauss rule (built once the work is known to be within limits).
struct Parameter {
    spread: Spread,
    count: usize,
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

/// How a circuit with slow noise is averaged: the qubits whose telegraph
/// noise acts (one bit of the joint state each), and the quasi-static
/// detunings and over-rotation spreads with their rules.
struct Plan {
    /// Per qubit: its bit in the joint state, if its telegraph noise acts.
    bit: Vec<Option<usize>>,
    /// `(amplitude, rate)` of each bit's telegraph process.
    telegraph: Vec<(f64, f64)>,
    /// Per qubit: its detuning parameter, if any.
    detuning: Vec<Option<usize>>,
    /// Per rule: its spread parameter, if any.
    spread: Vec<Option<usize>>,
    parameters: Vec<Parameter>,
    points: usize,
}

impl Plan {
    fn new(program: &Program) -> Result<Self> {
        let n = program.n_qubits;
        // How long each qubit is turned, and how far each spread rule's
        // gates turn: the reach of their parameters.
        let mut turned = vec![0.0f64; n];
        let mut angles = vec![0.0f64; program.spreads.len()];
        let mut used = vec![false; program.spreads.len()];
        for op in &program.ops {
            match op {
                Op::Phase { qubit, from, to } => turned[*qubit] += to - from,
                Op::SpreadGate { inst, rule, .. } => {
                    used[*rule] = true;
                    angles[*rule] += inst.angles(&[]).iter().map(|a| a.abs()).sum::<f64>();
                }
                _ => {}
            }
        }
        let mut bit = vec![None; n];
        let mut telegraph = Vec::new();
        let mut detuning = vec![None; n];
        let mut spread = vec![None; program.spreads.len()];
        let mut wanted: Vec<(Spread, f64)> = Vec::new();
        for q in 0..n {
            if turned[q] <= 0.0 {
                continue;
            }
            let slow = &program.slow[q];
            if let Some(t) = slow.telegraph
                && t.amplitude != 0.0
            {
                bit[q] = Some(telegraph.len());
                telegraph.push((t.amplitude, t.rate));
            }
            if let Some(s) = slow.detuning
                && width(&s) > 0.0
            {
                detuning[q] = Some(wanted.len());
                wanted.push((s, turned[q]));
            }
        }
        for (r, s) in program.spreads.iter().enumerate() {
            if let Some(s) = s
                && used[r]
                && width(s) > 0.0
            {
                spread[r] = Some(wanted.len());
                wanted.push((*s, angles[r]));
            }
        }
        // The quadrature errors of the parameters add up (positive weights
        // adding up to 1 do not amplify them).
        let eps = QUADRATURE_ERROR / wanted.len().max(1) as f64;
        let mut parameters = Vec::with_capacity(wanted.len());
        let mut points = 1usize;
        for (s, reach) in &wanted {
            let m = nodes_needed(s, *reach, eps).ok_or_else(|| {
                invalid(
                    "device",
                    format!(
                        "quasi-static noise of width {} over {reach} would need more than \
                         {MAX_QUADRATURE_NODES} quadrature nodes; simulate it with trajectories",
                        width(s)
                    ),
                )
            })?;
            points = points.saturating_mul(m);
            parameters.push(Parameter {
                spread: *s,
                count: m,
                nodes: Vec::new(),
                weights: Vec::new(),
            });
        }
        Ok(Plan {
            bit,
            telegraph,
            detuning,
            spread,
            parameters,
            points,
        })
    }

    /// The Gauss rules of the parameters.
    fn build_rules(&mut self) {
        for par in &mut self.parameters {
            (par.nodes, par.weights) = gauss_rule(&par.spread, par.count);
        }
    }

    /// The values of the parameters at quadrature point `p` (the first
    /// parameter varying slowest) and its weight.
    fn point(&self, mut p: usize) -> (Vec<f64>, f64) {
        let mut values = vec![0.0; self.parameters.len()];
        let mut weight = 1.0;
        for (k, par) in self.parameters.iter().enumerate().rev() {
            let m = par.nodes.len();
            values[k] = par.nodes[p % m];
            weight *= par.weights[p % m];
            p /= m;
        }
        (values, weight)
    }
}

/// Apply a compiled program with slow noise to `rho`, averaged exactly
/// over the telegraph processes and by Gauss quadrature over the
/// quasi-static parameters (see the module documentation).
pub(crate) fn run_program(rho: &mut DensityMatrix, program: &Program) -> Result<()> {
    let mut plan = Plan::new(program)?;
    let entries = rho.data.len();
    // A group of points in flight holds a density matrix per joint state
    // and its sum; besides, the start.
    let joint = 1usize
        .checked_shl(plan.telegraph.len() as u32)
        .filter(|j| (j + 2).saturating_mul(entries) <= MAX_SLOW_ENTRIES)
        .ok_or_else(|| {
            invalid(
                "device",
                format!(
                    "slow noise on {} qubits with telegraph noise on {} of them needs \
                     {} density matrices of {entries} entries at once, more than \
                     {MAX_SLOW_ENTRIES} entries in all (12 qubits at most)",
                    rho.n_qubits,
                    plan.telegraph.len(),
                    (1u128 << plan.telegraph.len().min(100)) + 2,
                ),
            )
        })?;
    // Passes over the entries of one density matrix: two per gate (rows
    // and columns), 4^k per channel on k qubits, a few per phase step.
    let passes: f64 = 1.0
        + program
            .ops
            .iter()
            .map(|op| match op {
                Op::Gates(c) => 2.0 * c.instructions().len() as f64,
                Op::Channel { qubits, .. } => 4f64.powi(qubits.len() as i32),
                Op::Phase { .. } => 4.0,
                Op::SpreadGate { .. } => 2.0,
            })
            .sum::<f64>();
    let work = plan.points as f64 * joint as f64 * entries as f64 * passes;
    if work > MAX_SLOW_WORK {
        return Err(invalid(
            "device",
            format!(
                "averaging the slow noise needs about {work:.1e} operations ({} quadrature \
                 points, {joint} joint states), more than {MAX_SLOW_WORK:.0e}; simulate it \
                 with trajectories",
                plan.points
            ),
        ));
    }
    plan.build_rules();
    // Groups of quadrature points, fixed by their number alone, run in
    // parallel; their sums are added in order (the same result on any
    // number of threads). Each holds `joint + 1` density matrices.
    let groups = plan
        .points
        .min(64)
        .min(((MAX_SLOW_ENTRIES / entries - 1) / (joint + 1)).max(1));
    let per_group = plan.points.div_ceil(groups);
    let partial: Vec<Vec<C64>> = map_tasks(groups, |g| {
        let mut sum = vec![ZERO; entries];
        for p in g * per_group..((g + 1) * per_group).min(plan.points) {
            let (values, weight) = plan.point(p);
            run_point(rho, program, &plan, &values, joint, weight, &mut sum);
        }
        sum
    });
    let mut parts = partial.into_iter();
    let mut total = parts.next().expect("at least one group");
    for part in parts {
        for (s, z) in total.iter_mut().zip(&part) {
            *s += z;
        }
    }
    rho.data = total;
    Ok(())
}

/// The program at one quadrature point: one density matrix per joint state
/// of the telegraph processes (starting stationary, so each holds
/// `ρ/2^K`), their sum added to `sum` with `weight`.
fn run_point(
    rho: &DensityMatrix,
    program: &Program,
    plan: &Plan,
    values: &[f64],
    joint: usize,
    weight: f64,
    sum: &mut [C64],
) {
    let n = rho.n_qubits;
    let mut start = rho.clone();
    start.scale(1.0 / joint as f64);
    let mut states = vec![start; joint];
    // The time up to which each telegraph process is known.
    let mut known = vec![0.0f64; plan.telegraph.len()];
    let mut i = 0;
    let ops = &program.ops;
    while i < ops.len() {
        match &ops[i] {
            Op::Phase { qubit, from, to } => {
                let q = *qubit;
                let shift = plan.detuning[q].map_or(0.0, |k| values[k]);
                let tau = to - from;
                match plan.bit[q] {
                    Some(b) => {
                        let (amplitude, rate) = plan.telegraph[b];
                        let gap = (from - known[b]).max(0.0);
                        known[b] = *to;
                        // Per element class s = b_r − b_c: the coupled
                        // evolution of the pair, after the free switching
                        // since the process was last needed.
                        let free = telegraph_propagator(amplitude, 0.0, rate, 0.0, gap);
                        let class = |s: f64| -> [[C64; 2]; 2] {
                            let m = telegraph_propagator(amplitude, s * shift, rate, s, tau);
                            let mut out = [[ZERO; 2]; 2];
                            for (r, row) in out.iter_mut().enumerate() {
                                for (c, x) in row.iter_mut().enumerate() {
                                    *x = m[r][0] * free[0][c] + m[r][1] * free[1][c];
                                }
                            }
                            out
                        };
                        let classes = [class(-1.0), class(0.0), class(1.0)];
                        for x0 in (0..joint).filter(|x| x >> b & 1 == 0) {
                            let x1 = x0 | 1 << b;
                            let (low, high) = states.split_at_mut(x1);
                            let (a, z) = (&mut low[x0].data, &mut high[0].data);
                            for (e, (u, v)) in a.iter_mut().zip(z.iter_mut()).enumerate() {
                                let s = (e >> q & 1) as isize - (e >> (q + n) & 1) as isize;
                                let m = &classes[(s + 1) as usize];
                                let (p, w) = (*u, *v);
                                *u = m[0][0] * p + m[0][1] * w;
                                *v = m[1][0] * p + m[1][1] * w;
                            }
                        }
                    }
                    None if shift != 0.0 => {
                        let (up, down) = (
                            C64::from_polar(1.0, shift * tau),
                            C64::from_polar(1.0, -shift * tau),
                        );
                        for state in &mut states {
                            for (e, z) in state.data.iter_mut().enumerate() {
                                match (e >> q & 1, e >> (q + n) & 1) {
                                    (1, 0) => *z *= up,
                                    (0, 1) => *z *= down,
                                    _ => {}
                                }
                            }
                        }
                    }
                    None => {}
                }
                i += 1;
            }
            Op::SpreadGate { inst, base, rule } => {
                let factor = base + plan.spread[*rule].map_or(0.0, |k| values[k]);
                let mut scaled: Instruction = inst.clone();
                for p in &mut scaled.params {
                    if let Param::Value(v) = p {
                        *v *= factor;
                    }
                }
                let angles = scaled.angles(&[]);
                for state in &mut states {
                    kernels::apply_instruction(&mut state.data, &scaled, &angles, false, 0);
                    kernels::apply_instruction(&mut state.data, &scaled, &angles, true, n);
                }
                i += 1;
            }
            _ => {
                // A run of gates and channels, the same in every joint
                // state.
                let mut j = i;
                while j < ops.len() && matches!(ops[j], Op::Gates(_) | Op::Channel { .. }) {
                    j += 1;
                }
                for state in &mut states {
                    state.run_ops(&ops[i..j], &program.channels);
                }
                i = j;
            }
        }
    }
    for state in &states {
        for (s, z) in sum.iter_mut().zip(&state.data) {
            *s += z * weight;
        }
    }
}

#[cfg(test)]
mod tests;
