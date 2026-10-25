//! Combinatorial optimisation: QUBO and Ising problems, and QAOA with
//! problem-specific mixers.
//!
//! * [`Qubo`]: minimise `f(x) = Σ_{i≤j} Q_ij x_i x_j + c` over
//!   `x ∈ {0, 1}ⁿ`. [`Ising`]: minimise `E(s) = Σ h_i s_i + Σ J_ij s_i s_j
//!   + c` over `s ∈ {±1}ⁿ`. With `s_i = 1 − 2x_i` the two are the same
//!   problem, and both become the diagonal cost Hamiltonian
//!   `Σ h_i Z_i + Σ J_ij Z_i Z_j + c` (bit `x_i = 1` is qubit state |1⟩,
//!   `Z = −1`).
//! * [`Qaoa`] alternates `e^{−iγ_l C}` with a mixer `e^{−iβ_l H_M}` for
//!   `l = 1 … p`, starting from the mixer's ground state:
//!   - [`Mixer::X`]: `Π RX(2β)` from `|+⟩ⁿ` (Farhi et al. 2014);
//!   - [`Mixer::XyRing`] and [`Mixer::XyComplete`]: `exp(−iβ(XX + YY)/2)`
//!     on the pairs of a ring (odd pairs, then even pairs) or of all pairs,
//!     from the Dicke state of Hamming weight `k`. Every factor conserves
//!     the Hamming weight, so the state never leaves the feasible strings
//!     of a "choose exactly k" constraint (Hadfield et al., Algorithms 12,
//!     34 (2019); Wang, Rubin, Dominy and Rieffel, Phys. Rev. A 101,
//!     012320 (2020));
//!   - [`Mixer::WarmStart`]: from `⊗ RY(θ_i)|0⟩` with
//!     `θ_i = 2 arcsin √c_i` for a relaxed or heuristic solution
//!     `c ∈ [0, 1]ⁿ`, mixing with `RY(θ_i) RZ(−2β) RY(−θ_i)`, whose
//!     generator has that state as its ground state (Egger, Mareček and
//!     Woerner, Quantum 5, 479 (2021)). Values of `c` are clipped to
//!     `[ε, 1 − ε]` so the state can still move.
//!
//! The circuit is an ordinary parametrised [`Circuit`] (`2p` parameters
//! `[γ_1 … γ_p, β_1 … β_p]`), optimised with exact adjoint gradients.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::GradientMethod;
use super::gradients::{expectation, expectation_and_gradient};
use crate::error::{QsoError, Result};
use crate::optimize::{Method, Objective, Options, minimize};
use crate::parallel::map_tasks;
use crate::quantum::{Circuit, MAX_QUBITS, Pauli, PauliString, PauliSum, StateVector};
use crate::rng::Rng;

/// Largest problem solved by enumeration.
pub const MAX_BRUTE_FORCE: usize = 30;
/// Most minimisers returned by enumeration (the smallest bit masks).
pub const MAX_MINIMISERS: usize = 1024;

/// Quadratic unconstrained binary optimisation, minimised.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Qubo {
    n: usize,
    /// Upper triangle `Q_ij` (`i ≤ j`), row-major `n × n`.
    q: Vec<f64>,
    offset: f64,
}

/// An Ising problem, minimised.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ising {
    pub n: usize,
    pub h: Vec<f64>,
    /// Couplings `(i, j, J_ij)` with `i < j`.
    pub j: Vec<(usize, usize, f64)>,
    pub offset: f64,
}

impl Qubo {
    /// `f(x) = Σ_ij Q_ij x_i x_j + offset` from a full `n × n` matrix
    /// (row-major; `Q_ij` and `Q_ji` add up).
    pub fn new(n: usize, matrix: &[f64], offset: f64) -> Result<Self> {
        if n == 0 || n > 64 {
            return Err(QsoError::invalid("n", "need 1 to 64 variables"));
        }
        if matrix.len() != n * n {
            return Err(QsoError::DimensionMismatch {
                expected: n * n,
                actual: matrix.len(),
            });
        }
        if !offset.is_finite() || matrix.iter().any(|v| !v.is_finite()) {
            return Err(QsoError::invalid("qubo", "coefficients must be finite"));
        }
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let (a, b) = if i <= j { (i, j) } else { (j, i) };
                q[a * n + b] += matrix[i * n + j];
            }
        }
        if q.iter().any(|v| !v.is_finite()) {
            return Err(QsoError::invalid(
                "qubo",
                "symmetrised coefficients overflow",
            ));
        }
        Ok(Self { n, q, offset })
    }

    /// From `(i, j, Q_ij)` entries (repeated or mirrored entries add up).
    pub fn from_entries(n: usize, entries: &[(usize, usize, f64)], offset: f64) -> Result<Self> {
        if n == 0 || n > 64 {
            return Err(QsoError::invalid("n", "need 1 to 64 variables"));
        }
        let mut m = vec![0.0; n * n];
        for &(i, j, v) in entries {
            if i >= n || j >= n {
                return Err(QsoError::invalid(
                    "entries",
                    format!("index ({i}, {j}) out of range for {n} variables"),
                ));
            }
            m[i * n + j] += v;
        }
        Self::new(n, &m, offset)
    }

    pub fn num_variables(&self) -> usize {
        self.n
    }

    /// `Q_ij` for `i ≤ j` (the upper triangle).
    pub fn coefficient(&self, i: usize, j: usize) -> f64 {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        self.q[a * self.n + b]
    }

    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// `f(x)` for the bit mask `x` (bit `i` = `x_i`).
    pub fn value(&self, x: u64) -> f64 {
        let n = self.n;
        let mut f = self.offset;
        for i in 0..n {
            if (x >> i) & 1 == 0 {
                continue;
            }
            for j in i..n {
                if (x >> j) & 1 == 1 {
                    f += self.q[i * n + j];
                }
            }
        }
        f
    }

    /// The equivalent Ising problem (`x_i = (1 − s_i)/2`).
    pub fn to_ising(&self) -> Ising {
        let n = self.n;
        let mut h = vec![0.0; n];
        let mut j = Vec::new();
        let mut offset = self.offset;
        for a in 0..n {
            let d = self.q[a * n + a];
            // Q x_a = Q (1 − s_a)/2
            offset += d / 2.0;
            h[a] -= d / 2.0;
            for b in a + 1..n {
                let v = self.q[a * n + b];
                if v == 0.0 {
                    continue;
                }
                // v x_a x_b = v (1 − s_a − s_b + s_a s_b)/4
                offset += v / 4.0;
                h[a] -= v / 4.0;
                h[b] -= v / 4.0;
                j.push((a, b, v / 4.0));
            }
        }
        Ising { n, h, j, offset }
    }

    /// The diagonal cost Hamiltonian.
    pub fn to_pauli_sum(&self) -> PauliSum {
        self.to_ising().to_pauli_sum()
    }

    /// Exact minimum and the minimising bit masks (at most
    /// [`MAX_MINIMISERS`], the smallest), by enumeration
    /// (`n ≤` [`MAX_BRUTE_FORCE`]).
    pub fn brute_force(&self) -> Result<(f64, Vec<u64>)> {
        brute_force(self.n, |x| self.value(x))
    }
}

impl Ising {
    /// `E(s)` for the bit mask `x` with `s_i = 1 − 2x_i`.
    pub fn energy(&self, x: u64) -> f64 {
        let s = |i: usize| if (x >> i) & 1 == 1 { -1.0 } else { 1.0 };
        let mut e = self.offset;
        for (i, h) in self.h.iter().enumerate() {
            e += h * s(i);
        }
        for &(a, b, v) in &self.j {
            e += v * s(a) * s(b);
        }
        e
    }

    /// `Σ h_i Z_i + Σ J_ij Z_i Z_j + offset`.
    pub fn to_pauli_sum(&self) -> PauliSum {
        let mut h = PauliSum::new();
        if self.offset != 0.0 {
            h.add_term(self.offset, PauliString::identity());
        }
        for (i, &v) in self.h.iter().enumerate() {
            if v != 0.0 {
                h.add_term(v, PauliString::single(i, Pauli::Z));
            }
        }
        for &(a, b, v) in &self.j {
            if v != 0.0 {
                h.add_term(
                    v,
                    PauliString::new(&[(a, Pauli::Z), (b, Pauli::Z)]).expect("distinct qubits"),
                );
            }
        }
        h
    }

    /// The equivalent QUBO (`s_i = 1 − 2x_i`).
    pub fn to_qubo(&self) -> Result<Qubo> {
        let n = self.n;
        let mut entries = Vec::new();
        let mut offset = self.offset;
        for (i, &v) in self.h.iter().enumerate() {
            // h s = h − 2h x
            offset += v;
            entries.push((i, i, -2.0 * v));
        }
        for &(a, b, v) in &self.j {
            if a >= n || b >= n || a == b {
                return Err(QsoError::invalid("ising", "invalid coupling indices"));
            }
            // J s_a s_b = J (1 − 2x_a − 2x_b + 4 x_a x_b)
            offset += v;
            entries.push((a, a, -2.0 * v));
            entries.push((b, b, -2.0 * v));
            entries.push((a, b, 4.0 * v));
        }
        Qubo::from_entries(n, &entries, offset)
    }
}

fn brute_force(n: usize, f: impl Fn(u64) -> f64 + Sync) -> Result<(f64, Vec<u64>)> {
    if n > MAX_BRUTE_FORCE {
        return Err(QsoError::invalid(
            "n",
            format!("enumeration is limited to {MAX_BRUTE_FORCE} variables"),
        ));
    }
    // Chunks keep their minimum and its first minimisers, so memory stays
    // small for any n.
    let total = 1u64 << n;
    let chunks = 256u64.min(total);
    let size = total.div_ceil(chunks);
    let parts = crate::parallel::map_tasks(chunks as usize, |c| {
        let (lo, hi) = (c as u64 * size, ((c as u64 + 1) * size).min(total));
        let mut best = f64::INFINITY;
        let mut values = Vec::new();
        for x in lo..hi {
            let v = f(x);
            if v < best {
                best = v;
            }
            values.push((v, x));
            if values.len() > 4 * MAX_MINIMISERS {
                values.retain(|&(w, _)| w - best <= 1e-9 * best.abs().max(1.0));
                values.truncate(MAX_MINIMISERS);
            }
        }
        (best, values)
    });
    let best = parts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let tol = 1e-9 * best.abs().max(1.0);
    let mut arg: Vec<u64> = parts
        .into_iter()
        .flat_map(|p| p.1)
        .filter(|&(v, _)| v - best <= tol)
        .map(|(_, x)| x)
        .collect();
    arg.sort_unstable();
    arg.truncate(MAX_MINIMISERS);
    Ok((best, arg))
}

/// Mixer and initial state of a [`Qaoa`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mixer {
    /// `Π RX(2β)` from `|+⟩ⁿ`.
    X,
    /// XY mixer on the ring `0–1–…–(n−1)–0` from the Dicke state of
    /// Hamming weight `weight`.
    XyRing { weight: usize },
    /// XY mixer on all pairs from the Dicke state of Hamming weight `weight`.
    XyComplete { weight: usize },
    /// Warm start from `c ∈ [0, 1]ⁿ`, clipped to `[epsilon, 1 − epsilon]`.
    WarmStart { solution: Vec<f64>, epsilon: f64 },
}

/// Settings of [`Qaoa::optimize`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QaoaOptions {
    pub restarts: usize,
    pub seed: u64,
    /// Starting point `[γ…, β…]` for the first restart.
    pub initial: Option<Vec<f64>>,
    pub optimizer: Options,
}

impl Default for QaoaOptions {
    fn default() -> Self {
        Self {
            restarts: 8,
            seed: 0,
            initial: None,
            optimizer: Options::default(),
        }
    }
}

impl QaoaOptions {
    pub fn with_restarts(mut self, restarts: usize) -> Self {
        self.restarts = restarts;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_initial(mut self, params: Vec<f64>) -> Self {
        self.initial = Some(params);
        self
    }
}

/// Statistics of a QAOA state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QaoaAnalysis {
    pub gammas: Vec<f64>,
    pub betas: Vec<f64>,
    /// `⟨C⟩`.
    pub expectation: f64,
    /// Exact minimum of `C` over the feasible strings.
    pub optimum: f64,
    /// Exact maximum of `C` over the feasible strings.
    pub worst: f64,
    /// `(worst − ⟨C⟩)/(worst − optimum)`: 1 at the optimum, 0 at the worst.
    pub approximation_ratio: f64,
    /// Probability of measuring an optimal feasible string.
    pub optimal_probability: f64,
    /// Probability of measuring a feasible string.
    pub feasible_probability: f64,
    pub most_likely: u64,
    pub most_likely_value: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub converged: bool,
}

/// QAOA for a diagonal cost Hamiltonian with a choice of mixer.
#[derive(Clone, Debug)]
pub struct Qaoa {
    n: usize,
    p: usize,
    cost: PauliSum,
    diagonal: Vec<f64>,
    feasible: Option<usize>,
    circuit: Circuit,
    initial: StateVector,
}

fn dicke(n: usize, k: usize) -> Result<StateVector> {
    if k > n {
        return Err(QsoError::invalid(
            "weight",
            format!("weight {k} exceeds {n} qubits"),
        ));
    }
    let count = (0..1usize << n)
        .filter(|b| b.count_ones() as usize == k)
        .count();
    let a = C64::new(1.0 / (count as f64).sqrt(), 0.0);
    StateVector::from_amplitudes(
        (0..1usize << n)
            .map(|b| {
                if b.count_ones() as usize == k {
                    a
                } else {
                    C64::new(0.0, 0.0)
                }
            })
            .collect(),
    )
}

impl Qaoa {
    /// QAOA of depth `p` for the diagonal `cost` on `n_qubits` qubits.
    pub fn new(cost: &PauliSum, n_qubits: usize, p: usize, mixer: Mixer) -> Result<Self> {
        let n = n_qubits;
        if n == 0 || n > MAX_QUBITS.min(MAX_BRUTE_FORCE) {
            return Err(QsoError::invalid(
                "n_qubits",
                format!("need 1 to {} qubits", MAX_QUBITS.min(MAX_BRUTE_FORCE)),
            ));
        }
        if p == 0 || p > super::qaoa::MAX_QAOA_DEPTH {
            return Err(QsoError::invalid("p", "depth must be positive"));
        }
        if cost.min_qubits() > n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: cost.min_qubits(),
            });
        }
        let diagonal = cost.diagonal(n)?.ok_or_else(|| {
            QsoError::invalid("cost", "QAOA needs a diagonal cost (Z terms only)")
        })?;
        let mut circuit = Circuit::new(n);
        let gammas = circuit.new_parameters(p);
        let betas = circuit.new_parameters(p);
        let (initial, feasible) = match &mixer {
            Mixer::X => {
                let mut c = Circuit::new(n);
                (0..n).for_each(|q| {
                    c.h(q);
                });
                (StateVector::from_circuit(&c, &[])?, None)
            }
            Mixer::XyRing { weight } | Mixer::XyComplete { weight } => {
                if n < 2 {
                    return Err(QsoError::invalid("n_qubits", "XY mixers need two qubits"));
                }
                (dicke(n, *weight)?, Some(*weight))
            }
            Mixer::WarmStart { solution, epsilon } => {
                if solution.len() != n {
                    return Err(QsoError::DimensionMismatch {
                        expected: n,
                        actual: solution.len(),
                    });
                }
                if !(0.0..=0.5).contains(epsilon)
                    || solution.iter().any(|c| !(0.0..=1.0).contains(c))
                {
                    return Err(QsoError::invalid(
                        "warm start",
                        "need c in [0, 1] and epsilon in [0, 0.5]",
                    ));
                }
                let mut c = Circuit::new(n);
                for (q, &v) in solution.iter().enumerate() {
                    c.ry(q, 2.0 * v.clamp(*epsilon, 1.0 - epsilon).sqrt().asin());
                }
                (StateVector::from_circuit(&c, &[])?, None)
            }
        };
        let pairs: Vec<(usize, usize)> = match &mixer {
            Mixer::XyRing { .. } => {
                let mut v: Vec<(usize, usize)> =
                    (0..n - 1).step_by(2).map(|i| (i, i + 1)).collect();
                v.extend((1..n - 1).step_by(2).map(|i| (i, i + 1)));
                if n > 2 {
                    v.push((n - 1, 0));
                }
                v
            }
            Mixer::XyComplete { .. } => (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .collect(),
            _ => Vec::new(),
        };
        let terms: Vec<(f64, PauliString)> = cost
            .terms()
            .iter()
            .filter(|(c, s)| *c != 0.0 && !s.is_identity())
            .copied()
            .collect();
        for l in 0..p {
            for &(c, s) in &terms {
                // e^{−iγ c Z…Z} = rotation by 2cγ
                circuit.pauli_rotation(&s, gammas[l] * (2.0 * c))?;
            }
            match &mixer {
                Mixer::X => {
                    for q in 0..n {
                        circuit.rx(q, betas[l] * 2.0);
                    }
                }
                Mixer::XyRing { .. } | Mixer::XyComplete { .. } => {
                    for &(a, b) in &pairs {
                        // exp(−iβ(XX + YY)/2) = RXX(β) RYY(β)
                        circuit.rxx(a, b, betas[l]).ryy(a, b, betas[l]);
                    }
                }
                Mixer::WarmStart { solution, epsilon } => {
                    for (q, &v) in solution.iter().enumerate() {
                        let theta = 2.0 * v.clamp(*epsilon, 1.0 - epsilon).sqrt().asin();
                        circuit.ry(q, -theta).rz(q, betas[l] * -2.0).ry(q, theta);
                    }
                }
            }
        }
        Ok(Self {
            n,
            p,
            cost: cost.clone(),
            diagonal,
            feasible,
            circuit,
            initial,
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    pub fn depth(&self) -> usize {
        self.p
    }

    /// The parametrised circuit (applied to [`Qaoa::initial_state`]).
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    pub fn initial_state(&self) -> &StateVector {
        &self.initial
    }

    fn is_feasible(&self, b: usize) -> bool {
        self.feasible.is_none_or(|k| b.count_ones() as usize == k)
    }

    /// The QAOA state for `[γ…, β…]`.
    pub fn state(&self, params: &[f64]) -> Result<StateVector> {
        let mut psi = self.initial.clone();
        psi.apply_circuit(&self.circuit, params)?;
        Ok(psi)
    }

    /// `⟨C⟩` and its gradient.
    pub fn expectation_and_gradient(&self, params: &[f64]) -> Result<(f64, Vec<f64>)> {
        expectation_and_gradient(
            &self.circuit,
            &self.cost,
            params,
            Some(&self.initial),
            GradientMethod::Adjoint,
        )
    }

    /// Statistics of the state at `params`.
    pub fn analyze(&self, params: &[f64]) -> Result<QaoaAnalysis> {
        let psi = self.state(params)?;
        let probs: Vec<f64> = psi.amplitudes().iter().map(|a| a.norm_sqr()).collect();
        let feasible: Vec<usize> = (0..probs.len()).filter(|&b| self.is_feasible(b)).collect();
        if feasible.is_empty() {
            return Err(QsoError::invalid("problem", "no feasible strings"));
        }
        let optimum = feasible
            .iter()
            .map(|&b| self.diagonal[b])
            .fold(f64::INFINITY, f64::min);
        let worst = feasible
            .iter()
            .map(|&b| self.diagonal[b])
            .fold(f64::NEG_INFINITY, f64::max);
        let tol = 1e-9 * optimum.abs().max(1.0);
        let expectation: f64 = probs.iter().zip(&self.diagonal).map(|(p, d)| p * d).sum();
        let optimal_probability = feasible
            .iter()
            .filter(|&&b| self.diagonal[b] - optimum <= tol)
            .map(|&b| probs[b])
            .sum();
        let feasible_probability = feasible.iter().map(|&b| probs[b]).sum();
        let most_likely = (0..probs.len())
            .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
            .expect("non-empty");
        Ok(QaoaAnalysis {
            gammas: params[..self.p].to_vec(),
            betas: params[self.p..].to_vec(),
            expectation,
            optimum,
            worst,
            approximation_ratio: if worst > optimum {
                (worst - expectation) / (worst - optimum)
            } else {
                1.0
            },
            optimal_probability,
            feasible_probability,
            most_likely: most_likely as u64,
            most_likely_value: self.diagonal[most_likely],
            iterations: 0,
            evaluations: 0,
            converged: true,
        })
    }

    /// Minimise `⟨C⟩` from seeded random starts (γ in `[−π, π]`, β in
    /// `[−π/2, π/2]`).
    pub fn optimize(&self, options: &QaoaOptions) -> Result<QaoaAnalysis> {
        crate::optimize::check_restarts(options.restarts)?;
        if let Some(x) = &options.initial
            && x.len() != 2 * self.p
        {
            return Err(QsoError::ParameterCount {
                expected: 2 * self.p,
                actual: x.len(),
            });
        }
        let objective = Energy { qaoa: self };
        let base = Rng::seed_from_u64(options.seed);
        let p = self.p;
        let runs = map_tasks(options.restarts.max(1), |r| {
            let x0: Vec<f64> = match (&options.initial, r) {
                (Some(init), 0) => init.clone(),
                _ => {
                    let mut rng = base.fork(r as u64);
                    let mut x: Vec<f64> = (0..p)
                        .map(|_| rng.uniform_range(-std::f64::consts::PI, std::f64::consts::PI))
                        .collect();
                    x.extend((0..p).map(|_| {
                        rng.uniform_range(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2)
                    }));
                    x
                }
            };
            minimize(&objective, &x0, &Method::lbfgs(), &options.optimizer)
        });
        let runs: Vec<_> = runs.into_iter().collect::<Result<Vec<_>>>()?;
        let evaluations = runs.iter().map(|r| r.evaluations).sum();
        let best = runs
            .into_iter()
            .min_by(|a, b| a.value.total_cmp(&b.value))
            .expect("at least one run");
        let mut out = self.analyze(&best.x)?;
        out.iterations = best.iterations;
        out.evaluations = evaluations;
        out.converged = best.converged;
        Ok(out)
    }
}

struct Energy<'a> {
    qaoa: &'a Qaoa,
}

impl Objective for Energy<'_> {
    fn dim(&self) -> usize {
        2 * self.qaoa.p
    }

    fn value(&self, x: &[f64]) -> f64 {
        expectation(
            &self.qaoa.circuit,
            &self.qaoa.cost,
            x,
            Some(&self.qaoa.initial),
        )
        .unwrap_or(f64::INFINITY)
    }

    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        self.qaoa.expectation_and_gradient(x).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_qubo(n: usize, seed: u64) -> Qubo {
        let mut rng = Rng::seed_from_u64(seed);
        let m: Vec<f64> = (0..n * n).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
        Qubo::new(n, &m, rng.uniform_range(-1.0, 1.0)).unwrap()
    }

    #[test]
    fn qubo_ising_and_hamiltonian_agree_on_every_string() {
        for seed in 0..5 {
            let q = random_qubo(6, seed);
            let ising = q.to_ising();
            let diag = q.to_pauli_sum().diagonal(6).unwrap().unwrap();
            let back = ising.to_qubo().unwrap();
            for x in 0..64u64 {
                let f = q.value(x);
                assert!((ising.energy(x) - f).abs() < 1e-12);
                assert!((diag[x as usize] - f).abs() < 1e-12);
                assert!((back.value(x) - f).abs() < 1e-12);
            }
            let (best, arg) = q.brute_force().unwrap();
            assert!(arg.iter().all(|&x| (q.value(x) - best).abs() < 1e-9));
            assert!((0..64).all(|x| q.value(x) >= best - 1e-12));
        }
        assert!(Qubo::new(2, &[1.0, f64::NAN, 0.0, 0.0], 0.0).is_err());
        assert!(Qubo::new(2, &[1e308, 1e308, 1e308, 1e308], 0.0).is_err());
        // Enumeration keeps few minimisers: the zero QUBO has 2^20 of them.
        let zero = Qubo::new(20, &vec![0.0; 400], 0.0).unwrap();
        let (best, arg) = zero.brute_force().unwrap();
        assert_eq!(best, 0.0);
        assert_eq!(arg.len(), MAX_MINIMISERS);
        assert_eq!(arg[..3], [0, 1, 2]);
        assert!(Qubo::from_entries(2, &[(0, 2, 1.0)], 0.0).is_err());
    }

    #[test]
    fn xy_mixers_conserve_the_hamming_weight() {
        // Choose 3 of 6 items minimising a random QUBO.
        let q = random_qubo(6, 11);
        let cost = q.to_pauli_sum();
        for mixer in [Mixer::XyRing { weight: 3 }, Mixer::XyComplete { weight: 3 }] {
            let qaoa = Qaoa::new(&cost, 6, 2, mixer).unwrap();
            let mut rng = Rng::seed_from_u64(1);
            for _ in 0..5 {
                let params: Vec<f64> = (0..4).map(|_| rng.uniform_range(-3.0, 3.0)).collect();
                let a = qaoa.analyze(&params).unwrap();
                assert!((a.feasible_probability - 1.0).abs() < 1e-12);
            }
            let best = qaoa.optimize(&QaoaOptions::default()).unwrap();
            let constrained = (0..64u64)
                .filter(|x| x.count_ones() == 3)
                .map(|x| q.value(x))
                .fold(f64::INFINITY, f64::min);
            assert!((best.optimum - constrained).abs() < 1e-12);
            assert!(
                best.approximation_ratio > 0.8,
                "{}",
                best.approximation_ratio
            );
        }
    }

    #[test]
    fn warm_start_begins_at_the_relaxed_value_and_beats_standard_qaoa() {
        // At γ = β = 0 the warm-start state is the product distribution
        // with P(x_i = 1) = c_i, so ⟨C⟩ = Σ_{i<j} Q_ij c_i c_j + Σ Q_ii c_i + c.
        let n = 6;
        let mut wins = 0;
        for seed in 0..6 {
            let q = random_qubo(n, 100 + seed);
            let cost = q.to_pauli_sum();
            let c: Vec<f64> = (0..n).map(|i| 0.2 + 0.1 * i as f64).collect();
            let qaoa = Qaoa::new(
                &cost,
                n,
                1,
                Mixer::WarmStart {
                    solution: c.clone(),
                    epsilon: 0.0,
                },
            )
            .unwrap();
            let e0 = qaoa.analyze(&[0.0, 0.0]).unwrap().expectation;
            let mut expected = q.offset();
            for i in 0..n {
                expected += q.coefficient(i, i) * c[i];
                for j in i + 1..n {
                    expected += q.coefficient(i, j) * c[i] * c[j];
                }
            }
            assert!((e0 - expected).abs() < 1e-12);
            // Warm start from the rounded optimum, regularised by ε = 0.25,
            // against standard QAOA at depth 1.
            let (_, arg) = q.brute_force().unwrap();
            let guess: Vec<f64> = (0..n).map(|i| ((arg[0] >> i) & 1) as f64).collect();
            let warm = Qaoa::new(
                &cost,
                n,
                1,
                Mixer::WarmStart {
                    solution: guess,
                    epsilon: 0.25,
                },
            )
            .unwrap()
            .optimize(&QaoaOptions::default())
            .unwrap();
            let plain = Qaoa::new(&cost, n, 1, Mixer::X)
                .unwrap()
                .optimize(&QaoaOptions::default())
                .unwrap();
            if warm.optimal_probability > plain.optimal_probability {
                wins += 1;
            }
        }
        assert_eq!(wins, 6);
    }

    #[test]
    fn x_mixer_reproduces_maxcut_qaoa() {
        // Ring of 6: MaxCut as minimisation of −cut.
        let g = crate::graph::generators::cycle(6);
        let cut = crate::algorithms::models::maxcut_cost(&g).unwrap();
        let qaoa = Qaoa::new(&(cut * -1.0), 6, 1, Mixer::X).unwrap();
        let r = qaoa.optimize(&QaoaOptions::default()).unwrap();
        // p = 1 on a ring: ⟨cut⟩ = 3/4 · 6.
        assert!((r.expectation + 4.5).abs() < 1e-6, "{}", r.expectation);
        assert!(Qaoa::new(&"X0".parse().unwrap(), 1, 1, Mixer::X).is_err());
        assert!(Qaoa::new(&"Z0".parse().unwrap(), 2, 1, Mixer::XyRing { weight: 3 }).is_err());
    }
}
