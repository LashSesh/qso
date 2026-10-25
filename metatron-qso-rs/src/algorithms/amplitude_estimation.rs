//! Amplitude estimation: canonical, iterative and maximum-likelihood.
//!
//! A state-preparation circuit `A` on `n` qubits and an objective qubit
//! define the amplitude `a = P(objective = 1)` of `A|0⟩`. Write
//! `a = sin²θ`. The Grover operator `Q = A S₀ A† S_χ`, with
//! `S_χ = Z` on the objective qubit and `S₀ = 2|0⟩⟨0| − I`, rotates by `2θ`
//! in the plane of the good and bad components, so measuring
//! `Q^k A|0⟩` gives 1 with probability `sin²((2k + 1)θ)`. Each `Q` calls
//! `A` and `A†` once; QSO counts oracle calls as applications of `A` or
//! `A†`, so one shot of `Q^k A|0⟩` costs `2k + 1` of them.
//!
//! * [`canonical`] — phase estimation of `Q` with `m` evaluation qubits
//!   (Brassard, Høyer, Mosca and Tapp, Contemp. Math. 305, 53 (2002)):
//!   outcome `y` gives `ã = sin²(πy/M)`, `M = 2^m`, and with probability at
//!   least `8/π²` the error is at most `2π√(a(1−a))/M + π²/M²`. QSO returns
//!   the exact outcome distribution.
//! * [`iterative`] — iterative amplitude estimation (Grinko, Gacon, Zoufal
//!   and Woerner, npj Quantum Inf. 7, 52 (2021)): powers `k` chosen so that
//!   the current interval for θ stays in one half-plane, Chernoff–Hoeffding
//!   intervals at level `1 − α/T`; returns a confidence interval of
//!   half-width at most `ε`.
//! * [`maximum_likelihood`] — maximum-likelihood amplitude estimation
//!   (Suzuki et al., Quantum Inf. Process. 19, 75 (2020)) with the powers
//!   `0, 1, 2, 4, …`; the error bar comes from the Fisher information
//!   `Σ_k 4 N_k (2m_k + 1)²`.
//!
//! The estimation error of all three falls as `1/N_q` in the number of
//! oracle calls, against `1/√N_q` for plain sampling.

use serde::{Deserialize, Serialize};

use super::qft::inverse_qft;
use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, Instruction, StateVector};
use crate::rng::Rng;

/// Largest power `k` of `Q` simulated.
pub const MAX_POWER: usize = 1 << 20;
/// Most levels of [`maximum_likelihood`] (largest power `2^{14}`).
pub const MAX_LEVELS: usize = 16;
/// Smallest `epsilon` of [`iterative`]; powers up to [`MAX_POWER`] resolve
/// θ to about 10⁻⁶.
pub const MIN_EPSILON: f64 = 1e-6;

/// `A` and the objective qubit.
#[derive(Clone, Debug)]
pub struct AmplitudeProblem {
    a: Circuit,
    objective: usize,
}

impl AmplitudeProblem {
    /// The amplitude of `objective = 1` in `a|0⟩` (`a` without parameters).
    pub fn new(a: Circuit, objective: usize) -> Result<Self> {
        if a.num_parameters() != 0 {
            return Err(QsoError::invalid("a", "bind the parameters of A first"));
        }
        if objective >= a.n_qubits() {
            return Err(QsoError::QubitOutOfRange {
                qubit: objective,
                n_qubits: a.n_qubits(),
            });
        }
        if a.n_qubits() > 24 {
            return Err(QsoError::invalid("a", "at most 24 qubits"));
        }
        Ok(Self { a, objective })
    }

    /// One qubit with `P(1) = p`: `A = RY(2 arcsin √p)`.
    pub fn bernoulli(p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(QsoError::invalid("p", "must lie in [0, 1]"));
        }
        let mut a = Circuit::new(1);
        a.ry(0, 2.0 * p.sqrt().asin());
        Self::new(a, 0)
    }

    pub fn n_qubits(&self) -> usize {
        self.a.n_qubits()
    }

    pub fn objective(&self) -> usize {
        self.objective
    }

    /// The exact amplitude `a`, by simulating `A|0⟩`.
    pub fn exact(&self) -> Result<f64> {
        Ok(objective_probability(
            &StateVector::from_circuit(&self.a, &[])?,
            self.objective,
        ))
    }

    /// The Grover operator `Q` as a circuit (global phase included, so that
    /// controlled powers are right).
    pub fn grover_operator(&self) -> Circuit {
        let n = self.a.n_qubits();
        let o = self.objective;
        let mut q = Circuit::new(n);
        q.z(o);
        q.append(&self.a.inverse()).expect("same register");
        for k in 0..n {
            q.x(k);
        }
        if n == 1 {
            q.z(0);
        } else {
            let all: Vec<usize> = (0..n).collect();
            q.mcz(&all);
        }
        for k in 0..n {
            q.x(k);
        }
        q.append(&self.a).expect("same register");
        // X·MCZ·X = I − 2|0⟩⟨0| = −S₀: multiply by −I = ZXZX.
        q.z(o).x(o).z(o).x(o);
        q
    }

    /// `P(objective = 1)` in `Q^k A|0⟩` for each `k` in `powers`
    /// (ascending), simulated by applying `Q` repeatedly.
    pub fn probabilities(&self, powers: &[usize]) -> Result<Vec<f64>> {
        if powers.windows(2).any(|w| w[0] > w[1]) {
            return Err(QsoError::invalid("powers", "must be ascending"));
        }
        if powers.last().is_some_and(|&k| k > MAX_POWER) {
            return Err(QsoError::invalid("powers", format!("at most {MAX_POWER}")));
        }
        let q = self.grover_operator();
        let mut psi = StateVector::from_circuit(&self.a, &[])?;
        let mut current = 0;
        let mut out = Vec::with_capacity(powers.len());
        for &k in powers {
            while current < k {
                psi.apply_circuit(&q, &[])?;
                current += 1;
            }
            out.push(objective_probability(&psi, self.objective));
        }
        Ok(out)
    }
}

fn objective_probability(psi: &StateVector, q: usize) -> f64 {
    psi.amplitudes()
        .iter()
        .enumerate()
        .filter(|(b, _)| (b >> q) & 1 == 1)
        .map(|(_, a)| a.norm_sqr())
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Number of successes in `shots` Bernoulli(`p`) trials.
fn binomial(rng: &mut Rng, shots: usize, p: f64) -> usize {
    (0..shots).filter(|_| rng.uniform() < p).count()
}

/// Outcome of [`canonical`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CanonicalEstimate {
    /// The most likely estimate `sin²(πy/M)`.
    pub estimate: f64,
    /// Distinct estimates and their probabilities, ascending.
    pub distribution: Vec<(f64, f64)>,
    /// `M − 1` applications of `Q`, i.e. `2(M − 1) + 1` oracle calls.
    pub oracle_calls: usize,
}

/// Canonical amplitude estimation with `m` evaluation qubits.
pub fn canonical(problem: &AmplitudeProblem, m: usize) -> Result<CanonicalEstimate> {
    let n = problem.n_qubits();
    if m == 0 || m > 12 || n + m > 24 {
        return Err(QsoError::invalid(
            "m",
            "need 1 to 12 evaluation qubits and at most 24 qubits in total",
        ));
    }
    let q = problem.grover_operator();
    let mut c = Circuit::new(n + m);
    let register: Vec<usize> = (0..n).collect();
    c.append_mapped(&problem.a, &register)?;
    for k in 0..m {
        c.h(n + k);
    }
    for k in 0..m {
        let control = n + k;
        for _ in 0..(1usize << k) {
            for inst in q.instructions() {
                let mut controlled: Instruction = inst.clone();
                controlled.controls.push(control);
                c.try_push(controlled)?;
            }
        }
    }
    let eval: Vec<usize> = (n..n + m).collect();
    c.append_mapped(&inverse_qft(m, true), &eval)?;
    let psi = StateVector::from_circuit(&c, &[])?;
    let big_m = 1usize << m;
    let mut by_y = vec![0.0; big_m];
    for (b, amp) in psi.amplitudes().iter().enumerate() {
        by_y[b >> n] += amp.norm_sqr();
    }
    let mut distribution: Vec<(f64, f64)> = Vec::new();
    for (y, p) in by_y.iter().enumerate() {
        let est = (std::f64::consts::PI * y as f64 / big_m as f64)
            .sin()
            .powi(2);
        match distribution
            .iter_mut()
            .find(|(e, _)| (e - est).abs() < 1e-12)
        {
            Some(entry) => entry.1 += p,
            None => distribution.push((est, *p)),
        }
    }
    distribution.sort_by(|a, b| a.0.total_cmp(&b.0));
    let estimate = distribution
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty")
        .0;
    Ok(CanonicalEstimate {
        estimate,
        distribution,
        oracle_calls: 2 * (big_m - 1) + 1,
    })
}

/// An estimate with a confidence interval and its cost.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AmplitudeEstimate {
    pub estimate: f64,
    /// Confidence interval for `a`.
    pub interval: (f64, f64),
    /// Applications of `A` or `A†` over all shots.
    pub oracle_calls: usize,
    pub shots: usize,
    /// Powers `k` of `Q` used, with their shots and hits.
    pub schedule: Vec<(usize, usize, usize)>,
}

/// Iterative amplitude estimation to half-width `epsilon` with confidence
/// `1 − alpha`, `shots` per round (Chernoff–Hoeffding intervals).
pub fn iterative(
    problem: &AmplitudeProblem,
    epsilon: f64,
    alpha: f64,
    shots: usize,
    seed: u64,
) -> Result<AmplitudeEstimate> {
    use std::f64::consts::{FRAC_PI_2, PI};
    if !(MIN_EPSILON..0.5).contains(&epsilon) || !(alpha > 0.0 && alpha < 1.0) || shots == 0 {
        return Err(QsoError::invalid(
            "iterative amplitude estimation",
            format!("need {MIN_EPSILON} <= epsilon < 0.5, 0 < alpha < 1 and shots > 0"),
        ));
    }
    let rounds = (PI / (8.0 * epsilon)).log2().ceil().max(1.0);
    let half_width = |n: usize| ((2.0 * rounds / alpha).ln() / (2.0 * n as f64)).sqrt();
    let mut rng = Rng::seed_from_u64(seed);
    let (mut lo, mut hi) = (0.0f64, FRAC_PI_2);
    let mut k = 0usize;
    let mut up = true;
    let mut schedule: Vec<(usize, usize, usize)> = Vec::new();
    let mut iterations = 0;
    while hi.sin().powi(2) - lo.sin().powi(2) > 2.0 * epsilon {
        iterations += 1;
        if iterations > 10_000 {
            return Err(QsoError::NoConvergence {
                method: "iterative amplitude estimation",
                iterations,
                residual: hi - lo,
            });
        }
        // FindNextK: the largest K = 4k + 2 ≥ 2K_i keeping K·[lo, hi] in
        // one half-plane.
        let big_k_now = 4 * k + 2;
        let mut big_k = ((PI / (hi - lo)).floor() as usize).min(4 * MAX_POWER + 2);
        big_k -= (big_k + 2) % 4; // ≡ 2 mod 4
        while big_k >= 2 * big_k_now {
            let l = (big_k as f64 * lo) % (2.0 * PI);
            let u = (big_k as f64 * hi) % (2.0 * PI);
            if l <= u && u <= PI {
                k = (big_k - 2) / 4;
                up = true;
                break;
            }
            if l <= u && l >= PI {
                k = (big_k - 2) / 4;
                up = false;
                break;
            }
            big_k -= 4;
        }
        let p = problem.probabilities(&[k])?[0];
        let hits = binomial(&mut rng, shots, p);
        match schedule.last_mut() {
            Some(last) if last.0 == k => {
                last.1 += shots;
                last.2 += hits;
            }
            _ => schedule.push((k, shots, hits)),
        }
        let &(_, n_k, h_k) = schedule.last().expect("just pushed");
        let freq = h_k as f64 / n_k as f64;
        let w = half_width(n_k);
        let (p_min, p_max) = ((freq - w).max(0.0), (freq + w).min(1.0));
        let big_k = (4 * k + 2) as f64;
        // P = (1 − cos(Kθ))/2 ⇒ Kθ mod 2π from arccos(1 − 2P).
        let (t_min, t_max) = if up {
            ((1.0 - 2.0 * p_min).acos(), (1.0 - 2.0 * p_max).acos())
        } else {
            (
                2.0 * PI - (1.0 - 2.0 * p_max).acos(),
                2.0 * PI - (1.0 - 2.0 * p_min).acos(),
            )
        };
        let turns_lo = (big_k * lo / (2.0 * PI)).floor();
        let turns_hi = (big_k * hi / (2.0 * PI)).floor();
        let new_lo = (2.0 * PI * turns_lo + t_min) / big_k;
        let new_hi = (2.0 * PI * turns_hi + t_max) / big_k;
        lo = lo.max(new_lo);
        hi = hi.min(new_hi).max(lo);
    }
    let (a_lo, a_hi) = (lo.sin().powi(2), hi.sin().powi(2));
    let shots_total = schedule.iter().map(|s| s.1).sum();
    Ok(AmplitudeEstimate {
        estimate: 0.5 * (a_lo + a_hi),
        interval: (a_lo, a_hi),
        oracle_calls: schedule.iter().map(|s| s.1 * (2 * s.0 + 1)).sum(),
        shots: shots_total,
        schedule,
    })
}

/// Log-likelihood of θ for `(k, shots, hits)` records.
fn log_likelihood(theta: f64, schedule: &[(usize, usize, usize)]) -> f64 {
    schedule
        .iter()
        .map(|&(k, n, h)| {
            let x = (2 * k + 1) as f64 * theta;
            let (s, c) = (x.sin().powi(2).max(1e-300), x.cos().powi(2).max(1e-300));
            h as f64 * s.ln() + (n - h) as f64 * c.ln()
        })
        .sum()
}

/// Maximum-likelihood amplitude estimation with the powers
/// `0, 1, 2, 4, …, 2^{levels−2}` (`levels` powers), `shots` each; the
/// interval is the normal interval at `confidence` from the Fisher
/// information.
pub fn maximum_likelihood(
    problem: &AmplitudeProblem,
    levels: usize,
    shots: usize,
    confidence: f64,
    seed: u64,
) -> Result<AmplitudeEstimate> {
    use std::f64::consts::FRAC_PI_2;
    if levels == 0 || levels > MAX_LEVELS || shots == 0 || !(confidence > 0.0 && confidence < 1.0) {
        return Err(QsoError::invalid(
            "maximum-likelihood amplitude estimation",
            format!("need 1 to {MAX_LEVELS} levels, shots > 0 and a confidence in (0, 1)"),
        ));
    }
    let powers: Vec<usize> = (0..levels)
        .map(|j| if j == 0 { 0 } else { 1 << (j - 1) })
        .collect();
    let probabilities = problem.probabilities(&powers)?;
    let mut rng = Rng::seed_from_u64(seed);
    let schedule: Vec<(usize, usize, usize)> = powers
        .iter()
        .zip(&probabilities)
        .map(|(&k, &p)| (k, shots, binomial(&mut rng, shots, p)))
        .collect();
    // Grid finer than the fastest oscillation, then golden-section search.
    let fastest = (2 * powers[levels - 1] + 1) as f64;
    let points = (200.0 * fastest) as usize;
    let mut best = (0.0, f64::NEG_INFINITY);
    for i in 0..=points {
        let t = FRAC_PI_2 * i as f64 / points as f64;
        let l = log_likelihood(t, &schedule);
        if l > best.1 {
            best = (t, l);
        }
    }
    let step = FRAC_PI_2 / points as f64;
    let (mut a, mut b) = ((best.0 - step).max(0.0), (best.0 + step).min(FRAC_PI_2));
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..100 {
        let (x1, x2) = (b - g * (b - a), a + g * (b - a));
        if log_likelihood(x1, &schedule) >= log_likelihood(x2, &schedule) {
            b = x2;
        } else {
            a = x1;
        }
    }
    let theta = 0.5 * (a + b);
    let fisher: f64 = schedule
        .iter()
        .map(|&(k, n, _)| 4.0 * n as f64 * ((2 * k + 1) as f64).powi(2))
        .sum();
    let z = -crate::estimation::normal_quantile((1.0 - confidence) / 2.0);
    let width = z / fisher.sqrt();
    let (lo, hi) = ((theta - width).max(0.0), (theta + width).min(FRAC_PI_2));
    Ok(AmplitudeEstimate {
        estimate: theta.sin().powi(2),
        interval: (lo.sin().powi(2), hi.sin().powi(2)),
        oracle_calls: schedule.iter().map(|s| s.1 * (2 * s.0 + 1)).sum(),
        shots: shots * levels,
        schedule,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn grover_powers_follow_the_rotation() {
        // A two-qubit problem: a = P(q1 = 1) after RY on q0 and a CRY.
        let mut a = Circuit::new(2);
        a.ry(0, 1.1).cry(0, 1, 0.9).h(0);
        let problem = AmplitudeProblem::new(a, 1).unwrap();
        let amp = problem.exact().unwrap();
        let theta = amp.sqrt().asin();
        let powers = [0, 1, 2, 5, 13];
        for (k, p) in powers.iter().zip(problem.probabilities(&powers).unwrap()) {
            let expected = ((2 * k + 1) as f64 * theta).sin().powi(2);
            assert!((p - expected).abs() < 1e-12, "k = {k}: {p} vs {expected}");
        }
    }

    #[test]
    fn canonical_estimation_meets_its_error_bound() {
        for (k, &a) in [0.1, 0.27, 0.5, 0.83].iter().enumerate() {
            let problem = AmplitudeProblem::bernoulli(a).unwrap();
            for m in [3, 5, 7] {
                let big_m = (1 << m) as f64;
                let r = canonical(&problem, m).unwrap();
                let total: f64 = r.distribution.iter().map(|e| e.1).sum();
                assert!((total - 1.0).abs() < 1e-10);
                let bound = 2.0 * PI * (a * (1.0 - a)).sqrt() / big_m + PI * PI / (big_m * big_m);
                let within: f64 = r
                    .distribution
                    .iter()
                    .filter(|(e, _)| (e - a).abs() <= bound)
                    .map(|e| e.1)
                    .sum();
                assert!(within >= 8.0 / (PI * PI), "a = {a}, m = {m}: {within}");
                assert!((r.estimate - a).abs() <= bound, "{k} {m}");
            }
        }
    }

    #[test]
    fn iterative_intervals_cover_and_shrink() {
        let alpha = 0.05;
        let mut misses = 0;
        let runs = 200;
        for seed in 0..runs {
            let a = 0.05 + 0.9 * (seed as f64 / runs as f64);
            let problem = AmplitudeProblem::bernoulli(a).unwrap();
            let r = iterative(&problem, 1e-3, alpha, 100, seed).unwrap();
            assert!(r.interval.1 - r.interval.0 <= 2e-3 + 1e-12);
            if a < r.interval.0 - 1e-12 || a > r.interval.1 + 1e-12 {
                misses += 1;
            }
        }
        assert!((misses as f64) <= alpha * runs as f64, "{misses} misses");
    }

    /// Root-mean-square error over seeds and the oracle calls of one run.
    fn rmse(f: impl Fn(u64) -> AmplitudeEstimate, a: f64, seeds: u64) -> (f64, f64) {
        let mut sq = 0.0;
        let mut calls = 0;
        for s in 0..seeds {
            let r = f(s);
            sq += (r.estimate - a).powi(2);
            calls = r.oracle_calls;
        }
        ((sq / seeds as f64).sqrt(), calls as f64)
    }

    #[test]
    fn maximum_likelihood_error_falls_as_one_over_the_oracle_calls() {
        // Slope of log(RMSE) against log(N_q): −1 with growing powers
        // (Heisenberg scaling), −1/2 with more shots of k = 0 only.
        let a = 0.3;
        let problem = AmplitudeProblem::bernoulli(a).unwrap();
        let fit = |points: &[(f64, f64)]| {
            let n = points.len() as f64;
            let (mx, my) = points
                .iter()
                .fold((0.0, 0.0), |s, (x, y)| (s.0 + x.ln() / n, s.1 + y.ln() / n));
            let (sxy, sxx) = points.iter().fold((0.0, 0.0), |s, (x, y)| {
                (
                    s.0 + (x.ln() - mx) * (y.ln() - my),
                    s.1 + (x.ln() - mx).powi(2),
                )
            });
            sxy / sxx
        };
        let quantum: Vec<(f64, f64)> = (3..9)
            .map(|levels| {
                let (e, n) = rmse(
                    |s| maximum_likelihood(&problem, levels, 100, 0.95, s).unwrap(),
                    a,
                    200,
                );
                (n, e)
            })
            .collect();
        let classical: Vec<(f64, f64)> = [100, 400, 1600, 6400]
            .iter()
            .map(|&shots| {
                let (e, n) = rmse(
                    |s| maximum_likelihood(&problem, 1, shots, 0.95, s).unwrap(),
                    a,
                    200,
                );
                (n, e)
            })
            .collect();
        let (sq, sc) = (fit(&quantum), fit(&classical));
        assert!((sq + 1.0).abs() < 0.2, "quantum slope {sq}");
        assert!((sc + 0.5).abs() < 0.1, "classical slope {sc}");
        // The Fisher interval covers in about 95 % of runs.
        let covered = (0..400)
            .filter(|&s| {
                let r = maximum_likelihood(&problem, 6, 100, 0.95, s).unwrap();
                r.interval.0 <= a && a <= r.interval.1
            })
            .count();
        assert!(covered >= 360, "{covered}/400");
    }

    #[test]
    fn invalid_arguments() {
        assert!(AmplitudeProblem::bernoulli(1.5).is_err());
        assert!(AmplitudeProblem::new(Circuit::new(1), 1).is_err());
        let p = AmplitudeProblem::bernoulli(0.2).unwrap();
        assert!(canonical(&p, 0).is_err());
        assert!(iterative(&p, 0.0, 0.05, 10, 0).is_err());
        assert!(maximum_likelihood(&p, 0, 10, 0.95, 0).is_err());
        assert!(p.probabilities(&[3, 1]).is_err());
    }
}
