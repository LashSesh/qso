//! Transport to a trap: efficiency and mean arrival time of an open walk.
//!
//! An open walk loses its walker to traps (sinks) and to losses
//! (recombination) while dephasing destroys coherences node by node. The
//! density matrix obeys
//!
//! ```text
//! dρ/dt = −i(H_eff ρ − ρ H_eff†) + Σ_j γ_j (P_j ρ P_j − ½{P_j, ρ})
//! H_eff = H − i Σ_t κ_t P_t − (i/2) Σ_j Γ_j P_j,   H = γ_hop·L or −γ_hop·A, plus diag(ε)
//! sink:  dp_t/dt = 2κ_t ρ_tt      loss: dp_loss/dt = Σ_j Γ_j ρ_jj
//! ```
//!
//! with `P_j = |j⟩⟨j|`, trap rates `κ_t`, loss rates `Γ_j`, dephasing rates
//! `γ_j` and site energies `ε_j`. The entries of `ρ` decay and rotate as
//!
//! | Entry | Decays at | Rotates at |
//! |---|---|---|
//! | `ρ_aa` | `2κ_a + Γ_a` | — |
//! | `ρ_ab` (`a ≠ b`) | `k_a + k_b + (γ_a + γ_b)/2`, `k = κ + Γ/2` | `−i(ε_a − ε_b)` |
//!
//! A uniform rate `γ_j = γ` is exactly the `rate` of
//! [`DephasingWalk`](super::DephasingWalk) and of `Dephasing::sites`.
//! Conventions of the literature convert as follows:
//!
//! * a dephasing term `2γ_C(PρP − ½{P, ρ})` (Caruso, Plenio and co-workers)
//!   is the rate `γ = 2γ_C` here;
//! * a loss term `−iΓ_R P` in the effective Hamiltonian (Rebentrost and
//!   co-workers) is the loss rate `Γ = 2Γ_R` here;
//! * a sink fed at `κ′ρ_tt` is the trap rate `κ = κ′/2` here.
//!
//! The integrals `X = ∫₀^∞ ρ dt` and `Y = ∫₀^∞ tρ dt` solve the linear
//! equations `𝓛(X) = −ρ₀` and `𝓛(Y) = −X`. From them come the efficiency
//! `η = 2Σ_t κ_t X_tt`, the population lost `Σ_j Γ_j X_jj` and the mean
//! arrival time of the trapped walker `τ = 2Σ_t κ_t Y_tt/η`. The rest of the
//! start's weight is never absorbed (it lies on dark states), so
//! `η + lost + dark = 1`; when every node loses population, nothing stays
//! dark and that weight counts as lost.
//!
//! The solvers never integrate in time: the dense path
//! ([`DephasingWalk::transport`](super::DephasingWalk::transport)) restricts
//! the problem to the bright states (the smallest space that holds the traps
//! and the lossy nodes and is closed under `H` and the dephasing), solves
//! the coherent part on one complex Schur form and the populations that
//! dephasing feeds by GMRES, and refines the result. Every solve reports its
//! backward error and a condition estimate ([`SolveReport`]).
//!
//! A detuned dimer (start at energy 3, trap at energy 0, loss 1/20 on both
//! sites, dephasing 1) reaches the trap with the exact efficiency
//! 32000/38881 after the mean time 139260/38881:
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::{DephasingWalk, Generator, TransportOptions};
//!
//! let walk = DephasingWalk::new(&generators::path(2), Generator::Adjacency, 1.0, 1.0)?
//!     .with_site_energies(&[3.0, 0.0])?
//!     .with_traps(&[(1, 1.0)])?
//!     .with_losses(&[0.05; 2])?;
//! let result = walk.transport(0, &TransportOptions::default())?;
//! assert!((result.efficiency - 32000.0 / 38881.0).abs() < 1e-12);
//! let tau = result.arrival_time.unwrap();
//! assert!((tau / (139260.0 / 38881.0) - 1.0).abs() < 1e-10);
//! # Ok::<(), metatron_qso::QsoError>(())
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};

pub(crate) mod dense;

#[cfg(test)]
mod tests;

/// Most estimated operations (8 per complex multiply-add) of one transport
/// call: GMRES iterations of every solve plus the orbital factorisation.
/// About 20 minutes on one thread at the measured 1.8·10⁹ complex
/// multiply-adds per second.
pub const MAX_TRANSPORT_WORK: f64 = 2e13;

/// Most complex entries (16 bytes, 2 GiB) of the banded LU that
/// preconditions the orbital solves: hypercubes up to d = 61 (measured:
/// 1.91 GiB, 47 s at d = 61).
pub const MAX_TRANSPORT_FACTOR_ENTRIES: usize = 1 << 27;

/// Settings of the transport solves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TransportOptions {
    /// Dense path: a state coupled to the traps and to nodes with more than
    /// the smallest loss more weakly than this (relative to the largest
    /// `|H_ij − c δ_ij|`) counts as dark. Orbitals: dark weights below it
    /// are reported as 0 (the rest goes to `error`). Default 1e-10, in
    /// (0, 1e-2].
    pub dark_tolerance: f64,
    /// GMRES stops at `‖b − Ax‖ ≤ tolerance·(‖b‖ + ‖A‖‖x‖)`; the dense path
    /// refines while `𝓛(X) + ρ₀` is above it. Default 1e-13, in
    /// [1e-16, 1e-6].
    pub tolerance: f64,
    /// Most GMRES iterations per solve. Default 1000, at least 1.
    pub max_iterations: usize,
}

impl Default for TransportOptions {
    fn default() -> Self {
        TransportOptions {
            dark_tolerance: 1e-10,
            tolerance: 1e-13,
            max_iterations: 1000,
        }
    }
}

impl TransportOptions {
    /// Check the ranges documented on the fields.
    pub(crate) fn validate(&self) -> Result<()> {
        if !(self.dark_tolerance > 0.0 && self.dark_tolerance <= 1e-2) {
            return Err(QsoError::invalid(
                "dark_tolerance",
                "the dark tolerance must lie in (0, 1e-2]",
            ));
        }
        if !(1e-16..=1e-6).contains(&self.tolerance) {
            return Err(QsoError::invalid(
                "tolerance",
                "the tolerance must lie in [1e-16, 1e-6]",
            ));
        }
        if self.max_iterations == 0 {
            return Err(QsoError::invalid(
                "max_iterations",
                "at least one GMRES iteration",
            ));
        }
        Ok(())
    }
}

/// How one linear solve of a transport calculation went.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SolveReport {
    /// `"schur"`, `"X"`, `"Y"` (dense); `"adjoint"`, `"adjoint loss"`,
    /// `"forward"` (orbitals).
    pub name: String,
    /// GMRES iterations, refinements included (0 for direct solves).
    pub iterations: usize,
    /// `‖b − Ax‖/(‖A‖‖x‖ + ‖b‖)`; for `"schur"`: `‖H′_B − UTU†‖_F/‖H′_B‖_F`.
    pub backward_error: f64,
    /// Lower estimate of the condition number: `‖A‖‖x‖/‖b‖` (`"schur"`:
    /// `‖𝓛‖/slowest_decay`).
    pub condition: f64,
}

/// Efficiency, arrival time and the rest of a transport calculation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TransportResult {
    /// `η = 2Σ_t κ_t X_tt`, the population that reaches the traps.
    pub efficiency: f64,
    /// Mean arrival time of the trapped walker, `τ = 2Σ_t κ_t Y_tt/η`;
    /// `None` when `η < max(1e-12, 1e3·error)`, where it is not accurate.
    pub arrival_time: Option<f64>,
    /// `2κ_t X_tt` per trap in the order of `traps()` (hypercube: per
    /// listed weight, all traps of that weight together). Sums to `η`
    /// (orbitals: within `error`).
    pub trapped: Vec<f64>,
    /// `Σ_j Γ_j X_jj` (plus the weight outside the bright states when every
    /// node loses population).
    pub lost: f64,
    /// The start's weight never absorbed; 0 when every node loses
    /// population.
    pub dark: f64,
    /// Uniform dephasing rate added to every node for this result (0 for
    /// `transport`).
    pub added_dephasing: f64,
    /// Bright dimension `r` (dense) or number of orbitals.
    pub dimension: usize,
    /// Dense: `2·min(−Im λ(H′_B))` plus the added rate, the slowest decay
    /// without dephasing jumps on the bright states (an almost dark state
    /// shows here). Orbitals: `None`.
    pub slowest_decay: Option<f64>,
    /// Largest `condition` of `solves`.
    pub condition: f64,
    /// Estimated absolute error of `η`.
    pub error: f64,
    /// Every linear solve of the calculation, in order.
    pub solves: Vec<SolveReport>,
}

/// The uniform added dephasing rate of highest efficiency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DephasingOptimum {
    /// The uniform added rate of highest efficiency
    /// (`= result.added_dephasing`).
    pub rate: f64,
    /// False if the best grid rate was an end of `[low, high]` or `η` was
    /// flat.
    pub interior: bool,
    /// Solves spent (grid, golden section and the final solve).
    pub evaluations: usize,
    /// The full result at `rate`.
    pub result: TransportResult,
}

/// The walk at one time of a transport trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TransportState {
    /// The time.
    pub time: f64,
    /// Population per node (hypercube: per Hamming weight `0..=d`).
    pub populations: Vec<f64>,
    /// Sink population of each trap entry at `time` (same order as
    /// `TransportResult::trapped`).
    pub trapped: Vec<f64>,
    /// Population lost by `time`.
    pub lost: f64,
}

/// Traps `(site, κ)`: sites below `sites`, rates positive and finite, no
/// site twice.
pub(crate) fn check_traps(traps: &[(usize, f64)], sites: usize) -> Result<()> {
    for (k, &(t, kappa)) in traps.iter().enumerate() {
        if t >= sites {
            return Err(QsoError::NodeOutOfRange {
                node: t,
                n_nodes: sites,
            });
        }
        if !(kappa.is_finite() && kappa > 0.0) {
            return Err(QsoError::invalid(
                "traps",
                "trap rates must be positive and finite",
            ));
        }
        if traps[..k].iter().any(|&(u, _)| u == t) {
            return Err(QsoError::invalid(
                "traps",
                format!("node {t} is a trap twice"),
            ));
        }
    }
    Ok(())
}

/// One rate per site, finite and non-negative (`name`: `"losses"` or
/// `"dephasing"`).
pub(crate) fn check_rates(name: &'static str, v: &[f64], sites: usize) -> Result<()> {
    if v.len() != sites {
        return Err(QsoError::DimensionMismatch {
            expected: sites,
            actual: v.len(),
        });
    }
    if v.iter().any(|&x| !(x.is_finite() && x >= 0.0)) {
        return Err(QsoError::invalid(name, "rates must be finite and ≥ 0"));
    }
    Ok(())
}

/// One finite energy per site.
pub(crate) fn check_energies(v: &[f64], sites: usize) -> Result<()> {
    if v.len() != sites {
        return Err(QsoError::DimensionMismatch {
            expected: sites,
            actual: v.len(),
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(QsoError::invalid(
            "energies",
            "site energies must be finite",
        ));
    }
    Ok(())
}

/// The added rates of a scan: finite and non-negative.
pub(crate) fn check_scan_rates(r: &[f64]) -> Result<()> {
    if r.iter().any(|&x| !(x.is_finite() && x >= 0.0)) {
        return Err(QsoError::invalid(
            "rates",
            "dephasing rates must be finite and ≥ 0",
        ));
    }
    Ok(())
}

/// The times of a trace: finite, non-negative and non-decreasing.
pub(crate) fn check_times(t: &[f64]) -> Result<()> {
    let ordered = t.windows(2).all(|w| w[0] <= w[1]);
    if !ordered || t.iter().any(|&x| !(x.is_finite() && x >= 0.0)) {
        return Err(QsoError::invalid(
            "times",
            "times must be finite, ≥ 0 and non-decreasing",
        ));
    }
    Ok(())
}

/// `τ = ητ/η`, or `None` when `η < max(1e-12, 1e3·error)`: below that the
/// arrival time changed sign or by orders of magnitude between equivalent
/// solver paths (cycles of 64 and 256 nodes under strong dephasing).
pub(crate) fn arrival_time(eta: f64, eta_tau: f64, error: f64) -> Option<f64> {
    if eta.is_nan() || eta < 1e-12f64.max(1e3 * error) {
        return None;
    }
    Some(eta_tau / eta)
}

/// Grid points of [`optimise`]: `max(13, ⌈4·log₁₀(high/low)⌉ + 1)`.
fn grid_points(low: f64, high: f64) -> usize {
    let decades = (high / low).log10();
    13usize.max((4.0 * decades).ceil() as usize + 1)
}

/// Golden-section steps of [`optimise`] at most.
const GOLDEN_STEPS: usize = 200;

/// Width in `ln γ` at which [`optimise`]'s golden section stops.
const GOLDEN_WIDTH: f64 = 1e-8;

/// The added uniform rate in `[low, high]` of highest efficiency: `eta(e)`
/// gives the efficiency alone, `full(e)` the full result. A log-spaced grid
/// of `max(13, ⌈4·log₁₀(high/low)⌉ + 1)` rates (ends included) brackets the
/// best rate; golden section on `ln γ` narrows the bracket to a width of
/// 1e-8, and one full solve at its midpoint gives the result. If no grid
/// rate beats another by more than `1e-12·max(1, η)`, the efficiency is
/// flat and the result is the one at `low`, with `interior = false`.
pub(crate) fn optimise(
    low: f64,
    high: f64,
    mut eta: impl FnMut(f64) -> Result<f64>,
    mut full: impl FnMut(f64) -> Result<TransportResult>,
) -> Result<DephasingOptimum> {
    if !(low.is_finite() && high.is_finite() && low > 0.0 && low < high) {
        return Err(QsoError::invalid(
            "range",
            "need 0 < low < high, both finite",
        ));
    }
    let k = grid_points(low, high);
    let (a, b) = (low.ln(), high.ln());
    let xs: Vec<f64> = (0..k)
        .map(|i| a + (b - a) * i as f64 / (k - 1) as f64)
        .collect();
    let mut evaluations = 0;
    let mut etas = Vec::with_capacity(k);
    for (i, &x) in xs.iter().enumerate() {
        let rate = match i {
            0 => low,
            _ if i == k - 1 => high,
            _ => x.exp(),
        };
        etas.push(eta(rate)?);
        evaluations += 1;
    }
    let mut best = 0;
    for (i, &e) in etas.iter().enumerate() {
        if e > etas[best] {
            best = i;
        }
    }
    let top = etas[best];
    if etas.iter().all(|&e| top - e <= 1e-12 * top.max(1.0)) {
        let result = full(low)?;
        return Ok(DephasingOptimum {
            rate: low,
            interior: false,
            evaluations: evaluations + 1,
            result,
        });
    }
    // Golden section for the maximum of η(e^u) on the bracket.
    let (mut lo, mut hi) = (xs[best.saturating_sub(1)], xs[(best + 1).min(k - 1)]);
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let mut f = |u: f64, evaluations: &mut usize| -> Result<f64> {
        *evaluations += 1;
        eta(u.exp().clamp(low, high))
    };
    let mut c = hi - ratio * (hi - lo);
    let mut d = lo + ratio * (hi - lo);
    let mut fc = f(c, &mut evaluations)?;
    let mut fd = f(d, &mut evaluations)?;
    let mut steps = 0;
    while hi - lo > GOLDEN_WIDTH && steps < GOLDEN_STEPS {
        steps += 1;
        if fc >= fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - ratio * (hi - lo);
            fc = f(c, &mut evaluations)?;
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + ratio * (hi - lo);
            fd = f(d, &mut evaluations)?;
        }
    }
    let rate = (0.5 * (lo + hi)).exp().clamp(low, high);
    let result = full(rate)?;
    Ok(DephasingOptimum {
        rate,
        interior: 0 < best && best < k - 1,
        evaluations: evaluations + 1,
        result,
    })
}
