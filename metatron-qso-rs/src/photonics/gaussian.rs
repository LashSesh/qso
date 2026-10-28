//! Gaussian states of light and Gaussian boson sampling.
//!
//! A Gaussian state of `m` modes is given by its covariance matrix `V` and
//! means `r̄` of the quadratures `(x₁…x_m, p₁…p_m)`, with `ħ = 2`: the
//! vacuum has `V = I`, and `x = a + a†`, `p = −i(a − a†)`. These are the
//! conventions of The Walrus and Strawberry Fields, which the
//! cross-validation compares against.
//!
//! Photon-number probabilities (Hamilton et al. 2017; Quesada et al.
//! 2019) are loop hafnians,
//!
//! ```text
//! p(n) = exp(−½ β Q⁻¹ β*) / (√det Q · Π n_i!) · lhaf(Ã_n)
//! ```
//!
//! with `Q = σ + I/2` (σ the covariance of `(a, a†)`), `A = X(I − Q⁻¹)*`,
//! `β = (⟨a⟩, ⟨a⟩*)`, and `Ã_n` the rows and columns of `A` repeated as
//! `(n, n)` with the diagonal replaced by `γ = β* − Aβ`. Threshold
//! detectors click with probability `Tor(O_S)/√det Q`, `O = I − Q⁻¹`
//! (Quesada, Arrazola and Killoran 2018).

use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, RMatrix};
use crate::{C64, Rng};

use super::hafnian::{MAX_HAFNIAN_DIM, loop_hafnian, torontonian};

/// Most modes of a Gaussian state.
pub const MAX_GAUSSIAN_MODES: usize = 1024;

/// Most photons in one probability: the loop hafnian of a mixed state has
/// dimension twice the photon number.
pub const MAX_GAUSSIAN_PHOTONS: usize = MAX_HAFNIAN_DIM / 2;

/// Most clicking detectors in one click probability.
pub const MAX_CLICKS: usize = MAX_HAFNIAN_DIM / 2;

/// A Gaussian state (covariance and means in `xxpp` order, `ħ = 2`).
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianState {
    n: usize,
    cov: RMatrix,
    means: Vec<f64>,
}

fn check_modes(n: usize) -> Result<()> {
    if n == 0 || n > MAX_GAUSSIAN_MODES {
        return Err(QsoError::invalid(
            "n_modes",
            format!("a Gaussian state has 1 to {MAX_GAUSSIAN_MODES} modes"),
        ));
    }
    Ok(())
}

fn finite(name: &'static str, v: f64) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(QsoError::invalid(name, "must be finite"))
    }
}

impl GaussianState {
    /// The vacuum of `n_modes` modes.
    pub fn vacuum(n_modes: usize) -> Result<Self> {
        check_modes(n_modes)?;
        Ok(Self {
            n: n_modes,
            cov: RMatrix::identity(2 * n_modes, 2 * n_modes),
            means: vec![0.0; 2 * n_modes],
        })
    }

    /// Thermal states with mean photon numbers `nbar` (one per mode).
    pub fn thermal(nbar: &[f64]) -> Result<Self> {
        let mut s = Self::vacuum(nbar.len())?;
        for (i, &v) in nbar.iter().enumerate() {
            if !(v.is_finite() && v >= 0.0) {
                return Err(QsoError::invalid(
                    "nbar",
                    "mean photon numbers must be finite and non-negative",
                ));
            }
            s.cov[(i, i)] = 2.0 * v + 1.0;
            s.cov[(i + s.n, i + s.n)] = 2.0 * v + 1.0;
        }
        Ok(s)
    }

    /// A state from its covariance matrix and means. `V` must be symmetric
    /// and satisfy the uncertainty principle `V + iΩ ≥ 0`.
    pub fn from_parts(cov: RMatrix, means: Vec<f64>) -> Result<Self> {
        let dim = cov.nrows();
        if dim % 2 == 1 || cov.ncols() != dim {
            return Err(QsoError::invalid(
                "cov",
                "the covariance matrix must be 2m × 2m",
            ));
        }
        let n = dim / 2;
        check_modes(n)?;
        if means.len() != dim {
            return Err(QsoError::DimensionMismatch {
                expected: dim,
                actual: means.len(),
            });
        }
        if cov.iter().chain(&means).any(|v| !v.is_finite()) {
            return Err(QsoError::invalid("cov", "entries must be finite"));
        }
        let scale = cov.amax().max(1.0);
        if (&cov - cov.transpose()).amax() > 1e-12 * scale {
            return Err(QsoError::invalid(
                "cov",
                "the covariance matrix is not symmetric",
            ));
        }
        let cov = (&cov + cov.transpose()) * 0.5;
        // V + iΩ ≥ 0 (Ω = [[0, I], [−I, 0]]) ⟺ V > 0 and every symplectic
        // eigenvalue ν ≥ 1; with V = L Lᵀ, ±iν are the eigenvalues of
        // Lᵀ Ω L. Rounding V to doubles moves ν by up to ε·κ(V) (a state
        // squeezed by r = 8 has κ = e^{32} and ν uncertain by 10⁻²), so
        // that is the tolerance.
        let (values, _) = crate::linalg::eigh_real(&cov);
        let chol = cov
            .clone()
            .cholesky()
            .filter(|_| values[0] > 0.0)
            .ok_or_else(|| {
                QsoError::invalid("cov", "the covariance matrix must be positive definite")
            })?;
        let l = chol.l();
        let h = CMatrix::from_fn(dim, dim, |r, c| {
            // i (Lᵀ Ω L)_rc = i Σ_k (L_{k,r} L_{k+n,c} − L_{k+n,r} L_{k,c}).
            let v: f64 = (0..n)
                .map(|k| l[(k, r)] * l[(k + n, c)] - l[(k + n, r)] * l[(k, c)])
                .sum();
            C64::new(0.0, v)
        });
        let nu = crate::linalg::eigh(&h)
            .0
            .iter()
            .fold(f64::INFINITY, |m, v| m.min(v.abs()));
        let kappa = values[dim - 1] / values[0];
        if nu < 1.0 - 1e-10 - f64::EPSILON * kappa {
            return Err(QsoError::invalid(
                "cov",
                format!(
                    "the covariance matrix violates the uncertainty principle (symplectic eigenvalue {nu:.6} < 1)"
                ),
            ));
        }
        Ok(Self { n, cov, means })
    }

    pub fn n_modes(&self) -> usize {
        self.n
    }

    /// Covariance matrix (`xxpp`, `ħ = 2`).
    pub fn cov(&self) -> &RMatrix {
        &self.cov
    }

    /// Means `(⟨x₁⟩…⟨x_m⟩, ⟨p₁⟩…⟨p_m⟩)`.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    fn check_mode(&self, mode: usize) -> Result<()> {
        if mode >= self.n {
            return Err(QsoError::invalid(
                "mode",
                format!("mode {mode} is outside the {} modes", self.n),
            ));
        }
        Ok(())
    }

    /// Apply the symplectic matrix `s` (`2k × 2k`, `xxpp` order of
    /// `modes`): `V ← S V Sᵀ`, `r̄ ← S r̄`, touching only the rows and
    /// columns of `modes`.
    fn apply_symplectic(&mut self, modes: &[usize], s: &RMatrix) {
        let idx: Vec<usize> = modes
            .iter()
            .copied()
            .chain(modes.iter().map(|&m| m + self.n))
            .collect();
        let (k2, dim) = (idx.len(), 2 * self.n);
        // Rows: (S V)[idx, :].
        let rows = RMatrix::from_fn(k2, dim, |a, col| {
            (0..k2).map(|b| s[(a, b)] * self.cov[(idx[b], col)]).sum()
        });
        for (a, &i) in idx.iter().enumerate() {
            for col in 0..dim {
                self.cov[(i, col)] = rows[(a, col)];
            }
        }
        // Columns: (S V Sᵀ)[:, idx].
        let cols = RMatrix::from_fn(dim, k2, |row, a| {
            (0..k2).map(|b| self.cov[(row, idx[b])] * s[(a, b)]).sum()
        });
        for row in 0..dim {
            for (a, &i) in idx.iter().enumerate() {
                self.cov[(row, i)] = cols[(row, a)];
            }
        }
        let old: Vec<f64> = idx.iter().map(|&i| self.means[i]).collect();
        for (a, &i) in idx.iter().enumerate() {
            self.means[i] = (0..k2).map(|b| s[(a, b)] * old[b]).sum();
        }
    }

    /// Squeeze `mode` by `r` at angle `phi` (`exp(½(ζ* a² − ζ a†²))`,
    /// `ζ = r e^{iφ}`); the squeezed vacuum has `⟨n⟩ = sinh² r`.
    pub fn squeeze(&mut self, mode: usize, r: f64, phi: f64) -> Result<&mut Self> {
        self.check_mode(mode)?;
        let (r, phi) = (finite("r", r)?, finite("phi", phi)?);
        let (ch, sh) = (r.cosh(), r.sinh());
        let s = RMatrix::from_row_slice(
            2,
            2,
            &[
                ch - sh * phi.cos(),
                -sh * phi.sin(),
                -sh * phi.sin(),
                ch + sh * phi.cos(),
            ],
        );
        self.apply_symplectic(&[mode], &s);
        Ok(self)
    }

    /// Displace `mode` by `alpha` (`⟨a⟩ += α`).
    pub fn displace(&mut self, mode: usize, alpha: C64) -> Result<&mut Self> {
        self.check_mode(mode)?;
        finite("alpha", alpha.re)?;
        finite("alpha", alpha.im)?;
        self.means[mode] += 2.0 * alpha.re;
        self.means[mode + self.n] += 2.0 * alpha.im;
        Ok(self)
    }

    /// Phase rotation `a → e^{iθ} a` of `mode`.
    pub fn rotate(&mut self, mode: usize, theta: f64) -> Result<&mut Self> {
        self.check_mode(mode)?;
        let t = finite("theta", theta)?;
        let s = RMatrix::from_row_slice(2, 2, &[t.cos(), -t.sin(), t.sin(), t.cos()]);
        self.apply_symplectic(&[mode], &s);
        Ok(self)
    }

    /// The interferometer `a → U a` on `modes` (`U` unitary,
    /// `modes.len() × modes.len()`).
    pub fn interferometer(&mut self, u: &CMatrix, modes: &[usize]) -> Result<&mut Self> {
        let k = modes.len();
        if u.nrows() != k || u.ncols() != k {
            return Err(QsoError::DimensionMismatch {
                expected: k,
                actual: u.nrows(),
            });
        }
        for &m in modes {
            self.check_mode(m)?;
        }
        let mut seen = vec![false; self.n];
        for &m in modes {
            if std::mem::replace(&mut seen[m], true) {
                return Err(QsoError::invalid("modes", format!("mode {m} is repeated")));
            }
        }
        if super::unitarity_error(u) > 1e-10 {
            return Err(QsoError::invalid("unitary", "the matrix is not unitary"));
        }
        let s = RMatrix::from_fn(2 * k, 2 * k, |r, c| {
            let z = u[(r % k, c % k)];
            match (r < k, c < k) {
                (true, true) | (false, false) => z.re,
                (true, false) => -z.im,
                (false, true) => z.im,
            }
        });
        self.apply_symplectic(modes, &s);
        Ok(self)
    }

    /// Loss on `mode`: transmission `eta`, the rest replaced by a thermal
    /// environment with `nbar` photons (0 for pure loss).
    pub fn loss(&mut self, mode: usize, eta: f64, nbar: f64) -> Result<&mut Self> {
        self.check_mode(mode)?;
        if !(0.0..=1.0).contains(&eta) || !(nbar.is_finite() && nbar >= 0.0) {
            return Err(QsoError::invalid(
                "eta",
                "transmission must lie in [0, 1] and nbar be non-negative",
            ));
        }
        let t = eta.sqrt();
        for m in [mode, mode + self.n] {
            self.means[m] *= t;
            for j in 0..2 * self.n {
                self.cov[(m, j)] *= t;
                self.cov[(j, m)] *= t;
            }
            self.cov[(m, m)] += (1.0 - eta) * (2.0 * nbar + 1.0);
        }
        Ok(self)
    }

    /// Mean photon number of every mode.
    pub fn mean_photon_numbers(&self) -> Vec<f64> {
        let n = self.n;
        (0..n)
            .map(|i| {
                let (x, p) = (self.means[i], self.means[i + n]);
                (self.cov[(i, i)] + self.cov[(i + n, i + n)] + x * x + p * p) / 4.0 - 0.5
            })
            .collect()
    }

    /// The state of `modes` alone (partial trace over the others).
    pub fn reduced(&self, modes: &[usize]) -> Result<Self> {
        if modes.is_empty() {
            return Err(QsoError::invalid("modes", "at least one mode"));
        }
        for (k, &m) in modes.iter().enumerate() {
            self.check_mode(m)?;
            if modes[..k].contains(&m) {
                return Err(QsoError::invalid("modes", format!("mode {m} is repeated")));
            }
        }
        let idx: Vec<usize> = modes
            .iter()
            .copied()
            .chain(modes.iter().map(|&m| m + self.n))
            .collect();
        Ok(Self {
            n: modes.len(),
            cov: RMatrix::from_fn(idx.len(), idx.len(), |r, c| self.cov[(idx[r], idx[c])]),
            means: idx.iter().map(|&i| self.means[i]).collect(),
        })
    }

    /// `ln det Q = ln det((V + I)/2) = Σ ln((1 + λ_i)/2)` over the
    /// eigenvalues of `V`. A determinant of `Q` itself cancels digits when
    /// the modes are strongly squeezed (10⁻⁹ at r = 5).
    fn log_det_q(&self) -> f64 {
        crate::linalg::eigh_real(&self.cov)
            .0
            .iter()
            .map(|l| ((1.0 + l) / 2.0).ln())
            .sum()
    }

    /// `Q = σ + I/2`, `β = (⟨a⟩, ⟨a⟩*)`.
    fn q_and_beta(&self) -> (CMatrix, Vec<C64>) {
        let n = self.n;
        let v = &self.cov;
        let i = C64::new(0.0, 1.0);
        let block = |r: usize, c: usize| {
            let (x, xp, px, p) = (v[(r, c)], v[(r, c + n)], v[(r + n, c)], v[(r + n, c + n)]);
            // ⟨a_r† a_c⟩ + δ/2 and ⟨a_r a_c⟩ in terms of the quadratures.
            let adag_a = (x + p + i * (xp - px)) / 4.0;
            let a_a = (x - p + i * (xp + px)) / 4.0;
            (adag_a, a_a)
        };
        let q = CMatrix::from_fn(2 * n, 2 * n, |r, c| {
            let (rr, cc) = (r % n, c % n);
            let (adag_a, a_a) = block(rr, cc);
            let delta = if r == c {
                C64::new(0.5, 0.0)
            } else {
                C64::new(0.0, 0.0)
            };
            match (r < n, c < n) {
                (true, true) => adag_a + delta,
                (true, false) => a_a.conj(),
                (false, true) => a_a,
                (false, false) => adag_a.conj() + delta,
            }
        });
        let beta: Vec<C64> = (0..n)
            .map(|k| C64::new(self.means[k], self.means[k + n]) / 2.0)
            .collect();
        let beta = beta
            .iter()
            .copied()
            .chain(beta.iter().map(|z| z.conj()))
            .collect();
        (q, beta)
    }

    /// Probability of the photon numbers `pattern` (one per mode).
    pub fn probability(&self, pattern: &[usize]) -> Result<f64> {
        let n = self.n;
        if pattern.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: pattern.len(),
            });
        }
        let total: usize = pattern.iter().sum();
        if total > MAX_GAUSSIAN_PHOTONS {
            return Err(QsoError::invalid(
                "pattern",
                format!("at most {MAX_GAUSSIAN_PHOTONS} photons in total"),
            ));
        }
        let (q, beta) = self.q_and_beta();
        let qinv = invert_q(&q)?;
        let x = |r: usize| if r < n { r + n } else { r - n };
        // A = X (I − Q⁻¹)*, symmetric up to rounding (made exact here).
        let a = CMatrix::from_fn(2 * n, 2 * n, |r, c| {
            let id = if x(r) == c {
                C64::new(1.0, 0.0)
            } else {
                C64::new(0.0, 0.0)
            };
            (id - qinv[(x(r), c)]).conj()
        });
        let a = (&a + a.transpose()) * C64::new(0.5, 0.0);
        let gamma: Vec<C64> = (0..2 * n)
            .map(|r| beta[r].conj() - (0..2 * n).map(|c| a[(r, c)] * beta[c]).sum::<C64>())
            .collect();
        // A pure state has A = B ⊕ B* and γ = (g, g*): the loop hafnian
        // factorises into |lhaf(B, g)|², half the rows (more accurate and
        // exponentially faster).
        let scale = a.iter().map(|z| z.norm()).fold(1.0, f64::max);
        let pure = (0..n).all(|r| (0..n).all(|c| a[(r, c + n)].norm() <= 1e-13 * scale));
        let lhaf = if pure {
            let b = a.view((0, 0), (n, n)).into_owned();
            repeated_loop_hafnian(&b, &gamma[..n], pattern)?.norm_sqr()
        } else {
            let rpt: Vec<usize> = pattern.iter().chain(pattern).copied().collect();
            repeated_loop_hafnian(&a, &gamma, &rpt)?.re
        };
        let quad: C64 = (0..2 * n)
            .map(|r| {
                beta[r]
                    * (0..2 * n)
                        .map(|c| qinv[(r, c)] * beta[c].conj())
                        .sum::<C64>()
            })
            .sum();
        // exp(−½ β† Q⁻¹ β) / √det Q, through log det Q (det Q overflows
        // for hundreds of squeezed modes).
        let factor = (quad * -0.5).exp().re * (-0.5 * self.log_det_q()).exp();
        let factorials: f64 = pattern.iter().map(|&k| factorial(k)).product();
        let p = lhaf * factor / factorials;
        Ok(p.max(0.0))
    }

    /// Probability that exactly the detectors with `clicks[i]` click
    /// (threshold detectors; zero means only).
    pub fn click_probability(&self, clicks: &[bool]) -> Result<f64> {
        let n = self.n;
        if clicks.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: clicks.len(),
            });
        }
        let scale = self.cov.amax().max(1.0).sqrt();
        if self.means.iter().any(|&v| v.abs() > 1e-12 * scale) {
            return Err(QsoError::invalid(
                "state",
                "click probabilities are implemented for states with zero means",
            ));
        }
        let on: Vec<usize> = (0..n).filter(|&i| clicks[i]).collect();
        if on.len() > MAX_CLICKS {
            return Err(QsoError::invalid(
                "clicks",
                format!("at most {MAX_CLICKS} clicking detectors"),
            ));
        }
        let (q, _) = self.q_and_beta();
        let qinv = invert_q(&q)?;
        let idx: Vec<usize> = on
            .iter()
            .copied()
            .chain(on.iter().map(|&i| i + n))
            .collect();
        let o = CMatrix::from_fn(idx.len(), idx.len(), |r, c| {
            let id = if idx[r] == idx[c] {
                C64::new(1.0, 0.0)
            } else {
                C64::new(0.0, 0.0)
            };
            id - qinv[(idx[r], idx[c])]
        });
        let o = (&o + o.adjoint()) * C64::new(0.5, 0.0);
        let p = torontonian(&o)?.re * (-0.5 * self.log_det_q()).exp();
        Ok(p.max(0.0))
    }

    /// `shots` exact samples of the photon numbers, drawn mode by mode
    /// from the conditional distributions of the marginal states. A sample
    /// with more than `max_photons` photons is an error.
    pub fn sample(
        &self,
        shots: usize,
        max_photons: usize,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<usize>>> {
        let max_photons = max_photons.min(MAX_GAUSSIAN_PHOTONS);
        let marginals: Vec<GaussianState> = (1..=self.n)
            .map(|k| self.reduced(&(0..k).collect::<Vec<_>>()))
            .collect::<Result<_>>()?;
        (0..shots)
            .map(|_| {
                let mut pattern: Vec<usize> = Vec::with_capacity(self.n);
                let mut prefix = 1.0;
                for state in &marginals {
                    let u = rng.uniform() * prefix;
                    let used: usize = pattern.iter().sum();
                    let mut cumulative = 0.0;
                    let mut chosen = None;
                    pattern.push(0);
                    for k in 0..=max_photons - used {
                        *pattern.last_mut().expect("pushed") = k;
                        let p = state.probability(&pattern)?;
                        cumulative += p;
                        if cumulative >= u {
                            chosen = Some(p);
                            break;
                        }
                    }
                    match chosen {
                        Some(p) => prefix = p,
                        None => {
                            return Err(QsoError::invalid(
                                "max_photons",
                                format!("a sample has more than {max_photons} photons"),
                            ));
                        }
                    }
                }
                Ok(pattern)
            })
            .collect()
    }

    /// `shots` exact samples of threshold-detector clicks (zero means).
    pub fn sample_clicks(&self, shots: usize, rng: &mut Rng) -> Result<Vec<Vec<bool>>> {
        let marginals: Vec<GaussianState> = (1..=self.n)
            .map(|k| self.reduced(&(0..k).collect::<Vec<_>>()))
            .collect::<Result<_>>()?;
        (0..shots)
            .map(|_| {
                let mut clicks: Vec<bool> = Vec::with_capacity(self.n);
                let mut prefix = 1.0;
                for state in &marginals {
                    clicks.push(false);
                    let p0 = state.click_probability(&clicks)?;
                    if rng.uniform() * prefix < p0 {
                        prefix = p0;
                    } else {
                        *clicks.last_mut().expect("pushed") = true;
                        prefix = (prefix - p0).max(0.0);
                        if clicks.iter().filter(|&&c| c).count() > MAX_CLICKS {
                            return Err(QsoError::invalid(
                                "clicks",
                                format!("a sample has more than {MAX_CLICKS} clicks"),
                            ));
                        }
                    }
                }
                Ok(clicks)
            })
            .collect()
    }
}

/// `Q⁻¹` by Cholesky (`Q` is Hermitian positive definite); nalgebra's
/// closed-form inverse of small matrices loses digits when the modes are
/// strongly squeezed (10⁻⁹ at r = 5).
fn invert_q(q: &CMatrix) -> Result<CMatrix> {
    q.clone().cholesky().map(|c| c.inverse()).ok_or_else(|| {
        QsoError::invalid(
            "state",
            "Q is not positive definite (not a valid Gaussian state)",
        )
    })
}

/// Loop hafnian of `a` with rows and columns repeated `rpt` times and the
/// diagonal replaced by `diag` (repeated likewise). Up to 20 rows, or
/// when it is cheaper, the general loop hafnian of the repeated matrix
/// (`2^{N/2}` terms of `O(N⁴)`); otherwise Kan's formula (`Π (rpt_i + 1)`
/// terms), which is fast for many photons in few modes but loses digits
/// to cancellation as the photon number grows.
fn repeated_loop_hafnian(a: &CMatrix, diag: &[C64], rpt: &[usize]) -> Result<C64> {
    let total: usize = rpt.iter().sum();
    let kan_terms: f64 = rpt.iter().map(|&k| (k + 1) as f64).product();
    let occupied = rpt.iter().filter(|&&k| k > 0).count() as f64;
    let kan_cost = kan_terms * occupied * occupied;
    let general_cost = 2f64.powf(total as f64 / 2.0) * (total as f64).powi(4);
    if total <= 20 || general_cost <= kan_cost {
        let rows: Vec<usize> = (0..rpt.len())
            .flat_map(|k| std::iter::repeat_n(k, rpt[k]))
            .collect();
        let m = rows.len();
        let mut sub = CMatrix::from_fn(m, m, |r, c| a[(rows[r], rows[c])]);
        for r in 0..m {
            sub[(r, r)] = diag[rows[r]];
        }
        loop_hafnian(&sub)
    } else {
        super::hafnian::loop_hafnian_repeated(a, diag, rpt)
    }
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|i| i as f64).product()
}

#[cfg(test)]
mod tests;
