//! Mixed states and noise.
//!
//! [`DensityMatrix`] stores ρ for up to [`MAX_DENSITY_QUBITS`] qubits as a
//! vector of 4ⁿ amplitudes with `vec(ρ)[r + (c << n)] = ρ[r, c]`. In this
//! layout `ρ ↦ UρU†` is the action of `U ⊗ U*` on a 2n-qubit "state", so the
//! statevector kernels are reused unchanged, and a single-qubit channel with
//! Kraus operators `{K_i}` is the 4×4 superoperator `Σ K_i ⊗ K_i*` on the
//! (row, column) pair of its qubit.
//!
//! ```
//! use metatron_qso::noise::{Channel, DensityMatrix};
//! use metatron_qso::quantum::Circuit;
//!
//! let mut c = Circuit::new(1);
//! c.x(0);
//! let mut rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
//! rho.apply_channel(&Channel::amplitude_damping(0.25).unwrap(), &[0]).unwrap();
//! assert!((rho.probabilities()[1] - 0.75).abs() < 1e-12);
//! ```

pub mod device;
pub mod trajectories;

pub use device::{ChannelSpec, DeviceModel, GateRule, QubitProperties};
pub use trajectories::TrajectorySimulator;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::parallel::sum_f64;
use crate::quantum::kernels;
use crate::quantum::{Circuit, Counts, PauliSum, StateVector};
use crate::rng::{Rng, sample_cumulative};

/// Largest register a [`DensityMatrix`] may hold (4¹⁴ entries = 4 GiB).
pub const MAX_DENSITY_QUBITS: usize = 14;

const ZERO: C64 = C64::new(0.0, 0.0);

fn c(re: f64) -> C64 {
    C64::new(re, 0.0)
}

/// A completely positive, trace-preserving map given by Kraus operators on
/// one or two qubits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    name: String,
    n_qubits: usize,
    /// Row-major Kraus matrices (dimension 2^n_qubits).
    kraus: Vec<Vec<C64>>,
}

fn check_probability(name: &'static str, p: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return Err(QsoError::invalid(name, format!("{p} is not a probability")));
    }
    Ok(())
}

impl Channel {
    /// Channel from explicit Kraus operators (checked for completeness
    /// `Σ K†K = I` to 1e-9).
    pub fn from_kraus(name: impl Into<String>, kraus: Vec<Vec<C64>>) -> Result<Self> {
        let first = kraus
            .first()
            .ok_or_else(|| QsoError::invalid("kraus", "at least one Kraus operator required"))?;
        let d = (first.len() as f64).sqrt().round() as usize;
        let n_qubits = match d {
            2 => 1,
            4 => 2,
            _ => {
                return Err(QsoError::invalid(
                    "kraus",
                    "only 1- and 2-qubit channels are supported",
                ));
            }
        };
        let mut sum = vec![ZERO; d * d];
        for k in &kraus {
            if k.len() != d * d {
                return Err(QsoError::DimensionMismatch {
                    expected: d * d,
                    actual: k.len(),
                });
            }
            for r in 0..d {
                for cc in 0..d {
                    for m in 0..d {
                        sum[r * d + cc] += k[m * d + r].conj() * k[m * d + cc];
                    }
                }
            }
        }
        for r in 0..d {
            for cc in 0..d {
                let expected = if r == cc { 1.0 } else { 0.0 };
                if (sum[r * d + cc] - c(expected)).norm() > 1e-9 {
                    return Err(QsoError::invalid(
                        "kraus",
                        "operators are not trace preserving",
                    ));
                }
            }
        }
        Ok(Self {
            name: name.into(),
            n_qubits,
            kraus,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of qubits the channel acts on.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn kraus_operators(&self) -> &[Vec<C64>] {
        &self.kraus
    }

    /// Depolarising channel `ρ ↦ (1−p)ρ + p/3 (XρX + YρY + ZρZ)`.
    pub fn depolarizing(p: f64) -> Result<Self> {
        check_probability("p", p)?;
        let a = (1.0 - p).sqrt();
        let b = (p / 3.0).sqrt();
        let i = C64::new(0.0, 1.0);
        Self::from_kraus(
            format!("depolarizing({p})"),
            vec![
                vec![c(a), ZERO, ZERO, c(a)],
                vec![ZERO, c(b), c(b), ZERO],
                vec![ZERO, -i * b, i * b, ZERO],
                vec![c(b), ZERO, ZERO, c(-b)],
            ],
        )
    }

    /// Two-qubit depolarising channel `ρ ↦ (1−p)ρ + p/15 Σ_{P≠II} PρP`.
    pub fn depolarizing_2q(p: f64) -> Result<Self> {
        check_probability("p", p)?;
        let paulis = [
            [c(1.0), ZERO, ZERO, c(1.0)],
            [ZERO, c(1.0), c(1.0), ZERO],
            [ZERO, C64::new(0.0, -1.0), C64::new(0.0, 1.0), ZERO],
            [c(1.0), ZERO, ZERO, c(-1.0)],
        ];
        let mut kraus = Vec::with_capacity(16);
        for (a, pa) in paulis.iter().enumerate() {
            for (b, pb) in paulis.iter().enumerate() {
                let w = if a == 0 && b == 0 {
                    (1.0 - p).sqrt()
                } else {
                    (p / 15.0).sqrt()
                };
                // Kronecker product pa ⊗ pb, local order |x_first x_second⟩.
                let mut k = vec![ZERO; 16];
                for r in 0..4 {
                    for cc in 0..4 {
                        k[r * 4 + cc] =
                            pa[(r >> 1) * 2 + (cc >> 1)] * pb[(r & 1) * 2 + (cc & 1)] * w;
                    }
                }
                kraus.push(k);
            }
        }
        Self::from_kraus(format!("depolarizing_2q({p})"), kraus)
    }

    /// Amplitude damping (energy relaxation) with decay probability `gamma`.
    pub fn amplitude_damping(gamma: f64) -> Result<Self> {
        check_probability("gamma", gamma)?;
        Self::from_kraus(
            format!("amplitude_damping({gamma})"),
            vec![
                vec![c(1.0), ZERO, ZERO, c((1.0 - gamma).sqrt())],
                vec![ZERO, c(gamma.sqrt()), ZERO, ZERO],
            ],
        )
    }

    /// Phase damping: off-diagonal elements shrink by `√(1−λ)`.
    pub fn phase_damping(lambda: f64) -> Result<Self> {
        check_probability("lambda", lambda)?;
        Self::from_kraus(
            format!("phase_damping({lambda})"),
            vec![
                vec![c(1.0), ZERO, ZERO, c((1.0 - lambda).sqrt())],
                vec![ZERO, ZERO, ZERO, c(lambda.sqrt())],
            ],
        )
    }

    /// Bit flip with probability `p`.
    pub fn bit_flip(p: f64) -> Result<Self> {
        check_probability("p", p)?;
        let a = (1.0 - p).sqrt();
        let b = p.sqrt();
        Self::from_kraus(
            format!("bit_flip({p})"),
            vec![vec![c(a), ZERO, ZERO, c(a)], vec![ZERO, c(b), c(b), ZERO]],
        )
    }

    /// Phase flip with probability `p`.
    pub fn phase_flip(p: f64) -> Result<Self> {
        check_probability("p", p)?;
        let a = (1.0 - p).sqrt();
        let b = p.sqrt();
        Self::from_kraus(
            format!("phase_flip({p})"),
            vec![vec![c(a), ZERO, ZERO, c(a)], vec![c(b), ZERO, ZERO, c(-b)]],
        )
    }

    /// Thermal relaxation towards |0⟩ for a duration `time` with relaxation
    /// time `t1` and dephasing time `t2 ≤ 2·t1`: populations relax as
    /// `e^{−t/T1}`, coherences as `e^{−t/T2}`.
    pub fn thermal_relaxation(t1: f64, t2: f64, time: f64) -> Result<Self> {
        if !(t1 > 0.0 && t2 > 0.0 && time >= 0.0) {
            return Err(QsoError::invalid(
                "t1/t2/time",
                "T1, T2 must be positive, time non-negative",
            ));
        }
        if t2 > 2.0 * t1 * (1.0 + 1e-12) {
            return Err(QsoError::invalid(
                "t2",
                "physical channels require T2 ≤ 2·T1",
            ));
        }
        let gamma = 1.0 - (-time / t1).exp();
        // Amplitude damping alone gives coherence decay e^{−t/(2T1)}; the
        // remaining pure dephasing supplies e^{−t(1/T2 − 1/(2T1))}.
        let rate = (1.0 / t2 - 0.5 / t1).max(0.0);
        let lambda = 1.0 - (-2.0 * time * rate).exp();
        let a = (1.0 - gamma).sqrt();
        let b = (1.0 - lambda).sqrt();
        // Compose amplitude damping followed by phase damping.
        let ad = [
            [c(1.0), ZERO, ZERO, c(a)],
            [ZERO, c(gamma.sqrt()), ZERO, ZERO],
        ];
        let pd = [
            [c(1.0), ZERO, ZERO, c(b)],
            [ZERO, ZERO, ZERO, c(lambda.sqrt())],
        ];
        let mut kraus = Vec::new();
        for p in &pd {
            for q in &ad {
                kraus.push(vec![
                    p[0] * q[0] + p[1] * q[2],
                    p[0] * q[1] + p[1] * q[3],
                    p[2] * q[0] + p[3] * q[2],
                    p[2] * q[1] + p[3] * q[3],
                ]);
            }
        }
        Self::from_kraus(format!("thermal_relaxation({t1},{t2},{time})"), kraus)
    }

    /// Pauli channel `ρ ↦ Σ_P p_P PρP` from the probabilities of
    /// non-identity Pauli strings, given by labels such as `"X"` or `"ZZ"`
    /// (the first letter acts on the first qubit of the pair). The identity
    /// keeps the remaining probability.
    ///
    /// ```
    /// use metatron_qso::noise::Channel;
    ///
    /// let crosstalk = Channel::pauli(&[("ZZ", 0.02), ("XI", 0.001)]).unwrap();
    /// assert_eq!(crosstalk.n_qubits(), 2);
    /// assert!((crosstalk.pauli_probabilities().unwrap()[15] - 0.02).abs() < 1e-15);
    /// ```
    pub fn pauli(probabilities: &[(&str, f64)]) -> Result<Self> {
        let k = probabilities
            .first()
            .map(|(label, _)| label.len())
            .ok_or_else(|| QsoError::invalid("probabilities", "no Pauli strings given"))?;
        if !(1..=2).contains(&k) {
            return Err(QsoError::invalid(
                "probabilities",
                "Pauli labels must have one or two letters",
            ));
        }
        let mut p = vec![0.0; 1 << (2 * k)];
        for &(label, prob) in probabilities {
            let index = pauli_index(label, k)?;
            if index == 0 {
                return Err(QsoError::invalid(
                    "probabilities",
                    "the identity keeps the remaining probability; do not list it",
                ));
            }
            check_probability("probability", prob)?;
            if p[index] != 0.0 {
                return Err(QsoError::invalid(
                    "probabilities",
                    format!("{label} is listed twice"),
                ));
            }
            p[index] = prob;
        }
        let rest = 1.0 - p.iter().sum::<f64>();
        if rest < -1e-12 {
            return Err(QsoError::invalid(
                "probabilities",
                "probabilities sum to more than 1",
            ));
        }
        p[0] = rest.max(0.0);
        let name = probabilities
            .iter()
            .map(|(l, q)| format!("{}:{q}", l.to_ascii_uppercase()))
            .collect::<Vec<_>>()
            .join(",");
        Self::from_pauli_probabilities(format!("pauli({name})"), &p)
    }

    /// Pauli channel from all `4^k` probabilities (index digits in base 4,
    /// `I, X, Y, Z`, first qubit most significant).
    fn from_pauli_probabilities(name: String, p: &[f64]) -> Result<Self> {
        let k = if p.len() == 4 { 1 } else { 2 };
        let kraus = p
            .iter()
            .enumerate()
            .filter(|&(_, &q)| q > 0.0)
            .map(|(index, &q)| {
                pauli_matrix(k, index)
                    .into_iter()
                    .map(|z| z * q.sqrt())
                    .collect()
            })
            .collect();
        Self::from_kraus(name, kraus)
    }

    /// Unitary channel `ρ ↦ UρU†` for a 2×2 or 4×4 unitary (row-major), for
    /// coherent errors.
    pub fn unitary(name: impl Into<String>, matrix: Vec<C64>) -> Result<Self> {
        if !crate::quantum::gates::is_unitary(&matrix, 1e-9) {
            return Err(QsoError::invalid("matrix", "not unitary"));
        }
        Self::from_kraus(name, vec![matrix])
    }

    /// Coherent rotation `exp(−iθP/2)` about a Pauli string such as `"X"`,
    /// `"Z"` or `"ZZ"` (a systematic over-rotation or a ZZ crosstalk
    /// term).
    pub fn rotation(label: &str, angle: f64) -> Result<Self> {
        if !angle.is_finite() {
            return Err(QsoError::invalid("angle", "must be finite"));
        }
        let k = label.len();
        let index = pauli_index(label, k)?;
        let (c, s) = ((angle / 2.0).cos(), (angle / 2.0).sin());
        let d = 1usize << k;
        let pm = pauli_matrix(k, index);
        let matrix = (0..d * d)
            .map(|i| {
                let id = if i % (d + 1) == 0 { 1.0 } else { 0.0 };
                C64::new(c * id, 0.0) - C64::new(0.0, s) * pm[i]
            })
            .collect();
        Self::from_kraus(
            format!("r{}({angle})", label.to_ascii_lowercase()),
            vec![matrix],
        )
    }

    /// The channel `then ∘ self` (first `self`, then `then`, on the same
    /// qubits), with a minimal set of Kraus operators (at most `4^k`, from
    /// the eigendecomposition of the Choi matrix).
    pub fn compose(&self, then: &Channel) -> Result<Channel> {
        if self.n_qubits != then.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: then.n_qubits,
            });
        }
        let name = format!("{}∘{}", then.name, self.name);
        // Pauli channels compose to a Pauli channel (Pauli products are
        // XORs of the base-4 indices up to phase).
        if let (Some(p), Some(q)) = (self.pauli_probabilities(), then.pauli_probabilities()) {
            let mut r = vec![0.0; p.len()];
            for (a, pa) in p.iter().enumerate() {
                for (b, qb) in q.iter().enumerate() {
                    r[a ^ b] += pa * qb;
                }
            }
            return Self::from_pauli_probabilities(name, &r);
        }
        let d = 1usize << self.n_qubits;
        let mut products = Vec::with_capacity(self.kraus.len() * then.kraus.len());
        for b in &then.kraus {
            for a in &self.kraus {
                let mut m = vec![ZERO; d * d];
                for r in 0..d {
                    for cc in 0..d {
                        for k in 0..d {
                            m[r * d + cc] += b[r * d + k] * a[k * d + cc];
                        }
                    }
                }
                products.push(m);
            }
        }
        // Mixtures of unitaries stay mixtures of unitaries (trajectories
        // sample them without looking at the state); other channels get a
        // minimal Kraus set.
        let kraus = if self.is_unitary_mixture() && then.is_unitary_mixture() {
            products
        } else {
            minimal_kraus(&products, d)
        };
        Self::from_kraus(name, kraus)
    }

    /// Whether every Kraus operator is a multiple of a unitary.
    fn is_unitary_mixture(&self) -> bool {
        let d = 1usize << self.n_qubits;
        self.kraus.iter().all(|k| {
            let mut g = vec![ZERO; d * d];
            for r in 0..d {
                for cc in 0..d {
                    for m in 0..d {
                        g[r * d + cc] += k[m * d + r].conj() * k[m * d + cc];
                    }
                }
            }
            let w = g[0].re;
            (0..d * d).all(|i| {
                let expected = if i % (d + 1) == 0 { w } else { 0.0 };
                (g[i] - c(expected)).norm() <= 1e-12
            })
        })
    }

    /// Process (χ) matrix in the Pauli basis: `χ[P][Q] = Σ_i k_iP k_iQ*`
    /// with `K_i = Σ_P k_iP P`.
    fn chi_matrix(&self) -> Vec<C64> {
        let k = self.n_qubits;
        let d = 1usize << k;
        let m = d * d;
        let paulis: Vec<Vec<C64>> = (0..m).map(|p| pauli_matrix(k, p)).collect();
        let mut chi = vec![ZERO; m * m];
        for kr in &self.kraus {
            // k_P = Tr(P K) / d (Paulis are Hermitian).
            let coeff: Vec<C64> = paulis
                .iter()
                .map(|pm| {
                    let mut t = ZERO;
                    for r in 0..d {
                        for cc in 0..d {
                            t += pm[r * d + cc] * kr[cc * d + r];
                        }
                    }
                    t / d as f64
                })
                .collect();
            for p in 0..m {
                for q in 0..m {
                    chi[p * m + q] += coeff[p] * coeff[q].conj();
                }
            }
        }
        chi
    }

    /// Probabilities `p_P` (index digits in base 4, first qubit most
    /// significant) if this is a Pauli channel `ρ ↦ Σ_P p_P PρP`, else
    /// `None`.
    pub fn pauli_probabilities(&self) -> Option<Vec<f64>> {
        let chi = self.chi_matrix();
        let m = 1usize << (2 * self.n_qubits);
        for p in 0..m {
            for q in 0..m {
                if p != q && chi[p * m + q].norm() > 1e-12 {
                    return None;
                }
            }
        }
        Some((0..m).map(|p| chi[p * m + p].re.max(0.0)).collect())
    }

    /// The Pauli-twirled channel: the Pauli channel with probabilities
    /// `χ[P][P]`, which is what the channel becomes on average when random
    /// Paulis are applied before it and undone after it.
    pub fn twirled(&self) -> Channel {
        let chi = self.chi_matrix();
        let m = 1usize << (2 * self.n_qubits);
        let p: Vec<f64> = (0..m).map(|p| chi[p * m + p].re.max(0.0)).collect();
        Self::from_pauli_probabilities(format!("twirled({})", self.name), &p)
            .expect("the diagonal of χ is a probability distribution")
    }

    /// Superoperator `Σ K ⊗ K*` in the local order of the density-matrix
    /// kernels: (row bits, then column bits).
    pub(crate) fn superoperator(&self) -> Vec<C64> {
        let d = 1usize << self.n_qubits;
        let dd = d * d;
        let mut s = vec![ZERO; dd * dd];
        for k in &self.kraus {
            for r in 0..d {
                for cc in 0..d {
                    for r2 in 0..d {
                        for c2 in 0..d {
                            s[(r * d + cc) * dd + (r2 * d + c2)] +=
                                k[r * d + r2] * k[cc * d + c2].conj();
                        }
                    }
                }
            }
        }
        s
    }
}

/// Product `a·b` of square row-major matrices.
fn mat_product(a: &[C64], b: &[C64]) -> Vec<C64> {
    let d = (a.len() as f64).sqrt().round() as usize;
    let mut m = vec![ZERO; d * d];
    for r in 0..d {
        for k in 0..d {
            let x = a[r * d + k];
            if x == ZERO {
                continue;
            }
            for cc in 0..d {
                m[r * d + cc] += x * b[k * d + cc];
            }
        }
    }
    m
}

/// A minimal Kraus representation of the channel `Σ K ρ K†`: the
/// eigenvectors of the Choi matrix `Σ vec(K) vec(K)†` with non-negligible
/// eigenvalues.
fn minimal_kraus(kraus: &[Vec<C64>], d: usize) -> Vec<Vec<C64>> {
    let m = d * d;
    if kraus.len() <= 1 {
        return kraus.to_vec();
    }
    let mut choi = CMatrix::zeros(m, m);
    for k in kraus {
        for i in 0..m {
            for j in 0..m {
                choi[(i, j)] += k[i] * k[j].conj();
            }
        }
    }
    let (values, vectors) = crate::linalg::eigh(&choi);
    let top = values.iter().copied().fold(0.0, f64::max);
    values
        .iter()
        .enumerate()
        .rev()
        .filter(|&(_, &v)| v > 1e-14 * top)
        .map(|(j, &v)| {
            let k: Vec<C64> = (0..m).map(|i| vectors[(i, j)] * v.sqrt()).collect();
            // Fix the arbitrary phase of the eigenvector (largest entry
            // real and positive) and drop rounding noise, so that diagonal
            // operators stay exactly diagonal.
            let big = k
                .iter()
                .copied()
                .fold(ZERO, |a, z| if z.norm() > a.norm() { z } else { a });
            let phase = if big.norm() > 0.0 {
                big.conj() / big.norm()
            } else {
                c(1.0)
            };
            k.into_iter()
                .map(|z| {
                    let z = z * phase;
                    C64::new(
                        if z.re.abs() < 1e-15 { 0.0 } else { z.re },
                        if z.im.abs() < 1e-15 { 0.0 } else { z.im },
                    )
                })
                .collect()
        })
        .collect()
}

/// Index of a Pauli label (`"X"`, `"ZZ"`, …) of length `k`: digits in base
/// 4 (`I, X, Y, Z`), first letter most significant.
pub(crate) fn pauli_index(label: &str, k: usize) -> Result<usize> {
    if label.len() != k {
        return Err(QsoError::invalid(
            "label",
            format!("Pauli labels must all have {k} letters, got '{label}'"),
        ));
    }
    label.chars().try_fold(0usize, |acc, ch| {
        let digit = match ch.to_ascii_uppercase() {
            'I' => 0,
            'X' => 1,
            'Y' => 2,
            'Z' => 3,
            _ => {
                return Err(QsoError::invalid(
                    "label",
                    format!("'{label}' is not a Pauli string"),
                ));
            }
        };
        Ok(acc * 4 + digit)
    })
}

/// Row-major matrix of the `k`-qubit Pauli string with base-4 index
/// `index` (first qubit = most significant local bit).
pub(crate) fn pauli_matrix(k: usize, index: usize) -> Vec<C64> {
    let single = |p: usize| -> [C64; 4] {
        let i = C64::new(0.0, 1.0);
        match p {
            0 => [c(1.0), ZERO, ZERO, c(1.0)],
            1 => [ZERO, c(1.0), c(1.0), ZERO],
            2 => [ZERO, -i, i, ZERO],
            _ => [c(1.0), ZERO, ZERO, c(-1.0)],
        }
    };
    let mut m = vec![c(1.0)];
    let mut d = 1;
    for j in (0..k).rev() {
        // Digits from the most significant: qubit j of the string.
        let p = single((index >> (2 * j)) & 3);
        let mut next = vec![ZERO; 4 * d * d];
        let nd = 2 * d;
        for r in 0..d {
            for cc in 0..d {
                for a in 0..2 {
                    for b in 0..2 {
                        next[(r * 2 + a) * nd + cc * 2 + b] = m[r * d + cc] * p[a * 2 + b];
                    }
                }
            }
        }
        m = next;
        d = nd;
    }
    m
}

/// Noise applied automatically after every gate of a circuit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NoiseModel {
    /// Applied to the qubit of every single-qubit gate.
    pub single_qubit: Vec<Channel>,
    /// Applied to every qubit (1-qubit channels) or the pair (2-qubit
    /// channels) of every multi-qubit gate.
    pub multi_qubit: Vec<Channel>,
    /// Classical readout flips `(P(read 1 | 0), P(read 0 | 1))`.
    pub readout: Option<(f64, f64)>,
}

impl NoiseModel {
    /// Uniform depolarising noise: `p1` after one-qubit gates, `p2`
    /// (two-qubit depolarising) after two-qubit gates.
    pub fn depolarizing(p1: f64, p2: f64) -> Result<Self> {
        Ok(Self {
            single_qubit: vec![Channel::depolarizing(p1)?],
            multi_qubit: vec![Channel::depolarizing_2q(p2)?],
            readout: None,
        })
    }

    /// Add classical readout flips `(P(read 1 | 0), P(read 0 | 1))`.
    pub fn with_readout(mut self, p01: f64, p10: f64) -> Result<Self> {
        check_probability("p01", p01)?;
        check_probability("p10", p10)?;
        self.readout = Some((p01, p10));
        Ok(self)
    }

    /// Check the readout probabilities (the channels are validated when
    /// they are constructed).
    pub fn validate(&self) -> Result<()> {
        if let Some(ch) = self.single_qubit.iter().find(|c| c.n_qubits != 1) {
            return Err(QsoError::invalid(
                "single_qubit",
                format!(
                    "channel {} acts on {} qubits; single-qubit gates need 1-qubit channels",
                    ch.name, ch.n_qubits
                ),
            ));
        }
        if let Some((p01, p10)) = self.readout {
            check_probability("p01", p01)?;
            check_probability("p10", p10)?;
        }
        Ok(())
    }
}

/// Density matrix of `n` qubits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "DensityMatrixData")]
pub struct DensityMatrix {
    n_qubits: usize,
    data: Vec<C64>,
}

/// Unvalidated serialised form of a [`DensityMatrix`].
#[derive(Deserialize)]
struct DensityMatrixData {
    n_qubits: usize,
    data: Vec<C64>,
}

impl TryFrom<DensityMatrixData> for DensityMatrix {
    type Error = QsoError;

    fn try_from(raw: DensityMatrixData) -> Result<Self> {
        DensityMatrix::check_size(raw.n_qubits)?;
        let expected = 1usize << (2 * raw.n_qubits);
        if raw.data.len() != expected {
            return Err(QsoError::DimensionMismatch {
                expected,
                actual: raw.data.len(),
            });
        }
        if raw
            .data
            .iter()
            .any(|a| !a.re.is_finite() || !a.im.is_finite())
        {
            return Err(QsoError::invalid("data", "entries must be finite"));
        }
        Ok(Self {
            n_qubits: raw.n_qubits,
            data: raw.data,
        })
    }
}

impl DensityMatrix {
    fn check_size(n_qubits: usize) -> Result<()> {
        if n_qubits > MAX_DENSITY_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "DensityMatrix",
                max: MAX_DENSITY_QUBITS,
                requested: n_qubits,
            });
        }
        Ok(())
    }

    /// `|0…0⟩⟨0…0|`.
    pub fn zero(n_qubits: usize) -> Result<Self> {
        Self::check_size(n_qubits)?;
        let mut data = vec![ZERO; 1 << (2 * n_qubits)];
        data[0] = c(1.0);
        Ok(Self { n_qubits, data })
    }

    /// Maximally mixed state `I/2ⁿ`.
    pub fn maximally_mixed(n_qubits: usize) -> Result<Self> {
        let mut rho = Self::zero(n_qubits)?;
        rho.data[0] = ZERO;
        let dim = 1usize << n_qubits;
        for b in 0..dim {
            rho.data[b + (b << n_qubits)] = c(1.0 / dim as f64);
        }
        Ok(rho)
    }

    /// `|ψ⟩⟨ψ|`.
    pub fn from_pure(state: &StateVector) -> Result<Self> {
        let n = state.n_qubits();
        Self::check_size(n)?;
        let psi = state.amplitudes();
        let dim = psi.len();
        let data = crate::parallel::map_collect(dim * dim, |i| {
            let r = i & (dim - 1);
            let cc = i >> n;
            psi[r] * psi[cc].conj()
        });
        Ok(Self { n_qubits: n, data })
    }

    /// Noiseless evolution of `|0…0⟩` under `circuit`.
    pub fn from_circuit(circuit: &Circuit, theta: &[f64]) -> Result<Self> {
        let mut rho = Self::zero(circuit.n_qubits())?;
        rho.apply_circuit(circuit, theta, None)?;
        Ok(rho)
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Entry `ρ[r, c]`.
    ///
    /// # Panics
    /// If `r` or `c` is not below `2ⁿ`.
    pub fn get(&self, r: usize, c: usize) -> C64 {
        let dim = 1usize << self.n_qubits;
        assert!(
            r < dim && c < dim,
            "index ({r}, {c}) outside a {dim}x{dim} density matrix"
        );
        self.data[r + (c << self.n_qubits)]
    }

    /// Dense copy of ρ.
    pub fn to_matrix(&self) -> CMatrix {
        let dim = 1usize << self.n_qubits;
        CMatrix::from_fn(dim, dim, |r, cc| self.get(r, cc))
    }

    /// Apply `circuit` (with parameters `theta`), inserting `noise` after
    /// every gate if given.
    pub fn apply_circuit(
        &mut self,
        circuit: &Circuit,
        theta: &[f64],
        noise: Option<&NoiseModel>,
    ) -> Result<()> {
        if circuit.n_qubits() > self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: circuit.n_qubits(),
            });
        }
        circuit.check_parameters(theta)?;
        if let Some(model) = noise {
            model.validate()?;
        }
        let n = self.n_qubits;
        for inst in circuit.instructions() {
            let angles = inst.angles(theta);
            kernels::apply_instruction(&mut self.data, inst, &angles, false, 0);
            kernels::apply_instruction(&mut self.data, inst, &angles, true, n);
            if let Some(model) = noise {
                let qubits: Vec<usize> = inst.qubits().collect();
                let channels = if qubits.len() == 1 {
                    &model.single_qubit
                } else {
                    &model.multi_qubit
                };
                for ch in channels {
                    match ch.n_qubits {
                        1 => {
                            for &q in &qubits {
                                self.apply_channel(ch, &[q])?;
                            }
                        }
                        _ => {
                            // Two-qubit channels act on consecutive pairs of
                            // the gate's qubits.
                            for pair in qubits.windows(2) {
                                self.apply_channel(ch, pair)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Multiply every entry by `factor` (for averaging states).
    pub fn scale(&mut self, factor: f64) {
        for z in &mut self.data {
            *z *= factor;
        }
    }

    /// Add `factor · other` (for averaging states over runs).
    ///
    /// # Panics
    /// If the registers differ in size.
    pub fn add_scaled(&mut self, other: &DensityMatrix, factor: f64) {
        assert_eq!(self.n_qubits, other.n_qubits, "registers differ in size");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b * factor;
        }
    }

    /// Density matrix from vectorised data (checked sizes, trusted values).
    pub(crate) fn from_data(n_qubits: usize, data: Vec<C64>) -> Result<Self> {
        Self::check_size(n_qubits)?;
        debug_assert_eq!(data.len(), 1 << (2 * n_qubits));
        Ok(Self { n_qubits, data })
    }

    /// Apply `circuit(theta)` with the noise of `device` (gate errors,
    /// relaxation during gates and idle times; see [`DeviceModel`]). Readout
    /// errors apply when sampling with [`DensityMatrix::sample_with_readout`].
    pub fn apply_device(
        &mut self,
        circuit: &Circuit,
        theta: &[f64],
        device: &DeviceModel,
    ) -> Result<()> {
        if circuit.n_qubits() != self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: circuit.n_qubits(),
            });
        }
        device.validate()?;
        let program = device::Program::compile(circuit, theta, device)?;
        self.run_program(&program);
        Ok(())
    }

    /// Apply a compiled program (sizes already checked).
    pub(crate) fn run_program(&mut self, program: &device::Program) {
        self.run_ops(&program.ops, &program.channels);
    }

    /// Apply compiled operations (sizes already checked). Consecutive
    /// channels on the same qubits are combined into one superoperator.
    pub(crate) fn run_ops(&mut self, ops: &[device::Op], channels: &[device::CompiledChannel]) {
        let n = self.n_qubits;
        let mut pending: Option<(Vec<C64>, &[usize])> = None;
        for op in ops {
            match op {
                device::Op::Gates(c) => {
                    if let Some((s, q)) = pending.take() {
                        self.apply_superoperator(&s, q);
                    }
                    for inst in c.instructions() {
                        let angles = inst.angles(&[]);
                        kernels::apply_instruction(&mut self.data, inst, &angles, false, 0);
                        kernels::apply_instruction(&mut self.data, inst, &angles, true, n);
                    }
                }
                device::Op::Channel { channel, qubits } => {
                    let s = &channels[*channel].superoperator;
                    pending = match pending.take() {
                        Some((prev, q)) if q == qubits.as_slice() => {
                            Some((mat_product(s, &prev), q))
                        }
                        Some((prev, q)) => {
                            self.apply_superoperator(&prev, q);
                            Some((s.clone(), qubits.as_slice()))
                        }
                        None => Some((s.clone(), qubits.as_slice())),
                    };
                }
            }
        }
        if let Some((s, q)) = pending {
            self.apply_superoperator(&s, q);
        }
    }

    /// Apply a 1- or 2-qubit superoperator (qubits already checked).
    fn apply_superoperator(&mut self, s: &[C64], qubits: &[usize]) {
        let n = self.n_qubits;
        match qubits {
            [q] => {
                let m: [C64; 16] = s.try_into().expect("4x4 superoperator");
                kernels::apply_2q(&mut self.data, *q, *q + n, 0, &m);
            }
            _ => {
                let targets = [qubits[0], qubits[1], qubits[0] + n, qubits[1] + n];
                kernels::apply_kq(&mut self.data, &targets, 0, s);
            }
        }
    }

    /// Apply a channel to `qubits` (one or two, matching the channel).
    pub fn apply_channel(&mut self, channel: &Channel, qubits: &[usize]) -> Result<()> {
        if qubits.len() != channel.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: channel.n_qubits,
                actual: qubits.len(),
            });
        }
        for &q in qubits {
            if q >= self.n_qubits {
                return Err(QsoError::QubitOutOfRange {
                    qubit: q,
                    n_qubits: self.n_qubits,
                });
            }
        }
        if qubits.len() == 2 && qubits[0] == qubits[1] {
            return Err(QsoError::DuplicateQubit {
                gate: "channel",
                qubit: qubits[0],
            });
        }
        self.apply_superoperator(&channel.superoperator(), qubits);
        Ok(())
    }

    /// `Tr ρ` (1 for a valid state).
    pub fn trace(&self) -> f64 {
        let dim = 1usize << self.n_qubits;
        (0..dim)
            .map(|b| self.data[b + (b << self.n_qubits)].re)
            .sum()
    }

    /// `Tr ρ²` (1 for pure states, 2⁻ⁿ for the maximally mixed state).
    pub fn purity(&self) -> f64 {
        sum_f64(self.data.len(), |i| self.data[i].norm_sqr())
    }

    /// Diagonal of ρ.
    pub fn probabilities(&self) -> Vec<f64> {
        let dim = 1usize << self.n_qubits;
        (0..dim)
            .map(|b| self.data[b + (b << self.n_qubits)].re.max(0.0))
            .collect()
    }

    /// `Tr(ρH)`.
    pub fn expectation(&self, observable: &PauliSum) -> Result<f64> {
        let needed = observable.min_qubits();
        if needed > self.n_qubits {
            return Err(QsoError::QubitOutOfRange {
                qubit: needed - 1,
                n_qubits: self.n_qubits,
            });
        }
        let n = self.n_qubits;
        let dim = 1usize << n;
        let mut total = 0.0;
        for &(coef, s) in observable.terms() {
            // Tr(Pρ) = Σ_c ⟨c⊕x|P|c⟩ ρ[c, c⊕x]
            let value = crate::parallel::sum_c64(dim, |col| {
                let (phase, row) = s.act(col);
                phase * self.data[col + (row << n)]
            });
            total += coef * value.re;
        }
        Ok(total)
    }

    /// `⟨ψ|ρ|ψ⟩`.
    pub fn fidelity_with_pure(&self, state: &StateVector) -> Result<f64> {
        if state.n_qubits() != self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: state.n_qubits(),
            });
        }
        let psi = state.amplitudes();
        let n = self.n_qubits;
        let dim = psi.len();
        let v = crate::parallel::sum_c64(dim * dim, |i| {
            let r = i & (dim - 1);
            let cc = i >> n;
            psi[r].conj() * self.data[i] * psi[cc]
        });
        Ok(v.re)
    }

    /// Sample `shots` measurements of all qubits with per-qubit readout
    /// errors `(P(read 1 | 0), P(read 0 | 1))` (qubits beyond the list read
    /// out perfectly).
    pub fn sample_with_readout(
        &self,
        shots: usize,
        readout: &[(f64, f64)],
        rng: &mut Rng,
    ) -> Counts {
        let probs = self.probabilities();
        let mut cumulative = Vec::with_capacity(probs.len());
        let mut acc = 0.0;
        for p in probs {
            acc += p;
            cumulative.push(acc);
        }
        let mut counts = Counts::new();
        for _ in 0..shots {
            let b = sample_cumulative(&cumulative, rng);
            let b = trajectories::apply_readout(b, readout, rng);
            *counts.entry(b).or_insert(0) += 1;
        }
        counts
    }

    /// Sample `shots` measurements of all qubits, including readout error.
    /// Readout probabilities are clamped to `[0, 1]`.
    pub fn sample(&self, shots: usize, readout: Option<(f64, f64)>, rng: &mut Rng) -> Counts {
        let probs = self.probabilities();
        let mut cumulative = Vec::with_capacity(probs.len());
        let mut acc = 0.0;
        for p in probs {
            acc += p;
            cumulative.push(acc);
        }
        let mut counts = Counts::new();
        for _ in 0..shots {
            let mut b = sample_cumulative(&cumulative, rng);
            if let Some((p01, p10)) = readout {
                for q in 0..self.n_qubits {
                    let flip = if (b >> q) & 1 == 0 { p01 } else { p10 }.clamp(0.0, 1.0);
                    if rng.uniform() < flip {
                        b ^= 1 << q;
                    }
                }
            }
            *counts.entry(b).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::Pauli;
    use crate::quantum::PauliString;

    #[test]
    fn unitary_evolution_matches_pure_state() {
        let mut c = Circuit::new(3);
        let t = c.new_parameter();
        c.h(0)
            .cx(0, 1)
            .ry(2, t)
            .crz(1, 2, 0.8)
            .swap(0, 2)
            .u(1, 0.2, 0.3, 0.4)
            .ccx(0, 1, 2);
        let theta = [0.66];
        let psi = StateVector::from_circuit(&c, &theta).unwrap();
        let rho = DensityMatrix::from_circuit(&c, &theta).unwrap();
        let expected = DensityMatrix::from_pure(&psi).unwrap();
        for (a, b) in rho.data.iter().zip(&expected.data) {
            assert!((a - b).norm() < 1e-12);
        }
        assert!((rho.purity() - 1.0).abs() < 1e-12);
        assert!((rho.fidelity_with_pure(&psi).unwrap() - 1.0).abs() < 1e-12);
        let h: PauliSum = "0.4 X0 Z1 - 0.3 Y2 + 1.1 Z0 Z2".parse().unwrap();
        assert!((rho.expectation(&h).unwrap() - h.expectation(&psi).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn depolarizing_shrinks_bloch_vector() {
        let p = 0.3;
        let mut rho = DensityMatrix::zero(1).unwrap();
        rho.apply_channel(&Channel::depolarizing(p).unwrap(), &[0])
            .unwrap();
        let z: PauliSum = PauliString::single(0, Pauli::Z).into();
        assert!((rho.expectation(&z).unwrap() - (1.0 - 4.0 * p / 3.0)).abs() < 1e-12);
        assert!((rho.trace() - 1.0).abs() < 1e-12);
        // p = 3/4 is the fully depolarising point.
        let mut rho = DensityMatrix::zero(1).unwrap();
        rho.apply_channel(&Channel::depolarizing(0.75).unwrap(), &[0])
            .unwrap();
        assert!((rho.purity() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn two_qubit_depolarizing_full_strength_is_maximally_mixed() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1);
        let mut rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
        rho.apply_channel(&Channel::depolarizing_2q(15.0 / 16.0).unwrap(), &[0, 1])
            .unwrap();
        assert!((rho.purity() - 0.25).abs() < 1e-12);
        assert!((rho.trace() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn thermal_relaxation_matches_t1_t2() {
        let (t1, t2, t) = (50.0, 30.0, 12.0);
        // |+⟩ for coherence, |1⟩ for population.
        let mut c = Circuit::new(1);
        c.h(0);
        let mut rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
        rho.apply_channel(&Channel::thermal_relaxation(t1, t2, t).unwrap(), &[0])
            .unwrap();
        assert!((rho.get(0, 1).norm() - 0.5 * (-t / t2).exp()).abs() < 1e-12);
        let mut c = Circuit::new(1);
        c.x(0);
        let mut rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
        rho.apply_channel(&Channel::thermal_relaxation(t1, t2, t).unwrap(), &[0])
            .unwrap();
        assert!((rho.probabilities()[1] - (-t / t1).exp()).abs() < 1e-12);
        assert!(Channel::thermal_relaxation(10.0, 25.0, 1.0).is_err());
    }

    #[test]
    fn noisy_bell_state_fidelity_and_readout() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1);
        let ideal = StateVector::from_circuit(&c, &[]).unwrap();
        let model = NoiseModel::depolarizing(0.01, 0.05).unwrap();
        let mut rho = DensityMatrix::zero(2).unwrap();
        rho.apply_circuit(&c, &[], Some(&model)).unwrap();
        let f = rho.fidelity_with_pure(&ideal).unwrap();
        assert!(f < 1.0 && f > 0.9, "fidelity {f}");
        assert!((rho.trace() - 1.0).abs() < 1e-12);
        // Hermiticity
        let m = rho.to_matrix();
        assert!((&m - m.adjoint()).norm() < 1e-12);
        let mut rng = Rng::seed_from_u64(1);
        let counts = rho.sample(20_000, Some((0.02, 0.05)), &mut rng);
        let odd: usize = counts
            .iter()
            .filter(|(b, _)| b.count_ones() == 1)
            .map(|(_, n)| n)
            .sum();
        assert!(odd > 0);
    }

    #[test]
    fn invalid_channels_are_rejected() {
        assert!(Channel::depolarizing(1.5).is_err());
        assert!(Channel::from_kraus("bad", vec![vec![c(2.0), ZERO, ZERO, c(1.0)]]).is_err());
        let mut rho = DensityMatrix::zero(1).unwrap();
        assert!(
            rho.apply_channel(&Channel::bit_flip(0.1).unwrap(), &[1])
                .is_err()
        );
        assert!(DensityMatrix::zero(MAX_DENSITY_QUBITS + 1).is_err());
    }
}
