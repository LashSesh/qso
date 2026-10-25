//! Classical shadows with random Pauli measurements (Huang, Kueng and
//! Preskill 2020).
//!
//! Each snapshot measures every qubit in a uniformly random basis X, Y or
//! Z. For a Pauli string `P` on `k` qubits, a snapshot whose bases match
//! `P` on its support contributes `3ᵏ·(−1)^{parity}` and every other
//! snapshot contributes 0; the mean is an unbiased estimate of `⟨P⟩` with
//! single-snapshot variance `3ᵏ − ⟨P⟩²`. One set of snapshots therefore
//! estimates many local observables at once.
//!
//! ```
//! use metatron_qso::estimation::ClassicalShadow;
//! use metatron_qso::quantum::{Circuit, PauliSum, StateVector};
//!
//! let mut c = Circuit::new(3);
//! c.h(0).cx(0, 1).cx(1, 2);
//! let ghz = StateVector::from_circuit(&c, &[]).unwrap();
//! let shadow = ClassicalShadow::from_state(&ghz, 20_000, 1).unwrap();
//! let zz: PauliSum = "Z0 Z1".parse().unwrap();
//! let e = shadow.estimate(&zz).unwrap();
//! assert!((e.value - 1.0).abs() < 5.0 * e.std_error);
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::Estimate;
use crate::backend::Backend;
use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, Counts, PauliSum, StateVector};
use crate::rng::Rng;

/// Largest register a shadow records (Pauli strings are 64-bit masks).
pub const MAX_SHADOW_QUBITS: usize = 64;

/// One snapshot: measurement bases (X where `x` is set, Y where `y` is set,
/// Z elsewhere) and the measured bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Snapshot {
    x: u64,
    y: u64,
    bits: u64,
}

/// A classical shadow: random-basis snapshots of a state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassicalShadow {
    n_qubits: usize,
    snapshots: Vec<Snapshot>,
}

/// Random bases for `count` snapshots, grouped by basis with their
/// multiplicity.
fn random_bases(n: usize, count: usize, rng: &mut Rng) -> BTreeMap<(u64, u64), usize> {
    let mut bases = BTreeMap::new();
    for _ in 0..count {
        let (mut x, mut y) = (0u64, 0u64);
        for q in 0..n {
            match rng.below(3) {
                0 => x |= 1 << q,
                1 => y |= 1 << q,
                _ => {}
            }
        }
        *bases.entry((x, y)).or_insert(0) += 1;
    }
    bases
}

/// The circuit rotating the bases `(x, y)` onto Z.
fn rotation(n: usize, x: u64, y: u64) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        if (x >> q) & 1 == 1 {
            c.h(q);
        } else if (y >> q) & 1 == 1 {
            c.sdg(q).h(q);
        }
    }
    c
}

impl ClassicalShadow {
    fn build(
        n: usize,
        snapshots: usize,
        seed: u64,
        mut measure: impl FnMut(u64, u64, usize, u64) -> Result<Counts>,
    ) -> Result<Self> {
        if n > MAX_SHADOW_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "classical shadow",
                max: MAX_SHADOW_QUBITS,
                requested: n,
            });
        }
        if snapshots == 0 {
            return Err(QsoError::invalid("snapshots", "need at least one snapshot"));
        }
        let base = Rng::seed_from_u64(seed);
        let mut rng = base.fork(0);
        let bases = random_bases(n, snapshots, &mut rng);
        let mut out = Vec::with_capacity(snapshots);
        for (k, (&(x, y), &count)) in bases.iter().enumerate() {
            let counts = measure(x, y, count, base.fork(k as u64 + 1).next_u64())?;
            for (b, m) in counts {
                out.extend(std::iter::repeat_n(
                    Snapshot {
                        x,
                        y,
                        bits: b as u64,
                    },
                    m,
                ));
            }
        }
        // Snapshots arrive grouped by basis; shuffle them so that blocks of
        // consecutive snapshots (median of means) are random samples.
        rng.shuffle(&mut out);
        Ok(Self {
            n_qubits: n,
            snapshots: out,
        })
    }

    /// Snapshots of `circuit(theta)` measured on `backend` (one run per
    /// distinct random basis).
    pub fn collect(
        backend: &dyn Backend,
        circuit: &Circuit,
        theta: &[f64],
        snapshots: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = circuit.n_qubits();
        Self::build(n, snapshots, seed, |x, y, count, s| {
            let mut c = circuit.clone();
            c.append(&rotation(n, x, y))?;
            backend.run(&c, theta, count, s)
        })
    }

    /// Snapshots of a state vector (rotating a copy per distinct basis).
    pub fn from_state(state: &StateVector, snapshots: usize, seed: u64) -> Result<Self> {
        let n = state.n_qubits();
        Self::build(n, snapshots, seed, |x, y, count, s| {
            let mut copy = state.clone();
            copy.apply_circuit(&rotation(n, x, y), &[])?;
            Ok(copy.sample(count, &mut Rng::seed_from_u64(s)))
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Per-snapshot estimates of `observable`.
    fn values(&self, observable: &PauliSum) -> Result<Vec<f64>> {
        if observable.min_qubits() > self.n_qubits {
            return Err(QsoError::QubitOutOfRange {
                qubit: observable.min_qubits() - 1,
                n_qubits: self.n_qubits,
            });
        }
        // (coefficient, support, X-basis mask, Y-basis mask, 3^weight)
        let terms: Vec<(f64, u64, u64, u64, f64)> = observable
            .terms()
            .iter()
            .map(|&(c, s)| {
                let (px, pz) = (s.x_mask(), s.z_mask());
                let support = px | pz;
                (
                    c,
                    support,
                    px & !pz,
                    px & pz,
                    3f64.powi(support.count_ones() as i32),
                )
            })
            .collect();
        Ok(self
            .snapshots
            .iter()
            .map(|snap| {
                terms
                    .iter()
                    .map(|&(c, support, xs, ys, scale)| {
                        if snap.x & support == xs && snap.y & support == ys {
                            let sign = if (snap.bits & support).count_ones().is_multiple_of(2) {
                                1.0
                            } else {
                                -1.0
                            };
                            c * scale * sign
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .collect())
    }

    /// Mean over the snapshots with its standard error.
    pub fn estimate(&self, observable: &PauliSum) -> Result<Estimate> {
        Ok(Estimate::from_samples(&self.values(observable)?))
    }

    /// Median of the means of `groups` equal blocks of snapshots, which
    /// is robust against the heavy tails of the single-snapshot estimates.
    pub fn median_of_means(&self, observable: &PauliSum, groups: usize) -> Result<f64> {
        if groups == 0 || groups > self.snapshots.len() {
            return Err(QsoError::invalid(
                "groups",
                "need between one group and one group per snapshot",
            ));
        }
        let values = self.values(observable)?;
        // Blocks of ⌊n/g⌋ or ⌈n/g⌉ snapshots, using all of them.
        let (n, mut start) = (values.len(), 0);
        let mut means: Vec<f64> = (0..groups)
            .map(|k| {
                let end = start + n / groups + usize::from(k < n % groups);
                let block = &values[start..end];
                start = end;
                block.iter().sum::<f64>() / block.len() as f64
            })
            .collect();
        means.sort_by(f64::total_cmp);
        let mid = means.len() / 2;
        Ok(if means.len() % 2 == 1 {
            means[mid]
        } else {
            0.5 * (means[mid - 1] + means[mid])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceBackend;
    use crate::noise::{DensityMatrix, DeviceModel, GateRule};

    fn state() -> StateVector {
        let mut c = Circuit::new(4);
        c.ry(0, 0.7)
            .h(1)
            .cx(1, 2)
            .rx(3, 1.1)
            .cx(0, 3)
            .rz(2, 0.4)
            .ry(2, -0.6);
        StateVector::from_circuit(&c, &[]).unwrap()
    }

    #[test]
    fn single_snapshot_variance_is_three_to_the_weight() {
        let psi = state();
        let shots = 60_000;
        let shadow = ClassicalShadow::from_state(&psi, shots, 3).unwrap();
        assert_eq!(shadow.len(), shots);
        for label in ["Z0", "X1 X2", "Y0 Z2 X3", "Z0 Z1 Z2 Z3"] {
            let h: PauliSum = label.parse().unwrap();
            let exact = h.expectation(&psi).unwrap();
            let values = shadow.values(&h).unwrap();
            let e = Estimate::from_samples(&values);
            assert!((e.value - exact).abs() < 4.0 * e.std_error, "{label}");
            let k = h.terms()[0].1.weight() as i32;
            let var = e.std_error.powi(2) * shots as f64;
            let bound = 3f64.powi(k) - exact * exact;
            // Relative spread of a sample variance ≈ √((3ᵏ − 1)/N).
            let tol = 4.0 * ((3f64.powi(k) - 1.0) / shots as f64).sqrt();
            assert!((var / bound - 1.0).abs() < tol, "{label}: {var} vs {bound}");
        }
    }

    #[test]
    fn sums_and_median_of_means() {
        let psi = state();
        let shadow = ClassicalShadow::from_state(&psi, 30_000, 8).unwrap();
        let h: PauliSum = "0.5 Z0 Z3 - 1.2 X1 X2 + 0.3 Y3 + 2.0".parse().unwrap();
        let exact = h.expectation(&psi).unwrap();
        let e = shadow.estimate(&h).unwrap();
        assert!((e.value - exact).abs() < 4.0 * e.std_error);
        let mom = shadow.median_of_means(&h, 10).unwrap();
        // Uneven blocks use every snapshot: with one group per snapshot but
        // one, the median of means differs from that over the first blocks.
        let small = ClassicalShadow::from_state(&psi, 10, 1).unwrap();
        assert!(small.median_of_means(&h, 6).unwrap().is_finite());
        assert!((mom - exact).abs() < 0.15, "{mom} vs {exact}");
        assert!(shadow.median_of_means(&h, 0).is_err());
    }

    #[test]
    fn shadows_of_noisy_devices() {
        let mut c = Circuit::new(3);
        c.h(0).cx(0, 1).cx(1, 2);
        // Noise on the entangling gates only: the basis rotations a shadow
        // appends are single-qubit gates and stay ideal here.
        let device = DeviceModel::new("dep")
            .with_rule(GateRule::any().with_num_qubits(2).with_depolarizing(0.05));
        let mut rho = DensityMatrix::zero(3).unwrap();
        rho.apply_device(&c, &[], &device).unwrap();
        let backend = DeviceBackend::density_matrix(device).unwrap();
        let shadow = ClassicalShadow::collect(&backend, &c, &[], 30_000, 2).unwrap();
        for label in ["Z0 Z1", "X0 X1 X2", "Z2"] {
            let h: PauliSum = label.parse().unwrap();
            let exact = rho.expectation(&h).unwrap();
            let e = shadow.estimate(&h).unwrap();
            assert!(
                (e.value - exact).abs() < 4.0 * e.std_error,
                "{label}: {} vs {exact}",
                e.value
            );
        }
    }
}
