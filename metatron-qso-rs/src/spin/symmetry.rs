//! Permutations of equivalent spins (the protons of a methyl or a
//! methylene group) that leave a spin system unchanged. Declared with
//! [`SpinSystem::set_symmetries`], they let relaxation propagate an
//! invariant state on the orbits of the elements `|a⟩⟨b|` instead of the
//! elements themselves (see the `krylov` source).

use super::SpinSystem;
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;

/// Shifts, couplings and relaxation times of equivalent spins agree to
/// this fraction.
const EQUAL: f64 = 1e-12;

fn close(x: f64, y: f64) -> bool {
    (x - y).abs() <= EQUAL * x.abs().max(y.abs())
}

impl SpinSystem {
    /// Declare permutations of the spins (`p[i]` the image of spin `i`)
    /// that leave the system unchanged, replacing earlier ones: each maps
    /// spins to spins of the same isotope with the same shift, relaxation
    /// times and couplings, and keeps the relaxing interactions of
    /// Redfield theory (dipolar couplings and shielding anisotropies) up to
    /// a rotation of the molecule. A methyl group's protons `h₁, h₂, h₃`
    /// with a threefold axis give the cycle `h₁ → h₂ → h₃ → h₁`. Relaxation
    /// then propagates states that the permutations leave unchanged on the
    /// orbits of their elements, which is exact and several times smaller;
    /// other states propagate in full. On an error the system keeps its
    /// symmetries.
    pub fn set_symmetries(&mut self, permutations: Vec<Vec<usize>>) -> Result<&mut Self> {
        let old = std::mem::replace(&mut self.symmetries, permutations);
        if let Err(e) = self.validate_symmetries() {
            self.symmetries = old;
            return Err(e);
        }
        Ok(self)
    }

    /// The declared permutations of the spins.
    pub fn symmetries(&self) -> &[Vec<usize>] {
        &self.symmetries
    }

    /// Check the declared permutations against the system.
    pub(super) fn validate_symmetries(&self) -> Result<()> {
        let n = self.spins.len();
        let refuse = |why: String| Err(QsoError::invalid("symmetries", why));
        for p in &self.symmetries {
            let mut seen = vec![false; n];
            if p.len() != n
                || p.iter()
                    .any(|&i| i >= n || std::mem::replace(&mut seen[i], true))
            {
                return refuse(format!("{p:?} is not a permutation of the {n} spins"));
            }
            for (i, &j) in p.iter().enumerate() {
                let (a, b) = (&self.spins[i], &self.spins[j]);
                let same_time = |x: Option<f64>, y: Option<f64>| match (x, y) {
                    (Some(x), Some(y)) => close(x, y),
                    (None, None) => true,
                    _ => false,
                };
                if a.isotope != b.isotope
                    || !close(a.shift, b.shift)
                    || !same_time(a.t1, b.t1)
                    || !same_time(a.t2, b.t2)
                {
                    return refuse(format!(
                        "{p:?} maps spin {i} onto spin {j}, which differs in isotope, shift or \
                         relaxation times"
                    ));
                }
            }
            let coupling = |i: usize, j: usize| {
                let key = (i.min(j), i.max(j));
                self.couplings
                    .iter()
                    .find(|&&(x, y, _)| (x, y) == key)
                    .map_or(0.0, |c| c.2)
            };
            for &(i, j, v) in &self.couplings {
                if !close(coupling(p[i], p[j]), v) {
                    return refuse(format!(
                        "{p:?} maps the coupling of spins {i} and {j} onto a different one"
                    ));
                }
            }
            if !self.redfield_invariant(p) {
                return refuse(format!(
                    "{p:?} changes the relaxing dipolar couplings or shielding anisotropies \
                     (positions and tensors must agree up to a rotation)"
                ));
            }
        }
        Ok(())
    }

    /// The permutation of the basis states a spin permutation induces:
    /// state `s` goes to the one with spin `p[i]` at the level of spin `i`.
    fn state_map(&self, p: &[usize]) -> Vec<usize> {
        (0..self.dimension())
            .map(|s| {
                p.iter()
                    .enumerate()
                    .map(|(i, &j)| {
                        let level = (s / self.stride(i)) % self.spins[i].isotope.levels();
                        level * self.stride(j)
                    })
                    .sum()
            })
            .collect()
    }

    /// The permutations of the basis states the declared symmetries
    /// induce, when `h` is unchanged by them (to 10⁻¹² of its largest
    /// entry); `None` without symmetries or when `h` breaks them (a pulse
    /// on one spin).
    pub(super) fn symmetry_maps(&self, h: &CMatrix) -> Option<Vec<Vec<usize>>> {
        if self.symmetries.is_empty() {
            return None;
        }
        let maps: Vec<Vec<usize>> = self.symmetries.iter().map(|p| self.state_map(p)).collect();
        let d = h.nrows();
        let tol = 1e-12 * h.iter().fold(0.0f64, |m, z| m.max(z.norm()));
        let invariant = maps
            .iter()
            .all(|m| (0..d).all(|c| (0..d).all(|r| (h[(m[r], m[c])] - h[(r, c)]).norm() <= tol)));
        invariant.then_some(maps)
    }
}
