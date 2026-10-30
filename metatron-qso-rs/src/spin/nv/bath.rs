//! The nuclear-spin bath of an NV centre: ¹³C on the diamond lattice and
//! the coherence of the electron by the cluster-correlation expansion.
//!
//! With ideal pulses on the qubit `{0, m}` the electron is in level `m` or
//! `0` at every moment of a sequence, so the bath evolves under
//!
//! ```text
//! H_a/2π = Σ_i [a·(A_zx I_x + A_zy I_y + A_zz I_z)_i − γ_C B I_z^i]
//!        + Σ_{i<j} d_ij [I_i·I_j − 3 (I_i·n_ij)(I_j·n_ij)],   a ∈ {m, 0},
//! ```
//!
//! the secular hyperfine couplings `A_z·` of every ¹³C and the dipolar
//! couplings `d_ij = (μ₀/4π) γ_C² h / r_ij³` between them (Hz). A sequence
//! — a pulse, free periods and π pulses, a pulse — splits the electron's
//! coherence into two branches, one starting in `m` and one in `0`, which
//! swap levels at every π pulse. The `m_s = 0` population at the end is
//!
//! ```text
//! S = |w_m|² + |w_0|² + 2 Re(w_m w̄_0 L),   L = Tr(U_m ρ U_0†),
//! ```
//!
//! with `w` from the pulses alone and `U_m`, `U_0` the bath's evolution
//! along each branch from the unpolarised state `ρ = 1/2^N`; for the
//! standard sequences `S = (1 + Re L)/2`. The coherence `L` is a product
//! over the clusters of the bath (Yang and Liu, Phys. Rev. B 78, 085315
//! (2008)): `L = Π_C L̃_C`, with `L_C` the coherence of cluster `C` alone
//! and `L̃_C = L_C / Π_{C' ⊊ C} L̃_{C'}`. A cluster is a set of spins
//! connected by pairs closer than a cutoff; only the couplings of those
//! pairs are kept, so clusters that are not connected factorise exactly.
//! Truncated at clusters of `order` spins, the expansion is exact once
//! `order` reaches the size of the largest connected cluster.

use std::collections::{HashMap, HashSet};

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{
    FieldNoise, NV_GYROMAGNETIC_RATIO, NoisePath, NvCenter, NvEvent, decay_exponent, segments,
};
use crate::error::{QsoError, Result};
use crate::linalg::dense::{cmul, from_spectrum};
use crate::linalg::{CMatrix, eigh};
use crate::rng::Rng;
use crate::spin::{HBAR, Isotope, MU0_OVER_4PI, TWO_PI};

/// Cubic lattice constant of diamond (m, at room temperature).
pub const DIAMOND_LATTICE_CONSTANT: f64 = 3.567e-10;

/// Natural abundance of ¹³C (IUPAC: 1.07 %, often quoted as 1.1 %).
pub const NATURAL_C13_ABUNDANCE: f64 = 0.0107;

/// Most ¹³C of a bath.
pub const MAX_BATH_SPINS: usize = 20_000;

/// Largest cluster of the expansion.
pub const MAX_CLUSTER_ORDER: usize = 8;

/// Most clusters of an expansion.
pub const MAX_CLUSTERS: usize = 1 << 21;

/// Most spins of an exact propagation of the whole bath.
pub const MAX_EXACT_BATH: usize = 10;

/// Most operations (estimated) of one coherence calculation: a few
/// minutes on one core.
pub const MAX_BATH_WORK: f64 = 5e11;

/// Most sampled noise paths per sequence.
pub const MAX_BATH_TRAJECTORIES: usize = 1 << 24;

/// Clusters evaluated together (bounds the memory of one batch).
const CLUSTER_BATCH: usize = 1 << 14;

/// Most lattice sites searched for a bath.
const MAX_SITES: f64 = 2e7;

/// A ¹³C of the bath: its position from the vacancy (m, in the frame of
/// the centre: z along the NV axis from the vacancy to the nitrogen) and
/// its secular hyperfine couplings `(A_zx, A_zy, A_zz)` (Hz).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BathSpin {
    pub position: [f64; 3],
    pub hyperfine: [f64; 3],
}

/// `(μ₀/4π) γ_e γ_C h` (Hz m³): the point-dipole hyperfine scale.
fn hyperfine_scale() -> f64 {
    MU0_OVER_4PI * NV_GYROMAGNETIC_RATIO * gamma_c() * TWO_PI * HBAR
}

/// `γ_C/2π` (Hz/T).
fn gamma_c() -> f64 {
    Isotope::C13.gamma() / TWO_PI
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

impl BathSpin {
    /// A ¹³C at `position` with the hyperfine coupling of a point dipole at
    /// the vacancy: `A_zj = (μ₀/4π) γ_e γ_C h (3 n_z n_j − δ_zj) / r³`
    /// (the electron's moment is opposite to its spin).
    pub fn point_dipole(position: [f64; 3]) -> Result<Self> {
        let r = norm(position);
        if !(r.is_finite() && r > 0.0) {
            return Err(QsoError::invalid(
                "position",
                "a finite position away from the vacancy",
            ));
        }
        let n = position.map(|x| x / r);
        let k = hyperfine_scale() / (r * r * r);
        Ok(BathSpin {
            position,
            hyperfine: [
                3.0 * n[2] * n[0] * k,
                3.0 * n[2] * n[1] * k,
                (3.0 * n[2] * n[2] - 1.0) * k,
            ],
        })
    }
}

/// Estimated number of lattice sites within `radius` (8 atoms per cubic
/// cell), checked against the search limit.
fn site_count(radius: f64) -> Result<f64> {
    let sites =
        8.0 * 4.0 / 3.0 * std::f64::consts::PI * (radius / DIAMOND_LATTICE_CONSTANT).powi(3);
    if !(radius.is_finite() && radius > 0.0) || sites > MAX_SITES {
        return Err(QsoError::invalid(
            "radius",
            format!("a positive radius with at most {MAX_SITES:.0e} sites (about 30 nm)"),
        ));
    }
    Ok(sites)
}

/// Visit every carbon site within `radius` with its lattice coordinates
/// (units of a/4) and its position in the frame of the centre.
fn lattice(radius: f64, mut visit: impl FnMut([i64; 3], [f64; 3])) -> Result<()> {
    site_count(radius)?;
    let (s2, s3, s6) = (2f64.sqrt(), 3f64.sqrt(), 6f64.sqrt());
    let (ex, ey, ez) = (
        [1.0 / s6, 1.0 / s6, -2.0 / s6],
        [-1.0 / s2, 1.0 / s2, 0.0],
        [1.0 / s3, 1.0 / s3, 1.0 / s3],
    );
    let q = DIAMOND_LATTICE_CONSTANT / 4.0;
    let reach = (radius / q).ceil() as i64;
    for i in -reach..=reach {
        for j in -reach..=reach {
            for k in -reach..=reach {
                // Face-centred cubic sites (even, sum ≡ 0 mod 4) and the
                // second basis atom (odd, sum ≡ 3 mod 4), in units of a/4.
                let even = i % 2 == 0 && j % 2 == 0 && k % 2 == 0;
                let odd = i % 2 != 0 && j % 2 != 0 && k % 2 != 0;
                let sum = (i + j + k).rem_euclid(4);
                if !((even && sum == 0) || (odd && sum == 3)) {
                    continue;
                }
                if (i, j, k) == (0, 0, 0) || (i, j, k) == (1, 1, 1) {
                    continue;
                }
                let p = [i as f64 * q, j as f64 * q, k as f64 * q];
                if norm(p) > radius {
                    continue;
                }
                let dot = |e: [f64; 3]| e[0] * p[0] + e[1] * p[1] + e[2] * p[2];
                visit([i, j, k], [dot(ex), dot(ey), dot(ez)]);
            }
        }
    }
    Ok(())
}

/// The carbon sites of diamond within `radius` (m) of the vacancy, in the
/// frame of the centre (the vacancy at a lattice site, the nitrogen at its
/// neighbour along [111], both left out), in a fixed order.
pub fn diamond_sites(radius: f64) -> Result<Vec<[f64; 3]>> {
    let mut out = Vec::new();
    lattice(radius, |_, p| out.push(p))?;
    Ok(out)
}

/// A uniform number in [0, 1) for one lattice site and seed (SplitMix64
/// over the seed and the coordinates), so that every site's draw does not
/// depend on the radius.
fn site_draw(seed: u64, site: [i64; 3]) -> f64 {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let x = site.iter().fold(mix(seed), |x, &c| mix(x ^ c as u64));
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A random bath: every site of [`diamond_sites`] within `radius` (m)
/// holds a ¹³C with probability `abundance`, with point-dipole hyperfine
/// couplings. Each site's draw depends only on the seed and the site, so
/// the bath within a smaller radius is the same for the same seed. Nuclei
/// much closer than about 0.5 nm have contact terms the point dipole
/// misses: replace their [`BathSpin::hyperfine`] with given values, or
/// leave them out ([`SpinBath::without_spins_within`]).
pub fn diamond_bath(abundance: f64, radius: f64, seed: u64) -> Result<Vec<BathSpin>> {
    if !(abundance > 0.0 && abundance <= 1.0) {
        return Err(QsoError::invalid("abundance", "a fraction in (0, 1]"));
    }
    let expected = abundance * site_count(radius)?;
    if expected > MAX_BATH_SPINS as f64 {
        return Err(QsoError::invalid(
            "radius",
            format!("about {expected:.0} spins exceed {MAX_BATH_SPINS}"),
        ));
    }
    let mut spins = Vec::new();
    let mut error = None;
    lattice(radius, |site, p| {
        if site_draw(seed, site) < abundance {
            match BathSpin::point_dipole(p) {
                Ok(s) => spins.push(s),
                Err(e) => error = Some(e),
            }
        }
    })?;
    if let Some(e) = error {
        return Err(e);
    }
    if spins.len() > MAX_BATH_SPINS {
        return Err(QsoError::invalid(
            "radius",
            format!("{} spins exceed {MAX_BATH_SPINS}", spins.len()),
        ));
    }
    Ok(spins)
}

/// The cluster-correlation expansion: clusters of up to `order` spins,
/// connected by pairs closer than `cutoff` (m), whose couplings are kept.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cce {
    pub order: usize,
    pub cutoff: f64,
}

impl Default for Cce {
    /// Pairs within 0.8 nm, the usual choice for natural diamond.
    fn default() -> Self {
        Cce {
            order: 2,
            cutoff: 8e-10,
        }
    }
}

/// A sequence as the two branches of the coherence: the free periods with
/// whether the branch that starts in `m` is in `m`, and the weights `w_m`,
/// `w_0` of the branches in the final population.
#[derive(Clone, Debug)]
struct Branches {
    periods: Vec<(f64, bool)>,
    weights: [C64; 2],
}

/// `exp(−iθ/2 (cos φ σ_x + sin φ σ_y))` on the qubit `(|m⟩, |0⟩)`, angles
/// in degrees.
fn rotation(angle: f64, phase: f64) -> [[C64; 2]; 2] {
    let (c, s) = (
        (angle.to_radians() / 2.0).cos(),
        (angle.to_radians() / 2.0).sin(),
    );
    let p = C64::from_polar(1.0, -phase.to_radians());
    let mi = C64::new(0.0, -s);
    [
        [C64::new(c, 0.0), mi * p],
        [mi * p.conj(), C64::new(c, 0.0)],
    ]
}

fn branches(events: &[NvEvent]) -> Result<Branches> {
    NvCenter::check_events(events)?;
    let bad = || {
        QsoError::invalid(
            "sequence",
            "a pulse, free periods and π pulses (or multiples of 2π), a pulse",
        )
    };
    let n = events.len();
    let (
        Some(&NvEvent::Pulse {
            angle: a0,
            phase: p0,
        }),
        Some(&NvEvent::Pulse {
            angle: a1,
            phase: p1,
        }),
    ) = (events.first(), events.last())
    else {
        return Err(bad());
    };
    if n < 2 {
        return Err(bad());
    }
    let (first, last) = (rotation(a0, p0), rotation(a1, p1));
    // The branch that starts in |m⟩ (index 0) and the one in |0⟩ (index 1),
    // from the pumped |0⟩: amplitude and current level of each.
    let mut amp = [first[0][1], first[1][1]];
    let mut at = [0usize, 1usize];
    let mut periods: Vec<(f64, bool)> = Vec::new();
    for e in &events[1..n - 1] {
        match *e {
            NvEvent::Free(t) => {
                let level = at[0] == 0;
                match periods.last_mut() {
                    Some((u, l)) if *l == level => *u += t,
                    _ => periods.push((t, level)),
                }
            }
            NvEvent::Pulse { angle, phase } => {
                let turns = angle.rem_euclid(360.0) / 90.0;
                let q = turns.round();
                if (turns - q).abs() > 1e-11 {
                    return Err(bad());
                }
                match q as i64 % 4 {
                    0 => {}
                    2 => {
                        let r = rotation(180.0, phase);
                        for b in 0..2 {
                            let (i, j) = (at[b], 1 - at[b]);
                            amp[b] *= r[j][i];
                            at[b] = j;
                        }
                    }
                    _ => return Err(bad()),
                }
            }
        }
    }
    Ok(Branches {
        periods,
        weights: [last[1][at[0]] * amp[0], last[1][at[1]] * amp[1]],
    })
}

impl Branches {
    fn signal(&self, l: C64) -> f64 {
        let [wm, w0] = self.weights;
        wm.norm_sqr() + w0.norm_sqr() + 2.0 * (wm * w0.conj() * l).re
    }
}

/// The bath of an NV centre in a field `field` (T) along its axis, seen
/// by the qubit `{0, transition}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpinBath {
    pub field: f64,
    pub transition: i8,
    pub spins: Vec<BathSpin>,
}

/// Pauli matrices / 2: `I_x`, `I_y`, `I_z`.
fn spin_half() -> [[[C64; 2]; 2]; 3] {
    let (z, h) = (C64::new(0.0, 0.0), C64::new(0.5, 0.0));
    let ih = C64::new(0.0, 0.5);
    [[[z, h], [h, z]], [[z, -ih], [ih, z]], [[h, z], [z, -h]]]
}

/// Add `op` (2 × 2) on spin `i` (bit `i`) to `h`.
fn add_one(h: &mut CMatrix, i: usize, op: &[[C64; 2]; 2]) {
    let bit = 1usize << i;
    for s in 0..h.nrows() {
        let a = (s >> i) & 1;
        for b in 0..2 {
            let z = op[b][a];
            if z != C64::new(0.0, 0.0) {
                h[((s & !bit) | (b << i), s)] += z;
            }
        }
    }
}

/// Add `op` (4 × 4, index `2·bit_i + bit_j`) on spins `i` and `j`.
fn add_two(h: &mut CMatrix, i: usize, j: usize, op: &[[C64; 4]; 4]) {
    let mask = !((1usize << i) | (1usize << j));
    for s in 0..h.nrows() {
        let col = 2 * ((s >> i) & 1) + ((s >> j) & 1);
        for (row, line) in op.iter().enumerate() {
            let z = line[col];
            if z != C64::new(0.0, 0.0) {
                let t = (s & mask) | ((row >> 1) << i) | ((row & 1) << j);
                h[(t, s)] += z;
            }
        }
    }
}

impl SpinBath {
    pub fn new(field: f64, transition: i8, spins: Vec<BathSpin>) -> Self {
        SpinBath {
            field,
            transition,
            spins,
        }
    }

    /// Check the field, the transition and the spins.
    pub fn validate(&self) -> Result<()> {
        if !self.field.is_finite() {
            return Err(QsoError::invalid("field", "the field must be finite"));
        }
        if self.transition != 1 && self.transition != -1 {
            return Err(QsoError::invalid("transition", "−1 or +1"));
        }
        if self.spins.len() > MAX_BATH_SPINS {
            return Err(QsoError::invalid(
                "spins",
                format!("at most {MAX_BATH_SPINS} spins"),
            ));
        }
        let nitrogen = [0.0, 0.0, DIAMOND_LATTICE_CONSTANT * 3f64.sqrt() / 4.0];
        for (k, s) in self.spins.iter().enumerate() {
            if s.position
                .iter()
                .chain(&s.hyperfine)
                .any(|x| !x.is_finite())
            {
                return Err(QsoError::invalid(
                    "spins",
                    format!("spin {k} needs a finite position and hyperfine coupling"),
                ));
            }
            let to_n = [0, 1, 2].map(|c| s.position[c] - nitrogen[c]);
            if norm(s.position) < 1e-12 || norm(to_n) < 1e-12 {
                return Err(QsoError::invalid(
                    "spins",
                    format!("spin {k} sits on the vacancy or the nitrogen"),
                ));
            }
        }
        // Two spins at one place: neighbours in x within 10⁻¹² m, after
        // sorting.
        let mut order: Vec<usize> = (0..self.spins.len()).collect();
        order.sort_by(|&i, &j| self.spins[i].position[0].total_cmp(&self.spins[j].position[0]));
        for (a, &i) in order.iter().enumerate() {
            let p = self.spins[i].position;
            for &j in &order[a + 1..] {
                let q = self.spins[j].position;
                if q[0] - p[0] >= 1e-12 {
                    break;
                }
                if norm([p[0] - q[0], p[1] - q[1], p[2] - q[2]]) < 1e-12 {
                    return Err(QsoError::invalid(
                        "spins",
                        format!("spins {} and {} sit at one place", i.min(j), i.max(j)),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The bath without the spins closer than `radius` (m) to the vacancy,
    /// whose contact couplings the point dipole misses.
    pub fn without_spins_within(mut self, radius: f64) -> Self {
        self.spins.retain(|s| norm(s.position) >= radius);
        self
    }

    fn check_cutoff(cutoff: f64) -> Result<()> {
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(QsoError::invalid(
                "cutoff",
                "a distance ≥ 0 (∞ for every pair)",
            ));
        }
        Ok(())
    }

    /// The spins closer than `cutoff` to each spin, found on a grid of
    /// cells of that size; at most [`MAX_CLUSTERS`] pairs.
    fn neighbours(&self, cutoff: f64) -> Result<Vec<Vec<usize>>> {
        let n = self.spins.len();
        let mut out = vec![Vec::new(); n];
        let too_many = || {
            QsoError::invalid(
                "cutoff",
                format!("more than {MAX_CLUSTERS} pairs within the cutoff: lower it"),
            )
        };
        let close = |i: usize, j: usize| {
            let (p, q) = (self.spins[i].position, self.spins[j].position);
            norm([p[0] - q[0], p[1] - q[1], p[2] - q[2]]) < cutoff
        };
        let mut pairs = 0usize;
        if cutoff == 0.0 {
            return Ok(out);
        }
        if !cutoff.is_finite() {
            if n * n.saturating_sub(1) / 2 > MAX_CLUSTERS {
                return Err(too_many());
            }
            for i in 0..n {
                for j in i + 1..n {
                    out[i].push(j);
                    out[j].push(i);
                }
            }
            return Ok(out);
        }
        let cell = |p: [f64; 3]| p.map(|x| (x / cutoff).floor() as i64);
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (i, s) in self.spins.iter().enumerate() {
            grid.entry(cell(s.position)).or_default().push(i);
        }
        for (i, s) in self.spins.iter().enumerate() {
            let c = cell(s.position);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(others) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                            continue;
                        };
                        for &j in others {
                            if j > i && close(i, j) {
                                out[i].push(j);
                                out[j].push(i);
                                pairs += 1;
                                if pairs > MAX_CLUSTERS {
                                    return Err(too_many());
                                }
                            }
                        }
                    }
                }
            }
        }
        for list in &mut out {
            list.sort_unstable();
        }
        Ok(out)
    }

    /// The connected clusters of up to `order` spins (pairs closer than
    /// `cutoff`), by size and then in increasing order of their spins.
    pub fn clusters(&self, order: usize, cutoff: f64) -> Result<Vec<Vec<usize>>> {
        self.validate()?;
        Self::check_cutoff(cutoff)?;
        if !(1..=MAX_CLUSTER_ORDER).contains(&order) {
            return Err(QsoError::invalid(
                "order",
                format!("clusters of 1 to {MAX_CLUSTER_ORDER} spins"),
            ));
        }
        if order == 1 {
            return Ok((0..self.spins.len()).map(|i| vec![i]).collect());
        }
        let neighbours = self.neighbours(cutoff)?;
        let mut all: Vec<Vec<usize>> = (0..self.spins.len()).map(|i| vec![i]).collect();
        let mut current = all.clone();
        for _ in 1..order {
            let mut next: HashSet<Vec<usize>> = HashSet::new();
            for c in &current {
                for &i in c {
                    for &j in &neighbours[i] {
                        if let Err(k) = c.binary_search(&j) {
                            let mut grown = c.clone();
                            grown.insert(k, j);
                            next.insert(grown);
                        }
                    }
                }
                if all.len() + next.len() > MAX_CLUSTERS {
                    return Err(QsoError::invalid(
                        "order",
                        format!("more than {MAX_CLUSTERS} clusters: lower the order or the cutoff"),
                    ));
                }
            }
            let mut next: Vec<Vec<usize>> = next.into_iter().collect();
            next.sort();
            all.extend(next.iter().cloned());
            current = next;
        }
        Ok(all)
    }

    /// The bath Hamiltonians `H_m` and `H_0` (rad/s) of the spins
    /// `members`, with the dipolar couplings of the pairs closer than
    /// `cutoff`, diagonalised.
    fn spectra(&self, members: &[usize], cutoff: f64) -> [(Vec<f64>, CMatrix); 2] {
        let k = members.len();
        let d = 1usize << k;
        let ops = spin_half();
        let gc = gamma_c();
        let dipolar = MU0_OVER_4PI * gc * gc * TWO_PI * HBAR;
        let mut base = CMatrix::zeros(d, d);
        for (a, &i) in members.iter().enumerate() {
            let zeeman = ops[2].map(|row| row.map(|z| z * (-TWO_PI * gc * self.field)));
            add_one(&mut base, a, &zeeman);
            for (b, &j) in members.iter().enumerate().skip(a + 1) {
                let (p, q) = (self.spins[i].position, self.spins[j].position);
                let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                let r = norm(v);
                if r >= cutoff {
                    continue;
                }
                let n = v.map(|x| x / r);
                let scale = TWO_PI * dipolar / (r * r * r);
                let mut op = [[C64::new(0.0, 0.0); 4]; 4];
                for (u, ou) in ops.iter().enumerate() {
                    for (w, ow) in ops.iter().enumerate() {
                        let c = scale * (f64::from(u8::from(u == w)) - 3.0 * n[u] * n[w]);
                        if c == 0.0 {
                            continue;
                        }
                        for (row, line) in op.iter_mut().enumerate() {
                            for (col, z) in line.iter_mut().enumerate() {
                                *z += ou[row >> 1][col >> 1] * ow[row & 1][col & 1] * c;
                            }
                        }
                    }
                }
                add_two(&mut base, a, b, &op);
            }
        }
        let level = |m: f64| {
            let mut h = base.clone();
            if m != 0.0 {
                for (a, &i) in members.iter().enumerate() {
                    let hf = self.spins[i].hyperfine;
                    let mut op = [[C64::new(0.0, 0.0); 2]; 2];
                    for (c, oc) in ops.iter().enumerate() {
                        for (row, line) in op.iter_mut().enumerate() {
                            for (col, z) in line.iter_mut().enumerate() {
                                *z += oc[row][col] * (TWO_PI * m * hf[c]);
                            }
                        }
                    }
                    add_one(&mut h, a, &op);
                }
            }
            eigh(&h)
        };
        [level(f64::from(self.transition)), level(0.0)]
    }

    /// `L_C = Tr(U_m U_0†)/2^k` of the spins `members` for every sequence.
    fn cluster_coherence(&self, members: &[usize], branches: &[Branches], cutoff: f64) -> Vec<C64> {
        let spectra = self.spectra(members, cutoff);
        let d = 1usize << members.len();
        // The exponentials of the distinct periods of one sequence at a
        // time, so that memory stays at a few matrices per sequence.
        let mut cache: [HashMap<u64, CMatrix>; 2] = [HashMap::new(), HashMap::new()];
        branches
            .iter()
            .map(|b| {
                cache.iter_mut().for_each(HashMap::clear);
                let (mut um, mut u0) = (CMatrix::identity(d, d), CMatrix::identity(d, d));
                for &(t, in_m) in &b.periods {
                    for (level, (values, vectors)) in spectra.iter().enumerate() {
                        cache[level].entry(t.to_bits()).or_insert_with(|| {
                            from_spectrum(values, vectors, |e| C64::from_polar(1.0, -e * t))
                        });
                    }
                    // Level index 0 is m, 1 is 0.
                    let (lm, l0) = if in_m { (0, 1) } else { (1, 0) };
                    um = cmul(&cache[lm][&t.to_bits()], &um);
                    u0 = cmul(&cache[l0][&t.to_bits()], &u0);
                }
                um.iter()
                    .zip(u0.iter())
                    .map(|(a, b)| a * b.conj())
                    .sum::<C64>()
                    / d as f64
            })
            .collect()
    }

    fn parse(sequences: &[Vec<NvEvent>]) -> Result<Vec<Branches>> {
        sequences.iter().map(|s| branches(s)).collect()
    }

    /// Operations (estimated) for clusters of the given sizes: two
    /// diagonalisations per cluster, and per free period two exponentials
    /// and two products, with a fixed cost per period.
    fn check_work(sizes: impl Iterator<Item = usize>, branches: &[Branches]) -> Result<()> {
        let periods: usize = branches.iter().map(|b| b.periods.len()).sum();
        let work: f64 = sizes
            .map(|k| {
                let d = (1u64 << k) as f64;
                60.0 * d * d * d + periods as f64 * (64.0 * d * d * d + 400.0)
            })
            .sum();
        if work > MAX_BATH_WORK {
            return Err(QsoError::invalid(
                "cce",
                format!(
                    "about {work:.1e} operations (limit {MAX_BATH_WORK:.1e}): lower the order, the cutoff or the number of points"
                ),
            ));
        }
        Ok(())
    }

    /// The coherence `L` of the bath after each sequence, by the
    /// cluster-correlation expansion `cce`. The expansion can diverge: a
    /// cluster whose subclusters have almost lost their coherence divides
    /// by nearly zero (strongly coupled pairs, at high abundance, with a
    /// large cutoff or a high order, and for free decays). A result that is
    /// not finite or whose modulus exceeds 1 (by more than 10⁻⁶) is
    /// refused; a lower order or a smaller cutoff usually helps.
    pub fn coherence(&self, sequences: &[Vec<NvEvent>], cce: &Cce) -> Result<Vec<C64>> {
        let branches = Self::parse(sequences)?;
        let clusters = self.clusters(cce.order, cce.cutoff)?;
        Self::check_work(clusters.iter().map(Vec::len), &branches)?;
        let mut total = vec![C64::new(1.0, 0.0); branches.len()];
        let mut reduced: HashMap<Vec<usize>, Vec<C64>> = HashMap::new();
        for size in 1..=cce.order {
            let list: Vec<&Vec<usize>> = clusters.iter().filter(|c| c.len() == size).collect();
            for batch in list.chunks(CLUSTER_BATCH) {
                let raw = crate::parallel::map_tasks(batch.len(), |k| {
                    self.cluster_coherence(batch[k], &branches, cce.cutoff)
                });
                for (&members, mut l) in batch.iter().zip(raw) {
                    // Divide by every connected proper subcluster.
                    for mask in 1..(1u32 << size) - 1 {
                        let sub: Vec<usize> = (0..size)
                            .filter(|&b| mask >> b & 1 == 1)
                            .map(|b| members[b])
                            .collect();
                        if let Some(s) = reduced.get(&sub) {
                            for (x, y) in l.iter_mut().zip(s) {
                                *x /= y;
                            }
                        }
                    }
                    for (t, x) in total.iter_mut().zip(&l) {
                        *t *= x;
                    }
                    if size < cce.order {
                        reduced.insert(members.clone(), l);
                    }
                }
            }
        }
        if let Some((k, l)) = total
            .iter()
            .enumerate()
            .find(|(_, l)| !(l.is_finite() && l.norm() <= 1.0 + 1e-6))
        {
            return Err(QsoError::invalid(
                "cce",
                format!(
                    "the expansion diverges at sequence {k} (|L| = {:.3e}): lower the order or the cutoff",
                    l.norm()
                ),
            ));
        }
        Ok(total)
    }

    /// The coherence of the whole bath propagated exactly (at most
    /// [`MAX_EXACT_BATH`] spins), with the couplings of the pairs closer
    /// than `cutoff`.
    pub fn exact_coherence(&self, sequences: &[Vec<NvEvent>], cutoff: f64) -> Result<Vec<C64>> {
        self.validate()?;
        Self::check_cutoff(cutoff)?;
        if self.spins.len() > MAX_EXACT_BATH {
            return Err(QsoError::invalid(
                "spins",
                format!("an exact propagation holds at most {MAX_EXACT_BATH} spins"),
            ));
        }
        let branches = Self::parse(sequences)?;
        if self.spins.is_empty() {
            return Ok(vec![C64::new(1.0, 0.0); branches.len()]);
        }
        Self::check_work(std::iter::once(self.spins.len()), &branches)?;
        let members: Vec<usize> = (0..self.spins.len()).collect();
        Ok(self.cluster_coherence(&members, &branches, cutoff))
    }

    /// The `m_s = 0` population after each sequence (ideal pulses on the
    /// qubit `{0, transition}`), by the expansion `cce`.
    pub fn signal(&self, sequences: &[Vec<NvEvent>], cce: &Cce) -> Result<Vec<f64>> {
        let branches = Self::parse(sequences)?;
        let l = self.coherence(sequences, cce)?;
        Ok(branches.iter().zip(&l).map(|(b, &l)| b.signal(l)).collect())
    }

    /// The population with classical field noise as well (the noise of
    /// [`FieldNoise`], coupling as `2π b(t) S_z`), with its standard error.
    /// The noise multiplies the coherence by `⟨e^{−i m φ}⟩`, `φ = 2π∫b y
    /// dt`: `e^{−χ}` exactly for Gaussian noise (error 0), an average over
    /// `trajectories` sampled paths (seeded) for telegraph noise. Sequences
    /// need π/2 pulses at both ends, as for [`decay_exponent`].
    pub fn signal_with_noise(
        &self,
        sequences: &[Vec<NvEvent>],
        cce: &Cce,
        noise: &FieldNoise,
        trajectories: usize,
        seed: u64,
    ) -> Result<Vec<(f64, f64)>> {
        noise.validate()?;
        let branches = Self::parse(sequences)?;
        let l = self.coherence(sequences, cce)?;
        let m = f64::from(self.transition);
        if noise.is_gaussian() {
            return sequences
                .iter()
                .zip(&branches)
                .zip(&l)
                .map(|((s, b), &l)| Ok((b.signal(l * (-decay_exponent(s, noise)?).exp()), 0.0)))
                .collect();
        }
        if !(1..=MAX_BATH_TRAJECTORIES).contains(&trajectories) {
            return Err(QsoError::invalid(
                "trajectories",
                format!("1 to {MAX_BATH_TRAJECTORIES} paths"),
            ));
        }
        // Sampling: a fixed cost per period and component and one per flip.
        let rates: Vec<f64> = noise.components().iter().map(|c| c.1 / 2.0).collect();
        let mut work = 0.0;
        for seq in sequences {
            for (t0, t1, _) in segments(seq)? {
                work += rates
                    .iter()
                    .map(|r| super::SAMPLE_WORK * (1.0 + r * (t1 - t0)))
                    .sum::<f64>();
            }
        }
        super::check_work(work * trajectories as f64)?;
        let root = Rng::seed_from_u64(seed);
        sequences
            .iter()
            .zip(&branches)
            .zip(&l)
            .enumerate()
            .map(|(k, ((s, b), &l))| {
                let segs = segments(s)?;
                let base = root.fork(k as u64);
                let values = crate::parallel::map_tasks(trajectories, |j| {
                    let mut rng = base.fork(j as u64);
                    let mut path = NoisePath::new(noise, &mut rng);
                    let phi: f64 = segs
                        .iter()
                        .map(|&(t0, t1, sign)| sign * path.integral(t1 - t0, &[], &mut rng))
                        .sum::<f64>()
                        * TWO_PI;
                    b.signal(l * C64::from_polar(1.0, -m * phi))
                });
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let error = if values.len() < 2 {
                    f64::NAN
                } else {
                    (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
                };
                Ok((mean, error))
            })
            .collect()
    }
}

/// The population after each sequence averaged over `configurations`
/// random baths of [`diamond_bath`] (seeded), with its standard error over
/// the configurations (NaN for one).
#[allow(clippy::too_many_arguments)]
pub fn ensemble_signal(
    field: f64,
    transition: i8,
    abundance: f64,
    radius: f64,
    configurations: usize,
    seed: u64,
    sequences: &[Vec<NvEvent>],
    cce: &Cce,
) -> Result<Vec<(f64, f64)>> {
    if configurations == 0 {
        return Err(QsoError::invalid("configurations", "at least one"));
    }
    let root = Rng::seed_from_u64(seed);
    // Every bath first, so that one out of range fails before any work.
    let baths = (0..configurations)
        .map(|k| {
            let bath_seed = root.fork(k as u64).next_u64();
            let bath = SpinBath::new(
                field,
                transition,
                diamond_bath(abundance, radius, bath_seed)?,
            );
            bath.validate()?;
            Ok(bath)
        })
        .collect::<Result<Vec<_>>>()?;
    let runs = baths
        .iter()
        .map(|bath| bath.signal(sequences, cce))
        .collect::<Result<Vec<_>>>()?;
    let n = configurations as f64;
    Ok((0..sequences.len())
        .map(|j| {
            let mean = runs.iter().map(|r| r[j]).sum::<f64>() / n;
            let error = if configurations < 2 {
                f64::NAN
            } else {
                (runs.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
            };
            (mean, error)
        })
        .collect())
}

#[cfg(test)]
mod tests;
