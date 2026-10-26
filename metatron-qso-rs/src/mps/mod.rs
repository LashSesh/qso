//! Matrix-product-state simulation of circuits with little entanglement.
//!
//! An n-qubit state is stored as n tensors `A[s]` of shape `l × 2 × r`
//! whose product gives every amplitude; the bond dimensions `l`, `r` grow
//! with the entanglement across each cut, not with n (Vidal, Phys. Rev.
//! Lett. 91, 147902 (2003); Schollwöck, Ann. Phys. 326, 96 (2011)). A GHZ
//! state of any size has bond dimension 2; a state with k ebits across a
//! cut needs at least 2^k there.
//!
//! Qubit q is stored at site q. The state is kept in mixed canonical form
//! around an orthogonality centre. A gate on several qubits routes them
//! to adjacent sites with swaps, contracts those sites with the centre,
//! applies the gate, splits the result again and swaps the qubits back.
//! A split keeps at most `max_bond` Schmidt values and drops the smallest
//! ones whose total weight is at most `cutoff` (relative to the state);
//! both are [`MpsOptions`]. A split computes the Schmidt basis from the
//! Hermitian eigendecomposition of a Gram matrix and a Householder QR
//! (see `factor` below for why not nalgebra's SVD).
//!
//! Every truncation is accounted for. If a truncation discards the
//! fraction ε of the norm², the normalised state moves by
//! `√(2 − 2√(1 − ε))` in 2-norm; gates preserve distances, so the distance
//! to the exact state is at most the sum δ of these, and the fidelity is at
//! least `(1 − δ²/2)²` ([`Mps::fidelity_lower_bound`]).

use std::collections::BTreeMap;

use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

use crate::error::{QsoError, Result};
use crate::quantum::gates::{matrix_1q, matrix_2q};
use crate::quantum::{Circuit, GateKind, Instruction, Pauli, PauliSum, StateVector};
use crate::rng::Rng;

pub mod program;

/// Largest register of a matrix-product state.
pub const MAX_MPS_QUBITS: usize = 1 << 16;
/// Largest bond dimension that can be requested.
pub const MAX_BOND_DIMENSION: usize = 1 << 12;
/// Most complex entries of all tensors together (4 GiB).
pub const MAX_MPS_ENTRIES: usize = 1 << 28;
/// Most qubits one gate may act on, controls included.
pub const MAX_GATE_QUBITS: usize = 8;
/// Largest register [`Mps::to_statevector`] converts.
pub const MAX_STATEVECTOR_QUBITS: usize = 24;
/// Most work (shots × tensor entries) one sampling call does.
pub const MAX_SAMPLING_WORK: u128 = 1 << 36;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// Truncation of a matrix-product state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpsOptions {
    /// Most singular values kept at a bond.
    pub max_bond: usize,
    /// Largest total weight (fraction of the norm²) of the singular values
    /// a split may drop.
    pub cutoff: f64,
}

impl Default for MpsOptions {
    /// Bond dimension up to 256; only numerically negligible weight
    /// (10⁻¹⁴) is dropped below that.
    fn default() -> Self {
        MpsOptions {
            max_bond: 256,
            cutoff: 1e-14,
        }
    }
}

impl MpsOptions {
    /// No truncation beyond numerical zeros (weight 10⁻¹⁶ per split), up to
    /// the largest bond dimension.
    pub fn exact() -> Self {
        MpsOptions {
            max_bond: MAX_BOND_DIMENSION,
            cutoff: 1e-16,
        }
    }

    fn check(&self) -> Result<()> {
        if self.max_bond == 0 || self.max_bond > MAX_BOND_DIMENSION {
            return Err(QsoError::invalid(
                "max_bond",
                format!("the bond dimension must lie in 1..={MAX_BOND_DIMENSION}"),
            ));
        }
        if !(0.0..1.0).contains(&self.cutoff) {
            return Err(QsoError::invalid("cutoff", "the cutoff must lie in [0, 1)"));
        }
        Ok(())
    }
}

/// One site tensor `l × 2 × r`, stored row-major with index
/// `(a·2 + s)·r + b`; the same buffer is the `(2l) × r` and the `l × (2r)`
/// matrix.
#[derive(Clone, Debug, PartialEq)]
struct Site {
    l: usize,
    r: usize,
    data: Vec<C64>,
}

/// A pure state of n qubits as a matrix-product state.
#[derive(Clone, Debug)]
pub struct Mps {
    sites: Vec<Site>,
    /// `qubit_at[s]`: the qubit stored at site `s`; `site_of` is its
    /// inverse.
    qubit_at: Vec<usize>,
    site_of: Vec<usize>,
    center: usize,
    options: MpsOptions,
    /// Σ of the discarded weights, and Σ of the distances they cause.
    discarded: f64,
    distance: f64,
    truncations: usize,
    /// A measurement after a truncation voids the fidelity bound.
    bound_void: bool,
}

/// Complex matrix product through three real products (Gauss), which
/// run on nalgebra's optimised real kernels: several times faster than
/// its generic complex product.
fn cmul(a: &DMatrix<C64>, b: &DMatrix<C64>) -> DMatrix<C64> {
    let (ar, ai) = (a.map(|z| z.re), a.map(|z| z.im));
    let (br, bi) = (b.map(|z| z.re), b.map(|z| z.im));
    let p1 = &ar * &br;
    let p2 = &ai * &bi;
    let p3 = (ar + ai) * (br + bi);
    DMatrix::from_fn(p1.nrows(), p1.ncols(), |i, j| {
        C64::new(
            p1[(i, j)] - p2[(i, j)],
            p3[(i, j)] - p1[(i, j)] - p2[(i, j)],
        )
    })
}

/// Row-major `m × k` times `k × n`.
fn matmul(a: &[C64], m: usize, k: usize, b: &[C64], n: usize) -> Vec<C64> {
    let a = DMatrix::from_row_slice(m, k, a);
    let b = DMatrix::from_row_slice(k, n, b);
    row_major(&cmul(&a, &b))
}

fn row_major(m: &DMatrix<C64>) -> Vec<C64> {
    m.transpose().as_slice().to_vec()
}

/// Whether a matrix is unitary to within `tol`.
fn is_unitary(m: &[C64], d: usize, tol: f64) -> bool {
    (0..d).all(|i| {
        (0..d).all(|j| {
            let dot: C64 = (0..d).map(|k| m[k * d + i].conj() * m[k * d + j]).sum();
            let target = if i == j { ONE } else { ZERO };
            (dot - target).norm() <= tol
        })
    })
}

/// The matrix of an instruction on `controls ++ targets`, the first
/// qubit most significant.
fn instruction_matrix(inst: &Instruction, angles: &[f64]) -> Vec<C64> {
    let t = inst.targets.len();
    let u: Vec<C64> = match inst.kind {
        GateKind::Unitary => inst.matrix.as_ref().expect("validated").to_vec(),
        k if k.num_targets() == Some(1) => matrix_1q(k, angles).to_vec(),
        k => matrix_2q(k, angles).to_vec(),
    };
    let c = inst.controls.len();
    if c == 0 {
        return u;
    }
    let (dt, d) = (1usize << t, 1usize << (c + t));
    let mut m = vec![ZERO; d * d];
    for (i, row) in m.chunks_mut(d).enumerate() {
        row[i] = ONE;
    }
    let base = (d - 1) & !(dt - 1); // all controls set
    for i in 0..dt {
        for j in 0..dt {
            m[(base + i) * d + base + j] = u[i * dt + j];
        }
    }
    m
}

impl Mps {
    /// `|0…0⟩` on `n_qubits` qubits.
    pub fn zero(n_qubits: usize, options: MpsOptions) -> Result<Self> {
        if n_qubits == 0 || n_qubits > MAX_MPS_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "matrix-product states (1 or more)",
                max: MAX_MPS_QUBITS,
                requested: n_qubits,
            });
        }
        options.check()?;
        Ok(Mps {
            sites: (0..n_qubits)
                .map(|_| Site {
                    l: 1,
                    r: 1,
                    data: vec![ONE, ZERO],
                })
                .collect(),
            qubit_at: (0..n_qubits).collect(),
            site_of: (0..n_qubits).collect(),
            center: 0,
            options,
            discarded: 0.0,
            distance: 0.0,
            truncations: 0,
            bound_void: false,
        })
    }

    /// Run `circuit` (parameters bound to `theta`) on `|0…0⟩`.
    pub fn from_circuit(circuit: &Circuit, theta: &[f64], options: MpsOptions) -> Result<Self> {
        let mut s = Mps::zero(circuit.n_qubits().max(1), options)?;
        s.apply_circuit(circuit, theta)?;
        Ok(s)
    }

    pub fn n_qubits(&self) -> usize {
        self.sites.len()
    }

    pub fn options(&self) -> MpsOptions {
        self.options
    }

    /// Bond dimensions between consecutive qubits (n − 1 values).
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.sites[..self.sites.len() - 1]
            .iter()
            .map(|s| s.r)
            .collect()
    }

    /// The largest bond dimension.
    pub fn max_bond_dimension(&self) -> usize {
        self.sites.iter().map(|s| s.r).max().unwrap_or(1)
    }

    /// Total discarded weight Σ ε of all truncations.
    pub fn discarded_weight(&self) -> f64 {
        self.discarded
    }

    /// Number of splits that discarded weight.
    pub fn truncations(&self) -> usize {
        self.truncations
    }

    /// A lower bound on the fidelity `|⟨ψ_exact|ψ⟩|²` with the state the
    /// gates prepare exactly: `(1 − δ²/2)²` with δ the sum of the distances
    /// of the truncations (0 if δ² ≥ 2). `None` after a measurement or
    /// reset that followed a truncation, where the exact state is not
    /// defined by the gates alone.
    pub fn fidelity_lower_bound(&self) -> Option<f64> {
        if self.bound_void {
            return None;
        }
        let d2 = self.distance * self.distance;
        Some(if d2 >= 2.0 {
            0.0
        } else {
            (1.0 - d2 / 2.0).powi(2)
        })
    }

    fn entries(&self) -> usize {
        self.sites.iter().map(|s| s.data.len()).sum()
    }

    fn check_qubit(&self, q: usize) -> Result<()> {
        if q >= self.n_qubits() {
            return Err(QsoError::QubitOutOfRange {
                qubit: q,
                n_qubits: self.n_qubits(),
            });
        }
        Ok(())
    }

    // ---- canonical form -------------------------------------------------

    /// Move the orthogonality centre one site to the right (QR).
    fn shift_right(&mut self) {
        let c = self.center;
        let Site { l, r, ref data } = self.sites[c];
        let qr = DMatrix::from_row_slice(2 * l, r, data).qr();
        let (q, rm) = (qr.q(), qr.r());
        let m = q.ncols();
        self.sites[c] = Site {
            l,
            r: m,
            data: row_major(&q),
        };
        let next = &self.sites[c + 1];
        let data = matmul(&row_major(&rm), m, r, &next.data, 2 * next.r);
        self.sites[c + 1] = Site {
            l: m,
            r: next.r,
            data,
        };
        self.center = c + 1;
    }

    /// Move the orthogonality centre one site to the left (QR of A†).
    fn shift_left(&mut self) {
        let c = self.center;
        let Site { l, r, ref data } = self.sites[c];
        let a = DMatrix::from_row_slice(l, 2 * r, data);
        let qr = a.adjoint().qr();
        let (q, rm) = (qr.q(), qr.r()); // A = R† Q†
        let m = q.ncols();
        self.sites[c] = Site {
            l: m,
            r,
            data: row_major(&q.adjoint()),
        };
        let prev = &self.sites[c - 1];
        let data = matmul(&prev.data, 2 * prev.l, l, &row_major(&rm.adjoint()), m);
        self.sites[c - 1] = Site {
            l: prev.l,
            r: m,
            data,
        };
        self.center = c - 1;
    }

    fn move_center(&mut self, to: usize) {
        while self.center < to {
            self.shift_right();
        }
        while self.center > to {
            self.shift_left();
        }
    }

    // ---- gates ----------------------------------------------------------

    /// Contract sites `p..p + k` into `θ` of shape `(l · 2^k) × r`.
    fn theta(&self, p: usize, k: usize) -> (usize, usize, Vec<C64>) {
        let l = self.sites[p].l;
        let mut t = self.sites[p].data.clone();
        let mut rows = 2 * l;
        for s in p + 1..p + k {
            let b = &self.sites[s];
            t = matmul(&t, rows, b.l, &b.data, 2 * b.r);
            rows *= 2;
        }
        (l, self.sites[p + k - 1].r, t)
    }

    /// Singular values kept of a split, by the options; returns their
    /// number and the discarded weight fraction.
    fn keep(&self, sv: &[f64]) -> (usize, f64) {
        let total: f64 = sv.iter().map(|s| s * s).sum();
        if total == 0.0 {
            return (1, 0.0);
        }
        let mut keep = sv.len().min(self.options.max_bond);
        let mut dropped: f64 = sv[keep..].iter().map(|s| s * s).sum();
        while keep > 1 {
            let w = sv[keep - 1] * sv[keep - 1];
            if dropped + w > self.options.cutoff * total {
                break;
            }
            dropped += w;
            keep -= 1;
        }
        (keep, dropped / total)
    }

    /// Factor the row-major `rows × cols` matrix `m` as `Q·F` with `Q` an
    /// isometry (`rows × keep`, orthonormal columns) and `F` normalised;
    /// returns `Q`, `keep`, `F` and the discarded weight fraction
    /// `1 − ‖Q†m‖²/‖m‖²`.
    ///
    /// The kept subspace comes from the Hermitian eigendecomposition of
    /// the Gram matrix on the smaller side (nalgebra's complex SVD is not
    /// reliable for rank-deficient matrices, which are the rule here), and
    /// `Q` from a Householder QR of its basis. The discarded weight is
    /// measured from `F` itself, so it is exact whatever the accuracy of
    /// the eigenvectors.
    fn factor(&self, m: &[C64], rows: usize, cols: usize) -> (Vec<C64>, usize, Vec<C64>, f64) {
        let a = DMatrix::from_row_slice(rows, cols, m);
        let total = a.norm_squared();
        let wide = rows <= cols;
        let gram = if wide {
            cmul(&a, &a.adjoint())
        } else {
            cmul(&a.adjoint(), &a)
        };
        let (values, vectors) = crate::linalg::eigh(&gram);
        let dim = values.len();
        // Descending singular values.
        let sv: Vec<f64> = values.iter().rev().map(|&x| x.max(0.0).sqrt()).collect();
        let (keep, _) = self.keep(&sv);
        let top = DMatrix::from_fn(dim, keep, |i, c| vectors[(i, dim - 1 - c)]);
        let basis = if wide { top } else { cmul(&a, &top) };
        let q = basis.qr().q();
        let f = cmul(&q.adjoint(), &a);
        let kept = f.norm_squared();
        let eps = if total > 0.0 {
            (1.0 - kept / total).max(0.0)
        } else {
            0.0
        };
        let scale = if kept > 0.0 { 1.0 / kept.sqrt() } else { 1.0 };
        (
            row_major(&q),
            q.ncols(),
            row_major(&(f * C64::new(scale, 0.0))),
            eps,
        )
    }

    /// Split `θ` back into sites `p..p + k`, the centre ending at
    /// `p + k − 1`.
    fn split(&mut self, p: usize, k: usize, l: usize, r: usize, mut theta: Vec<C64>) -> Result<()> {
        let mut left = l;
        for j in 0..k - 1 {
            let rows = 2 * left;
            let cols = theta.len() / rows;
            let (site, keep, rest, eps) = self.factor(&theta, rows, cols);
            if eps > 0.0 {
                self.discarded += eps;
                self.distance += (2.0 - 2.0 * (1.0 - eps).max(0.0).sqrt()).max(0.0).sqrt();
                if eps > 1e-14 {
                    self.truncations += 1;
                }
            }
            self.sites[p + j] = Site {
                l: left,
                r: keep,
                data: site,
            };
            theta = rest;
            left = keep;
        }
        self.sites[p + k - 1] = Site {
            l: left,
            r,
            data: theta,
        };
        self.center = p + k - 1;
        if self.entries() > MAX_MPS_ENTRIES {
            return Err(QsoError::invalid(
                "mps",
                format!(
                    "the state needs more than {MAX_MPS_ENTRIES} tensor entries; lower max_bond"
                ),
            ));
        }
        Ok(())
    }

    /// Apply a matrix (in site order) to the adjacent sites `p..p + k`.
    fn apply_sites(&mut self, p: usize, k: usize, m: &[C64]) -> Result<()> {
        if self.center < p {
            self.move_center(p);
        } else if self.center >= p + k {
            self.move_center(p + k - 1);
        }
        let (l, r, theta) = self.theta(p, k);
        let d = 1usize << k;
        if l * d * r > MAX_MPS_ENTRIES / 4 {
            return Err(QsoError::invalid(
                "mps",
                "a gate's tensor is too large; lower max_bond or use fewer qubits per gate",
            ));
        }
        let mut out = vec![ZERO; theta.len()];
        for a in 0..l {
            let block = &theta[a * d * r..(a + 1) * d * r];
            let dst = &mut out[a * d * r..(a + 1) * d * r];
            for (i, row) in m.chunks(d).enumerate() {
                for (j, &mij) in row.iter().enumerate() {
                    if mij == ZERO {
                        continue;
                    }
                    let src = &block[j * r..(j + 1) * r];
                    for (o, &v) in dst[i * r..(i + 1) * r].iter_mut().zip(src) {
                        *o += mij * v;
                    }
                }
            }
        }
        self.split(p, k, l, r, out)
    }

    /// Exchange the qubits at sites `s` and `s + 1`.
    fn swap_sites(&mut self, s: usize) -> Result<()> {
        let mut m = vec![ZERO; 16];
        m[0] = ONE;
        m[6] = ONE;
        m[9] = ONE;
        m[15] = ONE;
        self.apply_sites(s, 2, &m)?;
        let (a, b) = (self.qubit_at[s], self.qubit_at[s + 1]);
        self.qubit_at.swap(s, s + 1);
        self.site_of[a] = s + 1;
        self.site_of[b] = s;
        Ok(())
    }

    /// Move `qubits` to adjacent sites around their median; returns the
    /// first site and the swaps made (their left sites, in order).
    fn gather(&mut self, qubits: &[usize]) -> Result<(usize, Vec<usize>)> {
        let mut sites: Vec<usize> = qubits.iter().map(|&q| self.site_of[q]).collect();
        sites.sort_unstable();
        let k = sites.len();
        let m = k / 2;
        let anchor = sites[m];
        let mut swaps = Vec::new();
        for j in (0..m).rev() {
            let target = anchor - (m - j);
            let mut s = sites[j];
            while s < target {
                self.swap_sites(s)?;
                swaps.push(s);
                s += 1;
            }
        }
        for (j, &start) in sites.iter().enumerate().skip(m + 1) {
            let target = anchor + (j - m);
            let mut s = start;
            while s > target {
                self.swap_sites(s - 1)?;
                swaps.push(s - 1);
                s -= 1;
            }
        }
        Ok((anchor - m, swaps))
    }

    /// Apply a unitary (row-major, dimension `2^qubits.len()`, `qubits[0]`
    /// most significant) to distinct qubits.
    pub fn apply_matrix(&mut self, matrix: &[C64], qubits: &[usize]) -> Result<()> {
        let k = qubits.len();
        if k == 0 || k > MAX_GATE_QUBITS {
            return Err(QsoError::invalid(
                "qubits",
                format!("a gate acts on 1 to {MAX_GATE_QUBITS} qubits"),
            ));
        }
        for (i, &q) in qubits.iter().enumerate() {
            self.check_qubit(q)?;
            if qubits[..i].contains(&q) {
                return Err(QsoError::DuplicateQubit {
                    gate: "matrix",
                    qubit: q,
                });
            }
        }
        let d = 1usize << k;
        if matrix.len() != d * d {
            return Err(QsoError::DimensionMismatch {
                expected: d * d,
                actual: matrix.len(),
            });
        }
        if matrix
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
            || !is_unitary(matrix, d, 1e-9)
        {
            return Err(QsoError::invalid("matrix", "the matrix is not unitary"));
        }
        self.apply_unitary(matrix, qubits)
    }

    fn apply_unitary(&mut self, matrix: &[C64], qubits: &[usize]) -> Result<()> {
        let k = qubits.len();
        if k == 1 {
            // A one-site unitary keeps the canonical form.
            let site = &mut self.sites[self.site_of[qubits[0]]];
            let r = site.r;
            for a in 0..site.l {
                for b in 0..r {
                    let (x0, x1) = (site.data[(a * 2) * r + b], site.data[(a * 2 + 1) * r + b]);
                    site.data[(a * 2) * r + b] = matrix[0] * x0 + matrix[1] * x1;
                    site.data[(a * 2 + 1) * r + b] = matrix[2] * x0 + matrix[3] * x1;
                }
            }
            return Ok(());
        }
        let (p, swaps) = self.gather(qubits)?;
        // Site j holds gate position pos[j]; reorder the matrix to sites.
        let pos: Vec<usize> = (0..k)
            .map(|j| {
                let q = self.qubit_at[p + j];
                qubits.iter().position(|&x| x == q).expect("gathered")
            })
            .collect();
        let d = 1usize << k;
        let to_gate =
            |x: usize| (0..k).fold(0, |g, j| g | (((x >> (k - 1 - j)) & 1) << (k - 1 - pos[j])));
        let perm: Vec<usize> = (0..d).map(to_gate).collect();
        let mut m = vec![ZERO; d * d];
        for x in 0..d {
            for y in 0..d {
                m[x * d + y] = matrix[perm[x] * d + perm[y]];
            }
        }
        self.apply_sites(p, k, &m)?;
        // Swap back, so that qubit q stays at site q.
        for &s in swaps.iter().rev() {
            self.swap_sites(s)?;
        }
        Ok(())
    }

    /// Apply one instruction with parameters `theta`.
    pub fn apply_instruction(&mut self, inst: &Instruction, theta: &[f64]) -> Result<()> {
        inst.validate(self.n_qubits())?;
        for p in &inst.params {
            if let crate::quantum::Param::Symbol(s) = p
                && s.index >= theta.len()
            {
                return Err(QsoError::ParameterCount {
                    expected: s.index + 1,
                    actual: theta.len(),
                });
            }
        }
        if theta.iter().any(|t| !t.is_finite()) {
            return Err(QsoError::invalid("theta", "parameters must be finite"));
        }
        let qubits: Vec<usize> = inst.qubits().collect();
        if qubits.len() > MAX_GATE_QUBITS {
            return Err(QsoError::invalid(
                "gate",
                format!(
                    "{} acts on {} qubits; matrix-product states take gates on at most {MAX_GATE_QUBITS}",
                    inst.kind.name(),
                    qubits.len()
                ),
            ));
        }
        let m = instruction_matrix(inst, &inst.angles(theta));
        self.apply_unitary(&m, &qubits)
    }

    fn gate(
        &mut self,
        kind: GateKind,
        targets: &[usize],
        controls: &[usize],
        params: &[f64],
    ) -> Result<&mut Self> {
        let inst = Instruction {
            kind,
            targets: targets.to_vec(),
            controls: controls.to_vec(),
            params: params
                .iter()
                .map(|&p| crate::quantum::Param::Value(p))
                .collect(),
            matrix: None,
        };
        self.apply_instruction(&inst, &[])?;
        Ok(self)
    }

    pub fn h(&mut self, q: usize) -> Result<&mut Self> {
        self.gate(GateKind::H, &[q], &[], &[])
    }

    pub fn x(&mut self, q: usize) -> Result<&mut Self> {
        self.gate(GateKind::X, &[q], &[], &[])
    }

    pub fn y(&mut self, q: usize) -> Result<&mut Self> {
        self.gate(GateKind::Y, &[q], &[], &[])
    }

    pub fn z(&mut self, q: usize) -> Result<&mut Self> {
        self.gate(GateKind::Z, &[q], &[], &[])
    }

    pub fn s(&mut self, q: usize) -> Result<&mut Self> {
        self.gate(GateKind::S, &[q], &[], &[])
    }

    pub fn t(&mut self, q: usize) -> Result<&mut Self> {
        self.gate(GateKind::T, &[q], &[], &[])
    }

    pub fn rx(&mut self, q: usize, angle: f64) -> Result<&mut Self> {
        self.gate(GateKind::RX, &[q], &[], &[angle])
    }

    pub fn ry(&mut self, q: usize, angle: f64) -> Result<&mut Self> {
        self.gate(GateKind::RY, &[q], &[], &[angle])
    }

    pub fn rz(&mut self, q: usize, angle: f64) -> Result<&mut Self> {
        self.gate(GateKind::RZ, &[q], &[], &[angle])
    }

    pub fn cx(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        self.gate(GateKind::X, &[target], &[control], &[])
    }

    pub fn cz(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.gate(GateKind::Z, &[b], &[a], &[])
    }

    pub fn swap(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.gate(GateKind::Swap, &[a, b], &[], &[])
    }

    pub fn rzz(&mut self, a: usize, b: usize, angle: f64) -> Result<&mut Self> {
        self.gate(GateKind::RZZ, &[a, b], &[], &[angle])
    }

    /// Apply every gate of `circuit` with parameters `theta`.
    pub fn apply_circuit(&mut self, circuit: &Circuit, theta: &[f64]) -> Result<()> {
        if circuit.n_qubits() > self.n_qubits() {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits(),
                actual: circuit.n_qubits(),
            });
        }
        circuit.check_parameters(theta)?;
        for inst in circuit.instructions() {
            self.apply_instruction(inst, theta)?;
        }
        Ok(())
    }

    // ---- readout --------------------------------------------------------

    /// `⟨P⟩` of a Pauli string given as `(qubit, Pauli)` factors.
    ///
    /// The orthogonality centre moves to the string's support, so the
    /// contraction covers only the sites between its first and last
    /// factor; terms evaluated in order of position cost O(n χ³) in all.
    pub fn expectation(&mut self, ops: &[(usize, Pauli)]) -> Result<f64> {
        let n = self.n_qubits();
        let mut op_at: BTreeMap<usize, Pauli> = BTreeMap::new();
        for &(q, p) in ops {
            self.check_qubit(q)?;
            if op_at.insert(self.site_of[q], p).is_some() {
                return Err(QsoError::Parse(format!(
                    "qubit {q} appears twice in Pauli string"
                )));
            }
        }
        op_at.retain(|_, p| *p != Pauli::I);
        if let (Some(&first), Some(&last)) = (op_at.keys().next(), op_at.keys().next_back()) {
            if self.center < first {
                self.move_center(first);
            } else if self.center > last {
                self.move_center(last);
            }
        }
        let lo = op_at
            .keys()
            .next()
            .map_or(self.center, |&s| s.min(self.center));
        let hi = op_at
            .keys()
            .next_back()
            .map_or(self.center, |&s| s.max(self.center));
        debug_assert!(hi < n);
        // Environment at the left bond of `lo`: the identity, since every
        // site left of the centre is left-canonical.
        let l0 = self.sites[lo].l;
        let mut env = DMatrix::<C64>::identity(l0, l0);
        for s in lo..=hi {
            let Site { l, r, ref data } = self.sites[s];
            let slice = |sig: usize| DMatrix::from_fn(l, r, |a, b| data[(a * 2 + sig) * r + b]);
            let a = [slice(0), slice(1)];
            let o = pauli_matrix(op_at.get(&s).copied().unwrap_or(Pauli::I));
            let mut next = DMatrix::<C64>::zeros(r, r);
            for sp in 0..2 {
                if o[sp] == ZERO && o[2 + sp] == ZERO {
                    continue;
                }
                let t = cmul(&env, &a[sp]);
                for sg in 0..2 {
                    let w = o[sg * 2 + sp];
                    if w != ZERO {
                        next += cmul(&a[sg].adjoint(), &t) * w;
                    }
                }
            }
            env = next;
        }
        Ok(env.trace().re)
    }

    /// `⟨P⟩` of a Pauli string such as `"X0 Z1 Y1000"` (optional sign).
    pub fn expectation_str(&mut self, text: &str) -> Result<f64> {
        let (sign, ops) = crate::stabilizer::parse_pauli(text)?;
        Ok(sign * self.expectation(&ops)?)
    }

    /// `⟨H⟩` of a Pauli sum.
    pub fn expectation_sum(&mut self, h: &PauliSum) -> Result<f64> {
        let mut total = 0.0;
        for (c, p) in h.terms() {
            let ops: Vec<(usize, Pauli)> = (0..64)
                .filter_map(|q| match p.get(q) {
                    Pauli::I => None,
                    x => Some((q, x)),
                })
                .collect();
            total += c * self.expectation(&ops)?;
        }
        Ok(total)
    }

    /// The amplitude `⟨b|ψ⟩` of a basis state; `bits[q]` is qubit `q`.
    pub fn amplitude(&self, bits: &[bool]) -> Result<C64> {
        if bits.len() != self.n_qubits() {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits(),
                actual: bits.len(),
            });
        }
        let mut v = vec![ONE];
        for (s, site) in self.sites.iter().enumerate() {
            let sig = usize::from(bits[self.qubit_at[s]]);
            let mut w = vec![ZERO; site.r];
            for (a, &va) in v.iter().enumerate() {
                let row = &site.data[(a * 2 + sig) * site.r..(a * 2 + sig + 1) * site.r];
                for (o, &x) in w.iter_mut().zip(row) {
                    *o += va * x;
                }
            }
            v = w;
        }
        Ok(v[0])
    }

    /// The state vector (qubit 0 least significant), for at most
    /// [`MAX_STATEVECTOR_QUBITS`] qubits.
    pub fn to_statevector(&self) -> Result<StateVector> {
        let n = self.n_qubits();
        if n > MAX_STATEVECTOR_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "conversion of a matrix-product state",
                max: MAX_STATEVECTOR_QUBITS,
                requested: n,
            });
        }
        // Rows: configurations of the sites so far, site 0 most significant.
        let mut t = vec![ONE];
        let mut rows = 1;
        for site in &self.sites {
            t = matmul(&t, rows, site.l, &site.data, 2 * site.r);
            rows *= 2;
        }
        let mut amps = vec![ZERO; 1 << n];
        for (config, &a) in t.iter().enumerate() {
            let idx = (0..n).fold(0, |acc, s| {
                acc | (((config >> (n - 1 - s)) & 1) << self.qubit_at[s])
            });
            amps[idx] = a;
        }
        StateVector::from_amplitudes(amps)
    }

    /// Entanglement entropy in bits across every bond: entry `b` is the
    /// entropy between qubits `0..=b` and the rest.
    pub fn bond_entropies(&mut self) -> Result<Vec<f64>> {
        let n = self.n_qubits();
        self.move_center(0);
        let mut out = Vec::with_capacity(n - 1);
        for c in 0..n - 1 {
            let Site { l, r, ref data } = self.sites[c];
            let a = DMatrix::from_row_slice(2 * l, r, data);
            let gram = if 2 * l <= r {
                cmul(&a, &a.adjoint())
            } else {
                cmul(&a.adjoint(), &a)
            };
            let (values, _) = crate::linalg::eigh(&gram);
            let total: f64 = values.iter().map(|x| x.max(0.0)).sum();
            out.push(
                values
                    .iter()
                    .map(|x| x.max(0.0) / total)
                    .filter(|&p| p > 1e-300)
                    .map(|p| -p * p.log2())
                    .sum::<f64>()
                    .max(0.0),
            );
            self.shift_right();
        }
        Ok(out)
    }

    /// Measure `qubit` in the computational basis, collapsing the state.
    pub fn measure(&mut self, qubit: usize, rng: &mut Rng) -> Result<bool> {
        self.check_qubit(qubit)?;
        let s = self.site_of[qubit];
        self.move_center(s);
        let site = &mut self.sites[s];
        let r = site.r;
        let weight = |sig: usize, site: &Site| -> f64 {
            (0..site.l)
                .map(|a| {
                    site.data[(a * 2 + sig) * r..(a * 2 + sig + 1) * r]
                        .iter()
                        .map(|z| z.norm_sqr())
                        .sum::<f64>()
                })
                .sum()
        };
        let (p0, p1) = (weight(0, site), weight(1, site));
        let one = rng.uniform() * (p0 + p1) >= p0;
        let (drop, keep, p) = if one { (0, 1, p1) } else { (1, 0, p0) };
        let scale = 1.0 / p.sqrt();
        for a in 0..site.l {
            for b in 0..r {
                site.data[(a * 2 + drop) * r + b] = ZERO;
                site.data[(a * 2 + keep) * r + b] *= scale;
            }
        }
        if self.truncations > 0 {
            self.bound_void = true;
        }
        Ok(one)
    }

    /// Reset `qubit` to `|0⟩` (measure, then flip a 1).
    pub fn reset(&mut self, qubit: usize, rng: &mut Rng) -> Result<()> {
        if self.measure(qubit, rng)? {
            self.apply_unitary(&[ZERO, ONE, ONE, ZERO], &[qubit])?;
        }
        Ok(())
    }

    /// Sample every qubit `shots` times without changing the state; bit
    /// `q` of each sample is qubit `q`.
    pub fn sample(&mut self, shots: usize, rng: &mut Rng) -> Result<Vec<Vec<bool>>> {
        let work = shots as u128 * self.entries() as u128;
        if work > MAX_SAMPLING_WORK {
            return Err(QsoError::invalid(
                "shots",
                format!("shots × tensor entries must be at most {MAX_SAMPLING_WORK}"),
            ));
        }
        // With the centre at site 0, every other site is right-canonical:
        // the probability of a prefix is the norm² of its left vector.
        self.move_center(0);
        let n = self.n_qubits();
        let mut out = Vec::with_capacity(shots);
        for _ in 0..shots {
            let mut bits = vec![false; n];
            let mut v = vec![ONE];
            for (s, site) in self.sites.iter().enumerate() {
                let r = site.r;
                let mut w = [vec![ZERO; r], vec![ZERO; r]];
                for (a, &va) in v.iter().enumerate() {
                    for (sig, ws) in w.iter_mut().enumerate() {
                        let row = &site.data[(a * 2 + sig) * r..(a * 2 + sig + 1) * r];
                        for (o, &x) in ws.iter_mut().zip(row) {
                            *o += va * x;
                        }
                    }
                }
                let p: [f64; 2] = [
                    w[0].iter().map(|z| z.norm_sqr()).sum(),
                    w[1].iter().map(|z| z.norm_sqr()).sum(),
                ];
                let sig = usize::from(rng.uniform() * (p[0] + p[1]) >= p[0]);
                bits[self.qubit_at[s]] = sig == 1;
                let scale = 1.0 / p[sig].sqrt();
                v = w[sig].iter().map(|z| z * scale).collect();
            }
            out.push(bits);
        }
        Ok(out)
    }

    /// Sample `shots` times: `{bitstring: count}`, qubit n − 1 first.
    pub fn sample_counts(
        &mut self,
        shots: usize,
        rng: &mut Rng,
    ) -> Result<BTreeMap<String, usize>> {
        let mut counts = BTreeMap::new();
        for bits in self.sample(shots, rng)? {
            let s: String = bits
                .iter()
                .rev()
                .map(|&b| if b { '1' } else { '0' })
                .collect();
            *counts.entry(s).or_insert(0) += 1;
        }
        Ok(counts)
    }
}

fn pauli_matrix(p: Pauli) -> [C64; 4] {
    let i = C64::new(0.0, 1.0);
    match p {
        Pauli::I => [ONE, ZERO, ZERO, ONE],
        Pauli::X => [ZERO, ONE, ONE, ZERO],
        Pauli::Y => [ZERO, -i, i, ZERO],
        Pauli::Z => [ONE, ZERO, ZERO, -ONE],
    }
}

#[cfg(test)]
mod tests;
