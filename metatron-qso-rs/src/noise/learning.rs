//! Learning the noise of a layer of two-qubit Clifford gates as a sparse
//! Pauli–Lindblad model, and error mitigation with the learned model (van
//! den Berg, Minev, Kandala and Temme, Nat. Phys. 19, 1116 (2023)).
//!
//! Twirled, the noise of a layer `U` is a Pauli channel. The sparse model
//! writes it as a product of channels `ρ ↦ w_k ρ + (1 − w_k) P_k ρ P_k`
//! with `w_k = (1 + e^{−2λ_k})/2`, one per generator `P_k`: every
//! one-qubit Pauli and every two-qubit Pauli on a pair of coupled qubits.
//! Its Pauli fidelities are `f_b = exp(−2 Σ_{k: P_k anticommutes with b}
//! λ_k)`.
//!
//! Learning prepares product eigenstates in nine bases, applies the twirled
//! layer an even number of times `d` and measures in the same basis. For a
//! self-inverse layer, `⟨b⟩ = A·(f_b f_{U b U})^{d/2}`: fitting the decay
//! over several depths removes state preparation and measurement errors
//! from the product of fidelities. Where `U b U ≠ b` the two fidelities are
//! taken equal (the symmetry assumption of the paper; [`LearnedNoise`]
//! lists those Paulis), and the rates follow by non-negative least squares.
//! Standard errors come from a bootstrap over the twirls; an error the
//! layer moves to another Pauli (a ZZ error after a cx, say) is learnable
//! only as a product with its image, and the assumption splits it evenly.
//!
//! With a model, [`pec_layered`] cancels the noise of every layer of a
//! circuit (probabilistic error cancellation: one inverse channel per
//! generator, sampled with signs) and [`pea_layered`] amplifies it by
//! inserting Paulis with positive probabilities, for zero-noise
//! extrapolation. Inserted Paulis are single-qubit gates; they are
//! noiseless when the device has no one-qubit gate noise, as when they are
//! merged into neighbouring gates on hardware.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{DensityMatrix, DeviceModel};
use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::mitigation::pec::PecResult;
use crate::mitigation::zne::{Extrapolation, ZneResult, extrapolate};
use crate::parallel::map_tasks;
use crate::quantum::{Circuit, GateKind, Pauli, PauliString, PauliSum};
use crate::rng::Rng;

/// Largest register a layer may act on: every learning and mitigation
/// circuit is simulated with density matrices.
pub const MAX_LEARNING_QUBITS: usize = 10;
/// Most circuits one call simulates.
pub const MAX_LEARNING_CIRCUITS: usize = 1 << 20;
/// Most shots per circuit.
pub const MAX_LEARNING_SHOTS: usize = 1 << 24;
/// Most applications of the layer in one call (circuits times depth).
pub const MAX_LEARNING_LAYERS: usize = 1 << 28;
/// Bootstrap replicas for the standard errors of the rates.
const BOOTSTRAP: usize = 200;

/// A sparse Pauli–Lindblad noise model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparsePauliLindblad {
    pub n_qubits: usize,
    /// Generators `P_k` (non-identity, distinct).
    pub generators: Vec<PauliString>,
    /// Rates `λ_k ≥ 0`.
    pub rates: Vec<f64>,
}

impl SparsePauliLindblad {
    /// A model on `n_qubits` qubits; the generators must be distinct,
    /// non-identity and inside the register, and the rates finite and
    /// non-negative.
    pub fn new(n_qubits: usize, generators: Vec<PauliString>, rates: Vec<f64>) -> Result<Self> {
        let model = Self {
            n_qubits,
            generators,
            rates,
        };
        model.validate()?;
        Ok(model)
    }

    /// Check the conditions of [`SparsePauliLindblad::new`] (the fields are
    /// public and the model can be deserialised).
    pub fn validate(&self) -> Result<()> {
        let n_qubits = self.n_qubits;
        if n_qubits == 0 || n_qubits > 64 {
            return Err(QsoError::invalid(
                "n_qubits",
                "a model acts on 1 to 64 qubits",
            ));
        }
        if self.generators.len() != self.rates.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.generators.len(),
                actual: self.rates.len(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        for g in &self.generators {
            if g.is_identity() || g.min_qubits() > n_qubits || !seen.insert(*g) {
                return Err(QsoError::invalid(
                    "generators",
                    format!("{g} is the identity, outside {n_qubits} qubits or repeated"),
                ));
            }
        }
        if self.rates.iter().any(|r| !(r.is_finite() && *r >= 0.0)) {
            return Err(QsoError::invalid(
                "rates",
                "rates must be finite and non-negative",
            ));
        }
        Ok(())
    }

    /// The Pauli fidelity `f_b = exp(−2 Σ_{P_k anticommutes with b} λ_k)`.
    pub fn fidelity(&self, b: &PauliString) -> f64 {
        let s: f64 = self
            .generators
            .iter()
            .zip(&self.rates)
            .filter(|(g, _)| !g.commutes_with(b))
            .map(|(_, r)| r)
            .sum();
        (-2.0 * s).exp()
    }

    /// Sampling overhead of cancelling this noise once: `γ = exp(2 Σ λ_k)`.
    pub fn overhead(&self) -> f64 {
        (2.0 * self.rates.iter().sum::<f64>()).exp()
    }

    /// The model with every rate multiplied by `factor ≥ 0`.
    pub fn scaled(&self, factor: f64) -> Result<Self> {
        if !(factor.is_finite() && factor >= 0.0) {
            return Err(QsoError::invalid(
                "factor",
                "the factor must be finite and non-negative",
            ));
        }
        Ok(Self {
            rates: self.rates.iter().map(|r| r * factor).collect(),
            ..self.clone()
        })
    }

    /// The Pauli error probabilities of the channel (the identity
    /// included), for registers of at most [`MAX_LEARNING_QUBITS`] qubits.
    pub fn error_probabilities(&self) -> Result<Vec<(PauliString, f64)>> {
        if self.n_qubits > MAX_LEARNING_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "Pauli error probabilities",
                max: MAX_LEARNING_QUBITS,
                requested: self.n_qubits,
            });
        }
        let mut dist: HashMap<(u64, u64), f64> = HashMap::from([((0, 0), 1.0)]);
        for (g, &r) in self.generators.iter().zip(&self.rates) {
            let flip = (1.0 - (-2.0 * r).exp()) / 2.0;
            let mut next: HashMap<(u64, u64), f64> = HashMap::with_capacity(2 * dist.len());
            for (&(x, z), &p) in &dist {
                *next.entry((x, z)).or_insert(0.0) += p * (1.0 - flip);
                *next.entry((x ^ g.x_mask(), z ^ g.z_mask())).or_insert(0.0) += p * flip;
            }
            dist = next;
        }
        let mut out: Vec<(PauliString, f64)> = dist
            .into_iter()
            .map(|((x, z), p)| (PauliString::from_masks(x, z), p))
            .collect();
        out.sort_by_key(|e| e.0);
        Ok(out)
    }
}

/// A learned model with the data behind it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnedNoise {
    pub model: SparsePauliLindblad,
    /// Bootstrap standard error of each rate (of the unconstrained
    /// least-squares fit; see [`learn_layer`]).
    pub std_errors: Vec<f64>,
    /// Every generator's fidelity product `f_b f_{U b U}` per two layers,
    /// fitted over the depths, with its standard error.
    pub products: Vec<(PauliString, f64, f64)>,
    /// Generators `b` with `U b U ≠ b`, whose fidelity is assumed equal to
    /// that of `U b U`.
    pub symmetrised: Vec<PauliString>,
}

/// Options of [`learn_layer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningOptions {
    /// Numbers of layers (even, at least two of them).
    pub depths: Vec<usize>,
    /// Random Pauli twirls per basis and depth.
    pub twirls: usize,
    /// Shots per twirled circuit.
    pub shots: usize,
    /// Pairs of coupled qubits for two-qubit generators; by default the
    /// pairs of the layer's gates and of neighbouring qubits.
    pub edges: Option<Vec<(usize, usize)>>,
    pub seed: u64,
}

impl Default for LearningOptions {
    fn default() -> Self {
        LearningOptions {
            depths: vec![2, 4, 8, 16, 32],
            twirls: 16,
            shots: 256,
            edges: None,
            seed: 0,
        }
    }
}

/// A two-qubit Clifford gate of a layer.
#[derive(Clone, Copy, Debug)]
enum Gate {
    Cx(usize, usize),
    Cy(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
}

/// The gates of a layer: `cx`, `cy`, `cz` and `swap` on disjoint qubits
/// (so the layer is its own inverse).
fn layer_gates(layer: &Circuit) -> Result<Vec<Gate>> {
    let n = layer.n_qubits();
    if n == 0 || n > MAX_LEARNING_QUBITS {
        return Err(QsoError::TooManyQubits {
            what: "noise learning (1 or more)",
            max: MAX_LEARNING_QUBITS,
            requested: n,
        });
    }
    let mut used = 0u64;
    let mut gates = Vec::new();
    for inst in layer.instructions() {
        let gate = match (inst.kind, inst.controls.as_slice(), inst.targets.as_slice()) {
            (GateKind::X, [c], [t]) => Gate::Cx(*c, *t),
            (GateKind::Y, [c], [t]) => Gate::Cy(*c, *t),
            (GateKind::Z, [c], [t]) => Gate::Cz(*c, *t),
            (GateKind::Swap, [], [a, b]) => Gate::Swap(*a, *b),
            _ => {
                return Err(QsoError::invalid(
                    "layer",
                    format!(
                        "a layer holds cx, cy, cz and swap gates only, not {}",
                        inst.kind.name()
                    ),
                ));
            }
        };
        let (a, b) = pair(gate);
        let mask = (1u64 << a) | (1u64 << b);
        if used & mask != 0 {
            return Err(QsoError::invalid(
                "layer",
                "the gates of a layer act on disjoint qubits",
            ));
        }
        used |= mask;
        gates.push(gate);
    }
    Ok(gates)
}

fn pair(g: Gate) -> (usize, usize) {
    match g {
        Gate::Cx(a, b) | Gate::Cy(a, b) | Gate::Cz(a, b) | Gate::Swap(a, b) => (a, b),
    }
}

/// `U P U†` for the layer (up to sign).
fn conjugate(gates: &[Gate], p: PauliString) -> PauliString {
    let (mut x, mut z) = (p.x_mask(), p.z_mask());
    for &g in gates {
        let (a, b) = pair(g);
        let bit = |m: u64, q: usize| (m >> q) & 1 == 1;
        let (xa, za, xb, zb) = (bit(x, a), bit(z, a), bit(x, b), bit(z, b));
        let cx = |[xa, za, xb, zb]: [bool; 4]| [xa, za ^ zb, xb ^ xa, zb];
        let out = match g {
            Gate::Cx(..) => cx([xa, za, xb, zb]),
            Gate::Cz(..) => [xa, za ^ xb, xb, zb ^ xa],
            Gate::Cy(..) => {
                // CY = S_b · CX · S_b†; S conjugation maps z_b ↦ z_b ⊕ x_b.
                let [xa, za, xb, zb] = cx([xa, za, xb, zb ^ xb]);
                [xa, za, xb, zb ^ xb]
            }
            Gate::Swap(..) => [xb, zb, xa, za],
        };
        let set = |m: u64, q: usize, v: bool| (m & !(1 << q)) | (u64::from(v) << q);
        x = set(set(x, a, out[0]), b, out[2]);
        z = set(set(z, a, out[1]), b, out[3]);
    }
    PauliString::from_masks(x, z)
}

/// Single-qubit Pauli gate `(x, z)` on `q` (nothing for the identity).
fn push_pauli(c: &mut Circuit, q: usize, x: bool, z: bool) {
    match (x, z) {
        (true, false) => c.x(q),
        (true, true) => c.y(q),
        (false, true) => c.z(q),
        (false, false) => c,
    };
}

/// The Pauli string `p` as gates.
fn push_string(c: &mut Circuit, p: PauliString) {
    for q in 0..c.n_qubits() {
        push_pauli(c, q, (p.x_mask() >> q) & 1 == 1, (p.z_mask() >> q) & 1 == 1);
    }
}

/// The layer with a random Pauli frame on every qubit.
fn push_twirled_layer(c: &mut Circuit, layer: &Circuit, gates: &[Gate], rng: &mut Rng) {
    let n = c.n_qubits() as u32;
    let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let frame = PauliString::from_masks(rng.next_u64() & mask, rng.next_u64() & mask);
    push_string(c, frame);
    for inst in layer.instructions() {
        c.try_push(inst.clone()).expect("layer gates fit");
    }
    push_string(c, conjugate(gates, frame));
}

/// Generators: every one-qubit Pauli, and every two-qubit Pauli on `edges`.
fn generators(n: usize, edges: &[(usize, usize)]) -> Vec<PauliString> {
    let letters = [Pauli::X, Pauli::Y, Pauli::Z];
    let mut out = Vec::new();
    for q in 0..n {
        for p in letters {
            out.push(PauliString::single(q, p));
        }
    }
    for &(a, b) in edges {
        for pa in letters {
            for pb in letters {
                out.push(PauliString::new(&[(a, pa), (b, pb)]).expect("distinct qubits"));
            }
        }
    }
    out
}

/// Measurement bases (a letter per qubit) in which every generator is
/// measurable: an orthogonal array of strength two over a colouring of the
/// coupling graph, so that every pair of coupled qubits sees all nine
/// letter pairs. The rows are the vectors `r` of GF(3)ᵏ and colour `c`
/// gets letter `r·p_c` for distinct projective points `p_c`: 3 bases for
/// one colour, 9 for up to 4 colours and 27 for up to 13.
fn bases(n: usize, edges: &[(usize, usize)]) -> Result<Vec<Vec<Pauli>>> {
    let colour = colouring(n, edges);
    let colours = colour.iter().max().map_or(1, |c| c + 1);
    let (k, points): (u32, &[[usize; 3]]) = match colours {
        1 => (1, &[[1, 0, 0]]),
        2..=4 => (2, &[[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0]]),
        5..=13 => (
            3,
            &[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 2, 0],
                [1, 0, 1],
                [1, 0, 2],
                [0, 1, 1],
                [0, 1, 2],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
            ],
        ),
        _ => {
            return Err(QsoError::invalid(
                "edges",
                "the coupling graph needs more than 13 colours",
            ));
        }
    };
    let letters = [Pauli::X, Pauli::Y, Pauli::Z];
    Ok((0..3usize.pow(k))
        .map(|r| {
            let v = [r % 3, (r / 3) % 3, r / 9];
            (0..n)
                .map(|q| {
                    let p = points[colour[q]];
                    letters[(v[0] * p[0] + v[1] * p[1] + v[2] * p[2]) % 3]
                })
                .collect()
        })
        .collect())
}

/// A proper colouring of the coupling graph (DSatur: the next node is the
/// one with the most distinct colours among its neighbours).
fn colouring(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut neighbours = vec![Vec::new(); n];
    for &(a, b) in edges {
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    let mut colour = vec![usize::MAX; n];
    for _ in 0..n {
        let used = |q: usize| {
            let mut c: Vec<usize> = neighbours[q]
                .iter()
                .map(|&v| colour[v])
                .filter(|&c| c != usize::MAX)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let q = (0..n)
            .filter(|&q| colour[q] == usize::MAX)
            .max_by_key(|&q| (used(q).len(), neighbours[q].len(), usize::MAX - q))
            .expect("an uncoloured node");
        let taken = used(q);
        colour[q] = (0..).find(|c| !taken.contains(c)).expect("a free colour");
    }
    colour
}

/// Whether Pauli `b` is diagonal in the product basis `basis`.
fn measurable(b: &PauliString, basis: &[Pauli]) -> bool {
    (0..basis.len()).all(|q| {
        let p = b.get(q);
        p == Pauli::I || p == basis[q]
    })
}

/// Median (sorts `v`).
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len();
    if m == 0 {
        return f64::NAN;
    }
    if m % 2 == 1 {
        v[m / 2]
    } else {
        (v[m / 2 - 1] + v[m / 2]) / 2.0
    }
}

/// Non-negative least squares `argmin_{x ≥ 0} ‖A x − b‖` (Lawson and
/// Hanson); `a` holds the rows.
pub(crate) fn nnls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let k = a.first().map_or(0, Vec::len);
    let mat = nalgebra::DMatrix::from_fn(m, k, |i, j| a[i][j]);
    let rhs = nalgebra::DVector::from_column_slice(b);
    let mut x = vec![0.0; k];
    let mut passive = vec![false; k];
    // Thresholds relative to the scale of the gradient Aᵀb and of the
    // solution, so that rescaling rows leaves the result unchanged.
    let tol = 1e-12 * (mat.transpose() * &rhs).amax();
    let gradient = |x: &[f64]| {
        let r = &rhs - &mat * nalgebra::DVector::from_column_slice(x);
        mat.transpose() * r
    };
    // Least squares on the passive columns (others zero).
    let solve = |passive: &[bool]| -> Vec<f64> {
        let cols: Vec<usize> = (0..k).filter(|&j| passive[j]).collect();
        let mut s = vec![0.0; k];
        if cols.is_empty() {
            return s;
        }
        let sub = nalgebra::DMatrix::from_fn(m, cols.len(), |i, c| mat[(i, cols[c])]);
        let svd = sub.svd(true, true);
        let eps = 1e-13 * svd.singular_values.max();
        let sol = svd.solve(&rhs, eps).expect("both factors computed");
        for (c, &j) in cols.iter().enumerate() {
            s[j] = sol[c];
        }
        s
    };
    // Variables that came out non-positive right after entering; skipped
    // until the solution changes (prevents cycling on round-off).
    let mut skip = vec![false; k];
    for _ in 0..3 * k + 10 {
        let w = gradient(&x);
        let Some(j) = (0..k)
            .filter(|&j| !passive[j] && !skip[j] && w[j] > tol)
            .max_by(|&p, &q| w[p].total_cmp(&w[q]))
        else {
            break;
        };
        passive[j] = true;
        let mut first = true;
        loop {
            let s = solve(&passive);
            let bad: Vec<usize> = (0..k).filter(|&i| passive[i] && s[i] <= 0.0).collect();
            if bad.is_empty() {
                x = s;
                skip.iter_mut().for_each(|f| *f = false);
                break;
            }
            if first && s[j] <= 0.0 {
                passive[j] = false;
                skip[j] = true;
                break;
            }
            first = false;
            let (blocking, alpha) = bad.iter().map(|&i| (i, x[i] / (x[i] - s[i]))).fold(
                (bad[0], f64::INFINITY),
                |m, c| if c.1 < m.1 { c } else { m },
            );
            let scale = (0..k)
                .filter(|&i| passive[i])
                .map(|i| x[i].abs().max(s[i].abs()))
                .fold(0.0, f64::max);
            for i in 0..k {
                x[i] += alpha * (s[i] - x[i]);
                if passive[i] && (i == blocking || x[i] <= 1e-12 * scale) {
                    passive[i] = false;
                    x[i] = 0.0;
                }
            }
        }
    }
    x
}

/// Learn the noise of the twirled `layer` (two-qubit Clifford gates on
/// disjoint qubits) on `device` as a sparse Pauli–Lindblad model, from
/// simulated experiments with finite shots and the device's readout
/// errors (twirled: random bit flips before measurement, undone in the
/// data).
pub fn learn_layer(
    layer: &Circuit,
    device: &DeviceModel,
    options: &LearningOptions,
) -> Result<LearnedNoise> {
    let gates = layer_gates(layer)?;
    let n = layer.n_qubits();
    device.validate()?;
    let mut distinct = options.depths.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() < 2
        || distinct.len() != options.depths.len()
        || options
            .depths
            .iter()
            .any(|&d| d == 0 || d % 2 == 1 || d > 4096)
    {
        return Err(QsoError::invalid(
            "depths",
            "at least two distinct depths, each even and in 2..=4096",
        ));
    }
    if options.twirls < 2 {
        return Err(QsoError::invalid(
            "twirls",
            "at least two twirls (the standard errors come from their spread)",
        ));
    }
    if options.shots == 0 || options.shots > MAX_LEARNING_SHOTS {
        return Err(QsoError::invalid(
            "shots",
            format!("shots must lie in 1..={MAX_LEARNING_SHOTS}"),
        ));
    }
    let edges: Vec<(usize, usize)> = match &options.edges {
        Some(e) => {
            if e.iter().any(|&(a, b)| a == b || a >= n || b >= n) {
                return Err(QsoError::invalid(
                    "edges",
                    format!("edges join two different qubits below {n}"),
                ));
            }
            let mut e: Vec<(usize, usize)> = e.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
            e.sort_unstable();
            e.dedup();
            e
        }
        None => {
            let mut e: Vec<(usize, usize)> = (1..n).map(|q| (q - 1, q)).collect();
            for &g in &gates {
                let (a, b) = pair(g);
                e.push((a.min(b), a.max(b)));
            }
            e.sort_unstable();
            e.dedup();
            e
        }
    };
    let gens = generators(n, &edges);
    let bases = bases(n, &edges)?;
    let work = (bases.len() * options.depths.len()).saturating_mul(options.twirls);
    let layers = (bases.len() * distinct.iter().sum::<usize>()).saturating_mul(options.twirls);
    if work > MAX_LEARNING_CIRCUITS || layers > MAX_LEARNING_LAYERS {
        return Err(QsoError::invalid(
            "twirls",
            format!(
                "the experiment needs more than {MAX_LEARNING_CIRCUITS} circuits or \
                 {MAX_LEARNING_LAYERS} layers"
            ),
        ));
    }
    let readout = device.readout_errors(n);
    let base = Rng::seed_from_u64(options.seed);
    // One task per (basis, depth, twirl): the estimate of every
    // measurable generator.
    let results: Vec<Result<Vec<(usize, f64)>>> = map_tasks(work, |t| {
        let (bi, rest) = (
            t / (options.depths.len() * options.twirls),
            t % (options.depths.len() * options.twirls),
        );
        let di = rest / options.twirls;
        let basis = &bases[bi];
        let mut rng = base.fork(t as u64);
        let mut c = Circuit::new(n);
        for (q, &p) in basis.iter().enumerate() {
            match p {
                Pauli::X => {
                    c.h(q);
                }
                Pauli::Y => {
                    c.h(q).s(q);
                }
                _ => {}
            }
        }
        for _ in 0..options.depths[di] {
            push_twirled_layer(&mut c, layer, &gates, &mut rng);
        }
        for (q, &p) in basis.iter().enumerate() {
            match p {
                Pauli::X => {
                    c.h(q);
                }
                Pauli::Y => {
                    c.sdg(q).h(q);
                }
                _ => {}
            }
        }
        // Readout twirl: flipping random qubits before measurement and
        // their bits after it makes readout errors a factor per qubit.
        let flip = rng.next_u64() & ((1u64 << n) - 1);
        for q in (0..n).filter(|q| (flip >> q) & 1 == 1) {
            c.x(q);
        }
        let mut rho = DensityMatrix::zero(n)?;
        rho.apply_device(&c, &[], device)?;
        let counts = rho.sample_with_readout(options.shots, &readout, &mut rng);
        Ok(gens
            .iter()
            .enumerate()
            .filter(|(_, g)| measurable(g, basis))
            .map(|(k, g)| {
                let support = g.x_mask() | g.z_mask();
                let total: i64 = counts
                    .iter()
                    .map(|(&bits, &m)| {
                        let parity = ((bits as u64 ^ flip) & support).count_ones() % 2;
                        if parity == 0 { m as i64 } else { -(m as i64) }
                    })
                    .sum();
                (k, total as f64 / options.shots as f64)
            })
            .collect())
    });
    let results = results.into_iter().collect::<Result<Vec<_>>>()?;
    let n_depths = options.depths.len();
    // Mean of every generator at every depth over the twirls `pick(cell)`
    // of each (basis, depth) cell.
    let means = |pick: &mut dyn FnMut(usize) -> Vec<usize>| {
        let mut sum = vec![vec![0.0; n_depths]; gens.len()];
        let mut count = vec![vec![0usize; n_depths]; gens.len()];
        for cell in 0..bases.len() * n_depths {
            let di = cell % n_depths;
            for w in pick(cell) {
                for &(k, v) in &results[cell * options.twirls + w] {
                    sum[k][di] += v;
                    count[k][di] += 1;
                }
            }
        }
        (sum, count)
    };
    let (sum, count) = means(&mut |_| (0..options.twirls).collect());
    let mut sq = vec![vec![0.0; n_depths]; gens.len()];
    for (cell, chunk) in results.chunks(options.twirls).enumerate() {
        let di = cell % n_depths;
        for r in chunk {
            for &(k, v) in r {
                let mean = sum[k][di] / count[k][di] as f64;
                sq[k][di] += (v - mean) * (v - mean);
            }
        }
    }
    // Per generator: the depths that enter its fit (mean above three
    // standard errors) and their coefficients in the least-squares slope
    // of ln⟨b⟩ against d/2. The fit is unweighted: weights estimated from
    // a few twirls bias it when coherent errors spread the twirls.
    let mut coefficients: Vec<Vec<(usize, f64)>> = Vec::with_capacity(gens.len());
    for (k, g) in gens.iter().enumerate() {
        let points: Vec<(usize, f64)> = (0..n_depths)
            .filter(|&di| {
                let m = count[k][di];
                let mean = sum[k][di] / m as f64;
                let var = if m > 1 {
                    sq[k][di] / (m - 1) as f64
                } else {
                    0.0
                };
                let se = (var / m as f64)
                    .sqrt()
                    .max(0.5 / (m as f64 * options.shots as f64));
                mean > 3.0 * se
            })
            .map(|di| (di, options.depths[di] as f64 / 2.0))
            .collect();
        let xbar = points.iter().map(|p| p.1).sum::<f64>() / points.len() as f64;
        let sxx: f64 = points.iter().map(|p| (p.1 - xbar).powi(2)).sum();
        if points.len() < 2 || sxx.is_nan() || sxx <= 0.0 {
            return Err(QsoError::invalid(
                "depths",
                format!("the fidelity of {g} decays too fast to fit at these depths"),
            ));
        }
        coefficients.push(points.iter().map(|p| (p.0, (p.1 - xbar) / sxx)).collect());
    }
    // f_b = sqrt(product) with ln(product) the slope: −ln f_b / 2 is
    // −slope / 4.
    let halved_logs = |sum: &[Vec<f64>], count: &[Vec<usize>]| -> Vec<f64> {
        coefficients
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let slope: f64 = c
                    .iter()
                    .map(|&(di, ci)| {
                        let mean = sum[k][di] / count[k][di] as f64;
                        ci * mean.max(f64::MIN_POSITIVE).ln()
                    })
                    .sum();
                -slope / 4.0
            })
            .collect()
    };
    let y = halved_logs(&sum, &count);
    // Bootstrap over the twirls of every cell, which keeps the
    // correlations between generators measured on the same circuits.
    let mut rng = base.fork(u64::MAX);
    let replicas: Vec<Vec<f64>> = (0..BOOTSTRAP)
        .map(|_| {
            let (sum, count) = means(&mut |_| {
                (0..options.twirls)
                    .map(|_| rng.below(options.twirls as u64) as usize)
                    .collect()
            });
            halved_logs(&sum, &count)
        })
        .collect();
    // Robust spread (1.4826 × median absolute deviation, the standard
    // deviation for normal data): a replica whose mean at some depth falls
    // to zero or below has an absurd logarithm, and must not dominate.
    let spread = |values: &[Vec<f64>], k: usize| {
        let mut v: Vec<f64> = values.iter().map(|r| r[k]).collect();
        let centre = median(&mut v);
        let mut dev: Vec<f64> = v.iter().map(|x| (x - centre).abs()).collect();
        1.4826 * median(&mut dev)
    };
    // Floor: an exactly reproducible fidelity (no noise at all, as on an
    // idle qubit without readout errors) gets a weight at most 100 times
    // the typical one, which keeps the fit well conditioned.
    let mut sigma: Vec<f64> = (0..gens.len()).map(|k| spread(&replicas, k)).collect();
    let mut positive: Vec<f64> = sigma.iter().copied().filter(|s| *s > 0.0).collect();
    let floor = if positive.is_empty() {
        1.0
    } else {
        1e-2 * median(&mut positive)
    };
    sigma.iter_mut().for_each(|s| *s = s.max(floor));
    let products = gens
        .iter()
        .zip(&y)
        .zip(&sigma)
        .map(|((g, v), s)| {
            let product = (-4.0 * v).exp();
            (*g, product, 4.0 * s * product)
        })
        .collect();
    let rows: Vec<Vec<f64>> = gens
        .iter()
        .zip(&sigma)
        .map(|(b, s)| {
            gens.iter()
                .map(|g| if g.commutes_with(b) { 0.0 } else { 1.0 / s })
                .collect()
        })
        .collect();
    let fit = |y: &[f64]| {
        let rhs: Vec<f64> = y.iter().zip(&sigma).map(|(v, s)| v / s).collect();
        nnls(&rows, &rhs)
    };
    let rates = fit(&y);
    // Standard errors of the unconstrained least-squares rates: where the
    // constraint λ ≥ 0 is active, every replica of the constrained fit can
    // sit at zero, and their spread says nothing about the error. The
    // constrained fit is the projection of the unconstrained one onto
    // λ ≥ 0 in the metric of the fit, which only brings it closer to the
    // (non-negative) true rates in that metric.
    let a = nalgebra::DMatrix::from_fn(gens.len(), gens.len(), |i, j| rows[i][j]);
    let tolerance = 1e-12 * a.norm();
    let pinv = a
        .pseudo_inverse(tolerance)
        .expect("the tolerance is non-negative");
    let unconstrained: Vec<Vec<f64>> = replicas
        .iter()
        .map(|yb| {
            let rhs = nalgebra::DVector::from_iterator(
                gens.len(),
                yb.iter().zip(&sigma).map(|(v, s)| v / s),
            );
            (&pinv * rhs).iter().copied().collect()
        })
        .collect();
    let std_errors = (0..gens.len()).map(|k| spread(&unconstrained, k)).collect();
    let symmetrised = gens
        .iter()
        .filter(|g| conjugate(&gates, **g) != **g)
        .copied()
        .collect();
    Ok(LearnedNoise {
        model: SparsePauliLindblad::new(n, gens, rates)?,
        std_errors,
        products,
        symmetrised,
    })
}

/// Checks shared by [`pec_layered`] and [`pea_layered`].
fn check_layered(
    blocks: &[Circuit],
    layer: &Circuit,
    model: &SparsePauliLindblad,
    observable: &PauliSum,
    samples: usize,
) -> Result<Vec<Gate>> {
    let gates = layer_gates(layer)?;
    model.validate()?;
    let n = layer.n_qubits();
    if blocks.is_empty() || blocks.iter().any(|b| b.n_qubits() != n) || model.n_qubits != n {
        return Err(QsoError::invalid(
            "blocks",
            format!("at least one block, every block and the model on the layer's {n} qubits"),
        ));
    }
    if blocks.iter().any(|b| b.num_parameters() > 0) {
        return Err(QsoError::invalid(
            "blocks",
            "blocks must not have free parameters",
        ));
    }
    if observable.min_qubits() > n {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: n,
        });
    }
    if samples == 0 || samples.saturating_mul(blocks.len()) > MAX_LEARNING_CIRCUITS {
        return Err(QsoError::invalid(
            "samples",
            format!("samples × blocks must lie in 1..={MAX_LEARNING_CIRCUITS}"),
        ));
    }
    Ok(gates)
}

/// `⟨O⟩` of one sampled circuit `B₀ L B₁ L … B_m`: every layer twirled and
/// followed by the Paulis `insert(rng)` returns.
fn run_sample(
    blocks: &[Circuit],
    layer: &Circuit,
    gates: &[Gate],
    device: &DeviceModel,
    observable: &PauliSum,
    rng: &mut Rng,
    mut insert: impl FnMut(&mut Rng) -> PauliString,
) -> Result<f64> {
    let n = layer.n_qubits();
    let mut c = Circuit::new(n);
    for (i, block) in blocks.iter().enumerate() {
        if i > 0 {
            push_twirled_layer(&mut c, layer, gates, rng);
            let p = insert(rng);
            push_string(&mut c, p);
        }
        for inst in block.instructions() {
            c.try_push(inst.clone()).expect("block gates fit");
        }
    }
    let mut rho = DensityMatrix::zero(n)?;
    rho.apply_device(&c, &[], device)?;
    rho.expectation(observable)
}

/// Error-cancelled `⟨O⟩` of `B₀ L B₁ L … L B_m` (`blocks = [B₀, …, B_m]`,
/// `L` the twirled `layer`) on `device`, cancelling the noise of every
/// layer with the learned `model`. Each sample draws, after every layer
/// and for every generator, `P_k` with probability `(1 − e^{−2λ_k})/2`
/// (flipping the sign); the result is unbiased when the model is the
/// layer's noise. Blocks keep the device's noise.
pub fn pec_layered(
    blocks: &[Circuit],
    layer: &Circuit,
    model: &SparsePauliLindblad,
    device: &DeviceModel,
    observable: &PauliSum,
    samples: usize,
    seed: u64,
) -> Result<PecResult> {
    let gates = check_layered(blocks, layer, model, observable, samples)?;
    device.validate()?;
    let layers = blocks.len() - 1;
    let gamma = model.overhead().powi(layers as i32);
    if !gamma.is_finite() {
        return Err(QsoError::invalid(
            "model",
            "the sampling overhead exp(2·Σλ·layers) overflows",
        ));
    }
    let flips: Vec<f64> = model
        .rates
        .iter()
        .map(|r| (1.0 - (-2.0 * r).exp()) / 2.0)
        .collect();
    let base = Rng::seed_from_u64(seed);
    let values: Vec<Result<f64>> = map_tasks(samples, |s| {
        let mut rng = base.fork(s as u64);
        let mut sign = 1.0;
        let v = run_sample(blocks, layer, &gates, device, observable, &mut rng, |rng| {
            let mut p = PauliString::identity();
            for (g, &f) in model.generators.iter().zip(&flips) {
                if rng.uniform() < f {
                    sign = -sign;
                    p = PauliString::from_masks(p.x_mask() ^ g.x_mask(), p.z_mask() ^ g.z_mask());
                }
            }
            p
        })?;
        Ok(sign * gamma * v)
    });
    let values = values.into_iter().collect::<Result<Vec<f64>>>()?;
    Ok(PecResult {
        estimate: Estimate::from_samples(&values),
        overhead: gamma,
        noisy_operations: layers,
    })
}

/// Zero-noise extrapolation of `⟨O⟩` of `B₀ L B₁ L … L B_m` by
/// probabilistic error amplification: at scale `G ≥ 1`, after every
/// twirled layer, `P_k` is inserted with probability
/// `(1 − e^{−2(G−1)λ_k})/2`, which multiplies every rate of the model by
/// `G`. The values at the scales are extrapolated to zero noise.
#[allow(clippy::too_many_arguments)]
pub fn pea_layered(
    blocks: &[Circuit],
    layer: &Circuit,
    model: &SparsePauliLindblad,
    device: &DeviceModel,
    observable: &PauliSum,
    scales: &[f64],
    samples: usize,
    method: Extrapolation,
    seed: u64,
) -> Result<ZneResult> {
    let gates = check_layered(blocks, layer, model, observable, samples)?;
    device.validate()?;
    if scales.len() < 2 || scales.iter().any(|g| !(g.is_finite() && *g >= 1.0)) {
        return Err(QsoError::invalid(
            "scales",
            "at least two scales, each finite and at least 1",
        ));
    }
    let mut values = Vec::with_capacity(scales.len());
    for (i, &g) in scales.iter().enumerate() {
        let flips: Vec<f64> = model
            .rates
            .iter()
            .map(|r| (1.0 - (-2.0 * (g - 1.0) * r).exp()) / 2.0)
            .collect();
        let base = Rng::seed_from_u64(seed).fork(i as u64);
        let v: Vec<Result<f64>> = map_tasks(samples, |s| {
            let mut rng = base.fork(s as u64);
            run_sample(blocks, layer, &gates, device, observable, &mut rng, |rng| {
                let mut p = PauliString::identity();
                for (g, &f) in model.generators.iter().zip(&flips) {
                    if rng.uniform() < f {
                        p = PauliString::from_masks(
                            p.x_mask() ^ g.x_mask(),
                            p.z_mask() ^ g.z_mask(),
                        );
                    }
                }
                p
            })
        });
        let v = v.into_iter().collect::<Result<Vec<f64>>>()?;
        values.push(Estimate::from_samples(&v));
    }
    Ok(ZneResult {
        estimate: extrapolate(scales, &values, method)?,
        scales: scales.to_vec(),
        values,
    })
}

#[cfg(test)]
mod tests;
