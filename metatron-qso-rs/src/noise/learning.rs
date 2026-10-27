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
//! lists those Paulis), and the rates follow by non-negative least squares,
//! with standard errors from a parametric bootstrap.
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
        if n_qubits == 0 || n_qubits > 64 {
            return Err(QsoError::invalid(
                "n_qubits",
                "a model acts on 1 to 64 qubits",
            ));
        }
        if generators.len() != rates.len() {
            return Err(QsoError::DimensionMismatch {
                expected: generators.len(),
                actual: rates.len(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        for g in &generators {
            if g.is_identity() || g.min_qubits() > n_qubits || !seen.insert(*g) {
                return Err(QsoError::invalid(
                    "generators",
                    format!("{g} is the identity, outside {n_qubits} qubits or repeated"),
                ));
            }
        }
        if rates.iter().any(|r| !(r.is_finite() && *r >= 0.0)) {
            return Err(QsoError::invalid(
                "rates",
                "rates must be finite and non-negative",
            ));
        }
        Ok(Self {
            n_qubits,
            generators,
            rates,
        })
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
    /// Bootstrap standard error of each rate.
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
/// measurable: an orthogonal array over a colouring of the coupling graph,
/// so that every pair of coupled qubits sees all nine letter pairs.
fn bases(n: usize, edges: &[(usize, usize)]) -> Result<Vec<Vec<Pauli>>> {
    let mut colour = vec![usize::MAX; n];
    for q in 0..n {
        let taken: Vec<usize> = edges
            .iter()
            .filter_map(|&(a, b)| match (a == q, b == q) {
                (true, _) => Some(colour[b]),
                (_, true) => Some(colour[a]),
                _ => None,
            })
            .collect();
        colour[q] = (0..).find(|c| !taken.contains(c)).expect("a free colour");
    }
    let colours = colour.iter().max().map_or(1, |c| c + 1);
    if colours > 5 {
        return Err(QsoError::invalid(
            "edges",
            "the coupling graph needs more than five colours",
        ));
    }
    let letters = [Pauli::X, Pauli::Y, Pauli::Z];
    // Up to three colours: the nine rows (a, b, a + b mod 3); more: every
    // assignment of letters to colours.
    let rows: Vec<Vec<usize>> = if colours <= 3 {
        (0..9)
            .map(|r| vec![r / 3, r % 3, (r / 3 + r % 3) % 3])
            .collect()
    } else {
        (0..3usize.pow(colours as u32))
            .map(|r| {
                (0..colours)
                    .map(|c| (r / 3usize.pow(c as u32)) % 3)
                    .collect()
            })
            .collect()
    };
    Ok(rows
        .iter()
        .map(|row| (0..n).map(|q| letters[row[colour[q]]]).collect())
        .collect())
}

/// Whether Pauli `b` is diagonal in the product basis `basis`.
fn measurable(b: &PauliString, basis: &[Pauli]) -> bool {
    (0..basis.len()).all(|q| {
        let p = b.get(q);
        p == Pauli::I || p == basis[q]
    })
}

/// Weighted least squares of `y` against `x`: slope and its standard
/// error.
fn weighted_slope(points: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, s) in points {
        let w = 1.0 / (s * s);
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let det = sw * sxx - sx * sx;
    (points.len() >= 2 && det > 0.0).then(|| ((sw * sxy - sx * sy) / det, (sw / det).sqrt()))
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
    let tol = 1e-12 * (1.0 + rhs.norm()) * (1.0 + mat.norm());
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
        let sol = sub
            .svd(true, true)
            .solve(&rhs, 1e-14)
            .expect("both factors computed");
        for (c, &j) in cols.iter().enumerate() {
            s[j] = sol[c];
        }
        s
    };
    for _ in 0..3 * k + 10 {
        let w = gradient(&x);
        let Some(j) = (0..k)
            .filter(|&j| !passive[j] && w[j] > tol)
            .max_by(|&p, &q| w[p].total_cmp(&w[q]))
        else {
            break;
        };
        passive[j] = true;
        loop {
            let s = solve(&passive);
            let bad: Vec<usize> = (0..k).filter(|&i| passive[i] && s[i] <= 0.0).collect();
            if bad.is_empty() {
                x = s;
                break;
            }
            let alpha = bad
                .iter()
                .map(|&i| x[i] / (x[i] - s[i]))
                .fold(f64::INFINITY, f64::min);
            for i in 0..k {
                x[i] += alpha * (s[i] - x[i]);
                if passive[i] && x[i] <= tol {
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
    if options.depths.len() < 2
        || options
            .depths
            .iter()
            .any(|&d| d == 0 || d % 2 == 1 || d > 4096)
    {
        return Err(QsoError::invalid(
            "depths",
            "at least two depths, each even and in 2..=4096",
        ));
    }
    if options.twirls == 0 || options.shots == 0 {
        return Err(QsoError::invalid(
            "twirls",
            "twirls and shots must be positive",
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
    let work = bases.len() * options.depths.len() * options.twirls;
    if work > MAX_LEARNING_CIRCUITS {
        return Err(QsoError::invalid(
            "twirls",
            format!("the experiment needs more than {MAX_LEARNING_CIRCUITS} circuits"),
        ));
    }
    let readout = device.readout_errors(n);
    let base = Rng::seed_from_u64(options.seed);
    // One task per (basis, depth, twirl): the estimate of every
    // measurable generator.
    let results: Vec<Result<Vec<(usize, usize, f64)>>> = map_tasks(work, |t| {
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
                (k, di, total as f64 / options.shots as f64)
            })
            .collect())
    });
    // Estimates per (generator, depth).
    let mut samples: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); options.depths.len()]; gens.len()];
    for r in results {
        for (k, di, e) in r? {
            samples[k][di].push(e);
        }
    }
    let mut products = Vec::with_capacity(gens.len());
    let mut y = Vec::with_capacity(gens.len());
    let mut sigma = Vec::with_capacity(gens.len());
    for (k, g) in gens.iter().enumerate() {
        let points: Vec<(f64, f64, f64)> = options
            .depths
            .iter()
            .zip(&samples[k])
            .filter_map(|(&d, s)| {
                let e = Estimate::from_samples(s);
                let se = e.std_error.max(0.5 / (s.len() * options.shots) as f64);
                (e.value > 3.0 * se).then(|| (d as f64 / 2.0, e.value.ln(), se / e.value))
            })
            .collect();
        let Some((slope, se)) = weighted_slope(&points) else {
            return Err(QsoError::invalid(
                "depths",
                format!("the fidelity of {g} decays too fast to fit at these depths"),
            ));
        };
        products.push((*g, slope.exp(), slope.exp() * se));
        // f_b = sqrt(product): −ln f_b / 2 = −slope / 4.
        y.push(-slope / 4.0);
        sigma.push(se / 4.0);
    }
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
    let mut rng = base.fork(u64::MAX);
    let replicas: Vec<Vec<f64>> = (0..BOOTSTRAP)
        .map(|_| {
            let yb: Vec<f64> = y
                .iter()
                .zip(&sigma)
                .map(|(v, s)| v + s * rng.normal())
                .collect();
            fit(&yb)
        })
        .collect();
    let std_errors = (0..gens.len())
        .map(|k| {
            Estimate::from_samples(&replicas.iter().map(|r| r[k]).collect::<Vec<_>>()).std_error
                * (BOOTSTRAP as f64).sqrt()
        })
        .collect();
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
