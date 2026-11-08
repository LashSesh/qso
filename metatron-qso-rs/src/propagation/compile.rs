//! Compiling layers to ops: Clifford tables by the tableau's rules,
//! rotation tables, transfer matrices of the other gates and of channels
//! (see [`transfer`](super::transfer)) and the diagonal maps of
//! Pauli–Lindblad layers.

use std::f64::consts::FRAC_PI_4;
use std::sync::Arc;

use super::transfer::{GateKey, LocalMap, TableCache, gate_table};
use super::{Layer, MAX_TRANSFER_QUBITS, PauliOperator};
use crate::error::{QsoError, Result};
use crate::quantum::gates::instruction_matrix;
use crate::quantum::{GateKind, Instruction, Param};
use crate::stabilizer::{CliffordTarget, apply_clifford, rules};

/// One step of the propagation, on register qubits.
#[derive(Clone, Debug)]
pub(crate) enum Op {
    /// A gate or channel on `qubits` (the op's `j`-th qubit is bit `j` of
    /// the local index of its table).
    Local {
        qubits: Vec<usize>,
        map: Arc<LocalMap>,
    },
    /// A Pauli–Lindblad layer: a factor per string.
    Diag(Arc<DiagMap>),
}

/// The diagonal map of a Pauli–Lindblad layer: string `P` times
/// `exp(−2α Σ_{k: {P_k, P} = 0} λ_k)`.
///
/// Generators are stored sparsely, as their non-identity factors, so a
/// layer of low-weight generators on a wide register costs memory by its
/// factors, not by `2·⌈n/64⌉` words per generator.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiagMap {
    /// The factors `(qubit, letter)` of every generator, ascending in the
    /// qubit, with the letter `x | z << 1` (X, Z, Y as 1, 2, 3).
    factors: Vec<(u32, u8)>,
    /// The factors of generator `i` are `factors[start[i]..start[i + 1]]`.
    start: Vec<u32>,
    /// `λ_k > 0`, in generator order.
    rates: Vec<f64>,
    /// `α`.
    factor: f64,
}

impl DiagMap {
    /// The map of `rates` (checked: no identity term, rates positive) at
    /// factor `α`, in the canonical order of `rates`.
    pub(crate) fn new(rates: &PauliOperator, factor: f64) -> Self {
        let (keys, coefs) = rates.rows();
        let w = rates.words();
        let mut map = DiagMap {
            factors: Vec::new(),
            start: Vec::with_capacity(coefs.len() + 1),
            rates: coefs.to_vec(),
            factor,
        };
        map.start.push(0);
        for key in keys.chunks_exact(2 * w) {
            for q in mask_qubits(&(0..w).map(|j| key[j] | key[w + j]).collect::<Vec<u64>>()) {
                map.factors.push((q as u32, letter_at(key, w, q)));
            }
            map.start.push(map.factors.len() as u32);
        }
        map
    }

    /// The map of the generators given by their factors (ascending in the
    /// qubit, letters 1–3), equal generators merged by adding their rates
    /// in the order given (commuting Pauli channels compose so).
    fn from_generators(generators: Vec<Vec<(u32, u8)>>, rates: Vec<f64>, factor: f64) -> Self {
        let mut order: Vec<usize> = (0..generators.len()).collect();
        // Stable: equal generators keep the order given.
        order.sort_by(|&a, &b| generators[a].cmp(&generators[b]));
        let mut map = DiagMap {
            factors: Vec::new(),
            start: vec![0],
            rates: Vec::new(),
            factor,
        };
        let mut i = 0;
        while i < order.len() {
            let mut rate = 0.0;
            let mut j = i;
            while j < order.len() && generators[order[j]] == generators[order[i]] {
                rate += rates[order[j]];
                j += 1;
            }
            map.factors.extend_from_slice(&generators[order[i]]);
            map.start.push(map.factors.len() as u32);
            map.rates.push(rate);
            i = j;
        }
        map
    }

    /// Number of generators.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.rates.len()
    }

    /// The factors `(qubit, letter)` of generator `i`.
    pub(crate) fn generator(&self, i: usize) -> &[(u32, u8)] {
        &self.factors[self.start[i] as usize..self.start[i + 1] as usize]
    }

    /// The rates `λ_k`.
    pub(crate) fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// The factor `α`.
    pub(crate) fn factor(&self) -> f64 {
        self.factor
    }

    /// `Λ = Σ λ_k` over the generators that anticommute with the string
    /// `key` (x words, then z words, on the qubits of the generators),
    /// summed in generator order.
    pub(crate) fn exponent(&self, key: &[u64]) -> f64 {
        let w = key.len() / 2;
        let mut sum = 0.0;
        for (i, &rate) in self.rates.iter().enumerate() {
            let mut parity = 0u64;
            for &(q, letter) in self.generator(i) {
                let (word, bit) = ((q >> 6) as usize, q & 63);
                let (x, z) = ((key[word] >> bit) & 1, (key[w + word] >> bit) & 1);
                parity ^= (x & (letter as u64 >> 1)) ^ (z & (letter as u64 & 1));
            }
            if parity == 1 {
                sum += rate;
            }
        }
        sum
    }

    /// The factor `exp(−2αΛ)` of string `key`.
    pub(crate) fn fidelity(&self, key: &[u64]) -> f64 {
        (-2.0 * self.factor * self.exponent(key)).exp()
    }

    /// The norm bound `γ = exp(2|α| Σ λ_k)` of the map for `α < 0` (1 for
    /// `α ≥ 0`, a channel).
    #[cfg(test)]
    pub(crate) fn gamma(&self) -> f64 {
        if self.factor < 0.0 {
            (-2.0 * self.factor * self.rates.iter().sum::<f64>()).exp()
        } else {
            1.0
        }
    }

    /// The union of the generators' supports (one mask of `words` words).
    pub(crate) fn support(&self, words: usize) -> Vec<u64> {
        let mut s = vec![0u64; words];
        for &(q, _) in &self.factors {
            s[(q >> 6) as usize] |= 1 << (q & 63);
        }
        s
    }
}

/// One signed local string (at most 32 qubits), the target of the
/// tableau's rules: bit `j` of `x`/`z` is local qubit `j`, bit 0 of `r` the
/// sign.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalString {
    pub(crate) x: u64,
    pub(crate) z: u64,
    pub(crate) r: u64,
}

impl LocalString {
    /// Apply a one-qubit rule to qubit `q`.
    fn one(&mut self, q: usize, rule: fn(&mut u64, &mut u64, &mut u64)) {
        let (mut x, mut z, mut r) = ((self.x >> q) & 1, (self.z >> q) & 1, 0);
        rule(&mut x, &mut z, &mut r);
        self.x = (self.x & !(1 << q)) | ((x & 1) << q);
        self.z = (self.z & !(1 << q)) | ((z & 1) << q);
        self.r ^= r & 1;
    }
}

impl CliffordTarget for LocalString {
    fn h(&mut self, q: usize) {
        self.one(q, rules::h);
    }

    fn s(&mut self, q: usize) {
        self.one(q, rules::s);
    }

    fn sdg(&mut self, q: usize) {
        self.one(q, rules::sdg);
    }

    fn x(&mut self, q: usize) {
        self.one(q, rules::x);
    }

    fn y(&mut self, q: usize) {
        self.one(q, rules::y);
    }

    fn z(&mut self, q: usize) {
        self.one(q, rules::z);
    }

    fn cx(&mut self, c: usize, t: usize) {
        let (mut xc, mut zc) = ((self.x >> c) & 1, (self.z >> c) & 1);
        let (mut xt, mut zt) = ((self.x >> t) & 1, (self.z >> t) & 1);
        let mut r = 0;
        rules::cx(&mut xc, &mut zc, &mut xt, &mut zt, &mut r);
        self.x = (self.x & !(1 << c) & !(1 << t)) | ((xc & 1) << c) | ((xt & 1) << t);
        self.z = (self.z & !(1 << c) & !(1 << t)) | ((zc & 1) << c) | ((zt & 1) << t);
        self.r ^= r & 1;
    }

    fn swap(&mut self, a: usize, b: usize) {
        let (mut xa, mut za) = ((self.x >> a) & 1, (self.z >> a) & 1);
        let (mut xb, mut zb) = ((self.x >> b) & 1, (self.z >> b) & 1);
        rules::swap(&mut xa, &mut za, &mut xb, &mut zb);
        self.x = (self.x & !(1 << a) & !(1 << b)) | (xa << a) | (xb << b);
        self.z = (self.z & !(1 << a) & !(1 << b)) | (za << a) | (zb << b);
    }
}

/// How an instruction acts on strings.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Route {
    /// One string to one string, by the tableau's rules.
    Clifford,
    /// `exp(−iθG/2)` with the local generator `G` (`b`-index) and angle.
    Rotation(usize, f64),
    /// Transfer matrix of the instruction's matrix.
    Transfer,
}

fn route(inst: &Instruction, angles: &[f64]) -> Route {
    use GateKind::*;
    match (inst.kind, inst.controls.len()) {
        (H | X | Y | Z | S | Sdg | SX | SXdg | Swap, 0) | (X | Y | Z, 1) => Route::Clifford,
        // One qubit: b = x | z << 1; two qubits: b = x | z << 2.
        (RX, 0) => Route::Rotation(0b01, angles[0]),
        (RY, 0) => Route::Rotation(0b11, angles[0]),
        (RZ | P, 0) => Route::Rotation(0b10, angles[0]),
        (T, 0) => Route::Rotation(0b10, FRAC_PI_4),
        (Tdg, 0) => Route::Rotation(0b10, -FRAC_PI_4),
        (RXX, 0) => Route::Rotation(0b0011, angles[0]),
        (RYY, 0) => Route::Rotation(0b1111, angles[0]),
        (RZZ, 0) => Route::Rotation(0b1100, angles[0]),
        _ => Route::Transfer,
    }
}

/// The table of a Clifford instruction on local qubits `0..k`: string `b`
/// maps to the one string `inst† P_b inst` with its sign.
pub(crate) fn clifford_table(inst: &Instruction, k: usize) -> Result<LocalMap> {
    let inverse = inst.inverse();
    let mask = (1usize << k) - 1;
    let mut columns = Vec::with_capacity(1 << (2 * k));
    for b in 0..1usize << (2 * k) {
        let mut s = LocalString {
            x: (b & mask) as u64,
            z: (b >> k) as u64,
            r: 0,
        };
        // The Schrödinger rule of inst† is the Heisenberg rule of inst.
        apply_clifford(&mut s, &inverse, &[])?;
        let image = s.x as usize | (s.z as usize) << k;
        let f = if s.r & 1 == 1 { -1.0 } else { 1.0 };
        columns.push(vec![(image, f)]);
    }
    Ok(LocalMap::from_columns(k, columns))
}

/// `i^k` with `k mod 4` of the product `G·P = i^k P_{g ⊕ b}` of local
/// strings, by the phase rule of [`PauliString::multiply`](crate::quantum::PauliString::multiply).
fn product_phase(g: usize, b: usize, k: usize) -> u32 {
    let mask = (1usize << k) - 1;
    let y = |s: usize| ((s & mask) & (s >> k)).count_ones();
    let (gz, bx) = (g >> k, b & mask);
    (y(g) + y(b) + 2 * (gz & bx).count_ones() + 4 * 64 - y(g ^ b)) % 4
}

/// Whether local strings `a` and `b` anticommute.
fn anticommute(a: usize, b: usize, k: usize) -> bool {
    let mask = (1usize << k) - 1;
    (((a & mask) & (b >> k)).count_ones() + ((a >> k) & (b & mask)).count_ones()) % 2 == 1
}

/// The table of `R = exp(−iθG/2)` on `k` qubits: `R† P R = P` if `P`
/// commutes with `G`, else `cos θ·P + sin θ·(iGP)` with `iGP = ±P_{b⊕g}`.
pub(crate) fn rotation_table(g: usize, theta: f64, k: usize) -> LocalMap {
    let (sin, cos) = theta.sin_cos();
    let columns = (0..1usize << (2 * k))
        .map(|b| {
            if !anticommute(g, b, k) {
                return vec![(b, 1.0)];
            }
            // iGP = i^{k+1} P_{b⊕g}, and k is odd for anticommuting strings.
            let sign = if (product_phase(g, b, k) + 1).is_multiple_of(4) {
                1.0
            } else {
                -1.0
            };
            vec![(b, cos), (b ^ g, sign * sin)]
        })
        .collect();
    LocalMap::from_columns(k, columns)
}

/// `inst` on local qubits: controls `0..c`, targets `c..c + t`, angles
/// bound.
fn localised(inst: &Instruction, angles: &[f64]) -> Instruction {
    let c = inst.controls.len();
    Instruction {
        kind: inst.kind,
        targets: (c..c + inst.targets.len()).collect(),
        controls: (0..c).collect(),
        params: angles.iter().map(|&a| Param::Value(a)).collect(),
        matrix: inst.matrix.clone(),
    }
}

/// The table of a validated instruction with bound `angles` (at most
/// [`MAX_TRANSFER_QUBITS`] qubits), from the cache.
pub(crate) fn instruction_table(
    inst: &Instruction,
    angles: &[f64],
    cache: &mut TableCache,
) -> Result<Arc<LocalMap>> {
    let local = localised(inst, angles);
    let k = local.controls.len() + local.targets.len();
    let key: GateKey = (
        inst.kind as u8,
        local.controls.len(),
        local.targets.len(),
        angles.iter().map(|a| a.to_bits()).collect(),
        local
            .matrix
            .iter()
            .flat_map(|m| m.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]))
            .collect(),
    );
    cache.gate(key, || match route(&local, angles) {
        Route::Clifford => clifford_table(&local, k),
        Route::Rotation(g, theta) => Ok(rotation_table(g, theta, k)),
        Route::Transfer => Ok(gate_table(&instruction_matrix(&local, angles), k)),
    })
}

/// The op of one instruction of a circuit with parameters `theta`
/// (checked against the circuit), or `None` for an identity.
pub(crate) fn compile_instruction(
    inst: &Instruction,
    theta: &[f64],
    n_qubits: usize,
    cache: &mut TableCache,
) -> Result<Option<Op>> {
    inst.validate(n_qubits)?;
    if inst.is_identity() {
        return Ok(None);
    }
    let k = inst.controls.len() + inst.targets.len();
    if k > MAX_TRANSFER_QUBITS {
        return Err(QsoError::invalid(
            "circuit",
            format!(
                "{} acts on {k} qubits; Pauli propagation takes gates on at most \
                 {MAX_TRANSFER_QUBITS}",
                inst.name()
            ),
        ));
    }
    let angles = inst.angles(theta);
    if angles.iter().any(|a| !a.is_finite()) {
        return Err(QsoError::invalid("angle", "non-finite gate angle"));
    }
    let map = instruction_table(inst, &angles, cache)?;
    Ok(Some(Op::Local {
        qubits: inst.qubits().collect(),
        map,
    }))
}

/// Check the rates of a Pauli–Lindblad layer on `n_qubits` qubits.
fn check_rates(rates: &PauliOperator, factor: f64, n_qubits: usize) -> Result<()> {
    if rates.n_qubits() != n_qubits {
        return Err(QsoError::DimensionMismatch {
            expected: n_qubits,
            actual: rates.n_qubits(),
        });
    }
    if !factor.is_finite() {
        return Err(QsoError::invalid("factor", "the factor must be finite"));
    }
    let (keys, coefs) = rates.rows();
    let r = 2 * rates.words();
    for (i, &c) in coefs.iter().enumerate() {
        if keys[i * r..(i + 1) * r].iter().all(|&w| w == 0) {
            return Err(QsoError::invalid(
                "rates",
                "a Pauli–Lindblad model has no identity generator",
            ));
        }
        if c < 0.0 {
            return Err(QsoError::invalid(
                "rates",
                "rates must be non-negative (scale them with the factor)",
            ));
        }
    }
    Ok(())
}

/// The ops of `layers` on a register of `n_qubits` qubits, in time order.
pub(crate) fn compile_layers(
    layers: &[Layer],
    n_qubits: usize,
    cache: &mut TableCache,
) -> Result<Vec<Op>> {
    let mut ops = Vec::new();
    for layer in layers {
        match layer {
            Layer::Gates { circuit, theta } => {
                if circuit.n_qubits() > n_qubits {
                    return Err(QsoError::DimensionMismatch {
                        expected: n_qubits,
                        actual: circuit.n_qubits(),
                    });
                }
                circuit.check_parameters(theta)?;
                for inst in circuit.instructions() {
                    if let Some(op) = compile_instruction(inst, theta, circuit.n_qubits(), cache)? {
                        ops.push(op);
                    }
                }
            }
            Layer::Channel { channel, qubits } => {
                if qubits.len() != channel.n_qubits() {
                    return Err(QsoError::DimensionMismatch {
                        expected: channel.n_qubits(),
                        actual: qubits.len(),
                    });
                }
                for (k, &q) in qubits.iter().enumerate() {
                    if q >= n_qubits {
                        return Err(QsoError::QubitOutOfRange { qubit: q, n_qubits });
                    }
                    if qubits[..k].contains(&q) {
                        return Err(QsoError::DuplicateQubit {
                            gate: "channel",
                            qubit: q,
                        });
                    }
                }
                ops.push(Op::Local {
                    qubits: qubits.clone(),
                    map: cache.channel(channel),
                });
            }
            Layer::PauliLindblad { rates, factor } => {
                check_rates(rates, *factor, n_qubits)?;
                if !rates.is_empty() {
                    ops.push(Op::Diag(Arc::new(DiagMap::new(rates, *factor))));
                }
            }
        }
    }
    Ok(ops)
}

// ---------------------------------------------------------------------
// Schedule, light cone, relabelling and passes
// ---------------------------------------------------------------------

/// No op on a label of a [`Level`].
const NONE: u32 = u32::MAX;

/// Levels with at most this many local ops are searched op by op; larger
/// ones through a map from labels to ops.
const DIRECT_OPS: usize = 8;

/// A gate or channel of a [`Level`] on compact labels.
#[derive(Clone, Debug)]
pub(crate) struct LocalOp {
    /// The labels of the op's qubits, in the order of its table (the
    /// first `k`; inline, so that a level is one array).
    slots: [u32; MAX_TRANSFER_QUBITS],
    k: usize,
    pub(crate) map: Arc<LocalMap>,
}

impl LocalOp {
    fn new(qubits: &[u32], map: Arc<LocalMap>) -> Self {
        let mut slots = [0; MAX_TRANSFER_QUBITS];
        slots[..qubits.len()].copy_from_slice(qubits);
        LocalOp {
            slots,
            k: qubits.len(),
            map,
        }
    }

    /// The labels of the op's qubits, in the order of its table.
    #[inline]
    pub(crate) fn qubits(&self) -> &[u32] {
        &self.slots[..self.k]
    }
}

/// How a level finds the ops a string touches.
#[derive(Clone, Debug)]
enum Lookup {
    /// Every op is checked.
    Direct,
    /// The op of every label ([`NONE`] for none).
    Dense(Vec<u32>),
    /// `(label, op)`, ascending in the label.
    Sorted(Vec<(u32, u32)>),
}

/// Ops on disjoint qubits, applied to every string at once.
#[derive(Clone, Debug)]
pub(crate) struct Level {
    pub(crate) locals: Vec<LocalOp>,
    lookup: Lookup,
    /// Pauli–Lindblad layers (on compact labels).
    pub(crate) diags: Vec<Arc<DiagMap>>,
}

/// The local index `b = xbits | zbits << k` of string `key` (`w` words
/// per mask) on the labels `qubits`.
#[inline]
pub(crate) fn gather(key: &[u64], w: usize, qubits: &[u32]) -> usize {
    let k = qubits.len();
    let mut b = 0usize;
    for (j, &q) in qubits.iter().enumerate() {
        let (word, bit) = ((q >> 6) as usize, q & 63);
        b |= (((key[word] >> bit) & 1) as usize) << j;
        b |= (((key[w + word] >> bit) & 1) as usize) << (k + j);
    }
    b
}

/// Write local string `b` onto the labels `qubits` of `key`.
#[inline]
pub(crate) fn scatter(key: &mut [u64], w: usize, qubits: &[u32], b: usize) {
    let k = qubits.len();
    for (j, &q) in qubits.iter().enumerate() {
        let (word, bit) = ((q >> 6) as usize, q & 63);
        let x = ((b >> j) & 1) as u64;
        let z = ((b >> (k + j)) & 1) as u64;
        key[word] = (key[word] & !(1 << bit)) | (x << bit);
        key[w + word] = (key[w + word] & !(1 << bit)) | (z << bit);
    }
}

impl Level {
    fn new(locals: Vec<LocalOp>, diags: Vec<Arc<DiagMap>>, labels: usize) -> Self {
        let qubits: usize = locals.iter().map(|op| op.k).sum();
        let lookup = if locals.len() <= DIRECT_OPS {
            Lookup::Direct
        } else if labels <= 4096 || labels <= 16 * qubits {
            let mut of = vec![NONE; labels];
            for (i, op) in locals.iter().enumerate() {
                for &q in op.qubits() {
                    of[q as usize] = i as u32;
                }
            }
            Lookup::Dense(of)
        } else {
            let mut of: Vec<(u32, u32)> = locals
                .iter()
                .enumerate()
                .flat_map(|(i, op)| op.qubits().iter().map(move |&q| (q, i as u32)))
                .collect();
            of.sort_unstable();
            Lookup::Sorted(of)
        };
        Level {
            locals,
            lookup,
            diags,
        }
    }

    /// The local ops that string `key` touches and that change it, as
    /// `(op, b)` with `b ≠ 0` and `b` not fixed by the op,
    /// in ascending order of the op's lowest touched label (op order for
    /// small levels).
    #[inline]
    pub(crate) fn active(&self, key: &[u64], w: usize, out: &mut Vec<(u32, u32)>) {
        out.clear();
        let op_at = |q: u32| -> u32 {
            match &self.lookup {
                Lookup::Direct => NONE,
                Lookup::Dense(of) => of[q as usize],
                Lookup::Sorted(of) => match of.binary_search_by_key(&q, |&(l, _)| l) {
                    Ok(i) => of[i].1,
                    Err(_) => NONE,
                },
            }
        };
        // A string that touches many ops is cheaper to check op by op.
        let direct = match self.lookup {
            Lookup::Direct => true,
            _ => {
                let weight: usize = (0..w)
                    .map(|j| (key[j] | key[w + j]).count_ones() as usize)
                    .sum();
                2 * weight >= self.locals.len()
            }
        };
        if direct {
            for (i, op) in self.locals.iter().enumerate() {
                let b = gather(key, w, op.qubits());
                if b != 0 && !op.map.is_fixed(b) {
                    out.push((i as u32, b as u32));
                }
            }
            return;
        }
        for word in 0..w {
            let mut bits = key[word] | key[w + word];
            while bits != 0 {
                let q = (64 * word) as u32 + bits.trailing_zeros();
                bits &= bits - 1;
                let o = op_at(q);
                if o == NONE {
                    continue;
                }
                let op = &self.locals[o as usize];
                let b = gather(key, w, op.qubits());
                if op.map.is_fixed(b) {
                    continue;
                }
                // Take the op once, at its lowest label that the string
                // touches.
                let k = op.k;
                let support = (b | (b >> k)) & ((1 << k) - 1);
                let mut lowest = u32::MAX;
                for (j, &label) in op.qubits().iter().enumerate() {
                    if (support >> j) & 1 == 1 {
                        lowest = lowest.min(label);
                    }
                }
                if lowest == q {
                    out.push((o, b as u32));
                }
            }
        }
    }
}

/// A run of levels propagated together.
#[derive(Clone, Debug)]
pub(crate) struct Pass {
    /// Every op of every level is a bijection on strings (Clifford gates,
    /// Pauli channels, Pauli–Lindblad layers): the table is changed in
    /// place. Otherwise the pass has one level and splits strings.
    pub(crate) in_place: bool,
    pub(crate) levels: Vec<Level>,
    /// `Γ`: the norm bound of this pass and every later one (the product
    /// of `e^{2|α|Σλ_k}` of their inverted Pauli–Lindblad layers).
    pub(crate) gamma: f64,
}

/// A compiled propagation: the observable and the passes on the light
/// cone, its qubits relabelled `0..m`.
#[derive(Debug)]
pub(crate) struct Plan {
    /// Register qubit of every compact label, ascending.
    pub(crate) labels: Vec<usize>,
    /// Words per mask on the compact labels (at least 1).
    pub(crate) words: usize,
    /// The observable's rows on compact labels (`2·words` per row).
    pub(crate) keys: Vec<u64>,
    pub(crate) coefs: Vec<f64>,
    /// In propagation order (last op of the circuit first).
    pub(crate) passes: Vec<Pass>,
    /// `Γ` of the observable itself (every inverted layer).
    pub(crate) gamma: f64,
}

/// The register qubits of a mask.
fn mask_qubits(mask: &[u64]) -> Vec<usize> {
    let mut out = Vec::new();
    for (word, &m) in mask.iter().enumerate() {
        let mut bits = m;
        while bits != 0 {
            out.push(64 * word + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
    out
}

/// The local letter `x | z << 1` (I, X, Z, Y as 0, 1, 2, 3) of register
/// qubit `q` in `key` (`w` words per mask).
fn letter_at(key: &[u64], w: usize, q: usize) -> u8 {
    let (j, bit) = (q / 64, q % 64);
    (((key[j] >> bit) & 1) | (((key[w + j] >> bit) & 1) << 1)) as u8
}

/// As-soon-as-possible levels (from 1) of ops with the given supports, in
/// the given order: each op one level after the last op on any of its
/// qubits.
fn asap_levels(supports: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut last = vec![0usize; n];
    supports
        .iter()
        .map(|support| {
            let level = 1 + support.iter().map(|&q| last[q]).max().unwrap_or(0);
            for &q in support {
                last[q] = level;
            }
            level
        })
        .collect()
}

/// Indices sorted by (level, index).
fn by_level(levels: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..levels.len()).collect();
    order.sort_by_key(|&i| (levels[i], i));
    order
}

/// An op kept in the light cone, on register qubits.
enum Kept {
    Local {
        qubits: Vec<usize>,
        map: Arc<LocalMap>,
    },
    /// Generators restricted to the cone, as `(qubit, letter)` factors.
    Diag {
        generators: Vec<Vec<(u32, u8)>>,
        rates: Vec<f64>,
        factor: f64,
    },
}

/// Schedule `ops` (time order) for propagating `observable`, cut them to
/// its light cone, relabel the cone's qubits `0..m` and group the levels
/// into passes.
///
/// 1. In propagation order, every op gets the as-soon-as-possible level of
///    its full support (all generators of a Pauli–Lindblad layer); ops are
///    sorted by level, which only swaps ops on disjoint qubits.
/// 2. Walking that order from the observable, the letters every qubit
///    may carry are tracked (every string lies in their product): an op
///    that leaves every string it can meet as it is is removed (adjoint
///    channels and unitaries are unital, so this covers every op off the
///    strings' support, and gates that commute with them); a kept op
///    replaces the letters of its qubits by those of its images. A
///    Pauli–Lindblad generator acts on the strings as its factors on
///    qubits where some possible letter anticommutes with them, so it is
///    cut to those (and dropped if none is left).
/// 3. The kept ops get levels again on the cone; consecutive levels of
///    bijections form one in-place pass, every other level a splitting
///    pass.
pub(crate) fn plan(ops: Vec<Op>, observable: &PauliOperator) -> Result<Plan> {
    let n = observable.n_qubits();
    let wr = observable.words();
    let ops: Vec<Op> = ops.into_iter().rev().collect();
    let supports: Vec<Vec<usize>> = ops
        .iter()
        .map(|op| match op {
            Op::Local { qubits, .. } => qubits.clone(),
            Op::Diag(d) => mask_qubits(&d.support(wr)),
        })
        .collect();
    let order = by_level(&asap_levels(&supports, n));

    let (obs_keys, obs_coefs) = observable.rows();
    // The letters each register qubit may carry at the current op (bit
    // `l` for the local letter `l = x | z << 1`: I, X, Z, Y; every string
    // lies in their product), starting from the observable's. `I` is
    // always possible.
    let mut letters = vec![1u8; n];
    let mut support = vec![0u64; wr];
    for key in obs_keys.chunks_exact(2 * wr) {
        for j in 0..wr {
            support[j] |= key[j] | key[wr + j];
        }
        for q in mask_qubits(&(0..wr).map(|j| key[j] | key[wr + j]).collect::<Vec<u64>>()) {
            letters[q] |= 1 << letter_at(key, wr, q);
        }
    }
    let mut kept = Vec::new();
    for i in order {
        match &ops[i] {
            Op::Local { qubits, map } => {
                if qubits.iter().all(|&q| letters[q] == 1) {
                    continue;
                }
                // The local strings the op can meet, and the letters of
                // their images.
                let k = qubits.len();
                let mut images = [0u8; MAX_TRANSFER_QUBITS];
                let mut acts = false;
                for b in 0..1usize << (2 * k) {
                    let meets = (0..k).all(|j| {
                        let l = ((b >> j) & 1) | (((b >> (k + j)) & 1) << 1);
                        (letters[qubits[j]] >> l) & 1 == 1
                    });
                    if !meets {
                        continue;
                    }
                    acts |= !map.is_fixed(b);
                    // With the snapped images: the exact map carries them
                    // on, and the bound counts them through every later
                    // op, so the letters (and the generators cut by them)
                    // must cover them too.
                    let reach = map.reach(b);
                    for (j, set) in images[..k].iter_mut().enumerate() {
                        *set |= ((reach >> (4 * j)) & 0xF) as u8;
                    }
                }
                // An op that leaves every string it can meet as it is
                // (unitaries and unital channels off the strings'
                // support, gates commuting with every possible string) is
                // removed.
                if !acts {
                    continue;
                }
                for (j, &q) in qubits.iter().enumerate() {
                    letters[q] = images[j];
                }
                kept.push(Kept::Local {
                    qubits: qubits.clone(),
                    map: Arc::clone(map),
                });
            }
            Op::Diag(d) => {
                let mut generators = Vec::new();
                let mut rates = Vec::new();
                for (g, &rate) in d.rates().iter().enumerate() {
                    // Keep the factors that some possible letter of their
                    // qubit anticommutes with (letters other than I and
                    // the factor's own).
                    let factors: Vec<(u32, u8)> = d
                        .generator(g)
                        .iter()
                        .copied()
                        .filter(|&(q, l)| letters[q as usize] & !(1 | 1 << l) != 0)
                        .collect();
                    if !factors.is_empty() {
                        generators.push(factors);
                        rates.push(rate);
                    }
                }
                if !generators.is_empty() {
                    kept.push(Kept::Diag {
                        generators,
                        rates,
                        factor: d.factor(),
                    });
                }
            }
        }
    }

    // Relabel the observable's support and the qubits of the kept ops
    // (every qubit where a string can carry a letter, and qubits where a
    // kept op only meets `I`).
    let mut labelled = support;
    for k in &kept {
        if let Kept::Local { qubits, .. } = k {
            for &q in qubits {
                labelled[q / 64] |= 1 << (q % 64);
            }
        }
    }
    let labels = mask_qubits(&labelled);
    let m = labels.len();
    let w = m.div_ceil(64).max(1);
    let mut label_of = vec![NONE; n];
    for (l, &q) in labels.iter().enumerate() {
        label_of[q] = l as u32;
    }
    let mut keys = Vec::with_capacity(obs_coefs.len() * 2 * w);
    for key in obs_keys.chunks_exact(2 * wr) {
        let mut out = vec![0u64; 2 * w];
        for q in mask_qubits(&(0..wr).map(|j| key[j] | key[wr + j]).collect::<Vec<u64>>()) {
            let l = label_of[q] as usize;
            let (j, bit) = (q / 64, q % 64);
            out[l / 64] |= ((key[j] >> bit) & 1) << (l % 64);
            out[w + l / 64] |= ((key[wr + j] >> bit) & 1) << (l % 64);
        }
        keys.extend(out);
    }

    // Steps on compact labels, their supports and Γ exponents.
    enum Step {
        Local(LocalOp),
        Diag(Arc<DiagMap>, f64),
    }
    let mut steps = Vec::with_capacity(kept.len());
    let mut step_supports = Vec::with_capacity(kept.len());
    for k in kept {
        match k {
            Kept::Local { qubits, map } => {
                let qubits: Vec<u32> = qubits.iter().map(|&q| label_of[q]).collect();
                step_supports.push(qubits.iter().map(|&q| q as usize).collect());
                steps.push(Step::Local(LocalOp::new(&qubits, map)));
            }
            Kept::Diag {
                generators,
                rates,
                factor,
            } => {
                // Labels ascend with the qubits, so the factors stay
                // sorted. Equal restricted generators merge: commuting
                // Pauli channels compose by adding their rates.
                let generators = generators
                    .into_iter()
                    .map(|factors| {
                        factors
                            .into_iter()
                            .map(|(q, l)| (label_of[q as usize], l))
                            .collect()
                    })
                    .collect();
                let d = DiagMap::from_generators(generators, rates, factor);
                let exponent = if factor < 0.0 {
                    -2.0 * factor * d.rates().iter().sum::<f64>()
                } else {
                    0.0
                };
                step_supports.push(mask_qubits(&d.support(w)));
                steps.push(Step::Diag(Arc::new(d), exponent));
            }
        }
    }
    let levels = asap_levels(&step_supports, m.max(1));
    let n_levels = levels.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_levels];
    for i in by_level(&levels) {
        buckets[levels[i] - 1].push(i);
    }
    let mut steps: Vec<Option<Step>> = steps.into_iter().map(Some).collect();
    let mut passes: Vec<Pass> = Vec::new();
    let mut exponents: Vec<f64> = Vec::new();
    for bucket in buckets {
        let mut locals = Vec::new();
        let mut diags = Vec::new();
        let mut exponent = 0.0;
        for i in bucket {
            match steps[i].take().expect("each step once") {
                Step::Local(op) => locals.push(op),
                Step::Diag(d, e) => {
                    diags.push(d);
                    exponent += e;
                }
            }
        }
        let in_place = locals.iter().all(|op| op.map.is_bijective());
        let level = Level::new(locals, diags, m.max(1));
        match passes.last_mut() {
            Some(last) if in_place && last.in_place => {
                last.levels.push(level);
                *exponents.last_mut().expect("one per pass") += exponent;
            }
            _ => {
                passes.push(Pass {
                    in_place,
                    levels: vec![level],
                    gamma: 1.0,
                });
                exponents.push(exponent);
            }
        }
    }
    // Γ(p) = exp(Σ_{q ≥ p} exponent_q), and Γ of the observable over all.
    let mut suffix = 0.0;
    for (pass, e) in passes.iter_mut().zip(&exponents).rev() {
        suffix += e;
        pass.gamma = suffix.exp();
    }
    let gamma = suffix.exp();
    if !gamma.is_finite() {
        return Err(QsoError::invalid(
            "factor",
            format!(
                "the inverted Pauli–Lindblad layers amplify errors by e^{suffix:.3e}, \
                 beyond floating point"
            ),
        ));
    }
    // Every coefficient, the value and the bound are at most Γ·‖O‖₁ (up to
    // rounding): refuse what would overflow on the way.
    let norm = observable.one_norm();
    if !(gamma * norm).is_finite() {
        return Err(QsoError::invalid(
            if gamma > 1.0 { "factor" } else { "observable" },
            format!(
                "the observable's 1-norm {norm:.3e} times the amplification e^{suffix:.3e} \
                 of the inverted Pauli–Lindblad layers lies beyond floating point"
            ),
        ));
    }
    Ok(Plan {
        labels,
        words: w,
        keys,
        coefs: obs_coefs.to_vec(),
        passes,
        gamma,
    })
}
