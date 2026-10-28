//! CSS codes given by their checks: triangular colour codes, bivariate
//! bicycle codes, and memory experiments for any of them.

use super::circuit::{DetectorCircuit, MAX_OBSERVABLES, Noise};
use super::codes::{CircuitNoise, MAX_DISTANCE, MemoryBasis, Noisy, check_rounds, check_size};
use super::gf2::{self, Basis};
use crate::error::{QsoError, Result};
use serde::{Deserialize, Serialize};

/// The six directions from a face centre of [`CssCode::color_code`] to
/// its qubits, in order around the hexagon (lattice steps `(Δi, Δj)`).
pub const COLOR_DIRECTIONS: [(i64, i64); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];

/// Largest `l · m` of a bivariate bicycle code (it has `2 l m` qubits).
pub const MAX_BICYCLE_SIZE: usize = 1 << 12;
/// Most qubits of a [`CssCode`]: finding the logical operators costs
/// about n³/64 word operations (seconds at this size).
pub const MAX_CSS_QUBITS: usize = 1 << 13;

/// A CSS code: X-type and Z-type checks on `n` qubits, every X check
/// overlapping every Z check evenly, and a basis of logical operators,
/// paired so that X logical `i` and Z logical `j` overlap oddly exactly
/// when `i = j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CssCode {
    n: usize,
    x_checks: Vec<Vec<usize>>,
    z_checks: Vec<Vec<usize>>,
    x_logicals: Vec<Vec<usize>>,
    z_logicals: Vec<Vec<usize>>,
    /// For a bivariate bicycle code with three distinct monomials in each
    /// polynomial: its structure, for the depth-8 cycle.
    bicycle: Option<Bicycle>,
    /// For a colour code: the qubit of every face (X check `f` = Z check
    /// `f`) in each of the six directions around its centre.
    faces: Option<Vec<[Option<usize>; 6]>>,
}

/// The torus `l × m` and the monomials (reduced exponents, in the given
/// order) of a bivariate bicycle code.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Bicycle {
    l: usize,
    m: usize,
    a: [(usize, usize); 3],
    b: [(usize, usize); 3],
}

impl CssCode {
    /// The code of the given checks (supports of X-type and Z-type
    /// stabilizer generators; they may be dependent).
    pub fn new(n: usize, x_checks: Vec<Vec<usize>>, z_checks: Vec<Vec<usize>>) -> Result<Self> {
        if n == 0 || n > MAX_CSS_QUBITS {
            return Err(QsoError::invalid(
                "n",
                format!("between 1 and {MAX_CSS_QUBITS} qubits"),
            ));
        }
        let normal = |checks: Vec<Vec<usize>>, name: &'static str| -> Result<Vec<Vec<usize>>> {
            checks
                .into_iter()
                .map(|mut c| {
                    c.sort_unstable();
                    if c.windows(2).any(|w| w[0] == w[1]) {
                        return Err(QsoError::invalid(name, "a check lists a qubit twice"));
                    }
                    if c.last().is_some_and(|&q| q >= n) {
                        return Err(QsoError::invalid(
                            name,
                            format!("a check acts on a qubit outside 0..{n}"),
                        ));
                    }
                    if c.is_empty() {
                        return Err(QsoError::invalid(name, "a check acts on no qubit"));
                    }
                    Ok(c)
                })
                .collect()
        };
        let x_checks = normal(x_checks, "x_checks")?;
        let z_checks = normal(z_checks, "z_checks")?;
        let hx: Vec<Vec<u64>> = x_checks.iter().map(|c| gf2::from_ones(n, c)).collect();
        let hz: Vec<Vec<u64>> = z_checks.iter().map(|c| gf2::from_ones(n, c)).collect();
        for (i, x) in hx.iter().enumerate() {
            if let Some(j) = hz.iter().position(|z| gf2::dot(x, z)) {
                return Err(QsoError::invalid(
                    "z_checks",
                    format!("X check {i} and Z check {j} anticommute"),
                ));
            }
        }
        // Logical Z: commutes with the X checks, outside the Z checks'
        // span; logical X likewise.
        let complement = |kernel_of: &[Vec<u64>], span_of: &[Vec<u64>]| -> Vec<Vec<u64>> {
            let mut span = Basis::new(n, false, 0);
            for r in span_of {
                span.insert(r);
            }
            gf2::kernel(kernel_of, n)
                .into_iter()
                .filter(|v| span.insert(v).is_some())
                .collect()
        };
        let mut zl = complement(&hx, &hz);
        let mut xl = complement(&hz, &hx);
        debug_assert_eq!(zl.len(), xl.len());
        // Symplectic Gram–Schmidt: pair X logical i with Z logical i.
        let k = zl.len();
        for i in 0..k {
            let j = (i..k)
                .find(|&j| gf2::dot(&xl[i], &zl[j]))
                .expect("the logical operators pair up");
            zl.swap(i, j);
            for j in 0..k {
                if j != i && gf2::dot(&xl[i], &zl[j]) {
                    let zi = zl[i].clone();
                    gf2::xor_into(&mut zl[j], &zi);
                }
            }
            for l in 0..k {
                if l != i && gf2::dot(&xl[l], &zl[i]) {
                    let xi = xl[i].clone();
                    gf2::xor_into(&mut xl[l], &xi);
                }
            }
        }
        let supports = |v: Vec<Vec<u64>>| -> Vec<Vec<usize>> {
            v.iter().map(|x| gf2::ones(x).collect()).collect()
        };
        Ok(CssCode {
            n,
            x_checks,
            z_checks,
            x_logicals: supports(xl),
            z_logicals: supports(zl),
            bicycle: None,
            faces: None,
        })
    }

    /// The triangular colour code of odd distance `d` on the hexagonal
    /// (6.6.6) lattice: `(3d² + 1)/4` qubits, one logical qubit, the same
    /// faces of weight 4 and 6 as X and Z checks.
    ///
    /// Lattice points `(i, j)` with `i, j ≥ 0` and `i + j ≤ 3(d − 1)/2`;
    /// those with `i − j ≡ 1 (mod 3)` are face centres, the others qubits,
    /// and each face holds the qubits among its centre's six neighbours.
    pub fn color_code(distance: usize) -> Result<Self> {
        if distance < 3 || distance.is_multiple_of(2) || distance > MAX_DISTANCE {
            return Err(QsoError::invalid(
                "distance",
                format!("the distance must be odd and between 3 and {MAX_DISTANCE}"),
            ));
        }
        let l = 3 * (distance as i64 - 1) / 2;
        let centre = |i: i64, j: i64| (i - j - 1).rem_euclid(3) == 0;
        let inside = |i: i64, j: i64| i >= 0 && j >= 0 && i + j <= l;
        let mut index = std::collections::HashMap::new();
        for i in 0..=l {
            for j in 0..=l - i {
                if !centre(i, j) {
                    let k = index.len();
                    index.insert((i, j), k);
                }
            }
        }
        let mut faces = Vec::new();
        let mut around = Vec::new();
        for i in -2..=l + 2 {
            for j in -2..=l + 2 {
                if !centre(i, j) {
                    continue;
                }
                let slots = COLOR_DIRECTIONS
                    .map(|(a, b)| inside(i + a, j + b).then(|| index[&(i + a, j + b)]));
                let f: Vec<usize> = slots.iter().flatten().copied().collect();
                if f.len() >= 4 {
                    faces.push(f);
                    around.push(slots);
                }
            }
        }
        let mut code = CssCode::new(index.len(), faces.clone(), faces)?;
        code.faces = Some(around);
        Ok(code)
    }

    /// The bivariate bicycle code of Bravyi et al. (2024) on `2 l m`
    /// qubits: with `x = S_l ⊗ I_m` and `y = I_l ⊗ S_m` (cyclic shifts),
    /// polynomials `A = Σ x^a y^b` over the monomials `a` and `B` over
    /// `b`, the checks are the rows of `H_X = [A | B]` and
    /// `H_Z = [Bᵀ | Aᵀ]`. Qubit `(i, j)` of the left block is `i m + j`,
    /// of the right block `l m + i m + j`.
    ///
    /// For instance `l = m = 6`, `A = x³ + y + y²`, `B = y³ + x + x²`
    /// gives the [[72, 12, 6]] code, `l = 12, m = 6` with the same
    /// polynomials the [[144, 12, 12]] "gross" code.
    pub fn bivariate_bicycle(
        l: usize,
        m: usize,
        a: &[(usize, usize)],
        b: &[(usize, usize)],
    ) -> Result<Self> {
        if l == 0 || m == 0 || l.saturating_mul(m) > MAX_BICYCLE_SIZE {
            return Err(QsoError::invalid(
                "l",
                format!("l and m must be positive with l · m ≤ {MAX_BICYCLE_SIZE}"),
            ));
        }
        let (l64, m64) = (l as i64, m as i64);
        let idx = |i: i64, j: i64| (i.rem_euclid(l64) * m64 + j.rem_euclid(m64)) as usize;
        let lm = l * m;
        // Rows of a polynomial matrix: row (i, j) of x^a y^b has its one
        // at column (i + a, j + b); of its transpose at (i − a, j − b).
        let row = |terms: &[(usize, usize)], i: i64, j: i64, sign: i64, offset: usize| {
            let mut v = Vec::new();
            for &(p, q) in terms {
                // Exponents modulo the orders of x and y (x^l = y^m = 1).
                let (p, q) = ((p % l) as i64, (q % m) as i64);
                let c = offset + idx(i + sign * p, j + sign * q);
                // Equal monomials cancel.
                if let Some(pos) = v.iter().position(|&x| x == c) {
                    v.swap_remove(pos);
                } else {
                    v.push(c);
                }
            }
            v
        };
        let (mut hx, mut hz) = (Vec::new(), Vec::new());
        for i in 0..l64 {
            for j in 0..m64 {
                let mut x = row(a, i, j, 1, 0);
                x.extend(row(b, i, j, 1, lm));
                let mut z = row(b, i, j, -1, 0);
                z.extend(row(a, i, j, -1, lm));
                hx.push(x);
                hz.push(z);
            }
        }
        // A zero polynomial leaves empty rows: drop them.
        hx.retain(|r| !r.is_empty());
        hz.retain(|r| !r.is_empty());
        let mut code = CssCode::new(2 * lm, hx, hz)?;
        let reduce = |terms: &[(usize, usize)]| -> Option<[(usize, usize); 3]> {
            let t: Vec<(usize, usize)> = terms.iter().map(|&(p, q)| (p % l, q % m)).collect();
            let distinct = t.len() == 3 && t[0] != t[1] && t[0] != t[2] && t[1] != t[2];
            distinct.then(|| [t[0], t[1], t[2]])
        };
        if let (Some(a), Some(b)) = (reduce(a), reduce(b)) {
            code.bicycle = Some(Bicycle { l, m, a, b });
        }
        Ok(code)
    }

    /// One of the bivariate bicycle codes of Bravyi et al. (2024),
    /// table 3, by its number of qubits: [[72, 12, 6]], [[90, 8, 10]],
    /// [[108, 8, 10]], [[144, 12, 12]] or [[288, 12, 18]]. Returns the code
    /// and its distance (from the paper; not computed).
    pub fn bivariate_bicycle_code(n: usize) -> Result<(Self, usize)> {
        let gross_a = [(3, 0), (0, 1), (0, 2)];
        let gross_b = [(0, 3), (1, 0), (2, 0)];
        type Monomials<'a> = &'a [(usize, usize)];
        let (l, m, a, b, d): (usize, usize, Monomials<'_>, Monomials<'_>, usize) = match n {
            72 => (6, 6, &gross_a, &gross_b, 6),
            90 => (
                15,
                3,
                &[(9, 0), (0, 1), (0, 2)],
                &[(0, 0), (2, 0), (7, 0)],
                10,
            ),
            108 => (9, 6, &gross_a, &gross_b, 10),
            144 => (12, 6, &gross_a, &gross_b, 12),
            288 => (12, 12, &[(3, 0), (0, 2), (0, 7)], &gross_b, 18),
            _ => {
                return Err(QsoError::invalid(
                    "n",
                    "the bivariate bicycle codes of the table have 72, 90, 108, 144 or 288 qubits",
                ));
            }
        };
        Ok((CssCode::bivariate_bicycle(l, m, a, b)?, d))
    }

    /// Qubits.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Logical qubits.
    pub fn k(&self) -> usize {
        self.z_logicals.len()
    }

    pub fn x_checks(&self) -> &[Vec<usize>] {
        &self.x_checks
    }

    pub fn z_checks(&self) -> &[Vec<usize>] {
        &self.z_checks
    }

    pub fn x_logicals(&self) -> &[Vec<usize>] {
        &self.x_logicals
    }

    pub fn z_logicals(&self) -> &[Vec<usize>] {
        &self.z_logicals
    }
}

/// CNOT layers for the edges of `checks` (check, qubit): every qubit and
/// every check in at most one CNOT per layer (greedy edge colouring).
fn layers(checks: &[Vec<usize>], n: usize) -> Vec<Vec<(usize, usize)>> {
    let mut out: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut qubit_busy: Vec<Vec<bool>> = vec![Vec::new(); n];
    for (c, support) in checks.iter().enumerate() {
        let mut used = Vec::new();
        for &q in support {
            let t = (0..)
                .find(|&t| !used.contains(&t) && !qubit_busy[q].get(t).copied().unwrap_or(false))
                .expect("a free layer");
            if qubit_busy[q].len() <= t {
                qubit_busy[q].resize(t + 1, false);
            }
            qubit_busy[q][t] = true;
            used.push(t);
            if out.len() <= t {
                out.resize(t + 1, Vec::new());
            }
            out[t].push((c, q));
        }
    }
    out
}

/// The order of the CNOTs in a round of [`css_code_memory_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum CssSchedule {
    /// Every X check, then every Z check, each in CNOT layers found
    /// greedily (the schedule of [`css_code_memory`]).
    #[default]
    Greedy,
    /// The depth-8 cycle of Bravyi et al. (Nature 627, 778 (2024),
    /// table 5) for a bivariate bicycle code whose polynomials have three
    /// monomials each, `A = A1 + A2 + A3` and `B = B1 + B2 + B3` in the
    /// order given to [`CssCode::bivariate_bicycle`]: X and Z checks
    /// interleaved, seven CNOT layers, each X check visiting
    /// `A2, B2, B1, B3, A1, A3` and each Z check `A1ᵀ, A3ᵀ, B1ᵀ, B2ᵀ, B3ᵀ,
    /// A2ᵀ`.
    Bicycle,
    /// For a colour code of [`CssCode::color_code`]: every X check, then
    /// every Z check, each face visiting its qubits in this order of the
    /// directions [`COLOR_DIRECTIONS`] (a permutation of `0..6`), one
    /// direction per CNOT layer (faces of weight 4 skip their missing
    /// directions).
    Directions([u8; 6]),
}

/// A memory experiment for a CSS code: data qubits `0..n`, then one
/// ancilla per X check and one per Z check. Each of `rounds` rounds
/// measures every X check (ancilla in `|+⟩`, CNOTs ancilla → data, H)
/// and then every Z check (CNOTs data → ancilla), the CNOTs of each type
/// in layers that touch each qubit once; then every data qubit is
/// measured.
///
/// In the Z basis the data start in `|0⟩` and the detectors are the Z
/// checks, compared with their previous round (the first round with +1,
/// the last with the final data measurements); observable `i` is Z
/// logical `i`. The X basis prepares `|+⟩` and measures X (H with the
/// noise of a one-qubit gate) with the X checks and logicals. The
/// schedule is not tuned against hook errors, so the circuit distance may
/// be below the code distance.
pub fn css_code_memory(
    code: &CssCode,
    rounds: usize,
    noise: &CircuitNoise,
    basis: MemoryBasis,
) -> Result<DetectorCircuit> {
    check_rounds(rounds)?;
    check_logicals(code)?;
    check_size(rounds, |r| build_css(code, r, noise, basis))?;
    build_css(code, rounds, noise, basis)
}

fn check_logicals(code: &CssCode) -> Result<()> {
    if code.k() > MAX_OBSERVABLES {
        return Err(QsoError::invalid(
            "code",
            format!(
                "{} logical qubits; a memory experiment has at most {MAX_OBSERVABLES} observables",
                code.k()
            ),
        ));
    }
    if code.k() == 0 {
        return Err(QsoError::invalid("code", "the code has no logical qubit"));
    }
    Ok(())
}

/// [`css_code_memory`] with the given order of CNOTs: the same detectors
/// (each round's checks of the memory basis, compared with the previous
/// round) and observables. With [`CssSchedule::Bicycle`] (only for a
/// bivariate bicycle code with three monomials per polynomial) qubits are
/// the data `0..n`, then the X-check ancillas, then the Z-check ancillas;
/// each round is the depth-8 cycle with noise after every reset, gate and
/// measurement as in [`CircuitNoise`] and depolarizing noise on the data
/// before it.
pub fn css_code_memory_with(
    code: &CssCode,
    rounds: usize,
    noise: &CircuitNoise,
    basis: MemoryBasis,
    schedule: CssSchedule,
) -> Result<DetectorCircuit> {
    match schedule {
        CssSchedule::Greedy => css_code_memory(code, rounds, noise, basis),
        CssSchedule::Directions(order) => {
            check_rounds(rounds)?;
            check_logicals(code)?;
            let Some(faces) = &code.faces else {
                return Err(QsoError::invalid(
                    "schedule",
                    "a schedule by directions needs a colour code (CssCode::color_code)",
                ));
            };
            let mut seen = [false; 6];
            for &d in &order {
                if d >= 6 || std::mem::replace(&mut seen[d as usize], true) {
                    return Err(QsoError::invalid(
                        "schedule",
                        format!("{order:?} is not a permutation of the six directions"),
                    ));
                }
            }
            let layers: Vec<Vec<(usize, usize)>> = order
                .iter()
                .map(|&d| {
                    faces
                        .iter()
                        .enumerate()
                        .filter_map(|(f, slots)| slots[d as usize].map(|q| (f, q)))
                        .collect()
                })
                .collect();
            let build = |r: usize| build_layered(code, r, noise, basis, &layers, &layers);
            check_size(rounds, build)?;
            build(rounds)
        }
        CssSchedule::Bicycle => {
            check_rounds(rounds)?;
            check_logicals(code)?;
            let Some(bicycle) = &code.bicycle else {
                return Err(QsoError::invalid(
                    "schedule",
                    "the bicycle schedule needs a bivariate bicycle code with three monomials in each polynomial",
                ));
            };
            check_size(rounds, |r| build_bicycle(code, bicycle, r, noise, basis))?;
            build_bicycle(code, bicycle, rounds, noise, basis)
        }
    }
}

/// The depth-8 syndrome cycle of Bravyi et al. (table 5), `rounds` times.
fn build_bicycle(
    code: &CssCode,
    bicycle: &Bicycle,
    rounds: usize,
    noise: &CircuitNoise,
    basis: MemoryBasis,
) -> Result<DetectorCircuit> {
    let x_basis = basis == MemoryBasis::X;
    let Bicycle { l, m, a, b } = *bicycle;
    let lm = l * m;
    let n = code.n;
    debug_assert_eq!(
        (n, code.x_checks.len(), code.z_checks.len()),
        (2 * lm, lm, lm)
    );
    let (l64, m64) = (l as i64, m as i64);
    let idx = |i: i64, j: i64| (i.rem_euclid(l64) * m64 + j.rem_euclid(m64)) as usize;
    // Data qubit of check c under monomial t: c + t (forward) or c − t
    // (transposed); left register 0..lm, right lm..2lm.
    let shift = |c: usize, t: (usize, usize), sign: i64| {
        idx(
            (c / m) as i64 + sign * t.0 as i64,
            (c % m) as i64 + sign * t.1 as i64,
        )
    };
    let left = |c: usize, t: (usize, usize), sign: i64| shift(c, t, sign);
    let right = |c: usize, t: (usize, usize), sign: i64| lm + shift(c, t, sign);
    let x_anc = |c: usize| n + c;
    let z_anc = |c: usize| n + lm + c;
    let checks: Vec<usize> = (0..lm).collect();
    let xs: Vec<usize> = checks.iter().map(|&c| x_anc(c)).collect();
    let zs: Vec<usize> = checks.iter().map(|&c| z_anc(c)).collect();
    let data: Vec<usize> = (0..n).collect();
    let mut q = Noisy {
        c: DetectorCircuit::new(2 * n)?,
        noise,
    };
    q.reset(&(0..2 * n).collect::<Vec<_>>())?;
    if x_basis {
        q.h(&data)?;
    }
    // CNOT layers 1..=7: X checks (ancilla → data) and Z checks (data →
    // ancilla).
    let x_layer = |round: usize| -> Option<Vec<(usize, usize)>> {
        let target = |c: usize| match round {
            2 => Some(left(c, a[1], 1)),
            3 => Some(right(c, b[1], 1)),
            4 => Some(right(c, b[0], 1)),
            5 => Some(right(c, b[2], 1)),
            6 => Some(left(c, a[0], 1)),
            7 => Some(left(c, a[2], 1)),
            _ => None,
        };
        target(0)?;
        Some(
            checks
                .iter()
                .map(|&c| (x_anc(c), target(c).expect("a target")))
                .collect(),
        )
    };
    let z_layer = |round: usize| -> Option<Vec<(usize, usize)>> {
        let control = |c: usize| match round {
            1 => Some(right(c, a[0], -1)),
            2 => Some(right(c, a[2], -1)),
            3 => Some(left(c, b[0], -1)),
            4 => Some(left(c, b[1], -1)),
            5 => Some(left(c, b[2], -1)),
            6 => Some(right(c, a[1], -1)),
            _ => None,
        };
        control(0)?;
        Some(
            checks
                .iter()
                .map(|&c| (control(c).expect("a control"), z_anc(c)))
                .collect(),
        )
    };
    let mut previous: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        q.c.noise(Noise::Depolarize1(noise.before_round), &data)?;
        // Round 1: X ancillas to |+⟩.
        q.reset(&xs)?;
        q.h(&xs)?;
        let mut z_results = Vec::new();
        for round in 1..=7 {
            let mut layer = x_layer(round).unwrap_or_default();
            layer.extend(z_layer(round).unwrap_or_default());
            q.cx(&layer)?;
            if round == 7 {
                z_results = q.measure(&zs)?;
            }
        }
        // Round 8: X ancillas measured in the X basis, Z ancillas reset.
        q.h(&xs)?;
        let x_results = q.measure(&xs)?;
        q.reset(&zs)?;
        let detected = if x_basis { x_results } else { z_results };
        for (c, &r) in detected.iter().enumerate() {
            match &previous {
                None => q.c.detector(&[r])?,
                Some(p) => q.c.detector(&[r, p[c]])?,
            };
        }
        previous = Some(detected);
    }
    let last = previous.expect("at least one round");
    if x_basis {
        q.h(&data)?;
    }
    let measured = q.measure(&data)?;
    let checks_of_basis = if x_basis {
        &code.x_checks
    } else {
        &code.z_checks
    };
    for (c, support) in checks_of_basis.iter().enumerate() {
        let mut recs = vec![last[c]];
        recs.extend(support.iter().map(|&d| measured[d]));
        q.c.detector(&recs)?;
    }
    let logicals = if x_basis {
        &code.x_logicals
    } else {
        &code.z_logicals
    };
    for (i, lg) in logicals.iter().enumerate() {
        q.c.observable(i, &lg.iter().map(|&d| measured[d]).collect::<Vec<_>>())?;
    }
    Ok(q.c)
}

fn build_css(
    code: &CssCode,
    rounds: usize,
    noise: &CircuitNoise,
    basis: MemoryBasis,
) -> Result<DetectorCircuit> {
    let x_layers = layers(&code.x_checks, code.n);
    let z_layers = layers(&code.z_checks, code.n);
    build_layered(code, rounds, noise, basis, &x_layers, &z_layers)
}

/// Rounds of every X check, then every Z check, with these CNOT layers of
/// (check, qubit).
fn build_layered(
    code: &CssCode,
    rounds: usize,
    noise: &CircuitNoise,
    basis: MemoryBasis,
    x_layers: &[Vec<(usize, usize)>],
    z_layers: &[Vec<(usize, usize)>],
) -> Result<DetectorCircuit> {
    let x_basis = basis == MemoryBasis::X;
    let n = code.n;
    let (rx, rz) = (code.x_checks.len(), code.z_checks.len());
    let total = n + rx + rz;
    let mut b = Noisy {
        c: DetectorCircuit::new(total)?,
        noise,
    };
    let data: Vec<usize> = (0..n).collect();
    let x_anc: Vec<usize> = (n..n + rx).collect();
    let all_anc: Vec<usize> = (n..total).collect();
    b.reset(&(0..total).collect::<Vec<_>>())?;
    if x_basis {
        b.h(&data)?;
    }
    let detected: &[Vec<usize>] = if x_basis {
        &code.x_checks
    } else {
        &code.z_checks
    };
    // Measurement index of detected check c within the round's results.
    let detected_offset = if x_basis { 0 } else { rx };
    let mut previous: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        b.c.noise(Noise::Depolarize1(noise.before_round), &data)?;
        if rx > 0 {
            b.h(&x_anc)?;
            for layer in x_layers {
                b.cx(&layer.iter().map(|&(c, q)| (n + c, q)).collect::<Vec<_>>())?;
            }
            b.h(&x_anc)?;
        }
        for layer in z_layers {
            b.cx(&layer
                .iter()
                .map(|&(c, q)| (q, n + rx + c))
                .collect::<Vec<_>>())?;
        }
        let m = b.measure(&all_anc)?;
        for c in 0..detected.len() {
            let r = m[detected_offset + c];
            match &previous {
                None => b.c.detector(&[r])?,
                Some(p) => b.c.detector(&[r, p[detected_offset + c]])?,
            };
        }
        b.reset(&all_anc)?;
        previous = Some(m);
    }
    let last = previous.expect("at least one round");
    if x_basis {
        b.h(&data)?;
    }
    let m = b.measure(&data)?;
    for (c, support) in detected.iter().enumerate() {
        let mut recs = vec![last[detected_offset + c]];
        recs.extend(support.iter().map(|&q| m[q]));
        b.c.detector(&recs)?;
    }
    let logicals = if x_basis {
        &code.x_logicals
    } else {
        &code.z_logicals
    };
    for (i, l) in logicals.iter().enumerate() {
        b.c.observable(i, &l.iter().map(|&q| m[q]).collect::<Vec<_>>())?;
    }
    Ok(b.c)
}

#[cfg(test)]
mod tests;
