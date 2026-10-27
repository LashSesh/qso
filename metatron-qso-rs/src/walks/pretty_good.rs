//! Pretty good state transfer, decided exactly.
//!
//! A continuous-time walk has *pretty good state transfer* (PGST) from `u`
//! to `v` if `|⟨v|e^{−iHt}|u⟩|` comes arbitrarily close to 1 at suitable
//! times. With the support `θ_0, …, θ_m` of `u` (the distinct eigenvalues
//! with `E_θ e_u ≠ 0`), the amplitude is `Σ_r σ_r ‖E_r e_u‖² e^{−iθ_r t}`
//! when `u` and `v` are strongly cospectral (`E_r e_v = σ_r E_r e_u`,
//! `σ_r = ±1`), and it nears modulus 1 exactly when all phases
//! `t(θ_r − θ_0)/2π` near `δ_r/2 (mod 1)`, `δ_r = 0` if `σ_r = σ_0` and 1
//! otherwise. By Kronecker's theorem such times exist if and only if
//!
//! ```text
//! Σ_{r ≥ 1} ℓ_r (θ_r − θ_0) = 0  with integers ℓ_r   ⟹   Σ_r ℓ_r δ_r is even
//! ```
//!
//! (Godsil 2012; Banchi, Coutinho, Godsil and Severini, J. Math. Phys. 58,
//! 032202 (2017)). The test needs the integer relations among the
//! eigenvalues exactly, which floating point cannot give.
//! [`PrettyGoodTransfer`] computes them for graphs whose eigenvalues lie
//! in a cyclotomic field `Q(ζ_N)` and are known in closed form: paths
//! (adjacency and Laplacian), cycles, and Cayley graphs of finite abelian
//! groups (circulants, hypercubes, tori, Hamming graphs). Every
//! eigenvalue is an integer vector in the power basis `1, ζ, …, ζ^{φ(N)−1}`
//! (powers of `ζ` reduced modulo the cyclotomic polynomial `Φ_N`); the
//! relations form the integer kernel of those vectors, found by integer
//! column reduction, and the condition is checked on a basis of it.
//!
//! The decision is exact. [`PrettyGoodTransfer::obstruction`] returns a
//! relation that violates the condition when there is no PGST, and
//! [`PrettyGoodTransfer::best_transfer`] searches a time window for a high
//! fidelity: PGST times exist but can be far beyond any window (on long
//! paths they grow quickly with the required fidelity).
//!
//! ```
//! use metatron_qso::walks::{Generator, PrettyGoodTransfer};
//!
//! // Ends of the path on n nodes: PGST exactly when n + 1 is a prime, twice
//! // a prime or a power of two (Godsil, Kirkland, Severini and Smith 2012).
//! for (n, pgst) in [(4, true), (5, true), (7, true), (8, false), (9, true), (14, false)] {
//!     let t = PrettyGoodTransfer::path(n, Generator::Adjacency, 1.0, 0, n - 1).unwrap();
//!     assert_eq!(t.pretty_good(), pgst, "P{n}");
//! }
//! ```

use std::collections::HashMap;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::search::optimize_peak;
use super::{Generator, TransferResult};
use crate::error::{QsoError, Result};
use crate::graph::Graph;

/// Largest order `N` of the cyclotomic field.
pub const MAX_CYCLOTOMIC_ORDER: usize = 1 << 13;
/// Largest group (number of nodes) of [`PrettyGoodTransfer::cayley`].
pub const MAX_CAYLEY_NODES: usize = 1 << 16;
/// Largest number of integers the relation search holds: with `m`
/// distinct eigenvalues in the support and field degree `d`, it keeps
/// `m (m + d)` entries of 16 bytes (here at most 256 MiB).
pub const MAX_RELATION_ENTRIES: usize = 1 << 24;
/// Range of the hopping rate γ: outside it the transfer times under- or
/// overflow.
pub const GAMMA_RANGE: (f64, f64) = (1e-100, 1e100);

/// Integers in the power basis of `Q(ζ_N)`.
struct Cyclotomic {
    order: usize,
    /// `ζ^k` for `k < N`, reduced modulo `Φ_N`.
    powers: Vec<Vec<i64>>,
}

fn poly_mul(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let mut out = vec![0i64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] = out[i + j].checked_add(x.checked_mul(y)?)?;
        }
    }
    Some(out)
}

/// Exact quotient of `a` by the monic `b` (coefficients low to high).
fn poly_div_exact(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let mut rem = a.to_vec();
    let db = b.len() - 1;
    if rem.len() < b.len() {
        return None;
    }
    let mut q = vec![0i64; rem.len() - db];
    for k in (0..q.len()).rev() {
        let c = rem[k + db];
        q[k] = c;
        if c != 0 {
            for (i, &bi) in b.iter().enumerate() {
                rem[k + i] = rem[k + i].checked_sub(c.checked_mul(bi)?)?;
            }
        }
    }
    rem.iter().all(|&x| x == 0).then_some(q)
}

fn mobius(mut n: usize) -> i32 {
    let mut result = 1;
    let mut p = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            n /= p;
            if n.is_multiple_of(p) {
                return 0;
            }
            result = -result;
        }
        p += 1;
    }
    if n > 1 {
        result = -result;
    }
    result
}

/// `Φ_n = Π_{d | n} (x^d − 1)^{μ(n/d)}`.
fn cyclotomic_polynomial(n: usize) -> Option<Vec<i64>> {
    let divisors: Vec<usize> = (1..=n).filter(|&d| n.is_multiple_of(d)).collect();
    let factor = |d: usize| {
        let mut f = vec![0i64; d + 1];
        f[0] = -1;
        f[d] = 1;
        f
    };
    let mut num = vec![1i64];
    for &d in &divisors {
        if mobius(n / d) == 1 {
            num = poly_mul(&num, &factor(d))?;
        }
    }
    for &d in &divisors {
        if mobius(n / d) == -1 {
            num = poly_div_exact(&num, &factor(d))?;
        }
    }
    Some(num)
}

impl Cyclotomic {
    fn new(order: usize) -> Result<Self> {
        if order == 0 || order > MAX_CYCLOTOMIC_ORDER {
            return Err(QsoError::invalid(
                "order",
                format!("cyclotomic order must lie in 1..={MAX_CYCLOTOMIC_ORDER}"),
            ));
        }
        let overflow = || QsoError::invalid("order", "integer overflow in cyclotomic arithmetic");
        let phi = cyclotomic_polynomial(order).ok_or_else(overflow)?;
        let deg = phi.len() - 1;
        let mut powers = Vec::with_capacity(order);
        let mut cur = vec![0i64; deg];
        cur[0] = 1;
        for _ in 0..order {
            powers.push(cur.clone());
            // x · cur, then x^deg = −Σ_{i<deg} Φ_i x^i.
            let carry = cur[deg - 1];
            let mut next = vec![0i64; deg];
            next[1..deg].copy_from_slice(&cur[..deg - 1]);
            for i in 0..deg {
                next[i] = next[i]
                    .checked_sub(carry.checked_mul(phi[i]).ok_or_else(overflow)?)
                    .ok_or_else(overflow)?;
            }
            cur = next;
        }
        Ok(Self { order, powers })
    }

    /// `Σ c·ζ^e` for `(c, e)` in `terms`.
    fn element(&self, terms: &[(i64, i64)]) -> Result<Vec<i64>> {
        let n = self.order as i64;
        let mut out = vec![0i64; self.powers[0].len()];
        for &(c, e) in terms {
            let p = &self.powers[e.rem_euclid(n) as usize];
            for (o, &x) in out.iter_mut().zip(p) {
                *o = c
                    .checked_mul(x)
                    .and_then(|v| o.checked_add(v))
                    .ok_or_else(|| {
                        QsoError::invalid("order", "integer overflow in cyclotomic arithmetic")
                    })?;
            }
        }
        Ok(out)
    }

    /// Numerical value of `Σ c·ζ^e` (real part; the eigenvalues are real).
    fn value(&self, terms: &[(i64, i64)]) -> f64 {
        let n = self.order as f64;
        terms
            .iter()
            .map(|&(c, e)| c as f64 * (std::f64::consts::TAU * e as f64 / n).cos())
            .sum()
    }
}

/// A basis of the integer relations `Σ_c ℓ_c w_c = 0` among the columns
/// `w_c`, and the rank of the columns; `None` on integer overflow. Column
/// reduction by extended gcds (fast), and on overflow by Euclid's
/// remainders with the smallest pivot (slower, smaller entries).
fn integer_kernel(columns: &[Vec<i64>]) -> Option<(Vec<Vec<i128>>, usize)> {
    integer_kernel_with(columns, false).or_else(|| integer_kernel_with(columns, true))
}

fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    // Returns (g, x, y) with a x + b y = g ≥ 0.
    let (mut r0, mut r1) = (a, b);
    let (mut s0, mut s1) = (1i128, 0i128);
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0.div_euclid(r1);
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if r0 < 0 {
        (-r0, -s0, -t0)
    } else {
        (r0, s0, t0)
    }
}

fn integer_kernel_with(columns: &[Vec<i64>], euclid: bool) -> Option<(Vec<Vec<i128>>, usize)> {
    let m = columns.len();
    if m == 0 {
        return Some((Vec::new(), 0));
    }
    let d = columns[0].len();
    let mut a: Vec<Vec<i128>> = columns
        .iter()
        .map(|c| c.iter().map(|&x| x as i128).collect())
        .collect();
    let mut u: Vec<Vec<i128>> = (0..m)
        .map(|c| {
            let mut e = vec![0i128; m];
            e[c] = 1;
            e
        })
        .collect();
    let combine = |x: &[i128], y: &[i128], cx: i128, cy: i128| -> Option<Vec<i128>> {
        x.iter()
            .zip(y)
            .map(|(&p, &q)| p.checked_mul(cx)?.checked_add(q.checked_mul(cy)?))
            .collect()
    };
    let mut pivot = 0;
    for row in 0..d {
        if pivot == m {
            break;
        }
        if !euclid {
            for c in pivot + 1..m {
                if a[c][row] == 0 {
                    continue;
                }
                if a[pivot][row] == 0 {
                    a.swap(pivot, c);
                    u.swap(pivot, c);
                    continue;
                }
                let (g, x, y) = ext_gcd(a[pivot][row], a[c][row]);
                let (p, q) = (a[pivot][row] / g, a[c][row] / g);
                // [x y; −q p] has determinant xp + yq = 1.
                let new_pivot = combine(&a[pivot], &a[c], x, y)?;
                let new_c = combine(&a[pivot], &a[c], -q, p)?;
                let u_pivot = combine(&u[pivot], &u[c], x, y)?;
                let u_c = combine(&u[pivot], &u[c], -q, p)?;
                a[pivot] = new_pivot;
                a[c] = new_c;
                u[pivot] = u_pivot;
                u[c] = u_c;
            }
            if a[pivot][row] != 0 {
                pivot += 1;
            }
            continue;
        }
        // Euclid on the columns: the smallest nonzero entry of the row
        // becomes the pivot and reduces the others to their remainders,
        // which keeps the entries small.
        loop {
            let best = (pivot..m)
                .filter(|&c| a[c][row] != 0)
                .min_by_key(|&c| a[c][row].unsigned_abs());
            let Some(best) = best else { break };
            a.swap(pivot, best);
            u.swap(pivot, best);
            let mut done = true;
            for c in pivot + 1..m {
                if a[c][row] == 0 {
                    continue;
                }
                let q = a[c][row].div_euclid(a[pivot][row]);
                a[c] = combine(&a[c], &a[pivot], 1, -q)?;
                u[c] = combine(&u[c], &u[pivot], 1, -q)?;
                if a[c][row] != 0 {
                    done = false;
                }
            }
            if done {
                pivot += 1;
                break;
            }
        }
    }
    Some((u.split_off(pivot), pivot))
}

/// One eigenvalue in the support of the source node.
#[derive(Clone, Debug)]
struct Level {
    /// Exact coordinates in `Q(ζ_N)`.
    exact: Vec<i64>,
    /// Eigenvalue of the Hamiltonian (with γ and the generator's sign).
    energy: f64,
    /// `‖E_θ e_u‖²`.
    weight: f64,
    /// `⟨e_v|E_θ|e_u⟩` (real for these graphs).
    cross: f64,
    /// `σ_θ` (0: not strongly cospectral on this eigenspace).
    sign: i8,
}

/// Exact decision of pretty good (and perfect) state transfer between two
/// nodes (see the module documentation).
#[derive(Clone, Debug)]
pub struct PrettyGoodTransfer {
    levels: Vec<Level>,
    cospectral: bool,
    /// Whether a relation violates the parity condition.
    obstructed: bool,
    /// That relation, when its coefficients fit in `i64`.
    obstruction: Option<Vec<i64>>,
    /// Rank of the eigenvalue differences over `Q`.
    rank: usize,
    perfect_time: Option<f64>,
}

/// Summary of a [`PrettyGoodTransfer`] decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferDecision {
    pub strongly_cospectral: bool,
    pub pretty_good: bool,
    /// Earliest time of perfect state transfer, if there is one.
    pub perfect_time: Option<f64>,
    /// Dimension over `Q` of the span of the eigenvalue differences (0
    /// when the nodes are not strongly cospectral).
    pub rank: usize,
}

fn check_gamma(gamma: f64) -> Result<()> {
    if !(GAMMA_RANGE.0..=GAMMA_RANGE.1).contains(&gamma) {
        return Err(QsoError::invalid(
            "gamma",
            format!(
                "the hopping rate must lie between {:e} and {:e}",
                GAMMA_RANGE.0, GAMMA_RANGE.1
            ),
        ));
    }
    Ok(())
}

/// `sin(πp/m) = ±sin(πq/m)` or neither, exactly: `Some(±1)` or `None`.
fn sine_ratio(p: i64, q: i64, m: i64) -> Option<i8> {
    let r = |x: i64| x.rem_euclid(2 * m);
    if r(p - q) == 0 || r(p + q - m) == 0 {
        Some(1)
    } else if r(p + q) == 0 || r(p - q - m) == 0 {
        Some(-1)
    } else {
        None
    }
}

/// `cos(πp/m) = ±cos(πq/m)` or neither, exactly.
fn cosine_ratio(p: i64, q: i64, m: i64) -> Option<i8> {
    let r = |x: i64| x.rem_euclid(2 * m);
    if r(p - q) == 0 || r(p + q) == 0 {
        Some(1)
    } else if r(p - q - m) == 0 || r(p + q - m) == 0 {
        Some(-1)
    } else {
        None
    }
}

impl PrettyGoodTransfer {
    /// Transfer between nodes `u` and `v` of the path on `n` nodes (node
    /// `x` adjacent to `x ± 1`).
    pub fn path(n: usize, generator: Generator, gamma: f64, u: usize, v: usize) -> Result<Self> {
        check_gamma(gamma)?;
        if n < 2 {
            return Err(QsoError::invalid("n", "a path needs at least 2 nodes"));
        }
        if n >= MAX_CYCLOTOMIC_ORDER / 2 {
            return Err(QsoError::invalid(
                "n",
                format!(
                    "at most {} nodes (the eigenvalues lie in a cyclotomic field of order 2(n + 1) ≤ {MAX_CYCLOTOMIC_ORDER})",
                    MAX_CYCLOTOMIC_ORDER / 2 - 1
                ),
            ));
        }
        for node in [u, v] {
            if node >= n {
                return Err(QsoError::NodeOutOfRange { node, n_nodes: n });
            }
        }
        if u == v {
            return Err(QsoError::invalid("v", "source and target must differ"));
        }
        let (a, b) = (u as i64, v as i64);
        let n_ = n as i64;
        let mut levels = Vec::new();
        match generator {
            Generator::Adjacency => {
                // θ_j = 2cos(πj/(n+1)) = ζ^j + ζ^{−j}, ζ = ζ_{2(n+1)};
                // eigenvectors sin(πj(x+1)/(n+1)), all eigenvalues simple.
                let m = n_ + 1;
                let field = Cyclotomic::new(2 * (n + 1))?;
                for j in 1..=n_ {
                    let (p, q) = (j * (a + 1), j * (b + 1));
                    let su = (std::f64::consts::PI * p as f64 / m as f64).sin();
                    let sv = (std::f64::consts::PI * q as f64 / m as f64).sin();
                    let zero_u = p.rem_euclid(m) == 0;
                    let zero_v = q.rem_euclid(m) == 0;
                    if zero_u {
                        if !zero_v {
                            levels.push(Level {
                                exact: Vec::new(),
                                energy: 0.0,
                                weight: 0.0,
                                cross: 0.0,
                                sign: 0,
                            });
                        }
                        continue;
                    }
                    let terms = [(1, j), (1, -j)];
                    levels.push(Level {
                        exact: field.element(&terms)?,
                        energy: -gamma * field.value(&terms),
                        weight: 2.0 / m as f64 * su * su,
                        cross: 2.0 / m as f64 * su * sv,
                        sign: sine_ratio(p, q, m).unwrap_or(0),
                    });
                }
            }
            Generator::Laplacian => {
                // λ_j = 2 − 2cos(πj/n) = 2 − ζ^j − ζ^{−j}, ζ = ζ_{2n};
                // eigenvectors cos(πj(2x+1)/(2n)).
                let m = 2 * n_;
                let field = Cyclotomic::new(2 * n)?;
                for j in 0..n_ {
                    let (p, q) = (j * (2 * a + 1), j * (2 * b + 1));
                    let cu = (std::f64::consts::PI * p as f64 / m as f64).cos();
                    let cv = (std::f64::consts::PI * q as f64 / m as f64).cos();
                    let zero = |x: i64| (2 * x).rem_euclid(2 * m) == m;
                    if zero(p) {
                        if !zero(q) {
                            levels.push(Level {
                                exact: Vec::new(),
                                energy: 0.0,
                                weight: 0.0,
                                cross: 0.0,
                                sign: 0,
                            });
                        }
                        continue;
                    }
                    let terms = [(2, 0), (-1, j), (-1, -j)];
                    let norm = if j == 0 {
                        1.0 / n as f64
                    } else {
                        2.0 / n as f64
                    };
                    levels.push(Level {
                        exact: field.element(&terms)?,
                        energy: gamma * field.value(&terms),
                        weight: norm * cu * cu,
                        cross: norm * cu * cv,
                        sign: cosine_ratio(p, q, m).unwrap_or(0),
                    });
                }
            }
        }
        Self::decide(levels)
    }

    /// Transfer between `u` and `v` on the cycle with `n ≥ 3` nodes.
    pub fn cycle(n: usize, generator: Generator, gamma: f64, u: usize, v: usize) -> Result<Self> {
        if n < 3 {
            return Err(QsoError::invalid("n", "a cycle needs at least 3 nodes"));
        }
        if n > MAX_CYCLOTOMIC_ORDER {
            return Err(QsoError::invalid(
                "n",
                format!("at most {MAX_CYCLOTOMIC_ORDER} nodes"),
            ));
        }
        Self::cayley(
            &[n],
            &[vec![1], vec![n as i64 - 1]],
            generator,
            gamma,
            &[u],
            &[v],
        )
    }

    /// Transfer between the elements `u` and `v` of the Cayley graph of
    /// `Z_{m_1} × … × Z_{m_k}` (`moduli`) whose edges join `g` and `g + s`
    /// for every `s` in the symmetric `connection` set (circulants for one
    /// modulus, the hypercube for `[2; d]` with the unit vectors).
    pub fn cayley(
        moduli: &[usize],
        connection: &[Vec<i64>],
        generator: Generator,
        gamma: f64,
        u: &[usize],
        v: &[usize],
    ) -> Result<Self> {
        check_gamma(gamma)?;
        let k = moduli.len();
        if k == 0 || moduli.iter().any(|&m| m < 2) {
            return Err(QsoError::invalid(
                "moduli",
                "at least one modulus, each at least 2",
            ));
        }
        let size = moduli
            .iter()
            .try_fold(1usize, |acc, &m| acc.checked_mul(m))
            .filter(|&s| s <= MAX_CAYLEY_NODES)
            .ok_or_else(|| {
                QsoError::invalid(
                    "moduli",
                    format!("at most {MAX_CAYLEY_NODES} group elements"),
                )
            })?;
        for (name, x) in [("u", u), ("v", v)] {
            if x.len() != k || x.iter().zip(moduli).any(|(&a, &m)| a >= m) {
                return Err(QsoError::invalid(
                    name,
                    "elements need one coordinate below each modulus",
                ));
            }
        }
        if u == v {
            return Err(QsoError::invalid("v", "source and target must differ"));
        }
        let reduce = |s: &[i64]| -> Vec<i64> {
            s.iter()
                .zip(moduli)
                .map(|(&x, &m)| x.rem_euclid(m as i64))
                .collect()
        };
        let set: Vec<Vec<i64>> = connection.iter().map(|s| reduce(s)).collect();
        if set.iter().any(|s| s.len() != k) {
            return Err(QsoError::invalid(
                "connection",
                "elements need one coordinate per modulus",
            ));
        }
        if set.iter().any(|s| s.iter().all(|&x| x == 0)) {
            return Err(QsoError::invalid(
                "connection",
                "the identity would be a loop",
            ));
        }
        if set.is_empty() {
            return Err(QsoError::invalid("connection", "the set is empty"));
        }
        let mut sorted = set.clone();
        sorted.sort();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(QsoError::invalid(
                "connection",
                "an element is listed twice (after reduction modulo the moduli)",
            ));
        }
        let mut negated: Vec<Vec<i64>> = set
            .iter()
            .map(|s| reduce(&s.iter().map(|x| -x).collect::<Vec<_>>()))
            .collect();
        negated.sort();
        if sorted != negated {
            return Err(QsoError::invalid(
                "connection",
                "the set must be symmetric (s and −s)",
            ));
        }
        // Q(ζ_N) for the exponent N of the group (for odd N it equals
        // Q(ζ_2N), and characters never take the value −1).
        let order = moduli.iter().fold(1usize, |l, &m| l / gcd(l, m) * m);
        if order > MAX_CYCLOTOMIC_ORDER {
            return Err(QsoError::invalid(
                "moduli",
                format!(
                    "the exponent of the group (the lcm of the moduli, {order}) must be at most {MAX_CYCLOTOMIC_ORDER}"
                ),
            ));
        }
        let field = Cyclotomic::new(order)?;
        let scale: Vec<i64> = moduli.iter().map(|&m| (order / m) as i64).collect();
        let degree = set.len() as i64;
        let diff: Vec<i64> = v
            .iter()
            .zip(u)
            .zip(moduli)
            .map(|((&b, &a), &m)| (b as i64 - a as i64).rem_euclid(m as i64))
            .collect();
        // Characters χ_c(g) = ζ^{Σ c_i g_i N/m_i}; eigenvalue Σ_s χ_c(s).
        let mut groups: HashMap<Vec<i64>, (f64, usize, i8, f64)> = HashMap::new();
        let mut order_seen: Vec<Vec<i64>> = Vec::new();
        let mut c = vec![0usize; k];
        for _ in 0..size {
            let exponent = |g: &[i64]| -> i64 {
                g.iter()
                    .zip(&c)
                    .zip(&scale)
                    .map(|((&gi, &ci), &si)| gi * ci as i64 * si)
                    .sum::<i64>()
                    .rem_euclid(order as i64)
            };
            let mut terms: Vec<(i64, i64)> = set.iter().map(|s| (1, exponent(s))).collect();
            if generator == Generator::Laplacian {
                terms.push((-degree, 0));
                terms.iter_mut().for_each(|t| t.0 = -t.0);
            }
            let exact = field.element(&terms)?;
            let energy = match generator {
                Generator::Adjacency => -gamma * field.value(&terms),
                Generator::Laplacian => gamma * field.value(&terms),
            };
            let e = exponent(&diff);
            let sign = if e == 0 {
                1
            } else if 2 * e == order as i64 {
                -1
            } else {
                0
            };
            // ⟨e_v|E|e_u⟩ sums χ_c(v − u)/|G| over the eigenspace; the
            // imaginary parts cancel (c and −c share the eigenvalue).
            let cross = (2.0 * std::f64::consts::PI * e as f64 / order as f64).cos();
            match groups.get_mut(&exact) {
                Some(entry) => {
                    entry.1 += 1;
                    entry.3 += cross;
                    if entry.2 != sign {
                        entry.2 = 0;
                    }
                }
                None => {
                    order_seen.push(exact.clone());
                    groups.insert(exact, (energy, 1, sign, cross));
                }
            }
            // Next character (mixed radix).
            for (ci, &m) in c.iter_mut().zip(moduli) {
                *ci += 1;
                if *ci < m {
                    break;
                }
                *ci = 0;
            }
        }
        let levels = order_seen
            .into_iter()
            .map(|exact| {
                let (energy, count, sign, cross) = groups[&exact];
                Level {
                    exact,
                    energy,
                    weight: count as f64 / size as f64,
                    cross: cross / size as f64,
                    sign,
                }
            })
            .collect();
        Self::decide(levels)
    }

    /// Transfer on a graph that is a path or a cycle (recognised from its
    /// structure, nodes in any order); other graphs are not decided.
    pub fn from_graph(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        u: usize,
        v: usize,
    ) -> Result<Self> {
        let n = graph.num_nodes();
        for node in [u, v] {
            if node >= n {
                return Err(QsoError::NodeOutOfRange { node, n_nodes: n });
            }
        }
        let unsupported = || {
            QsoError::invalid(
                "graph",
                "exact decisions cover paths and cycles here (and Cayley graphs of abelian groups through PrettyGoodTransfer::cayley)",
            )
        };
        if !graph.is_unweighted() || !graph.is_connected() || n < 2 {
            return Err(unsupported());
        }
        let degrees = graph.degrees();
        let is_path = graph.num_edges() == n - 1 && degrees.iter().all(|&d| d <= 2);
        let is_cycle = n >= 3 && graph.num_edges() == n && degrees.iter().all(|&d| d == 2);
        if !(is_path || is_cycle) {
            return Err(unsupported());
        }
        // Walk the graph from an end (or node 0 for a cycle) to number it.
        let start = if is_path {
            degrees
                .iter()
                .position(|&d| d == 1)
                .ok_or_else(unsupported)?
        } else {
            0
        };
        let mut position = vec![usize::MAX; n];
        let (mut prev, mut cur) = (usize::MAX, start);
        for p in 0..n {
            position[cur] = p;
            let next = graph
                .neighbors(cur)
                .iter()
                .map(|&(w, _)| w)
                .find(|&w| w != prev && position[w] == usize::MAX);
            match next {
                Some(w) => (prev, cur) = (cur, w),
                None => break,
            }
        }
        if is_path {
            Self::path(n, generator, gamma, position[u], position[v])
        } else {
            Self::cycle(n, generator, gamma, position[u], position[v])
        }
    }

    fn decide(levels: Vec<Level>) -> Result<Self> {
        let cospectral =
            !levels.is_empty() && levels.iter().all(|l| l.sign != 0 && !l.exact.is_empty());
        let mut out = Self {
            levels,
            cospectral,
            obstructed: false,
            obstruction: None,
            rank: 0,
            perfect_time: None,
        };
        if !cospectral {
            return Ok(out);
        }
        let (m, d) = (out.levels.len() - 1, out.levels[0].exact.len());
        if m.saturating_mul(m + d) > MAX_RELATION_ENTRIES {
            return Err(QsoError::invalid(
                "graph",
                format!(
                    "{} distinct eigenvalues in a field of degree {d}: the relation search would hold {} integers, more than {MAX_RELATION_ENTRIES}",
                    m + 1,
                    m.saturating_mul(m + d)
                ),
            ));
        }
        let base = out.levels[0].exact.clone();
        let columns: Vec<Vec<i64>> = out.levels[1..]
            .iter()
            .map(|l| l.exact.iter().zip(&base).map(|(x, y)| x - y).collect())
            .collect();
        let (kernel, rank) = integer_kernel(&columns).ok_or_else(|| {
            QsoError::invalid(
                "graph",
                "the integer relations among the eigenvalues overflow 128-bit integers; the pair is not decided",
            )
        })?;
        out.rank = rank;
        let delta: Vec<i128> = out.levels[1..]
            .iter()
            .map(|l| i128::from(l.sign != out.levels[0].sign))
            .collect();
        for relation in &kernel {
            let parity: i128 = relation.iter().zip(&delta).map(|(l, d)| l * d).sum();
            if parity.rem_euclid(2) == 1 {
                out.obstructed = true;
                out.obstruction = relation
                    .iter()
                    .map(|&x| i64::try_from(x).ok())
                    .collect::<Option<Vec<i64>>>();
                break;
            }
        }
        if !out.obstructed && rank <= 1 && out.levels.len() >= 2 {
            // All differences are rational multiples of θ_1 − θ_0: PST at
            // π/g with g their greatest common divisor (as a real number).
            let i = columns[0].iter().position(|&x| x != 0).expect("rank 1");
            let d1 = out.levels[1].energy - out.levels[0].energy;
            let den = columns[0][i];
            // Differences D_r = (columns[r][i] / den)·D_1; g = D_1·gcd(num)/den.
            let g_num = columns.iter().fold(0i64, |g, c| gcd_i64(g, c[i]));
            // Without an obstruction, each D_r/g is an integer of the
            // parity of δ_r: PST at π/g, exactly, and no earlier (some
            // integer combination of the D_r is g).
            let g = (d1 / den as f64 * g_num as f64).abs();
            out.perfect_time = Some(std::f64::consts::PI / g);
        }
        Ok(out)
    }

    /// Whether the two nodes are strongly cospectral.
    pub fn strongly_cospectral(&self) -> bool {
        self.cospectral
    }

    /// Whether there is pretty good state transfer (exact).
    pub fn pretty_good(&self) -> bool {
        self.cospectral && !self.obstructed
    }

    /// Earliest time of perfect state transfer, if there is one.
    pub fn perfect_time(&self) -> Option<f64> {
        self.perfect_time
    }

    /// Dimension over `Q` of the span of the differences `θ_r − θ_0` (0
    /// when the nodes are not strongly cospectral).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// An integer relation `Σ_{r ≥ 1} ℓ_r (θ_r − θ_0) = 0` with odd
    /// `Σ ℓ_r δ_r`, which rules out PGST (strongly cospectral nodes only;
    /// coefficient `i` belongs to entry `i + 1` of
    /// [`PrettyGoodTransfer::support`], whose entry 0 is `θ_0`).
    /// `None` also when the relation found has coefficients beyond `i64`;
    /// [`PrettyGoodTransfer::pretty_good`] is decided either way.
    pub fn obstruction(&self) -> Option<&[i64]> {
        self.obstruction.as_deref()
    }

    /// Eigenvalues of the Hamiltonian in the support (ascending order not
    /// guaranteed) and their signs `σ_r`.
    pub fn support(&self) -> Vec<(f64, i8)> {
        self.levels
            .iter()
            .filter(|l| l.weight > 0.0)
            .map(|l| (l.energy, l.sign))
            .collect()
    }

    /// The decision as plain data.
    pub fn decision(&self) -> TransferDecision {
        TransferDecision {
            strongly_cospectral: self.cospectral,
            pretty_good: self.pretty_good(),
            perfect_time: self.perfect_time,
            rank: self.rank,
        }
    }

    /// Transfer fidelity `|⟨v|e^{−iHt}|u⟩|² = |Σ_θ ⟨v|E_θ|u⟩ e^{−iθt}|²`,
    /// from the closed-form eigenvectors (any pair of nodes).
    pub fn fidelity(&self, t: f64) -> f64 {
        let a: C64 = self
            .levels
            .iter()
            .map(|l| C64::from_polar(l.cross, -l.energy * t))
            .sum();
        a.norm_sqr()
    }

    /// Highest fidelity over `t ∈ [0, t_max]` (grid scan with `resolution`
    /// points, then golden-section refinement).
    pub fn best_transfer(&self, t_max: f64, resolution: usize) -> Result<TransferResult> {
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(QsoError::invalid("t_max", "must be finite and positive"));
        }
        let r = optimize_peak(0.0, t_max, resolution, |t| self.fidelity(t));
        Ok(TransferResult {
            time: r.time,
            fidelity: r.success_probability,
        })
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests;
