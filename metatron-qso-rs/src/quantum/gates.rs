//! The standard gate set and its matrices.
//!
//! Conventions (identical to OpenQASM / Qiskit):
//!
//! * `RX(θ) = exp(−iθX/2)`, `RY(θ) = exp(−iθY/2)`, `RZ(θ) = exp(−iθZ/2)`
//! * `P(λ) = diag(1, e^{iλ})`
//! * `U(θ, φ, λ) = [[cos θ/2, −e^{iλ} sin θ/2], [e^{iφ} sin θ/2, e^{i(φ+λ)} cos θ/2]]`
//! * `RXX(θ) = exp(−iθ X⊗X/2)`, likewise `RYY`, `RZZ`
//!
//! Controlled variants are not separate kinds: any gate can carry control
//! qubits (see [`crate::quantum::Instruction`]).

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

/// Base gate kinds. Controls are attached separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateKind {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    /// √X.
    SX,
    /// (√X)†.
    SXdg,
    RX,
    RY,
    RZ,
    /// Phase gate `diag(1, e^{iλ})`.
    P,
    /// Generic single-qubit gate `U(θ, φ, λ)`.
    U,
    Swap,
    RXX,
    RYY,
    RZZ,
    /// User-supplied unitary on one or more targets.
    Unitary,
}

impl GateKind {
    /// Number of target qubits (`None` for [`GateKind::Unitary`], which is
    /// determined by its matrix).
    pub fn num_targets(self) -> Option<usize> {
        use GateKind::*;
        match self {
            Swap | RXX | RYY | RZZ => Some(2),
            Unitary => None,
            _ => Some(1),
        }
    }

    /// Number of real parameters.
    pub fn num_params(self) -> usize {
        use GateKind::*;
        match self {
            RX | RY | RZ | P | RXX | RYY | RZZ => 1,
            U => 3,
            _ => 0,
        }
    }

    /// Lower-case name (OpenQASM spelling where one exists).
    pub fn name(self) -> &'static str {
        use GateKind::*;
        match self {
            H => "h",
            X => "x",
            Y => "y",
            Z => "z",
            S => "s",
            Sdg => "sdg",
            T => "t",
            Tdg => "tdg",
            SX => "sx",
            SXdg => "sxdg",
            RX => "rx",
            RY => "ry",
            RZ => "rz",
            P => "p",
            U => "u",
            Swap => "swap",
            RXX => "rxx",
            RYY => "ryy",
            RZZ => "rzz",
            Unitary => "unitary",
        }
    }

    /// Whether the gate is diagonal in the computational basis.
    pub fn is_diagonal(self) -> bool {
        use GateKind::*;
        matches!(self, Z | S | Sdg | T | Tdg | RZ | P | RZZ)
    }
}

/// `cos(θ/2)·I − i·sin(θ/2)·G` coefficients for `exp(−iθG/2)` with `G² = I`.
#[inline]
fn half_angle(theta: f64) -> (f64, f64) {
    ((theta / 2.0).cos(), (theta / 2.0).sin())
}

/// 2×2 matrix (row-major) of a single-qubit gate kind.
pub(crate) fn matrix_1q(kind: GateKind, p: &[f64]) -> [C64; 4] {
    use GateKind::*;
    let r = std::f64::consts::FRAC_1_SQRT_2;
    match kind {
        H => [
            C64::new(r, 0.0),
            C64::new(r, 0.0),
            C64::new(r, 0.0),
            C64::new(-r, 0.0),
        ],
        X => [ZERO, ONE, ONE, ZERO],
        Y => [ZERO, -I, I, ZERO],
        Z => [ONE, ZERO, ZERO, -ONE],
        S => [ONE, ZERO, ZERO, I],
        Sdg => [ONE, ZERO, ZERO, -I],
        T => [
            ONE,
            ZERO,
            ZERO,
            C64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ],
        Tdg => [
            ONE,
            ZERO,
            ZERO,
            C64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        ],
        SX => {
            let a = C64::new(0.5, 0.5);
            let b = C64::new(0.5, -0.5);
            [a, b, b, a]
        }
        SXdg => {
            let a = C64::new(0.5, -0.5);
            let b = C64::new(0.5, 0.5);
            [a, b, b, a]
        }
        RX => {
            let (c, s) = half_angle(p[0]);
            [
                C64::new(c, 0.0),
                C64::new(0.0, -s),
                C64::new(0.0, -s),
                C64::new(c, 0.0),
            ]
        }
        RY => {
            let (c, s) = half_angle(p[0]);
            [
                C64::new(c, 0.0),
                C64::new(-s, 0.0),
                C64::new(s, 0.0),
                C64::new(c, 0.0),
            ]
        }
        RZ => [
            C64::from_polar(1.0, -p[0] / 2.0),
            ZERO,
            ZERO,
            C64::from_polar(1.0, p[0] / 2.0),
        ],
        P => [ONE, ZERO, ZERO, C64::from_polar(1.0, p[0])],
        U => {
            let (theta, phi, lambda) = (p[0], p[1], p[2]);
            let (c, s) = half_angle(theta);
            [
                C64::new(c, 0.0),
                -C64::from_polar(s, lambda),
                C64::from_polar(s, phi),
                C64::from_polar(c, phi + lambda),
            ]
        }
        Swap | RXX | RYY | RZZ | Unitary => {
            unreachable!("matrix_1q called for multi-qubit kind {kind:?}")
        }
    }
}

/// Derivative of [`matrix_1q`] with respect to parameter `which`.
pub(crate) fn dmatrix_1q(kind: GateKind, p: &[f64], which: usize) -> [C64; 4] {
    use GateKind::*;
    match kind {
        RX => {
            let (c, s) = half_angle(p[0]);
            [
                C64::new(-s / 2.0, 0.0),
                C64::new(0.0, -c / 2.0),
                C64::new(0.0, -c / 2.0),
                C64::new(-s / 2.0, 0.0),
            ]
        }
        RY => {
            let (c, s) = half_angle(p[0]);
            [
                C64::new(-s / 2.0, 0.0),
                C64::new(-c / 2.0, 0.0),
                C64::new(c / 2.0, 0.0),
                C64::new(-s / 2.0, 0.0),
            ]
        }
        RZ => [
            C64::from_polar(0.5, -p[0] / 2.0) * -I,
            ZERO,
            ZERO,
            C64::from_polar(0.5, p[0] / 2.0) * I,
        ],
        P => [ZERO, ZERO, ZERO, C64::from_polar(1.0, p[0]) * I],
        U => {
            let (theta, phi, lambda) = (p[0], p[1], p[2]);
            let (c, s) = half_angle(theta);
            match which {
                0 => [
                    C64::new(-s / 2.0, 0.0),
                    -C64::from_polar(c / 2.0, lambda),
                    C64::from_polar(c / 2.0, phi),
                    C64::from_polar(-s / 2.0, phi + lambda),
                ],
                1 => [
                    ZERO,
                    ZERO,
                    C64::from_polar(s, phi) * I,
                    C64::from_polar(c, phi + lambda) * I,
                ],
                2 => [
                    ZERO,
                    -C64::from_polar(s, lambda) * I,
                    ZERO,
                    C64::from_polar(c, phi + lambda) * I,
                ],
                _ => unreachable!("U has three parameters"),
            }
        }
        _ => unreachable!("gate kind {kind:?} has no parameters"),
    }
}

/// Two-qubit Pauli products `X⊗X`, `Y⊗Y`, `Z⊗Z` as 4×4 matrices.
fn pauli_pair(kind: GateKind) -> [C64; 16] {
    let mut m = [ZERO; 16];
    match kind {
        GateKind::RXX => {
            m[3] = ONE;
            m[6] = ONE;
            m[9] = ONE;
            m[12] = ONE;
        }
        GateKind::RYY => {
            // Y⊗Y = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
            m[3] = -ONE;
            m[6] = ONE;
            m[9] = ONE;
            m[12] = -ONE;
        }
        GateKind::RZZ => {
            m[0] = ONE;
            m[5] = -ONE;
            m[10] = -ONE;
            m[15] = ONE;
        }
        _ => unreachable!(),
    }
    m
}

/// 4×4 matrix of a two-qubit gate kind, local order `|x_first x_second⟩`.
pub(crate) fn matrix_2q(kind: GateKind, p: &[f64]) -> [C64; 16] {
    match kind {
        GateKind::Swap => {
            let mut m = [ZERO; 16];
            m[0] = ONE;
            m[6] = ONE;
            m[9] = ONE;
            m[15] = ONE;
            m
        }
        GateKind::RXX | GateKind::RYY | GateKind::RZZ => {
            let (c, s) = half_angle(p[0]);
            let g = pauli_pair(kind);
            let mut m = [ZERO; 16];
            for (k, entry) in m.iter_mut().enumerate() {
                let id = if k % 5 == 0 { c } else { 0.0 };
                *entry = C64::new(id, 0.0) + C64::new(0.0, -s) * g[k];
            }
            m
        }
        _ => unreachable!("matrix_2q called for {kind:?}"),
    }
}

/// Derivative of [`matrix_2q`] with respect to its (single) parameter.
pub(crate) fn dmatrix_2q(kind: GateKind, p: &[f64]) -> [C64; 16] {
    match kind {
        GateKind::RXX | GateKind::RYY | GateKind::RZZ => {
            let (c, s) = half_angle(p[0]);
            let g = pauli_pair(kind);
            let mut m = [ZERO; 16];
            for (k, entry) in m.iter_mut().enumerate() {
                let id = if k % 5 == 0 { -s / 2.0 } else { 0.0 };
                *entry = C64::new(id, 0.0) + C64::new(0.0, -c / 2.0) * g[k];
            }
            m
        }
        _ => unreachable!("gate kind {kind:?} has no parameters"),
    }
}

/// Conjugate transpose of a square row-major matrix.
pub(crate) fn dagger(m: &[C64]) -> Vec<C64> {
    let d = (m.len() as f64).sqrt().round() as usize;
    let mut out = vec![ZERO; m.len()];
    for r in 0..d {
        for c in 0..d {
            out[c * d + r] = m[r * d + c].conj();
        }
    }
    out
}

/// Check that a square row-major matrix is unitary to tolerance `tol`.
pub(crate) fn is_unitary(m: &[C64], tol: f64) -> bool {
    let d = (m.len() as f64).sqrt().round() as usize;
    if d * d != m.len() || !d.is_power_of_two() {
        return false;
    }
    for r in 0..d {
        for c in 0..d {
            let mut acc = ZERO;
            for k in 0..d {
                acc += m[k * d + r].conj() * m[k * d + c];
            }
            let expected = if r == c { ONE } else { ZERO };
            if (acc - expected).norm() > tol {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_1Q: [GateKind; 15] = [
        GateKind::H,
        GateKind::X,
        GateKind::Y,
        GateKind::Z,
        GateKind::S,
        GateKind::Sdg,
        GateKind::T,
        GateKind::Tdg,
        GateKind::SX,
        GateKind::SXdg,
        GateKind::RX,
        GateKind::RY,
        GateKind::RZ,
        GateKind::P,
        GateKind::U,
    ];

    fn params(kind: GateKind) -> Vec<f64> {
        [0.731, -1.29, 2.05][..kind.num_params()].to_vec()
    }

    fn mul2(a: &[C64; 4], b: &[C64; 4]) -> [C64; 4] {
        [
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ]
    }

    fn close(a: &[C64], b: &[C64], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
    }

    #[test]
    fn all_gates_are_unitary() {
        for kind in ALL_1Q {
            assert!(
                is_unitary(&matrix_1q(kind, &params(kind)), 1e-12),
                "{kind:?}"
            );
        }
        for kind in [GateKind::Swap, GateKind::RXX, GateKind::RYY, GateKind::RZZ] {
            assert!(
                is_unitary(&matrix_2q(kind, &params(kind)), 1e-12),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn algebraic_identities() {
        let h = matrix_1q(GateKind::H, &[]);
        let x = matrix_1q(GateKind::X, &[]);
        let z = matrix_1q(GateKind::Z, &[]);
        let s = matrix_1q(GateKind::S, &[]);
        let t = matrix_1q(GateKind::T, &[]);
        let sx = matrix_1q(GateKind::SX, &[]);
        assert!(close(&mul2(&mul2(&h, &z), &h), &x, 1e-12));
        assert!(close(&mul2(&t, &t), &s, 1e-12));
        assert!(close(&mul2(&s, &s), &z, 1e-12));
        assert!(close(&mul2(&sx, &sx), &x, 1e-12));
        // RX(π) = −iX, U(θ,−π/2,π/2) = RX(θ).
        let rx_pi = matrix_1q(GateKind::RX, &[std::f64::consts::PI]);
        assert!(close(&rx_pi, &x.map(|v| v * -I), 1e-12));
        let theta = 0.83;
        let u = matrix_1q(
            GateKind::U,
            &[
                theta,
                -std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_2,
            ],
        );
        assert!(close(&u, &matrix_1q(GateKind::RX, &[theta]), 1e-12));
        // RZ(λ) = e^{−iλ/2} P(λ).
        let lam = 1.1;
        let p = matrix_1q(GateKind::P, &[lam]).map(|v| v * C64::from_polar(1.0, -lam / 2.0));
        assert!(close(&p, &matrix_1q(GateKind::RZ, &[lam]), 1e-12));
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let h = 1e-6;
        for kind in [
            GateKind::RX,
            GateKind::RY,
            GateKind::RZ,
            GateKind::P,
            GateKind::U,
        ] {
            let p0 = params(kind);
            for which in 0..kind.num_params() {
                let mut plus = p0.clone();
                let mut minus = p0.clone();
                plus[which] += h;
                minus[which] -= h;
                let mp = matrix_1q(kind, &plus);
                let mm = matrix_1q(kind, &minus);
                let fd: Vec<C64> = mp
                    .iter()
                    .zip(&mm)
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect();
                assert!(
                    close(&dmatrix_1q(kind, &p0, which), &fd, 1e-8),
                    "{kind:?}/{which}"
                );
            }
        }
        for kind in [GateKind::RXX, GateKind::RYY, GateKind::RZZ] {
            let p0 = [0.42];
            let mp = matrix_2q(kind, &[0.42 + h]);
            let mm = matrix_2q(kind, &[0.42 - h]);
            let fd: Vec<C64> = mp
                .iter()
                .zip(&mm)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect();
            assert!(close(&dmatrix_2q(kind, &p0), &fd, 1e-8), "{kind:?}");
        }
    }

    #[test]
    fn dagger_inverts() {
        let u = matrix_1q(GateKind::U, &[0.3, 1.2, -0.4]);
        let ud = dagger(&u);
        let prod = mul2(&u, &ud.clone().try_into().unwrap());
        assert!(close(&prod, &[ONE, ZERO, ZERO, ONE], 1e-12));
    }
}
