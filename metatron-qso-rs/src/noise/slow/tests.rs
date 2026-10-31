//! Tests of milestone 1.44: slow noise averaged exactly.

use num_complex::Complex64 as C64;

use super::*;
use crate::linalg::dense::expm;
use crate::linalg::expm_hermitian;
use crate::noise::device::{GateRule, QubitProperties, Telegraph};
use crate::noise::{DeviceModel, TrajectorySimulator};
use crate::quantum::{Circuit, PauliSum};
use crate::rng::Rng;

fn c(re: f64) -> C64 {
    C64::new(re, 0.0)
}

/// Free induction under symmetric telegraph noise `±a` flipping at `γ`:
/// `e^{−γt}(cosh μt + γ sinh(μt)/μ)`, `μ = √(γ² − a²)` (Paladino et al.,
/// Rev. Mod. Phys. 86, 361 (2014)).
/// Written with `e^{(μ−γ)t}` and `e^{−(μ+γ)t}` (`μ − γ = −a²/(μ + γ)`) so
/// that fast switching does not overflow.
fn telegraph_free(a: f64, g: f64, t: f64) -> f64 {
    let mu = c(g * g - a * a).sqrt();
    let (slow, fast) = ((c(-a * a) / (mu + g) * t).exp(), (-(mu + g) * t).exp());
    (0.5 * ((1.0 + g / mu) * slow + (1.0 - g / mu) * fast)).re
}

/// The Hahn echo of the same noise (π pulse at `t/2`):
/// `e^{−γt}(1 + γ sinh(μt)/μ + γ²(cosh μt − 1)/μ²)`, written the same way.
fn telegraph_echo(a: f64, g: f64, t: f64) -> f64 {
    let mu = c(g * g - a * a).sqrt();
    let (slow, fast) = ((c(-a * a) / (mu + g) * t).exp(), (-(mu + g) * t).exp());
    let r = g / mu;
    ((-g * t).exp() * (1.0 - r * r) + slow * 0.5 * (r + r * r) + fast * 0.5 * (r * r - r)).re
}

/// Kubo's free induction and echo under Gaussian noise with
/// `⟨ν(0)ν(τ)⟩ = σ² e^{−|τ|/τ_c}`.
fn kubo_free(sigma: f64, tc: f64, t: f64) -> f64 {
    (-sigma * sigma * tc * tc * (t / tc - 1.0 + (-t / tc).exp())).exp()
}

fn kubo_echo(sigma: f64, tc: f64, t: f64) -> f64 {
    (-sigma * sigma * tc * tc * (t / tc - 3.0 + 4.0 * (-t / (2.0 * tc)).exp() - (-t / tc).exp()))
        .exp()
}

#[test]
fn telegraph_noise_follows_the_closed_forms() {
    // From the static limit through the crossover to motional narrowing
    // and switching ten thousand times faster than the amplitude.
    for (a, g) in [
        (1.0, 0.0),
        (1.0, 0.05),
        (1.0, 0.6),
        (1.0, 2.5),
        (0.3, 40.0),
        (1.0, 1e4),
    ] {
        let process = MarkovProcess::telegraph(a, g).unwrap();
        for t in [0.4, 2.0, 7.0] {
            let free = process.dephasing(&[], t).unwrap();
            let echo = process.dephasing(&[t / 2.0], t).unwrap();
            assert!(
                (free - c(telegraph_free(a, g, t))).norm() < 1e-12,
                "free {a} {g} {t}: {free} vs {}",
                telegraph_free(a, g, t)
            );
            assert!(
                (echo - c(telegraph_echo(a, g, t))).norm() < 1e-12,
                "echo {a} {g} {t}"
            );
        }
    }
    // Values shifted to shift ± a: the shift adds the phase e^{−i·shift·∫f},
    // which the echo removes; the closed form and the general path agree.
    let (a, g, shift) = (0.5, 0.8, 0.7);
    let shifted = MarkovProcess::new(
        vec![shift + a, shift - a],
        vec![(0, 1, g), (1, 0, g)],
        vec![0.5, 0.5],
    )
    .unwrap();
    let general = shifted
        .product(&MarkovProcess::telegraph(0.0, 1.0).unwrap())
        .unwrap();
    for t in [0.4, 2.0, 7.0] {
        let free = C64::from_polar(telegraph_free(a, g, t), -shift * t);
        for p in [&shifted, &general] {
            assert!((p.dephasing(&[], t).unwrap() - free).norm() < 1e-12, "{t}");
            assert!(
                (p.dephasing(&[t / 2.0], t).unwrap() - c(telegraph_echo(a, g, t))).norm() < 1e-12
            );
        }
    }
    // The general path (Krylov steps on the generator) on the same noise,
    // as a process of four states (a second, silent telegraph).
    for (a, g) in [(1.0, 0.05), (1.0, 0.6), (1.0, 2.5)] {
        let process = MarkovProcess::telegraph(a, g)
            .unwrap()
            .product(&MarkovProcess::telegraph(0.0, 1.0).unwrap())
            .unwrap();
        for t in [0.4, 2.0, 7.0] {
            let free = process.dephasing(&[], t).unwrap();
            let echo = process.dephasing(&[t / 2.0], t).unwrap();
            assert!((free - c(telegraph_free(a, g, t))).norm() < 1e-12);
            assert!((echo - c(telegraph_echo(a, g, t))).norm() < 1e-12);
        }
    }
}

#[test]
fn the_telegraph_propagator_is_the_exponential() {
    for (a, shift, g, s, tau) in [
        (1.0, 0.0, 0.3, 1.0, 0.7),
        (1.0, 0.4, 0.3, -1.0, 2.0),
        (0.7, -0.2, 0.7, 1.0, 1.3),
        (2.0, 0.1, 5.0, -1.0, 0.9),
        (1.0, 0.0, 1.0, 1.0, 0.5),
        (1.5, 0.3, 0.8, 0.0, 1.1),
        (1.0, 0.0, 0.0, 1.0, 3.0),
    ] {
        let m = telegraph_propagator(a, shift, g, s, tau);
        let generator = CMatrix::from_row_slice(
            2,
            2,
            &[
                C64::new(-g, s * a + shift) * tau,
                c(g * tau),
                c(g * tau),
                C64::new(-g, -s * a + shift) * tau,
            ],
        );
        let e = expm(&generator);
        for r in 0..2 {
            for col in 0..2 {
                assert!(
                    (m[r][col] - e[(r, col)]).norm() < 1e-13,
                    "{a} {g} {s} {tau}"
                );
            }
        }
    }
    // Switching alone keeps probabilities, however fast.
    let m = telegraph_propagator(1.0, 0.0, 1e8, 0.0, 3.0);
    assert!((m[0][0] - c(0.5)).norm() < 1e-15 && (m[1][0] - c(0.5)).norm() < 1e-15);
    // Fast switching narrows the line: e^{−a²τ/(2γ)} for the coherence.
    let m = telegraph_propagator(1.0, 0.0, 1e8, -1.0, 3.0);
    let sum = 0.5 * (m[0][0] + m[0][1] + m[1][0] + m[1][1]);
    assert!((sum - c((-1.5e-8f64).exp())).norm() < 1e-15);
}

#[test]
fn ornstein_uhlenbeck_noise_is_a_product_of_fluctuators_and_tends_to_kubo() {
    let (sigma, tc) = (1.3, 0.7);
    for k in [1usize, 4, 16, 64] {
        let chain = MarkovProcess::ornstein_uhlenbeck(sigma, tc, k + 1).unwrap();
        let one = MarkovProcess::telegraph(sigma / (k as f64).sqrt(), 0.5 / tc).unwrap();
        for t in [0.5, 2.0] {
            let cpmg: Vec<f64> = (0..4).map(|j| (j as f64 + 0.5) * t / 4.0).collect();
            for pulses in [vec![], vec![t / 2.0], cpmg] {
                let product = one.dephasing(&pulses, t).unwrap().powu(k as u32);
                let chained = chain.dephasing(&pulses, t).unwrap();
                assert!((chained - product).norm() < 1e-12, "K = {k}, t = {t}");
            }
        }
        // The correlation is exponential at every K.
        for tau in [0.0, 0.3, 2.0] {
            let expected = sigma * sigma * (-tau / tc).exp();
            assert!((chain.correlation(tau).unwrap() - expected).abs() < 1e-12);
        }
        assert!(chain.mean(1.0).unwrap().abs() < 1e-12);
    }
    // The error against Kubo's Gaussian forms falls as 1/K: K·error tends
    // to a constant (from the product of exact factors, K up to 4096).
    for t in [0.5, 2.0] {
        let error = |k: usize, echo: bool| -> f64 {
            let one = MarkovProcess::telegraph(sigma / (k as f64).sqrt(), 0.5 / tc).unwrap();
            let pulses = if echo { vec![t / 2.0] } else { vec![] };
            let d = one.dephasing(&pulses, t).unwrap().powu(k as u32);
            let exact = if echo {
                kubo_echo(sigma, tc, t)
            } else {
                kubo_free(sigma, tc, t)
            };
            (d - c(exact)).norm()
        };
        for echo in [false, true] {
            let (e256, e1024, e4096) = (error(256, echo), error(1024, echo), error(4096, echo));
            assert!(4096.0 * e4096 < 0.2, "{e4096}");
            let r1 = 256.0 * e256 / (1024.0 * e1024);
            let r2 = 1024.0 * e1024 / (4096.0 * e4096);
            assert!(
                (r1 - 1.0).abs() < 0.02 && (r2 - 1.0).abs() < 0.01,
                "{r1} {r2}"
            );
        }
    }
}

#[test]
fn quasi_static_noise_is_the_limit_of_slow_switching() {
    // A frozen telegraph process is a static ±a: cos(at), echoed away.
    let frozen = MarkovProcess::telegraph(0.8, 0.0).unwrap();
    for t in [0.5, 3.0] {
        assert!((frozen.dephasing(&[], t).unwrap() - c((0.8 * t).cos())).norm() < 1e-14);
        assert!((frozen.dephasing(&[t / 2.0], t).unwrap() - c(1.0)).norm() < 1e-14);
    }
    // Slow switching approaches it linearly in the rate.
    let t = 3.0;
    let errors: Vec<f64> = [1e-2, 1e-3, 1e-4]
        .iter()
        .map(|&g| {
            let p = MarkovProcess::telegraph(0.8, g).unwrap();
            (p.dephasing(&[], t).unwrap() - c((0.8 * t).cos())).norm()
        })
        .collect();
    assert!(
        (errors[0] / errors[1] - 10.0).abs() < 0.2 && (errors[1] / errors[2] - 10.0).abs() < 0.02
    );
    // Ornstein–Uhlenbeck noise with a long correlation time: the static
    // binomial levels, cos(σt/√K)^K.
    let (sigma, k) = (0.9, 20usize);
    let slow = MarkovProcess::ornstein_uhlenbeck(sigma, 1e9, k + 1).unwrap();
    for t in [0.5, 2.0] {
        let expected = (sigma * t / (k as f64).sqrt()).cos().powi(k as i32);
        assert!((slow.dephasing(&[], t).unwrap() - c(expected)).norm() < 1e-8);
    }
    // Quasi-static noise on Gauss nodes: e^{−σ²t²/2} and sin(ht)/(ht) to
    // the a-priori bound, with the nodes it asks for.
    for (spread, exact) in [
        (
            Spread::Gaussian { sigma: 1.1 },
            Box::new(|t: f64| (-1.1f64 * 1.1 * t * t / 2.0).exp()) as Box<dyn Fn(f64) -> f64>,
        ),
        (
            Spread::Uniform { half_width: 1.1 },
            Box::new(|t: f64| (1.1 * t).sin() / (1.1 * t)),
        ),
    ] {
        for t in [0.5, 3.0, 8.0] {
            let m = nodes_needed(&spread, t, 1e-13).unwrap();
            let p = MarkovProcess::quasi_static(spread, m).unwrap();
            assert!(
                (p.dephasing(&[], t).unwrap() - c(exact(t))).norm() < 1e-13,
                "{spread:?} {t}"
            );
            // Static noise is refocused by an echo.
            assert!((p.dephasing(&[t / 2.0], t).unwrap() - c(1.0)).norm() < 1e-13);
        }
    }
}

#[test]
fn gauss_rules_and_their_bounds() {
    // Moments up to degree 2m − 1 are exact.
    let m = 12;
    let (x, w) = gauss_rule(&Spread::Gaussian { sigma: 1.0 }, m);
    let mut double_factorial = 1.0;
    for k in 0..m {
        let moment: f64 = x
            .iter()
            .zip(&w)
            .map(|(x, w)| w * x.powi(2 * k as i32))
            .sum();
        assert!((moment / double_factorial - 1.0).abs() < 1e-12, "{k}");
        let odd: f64 = x
            .iter()
            .zip(&w)
            .map(|(x, w)| w * x.powi(2 * k as i32 + 1))
            .sum();
        assert!(odd.abs() < 1e-12 * double_factorial);
        double_factorial *= (2 * k + 1) as f64;
    }
    let (x, w) = gauss_rule(&Spread::Uniform { half_width: 2.0 }, m);
    for k in 0..m {
        let moment: f64 = x
            .iter()
            .zip(&w)
            .map(|(x, w)| w * x.powi(2 * k as i32))
            .sum();
        let exact = 2f64.powi(2 * k as i32) / (2 * k + 1) as f64;
        assert!((moment / exact - 1.0).abs() < 1e-12, "{k}");
    }
    // The bound holds for e^{icx}, and is not far from the nodes needed.
    for cc in [0.5, 2.0, 10.0, 25.0] {
        for spread in [
            Spread::Gaussian { sigma: 1.0 },
            Spread::Uniform { half_width: 1.0 },
        ] {
            let m = nodes_needed(&spread, cc, 1e-13).unwrap();
            let (x, w) = gauss_rule(&spread, m);
            let avg: C64 = x
                .iter()
                .zip(&w)
                .map(|(x, w)| C64::from_polar(*w, cc * x))
                .sum();
            let exact = match spread {
                Spread::Gaussian { .. } => (-cc * cc / 2.0).exp(),
                Spread::Uniform { .. } => cc.sin() / cc,
            };
            assert!((avg - c(exact)).norm() < 1e-13, "{spread:?} {cc} {m}");
        }
    }
    assert_eq!(
        nodes_needed(&Spread::Gaussian { sigma: 0.0 }, 5.0, 1e-13),
        Some(1)
    );
    assert_eq!(
        nodes_needed(&Spread::Gaussian { sigma: 1.0 }, 60.0, 1e-13),
        None
    );
}

#[test]
fn fluctuators_multiply() {
    // A few strong 1/f fluctuators: their product equals the joint
    // process of all of them, and their correlations add.
    let noise = Fluctuators::one_over_f(0.3, 0.01, 10.0, 6).unwrap();
    let joint = noise.combined().unwrap();
    assert_eq!(joint.num_states(), 64);
    for t in [1.0, 6.0] {
        for pulses in [vec![], vec![t / 2.0], vec![t / 4.0, 3.0 * t / 4.0]] {
            let a = noise.dephasing(&pulses, t).unwrap();
            let b = joint.dephasing(&pulses, t).unwrap();
            assert!((a - b).norm() < 1e-12, "{t} {pulses:?}");
        }
    }
    for tau in [0.0, 0.5, 4.0] {
        let expected: f64 = (0..6)
            .map(|k| {
                let rate = 0.01 * 1000f64.powf(k as f64 / 5.0);
                0.09 * (-2.0 * rate * tau).exp()
            })
            .sum();
        assert!((noise.correlation(tau).unwrap() - expected).abs() < 1e-12);
        assert!((joint.correlation(tau).unwrap() - expected).abs() < 1e-12);
    }
    // Many weak fluctuators of fixed total variance: Gaussian dephasing
    // e^{−χ}, χ = Σ c²(θt − 1 + e^{−θt})/θ² with θ = 2γ, approached as
    // 1/count.
    let (sigma, t) = (1.0, 2.0);
    let deviation = |count: usize| -> f64 {
        let noise =
            Fluctuators::one_over_f(sigma / (count as f64).sqrt(), 0.05, 5.0, count).unwrap();
        let chi: f64 = noise
            .processes()
            .iter()
            .map(|p| {
                let theta = 2.0 * p.rates()[0].2;
                let c2 = p.values()[0].powi(2);
                c2 * (theta * t - 1.0 + (-theta * t).exp()) / (theta * theta)
            })
            .sum();
        (noise.dephasing(&[], t).unwrap() - c((-chi).exp())).norm()
    };
    let (d100, d400) = (deviation(100), deviation(400));
    assert!(d400 < 1e-3 && (100.0 * d100 / (400.0 * d400) - 1.0).abs() < 0.05);
}

/// `σ_x`, `σ_y`, `σ_z` and `I` of one qubit.
fn pauli(k: usize) -> CMatrix {
    let z = C64::new(0.0, 0.0);
    let one = c(1.0);
    let i = C64::new(0.0, 1.0);
    match k {
        0 => CMatrix::from_row_slice(2, 2, &[one, z, z, one]),
        1 => CMatrix::from_row_slice(2, 2, &[z, one, one, z]),
        2 => CMatrix::from_row_slice(2, 2, &[z, -i, i, z]),
        _ => CMatrix::from_row_slice(2, 2, &[one, z, z, -one]),
    }
}

fn plus() -> CMatrix {
    CMatrix::from_element(2, 2, c(0.5))
}

fn max_diff(a: &CMatrix, b: &CMatrix) -> f64 {
    (a - b).iter().map(|z| z.norm()).fold(0.0, f64::max)
}

#[test]
fn the_liouville_equation_dephases_and_echoes() {
    // Pure dephasing: ρ₀₁ = ½ ⟨e^{−iφ}⟩, and after an echo the conjugate
    // of the echo factor (the pulse swaps ρ₀₁ and ρ₁₀). An asymmetric
    // process makes the factors complex.
    let asymmetric = MarkovProcess::new(
        vec![1.2, -0.4],
        vec![(0, 1, 0.3), (1, 0, 0.9)],
        vec![0.25, 0.75],
    )
    .unwrap();
    for process in [
        MarkovProcess::ornstein_uhlenbeck(0.8, 0.5, 8).unwrap(),
        asymmetric,
    ] {
        let sle = StochasticLiouville::new(2, vec![(process.clone(), pauli(3) * c(0.5))]).unwrap();
        let zero = CMatrix::zeros(2, 2);
        let t = 1.7;
        let start = sle.start(&plus()).unwrap();
        let free = sle.evolve(&start, &zero, t).unwrap().average();
        assert!((free[(0, 1)] - process.dephasing(&[], t).unwrap() * 0.5).norm() < 1e-11);
        let half = sle.evolve(&start, &zero, t / 2.0).unwrap();
        let flipped = sle.pulse(&half, &pauli(1)).unwrap();
        let echo = sle.evolve(&flipped, &zero, t / 2.0).unwrap().average();
        let factor = process.dephasing(&[t / 2.0], t).unwrap().conj();
        assert!((echo[(0, 1)] - factor * 0.5).norm() < 1e-11);
        // The process keeps its distribution only if it starts stationary;
        // the traces of the parts follow it either way.
        let probabilities = sle.evolve(&start, &zero, t).unwrap().probabilities();
        let p = process
            .propagate(&process.initial_vector(), t, 0.0)
            .unwrap();
        for (a, b) in probabilities.iter().zip(&p) {
            assert!((a - b.re).abs() < 1e-11);
        }
    }
}

/// `exp(t L) vec(ρ)` for the Lindblad generator of `h` and `ops`, built
/// with Kronecker products (columns stacked: `vec(AρB) = (Bᵀ ⊗ A) vec ρ`).
fn lindblad(h: &CMatrix, ops: &[CMatrix], rho: &CMatrix, t: f64) -> CMatrix {
    let d = h.nrows();
    let id = CMatrix::identity(d, d);
    let i = C64::new(0.0, 1.0);
    let mut l = (id.kronecker(h) - h.transpose().kronecker(&id)) * -i;
    for op in ops {
        let ldl = op.adjoint() * op;
        l += op.conjugate().kronecker(op)
            - id.kronecker(&ldl) * c(0.5)
            - ldl.transpose().kronecker(&id) * c(0.5);
    }
    let v = CMatrix::from_iterator(d * d, 1, rho.iter().copied());
    let out = expm(&(l * c(t))) * v;
    CMatrix::from_iterator(d, d, out.iter().copied())
}

#[test]
fn without_noise_the_liouville_equation_is_the_master_equation() {
    // Silent noise (values 0): the average obeys the Lindblad equation of
    // a driven, decaying qubit, built independently.
    let h = pauli(1) * c(0.6) + pauli(3) * c(0.25);
    let lowering = CMatrix::from_row_slice(2, 2, &[c(0.0), c(1.0), c(0.0), c(0.0)]) * c(0.3);
    let process = MarkovProcess::telegraph(0.0, 1.3).unwrap();
    let sle = StochasticLiouville::new(2, vec![(process, pauli(3))])
        .unwrap()
        .with_dissipators(vec![lowering.clone()])
        .unwrap();
    let mut rho0 = CMatrix::zeros(2, 2);
    rho0[(0, 0)] = c(1.0);
    for t in [0.5, 3.0] {
        let got = sle.evolve(&sle.start(&rho0).unwrap(), &h, t).unwrap();
        let expected = lindblad(&h, std::slice::from_ref(&lowering), &rho0, t);
        assert!(max_diff(&got.average(), &expected) < 1e-11, "{t}");
        assert!(got.probabilities().iter().all(|p| (p - 0.5).abs() < 1e-11));
    }
}

#[test]
fn fast_telegraph_noise_narrows_to_lindblad_dephasing() {
    // A driven qubit under ±a flipping at γ, with a²/(2γ) = Γ fixed:
    // as γ grows the noise becomes white, and the qubit dephases as under
    // the Lindblad operator √(Γ/2)·Z (ρ₀₁ decays at Γ), with an error that
    // falls as 1/γ.
    let gamma_phi = 0.2;
    let h = pauli(1) * c(0.5);
    let mut rho0 = CMatrix::zeros(2, 2);
    rho0[(0, 0)] = c(1.0);
    let t = 4.0;
    let white = lindblad(&h, &[pauli(3) * c((gamma_phi / 2.0f64).sqrt())], &rho0, t);
    let errors: Vec<f64> = [10.0, 100.0, 1000.0]
        .iter()
        .map(|&g| {
            let a = (2.0 * gamma_phi * g).sqrt();
            let process = MarkovProcess::telegraph(a, g).unwrap();
            let sle = StochasticLiouville::new(2, vec![(process, pauli(3) * c(0.5))]).unwrap();
            let got = sle.evolve(&sle.start(&rho0).unwrap(), &h, t).unwrap();
            max_diff(&got.average(), &white)
        })
        .collect();
    assert!(errors[2] < 1e-3, "{errors:?}");
    assert!(
        errors[0] / errors[1] > 7.0 && errors[1] / errors[2] > 7.0,
        "{errors:?}"
    );
}

#[test]
fn independent_noise_on_two_qubits_factorises() {
    let tel = MarkovProcess::telegraph(0.9, 0.4).unwrap();
    let ou = MarkovProcess::ornstein_uhlenbeck(0.7, 1.5, 4).unwrap();
    let id = pauli(0);
    let half = c(0.5);
    let sle = StochasticLiouville::new(
        4,
        vec![
            (tel.clone(), pauli(3).kronecker(&id) * half),
            (ou.clone(), id.kronecker(&pauli(3)) * half),
        ],
    )
    .unwrap();
    assert_eq!(sle.num_joint_states(), 8);
    let (h0, h1) = (pauli(1) * c(0.35), pauli(1) * c(0.8) + pauli(3) * c(0.1));
    let h = h0.kronecker(&id) + id.kronecker(&h1);
    let mut zero = CMatrix::zeros(2, 2);
    zero[(0, 0)] = c(1.0);
    let t = 2.5;
    let both = sle
        .evolve(&sle.start(&zero.kronecker(&zero)).unwrap(), &h, t)
        .unwrap();
    let one = |p: &MarkovProcess, h: &CMatrix| -> CMatrix {
        let s = StochasticLiouville::new(2, vec![(p.clone(), pauli(3) * half)]).unwrap();
        s.evolve(&s.start(&zero).unwrap(), h, t).unwrap().average()
    };
    let expected = one(&tel, &h0).kronecker(&one(&ou, &h1));
    assert!(max_diff(&both.average(), &expected) < 1e-11);
}

#[test]
fn the_liouville_equation_agrees_with_sampled_paths() {
    // A driven qubit under strong, slow telegraph noise and three-level
    // noise: the exact average against 4000 paths sampled event by event
    // and propagated exactly between switches, entry by entry within
    // five standard errors.
    let tel = MarkovProcess::telegraph(1.0, 0.5).unwrap();
    let three = MarkovProcess::new(
        vec![-0.6, 0.0, 0.9],
        vec![(0, 1, 0.7), (1, 0, 0.2), (1, 2, 0.4), (2, 1, 1.1)],
        vec![0.2, 0.5, 0.3],
    )
    .unwrap();
    let couplings = [pauli(3) * c(0.5), pauli(1) * c(0.5)];
    let processes = [tel, three];
    let sle = StochasticLiouville::new(
        2,
        processes
            .iter()
            .cloned()
            .zip(couplings.iter().cloned())
            .collect(),
    )
    .unwrap();
    let h = pauli(1) * c(0.4) + pauli(3) * c(0.3);
    let mut rho0 = CMatrix::zeros(2, 2);
    rho0[(0, 0)] = c(1.0);
    let t = 3.0;
    let exact = sle
        .evolve(&sle.start(&rho0).unwrap(), &h, t)
        .unwrap()
        .average();
    let mut rng = Rng::seed_from_u64(5);
    let paths = 4000;
    let mut sum = CMatrix::zeros(2, 2);
    let mut squares = CMatrix::zeros(2, 2);
    let draw = |p: &[f64], rng: &mut Rng| -> usize {
        let u = rng.uniform();
        let mut acc = 0.0;
        for (i, x) in p.iter().enumerate() {
            acc += x;
            if u < acc {
                return i;
            }
        }
        p.len() - 1
    };
    for _ in 0..paths {
        let mut states: Vec<usize> = processes
            .iter()
            .map(|p| draw(p.initial(), &mut rng))
            .collect();
        let mut psi = rho0.clone();
        let mut now = 0.0;
        loop {
            // The total rate out of the joint state, and the next switch.
            let outs: Vec<Vec<(usize, f64)>> = processes
                .iter()
                .zip(&states)
                .map(|(p, &s)| {
                    p.rates()
                        .iter()
                        .filter(|r| r.0 == s)
                        .map(|r| (r.1, r.2))
                        .collect()
                })
                .collect();
            let total: f64 = outs.iter().flatten().map(|r| r.1).sum();
            let wait = -(1.0 - rng.uniform()).ln() / total;
            let step = wait.min(t - now);
            let mut hx = h.clone();
            for ((p, &s), b) in processes.iter().zip(&states).zip(&couplings) {
                hx += b * c(p.values()[s]);
            }
            let u = expm_hermitian(&hx, step);
            psi = &u * psi * u.adjoint();
            now += step;
            if now >= t {
                break;
            }
            let mut pick = rng.uniform() * total;
            'choose: for (k, list) in outs.iter().enumerate() {
                for &(to, rate) in list {
                    if pick < rate {
                        states[k] = to;
                        break 'choose;
                    }
                    pick -= rate;
                }
            }
        }
        sum += &psi;
        squares += psi.map(|z| C64::new(z.re * z.re, z.im * z.im));
    }
    let n = paths as f64;
    for r in 0..2 {
        for col in 0..2 {
            let mean = sum[(r, col)] / n;
            let var = |sq: f64, m: f64| (sq / n - m * m).max(0.0) / (n - 1.0);
            let se_re = var(squares[(r, col)].re, mean.re).sqrt();
            let se_im = var(squares[(r, col)].im, mean.im).sqrt();
            let e = exact[(r, col)];
            assert!(
                (mean.re - e.re).abs() < 5.0 * se_re + 1e-12,
                "{r}{col}: {mean} vs {e}"
            );
            assert!((mean.im - e.im).abs() < 5.0 * se_im + 1e-12);
            assert!(se_re < 0.02);
        }
    }
}

/// `⟨Z⟩` of one qubit after `circuit` on a device, averaged exactly.
fn exact_z(device: &DeviceModel, circuit: &Circuit) -> f64 {
    let mut rho = DensityMatrix::zero(circuit.n_qubits()).unwrap();
    rho.apply_device(circuit, &[], device).unwrap();
    let z: PauliSum = "Z0".parse().unwrap();
    rho.expectation(&z).unwrap()
}

fn ramsey(qubit: QubitProperties, t: f64) -> f64 {
    let device = DeviceModel::new("ramsey")
        .with_default_qubit(qubit)
        .with_rule(GateRule::for_gate("id").with_duration(t));
    let mut circuit = Circuit::new(1);
    circuit.h(0).id(0).h(0);
    exact_z(&device, &circuit)
}

fn echo(qubit: QubitProperties, t: f64) -> f64 {
    let device = DeviceModel::new("echo")
        .with_default_qubit(qubit)
        .with_rule(GateRule::for_gate("id").with_duration(t / 2.0));
    let mut circuit = Circuit::new(1);
    circuit.h(0).id(0).x(0).id(0).h(0);
    exact_z(&device, &circuit)
}

#[test]
fn circuits_follow_the_closed_forms() {
    // The Ramsey and echo circuits of 1.30, now on density matrices:
    // telegraph noise as Kubo and Anderson, quasi-static detunings as a
    // Gaussian and a sinc, and an over-rotation spread, all to 10⁻¹².
    for (nu, g) in [(1.0, 0.0), (1.0, 0.3), (1.0, 3.0), (0.5, 20.0)] {
        let qubit = QubitProperties::new().with_telegraph(Telegraph::new(nu, g));
        for t in [0.5, 2.0, 6.0] {
            assert!((ramsey(qubit.clone(), t) - telegraph_free(nu, g, t)).abs() < 1e-12);
            assert!((echo(qubit.clone(), t) - telegraph_echo(nu, g, t)).abs() < 1e-12);
        }
    }
    let gauss = QubitProperties::new().with_detuning(Spread::Gaussian { sigma: 0.8 });
    let uniform = QubitProperties::new().with_detuning(Spread::Uniform { half_width: 1.3 });
    for t in [0.5, 2.0, 6.0] {
        assert!((ramsey(gauss.clone(), t) - (-0.32 * t * t).exp()).abs() < 1e-12);
        assert!((ramsey(uniform.clone(), t) - (1.3 * t).sin() / (1.3 * t)).abs() < 1e-12);
        assert!((echo(gauss.clone(), t) - 1.0).abs() < 1e-12);
    }
    let (theta, eps, sigma) = (2.3, 0.05, 0.1);
    let device = DeviceModel::new("spread").with_rule(
        GateRule::for_gate("rx")
            .with_over_rotation(eps)
            .with_over_rotation_spread(Spread::Gaussian { sigma }),
    );
    let mut circuit = Circuit::new(1);
    circuit.rx(0, theta);
    let expected = (theta * (1.0 + eps)).cos() * (-(theta * sigma).powi(2) / 2.0).exp();
    assert!((exact_z(&device, &circuit) - expected).abs() < 1e-12);
}

#[test]
fn circuits_agree_with_trajectories() {
    // Three qubits with every kind of slow noise and Markovian noise
    // besides: the exact average against 20 000 trajectories of 1.30.
    let device = DeviceModel::new("slow")
        .with_default_qubit(QubitProperties::new().with_t1_t2(40.0, 30.0))
        .with_qubit(
            0,
            QubitProperties::new().with_telegraph(Telegraph::new(0.6, 0.2)),
        )
        .with_qubit(
            1,
            QubitProperties::new()
                .with_telegraph(Telegraph::new(0.3, 1.5))
                .with_detuning(Spread::Uniform { half_width: 0.2 }),
        )
        .with_qubit(
            2,
            QubitProperties::new().with_detuning(Spread::Gaussian { sigma: 0.25 }),
        )
        .with_rule(
            GateRule::for_gate("rx")
                .with_duration(0.5)
                .with_over_rotation_spread(Spread::Gaussian { sigma: 0.05 }),
        )
        .with_rule(
            GateRule::for_gate("cx")
                .with_duration(1.0)
                .with_depolarizing(0.02),
        )
        .with_rule(GateRule::any().with_duration(0.3));
    let mut circuit = Circuit::new(3);
    circuit
        .h(0)
        .h(1)
        .rx(2, 1.1)
        .cx(0, 1)
        .rx(0, 0.7)
        .cx(1, 2)
        .h(0)
        .rx(1, 2.0)
        .h(2)
        .cx(0, 2);
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_device(&circuit, &[], &device).unwrap();
    assert!((rho.trace() - 1.0).abs() < 1e-12);
    let observables: Vec<PauliSum> = ["Z0", "Z1", "Z2", "X0 X1", "Z0 Z2", "Y1 Y2"]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let sampled = TrajectorySimulator::new(device, 20_000, 11)
        .unwrap()
        .expectations(&circuit, &[], &observables)
        .unwrap();
    for (o, e) in observables.iter().zip(&sampled) {
        let exact = rho.expectation(o).unwrap();
        assert!(
            (e.value - exact).abs() < 5.0 * e.std_error && e.std_error < 0.01,
            "{o}: {} ± {} vs {exact}",
            e.value,
            e.std_error
        );
    }
}

#[test]
fn markovian_parts_and_limits() {
    // Telegraph noise of amplitude 0 and a detuning of width 0 change
    // nothing: the same state as without them.
    let plain = DeviceModel::new("plain")
        .with_default_qubit(QubitProperties::new().with_t1_t2(20.0, 10.0))
        .with_rule(GateRule::any().with_duration(0.7).with_depolarizing(0.01));
    let silent = DeviceModel::new("silent")
        .with_default_qubit(
            QubitProperties::new()
                .with_t1_t2(20.0, 10.0)
                .with_telegraph(Telegraph::new(0.0, 1.0))
                .with_detuning(Spread::Gaussian { sigma: 0.0 }),
        )
        .with_rule(GateRule::any().with_duration(0.7).with_depolarizing(0.01));
    assert!(plain.is_markovian() && !silent.is_markovian());
    let mut circuit = Circuit::new(2);
    circuit.h(0).cx(0, 1).ry(1, 0.4).cx(1, 0);
    let mut a = DensityMatrix::zero(2).unwrap();
    a.apply_device(&circuit, &[], &plain).unwrap();
    let mut b = DensityMatrix::zero(2).unwrap();
    b.apply_device(&circuit, &[], &silent).unwrap();
    assert!(max_diff(&a.to_matrix(), &b.to_matrix()) < 1e-14);
    // Telegraph noise on twelve qubits: 2¹² density matrices of 4¹²
    // entries, refused.
    let crowded = DeviceModel::new("crowded")
        .with_default_qubit(QubitProperties::new().with_telegraph(Telegraph::new(0.1, 1.0)))
        .with_rule(GateRule::any().with_duration(1.0));
    let mut wide = Circuit::new(12);
    for q in 0..12 {
        wide.h(q);
    }
    let mut rho = DensityMatrix::zero(12).unwrap();
    assert!(rho.apply_device(&wide, &[], &crowded).is_err());
    // A detuning too strong for the quadrature.
    let strong = DeviceModel::new("strong")
        .with_default_qubit(QubitProperties::new().with_detuning(Spread::Gaussian { sigma: 50.0 }))
        .with_rule(GateRule::any().with_duration(1.0));
    let mut one = Circuit::new(1);
    one.h(0).h(0);
    let mut rho = DensityMatrix::zero(1).unwrap();
    assert!(rho.apply_device(&one, &[], &strong).is_err());
}

#[test]
fn invalid_input() {
    assert!(MarkovProcess::new(vec![], vec![], vec![]).is_err());
    assert!(MarkovProcess::new(vec![f64::NAN], vec![], vec![1.0]).is_err());
    assert!(MarkovProcess::new(vec![0.0, 1.0], vec![(0, 0, 1.0)], vec![0.5, 0.5]).is_err());
    assert!(MarkovProcess::new(vec![0.0, 1.0], vec![(0, 2, 1.0)], vec![0.5, 0.5]).is_err());
    assert!(MarkovProcess::new(vec![0.0, 1.0], vec![(0, 1, -1.0)], vec![0.5, 0.5]).is_err());
    assert!(MarkovProcess::new(vec![0.0, 1.0], vec![], vec![0.5, 0.6]).is_err());
    assert!(MarkovProcess::new(vec![0.0, 1.0], vec![], vec![1.0]).is_err());
    assert!(MarkovProcess::telegraph(1.0, -1.0).is_err());
    assert!(MarkovProcess::ornstein_uhlenbeck(1.0, 0.0, 4).is_err());
    assert!(MarkovProcess::ornstein_uhlenbeck(1.0, 1.0, 1).is_err());
    assert!(MarkovProcess::quasi_static(Spread::Gaussian { sigma: 1.0 }, 0).is_err());
    assert!(MarkovProcess::quasi_static(Spread::Gaussian { sigma: -1.0 }, 4).is_err());
    let p = MarkovProcess::telegraph(1.0, 1.0).unwrap();
    assert!(p.dephasing(&[2.0], 1.0).is_err());
    assert!(p.dephasing(&[0.6, 0.4], 1.0).is_err());
    assert!(p.dephasing(&[], f64::NAN).is_err());
    assert!(p.correlation(-1.0).is_err());
    assert!(Fluctuators::new(vec![]).is_err());
    assert!(Fluctuators::one_over_f(1.0, 2.0, 1.0, 4).is_err());
    assert!(Fluctuators::one_over_f(1.0, 1.0, 2.0, 0).is_err());
    let not_hermitian = CMatrix::from_row_slice(2, 2, &[c(0.0), c(1.0), c(0.0), c(0.0)]);
    assert!(StochasticLiouville::new(2, vec![(p.clone(), not_hermitian.clone())]).is_err());
    assert!(StochasticLiouville::new(3, vec![(p.clone(), pauli(3))]).is_err());
    let sle = StochasticLiouville::new(2, vec![(p, pauli(3))]).unwrap();
    let start = sle.start(&plus()).unwrap();
    assert!(sle.evolve(&start, &not_hermitian, 1.0).is_err());
    assert!(sle.evolve(&start, &pauli(1), -1.0).is_err());
    assert!(sle.pulse(&start, &(pauli(1) * c(2.0))).is_err());
    assert!(sle.start(&CMatrix::zeros(3, 3)).is_err());
}
