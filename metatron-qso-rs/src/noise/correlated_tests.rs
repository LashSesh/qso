//! Tests of milestone 1.30: channels on three and four qubits, crosstalk
//! on spectator qubits, and noise correlated in time (quasi-static
//! detunings and over-rotations, telegraph noise).

use super::*;
use crate::linalg::CMatrix;
use crate::rng::Rng;

/// Full `2ⁿ × 2ⁿ` matrix of a local operator `m` on `qubits` (first =
/// most significant local bit), built index by index.
fn embed(m: &[C64], qubits: &[usize], n: usize) -> CMatrix {
    let k = qubits.len();
    let dim = 1usize << n;
    let local =
        |b: usize| -> usize { qubits.iter().fold(0, |acc, &q| (acc << 1) | ((b >> q) & 1)) };
    let rest = |b: usize| qubits.iter().fold(b, |acc, &q| acc & !(1 << q));
    CMatrix::from_fn(dim, dim, |r, c| {
        if rest(r) == rest(c) {
            m[local(r) * (1 << k) + local(c)]
        } else {
            ZERO
        }
    })
}

fn max_diff(a: &CMatrix, b: &CMatrix) -> f64 {
    (a - b).iter().map(|z| z.norm()).fold(0.0, f64::max)
}

fn random_state_matrix(n: usize, seed: u64) -> DensityMatrix {
    let mut c = Circuit::new(n);
    let mut rng = Rng::seed_from_u64(seed);
    for layer in 0..3 {
        for q in 0..n {
            c.ry(q, rng.uniform_range(0.0, 3.0))
                .rz(q, rng.uniform_range(0.0, 3.0));
        }
        for q in (layer % 2..n - 1).step_by(2) {
            c.cx(q, q + 1);
        }
    }
    let mut rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
    // Mix it a little so that it is not pure.
    rho.apply_channel(&Channel::depolarizing(0.2).unwrap(), &[1])
        .unwrap();
    rho
}

/// A non-unitary three-qubit channel that is not a product: amplitude
/// damping on each qubit, then a correlated ZZZ flip and an XYZ rotation.
fn correlated_three() -> Channel {
    let ad = Channel::amplitude_damping(0.2).unwrap();
    let mut kraus = Vec::new();
    for a in ad.kraus_operators() {
        for b in ad.kraus_operators() {
            for c in ad.kraus_operators() {
                kraus.push(kron(&kron(a, b), c));
            }
        }
    }
    let damp = Channel::from_kraus("ad3", kraus).unwrap();
    damp.compose(&Channel::pauli(&[("ZZZ", 0.1), ("XIX", 0.05)]).unwrap())
        .unwrap()
        .compose(&Channel::rotation("XYZ", 0.3).unwrap())
        .unwrap()
}

fn kron(a: &[C64], b: &[C64]) -> Vec<C64> {
    let (da, db) = (
        (a.len() as f64).sqrt() as usize,
        (b.len() as f64).sqrt() as usize,
    );
    let d = da * db;
    let mut m = vec![ZERO; d * d];
    for r in 0..d {
        for c in 0..d {
            m[r * d + c] = a[(r / db) * da + c / db] * b[(r % db) * db + c % db];
        }
    }
    m
}

#[test]
fn channels_on_three_and_four_qubits_match_their_kraus_operators() {
    let four = Channel::pauli(&[("ZZZZ", 0.07), ("XXII", 0.03), ("IYZX", 0.02)])
        .unwrap()
        .compose(&Channel::rotation("ZXZY", 0.4).unwrap())
        .unwrap();
    assert_eq!(four.n_qubits(), 4);
    for (channel, qubits) in [
        (correlated_three(), vec![4, 0, 2]),
        (four, vec![1, 5, 3, 0]),
    ] {
        let n = 6;
        let mut rho = random_state_matrix(n, 11);
        let before = rho.to_matrix();
        rho.apply_channel(&channel, &qubits).unwrap();
        let mut expected = CMatrix::zeros(1 << n, 1 << n);
        for k in channel.kraus_operators() {
            let full = embed(k, &qubits, n);
            expected += &full * &before * full.adjoint();
        }
        let err = max_diff(&rho.to_matrix(), &expected);
        assert!(err < 1e-13, "{}: {err:e}", channel.name());
        assert!((rho.trace() - 1.0).abs() < 1e-13);
    }
    // Labels and operators of more than four qubits are refused.
    assert!(Channel::pauli(&[("ZZZZZ", 0.1)]).is_err());
    assert!(Channel::from_kraus("big", vec![vec![ZERO; 1024]]).is_err());
    let mut rho = DensityMatrix::zero(3).unwrap();
    assert!(rho.apply_channel(&correlated_three(), &[0, 1, 0]).is_err());
}

#[test]
fn trajectories_follow_three_qubit_channels() {
    // A general (non-unitary) three-qubit channel as crosstalk after a
    // two-qubit gate: trajectories average to the density matrix.
    let spec = ChannelSpec::from(&correlated_three());
    let device = DeviceModel::new("correlated")
        .with_rule(GateRule::for_gate("cx").with_spectator(&[0, 1, 2], spec));
    let mut c = Circuit::new(3);
    c.h(0).ry(1, 0.7).cx(0, 1).rx(2, 0.4).cx(1, 2);
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let sim = TrajectorySimulator::new(device, 20_000, 5).unwrap();
    let estimate = sim.density_matrix(&c, &[]).unwrap();
    let err = max_diff(&estimate.to_matrix(), &rho.to_matrix());
    // Entries are averages of bounded values: 4/√N is a generous bound.
    assert!(err < 4.0 / (20_000f64).sqrt(), "{err}");

    // A general four-qubit channel: damping on every qubit with a
    // correlated rotation, as crosstalk after a two-qubit gate.
    // Different damping on every qubit, so that the order of the qubits
    // matters.
    let ad: Vec<Channel> = [0.1, 0.25, 0.4, 0.55]
        .iter()
        .map(|&g| Channel::amplitude_damping(g).unwrap())
        .collect();
    let mut kraus = Vec::new();
    for a in ad[0].kraus_operators() {
        for b in ad[1].kraus_operators() {
            for c in ad[2].kraus_operators() {
                for d in ad[3].kraus_operators() {
                    kraus.push(kron(&kron(&kron(a, b), c), d));
                }
            }
        }
    }
    let four = Channel::from_kraus("ad4", kraus)
        .unwrap()
        .compose(&Channel::rotation("XZZY", 0.5).unwrap())
        .unwrap();
    let device = DeviceModel::new("four").with_rule(
        GateRule::for_gate("cz").with_spectator(&[3, 1, 0, 2], ChannelSpec::from(&four)),
    );
    let mut c = Circuit::new(4);
    c.h(0).h(1).h(2).ry(3, 1.2).cz(1, 2).rx(0, 0.3);
    let mut rho = DensityMatrix::zero(4).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let estimate = TrajectorySimulator::new(device, 20_000, 6)
        .unwrap()
        .density_matrix(&c, &[])
        .unwrap();
    let err = max_diff(&estimate.to_matrix(), &rho.to_matrix());
    assert!(err < 4.0 / (20_000f64).sqrt(), "{err}");
}

#[test]
fn noise_models_spread_three_qubit_channels() {
    // A three-qubit channel after a Toffoli acts on its three qubits; after
    // two-qubit gates it cannot act and is an error.
    let ch = correlated_three();
    let model = NoiseModel {
        single_qubit: Vec::new(),
        multi_qubit: vec![ch.clone()],
        readout: None,
    };
    let mut c = Circuit::new(4);
    c.h(1).ccx(1, 3, 0);
    let mut rho = DensityMatrix::zero(4).unwrap();
    rho.apply_circuit(&c, &[], Some(&model)).unwrap();
    let mut expected = DensityMatrix::zero(4).unwrap();
    expected.apply_circuit(&c, &[], None).unwrap();
    expected.apply_channel(&ch, &[1, 3, 0]).unwrap();
    assert!(max_diff(&rho.to_matrix(), &expected.to_matrix()) < 1e-13);
    let device = DeviceModel::new("bad").with_rule(
        GateRule::any()
            .with_num_qubits(2)
            .with_channel(ChannelSpec::from(&ch)),
    );
    assert!(device.validate().is_err());
    // The same noise model as a device gives the same state, and both
    // refuse the channel after a two-qubit gate instead of dropping it.
    let converted = DeviceModel::from(&model);
    let mut via_device = DensityMatrix::zero(4).unwrap();
    via_device.apply_device(&c, &[], &converted).unwrap();
    assert!(max_diff(&via_device.to_matrix(), &rho.to_matrix()) < 1e-13);
    let mut two = Circuit::new(4);
    two.cx(0, 1);
    let mut rho = DensityMatrix::zero(4).unwrap();
    assert!(rho.apply_circuit(&two, &[], Some(&model)).is_err());
    assert!(rho.apply_device(&two, &[], &converted).is_err());
}

#[test]
fn spectators_wait_and_receive_crosstalk() {
    // ZZ crosstalk on (1, 2) during a CX on (0, 1); qubit 2 waits for the
    // gate and relaxes during it.
    let zz = ChannelSpec::Rotation {
        pauli: "ZZ".into(),
        angle: 0.3,
    };
    let device = DeviceModel::new("spectator")
        .with_default_qubit(QubitProperties::new().with_t1_t2(20.0, 15.0))
        .with_rule(
            GateRule::for_gate("cx")
                .with_duration(2.0)
                .with_spectator(&[1, 2], zz.clone()),
        );
    let mut c = Circuit::new(3);
    c.h(0).h(2).cx(0, 1).h(2);
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let mut expected = DensityMatrix::zero(3).unwrap();
    let mut first = Circuit::new(3);
    first.h(0).h(2).cx(0, 1);
    expected.apply_circuit(&first, &[], None).unwrap();
    expected
        .apply_channel(&zz.to_channel().unwrap(), &[1, 2])
        .unwrap();
    let relax = Channel::thermal_relaxation(20.0, 15.0, 2.0).unwrap();
    for q in 0..3 {
        expected.apply_channel(&relax, &[q]).unwrap();
    }
    let mut last = Circuit::new(3);
    last.h(2);
    expected.apply_circuit(&last, &[], None).unwrap();
    assert!(max_diff(&rho.to_matrix(), &expected.to_matrix()) < 1e-13);
    // Spectators beyond the register and mismatched channels are errors.
    let far =
        DeviceModel::new("far").with_rule(GateRule::any().with_spectator(&[0, 7], zz.clone()));
    assert!(rho.apply_device(&c, &[], &far).is_err());
    let wrong =
        DeviceModel::new("wrong").with_rule(GateRule::any().with_spectator(&[0], zz.clone()));
    assert!(wrong.validate().is_err());
    let twice = DeviceModel::new("twice").with_rule(GateRule::any().with_spectator(&[1, 1], zz));
    assert!(twice.validate().is_err());
}

/// One qubit with `qubit` noise: H, a wait of `t` (an `id` gate), H.
/// Returns ⟨Z⟩ over `n` trajectories.
fn ramsey(qubit: QubitProperties, t: f64, n: usize) -> crate::estimation::Estimate {
    let device = DeviceModel::new("ramsey")
        .with_default_qubit(qubit)
        .with_rule(GateRule::for_gate("id").with_duration(t));
    let mut c = Circuit::new(1);
    c.h(0).id(0).h(0);
    let z: crate::quantum::PauliSum = "Z0".parse().unwrap();
    TrajectorySimulator::new(device, n, 17)
        .unwrap()
        .expectation(&c, &[], &z)
        .unwrap()
}

/// Hahn echo: H, wait t/2, X, wait t/2, H. ⟨Z⟩ = ⟨cos(φ₁ − φ₂)⟩.
fn echo(qubit: QubitProperties, t: f64, n: usize) -> crate::estimation::Estimate {
    let device = DeviceModel::new("echo")
        .with_default_qubit(qubit)
        .with_rule(GateRule::for_gate("id").with_duration(t / 2.0));
    let mut c = Circuit::new(1);
    c.h(0).id(0).x(0).id(0).h(0);
    let z: crate::quantum::PauliSum = "Z0".parse().unwrap();
    TrajectorySimulator::new(device, n, 23)
        .unwrap()
        .expectation(&c, &[], &z)
        .unwrap()
}

fn within(e: crate::estimation::Estimate, exact: f64, what: &str) {
    assert!(
        (e.value - exact).abs() < 4.0 * e.std_error + 1e-12,
        "{what}: {} ± {} vs {exact}",
        e.value,
        e.std_error
    );
}

#[test]
fn quasi_static_detuning_decays_as_a_gaussian_and_echoes_away() {
    let sigma = 0.8;
    let gauss = QubitProperties::new().with_detuning(Spread::Gaussian { sigma });
    let a = 1.3;
    let uniform = QubitProperties::new().with_detuning(Spread::Uniform { half_width: a });
    for t in [0.5, 1.0, 2.0, 3.0] {
        within(
            ramsey(gauss.clone(), t, 8000),
            (-sigma * sigma * t * t / 2.0).exp(),
            &format!("Gaussian, t = {t}"),
        );
        within(
            ramsey(uniform.clone(), t, 8000),
            (a * t).sin() / (a * t),
            &format!("uniform, t = {t}"),
        );
        // A static detuning is refocused exactly in every trajectory.
        let e = echo(gauss.clone(), t, 200);
        assert!(
            (e.value - 1.0).abs() < 1e-12 && e.std_error < 1e-12,
            "echo, t = {t}"
        );
    }
}

/// `⟨e^{iν∫s}⟩` of a symmetric telegraph process (flip rate `γ`, started
/// in equilibrium) over `t`: the Kubo–Anderson form
/// `e^{−γt} (cosh μt + (γ/μ) sinh μt)`, `μ = √(γ² − ν²)`.
fn telegraph_free(nu: f64, gamma: f64, t: f64) -> f64 {
    let mu2 = gamma * gamma - nu * nu;
    let (c, s) = if mu2 > 0.0 {
        let mu = mu2.sqrt();
        ((mu * t).cosh(), gamma * (mu * t).sinh() / mu)
    } else {
        let w = (-mu2).sqrt();
        ((w * t).cos(), gamma * (w * t).sin() / w)
    };
    (-gamma * t).exp() * (c + s)
}

/// Hahn echo of the same process: the two-state equations
/// `d f/dt = (∓iν·σ_z-like + γ·flip − γ) f`, solved in closed form for each
/// half (exp(Bτ) = cosh(μτ) + B sinh(μτ)/μ with B² = μ²).
fn telegraph_echo(nu: f64, gamma: f64, t: f64) -> f64 {
    use num_complex::Complex64 as Z;
    let half = t / 2.0;
    let step = |f: [Z; 2], sign: f64| -> [Z; 2] {
        let mu = Z::new(gamma * gamma - nu * nu, 0.0).sqrt();
        let (ch, sh) = ((mu * half).cosh(), (mu * half).sinh());
        let k = if mu.norm() < 1e-14 {
            Z::new(half, 0.0)
        } else {
            sh / mu
        };
        // B = [[iν·sign, γ], [γ, −iν·sign]].
        let i = Z::new(0.0, nu * sign);
        let b = [[i, Z::new(gamma, 0.0)], [Z::new(gamma, 0.0), -i]];
        let decay = (-gamma * half).exp();
        let mut out = [Z::new(0.0, 0.0); 2];
        for r in 0..2 {
            out[r] = ch * f[r];
            for c in 0..2 {
                out[r] += k * b[r][c] * f[c];
            }
            out[r] *= decay;
        }
        out
    };
    let f = step([Z::new(0.5, 0.0); 2], 1.0);
    let f = step(f, -1.0);
    (f[0] + f[1]).re
}

#[test]
fn telegraph_noise_follows_the_closed_forms() {
    // Slow switching (γ < ν: coherence oscillates and the echo helps) and
    // fast switching (γ > ν: motional narrowing).
    for (nu, gamma) in [(1.0, 0.2), (1.0, 5.0)] {
        let qubit = QubitProperties::new().with_telegraph(Telegraph::new(nu, gamma));
        for t in [0.5, 1.5, 3.0, 6.0] {
            within(
                ramsey(qubit.clone(), t, 8000),
                telegraph_free(nu, gamma, t),
                &format!("free, ν = {nu}, γ = {gamma}, t = {t}"),
            );
            within(
                echo(qubit.clone(), t, 8000),
                telegraph_echo(nu, gamma, t),
                &format!("echo, ν = {nu}, γ = {gamma}, t = {t}"),
            );
        }
    }
    // Without switching the noise is static and the echo is perfect.
    let frozen = QubitProperties::new().with_telegraph(Telegraph::new(0.7, 0.0));
    assert!((echo(frozen.clone(), 2.0, 50).value - 1.0).abs() < 1e-12);
    within(ramsey(frozen, 2.0, 50), (1.4f64).cos(), "frozen");
    // The closed form of the echo reduces to the free decay's structure:
    // at t → 0 both are 1.
    assert!((telegraph_echo(1.0, 0.3, 1e-9) - 1.0).abs() < 1e-12);
}

#[test]
fn over_rotation_spread_averages_the_rotation() {
    // rx(θ) with θ ↦ θ(1 + ε + η), η Gaussian: ⟨Z⟩ = cos(θ(1+ε)) e^{−θ²σ²/2}.
    let (theta, eps, sigma) = (1.1, 0.05, 0.3);
    let device = DeviceModel::new("spread").with_rule(
        GateRule::for_gate("rx")
            .with_over_rotation(eps)
            .with_over_rotation_spread(Spread::Gaussian { sigma }),
    );
    assert!(!device.is_markovian());
    let mut c = Circuit::new(1);
    c.rx(0, theta);
    let z: crate::quantum::PauliSum = "Z0".parse().unwrap();
    let e = TrajectorySimulator::new(device.clone(), 8000, 3)
        .unwrap()
        .expectation(&c, &[], &z)
        .unwrap();
    within(
        e,
        (theta * (1.0 + eps)).cos() * (-(theta * sigma).powi(2) / 2.0).exp(),
        "spread",
    );
    // Density matrices average the spread exactly (1.44); PEC refuses
    // noise correlated in time.
    let mut rho = DensityMatrix::zero(1).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let exact = (theta * (1.0 + eps)).cos() * (-(theta * sigma).powi(2) / 2.0).exp();
    assert!((rho.expectation(&z).unwrap() - exact).abs() < 1e-12);
    assert!(crate::mitigation::pec::pec_expectation(&c, &[], &device, &z, 10, 1).is_err());
}

#[test]
fn slow_noise_round_trips_through_json_and_is_validated() {
    let device = DeviceModel::new("slow")
        .with_default_qubit(
            QubitProperties::new()
                .with_detuning(Spread::Uniform { half_width: 0.2 })
                .with_telegraph(Telegraph::new(0.5, 0.1)),
        )
        .with_qubit(
            1,
            QubitProperties::new().with_detuning(Spread::Gaussian { sigma: 0.3 }),
        )
        .with_rule(
            GateRule::any()
                .with_over_rotation_spread(Spread::Gaussian { sigma: 0.01 })
                .with_spectator(&[0, 2], ChannelSpec::Pauli([("ZZ".into(), 0.01)].into())),
        );
    let json = device.to_json();
    assert_eq!(DeviceModel::from_json(&json).unwrap(), device);
    assert!(json.contains("\"gaussian\"") && json.contains("\"telegraph\""));
    // Qubit 1 keeps the default telegraph noise and its own detuning.
    let q1 = device.qubit(1);
    assert_eq!(q1.detuning, Some(Spread::Gaussian { sigma: 0.3 }));
    assert_eq!(q1.telegraph, Some(Telegraph::new(0.5, 0.1)));
    assert!(!device.is_markovian());
    assert!(DeviceModel::new("plain").is_markovian());
    for bad in [
        QubitProperties::new().with_detuning(Spread::Gaussian { sigma: -1.0 }),
        QubitProperties::new().with_detuning(Spread::Uniform {
            half_width: f64::NAN,
        }),
        QubitProperties::new().with_telegraph(Telegraph::new(1.0, -0.1)),
        QubitProperties::new().with_telegraph(Telegraph::new(f64::INFINITY, 1.0)),
    ] {
        assert!(
            DeviceModel::new("bad")
                .with_default_qubit(bad)
                .validate()
                .is_err()
        );
    }
    // Only an identity is the gate "id".
    let mut c = Circuit::new(1);
    c.id(0).x(0);
    let rule = GateRule::for_gate("id");
    assert!(rule.matches(&c.instructions()[0]) && !rule.matches(&c.instructions()[1]));
}

#[test]
fn markovian_programs_draw_no_slow_noise() {
    // Without slow noise a trajectory draws nothing extra, so its random
    // stream (and with it every result) is the same as before 1.30; the
    // review compared all trajectory outputs with the previous build bit
    // for bit.
    let device = DeviceModel::new("plain")
        .with_default_qubit(QubitProperties::new().with_t1_t2(30.0, 20.0))
        .with_rule(GateRule::any().with_duration(1.0).with_depolarizing(0.05));
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1).ry(1, 0.3).id(0);
    let program = super::device::Program::compile(&c, &[], &device).unwrap();
    let skeleton = super::trajectories::Skeleton::for_program(&program);
    let mut rng = Rng::seed_from_u64(9);
    let mut untouched = rng.clone();
    let (phases, etas) = skeleton.slow_sample(&mut rng);
    assert!(phases.is_empty() && etas.is_empty());
    assert_eq!(rng.next_u64(), untouched.next_u64());
}

#[test]
fn a_pulse_in_the_middle_of_an_echo_still_refocuses() {
    // Turns during a gate are split around it: an X of finite duration in
    // the middle of a Hahn echo refocuses a static detuning exactly.
    let device = DeviceModel::new("pulse")
        .with_default_qubit(QubitProperties::new().with_detuning(Spread::Gaussian { sigma: 0.8 }))
        .with_rule(GateRule::for_gate("id").with_duration(1.0))
        .with_rule(GateRule::for_gate("x").with_duration(0.4));
    let mut c = Circuit::new(1);
    c.h(0).id(0).x(0).id(0).h(0);
    let z: crate::quantum::PauliSum = "Z0".parse().unwrap();
    let e = TrajectorySimulator::new(device, 200, 1)
        .unwrap()
        .expectation(&c, &[], &z)
        .unwrap();
    assert!((e.value - 1.0).abs() < 1e-12, "{}", e.value);
}

#[test]
fn pec_cancels_three_qubit_pauli_crosstalk() {
    let device = DeviceModel::new("pec").with_rule(GateRule::for_gate("cx").with_spectator(
        &[0, 1, 2],
        ChannelSpec::Pauli([("ZZZ".into(), 0.05), ("XIX".into(), 0.03)].into()),
    ));
    let mut c = Circuit::new(3);
    c.h(0).h(2).cx(0, 1);
    let x: crate::quantum::PauliSum = "X0 X1 X2".parse().unwrap();
    let exact = x
        .expectation(&crate::quantum::StateVector::from_circuit(&c, &[]).unwrap())
        .unwrap();
    let r = crate::mitigation::pec::pec_expectation(&c, &[], &device, &x, 20_000, 4).unwrap();
    assert_eq!(r.noisy_operations, 1);
    assert!(
        (r.estimate.value - exact).abs() < 4.0 * r.estimate.std_error
            && r.estimate.std_error < 0.05,
        "{} ± {} vs {exact}",
        r.estimate.value,
        r.estimate.std_error
    );
}

#[test]
fn identities_survive_qasm_and_the_stabilizer_simulator() {
    let mut c = Circuit::new(2);
    c.h(0).id(1).cx(0, 1);
    assert_eq!(c.gate_counts().get("id"), Some(&1));
    let two = crate::quantum::to_qasm2(&c, &[], false).unwrap();
    assert!(two.contains("id q[1];"), "{two}");
    let back = crate::quantum::from_qasm2(&two).unwrap();
    assert!(back.instructions()[1].is_identity());
    let three = crate::quantum::to_qasm3(&c, &[], false).unwrap();
    let back3 = crate::quantum::Program::from_qasm(&three)
        .unwrap()
        .gates_skipping_measurements()
        .unwrap();
    assert!(back3.instructions().iter().any(|i| i.is_identity()));
    let mut s = crate::stabilizer::StabilizerState::zero(2).unwrap();
    for inst in c.instructions() {
        s.apply_instruction(inst, &[]).unwrap();
    }
    assert!((s.expectation_str("Z0 Z1").unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn four_qubit_channels_compose_through_superoperators() {
    // Two general four-qubit channels: the composition acts like applying
    // one after the other, and takes seconds at most.
    let ad = Channel::amplitude_damping(0.2).unwrap();
    let mut kraus = Vec::new();
    for a in ad.kraus_operators() {
        for b in ad.kraus_operators() {
            for c in ad.kraus_operators() {
                for d in ad.kraus_operators() {
                    kraus.push(kron(&kron(&kron(a, b), c), d));
                }
            }
        }
    }
    let first = Channel::from_kraus("ad4", kraus).unwrap();
    let then = Channel::rotation("XZZY", 0.7)
        .unwrap()
        .compose(&Channel::pauli(&[("ZZZZ", 0.1), ("XIIX", 0.2)]).unwrap())
        .unwrap()
        .compose(&first)
        .unwrap();
    let start = std::time::Instant::now();
    let both = first.compose(&then).unwrap();
    assert!(start.elapsed().as_secs_f64() < 10.0);
    let mut a = random_state_matrix(5, 3);
    let mut b = a.clone();
    a.apply_channel(&both, &[4, 1, 0, 2]).unwrap();
    b.apply_channel(&first, &[4, 1, 0, 2]).unwrap();
    b.apply_channel(&then, &[4, 1, 0, 2]).unwrap();
    assert!(max_diff(&a.to_matrix(), &b.to_matrix()) < 1e-12);
    assert!(both.kraus_operators().len() <= 256);
}
