//! Regression tests for invalid or extreme input reaching the public API:
//! it must produce an error, never a panic or a silently wrong result.

use metatron_qso::algorithms;
use metatron_qso::algorithms::classifier::VariationalClassifier;
use metatron_qso::algorithms::gradients::{GradientMethod, expectation_and_gradient};
use metatron_qso::prelude::*;

#[test]
fn empty_and_tiny_graphs() {
    let empty = Graph::new(0);
    let s = empty.summary();
    assert_eq!(
        (s.nodes, s.edges, s.regular, s.diameter),
        (0, 0, false, None)
    );
    assert_eq!(generators::star(0).num_nodes(), 0);
    assert_eq!(generators::star(1).num_edges(), 0);
    assert!(SpatialSearch::critical_gamma(&Graph::new(1)).is_err());
}

#[test]
fn oversized_registers_are_errors() {
    let big = Circuit::new(40);
    let z: PauliSum = "Z0".parse().unwrap();
    assert!(matches!(
        expectation_and_gradient(&big, &z, &[], None, GradientMethod::Adjoint),
        Err(QsoError::TooManyQubits { .. })
    ));
    let mut ansatz = Circuit::new(40);
    let p = ansatz.new_parameter();
    ansatz.ry(0, p);
    assert!(Vqe::new(z.clone(), ansatz).is_err());
    assert!(z.diagonal(40).is_err());
    assert!(z.ground_state_energy(40).is_err());
    assert!(VariationalClassifier::new(40, 2, 1).is_err());
}

#[test]
#[should_panic(expected = "qubits 0..64")]
fn pauli_string_rejects_qubit_64_instead_of_wrapping() {
    let _ = PauliString::single(64, Pauli::Z);
}

#[test]
fn classifier_validates_data() {
    let mut model = VariationalClassifier::new(2, 2, 1).unwrap();
    let xs = vec![vec![0.1, 0.2], vec![-0.3, 0.4]];
    let ys = vec![true, false];
    assert!(
        model
            .fit(&[vec![f64::NAN, 0.0]], &[true], &Options::default(), 0)
            .is_err()
    );
    model
        .fit(
            &xs,
            &ys,
            &Options {
                max_iterations: 5,
                ..Default::default()
            },
            0,
        )
        .unwrap();
    assert!(model.predict(&[f64::INFINITY, 0.0]).is_err());
    assert!(model.accuracy(&xs, &ys[..1]).is_err());
}

#[test]
fn readout_probabilities_are_validated() {
    assert!(NoiseModel::default().with_readout(0.1, 1.5).is_err());
    let bad = NoiseModel {
        readout: Some((-0.1, 0.0)),
        ..Default::default()
    };
    let mut c = Circuit::new(1);
    c.h(0);
    assert!(DensityMatrixBackend::new(bad).run(&c, &[], 10, 0).is_err());
}

#[test]
fn error_messages_are_specific() {
    let msg = SpatialSearch::critical_gamma(&Graph::new(1))
        .unwrap_err()
        .to_string();
    assert!(msg.contains("2048"), "{msg}");
}

// ---- Findings of the independent review -------------------------------

#[test]
fn crafted_json_cannot_reach_the_kernels() {
    let mut c = Circuit::new(2);
    c.cx(0, 1);
    let json = serde_json::to_string(&c).unwrap();
    let crafted = json.replace("\"targets\":[1]", "\"targets\":[40]");
    assert_ne!(json, crafted);
    assert!(serde_json::from_str::<Circuit>(&crafted).is_err());
    let duplicate = json.replace("\"targets\":[1]", "\"targets\":[0]");
    assert!(serde_json::from_str::<Circuit>(&duplicate).is_err());
    // A state vector whose amplitude count does not match its register.
    let bad_state = r#"{"n_qubits":30,"amps":[[1.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]}"#;
    assert!(serde_json::from_str::<StateVector>(bad_state).is_err());
    let bad_rho = r#"{"n_qubits":1,"data":[[1.0,0.0],[0.0,0.0]]}"#;
    assert!(serde_json::from_str::<DensityMatrix>(bad_rho).is_err());
    // Valid objects round-trip.
    let back: Circuit = serde_json::from_str(&json).unwrap();
    assert_eq!(back, c);
    let psi = StateVector::from_circuit(&c, &[]).unwrap();
    let psi_back: StateVector =
        serde_json::from_str(&serde_json::to_string(&psi).unwrap()).unwrap();
    assert_eq!(psi_back, psi);
}

#[test]
fn unvalidated_instructions_are_rejected() {
    use metatron_qso::quantum::Instruction;
    let mut psi = StateVector::zero(2);
    let control_is_target = Instruction {
        kind: GateKind::X,
        targets: vec![1],
        controls: vec![1],
        params: vec![],
        matrix: None,
    };
    assert!(psi.apply_instruction(&control_is_target, &[]).is_err());
    let rxx_same = Instruction {
        kind: GateKind::RXX,
        targets: vec![1, 1],
        controls: vec![],
        params: vec![0.3.into()],
        matrix: None,
    };
    assert!(psi.apply_instruction(&rxx_same, &[]).is_err());
    let missing_param = Instruction {
        kind: GateKind::RX,
        targets: vec![0],
        controls: vec![],
        params: vec![],
        matrix: None,
    };
    assert!(psi.apply_instruction(&missing_param, &[]).is_err());
}

#[test]
fn custom_unitaries_survive_serialisation() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let h = vec![
        C64::new(s, 0.0),
        C64::new(s, 0.0),
        C64::new(s, 0.0),
        C64::new(-s, 0.0),
    ];
    let mut c = Circuit::new(2);
    c.unitary(h, &[0], &[]).unwrap().cx(0, 1);
    let back: Circuit = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back, c);
    let a = StateVector::from_circuit(&c, &[]).unwrap();
    let b = StateVector::from_circuit(&back, &[]).unwrap();
    assert!((a.fidelity(&b).unwrap() - 1.0).abs() < 1e-14);
}

#[test]
fn controlled_gate_export_is_convention_independent() {
    // Controlled S, Tdg, RY(4), RX(π) and U: exported without `cu3`, built
    // from uncontrolled u1/u3/cx only, so reinterpreting u1 as RZ (the
    // original qelib1 phase convention) changes at most a global phase.
    let mut c = Circuit::new(2);
    c.h(0).h(1);
    c.controlled(GateKind::S, &[1], &[0], &[]).unwrap();
    c.controlled(GateKind::Tdg, &[0], &[1], &[]).unwrap();
    c.cry(0, 1, 4.0).crx(1, 0, std::f64::consts::PI);
    c.controlled(
        GateKind::U,
        &[1],
        &[0],
        &[0.83.into(), (-1.37).into(), 2.41.into()],
    )
    .unwrap();
    let qasm = to_qasm2(&c, &[], false).unwrap();
    assert!(!qasm.contains("cu3"), "{qasm}");
    let parsed = from_qasm2(&qasm).unwrap();
    let reference = StateVector::from_circuit(&c, &[]).unwrap();
    let qiskit = StateVector::from_circuit(&parsed, &[]).unwrap();
    assert!((reference.fidelity(&qiskit).unwrap() - 1.0).abs() < 1e-12);
    // Original-qelib1 reading: u1(λ) = RZ(λ) (differs by e^{iλ/2}).
    let rz_text = qasm.replace("u1(", "rz(");
    let original = StateVector::from_circuit(&from_qasm2(&rz_text).unwrap(), &[]).unwrap();
    assert!((reference.fidelity(&original).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn qasm_parameters_may_nest_parentheses() {
    let src = "OPENQASM 2.0;\nqreg q[1];\nrz(-(pi/2)) q[0];\nu3(sin(0.5),0,0) q[0];\nrz(2*(pi/4)) q[0];\nrz(-2^2) q[0];\n";
    let c = from_qasm2(src).unwrap();
    assert_eq!(c.len(), 4);
    let angle = |k: usize| match c.instructions()[k].params[0] {
        Param::Value(v) => v,
        _ => unreachable!(),
    };
    assert!((angle(0) + std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!((angle(1) - 0.5f64.sin()).abs() < 1e-15);
    assert!((angle(3) + 4.0).abs() < 1e-15, "-2^2 must be -4");
    assert!(from_qasm2("OPENQASM 2.0;\nqreg q[1];\nrz((pi q[0];\n").is_err());
}

#[test]
fn pauli_sign_after_star_is_unary() {
    let a: PauliSum = "2*-X0".parse().unwrap();
    assert_eq!(a.terms(), &[(-2.0, PauliString::single(0, Pauli::X))]);
    let b: PauliSum = "0.5 * -Z0 + 1".parse().unwrap();
    assert_eq!(b.constant(), 1.0);
    assert!(
        b.terms()
            .contains(&(-0.5, PauliString::single(0, Pauli::Z)))
    );
    assert!("X0 *".parse::<PauliSum>().is_err());
    assert!("* X0".parse::<PauliSum>().is_err());
}

#[test]
fn vqe_reports_unsupported_gradient_methods() {
    let mut ansatz = Circuit::new(3);
    let t = ansatz.new_parameter();
    ansatz.h(0).h(1);
    ansatz
        .controlled(GateKind::RY, &[2], &[0, 1], &[t.into()])
        .unwrap();
    let h: PauliSum = "Z2".parse().unwrap();
    let vqe = Vqe::new(h, ansatz).unwrap();
    assert!(
        vqe.clone()
            .with_gradient(GradientMethod::ParameterShift)
            .run()
            .is_err()
    );
    assert!(vqe.with_gradient(GradientMethod::Adjoint).run().is_ok());
}

#[test]
fn non_finite_values_are_errors() {
    use metatron_qso::linalg::{KrylovOptions, expm_multiply};
    let v = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
    let id = |x: &[C64], y: &mut [C64]| y.copy_from_slice(x);
    for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(expm_multiply(id, &v, t, &KrylovOptions::default()).is_err());
    }
    let mut c = Circuit::new(1);
    let p = c.new_parameter();
    c.rx(0, p);
    assert!(to_qasm2(&c, &[f64::NAN], false).is_err());
    assert!(StateVector::from_circuit(&c, &[f64::INFINITY]).is_err());
    let bad_scale = metatron_qso::quantum::Instruction {
        kind: GateKind::RX,
        targets: vec![0],
        controls: vec![],
        params: vec![Param::Symbol(Parameter {
            index: 0,
            scale: f64::NAN,
            offset: 0.0,
        })],
        matrix: None,
    };
    assert!(c.try_push(bad_scale).is_err());
}

#[test]
fn large_sizes_do_not_overflow() {
    let c = algorithms::qft::qft(70, false);
    assert!(c.instructions().iter().all(|i| {
        i.params
            .iter()
            .all(|p| matches!(p, Param::Value(v) if v.is_finite() && *v > 0.0))
    }));
    let k = grover::optimal_iterations(64, 1);
    assert!(k > 1_000_000_000);
    assert!(grover::theoretical_success(64, 1, 0) > 0.0);
}

#[test]
#[should_panic(expected = "outside")]
fn density_matrix_index_is_checked() {
    let rho = DensityMatrix::zero(1).unwrap();
    let _ = rho.get(2, 0);
}

#[test]
fn noise_models_reject_misplaced_channels() {
    let model = NoiseModel {
        single_qubit: vec![Channel::depolarizing_2q(0.1).unwrap()],
        ..Default::default()
    };
    let mut c = Circuit::new(2);
    c.h(0);
    assert!(
        DensityMatrixBackend::new(model)
            .run(&c, &[], 10, 0)
            .is_err()
    );
}
