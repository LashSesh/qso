//! Regression tests for invalid or extreme input reaching the public API:
//! it must produce an error, never a panic or a silently wrong result.

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
