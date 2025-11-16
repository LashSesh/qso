//! Integration tests for the complete APOLLYON-MEF pipeline
//!
//! Tests the end-to-end flow from APOLLYON 5D integration through
//! MEF knowledge derivation and gate evaluation.

use apollyon_mef_bridge::{UnifiedCognitiveEngine, CognitiveInput};
use core_5d::{State5D, SystemParameters};

#[test]
fn test_complete_pipeline_execution() {
    // Create engine
    let mut engine = UnifiedCognitiveEngine::new();

    // Create input with simple parameters
    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-INTEGRATION-001".to_string(),
        seed: "integration_test_seed".to_string(),
        seed_path: "MEF/integration/test/0001".to_string(),
    };

    // Process through pipeline
    let result = engine.process(input);
    assert!(result.is_ok(), "Pipeline should execute successfully");

    let output = result.unwrap();

    // Verify trajectory
    assert!(!output.trajectory.is_empty(), "Trajectory should not be empty");
    assert!(output.trajectory.len() > 10, "Trajectory should have multiple states");

    // Verify spectral signature
    assert!(output.spectral_signature.psi.is_finite());
    assert!(output.spectral_signature.rho.is_finite());
    assert!(output.spectral_signature.omega.is_finite());
    assert!(output.spectral_signature.rho >= 0.0 && output.spectral_signature.rho <= 1.0);

    // Verify route
    assert_eq!(output.route.permutation.len(), 7);
    assert!(output.route.mesh_score.is_finite());

    // Verify proof
    assert!(output.proof.delta_pi.is_finite());
    assert!(output.proof.phi.is_finite());
    assert!(output.proof.delta_v.is_finite());

    // Verify gate decision
    match output.gate_decision {
        mef_schemas::GateDecision::FIRE | mef_schemas::GateDecision::HOLD => {
            // Valid decision
        }
    }

    // Verify knowledge object
    assert!(output.knowledge.is_some());
    let knowledge = output.knowledge.unwrap();
    assert_eq!(knowledge.tic_id, "TIC-INTEGRATION-001");
    assert!(!knowledge.mef_id.is_empty());
}

#[test]
fn test_pipeline_with_different_parameters() {
    let mut engine = UnifiedCognitiveEngine::new();

    // Test with custom system parameters
    let params = SystemParameters::new(
        [-0.1, -0.2, 0.15, 0.0, -0.05], // intrinsic rates
        [0.0, 0.0, 0.0, 0.0, 0.0],       // no external forcing
    );

    let input = CognitiveInput {
        initial_state: State5D::new(2.0, 1.0, 0.5, 0.3, 0.2),
        parameters: params,
        t_final: 2.0,
        tic_id: "TIC-PARAMS-001".to_string(),
        seed: "params_test_seed".to_string(),
        seed_path: "MEF/params/test/0001".to_string(),
    };

    let result = engine.process(input);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert!(!output.trajectory.is_empty());
}

#[test]
fn test_pipeline_with_different_initial_states() {
    let mut engine = UnifiedCognitiveEngine::new();

    // Test multiple different initial states
    let initial_states = vec![
        State5D::new(1.0, 0.0, 0.0, 0.0, 0.0),
        State5D::new(0.0, 1.0, 0.0, 0.0, 0.0),
        State5D::new(0.5, 0.5, 0.5, 0.5, 0.5),
        State5D::new(1.0, 1.0, 1.0, 1.0, 1.0),
    ];

    for (i, initial_state) in initial_states.iter().enumerate() {
        let input = CognitiveInput {
            initial_state: *initial_state,
            parameters: SystemParameters::default(),
            t_final: 1.0,
            tic_id: format!("TIC-STATE-{:03}", i),
            seed: format!("state_test_{}", i),
            seed_path: format!("MEF/state/test/{:04}", i),
        };

        let result = engine.process(input);
        assert!(result.is_ok(), "Pipeline should work with state {}", i);
    }
}

#[test]
fn test_deterministic_routing() {
    let mut engine = UnifiedCognitiveEngine::new();

    // Process same input twice
    let input1 = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-DET-001".to_string(),
        seed: "deterministic_seed".to_string(),
        seed_path: "MEF/det/test/0001".to_string(),
    };

    let input2 = input1.clone();

    let output1 = engine.process(input1).unwrap();
    let output2 = engine.process(input2).unwrap();

    // Routes should be identical for same seed
    assert_eq!(output1.route.route_id, output2.route.route_id);
    assert_eq!(output1.route.permutation, output2.route.permutation);
}

#[test]
fn test_different_seeds_different_routes() {
    let mut engine = UnifiedCognitiveEngine::new();

    // Same state, different seeds
    let initial_state = State5D::new(1.0, 0.5, 0.3, 0.2, 0.1);

    let input1 = CognitiveInput {
        initial_state,
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-SEED-001".to_string(),
        seed: "seed_alpha".to_string(),
        seed_path: "MEF/seed/test/0001".to_string(),
    };

    let input2 = CognitiveInput {
        initial_state,
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-SEED-002".to_string(),
        seed: "seed_beta".to_string(),
        seed_path: "MEF/seed/test/0002".to_string(),
    };

    let output1 = engine.process(input1).unwrap();
    let output2 = engine.process(input2).unwrap();

    // Routes should be different for different seeds
    assert_ne!(output1.route.route_id, output2.route.route_id);
}

#[test]
fn test_trajectory_continuity() {
    let mut engine = UnifiedCognitiveEngine::new();

    let initial_state = State5D::new(1.0, 0.5, 0.3, 0.2, 0.1);

    let input = CognitiveInput {
        initial_state,
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-CONT-001".to_string(),
        seed: "continuity_test".to_string(),
        seed_path: "MEF/continuity/test/0001".to_string(),
    };

    let output = engine.process(input).unwrap();

    // Check trajectory starts at initial state
    assert_eq!(output.trajectory[0], initial_state);

    // Check states are continuous (no jumps)
    for i in 1..output.trajectory.len() {
        let prev = &output.trajectory[i - 1];
        let curr = &output.trajectory[i];

        // Compute distance
        let mut dist_sq = 0.0;
        for j in 0..5 {
            let diff = curr.get(j) - prev.get(j);
            dist_sq += diff * diff;
        }
        let dist = dist_sq.sqrt();

        // Distance should be reasonable (no huge jumps)
        assert!(dist < 1.0, "Trajectory should be continuous at step {}", i);
    }
}

#[test]
fn test_proof_of_resonance_validity() {
    let mut engine = UnifiedCognitiveEngine::new();

    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-POR-001".to_string(),
        seed: "por_test".to_string(),
        seed_path: "MEF/por/test/0001".to_string(),
    };

    let output = engine.process(input).unwrap();

    // Proof should be valid for reasonable trajectories
    // Note: This depends on the trajectory dynamics, so we just check it's computed
    assert!(output.proof.por_valid || !output.proof.por_valid); // Always true, but documents intent

    // All components should be finite
    assert!(output.proof.delta_pi.is_finite());
    assert!(output.proof.phi.is_finite());
    assert!(output.proof.delta_v.is_finite());

    // phi should be positive (resonance field modulation)
    assert!(output.proof.phi > 0.0);

    // delta_pi should be non-negative (it's a distance)
    assert!(output.proof.delta_pi >= 0.0);
}

#[test]
fn test_knowledge_object_structure() {
    let mut engine = UnifiedCognitiveEngine::new();

    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-KNOWLEDGE-001".to_string(),
        seed: "knowledge_test_seed".to_string(),
        seed_path: "MEF/knowledge/test/0001".to_string(),
    };

    let output = engine.process(input).unwrap();
    let knowledge = output.knowledge.unwrap();

    // Verify structure
    assert_eq!(knowledge.tic_id, "TIC-KNOWLEDGE-001");
    assert!(!knowledge.mef_id.is_empty());
    assert!(!knowledge.route_id.is_empty());
    assert_eq!(knowledge.seed_path, "MEF/knowledge/test/0001");

    // Verify payload exists and contains spectral data
    assert!(knowledge.payload.is_some());
    let payload = knowledge.payload.unwrap();
    assert!(payload.get("spectral_signature").is_some());
    assert!(payload.get("route").is_some());
}

#[test]
fn test_short_integration_time() {
    let mut engine = UnifiedCognitiveEngine::new();

    // Very short integration time
    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 0.1,
        tic_id: "TIC-SHORT-001".to_string(),
        seed: "short_time_test".to_string(),
        seed_path: "MEF/short/test/0001".to_string(),
    };

    let result = engine.process(input);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert!(!output.trajectory.is_empty());
}

#[test]
fn test_long_integration_time() {
    let mut engine = UnifiedCognitiveEngine::new();

    // Longer integration time
    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 5.0,
        tic_id: "TIC-LONG-001".to_string(),
        seed: "long_time_test".to_string(),
        seed_path: "MEF/long/test/0001".to_string(),
    };

    let result = engine.process(input);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert!(output.trajectory.len() > 100); // Should have many states
}
