//! Integration tests for MEF core functionality
//!
//! This test suite validates the integration between the core MEF components:
//! - Spiral snapshots
//! - Ledger blocks
//! - TIC (Temporal Information Crystals)
//! - HDAG (Hierarchical Directed Acyclic Graph)

use anyhow::Result;
use mef_ledger::MEFLedger;
use mef_spiral::SpiralSnapshot;
use serde_json::json;

#[test]
fn test_spiral_ledger_integration() -> Result<()> {
    // Create a temporary directory for test data
    let temp_dir = tempfile::tempdir()?;
    let storage_path = temp_dir.path().join("storage");
    let ledger_path = temp_dir.path().join("ledger");

    // Initialize spiral snapshot system
    let spiral_config = mef_spiral::SpiralConfig::default();
    let spiral = SpiralSnapshot::new(spiral_config, storage_path.to_str().unwrap())?;

    // Initialize ledger
    let mut ledger = MEFLedger::new(ledger_path.to_str().unwrap())?;

    // Create test data
    let test_data = json!({
        "type": "test_event",
        "value": 42,
        "timestamp": "2025-10-15T18:00:00Z"
    });

    // Create a spiral snapshot
    let snapshot = spiral.create_snapshot(&test_data, "TEST_SEED_001", None)?;

    // Create TIC data for ledger
    let tic = json!({
        "tic_id": format!("tic-{}", snapshot.id),
        "seed": "TEST_SEED_001",
        "fixpoint": snapshot.coordinates,
        "invariants": {"variance": 0.1},
        "sigma_bar": {"psi": 0.5},
        "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
        "proof": {"merkle_root": "test_root"}
    });

    // Create snapshot data for ledger
    let snapshot_data = json!({
        "id": snapshot.id,
        "coordinates": snapshot.coordinates,
        "phase": snapshot.phase
    });

    // Append block to ledger
    let block = ledger.append_block(&tic, &snapshot_data)?;

    // Verify the block was created correctly
    assert_eq!(block.index, 0);
    assert!(block.hash.len() > 0);

    // Verify chain integrity
    let valid = ledger.verify_chain_integrity(0)?;
    assert!(valid, "Chain integrity verification failed");

    // Get chain statistics
    let stats = ledger.get_chain_statistics()?;
    assert_eq!(stats.total_blocks, 1);

    // Save the snapshot
    let snapshot_file = spiral.save_snapshot(&snapshot)?;
    assert!(snapshot_file.exists(), "Snapshot file not created");

    // Load the snapshot
    let loaded_snapshot = spiral.load_snapshot(&snapshot.id)?;
    assert!(loaded_snapshot.is_some(), "Failed to load snapshot");

    let loaded = loaded_snapshot.unwrap();
    assert_eq!(loaded.id, snapshot.id);
    assert_eq!(loaded.coordinates, snapshot.coordinates);

    Ok(())
}

#[test]
fn test_multiple_blocks_chain_integrity() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let ledger_path = temp_dir.path().join("ledger");

    let mut ledger = MEFLedger::new(ledger_path.to_str().unwrap())?;

    // Create multiple blocks
    for i in 0..10 {
        let tic = json!({
            "tic_id": format!("tic-{}", i),
            "seed": format!("SEED_{}", i),
            "fixpoint": [0.1 * i as f64, 0.2 * i as f64, 0.3 * i as f64],
            "invariants": {"variance": 0.1},
            "sigma_bar": {"psi": 0.5},
            "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
            "proof": {"merkle_root": format!("root_{}", i)}
        });

        let snapshot = json!({
            "id": format!("snap-{}", i),
            "coordinates": [0.1 * i as f64, 0.2 * i as f64, 0.3 * i as f64, 0.4 * i as f64, 0.5 * i as f64]
        });

        let block = ledger.append_block(&tic, &snapshot)?;
        assert_eq!(block.index, i as i32);
    }

    // Verify chain integrity for all blocks
    for i in 0..10 {
        let valid = ledger.verify_chain_integrity(i)?;
        assert!(valid, "Chain integrity failed at block {}", i);
    }

    let stats = ledger.get_chain_statistics()?;
    assert_eq!(stats.total_blocks, 10);

    Ok(())
}

#[test]
fn test_deterministic_snapshot_creation() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let storage_path = temp_dir.path().join("storage");

    let spiral_config = mef_spiral::SpiralConfig::default();
    let spiral = SpiralSnapshot::new(spiral_config.clone(), storage_path.to_str().unwrap())?;

    let test_data = json!({
        "value": "deterministic_test",
        "number": 12345
    });

    // Create two snapshots with the same seed and data
    let snapshot1 = spiral.create_snapshot(&test_data, "DETERMINISTIC_SEED", None)?;

    // Create a new spiral instance to ensure fresh state
    let storage_path2 = temp_dir.path().join("storage2");
    let spiral2 = SpiralSnapshot::new(spiral_config, storage_path2.to_str().unwrap())?;
    let snapshot2 = spiral2.create_snapshot(&test_data, "DETERMINISTIC_SEED", None)?;

    // Verify both snapshots are identical
    assert_eq!(snapshot1.coordinates, snapshot2.coordinates);
    assert_eq!(snapshot1.phase, snapshot2.phase);

    Ok(())
}
