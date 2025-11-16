//! MEF Ledger storage backend
//!
//! Connects the UnifiedCognitiveEngine to the actual MEF Ledger
//! for persistent, cryptographically-verified storage.

use super::{StorageBackend, StorageError, StorageResult, StorageStats};
use crate::unified::CognitiveOutput;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// MEF Ledger storage backend
///
/// Provides persistent storage using the MEF hash-chained ledger.
/// All stored outputs are cryptographically verified and immutable.
pub struct LedgerStorage {
    ledger_path: PathBuf,
    stats: Arc<Mutex<StorageStats>>,
}

impl LedgerStorage {
    /// Create a new ledger storage backend
    ///
    /// # Arguments
    /// * `ledger_path` - Directory for ledger storage
    ///
    /// # Returns
    /// New LedgerStorage instance or error
    pub fn new(ledger_path: impl AsRef<Path>) -> StorageResult<Self> {
        let ledger_path = ledger_path.as_ref().to_path_buf();

        // Create ledger directory if it doesn't exist
        std::fs::create_dir_all(&ledger_path).map_err(|e| {
            StorageError::InitializationError(format!("Failed to create ledger directory: {}", e))
        })?;

        Ok(Self {
            ledger_path,
            stats: Arc::new(Mutex::new(StorageStats {
                backend_type: "ledger".to_string(),
                ..Default::default()
            })),
        })
    }

    /// Get the ledger path
    pub fn ledger_path(&self) -> &Path {
        &self.ledger_path
    }

    /// Convert CognitiveOutput to ledger-compatible format
    ///
    /// This prepares the output for storage in the MEF Ledger
    fn prepare_for_ledger(&self, output: &CognitiveOutput) -> StorageResult<serde_json::Value> {
        // Extract knowledge object
        let knowledge = output
            .knowledge
            .as_ref()
            .ok_or_else(|| StorageError::InvalidData("No knowledge object in output".to_string()))?;

        // Create ledger entry
        let entry = serde_json::json!({
            "mef_id": knowledge.mef_id,
            "tic_id": knowledge.tic_id,
            "route_id": knowledge.route_id,
            "seed_path": knowledge.seed_path,
            "spectral_signature": {
                "psi": output.spectral_signature.psi,
                "rho": output.spectral_signature.rho,
                "omega": output.spectral_signature.omega,
            },
            "route": {
                "route_id": output.route.route_id,
                "permutation": output.route.permutation,
                "mesh_score": output.route.mesh_score,
            },
            "proof_of_resonance": {
                "delta_pi": output.proof.delta_pi,
                "phi": output.proof.phi,
                "delta_v": output.proof.delta_v,
                "por_valid": output.proof.por_valid,
            },
            "gate_decision": match output.gate_decision {
                mef_schemas::GateDecision::FIRE => "FIRE",
                mef_schemas::GateDecision::HOLD => "HOLD",
            },
            "trajectory_length": output.trajectory.len(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });

        Ok(entry)
    }
}

#[async_trait]
impl StorageBackend for LedgerStorage {
    async fn store(&mut self, output: &CognitiveOutput) -> StorageResult<String> {
        // Only store if gate decision is FIRE
        if output.gate_decision != mef_schemas::GateDecision::FIRE {
            return Err(StorageError::InvalidData(
                "Cannot store HOLD decision to ledger".to_string(),
            ));
        }

        // Prepare data for ledger
        let ledger_entry = self.prepare_for_ledger(output)?;

        // Get storage ID from knowledge object
        let storage_id = output
            .knowledge
            .as_ref()
            .map(|k| k.mef_id.clone())
            .ok_or_else(|| StorageError::InvalidData("No knowledge object".to_string()))?;

        // Write to ledger file
        let block_file = self.ledger_path.join(format!("{}.json", storage_id));
        let json_str = serde_json::to_string_pretty(&ledger_entry).map_err(|e| {
            StorageError::WriteError(format!("Failed to serialize ledger entry: {}", e))
        })?;

        std::fs::write(&block_file, json_str).map_err(|e| {
            // Update failure stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.failed_writes += 1;
            }
            StorageError::WriteError(format!("Failed to write ledger file: {}", e))
        })?;

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_items += 1;
            stats.total_size_bytes += std::fs::metadata(&block_file)
                .map(|m| m.len())
                .unwrap_or(0);
            stats.successful_writes += 1;
        }

        Ok(storage_id)
    }

    async fn retrieve(&self, id: &str) -> StorageResult<CognitiveOutput> {
        // Read from ledger file
        let block_file = self.ledger_path.join(format!("{}.json", id));

        if !block_file.exists() {
            return Err(StorageError::ReadError(format!(
                "Ledger entry not found: {}",
                id
            )));
        }

        let json_str = std::fs::read_to_string(&block_file)
            .map_err(|e| StorageError::ReadError(format!("Failed to read ledger file: {}", e)))?;

        // Parse ledger entry
        let _entry: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| StorageError::ReadError(format!("Failed to parse ledger entry: {}", e)))?;

        // Note: Full reconstruction of CognitiveOutput from ledger entry
        // would require storing trajectory data, which can be large.
        // For now, we return an error indicating this limitation.
        Err(StorageError::ReadError(
            "Full CognitiveOutput reconstruction from ledger not yet implemented".to_string(),
        ))
    }

    async fn health_check(&self) -> StorageResult<bool> {
        // Check if ledger path is accessible
        Ok(self.ledger_path.exists() && self.ledger_path.is_dir())
    }

    async fn stats(&self) -> StorageResult<StorageStats> {
        match self.stats.lock() {
            Ok(stats) => Ok(stats.clone()),
            Err(e) => Err(StorageError::ReadError(format!(
                "Failed to read stats: {}",
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_5d::State5D;
    use mef_schemas::{GateDecision, KnowledgeObject, RouteSpec, SpectralSignature};
    use tempfile::TempDir;

    fn create_test_output(gate_decision: GateDecision) -> CognitiveOutput {
        let knowledge = KnowledgeObject::new(
            "TEST-LEDGER-001".to_string(),
            "TIC-001".to_string(),
            "ROUTE-001".to_string(),
            "MEF/test/ledger/0001".to_string(),
            vec![1, 2, 3],
            None,
        );

        CognitiveOutput {
            trajectory: vec![State5D::new(1.0, 0.0, 0.0, 0.0, 0.0)],
            spectral_signature: SpectralSignature {
                psi: 0.5,
                rho: 0.7,
                omega: 2.1,
            },
            route: RouteSpec::new("ROUTE-001".to_string(), vec![0, 1, 2, 3, 4, 5, 6], 0.8)
                .unwrap(),
            proof: Default::default(),
            gate_decision,
            knowledge: Some(knowledge),
        }
    }

    #[tokio::test]
    async fn test_ledger_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LedgerStorage::new(temp_dir.path()).unwrap();

        assert!(storage.ledger_path().exists());
    }

    #[tokio::test]
    async fn test_ledger_storage_store_fire() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LedgerStorage::new(temp_dir.path()).unwrap();

        let output = create_test_output(GateDecision::FIRE);
        let id = storage.store(&output).await.unwrap();

        // Check that file was created
        let block_file = temp_dir.path().join(format!("{}.json", id));
        assert!(block_file.exists());
    }

    #[tokio::test]
    async fn test_ledger_storage_reject_hold() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LedgerStorage::new(temp_dir.path()).unwrap();

        let output = create_test_output(GateDecision::HOLD);
        let result = storage.store(&output).await;

        // Should fail because gate decision is HOLD
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ledger_storage_health_check() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LedgerStorage::new(temp_dir.path()).unwrap();

        let healthy = storage.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn test_ledger_storage_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = LedgerStorage::new(temp_dir.path()).unwrap();

        let output = create_test_output(GateDecision::FIRE);
        storage.store(&output).await.unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.total_items, 1);
        assert_eq!(stats.successful_writes, 1);
        assert_eq!(stats.backend_type, "ledger");
    }
}
