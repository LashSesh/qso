//! Spiral Storage management for 5D snapshots.
//! File-based persistence with indexing and retrieval.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Snapshot metadata in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub id: String,
    pub timestamp: String,
    pub seed: String,
    pub phase: f64,
    pub por: String,
    pub file: String,
}

/// TIC metadata in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicMetadata {
    pub id: String,
    pub seed: String,
    pub source_snapshot: String,
    pub window: usize,
    pub por: String,
    pub file: String,
}

/// Storage index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub created: String,
    pub last_updated: String,
    pub version: String,
}

/// Storage index structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageIndex {
    pub snapshots: HashMap<String, SnapshotMetadata>,
    pub tics: HashMap<String, TicMetadata>,
    pub metadata: IndexMetadata,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatistics {
    pub snapshots: SnapshotStats,
    pub tics: TicStats,
    pub storage: StorageInfo,
    pub metadata: IndexMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotStats {
    pub count: usize,
    pub indexed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicStats {
    pub count: usize,
    pub indexed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub total_size_mb: f64,
}

/// Integrity check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityResults {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Storage backend for Spiral snapshots
pub struct SpiralStorage {
    store_path: PathBuf,
    index_file: PathBuf,
    index: StorageIndex,
}

impl SpiralStorage {
    /// Create new storage system
    ///
    /// # Arguments
    /// * `store_path` - Root directory for storage
    ///
    /// # Returns
    /// New SpiralStorage instance
    pub fn new<P: AsRef<Path>>(store_path: P) -> Result<Self, std::io::Error> {
        let store_path = store_path.as_ref().to_path_buf();
        fs::create_dir_all(&store_path)?;

        let index_file = store_path.join("index.json");
        let index = Self::load_index(&index_file)?;

        Ok(Self {
            store_path,
            index_file,
            index,
        })
    }

    /// Load storage index from disk
    fn load_index(index_file: &Path) -> Result<StorageIndex, std::io::Error> {
        if index_file.exists() {
            let content = fs::read_to_string(index_file)?;
            serde_json::from_str(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        } else {
            Ok(StorageIndex {
                snapshots: HashMap::new(),
                tics: HashMap::new(),
                metadata: IndexMetadata {
                    created: Utc::now().to_rfc3339(),
                    last_updated: Utc::now().to_rfc3339(),
                    version: "1.0.0".to_string(),
                },
            })
        }
    }

    /// Save index to disk
    fn save_index(&mut self) -> Result<(), std::io::Error> {
        self.index.metadata.last_updated = Utc::now().to_rfc3339();
        let content = serde_json::to_string_pretty(&self.index)?;
        fs::write(&self.index_file, content)?;
        Ok(())
    }

    /// Store a snapshot and update index
    ///
    /// # Arguments
    /// * `snapshot` - Snapshot data as JSON
    ///
    /// # Returns
    /// File path of stored snapshot
    pub fn store_snapshot(&mut self, snapshot: &serde_json::Value) -> Result<String, String> {
        let snapshot_id = snapshot["id"]
            .as_str()
            .ok_or("Missing snapshot id")?
            .to_string();
        let file_path = self.store_path.join(format!("{}.spiral", snapshot_id));

        // Write snapshot file
        let content = serde_json::to_string_pretty(snapshot)
            .map_err(|e| format!("Failed to serialize snapshot: {}", e))?;
        fs::write(&file_path, content)
            .map_err(|e| format!("Failed to write snapshot file: {}", e))?;

        // Update index
        let metadata = SnapshotMetadata {
            id: snapshot_id.clone(),
            timestamp: snapshot["timestamp"].as_str().unwrap_or("").to_string(),
            seed: snapshot["seed"].as_str().unwrap_or("").to_string(),
            phase: snapshot["phase"].as_f64().unwrap_or(0.0),
            por: snapshot["metrics"]["por"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            file: file_path
                .strip_prefix(&self.store_path)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string(),
        };

        self.index.snapshots.insert(snapshot_id, metadata);
        self.save_index()
            .map_err(|e| format!("Failed to save index: {}", e))?;

        Ok(file_path.to_string_lossy().to_string())
    }

    /// Retrieve a snapshot by ID
    ///
    /// # Arguments
    /// * `snapshot_id` - Snapshot UUID
    ///
    /// # Returns
    /// Snapshot data or None
    pub fn retrieve_snapshot(
        &self,
        snapshot_id: &str,
    ) -> Result<Option<serde_json::Value>, String> {
        let file_path = self.store_path.join(format!("{}.spiral", snapshot_id));

        if !file_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read snapshot file: {}", e))?;
        let snapshot = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse snapshot JSON: {}", e))?;

        Ok(Some(snapshot))
    }

    /// List snapshots with optional filtering
    ///
    /// # Arguments
    /// * `seed` - Optional filter by seed
    /// * `por_filter` - Optional filter by PoR status
    ///
    /// # Returns
    /// List of snapshot metadata
    pub fn list_snapshots(
        &self,
        seed: Option<&str>,
        por_filter: Option<&str>,
    ) -> Vec<SnapshotMetadata> {
        let mut results: Vec<SnapshotMetadata> = self
            .index
            .snapshots
            .values()
            .filter(|meta| {
                // Apply filters
                if let Some(s) = seed {
                    if meta.seed != s {
                        return false;
                    }
                }
                if let Some(por) = por_filter {
                    if meta.por != por {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // Sort by timestamp
        results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        results
    }

    /// Store a TIC and update index
    ///
    /// # Arguments
    /// * `tic` - TIC data as JSON
    ///
    /// # Returns
    /// File path of stored TIC
    pub fn store_tic(&mut self, tic: &serde_json::Value) -> Result<String, String> {
        let tic_id = tic["tic_id"].as_str().ok_or("Missing tic_id")?.to_string();
        let file_path = self.store_path.join(format!("{}.tic", tic_id));

        // Write TIC file
        let content = serde_json::to_string_pretty(tic)
            .map_err(|e| format!("Failed to serialize TIC: {}", e))?;
        fs::write(&file_path, content).map_err(|e| format!("Failed to write TIC file: {}", e))?;

        // Update index
        let metadata = TicMetadata {
            id: tic_id.clone(),
            seed: tic["seed"].as_str().unwrap_or("").to_string(),
            source_snapshot: tic["source_snapshot"].as_str().unwrap_or("").to_string(),
            window: tic["window"].as_u64().unwrap_or(0) as usize,
            por: tic["proof"]["por"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            file: file_path
                .strip_prefix(&self.store_path)
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string(),
        };

        self.index.tics.insert(tic_id, metadata);
        self.save_index()
            .map_err(|e| format!("Failed to save index: {}", e))?;

        Ok(file_path.to_string_lossy().to_string())
    }

    /// Retrieve a TIC by ID
    ///
    /// # Arguments
    /// * `tic_id` - TIC UUID
    ///
    /// # Returns
    /// TIC data or None
    pub fn retrieve_tic(&self, tic_id: &str) -> Result<Option<serde_json::Value>, String> {
        let file_path = self.store_path.join(format!("{}.tic", tic_id));

        if !file_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read TIC file: {}", e))?;
        let tic = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse TIC JSON: {}", e))?;

        Ok(Some(tic))
    }

    /// Get storage statistics
    ///
    /// # Returns
    /// Dictionary with storage metrics
    pub fn get_statistics(&self) -> Result<StorageStatistics, std::io::Error> {
        let snapshot_files: Vec<_> = fs::read_dir(&self.store_path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("spiral"))
            .collect();

        let tic_files: Vec<_> = fs::read_dir(&self.store_path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("tic"))
            .collect();

        let total_size: u64 = snapshot_files
            .iter()
            .chain(tic_files.iter())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        Ok(StorageStatistics {
            snapshots: SnapshotStats {
                count: snapshot_files.len(),
                indexed: self.index.snapshots.len(),
            },
            tics: TicStats {
                count: tic_files.len(),
                indexed: self.index.tics.len(),
            },
            storage: StorageInfo {
                total_files: snapshot_files.len() + tic_files.len(),
                total_size_bytes: total_size,
                total_size_mb: total_size as f64 / (1024.0 * 1024.0),
            },
            metadata: self.index.metadata.clone(),
        })
    }

    /// Verify storage integrity
    ///
    /// # Returns
    /// Integrity check results
    pub fn verify_integrity(&self) -> IntegrityResults {
        let mut results = IntegrityResults {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Check indexed snapshots exist
        for snap_id in self.index.snapshots.keys() {
            let file_path = self.store_path.join(format!("{}.spiral", snap_id));
            if !file_path.exists() {
                results
                    .errors
                    .push(format!("Missing snapshot file: {}", snap_id));
                results.valid = false;
            }
        }

        // Check indexed TICs exist
        for tic_id in self.index.tics.keys() {
            let file_path = self.store_path.join(format!("{}.tic", tic_id));
            if !file_path.exists() {
                results.errors.push(format!("Missing TIC file: {}", tic_id));
                results.valid = false;
            }
        }

        // Check for orphaned files
        if let Ok(entries) = fs::read_dir(&self.store_path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if ext == "spiral" {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if !self.index.snapshots.contains_key(stem) {
                                results
                                    .warnings
                                    .push(format!("Unindexed snapshot: {}", stem));
                            }
                        }
                    } else if ext == "tic" {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if !self.index.tics.contains_key(stem) {
                                results.warnings.push(format!("Unindexed TIC: {}", stem));
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Remove unindexed files
    ///
    /// # Returns
    /// Number of files removed
    pub fn cleanup_orphaned_files(&self) -> Result<usize, std::io::Error> {
        let mut removed_count = 0;

        // Find and remove orphaned snapshots
        for entry in fs::read_dir(&self.store_path)?.filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext == "spiral" {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if !self.index.snapshots.contains_key(stem) {
                            fs::remove_file(&path)?;
                            removed_count += 1;
                        }
                    }
                } else if ext == "tic" {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if !self.index.tics.contains_key(stem) {
                            fs::remove_file(&path)?;
                            removed_count += 1;
                        }
                    }
                }
            }
        }

        Ok(removed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_snapshot() -> serde_json::Value {
        serde_json::json!({
            "id": "test-snapshot-123",
            "timestamp": "2025-01-01T00:00:00Z",
            "seed": "test-seed",
            "phase": 1.5,
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metrics": {
                "resonance": 0.5,
                "stability": 0.8,
                "por": "valid"
            }
        })
    }

    fn create_test_tic() -> serde_json::Value {
        serde_json::json!({
            "tic_id": "test-tic-456",
            "seed": "test-seed",
            "source_snapshot": "test-snapshot-123",
            "window": 10,
            "proof": {
                "por": "valid"
            }
        })
    }

    #[test]
    fn test_create_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = SpiralStorage::new(temp_dir.path()).unwrap();

        assert!(temp_dir.path().exists());
        assert_eq!(storage.index.snapshots.len(), 0);
    }

    #[test]
    fn test_store_and_retrieve_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let snapshot = create_test_snapshot();
        let file_path = storage.store_snapshot(&snapshot).unwrap();

        assert!(PathBuf::from(&file_path).exists());

        let retrieved = storage.retrieve_snapshot("test-snapshot-123").unwrap();
        assert!(retrieved.is_some());

        let retrieved_snap = retrieved.unwrap();
        assert_eq!(retrieved_snap["id"], "test-snapshot-123");
    }

    #[test]
    fn test_list_snapshots() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let snapshot = create_test_snapshot();
        storage.store_snapshot(&snapshot).unwrap();

        let list = storage.list_snapshots(None, None);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "test-snapshot-123");
    }

    #[test]
    fn test_list_snapshots_with_filter() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let snapshot = create_test_snapshot();
        storage.store_snapshot(&snapshot).unwrap();

        let list = storage.list_snapshots(Some("test-seed"), None);
        assert_eq!(list.len(), 1);

        let list_wrong_seed = storage.list_snapshots(Some("wrong-seed"), None);
        assert_eq!(list_wrong_seed.len(), 0);
    }

    #[test]
    fn test_store_and_retrieve_tic() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let tic = create_test_tic();
        let file_path = storage.store_tic(&tic).unwrap();

        assert!(PathBuf::from(&file_path).exists());

        let retrieved = storage.retrieve_tic("test-tic-456").unwrap();
        assert!(retrieved.is_some());

        let retrieved_tic = retrieved.unwrap();
        assert_eq!(retrieved_tic["tic_id"], "test-tic-456");
    }

    #[test]
    fn test_get_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let snapshot = create_test_snapshot();
        storage.store_snapshot(&snapshot).unwrap();

        let stats = storage.get_statistics().unwrap();
        assert_eq!(stats.snapshots.count, 1);
        assert_eq!(stats.snapshots.indexed, 1);
        assert!(stats.storage.total_size_bytes > 0);
    }

    #[test]
    fn test_verify_integrity() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let snapshot = create_test_snapshot();
        storage.store_snapshot(&snapshot).unwrap();

        let results = storage.verify_integrity();
        assert!(results.valid);
        assert_eq!(results.errors.len(), 0);
    }

    #[test]
    fn test_verify_integrity_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = SpiralStorage::new(temp_dir.path()).unwrap();

        let snapshot = create_test_snapshot();
        storage.store_snapshot(&snapshot).unwrap();

        // Remove the file
        let file_path = temp_dir.path().join("test-snapshot-123.spiral");
        fs::remove_file(file_path).unwrap();

        let results = storage.verify_integrity();
        assert!(!results.valid);
        assert!(!results.errors.is_empty());
    }

    #[test]
    fn test_cleanup_orphaned_files() {
        let temp_dir = TempDir::new().unwrap();
        let storage = SpiralStorage::new(temp_dir.path()).unwrap();

        // Create an orphaned file
        let orphan_path = temp_dir.path().join("orphan.spiral");
        fs::write(&orphan_path, "{}").unwrap();

        let removed = storage.cleanup_orphaned_files().unwrap();
        assert_eq!(removed, 1);
        assert!(!orphan_path.exists());
    }
}
