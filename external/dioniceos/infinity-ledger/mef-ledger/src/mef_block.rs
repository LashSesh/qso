//! MEF Ledger implementation with hash-chained blocks.
//! Immutable audit log for TICs with deterministic hashing.
//!
//! Migrated from: MEF-Core_v1.0/src/ledger/mef_block.py

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Ledger index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerMetadata {
    pub created: String,
    pub last_updated: String,
    pub version: String,
}

impl Default for LedgerMetadata {
    fn default() -> Self {
        let now = Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string();
        Self {
            created: now.clone(),
            last_updated: now,
            version: "1.0.0".to_string(),
        }
    }
}

/// Block summary for index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSummary {
    pub index: i32,
    pub hash: String,
    pub tic_id: String,
    pub timestamp: String,
    pub file: String,
}

/// Ledger index structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerIndex {
    pub blocks: Vec<BlockSummary>,
    pub current_index: i32,
    pub metadata: LedgerMetadata,
}

impl Default for LedgerIndex {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            current_index: -1,
            metadata: LedgerMetadata::default(),
        }
    }
}

/// Compact TIC data for ledger storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactTic {
    pub tic_id: String,
    pub seed: String,
    /// Stored with deterministic precision to ensure hash consistency
    pub fixpoint_norm: f64,
    pub invariants: JsonValue,
    pub sigma_bar: JsonValue,
    pub window: Vec<String>,
}

/// MEF Block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MefBlock {
    pub index: i32,
    pub previous_hash: String,
    pub timestamp: String,
    pub tic_id: String,
    pub snapshot_hash: String,
    pub data: CompactTic,
    pub proof: JsonValue,
    pub hash: String,
}

/// Chain statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStatistics {
    pub total_blocks: u64,
    pub current_index: i32,
    pub total_size_bytes: u64,
    pub total_size_mb: f64,
    pub time_range: TimeRange,
    pub chain_valid: bool,
    pub metadata: LedgerMetadata,
}

/// Time range for chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub first: Option<String>,
    pub last: Option<String>,
}

/// MEF Ledger system implementing hash-chained blocks
/// B_i = H(tic_i, snapshot_i, B_{i-1})
pub struct MEFLedger {
    ledger_path: PathBuf,
    index: LedgerIndex,
    genesis_hash: String,
}

impl MEFLedger {
    /// Create a new MEF Ledger
    ///
    /// # Arguments
    /// * `ledger_path` - Directory for ledger storage
    pub fn new(ledger_path: impl AsRef<Path>) -> Result<Self> {
        let ledger_path = ledger_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&ledger_path).context("Failed to create ledger directory")?;

        let index_file = ledger_path.join("ledger_index.json");
        let index = Self::load_index(&index_file)?;
        let genesis_hash = "0".repeat(64);

        Ok(Self {
            ledger_path,
            index,
            genesis_hash,
        })
    }

    /// Load ledger index from disk
    fn load_index(index_file: &Path) -> Result<LedgerIndex> {
        if index_file.exists() {
            let contents =
                std::fs::read_to_string(index_file).context("Failed to read ledger index")?;
            let index: LedgerIndex =
                serde_json::from_str(&contents).context("Failed to parse ledger index")?;
            Ok(index)
        } else {
            Ok(LedgerIndex::default())
        }
    }

    /// Save ledger index to disk
    fn save_index(&mut self) -> Result<()> {
        self.index.metadata.last_updated = Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string();

        let index_file = self.ledger_path.join("ledger_index.json");
        let json = serde_json::to_string_pretty(&self.index)
            .context("Failed to serialize ledger index")?;
        std::fs::write(&index_file, json).context("Failed to write ledger index")?;

        Ok(())
    }

    /// Compute SHA256 hash of block data
    ///
    /// # Arguments
    /// * `block` - Block data (without hash field)
    ///
    /// Canonicalize JSON by recursively sorting all object keys
    fn canonicalize_json(value: &JsonValue) -> JsonValue {
        match value {
            JsonValue::Object(map) => {
                let mut sorted_map = serde_json::Map::new();
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                for key in keys {
                    sorted_map.insert(key.clone(), Self::canonicalize_json(&map[key]));
                }
                JsonValue::Object(sorted_map)
            }
            JsonValue::Array(arr) => {
                JsonValue::Array(arr.iter().map(Self::canonicalize_json).collect())
            }
            _ => value.clone(),
        }
    }

    /// Normalize floating point numbers in JSON for deterministic hashing
    fn normalize_floats_in_json(value: &mut JsonValue) {
        match value {
            JsonValue::Object(map) => {
                for (_key, val) in map.iter_mut() {
                    Self::normalize_floats_in_json(val);
                }
            }
            JsonValue::Array(arr) => {
                for val in arr.iter_mut() {
                    Self::normalize_floats_in_json(val);
                }
            }
            JsonValue::Number(n) => {
                // Only normalize if it's a float, not an integer
                if let Some(f) = n.as_f64() {
                    if !n.is_i64() && !n.is_u64() {
                        // It's a float, normalize it
                        let formatted = format!("{:.16e}", f);
                        *value = JsonValue::String(formatted);
                    }
                }
            }
            _ => {}
        }
    }

    pub fn compute_block_hash(block: &JsonValue) -> String {
        // Remove hash field if present
        let mut block_data = block.clone();
        if let Some(obj) = block_data.as_object_mut() {
            obj.remove("hash");
        }

        // Canonicalize JSON to ensure deterministic serialization
        let mut canonical_block = Self::canonicalize_json(&block_data);

        // Normalize all floating point numbers to strings for determinism
        Self::normalize_floats_in_json(&mut canonical_block);

        // Create deterministic string representation
        let block_str = serde_json::to_string(&canonical_block).unwrap();

        // Compute SHA256
        let mut hasher = Sha256::new();
        hasher.update(block_str.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get the most recent block in the ledger
    pub fn get_last_block(&self) -> Result<Option<MefBlock>> {
        if self.index.current_index < 0 {
            return Ok(None);
        }

        let block_file = self
            .ledger_path
            .join(format!("block_{:06}.mef", self.index.current_index));

        if !block_file.exists() {
            return Ok(None);
        }

        let contents = std::fs::read_to_string(&block_file).context("Failed to read block file")?;
        let block: MefBlock = serde_json::from_str(&contents).context("Failed to parse block")?;

        Ok(Some(block))
    }

    /// Get hash of the last block
    pub fn get_last_hash(&self) -> Result<String> {
        if let Some(block) = self.get_last_block()? {
            Ok(block.hash)
        } else {
            Ok(self.genesis_hash.clone())
        }
    }

    /// Create compact representation of TIC for ledger storage
    ///
    /// # Arguments
    /// * `tic` - Full TIC data as JSON
    pub fn compact_tic_data(tic: &JsonValue) -> Result<CompactTic> {
        let tic_id = tic["tic_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing tic_id"))?
            .to_string();

        let seed = tic["seed"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing seed"))?
            .to_string();

        // Compute fixpoint norm and round to fixed precision for determinism
        let fixpoint = tic["fixpoint"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing fixpoint"))?;
        let fixpoint_norm_raw: f64 = fixpoint
            .iter()
            .filter_map(|v| v.as_f64())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        // Round to 15 significant digits to ensure deterministic serialization
        let fixpoint_norm = format!("{:.15e}", fixpoint_norm_raw)
            .parse::<f64>()
            .unwrap();

        let window = tic["window"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing window"))?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();

        Ok(CompactTic {
            tic_id,
            seed,
            fixpoint_norm,
            invariants: Self::canonicalize_json(&tic["invariants"]),
            sigma_bar: Self::canonicalize_json(&tic["sigma_bar"]),
            window,
        })
    }

    /// Create a new MEF block
    ///
    /// # Arguments
    /// * `tic` - TIC data as JSON
    /// * `snapshot` - Snapshot data as JSON
    pub fn create_block(&self, tic: &JsonValue, snapshot: &JsonValue) -> Result<MefBlock> {
        // Get next index
        let next_index = self.index.current_index + 1;

        // Get previous hash
        let previous_hash = self.get_last_hash()?;

        // Compute snapshot hash with canonical JSON
        let canonical_snapshot = Self::canonicalize_json(snapshot);
        let snapshot_str =
            serde_json::to_string(&canonical_snapshot).context("Failed to serialize snapshot")?;
        let mut hasher = Sha256::new();
        hasher.update(snapshot_str.as_bytes());
        let snapshot_hash = format!("{:x}", hasher.finalize());

        let tic_id = tic["tic_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing tic_id"))?
            .to_string();

        // Create block structure with canonicalized proof
        let mut block_json = serde_json::json!({
            "index": next_index,
            "previous_hash": previous_hash,
            "timestamp": Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string(),
            "tic_id": tic_id,
            "snapshot_hash": snapshot_hash,
            "data": Self::compact_tic_data(tic)?,
            "proof": Self::canonicalize_json(&tic["proof"]),
        });

        // Compute block hash
        let hash = Self::compute_block_hash(&block_json);
        block_json["hash"] = serde_json::json!(hash);

        // Deserialize to MefBlock
        let block: MefBlock =
            serde_json::from_value(block_json).context("Failed to create block")?;

        Ok(block)
    }

    /// Append a new block to the ledger
    ///
    /// # Arguments
    /// * `tic` - TIC data as JSON
    /// * `snapshot` - Snapshot data as JSON
    pub fn append_block(&mut self, tic: &JsonValue, snapshot: &JsonValue) -> Result<MefBlock> {
        // Create new block
        let block = self.create_block(tic, snapshot)?;

        // Verify chain integrity before appending
        if !self.verify_chain_integrity(0)? {
            anyhow::bail!("Chain integrity check failed");
        }

        // Save block to disk in standard format (not normalized)
        let block_file = self
            .ledger_path
            .join(format!("block_{:06}.mef", block.index));
        let json = serde_json::to_string_pretty(&block).context("Failed to serialize block")?;
        std::fs::write(&block_file, json).context("Failed to write block file")?;

        // Update index
        self.index.blocks.push(BlockSummary {
            index: block.index,
            hash: block.hash.clone(),
            tic_id: block.tic_id.clone(),
            timestamp: block.timestamp.clone(),
            file: block_file
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
        });
        self.index.current_index = block.index;
        self.save_index()?;

        Ok(block)
    }

    /// Retrieve a block by index
    ///
    /// # Arguments
    /// * `index` - Block index
    pub fn get_block(&self, index: i32) -> Result<Option<MefBlock>> {
        let block_file = self.ledger_path.join(format!("block_{:06}.mef", index));

        if !block_file.exists() {
            return Ok(None);
        }

        let contents = std::fs::read_to_string(&block_file).context("Failed to read block file")?;
        let block: MefBlock = serde_json::from_str(&contents).context("Failed to parse block")?;

        Ok(Some(block))
    }

    /// Verify that a block's hash is correct
    ///
    /// # Arguments
    /// * `block` - Block data
    pub fn verify_block_hash(&self, block: &MefBlock) -> bool {
        let block_json = serde_json::to_value(block).unwrap();
        let computed_hash = Self::compute_block_hash(&block_json);
        block.hash == computed_hash
    }

    /// Verify integrity of the entire chain or from a specific index
    ///
    /// # Arguments
    /// * `start_index` - Starting block index for verification
    pub fn verify_chain_integrity(&self, start_index: i32) -> Result<bool> {
        if self.index.current_index < 0 {
            // Empty ledger is valid
            return Ok(true);
        }

        let mut prev_hash = if start_index == 0 {
            Some(self.genesis_hash.clone())
        } else {
            None
        };

        for i in start_index..=self.index.current_index {
            let block = self.get_block(i)?;

            if block.is_none() {
                eprintln!("Missing block at index {}", i);
                return Ok(false);
            }

            let block = block.unwrap();

            // Verify block hash
            if !self.verify_block_hash(&block) {
                eprintln!("Invalid hash for block {}", i);
                return Ok(false);
            }

            // Verify chain linkage
            if let Some(ref expected_prev) = prev_hash {
                if block.previous_hash != *expected_prev {
                    eprintln!("Chain break at block {}", i);
                    return Ok(false);
                }
            }

            prev_hash = Some(block.hash.clone());
        }

        Ok(true)
    }

    /// Get ledger chain statistics
    pub fn get_chain_statistics(&self) -> Result<ChainStatistics> {
        let total_blocks = if self.index.current_index >= 0 {
            (self.index.current_index + 1) as u64
        } else {
            0
        };

        // Calculate chain file size
        let mut total_size = 0u64;
        for block_info in &self.index.blocks {
            let block_file = self.ledger_path.join(&block_info.file);
            if let Ok(metadata) = block_file.metadata() {
                total_size += metadata.len();
            }
        }

        // Get time range
        let time_range = if total_blocks > 0 {
            let first_block = self.get_block(0)?;
            let last_block = self.get_block(self.index.current_index)?;
            TimeRange {
                first: first_block.map(|b| b.timestamp),
                last: last_block.map(|b| b.timestamp),
            }
        } else {
            TimeRange {
                first: None,
                last: None,
            }
        };

        Ok(ChainStatistics {
            total_blocks,
            current_index: self.index.current_index,
            total_size_bytes: total_size,
            total_size_mb: total_size as f64 / (1024.0 * 1024.0),
            time_range,
            chain_valid: self.verify_chain_integrity(0)?,
            metadata: self.index.metadata.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_create_ledger() {
        let temp_dir = std::env::temp_dir().join("test_ledger");
        let ledger = MEFLedger::new(&temp_dir).unwrap();

        assert_eq!(ledger.index.current_index, -1);
        assert_eq!(ledger.index.blocks.len(), 0);
    }

    #[test]
    fn test_compute_block_hash() {
        let block = json!({
            "index": 0,
            "previous_hash": "0".repeat(64),
            "data": {"test": "data"}
        });

        let hash1 = MEFLedger::compute_block_hash(&block);
        let hash2 = MEFLedger::compute_block_hash(&block);

        // Determinism test
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn test_append_block() {
        let temp_dir = std::env::temp_dir().join("test_ledger_append");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let mut ledger = MEFLedger::new(&temp_dir).unwrap();

        let tic = json!({
            "tic_id": "tic-001",
            "seed": "MEF_SEED_42",
            "fixpoint": [0.1, 0.2, 0.3],
            "invariants": {"variance": 0.1},
            "sigma_bar": {"psi": 0.5},
            "window": ["2025-01-01T00:00:00", "2025-01-01T01:00:00"],
            "proof": {"merkle_root": "abc123"}
        });

        let snapshot = json!({
            "id": "snap-001",
            "coordinates": [0.1, 0.2, 0.3, 0.4, 0.5]
        });

        let block = ledger.append_block(&tic, &snapshot).unwrap();

        assert_eq!(block.index, 0);
        assert_eq!(block.tic_id, "tic-001");
        assert_eq!(ledger.index.current_index, 0);
    }

    #[test]
    fn test_chain_integrity() {
        let temp_dir = std::env::temp_dir().join("test_ledger_integrity");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let mut ledger = MEFLedger::new(&temp_dir).unwrap();

        // Add multiple blocks (testing with 10 like the failing integration test)
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

            ledger.append_block(&tic, &snapshot).unwrap();
        }

        // Verify chain
        assert!(ledger.verify_chain_integrity(0).unwrap());
        assert_eq!(ledger.index.current_index, 9);
    }

    #[test]
    fn test_deterministic_hash_golden() {
        // Golden test: Verify that hash computation is deterministic
        // Given the same block JSON, we should always get the same hash

        // Test 1: Verify hash function is deterministic
        let block_json = json!({
            "index": 0,
            "previous_hash": "0".repeat(64),
            "timestamp": "2025-10-16T00:00:00.000000Z",
            "tic_id": "golden-tic-001",
            "snapshot_hash": "abc123",
            "data": {
                "tic_id": "golden-tic-001",
                "seed": "GOLDEN_SEED",
                "fixpoint_norm": "4.1231056256176610",
                "invariants": {"variance": 0.1, "alpha": 0.05},
                "sigma_bar": {"psi": 0.5, "theta": 0.3},
                "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"]
            },
            "proof": {"merkle_root": "golden_root", "depth": 5}
        });

        // Compute hash multiple times - should be identical
        let hash1 = MEFLedger::compute_block_hash(&block_json);
        let hash2 = MEFLedger::compute_block_hash(&block_json);
        let hash3 = MEFLedger::compute_block_hash(&block_json);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);

        // Test 2: Verify JSON canonicalization handles different key orders
        let block_json_reordered = json!({
            "proof": {"depth": 5, "merkle_root": "golden_root"},
            "data": {
                "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
                "sigma_bar": {"theta": 0.3, "psi": 0.5},
                "invariants": {"alpha": 0.05, "variance": 0.1},
                "fixpoint_norm": "4.1231056256176610",
                "seed": "GOLDEN_SEED",
                "tic_id": "golden-tic-001"
            },
            "snapshot_hash": "abc123",
            "tic_id": "golden-tic-001",
            "timestamp": "2025-10-16T00:00:00.000000Z",
            "previous_hash": "0".repeat(64),
            "index": 0
        });

        let hash_reordered = MEFLedger::compute_block_hash(&block_json_reordered);
        assert_eq!(
            hash1, hash_reordered,
            "Hash should be same regardless of JSON key order"
        );

        // Test 3: Verify changing data changes hash
        let mut block_json_modified = block_json.clone();
        block_json_modified["data"]["seed"] = json!("DIFFERENT_SEED");
        let hash_modified = MEFLedger::compute_block_hash(&block_json_modified);
        assert_ne!(hash1, hash_modified, "Hash should change when data changes");

        // Test 4: Verify loaded blocks maintain their hash
        let temp_dir = std::env::temp_dir().join("test_golden_hash");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let mut ledger = MEFLedger::new(&temp_dir).unwrap();

        let tic = json!({
            "tic_id": "golden-tic-001",
            "seed": "GOLDEN_SEED",
            "fixpoint": [1.234567890123456, 2.345678901234567, 3.456789012345678],
            "invariants": {"variance": 0.1, "alpha": 0.05},
            "sigma_bar": {"psi": 0.5, "theta": 0.3},
            "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
            "proof": {"merkle_root": "golden_root", "depth": 5}
        });

        let snapshot = json!({
            "id": "golden-snap-001",
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metadata": {"version": "1.0"}
        });

        let block = ledger.append_block(&tic, &snapshot).unwrap();
        let original_hash = block.hash.clone();

        // Load the block back and verify hash is unchanged
        let loaded_block = ledger.get_block(0).unwrap().unwrap();
        assert_eq!(original_hash, loaded_block.hash);

        // Verify the block hash is correct after round-trip
        assert!(ledger.verify_block_hash(&loaded_block));
    }
}
