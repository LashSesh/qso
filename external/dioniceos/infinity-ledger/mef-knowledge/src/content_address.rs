//! Content-addressed knowledge IDs via SHA256 hashing

use crate::canonical::canonical_json;
use sha2::{Digest, Sha256};

/// Compute a content-addressed MEF ID from TIC, route, and seed path
/// Uses SHA256 hash of canonical JSON representation
pub fn compute_mef_id(tic_id: &str, route_id: &str, seed_path: &str) -> crate::Result<String> {
    let data = serde_json::json!({
        "tic_id": tic_id,
        "route_id": route_id,
        "seed_path": seed_path
    });

    let canonical = canonical_json(&data)?;
    let hash = Sha256::digest(canonical.as_bytes());
    Ok(format!("mef_{}", hex::encode(hash[..16].to_vec()))) // Use first 16 bytes (32 hex chars)
}

/// Compute SHA256 hash of arbitrary data
pub fn compute_hash(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    hex::encode(hash.to_vec())
}

// Simple hex encoding helper
mod hex {
    pub fn encode(bytes: Vec<u8>) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mef_id() {
        let mef_id = compute_mef_id("tic_001", "route_001", "MEF/domain/stage/0001");
        assert!(mef_id.is_ok());

        let id = mef_id.unwrap();
        assert!(id.starts_with("mef_"));
        assert_eq!(id.len(), 36); // "mef_" + 32 hex chars
    }

    #[test]
    fn test_determinism() {
        let id1 = compute_mef_id("tic_001", "route_001", "MEF/domain/stage/0001").unwrap();
        let id2 = compute_mef_id("tic_001", "route_001", "MEF/domain/stage/0001").unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_different_inputs_different_ids() {
        let id1 = compute_mef_id("tic_001", "route_001", "MEF/domain/stage/0001").unwrap();
        let id2 = compute_mef_id("tic_002", "route_001", "MEF/domain/stage/0001").unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_compute_hash() {
        let hash = compute_hash(b"test data");
        assert_eq!(hash.len(), 64); // SHA256 = 32 bytes = 64 hex chars
    }
}
