//! # Deterministic Primitives
//!
//! Core deterministic functions for content addressing and serialization.
//!
//! ## SPEC-006 Reference
//!
//! From Part 4, Section 1:
//! - Canonical JSON with stable key order and fixed float precision
//! - Content hash using BLAKE3 (or SHA256 fallback)
//! - HD seed derivation using HMAC-SHA256

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Canonical JSON serialization with stable key order and fixed float precision
///
/// ## Properties
///
/// - UTF-8 encoding, no BOM
/// - Keys sorted in ASCII order
/// - Floats with 6 fixed decimals
/// - No trailing spaces
/// - Deterministic across platforms
///
/// ## Example
///
/// ```rust,ignore
/// use mef_knowledge::canonical_json;
/// use serde_json::json;
///
/// let obj = json!({"b": 1.23456789, "a": 42});
/// let canonical = canonical_json(&obj).unwrap();
/// // Result: {"a":42,"b":1.234568}
/// ```
pub fn canonical_json<T: Serialize>(obj: &T) -> anyhow::Result<String> {
    // Serialize to serde_json::Value first to enable transformation
    let value = serde_json::to_value(obj)?;
    let rounded = round_floats(value);

    // Serialize with no whitespace, sorted keys
    let json = serde_json::to_string(&rounded)?;

    Ok(json)
}

/// Round all floats in a JSON value to 6 decimal places
fn round_floats(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                // Format with 6 decimals and parse back
                let rounded = format!("{:.6}", f);
                let parsed: f64 = rounded.parse().unwrap_or(f);
                serde_json::json!(parsed)
            } else {
                serde_json::Value::Number(n)
            }
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(round_floats).collect())
        }
        serde_json::Value::Object(obj) => {
            // Use BTreeMap to ensure key ordering
            let mut sorted: BTreeMap<String, serde_json::Value> = BTreeMap::new();
            for (k, v) in obj {
                sorted.insert(k, round_floats(v));
            }
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        other => other,
    }
}

/// Compute content hash using SHA256 (BLAKE3 would be preferred but keeping dependencies minimal)
///
/// ## SPEC-006 Reference
///
/// From Part 4, Section 1.2:
/// - Use BLAKE3 if available, otherwise SHA256
/// - Return hexadecimal string
pub fn compute_content_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Compute MEF ID from TIC, route, and seed path
///
/// ## Formula
///
/// ```text
/// mef_id = HASH(canonical(TIC) || "|" || route_id || "|" || seed_path)[:32]
/// ```
///
/// ## SPEC-006 Reference
///
/// From Part 4, Section 1.2
pub fn compute_mef_id<T: Serialize>(
    tic: &T,
    route_id: &str,
    seed_path: &str,
) -> anyhow::Result<String> {
    let tic_json = canonical_json(tic)?;
    let blob = format!("{}|{}|{}", tic_json, route_id, seed_path);
    let hash = compute_content_hash(blob.as_bytes());

    // Take first 32 characters as specified
    Ok(hash.chars().take(32).collect())
}

/// Derive sub-seed from root seed using HD-style derivation
///
/// ## Formula
///
/// ```text
/// ssub = HMAC_SHA256(sroot, "MEF/<domain>/<stage>/<index>")
/// ```
///
/// ## SPEC-006 Reference
///
/// From Part 2, Section 2.1 and Part 4, Section 1.3
///
/// ## Security Note
///
/// The root seed (BIP-39 mnemonic) MUST NEVER be logged or persisted.
/// Only derived seeds and path IDs should be stored.
pub fn derive_seed(root_seed: &[u8], path: &str) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;

    let mut mac = HmacSha256::new_from_slice(root_seed).expect("HMAC can take key of any size");
    mac.update(path.as_bytes());

    mac.finalize().into_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_canonical_json_ordering() {
        let obj = json!({"z": 1, "a": 2, "m": 3});
        let canonical = canonical_json(&obj).unwrap();
        // Note: integers may be preserved as integers (not floats)
        assert!(canonical.starts_with(r#"{"a":"#));
        assert!(canonical.contains(r#""m":"#));
        assert!(canonical.ends_with(r#""z":1}"#) || canonical.ends_with(r#""z":1.0}"#));
    }

    #[test]
    fn test_canonical_json_float_precision() {
        let obj = json!({"value": 1.23456789});
        let canonical = canonical_json(&obj).unwrap();
        assert!(canonical.contains("1.234568"));
    }

    #[test]
    fn test_canonical_json_nested() {
        let obj = json!({
            "outer": {
                "z": 9.87654321,
                "a": 1.11111111
            }
        });
        let canonical = canonical_json(&obj).unwrap();
        assert!(canonical.contains(r#"{"a":1.111111,"z":9.876543}"#));
    }

    #[test]
    fn test_content_hash_deterministic() {
        let data = b"hello world";
        let hash1 = compute_content_hash(data);
        let hash2 = compute_content_hash(data);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 produces 64 hex chars
    }

    #[test]
    fn test_mef_id_deterministic() {
        let tic = json!({"tic_id": "TIC-123", "timestamp": "2025-10-16T22:00:00Z"});
        let route_id = "route-abc";
        let seed_path = "MEF/test/spiral/0001";

        let id1 = compute_mef_id(&tic, route_id, seed_path).unwrap();
        let id2 = compute_mef_id(&tic, route_id, seed_path).unwrap();

        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 32);
    }

    #[test]
    fn test_derive_seed_deterministic() {
        let root = b"test_root_seed_32_bytes_long____";
        let path = "MEF/text/spiral/0001";

        let seed1 = derive_seed(root, path);
        let seed2 = derive_seed(root, path);

        assert_eq!(seed1, seed2);
        assert_eq!(seed1.len(), 32); // SHA256 output
    }

    #[test]
    fn test_derive_seed_different_paths() {
        let root = b"test_root_seed_32_bytes_long____";

        let seed1 = derive_seed(root, "MEF/text/spiral/0001");
        let seed2 = derive_seed(root, "MEF/text/spiral/0002");

        assert_ne!(seed1, seed2);
    }
}
