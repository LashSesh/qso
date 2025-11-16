//! HD-style seed derivation using HMAC-SHA256

use sha2::{Digest, Sha256};

/// Derive a child seed from a parent seed using HD-style derivation
/// Following BIP-39 principles: derived_seed = HMAC-SHA256(parent_seed, path)
///
/// Important: The root seed should NEVER be logged or persisted.
/// Only derived seeds should be stored.
pub fn derive_seed(parent_seed: &[u8], path: &str) -> crate::Result<Vec<u8>> {
    if parent_seed.is_empty() {
        return Err(crate::KnowledgeError::SeedDerivation(
            "Parent seed cannot be empty".to_string(),
        ));
    }

    if path.is_empty() {
        return Err(crate::KnowledgeError::SeedDerivation(
            "Derivation path cannot be empty".to_string(),
        ));
    }

    // Simple HMAC-SHA256 implementation
    let derived = hmac_sha256(parent_seed, path.as_bytes());
    Ok(derived)
}

/// Simple HMAC-SHA256 implementation
fn hmac_sha256(key: &[u8], message: &[u8]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 64;

    // Prepare key
    let mut key_block = vec![0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hash = Sha256::digest(key);
        key_block[..32].copy_from_slice(&hash);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    // Create inner and outer keys
    let mut inner_key = key_block.clone();
    let mut outer_key = key_block;

    for i in 0..BLOCK_SIZE {
        inner_key[i] ^= 0x36;
        outer_key[i] ^= 0x5c;
    }

    // Compute inner hash
    let mut inner_hasher = Sha256::new();
    inner_hasher.update(&inner_key);
    inner_hasher.update(message);
    let inner_hash = inner_hasher.finalize();

    // Compute outer hash
    let mut outer_hasher = Sha256::new();
    outer_hasher.update(&outer_key);
    outer_hasher.update(&inner_hash);
    let outer_hash = outer_hasher.finalize();

    outer_hash.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed() {
        let parent = b"test_parent_seed";
        let path = "MEF/domain/stage/0001";

        let derived = derive_seed(parent, path);
        assert!(derived.is_ok());
        assert_eq!(derived.unwrap().len(), 32); // SHA256 output
    }

    #[test]
    fn test_determinism() {
        let parent = b"test_parent_seed";
        let path = "MEF/domain/stage/0001";

        let derived1 = derive_seed(parent, path).unwrap();
        let derived2 = derive_seed(parent, path).unwrap();
        assert_eq!(derived1, derived2);
    }

    #[test]
    fn test_different_paths_different_seeds() {
        let parent = b"test_parent_seed";

        let derived1 = derive_seed(parent, "MEF/domain/stage/0001").unwrap();
        let derived2 = derive_seed(parent, "MEF/domain/stage/0002").unwrap();
        assert_ne!(derived1, derived2);
    }

    #[test]
    fn test_empty_parent_error() {
        let result = derive_seed(&[], "MEF/domain/stage/0001");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_path_error() {
        let result = derive_seed(b"test_seed", "");
        assert!(result.is_err());
    }
}
