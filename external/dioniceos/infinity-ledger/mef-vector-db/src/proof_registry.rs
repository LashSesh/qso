/*!
 * Deterministic membership proof construction for vector collections
 *
 * The proof registry provides a Sparse-Merkle-like commitment where each leaf
 * corresponds to a single vector entry. Proofs are generated lazily and cached
 * together with a global commit root so callers can validate lookups against a
 * signed digest that is stable for a given collection state.
 *
 * The implementation uses standard SHA-256 hashing and stable JSON serialization
 * to guarantee deterministic outputs across platforms.
 */

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// Volatile metadata keys that should be excluded from proofs
const VOLATILE_KEYS: &[&str] = &[
    "timestamp",
    "created",
    "created_at",
    "updated",
    "updated_at",
    "ingested",
    "ingested_at",
    "acquired",
    "activated",
    "generated",
    "refreshed",
];

const VOLATILE_SUFFIXES: &[&str] = &["_at", "_time", "_timestamp", "_ts"];

/// SHA256 hash helper
fn sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Stable JSON serialization
fn stable_json(data: &Value) -> Vec<u8> {
    serde_json::to_string(data).unwrap().into_bytes()
}

/// Check if a key is volatile (should be excluded from proofs)
fn is_volatile_key(key: &str) -> bool {
    let last_fragment = key.rsplit('.').next().unwrap_or(key);
    let normalized = normalize_key_fragment(last_fragment);

    if VOLATILE_KEYS.contains(&normalized.as_str()) {
        return true;
    }

    for suffix in VOLATILE_SUFFIXES {
        if normalized.ends_with(suffix) {
            return true;
        }
    }

    false
}

/// Normalize a key fragment for checking
fn normalize_key_fragment(fragment: &str) -> String {
    let cleaned = fragment.replace('-', "_");
    if cleaned.is_empty() {
        return String::new();
    }

    if cleaned.chars().all(|c| c.is_uppercase() || c == '_') {
        return cleaned.to_lowercase();
    }

    // Convert camelCase to snake_case manually
    let mut result = String::new();
    let mut prev_lower = false;

    for ch in cleaned.chars() {
        if ch.is_uppercase() {
            if prev_lower {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
            prev_lower = false;
        } else {
            result.push(ch);
            prev_lower = ch.is_lowercase() || ch.is_numeric();
        }
    }

    result
}

/// Canonicalize a vector (convert to Vec<f64>)
fn canonicalize_vector(vector: Option<&Value>) -> Vec<f64> {
    match vector {
        Some(Value::Array(arr)) => arr.iter().filter_map(|v| v.as_f64()).collect(),
        _ => Vec::new(),
    }
}

/// Canonicalize metadata by removing volatile keys and sorting
fn canonicalize_metadata(metadata: Option<&Value>) -> Value {
    fn strip(value: &Value, prefix: &str) -> Option<Value> {
        match value {
            Value::Object(map) => {
                let mut canonical = serde_json::Map::new();
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();

                for key in keys {
                    let full_key = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", prefix, key)
                    };

                    if is_volatile_key(&full_key) {
                        continue;
                    }

                    if let Some(stripped) = strip(&map[key], &full_key) {
                        canonical.insert(key.clone(), stripped);
                    }
                }

                Some(Value::Object(canonical))
            }
            Value::Array(arr) => {
                let mut canonical: Vec<Value> =
                    arr.iter().filter_map(|item| strip(item, prefix)).collect();

                // Sort objects in array by their JSON representation
                if canonical.iter().all(|v| v.is_object()) {
                    canonical.sort_by(|a, b| {
                        let a_str = serde_json::to_string(a).unwrap();
                        let b_str = serde_json::to_string(b).unwrap();
                        a_str.cmp(&b_str)
                    });
                }

                Some(Value::Array(canonical))
            }
            _ => Some(value.clone()),
        }
    }

    strip(metadata.unwrap_or(&Value::Null), "").unwrap_or(Value::Object(serde_json::Map::new()))
}

/// Compute leaf hash for a vector entry
fn leaf_hash(vector_id: &str, payload: &HashMap<String, Value>) -> String {
    let vector = payload.get("vector");
    let metadata = payload.get("metadata");
    let epoch = payload.get("epoch");

    let material = serde_json::json!({
        "id": vector_id,
        "epoch": epoch,
        "vector": canonicalize_vector(vector),
        "metadata": canonicalize_metadata(metadata),
    });

    sha256(&stable_json(&material))
}

/// Combine two hashes
fn combine_hash(left: &str, right: &str) -> String {
    sha256(format!("{}|{}", left, right).as_bytes())
}

/// Membership proof error
#[derive(Debug, thiserror::Error)]
pub enum ProofError {
    #[error("Proof verification failed")]
    VerificationFailed,
    #[error("Invalid proof data: {0}")]
    InvalidProof(String),
}

/// Sparse Merkle style proof for a single vector entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipProof {
    /// Collection name
    pub collection: String,
    /// Vector ID
    pub vector_id: String,
    /// Leaf hash
    pub leaf: String,
    /// Sibling hashes with positions (left or right)
    pub siblings: Vec<(String, String)>,
    /// Collection root hash
    pub collection_root: String,
    /// Commit root hash
    pub commit_root: String,
}

impl MembershipProof {
    /// Verify the proof against an optional externally supplied leaf/root
    pub fn verify(&self, leaf_hash: Option<&str>, commit_root: Option<&str>) -> bool {
        let mut running = leaf_hash.unwrap_or(&self.leaf).to_string();

        for (position, sibling_hash) in &self.siblings {
            running = if position == "left" {
                combine_hash(sibling_hash, &running)
            } else if position == "right" {
                combine_hash(&running, sibling_hash)
            } else {
                return false; // Invalid position
            };
        }

        if running != self.collection_root {
            return false;
        }

        let expected_root = commit_root.unwrap_or(&self.commit_root);
        expected_root == self.commit_root
    }
}

/// Simplified collection state for proof registry
#[derive(Debug, Clone)]
pub struct CollectionState {
    /// Vectors indexed by ID
    pub vectors: HashMap<String, HashMap<String, Value>>,
    /// Index metadata
    pub indexes: HashMap<String, Value>,
}

/// Construct and cache membership proofs for vector collections
pub struct ProofRegistry {
    /// Key ID for signing
    kid: String,
    /// Secret for HMAC signing
    secret: Vec<u8>,
    /// Thread-safe lock
    lock: Mutex<ProofRegistryState>,
}

struct ProofRegistryState {
    /// Collection roots indexed by name
    collection_roots: HashMap<String, String>,
    /// Collection versions for cache invalidation
    collection_versions: HashMap<String, i64>,
    /// Cached proofs indexed by (collection, vector_id)
    proofs: HashMap<(String, String), MembershipProof>,
    /// Global commit root
    commit_root: String,
    /// HMAC signature of commit root
    signature: String,
}

impl ProofRegistry {
    /// Create a new proof registry
    ///
    /// # Arguments
    ///
    /// * `kid` - Key ID for signing (defaults to "ledger-root")
    /// * `secret` - Secret for HMAC signing (defaults to "MEF-SEED-COMMIT")
    pub fn new(kid: Option<String>, secret: Option<String>) -> Self {
        let kid = kid.unwrap_or_else(|| {
            std::env::var("MEF_COMMIT_KID").unwrap_or_else(|_| "ledger-root".to_string())
        });
        let secret_str = secret.unwrap_or_else(|| {
            std::env::var("MEF_COMMIT_SECRET").unwrap_or_else(|_| "MEF-SEED-COMMIT".to_string())
        });
        let secret = secret_str.into_bytes();

        let commit_root = "0".repeat(64);
        let signature = Self::sign_commit(&secret, &commit_root);

        Self {
            kid,
            secret,
            lock: Mutex::new(ProofRegistryState {
                collection_roots: HashMap::new(),
                collection_versions: HashMap::new(),
                proofs: HashMap::new(),
                commit_root,
                signature,
            }),
        }
    }

    /// Get the key ID
    pub fn kid(&self) -> String {
        self.kid.clone()
    }

    /// Get the current commit root
    pub fn commit_root(&self) -> String {
        let state = self.lock.lock().unwrap();
        state.commit_root.clone()
    }

    /// Get the current signature
    pub fn signature(&self) -> String {
        let state = self.lock.lock().unwrap();
        state.signature.clone()
    }

    /// Refresh proofs from collection states
    pub fn refresh_from_collections(&self, collections: &HashMap<String, CollectionState>) {
        let mut state = self.lock.lock().unwrap();
        let _previous_root = state.commit_root.clone();
        let mut changed = false;

        let active_collections: HashSet<_> = collections.keys().cloned().collect();

        // Remove inactive collections
        state
            .collection_roots
            .retain(|k, _| active_collections.contains(k));
        state
            .collection_versions
            .retain(|k, _| active_collections.contains(k));
        state
            .proofs
            .retain(|(coll, _), _| active_collections.contains(coll));

        // Update or add collections
        for (collection, coll_state) in collections {
            let version = coll_state
                .indexes
                .get("proof_version")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let previous_version = state.collection_versions.get(collection);

            if previous_version.is_none()
                || previous_version != Some(&version)
                || !state.collection_roots.contains_key(collection)
            {
                let (root, proofs) = Self::build_collection_root(collection, &coll_state.vectors);
                state.collection_roots.insert(collection.clone(), root);

                // Remove old proofs for this collection
                state.proofs.retain(|(coll, _), _| coll != collection);

                // Add new proofs
                state.proofs.extend(proofs);
                state
                    .collection_versions
                    .insert(collection.clone(), version);
                changed = true;
            }
        }

        // Recompute commit root if changed
        if changed || state.collection_roots.is_empty() {
            // Clone the roots to avoid borrow issues
            let roots_clone = state.collection_roots.clone();
            let combined_root = Self::combine_collection_roots(&roots_clone, &mut state.proofs);
            state.commit_root = combined_root.clone();
            state.signature = Self::sign_commit(&self.secret, &combined_root);
        }
    }

    /// Get membership proof for a vector
    pub fn get_membership_proof(
        &self,
        collection: &str,
        vector_id: &str,
    ) -> Option<MembershipProof> {
        let state = self.lock.lock().unwrap();
        state
            .proofs
            .get(&(collection.to_string(), vector_id.to_string()))
            .cloned()
    }

    /// Get collection root hash
    pub fn get_collection_root(&self, collection: &str) -> Option<String> {
        let state = self.lock.lock().unwrap();
        state.collection_roots.get(collection).cloned()
    }

    /// Get commit snapshot
    pub fn get_commit_snapshot(&self) -> HashMap<String, String> {
        let state = self.lock.lock().unwrap();
        let mut snapshot = HashMap::new();
        snapshot.insert("commit_root".to_string(), state.commit_root.clone());
        snapshot.insert("kid".to_string(), self.kid.clone());
        snapshot.insert("signature".to_string(), state.signature.clone());
        snapshot
    }

    /// Rotate secret and regenerate signature
    pub fn rotate_secret(
        &mut self,
        kid: Option<String>,
        secret: Option<String>,
    ) -> HashMap<String, String> {
        if let Some(new_kid) = kid {
            self.kid = new_kid;
        }
        if let Some(new_secret) = secret {
            self.secret = new_secret.into_bytes();
        }

        let mut state = self.lock.lock().unwrap();
        state.signature = Self::sign_commit(&self.secret, &state.commit_root);

        let mut result = HashMap::new();
        result.insert("commit_root".to_string(), state.commit_root.clone());
        result.insert("kid".to_string(), self.kid.clone());
        result.insert("signature".to_string(), state.signature.clone());
        result
    }

    // ------------------------------------------------------------------
    // Private methods
    // ------------------------------------------------------------------

    /// Build Merkle tree for a collection
    fn build_collection_root(
        collection: &str,
        vectors: &HashMap<String, HashMap<String, Value>>,
    ) -> (String, HashMap<(String, String), MembershipProof>) {
        let mut vector_items: Vec<_> = vectors.iter().collect();
        vector_items.sort_by_key(|(k, _)| k.as_str());

        if vector_items.is_empty() {
            let empty_root = sha256(format!("{}|empty", collection).as_bytes());
            return (empty_root, HashMap::new());
        }

        // Build leaf hashes
        let leaves: Vec<(String, String)> = vector_items
            .iter()
            .map(|(id, payload)| (id.to_string(), leaf_hash(id, payload)))
            .collect();

        // Build Merkle tree levels
        let mut level: Vec<String> = leaves.iter().map(|(_, hash)| hash.clone()).collect();
        let mut tree_levels = vec![level.clone()];

        while level.len() > 1 {
            let mut next_level = Vec::new();
            for i in (0..level.len()).step_by(2) {
                let left = &level[i];
                let right = if i + 1 < level.len() {
                    &level[i + 1]
                } else {
                    left
                };
                next_level.push(combine_hash(left, right));
            }
            level = next_level;
            tree_levels.push(level.clone());
        }

        let root_hash = level[0].clone();

        // Build proofs for each leaf
        let mut proofs = HashMap::new();
        for (position, (vector_id, leaf_hash)) in leaves.iter().enumerate() {
            let mut siblings = Vec::new();
            let mut index = position;

            for level_values in &tree_levels[..tree_levels.len() - 1] {
                if index % 2 == 0 {
                    let sibling_index = if index + 1 < level_values.len() {
                        index + 1
                    } else {
                        index
                    };
                    siblings.push(("right".to_string(), level_values[sibling_index].clone()));
                } else {
                    siblings.push(("left".to_string(), level_values[index - 1].clone()));
                }
                index /= 2;
            }

            let proof = MembershipProof {
                collection: collection.to_string(),
                vector_id: vector_id.clone(),
                leaf: leaf_hash.clone(),
                siblings,
                collection_root: root_hash.clone(),
                commit_root: String::new(), // Filled later
            };

            proofs.insert((collection.to_string(), vector_id.clone()), proof);
        }

        (root_hash, proofs)
    }

    /// Combine collection roots into a single commit root
    fn combine_collection_roots(
        roots: &HashMap<String, String>,
        proofs: &mut HashMap<(String, String), MembershipProof>,
    ) -> String {
        if roots.is_empty() {
            return sha256(b"empty-commit");
        }

        let mut digest = String::new();
        let mut sorted_names: Vec<_> = roots.keys().cloned().collect();
        sorted_names.sort();

        for name in &sorted_names {
            digest = sha256(format!("{}|{}|{}", digest, name, roots[name]).as_bytes());
        }

        // Update all proofs with the commit root
        for proof in proofs.values_mut() {
            proof.commit_root = digest.clone();
        }

        digest
    }

    /// Sign commit root with HMAC-SHA256
    fn sign_commit(secret: &[u8], commit_root: &str) -> String {
        use sha2::digest::Mac;
        type HmacSha256 = hmac::Hmac<Sha256>;

        let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC can take key of any size");
        mac.update(commit_root.as_bytes());
        format!("{:x}", mac.finalize().into_bytes())
    }
}

impl Default for ProofRegistry {
    fn default() -> Self {
        Self::new(None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        let hash = sha256(b"test");
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_is_volatile_key() {
        assert!(is_volatile_key("timestamp"));
        assert!(is_volatile_key("created_at"));
        assert!(is_volatile_key("user.updated_at"));
        assert!(is_volatile_key("some_ts"));
        assert!(!is_volatile_key("username"));
        assert!(!is_volatile_key("data"));
    }

    #[test]
    fn test_canonicalize_vector() {
        let vec = serde_json::json!([1.0, 2.0, 3.0]);
        let result = canonicalize_vector(Some(&vec));
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_canonicalize_metadata() {
        let metadata = serde_json::json!({
            "name": "test",
            "timestamp": "2025-01-01",
            "count": 42
        });
        let result = canonicalize_metadata(Some(&metadata));
        assert!(result.get("name").is_some());
        assert!(result.get("count").is_some());
        assert!(result.get("timestamp").is_none()); // Volatile key removed
    }

    #[test]
    fn test_proof_registry_creation() {
        let registry = ProofRegistry::new(None, None);
        assert_eq!(registry.kid(), "ledger-root");
        assert_eq!(registry.commit_root().len(), 64);
    }

    #[test]
    fn test_membership_proof_verify() {
        let proof = MembershipProof {
            collection: "test".to_string(),
            vector_id: "vec1".to_string(),
            leaf: "abc".to_string(),
            siblings: vec![],
            collection_root: "abc".to_string(),
            commit_root: "xyz".to_string(),
        };

        assert!(proof.verify(Some("abc"), Some("xyz")));
        assert!(!proof.verify(Some("wrong"), Some("xyz")));
    }

    #[test]
    fn test_refresh_from_collections() {
        let registry = ProofRegistry::new(None, None);

        let mut collections = HashMap::new();
        let mut vectors = HashMap::new();
        let mut vector_data = HashMap::new();
        vector_data.insert("vector".to_string(), serde_json::json!([1.0, 2.0, 3.0]));
        vectors.insert("vec1".to_string(), vector_data);

        let state = CollectionState {
            vectors,
            indexes: HashMap::new(),
        };
        collections.insert("test_collection".to_string(), state);

        registry.refresh_from_collections(&collections);

        let proof = registry.get_membership_proof("test_collection", "vec1");
        assert!(proof.is_some());
    }

    #[test]
    fn test_get_commit_snapshot() {
        let registry = ProofRegistry::new(Some("test-kid".to_string()), None);
        let snapshot = registry.get_commit_snapshot();

        assert_eq!(snapshot.get("kid").unwrap(), "test-kid");
        assert!(snapshot.contains_key("commit_root"));
        assert!(snapshot.contains_key("signature"));
    }
}
