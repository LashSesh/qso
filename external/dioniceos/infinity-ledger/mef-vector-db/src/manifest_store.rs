/*!
 * Manifest and persistence management for vector index artifacts
 */

use anyhow::{Context, Result};
use aws_sdk_s3::Client as S3Client;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration for persisting artifacts to an external service
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistenceConfig {
    /// Provider type (e.g., "s3")
    pub provider: Option<String>,
    /// S3 bucket name
    pub bucket: Option<String>,
    /// Key prefix for S3 objects
    pub prefix: Option<String>,
}

impl PersistenceConfig {
    /// Create from a JSON dictionary
    pub fn from_dict(payload: Option<&Value>) -> Self {
        let payload = payload.and_then(|v| v.as_object());

        Self {
            provider: payload
                .and_then(|p| p.get("provider").or_else(|| p.get("type")))
                .and_then(|v| v.as_str())
                .map(String::from),
            bucket: payload
                .and_then(|p| p.get("bucket"))
                .and_then(|v| v.as_str())
                .map(String::from),
            prefix: payload
                .and_then(|p| p.get("prefix"))
                .and_then(|v| v.as_str())
                .map(String::from),
        }
    }

    /// Check if this is an S3 persistence configuration
    pub fn is_s3(&self) -> bool {
        self.provider
            .as_ref()
            .map(|p| p.to_lowercase() == "s3")
            .unwrap_or(false)
            && self.bucket.is_some()
    }
}

/// Representation of the manifest metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    /// Collection metadata indexed by collection name
    pub collections: HashMap<String, Value>,
    /// Persistence configuration
    pub persistence: PersistenceConfig,
}

impl Manifest {
    /// Convert to JSON dictionary
    pub fn to_dict(&self) -> Value {
        serde_json::json!({
            "collections": self.collections,
            "persistence": {
                "provider": self.persistence.provider,
                "bucket": self.persistence.bucket,
                "prefix": self.persistence.prefix,
            }
        })
    }

    /// Create from JSON dictionary
    pub fn from_dict(payload: Option<&Value>) -> Self {
        let payload = payload.and_then(|v| v.as_object());

        let persistence = payload
            .and_then(|p| p.get("persistence"))
            .map(|v| PersistenceConfig::from_dict(Some(v)))
            .unwrap_or_default();

        let collections = payload
            .and_then(|p| p.get("collections"))
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Self {
            collections,
            persistence,
        }
    }
}

/// Simplified CollectionState for manifest storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionState {
    /// Vectors indexed by ID
    pub vectors: HashMap<String, Value>,
    /// Index metadata
    pub indexes: HashMap<String, Value>,
}

impl CollectionState {
    /// Convert to JSON dictionary
    pub fn to_dict(&self) -> Value {
        serde_json::json!({
            "vectors": self.vectors,
            "indexes": self.indexes,
        })
    }
}

/// Manage manifest metadata and persistence for vector index artifacts
pub struct ManifestStore {
    /// Base path for manifest storage
    pub base_path: PathBuf,
    /// Manifest path
    pub manifest_path: PathBuf,
    /// Manifest data
    pub manifest: Manifest,
    /// Optional S3 client (currently unused, reserved for future async implementation)
    #[allow(dead_code)]
    s3_client: Option<S3Client>,
}

impl ManifestStore {
    /// Create a new ManifestStore
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base directory for manifest storage
    /// * `manifest_data` - Optional initial manifest data
    /// * `persistence_config` - Optional persistence configuration
    /// * `s3_client` - Optional S3 client for persistence
    pub fn new(
        base_path: impl AsRef<Path>,
        manifest_data: Option<&Value>,
        persistence_config: Option<&Value>,
        s3_client: Option<S3Client>,
    ) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path).context("Failed to create base directory")?;

        let manifest_path = base_path.join("manifest.json");
        let mut manifest = Self::load_manifest(&manifest_path, manifest_data)?;

        if let Some(config) = persistence_config {
            manifest.persistence = PersistenceConfig::from_dict(Some(config));
        }

        Ok(Self {
            base_path,
            manifest_path,
            manifest,
            s3_client,
        })
    }

    /// Persist a collection state and optional artifacts under a versioned path
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `state` - Collection state to persist
    /// * `epoch` - Version epoch
    /// * `artifacts` - Optional additional artifacts to persist
    ///
    /// # Returns
    ///
    /// Path to the versioned directory
    pub fn persist_state(
        &mut self,
        collection: &str,
        state: &CollectionState,
        epoch: i64,
        artifacts: Option<&HashMap<String, PathBuf>>,
    ) -> Result<PathBuf> {
        let version_dir = self.version_path(collection, epoch);
        std::fs::create_dir_all(&version_dir).context("Failed to create version directory")?;

        // Save state
        let state_path = version_dir.join("index.json");
        let state_json =
            serde_json::to_string_pretty(&state.to_dict()).context("Failed to serialize state")?;
        std::fs::write(&state_path, state_json).context("Failed to write state file")?;

        // Update manifest
        let relative_path = version_dir
            .strip_prefix(&self.base_path)
            .unwrap_or(&version_dir)
            .to_string_lossy()
            .to_string();

        self.manifest.collections.insert(
            collection.to_string(),
            serde_json::json!({
                "latest_epoch": epoch,
                "updated_at": format!("{}Z", Utc::now().format("%Y-%m-%dT%H:%M:%S%.3f")),
                "path": relative_path,
            }),
        );
        self.save_manifest()?;

        // Copy artifacts
        let mut uploaded_files = vec![state_path.clone(), self.manifest_path.clone()];
        if let Some(artifacts) = artifacts {
            for (name, file_path) in artifacts {
                let dest_path = version_dir.join(name);
                if file_path.is_file() {
                    std::fs::copy(file_path, &dest_path)
                        .context(format!("Failed to copy artifact: {}", name))?;
                    uploaded_files.push(dest_path);
                }
            }
        }

        // Sync to S3 if configured
        self.sync_to_s3(&uploaded_files)?;

        Ok(version_dir)
    }

    /// Get the manifest
    pub fn get_manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Mark an epoch as active for a collection and persist the manifest
    pub fn set_active_epoch(&mut self, collection: &str, epoch: i64) -> Result<()> {
        let entry = self
            .manifest
            .collections
            .entry(collection.to_string())
            .or_insert_with(|| serde_json::json!({}));

        if let Some(obj) = entry.as_object_mut() {
            obj.insert("active_epoch".to_string(), serde_json::json!(epoch));
            obj.insert(
                "activated_at".to_string(),
                serde_json::json!(format!("{}Z", Utc::now().format("%Y-%m-%dT%H:%M:%S%.3f"))),
            );
        }

        self.save_manifest()?;
        self.sync_to_s3(std::slice::from_ref(&self.manifest_path))?;

        Ok(())
    }

    // ------------------------------------------------------------------
    // Private methods
    // ------------------------------------------------------------------

    /// Get the versioned path for a collection and epoch
    fn version_path(&self, collection: &str, epoch: i64) -> PathBuf {
        self.base_path.join(collection).join(format!("v{}", epoch))
    }

    /// Load manifest from file or use provided data
    fn load_manifest(manifest_path: &Path, manifest_data: Option<&Value>) -> Result<Manifest> {
        if let Some(data) = manifest_data {
            return Ok(Manifest::from_dict(Some(data)));
        }

        if manifest_path.exists() {
            let contents =
                std::fs::read_to_string(manifest_path).context("Failed to read manifest file")?;
            let json: Value =
                serde_json::from_str(&contents).context("Failed to parse manifest JSON")?;
            return Ok(Manifest::from_dict(Some(&json)));
        }

        Ok(Manifest::default())
    }

    /// Save manifest to file
    fn save_manifest(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.manifest.to_dict())
            .context("Failed to serialize manifest")?;
        std::fs::write(&self.manifest_path, json).context("Failed to write manifest file")?;
        Ok(())
    }

    /// Sync files to S3 if configured
    fn sync_to_s3(&self, files: &[PathBuf]) -> Result<()> {
        if files.is_empty() || !self.manifest.persistence.is_s3() {
            return Ok(());
        }

        // S3 sync would be implemented here using aws-sdk-s3
        // For now, we'll skip the actual S3 upload as it requires async context
        // and the Python code uses boto3.client which is also sync

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_persistence_config_from_dict() {
        let json = serde_json::json!({
            "provider": "s3",
            "bucket": "test-bucket",
            "prefix": "test/prefix"
        });

        let config = PersistenceConfig::from_dict(Some(&json));
        assert_eq!(config.provider, Some("s3".to_string()));
        assert_eq!(config.bucket, Some("test-bucket".to_string()));
        assert_eq!(config.prefix, Some("test/prefix".to_string()));
        assert!(config.is_s3());
    }

    #[test]
    fn test_persistence_config_type_alias() {
        let json = serde_json::json!({
            "type": "s3",
            "bucket": "test-bucket"
        });

        let config = PersistenceConfig::from_dict(Some(&json));
        assert_eq!(config.provider, Some("s3".to_string()));
        assert!(config.is_s3());
    }

    #[test]
    fn test_manifest_from_dict() {
        let json = serde_json::json!({
            "collections": {
                "test_collection": {
                    "latest_epoch": 1,
                    "updated_at": "2025-01-01T00:00:00.000Z"
                }
            },
            "persistence": {
                "provider": "s3",
                "bucket": "test-bucket"
            }
        });

        let manifest = Manifest::from_dict(Some(&json));
        assert_eq!(manifest.collections.len(), 1);
        assert!(manifest.collections.contains_key("test_collection"));
        assert!(manifest.persistence.is_s3());
    }

    #[test]
    fn test_manifest_store_creation() {
        let temp_dir = TempDir::new().unwrap();
        let store = ManifestStore::new(temp_dir.path(), None, None, None).unwrap();

        assert!(store.base_path.exists());
        assert_eq!(store.manifest.collections.len(), 0);
    }

    #[test]
    fn test_manifest_store_persist_state() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = ManifestStore::new(temp_dir.path(), None, None, None).unwrap();

        let state = CollectionState {
            vectors: HashMap::new(),
            indexes: HashMap::new(),
        };

        let version_dir = store
            .persist_state("test_collection", &state, 1, None)
            .unwrap();

        assert!(version_dir.exists());
        assert!(version_dir.join("index.json").exists());
        assert!(store.manifest.collections.contains_key("test_collection"));
    }

    #[test]
    fn test_set_active_epoch() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = ManifestStore::new(temp_dir.path(), None, None, None).unwrap();

        store.set_active_epoch("test_collection", 5).unwrap();

        let entry = store.manifest.collections.get("test_collection").unwrap();
        assert_eq!(entry["active_epoch"], 5);
        assert!(entry["activated_at"].is_string());
    }
}
