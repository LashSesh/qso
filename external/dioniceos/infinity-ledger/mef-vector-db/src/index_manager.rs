/*!
 * Utilities for managing persisted vector collections and index metadata.
 */

use anyhow::{Context, Result};
use chrono::Utc;
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use crate::providers::{get_provider, get_providers, IndexProvider};

/// Default vector database path
fn default_vector_db_path() -> PathBuf {
    env::var("VECTOR_DB_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("mef")
                .join("vector_db")
        })
}

/// Representation of a single vector record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub values: Vec<f64>,
    pub metadata: HashMap<String, Value>,
    pub epoch: Option<i64>,
}

impl VectorRecord {
    /// Create a new VectorRecord
    pub fn new(
        id: String,
        values: Vec<f64>,
        metadata: HashMap<String, Value>,
        epoch: Option<i64>,
    ) -> Self {
        Self {
            id,
            values,
            metadata,
            epoch,
        }
    }

    /// Convert the record to a serializable dictionary
    pub fn to_dict(&self) -> HashMap<String, Value> {
        let mut dict = HashMap::new();
        dict.insert("id".to_string(), Value::from(self.id.clone()));
        dict.insert("vector".to_string(), Value::from(self.values.clone()));
        dict.insert(
            "metadata".to_string(),
            serde_json::to_value(&self.metadata).unwrap(),
        );
        dict.insert(
            "epoch".to_string(),
            self.epoch.map(Value::from).unwrap_or(Value::Null),
        );
        dict
    }

    /// Create a VectorRecord from a dictionary
    pub fn from_dict(payload: &HashMap<String, Value>) -> Result<Self> {
        let id = payload
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'id' field"))?
            .to_string();

        let values = payload
            .get("vector")
            .or_else(|| payload.get("values"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let metadata = payload
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let epoch = payload.get("epoch").and_then(|v| v.as_i64());

        Ok(Self {
            id,
            values,
            metadata,
            epoch,
        })
    }
}

/// In-memory representation of a collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionState {
    pub vectors: HashMap<String, HashMap<String, Value>>,
    pub indexes: HashMap<String, Value>,
}

impl CollectionState {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            indexes: HashMap::new(),
        }
    }

    pub fn to_dict(&self) -> HashMap<String, Value> {
        let mut dict = HashMap::new();
        dict.insert(
            "vectors".to_string(),
            serde_json::to_value(&self.vectors).unwrap(),
        );
        dict.insert(
            "indexes".to_string(),
            serde_json::to_value(&self.indexes).unwrap(),
        );
        dict
    }

    pub fn from_dict(payload: &HashMap<String, Value>) -> Self {
        let vectors = payload
            .get("vectors")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let indexes = payload
            .get("indexes")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Self { vectors, indexes }
    }
}

impl Default for CollectionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Manage persisted vector sets and related index metadata
pub struct IndexManager {
    pub base_path: PathBuf,
    pub collections: HashMap<String, CollectionState>,
    pub collection_providers: HashMap<String, String>,
    provider_instances: HashMap<String, Box<dyn IndexProvider>>,
    #[allow(dead_code)]
    ephemeral_provider_cache: HashMap<String, Box<dyn IndexProvider>>,
    #[allow(dead_code)]
    ephemeral_cache_limit: usize,
    last_search_plan: HashMap<String, Value>,
    index_status: HashMap<String, HashMap<String, Value>>,
}

// Volatile key names for metadata canonicalization
const VOLATILE_KEY_NAMES: &[&str] = &[
    "timestamp",
    "created",
    "created_at",
    "updated",
    "updated_at",
    "stored_at",
    "ingested_at",
    "acquired_at",
    "activated_at",
    "generated_at",
    "refreshed_at",
    "expires_at",
    "expiration",
    "last_updated",
    "last_modified",
];

const VOLATILE_KEY_SUFFIXES: &[&str] = &["_ts", "_timestamp"];

impl IndexManager {
    /// Create a new IndexManager
    pub fn new(base_path: Option<PathBuf>) -> Result<Self> {
        let base_path = base_path.unwrap_or_else(default_vector_db_path);
        fs::create_dir_all(&base_path).context("Failed to create base directory")?;

        let ephemeral_cache_limit = env::var("INDEX_EPHEMERAL_CACHE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);

        let mut manager = Self {
            base_path,
            collections: HashMap::new(),
            collection_providers: HashMap::new(),
            provider_instances: HashMap::new(),
            ephemeral_provider_cache: HashMap::new(),
            ephemeral_cache_limit,
            last_search_plan: HashMap::new(),
            index_status: HashMap::new(),
        };

        manager.load_existing_state()?;
        Ok(manager)
    }

    /// Insert or update vector records for a collection
    pub fn upsert_vectors(
        &mut self,
        collection: &str,
        records: Vec<VectorRecord>,
        epoch: Option<i64>,
        indexes: Option<HashMap<String, Value>>,
    ) -> Result<CollectionState> {
        if records.is_empty() {
            return Ok(self
                .collections
                .get(collection)
                .cloned()
                .unwrap_or_default());
        }

        // Build updates first
        let mut updates = Vec::new();
        for record in records {
            let record_epoch = record.epoch.or(epoch).ok_or_else(|| {
                anyhow::anyhow!("Epoch must be provided either per record or as an argument")
            })?;

            let deterministic_updated_at = self
                .collections
                .get(collection)
                .and_then(|s| s.vectors.get(&record.id))
                .and_then(|v| {
                    v.get("updated_at")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                })
                .unwrap_or_else(|| format!("epoch-{}", record_epoch));

            let metadata = Self::canonicalize_metadata(
                &record.metadata,
                &format!("{}:{}", collection, record.id),
            );

            let mut vector_payload = HashMap::new();
            vector_payload.insert("vector".to_string(), Value::from(record.values.clone()));
            vector_payload.insert(
                "metadata".to_string(),
                serde_json::to_value(&metadata).unwrap(),
            );
            vector_payload.insert("epoch".to_string(), Value::from(record_epoch));
            vector_payload.insert(
                "updated_at".to_string(),
                Value::from(deterministic_updated_at),
            );

            updates.push((record.id.clone(), vector_payload));
        }

        // Now apply updates
        let state = self.collections.entry(collection.to_string()).or_default();

        for (id, payload) in &updates {
            state.vectors.insert(id.clone(), payload.clone());
        }

        if let Some(idx) = indexes {
            state.indexes.extend(idx);
        }

        let provider_name = state
            .indexes
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("hnsw")
            .to_string();

        state
            .indexes
            .entry("provider".to_string())
            .or_insert_with(|| Value::from(provider_name.clone()));

        let proof_version = state
            .indexes
            .get("proof_version")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + 1;
        state
            .indexes
            .insert("proof_version".to_string(), Value::from(proof_version));

        let result = state.clone();

        // Must persist after getting the result but before updating provider
        let collection_str = collection.to_string();
        self.persist_collection(&collection_str, &result)?;
        self.index_status.remove(collection);

        // Update provider after persisting
        if let Ok(provider) = self.ensure_provider(collection) {
            for (id, payload) in &updates {
                provider.upsert(id, payload);
            }
        }

        Ok(result)
    }

    /// Delete vectors from a collection
    pub fn delete_vectors(
        &mut self,
        collection: &str,
        vector_ids: &[String],
        epoch: Option<i64>,
    ) -> Result<CollectionState> {
        let state = self.collections.entry(collection.to_string()).or_default();

        let mut removed = false;
        for vector_id in vector_ids {
            if state.vectors.remove(vector_id).is_some() {
                removed = true;
            }
        }

        if removed {
            if let Some(ep) = epoch {
                let deletion_epochs = state
                    .indexes
                    .entry("deletion_epochs".to_string())
                    .or_insert_with(|| Value::Array(vec![]));
                if let Some(arr) = deletion_epochs.as_array_mut() {
                    arr.push(Value::from(ep));
                }
            }

            let proof_version = state
                .indexes
                .get("proof_version")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                + 1;
            state
                .indexes
                .insert("proof_version".to_string(), Value::from(proof_version));

            let result = state.clone();
            let collection_str = collection.to_string();
            self.persist_collection(&collection_str, &result)?;

            // Update provider after persisting
            if let Ok(provider) = self.ensure_provider(collection) {
                for vector_id in vector_ids {
                    provider.delete(vector_id);
                }
            }

            return Ok(result);
        }

        Ok(state.clone())
    }

    /// Retrieve the current in-memory state for a collection
    pub fn get_collection_state(&mut self, collection: &str) -> CollectionState {
        self.collections
            .entry(collection.to_string())
            .or_default()
            .clone()
    }

    /// Run a similarity search without mutating collection state
    pub fn search_vectors(
        &mut self,
        collection: &str,
        query: &[f64],
        top_k: usize,
        provider_name: Option<&str>,
        mode: Option<&str>,
        ef_search: Option<i64>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        let state = self.collections.get(collection);
        if state.is_none() || state.unwrap().vectors.is_empty() {
            debug!(
                "search requested for empty collection; collection={} has_state={}",
                collection,
                state.is_some()
            );
            return Ok(Vec::new());
        }

        // Clone state to avoid borrow issues
        let state = state.unwrap().clone();
        let _use_exact = mode.map(|m| m.to_lowercase() == "exact").unwrap_or(false);

        // Exact search implementation would go here (omitted for brevity)
        // For now, delegate to provider

        let provider_name_str = provider_name.unwrap_or("").to_string();
        let use_ephemeral = !provider_name_str.is_empty();

        let mut extra_params = HashMap::new();
        if let Some(ef) = ef_search {
            extra_params.insert("ef_search".to_string(), Value::from(ef));
        }

        let start = std::time::Instant::now();

        let results = if use_ephemeral {
            let mut provider = self.get_ephemeral_provider(collection, &provider_name_str)?;
            provider.search(query, &state.vectors, top_k, &extra_params)
        } else {
            let provider = self.ensure_provider(collection)?;
            provider.search(query, &state.vectors, top_k, &extra_params)
        };

        let _total_ms = start.elapsed().as_secs_f64() * 1000.0;

        let plan = if use_ephemeral {
            HashMap::new() // Ephemeral provider doesn't persist plan
        } else {
            self.provider_instances
                .get(collection)
                .and_then(|p| p.get_last_plan())
                .unwrap_or_default()
        };
        self.last_search_plan = plan;

        let ranked: Vec<HashMap<String, Value>> = results
            .iter()
            .map(|(vector_id, score)| {
                let payload = state.vectors.get(vector_id).unwrap();
                let mut result = HashMap::new();
                result.insert("id".to_string(), Value::from(vector_id.clone()));
                result.insert("score".to_string(), Value::from(*score));
                result.insert(
                    "epoch".to_string(),
                    payload.get("epoch").cloned().unwrap_or(Value::Null),
                );
                result.insert(
                    "metadata".to_string(),
                    payload.get("metadata").cloned().unwrap_or(Value::Null),
                );
                result
            })
            .collect();

        Ok(ranked)
    }

    /// Get last search plan
    pub fn last_search_plan(&self) -> HashMap<String, Value> {
        self.last_search_plan.clone()
    }

    /// Build index for a collection
    pub fn build_index(&mut self, collection: &str) -> Result<HashMap<String, Value>> {
        let state = self
            .collections
            .get(collection)
            .ok_or_else(|| anyhow::anyhow!("Collection not found: {}", collection))?
            .clone();

        let start = std::time::Instant::now();
        if let Ok(provider) = self.ensure_provider(collection) {
            provider.build(&state.vectors);
        }
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        let provider_name = self
            .collection_providers
            .get(collection)
            .cloned()
            .unwrap_or_else(|| "hnsw".to_string());

        let mut params = HashMap::new();
        params.insert(
            "metric".to_string(),
            state
                .indexes
                .get("metric")
                .cloned()
                .unwrap_or_else(|| Value::from("cosine")),
        );
        params.insert("provider".to_string(), Value::from(provider_name.clone()));

        let mut status = HashMap::new();
        status.insert("collection".to_string(), Value::from(collection));
        status.insert("ready".to_string(), Value::from(true));
        status.insert(
            "points_indexed".to_string(),
            Value::from(state.vectors.len()),
        );
        status.insert("provider".to_string(), Value::from(provider_name));
        status.insert("params".to_string(), serde_json::to_value(&params).unwrap());
        status.insert("duration_ms".to_string(), Value::from(duration_ms));
        status.insert(
            "updated_at".to_string(),
            Value::from(Utc::now().to_rfc3339()),
        );
        status.insert(
            "proof_version".to_string(),
            state
                .indexes
                .get("proof_version")
                .cloned()
                .unwrap_or(Value::from(0)),
        );

        self.index_status
            .insert(collection.to_string(), status.clone());
        Ok(status)
    }

    /// Get index status for a collection
    pub fn get_index_status(&self, collection: &str) -> HashMap<String, Value> {
        let state = self.collections.get(collection);
        let points_indexed = state.map(|s| s.vectors.len()).unwrap_or(0);
        let proof_version = state
            .and_then(|s| s.indexes.get("proof_version").and_then(|v| v.as_i64()))
            .unwrap_or(0);

        if let Some(status) = self.index_status.get(collection) {
            let mut status = status.clone();
            status.insert("points_indexed".to_string(), Value::from(points_indexed));
            status.insert("proof_version".to_string(), Value::from(proof_version));
            return status;
        }

        let provider_name = self.collection_providers.get(collection);
        let mut status = HashMap::new();
        status.insert("collection".to_string(), Value::from(collection));
        status.insert("ready".to_string(), Value::from(false));
        status.insert("points_indexed".to_string(), Value::from(points_indexed));
        status.insert(
            "provider".to_string(),
            provider_name
                .map(|s| Value::from(s.clone()))
                .unwrap_or(Value::Null),
        );
        status.insert("params".to_string(), Value::Object(Default::default()));
        status.insert("updated_at".to_string(), Value::Null);
        status.insert("proof_version".to_string(), Value::from(proof_version));

        status
    }

    /// List available providers
    pub fn list_providers(&self) -> HashMap<String, HashMap<String, Value>> {
        let providers = get_providers();
        let mut catalogue = HashMap::new();

        for (name, (_, config)) in providers {
            let mut info = HashMap::new();
            info.insert(
                "default_config".to_string(),
                serde_json::to_value(&config).unwrap(),
            );
            catalogue.insert(name, info);
        }

        catalogue
    }

    /// Set provider for a collection
    pub fn set_collection_provider(
        &mut self,
        collection: &str,
        provider_name: &str,
    ) -> Result<HashMap<String, Value>> {
        let providers = get_providers();
        if !providers.contains_key(provider_name) {
            return Err(anyhow::anyhow!("unknown provider: {}", provider_name));
        }

        let mut state = self.get_collection_state(collection);
        let previous_provider = state
            .indexes
            .get("provider")
            .and_then(|v| v.as_str())
            .map(String::from);

        let proof_version = state
            .indexes
            .get("proof_version")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + 1;

        state
            .indexes
            .insert("provider".to_string(), Value::from(provider_name));
        state
            .indexes
            .insert("proof_version".to_string(), Value::from(proof_version));

        self.collections
            .insert(collection.to_string(), state.clone());
        self.collection_providers
            .insert(collection.to_string(), provider_name.to_string());
        self.provider_instances.remove(collection);

        let mut status = HashMap::new();
        status.insert("collection".to_string(), Value::from(collection));
        status.insert("provider".to_string(), Value::from(provider_name));
        status.insert(
            "previous_provider".to_string(),
            previous_provider.map(Value::from).unwrap_or(Value::Null),
        );
        status.insert("proof_version".to_string(), Value::from(proof_version));
        status.insert("ready".to_string(), Value::from(false));
        status.insert(
            "updated_at".to_string(),
            Value::from(Utc::now().to_rfc3339()),
        );

        self.index_status
            .insert(collection.to_string(), status.clone());
        self.persist_collection(collection, &state)?;

        Ok(status)
    }

    // Internal helpers

    fn collection_path(&self, collection: &str) -> PathBuf {
        self.base_path.join(format!("{}.json", collection))
    }

    fn persist_collection(&self, collection: &str, state: &CollectionState) -> Result<()> {
        let path = self.collection_path(collection);
        let data = state.to_dict();
        let json = serde_json::to_string_pretty(&data)?;
        fs::write(&path, json).context(format!("Failed to write collection {}", collection))?;
        Ok(())
    }

    fn load_existing_state(&mut self) -> Result<()> {
        let entries = fs::read_dir(&self.base_path).context("Failed to read base directory")?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match fs::read_to_string(&path) {
                    Ok(content) => {
                        if let Ok(payload) =
                            serde_json::from_str::<HashMap<String, Value>>(&content)
                        {
                            let collection = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let mut state = CollectionState::from_dict(&payload);

                            // Canonicalize metadata for all vectors
                            for (vector_id, vector_payload) in state.vectors.iter_mut() {
                                if let Some(metadata_val) = vector_payload.get("metadata") {
                                    if let Some(metadata_obj) = metadata_val.as_object() {
                                        let metadata: HashMap<String, Value> = metadata_obj
                                            .iter()
                                            .map(|(k, v)| (k.clone(), v.clone()))
                                            .collect();
                                        let canonical = Self::canonicalize_metadata(
                                            &metadata,
                                            &format!("{}:{}", collection, vector_id),
                                        );
                                        vector_payload.insert(
                                            "metadata".to_string(),
                                            serde_json::to_value(&canonical).unwrap(),
                                        );
                                    }
                                }
                            }

                            self.collections.insert(collection.clone(), state.clone());

                            if let Some(provider_val) = state.indexes.get("provider") {
                                if let Some(provider_name) = provider_val.as_str() {
                                    self.collection_providers
                                        .insert(collection, provider_name.to_string());
                                }
                            }
                        }
                    }
                    Err(_) => continue, // Skip corrupted files
                }
            }
        }

        Ok(())
    }

    fn ensure_provider(&mut self, collection: &str) -> Result<&mut Box<dyn IndexProvider>> {
        let provider_name = self
            .collection_providers
            .get(collection)
            .cloned()
            .or_else(|| {
                self.collections
                    .get(collection)
                    .and_then(|s| s.indexes.get("provider"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .or_else(|| {
                let providers = get_providers();
                providers.keys().next().cloned()
            })
            .unwrap_or_else(|| "hnsw".to_string());

        if !self.provider_instances.contains_key(collection) {
            let mut provider = get_provider(Some(&provider_name));
            let state = self.collections.entry(collection.to_string()).or_default();
            provider.build(&state.vectors);
            self.provider_instances
                .insert(collection.to_string(), provider);
            self.collection_providers
                .insert(collection.to_string(), provider_name);
        }

        Ok(self.provider_instances.get_mut(collection).unwrap())
    }

    fn get_ephemeral_provider(
        &mut self,
        collection: &str,
        provider_name: &str,
    ) -> Result<Box<dyn IndexProvider>> {
        let providers = get_providers();
        if !providers.contains_key(provider_name) {
            return Err(anyhow::anyhow!("unknown provider: {}", provider_name));
        }

        let state = self
            .collections
            .get(collection)
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        let _cache_key = format!("{}:{}", collection, provider_name);

        // For now, skip caching and just create a new provider
        let mut provider = get_provider(Some(provider_name));
        provider.build(&state.vectors);

        Ok(provider)
    }

    fn canonicalize_metadata(
        metadata: &HashMap<String, Value>,
        context: &str,
    ) -> HashMap<String, Value> {
        let mut removed = Vec::new();
        let canonical = Self::strip_volatile(metadata, "", &mut removed);

        if !removed.is_empty() {
            debug!(
                "canonicalized metadata for {}; stripped volatile keys={:?}",
                context, removed
            );
        }

        canonical
    }

    fn strip_volatile(
        value: &HashMap<String, Value>,
        prefix: &str,
        removed: &mut Vec<String>,
    ) -> HashMap<String, Value> {
        let mut canonical = HashMap::new();

        for (key, val) in value {
            let lower = key.to_lowercase();
            let full_key = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{}.{}", prefix, key)
            };

            if VOLATILE_KEY_NAMES.contains(&lower.as_str())
                || VOLATILE_KEY_SUFFIXES
                    .iter()
                    .any(|suffix| lower.ends_with(suffix))
            {
                removed.push(full_key);
                continue;
            }

            if let Some(obj) = val.as_object() {
                let nested: HashMap<String, Value> =
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                canonical.insert(
                    key.clone(),
                    serde_json::to_value(Self::strip_volatile(&nested, &full_key, removed))
                        .unwrap(),
                );
            } else if let Some(arr) = val.as_array() {
                canonical.insert(key.clone(), Value::Array(arr.clone()));
            } else {
                canonical.insert(key.clone(), val.clone());
            }
        }

        canonical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_vector_record_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::from("test"));

        let record = VectorRecord::new(
            "vec1".to_string(),
            vec![1.0, 2.0, 3.0],
            metadata.clone(),
            Some(1),
        );

        assert_eq!(record.id, "vec1");
        assert_eq!(record.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(record.epoch, Some(1));
    }

    #[test]
    fn test_vector_record_to_from_dict() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::from("test"));

        let record = VectorRecord::new(
            "vec1".to_string(),
            vec![1.0, 2.0, 3.0],
            metadata.clone(),
            Some(1),
        );

        let dict = record.to_dict();
        let restored = VectorRecord::from_dict(&dict).unwrap();

        assert_eq!(restored.id, "vec1");
        assert_eq!(restored.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(restored.epoch, Some(1));
    }

    #[test]
    fn test_collection_state() {
        let state = CollectionState::new();
        assert!(state.vectors.is_empty());
        assert!(state.indexes.is_empty());
    }

    #[test]
    fn test_index_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = IndexManager::new(Some(temp_dir.path().to_path_buf())).unwrap();
        assert_eq!(manager.collections.len(), 0);
    }

    #[test]
    fn test_upsert_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = IndexManager::new(Some(temp_dir.path().to_path_buf())).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::from("test"));

        let record = VectorRecord::new("vec1".to_string(), vec![1.0, 0.0, 0.0], metadata, Some(1));

        let state = manager
            .upsert_vectors("test_collection", vec![record], None, None)
            .unwrap();
        assert_eq!(state.vectors.len(), 1);
        assert!(state.vectors.contains_key("vec1"));
    }

    #[test]
    fn test_delete_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = IndexManager::new(Some(temp_dir.path().to_path_buf())).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::from("test"));

        let record = VectorRecord::new("vec1".to_string(), vec![1.0, 0.0, 0.0], metadata, Some(1));

        manager
            .upsert_vectors("test_collection", vec![record], None, None)
            .unwrap();
        let state = manager
            .delete_vectors("test_collection", &["vec1".to_string()], Some(2))
            .unwrap();

        assert_eq!(state.vectors.len(), 0);
    }

    #[test]
    fn test_canonicalize_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), Value::from("test"));
        metadata.insert("timestamp".to_string(), Value::from("2024-01-01"));
        metadata.insert("created_at".to_string(), Value::from("2024-01-01"));
        metadata.insert("value_ts".to_string(), Value::from(12345));

        let canonical = IndexManager::canonicalize_metadata(&metadata, "test");

        assert!(canonical.contains_key("name"));
        assert!(!canonical.contains_key("timestamp"));
        assert!(!canonical.contains_key("created_at"));
        assert!(!canonical.contains_key("value_ts"));
    }

    #[test]
    fn test_list_providers() {
        let temp_dir = TempDir::new().unwrap();
        let manager = IndexManager::new(Some(temp_dir.path().to_path_buf())).unwrap();
        let providers = manager.list_providers();

        assert!(providers.contains_key("hnsw"));
        assert!(providers.contains_key("ivf_pq"));
    }
}
