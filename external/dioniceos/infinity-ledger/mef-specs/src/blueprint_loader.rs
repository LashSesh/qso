/*!
 * Utilities for loading and validating MEF-Core evolution blueprints.
 */

use crate::blueprint_models::Blueprint;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Required top-level keys in a blueprint
pub const REQUIRED_TOP_LEVEL_KEYS: &[&str] = &[
    "spec",
    "priorities",
    "components",
    "storage",
    "api",
    "index_backends",
    "consistency",
    "merkaba_gate",
    "workflows",
    "config",
];

/// Blueprint validation error
#[derive(Debug, Error)]
pub enum BlueprintValidationError {
    #[error("Blueprint schema error: {0}")]
    Schema(String),
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("{0}")]
    Other(String),
}

/// Blueprint schema error (specific type of validation error)
pub type BlueprintSchemaError = BlueprintValidationError;

/// Container for a validated blueprint and its derived artifacts
#[derive(Debug, Clone)]
pub struct BlueprintDocument {
    pub model: Blueprint,
    pub raw: HashMap<String, Value>,
    pub normalized_yaml: String,
    pub spec_hash: String,
}

/// Validate blueprint data against the required structural constraints
fn validate_schema(data: &HashMap<String, Value>) -> Result<(), BlueprintValidationError> {
    // Check required top-level keys
    let missing: Vec<&str> = REQUIRED_TOP_LEVEL_KEYS
        .iter()
        .filter(|key| !data.contains_key(**key))
        .copied()
        .collect();

    if !missing.is_empty() {
        return Err(BlueprintValidationError::Schema(format!(
            "Missing required top-level keys: {}",
            missing.join(", ")
        )));
    }

    // Validate spec
    let spec = data
        .get("spec")
        .and_then(|v| v.as_object())
        .ok_or_else(|| BlueprintValidationError::Schema("spec must be a mapping".to_string()))?;

    let spec_required = vec!["id", "title", "version", "date"];
    let spec_missing: Vec<&str> = spec_required
        .iter()
        .filter(|key| !spec.contains_key(**key))
        .copied()
        .collect();

    if !spec_missing.is_empty() {
        return Err(BlueprintValidationError::Schema(format!(
            "spec missing fields: {}",
            spec_missing.join(", ")
        )));
    }

    // Validate priorities
    let priorities = data
        .get("priorities")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            BlueprintValidationError::Schema("priorities must be a mapping".to_string())
        })?;

    for field in &["must", "should", "could"] {
        if !priorities.contains_key(*field) {
            return Err(BlueprintValidationError::Schema(format!(
                "priorities missing '{}' list",
                field
            )));
        }
    }

    // Validate components
    let components = data
        .get("components")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BlueprintValidationError::Schema("components must be a list".to_string()))?;

    for (index, component) in components.iter().enumerate() {
        let comp_obj = component.as_object().ok_or_else(|| {
            BlueprintValidationError::Schema(format!("components[{}] must be a mapping", index))
        })?;

        for field in &["name", "type"] {
            if !comp_obj.contains_key(*field) {
                return Err(BlueprintValidationError::Schema(format!(
                    "components[{}] missing '{}'",
                    index, field
                )));
            }
        }
    }

    // Validate storage
    let storage = data
        .get("storage")
        .and_then(|v| v.as_object())
        .ok_or_else(|| BlueprintValidationError::Schema("storage must be a mapping".to_string()))?;

    for field in &["fs_root", "s3", "layout"] {
        if !storage.contains_key(*field) {
            return Err(BlueprintValidationError::Schema(format!(
                "storage missing '{}'",
                field
            )));
        }
    }

    // Validate api
    let api = data
        .get("api")
        .and_then(|v| v.as_object())
        .ok_or_else(|| BlueprintValidationError::Schema("api must be a mapping".to_string()))?;

    for field in &["rest", "grpc"] {
        if !api.contains_key(*field) {
            return Err(BlueprintValidationError::Schema(format!(
                "api missing '{}'",
                field
            )));
        }
    }

    // Validate index_backends
    let backends = data
        .get("index_backends")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            BlueprintValidationError::Schema("index_backends must be a mapping".to_string())
        })?;

    for field in &["hnsw", "faiss"] {
        if !backends.contains_key(*field) {
            return Err(BlueprintValidationError::Schema(format!(
                "index_backends missing '{}'",
                field
            )));
        }
    }

    // Validate consistency
    if !data
        .get("consistency")
        .and_then(|v| v.as_object())
        .is_some()
    {
        return Err(BlueprintValidationError::Schema(
            "consistency must be a mapping".to_string(),
        ));
    }

    // Validate merkaba_gate
    let merkaba_gate = data
        .get("merkaba_gate")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            BlueprintValidationError::Schema("merkaba_gate must be a mapping".to_string())
        })?;

    for field in &["graph", "on_fail"] {
        if !merkaba_gate.contains_key(*field) {
            return Err(BlueprintValidationError::Schema(format!(
                "merkaba_gate missing '{}'",
                field
            )));
        }
    }

    // Validate workflows
    let workflows = data
        .get("workflows")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            BlueprintValidationError::Schema("workflows must be a mapping".to_string())
        })?;

    for field in &["upsert", "query", "rebuild"] {
        if !workflows.contains_key(*field) {
            return Err(BlueprintValidationError::Schema(format!(
                "workflows missing '{}'",
                field
            )));
        }
    }

    // Validate config
    let config = data
        .get("config")
        .and_then(|v| v.as_object())
        .ok_or_else(|| BlueprintValidationError::Schema("config must be a mapping".to_string()))?;

    if !config.contains_key("env") {
        return Err(BlueprintValidationError::Schema(
            "config missing 'env'".to_string(),
        ));
    }

    Ok(())
}

/// Render a normalized YAML string with sorted keys
fn normalize_yaml(data: &HashMap<String, Value>) -> String {
    // Convert to Value for serialization
    let value = serde_json::to_value(data).unwrap();

    // Use YAML serialization with sorted keys
    serde_yaml::to_string(&value).unwrap_or_else(|_| {
        // Fallback to JSON if YAML fails
        serde_json::to_string_pretty(&value).unwrap()
    })
}

/// Parse YAML (or JSON) into a dictionary
fn load_yaml(text: &str) -> Result<HashMap<String, Value>, BlueprintValidationError> {
    // Try YAML first
    let loaded: Value = serde_yaml::from_str(text).or_else(|_| serde_json::from_str(text))?;

    if let Value::Object(map) = loaded {
        let result: HashMap<String, Value> = map.into_iter().collect();
        Ok(result)
    } else {
        Err(BlueprintValidationError::Other(
            "Blueprint root must be a mapping".to_string(),
        ))
    }
}

/// Compute the BLAKE2b hash of the normalized YAML representation
fn compute_hash(normalized_yaml: &str) -> String {
    use sha2::{Digest, Sha256};

    // Use SHA256 for compatibility (BLAKE3 is not in workspace dependencies)
    let mut hasher = Sha256::new();
    hasher.update(normalized_yaml.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Load, validate, and normalize a blueprint from disk
pub fn load_blueprint(
    path: impl AsRef<Path>,
) -> Result<BlueprintDocument, BlueprintValidationError> {
    let blueprint_path = path.as_ref();

    if !blueprint_path.exists() {
        return Err(BlueprintValidationError::FileNotFound(
            blueprint_path.display().to_string(),
        ));
    }

    let raw_yaml = std::fs::read_to_string(blueprint_path)?;
    let data = load_yaml(&raw_yaml)?;

    validate_schema(&data)?;

    let model = Blueprint::from_dict(&data);
    let normalized_yaml = normalize_yaml(&data);
    let spec_hash = compute_hash(&normalized_yaml);

    Ok(BlueprintDocument {
        model,
        raw: data,
        normalized_yaml,
        spec_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_minimal_blueprint() -> HashMap<String, Value> {
        serde_json::from_value(serde_json::json!({
            "spec": {
                "id": "SPEC-002",
                "title": "Test Blueprint",
                "version": "1.0",
                "date": "2025-01-01"
            },
            "priorities": {
                "must": ["feature1"],
                "should": ["feature2"],
                "could": ["feature3"]
            },
            "components": [
                {
                    "name": "test-component",
                    "type": "service"
                }
            ],
            "storage": {
                "fs_root": "/data",
                "s3": {"bucket": "test"},
                "layout": {"type": "versioned"}
            },
            "api": {
                "rest": ["/v1/test"],
                "grpc": {"service": "TestService"}
            },
            "index_backends": {
                "hnsw": {"m": 16},
                "faiss": {"index_type": "Flat"}
            },
            "consistency": {
                "mode": "strong"
            },
            "merkaba_gate": {
                "graph": "metatron",
                "on_fail": "reject"
            },
            "workflows": {
                "upsert": ["validate", "index"],
                "query": ["search", "rank"],
                "rebuild": ["backup", "reindex"]
            },
            "config": {
                "env": "production"
            }
        }))
        .unwrap()
    }

    #[test]
    fn test_validate_schema_success() {
        let data = create_minimal_blueprint();
        assert!(validate_schema(&data).is_ok());
    }

    #[test]
    fn test_validate_schema_missing_top_level_key() {
        let mut data = create_minimal_blueprint();
        data.remove("workflows");
        let result = validate_schema(&data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Missing required top-level keys"));
    }

    #[test]
    fn test_validate_schema_missing_spec_field() {
        let mut data = create_minimal_blueprint();
        if let Some(Value::Object(spec)) = data.get_mut("spec") {
            spec.remove("version");
        }
        let result = validate_schema(&data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("spec missing fields"));
    }

    #[test]
    fn test_normalize_yaml() {
        let data = create_minimal_blueprint();
        let yaml = normalize_yaml(&data);
        assert!(!yaml.is_empty());
        assert!(yaml.contains("spec:"));
    }

    #[test]
    fn test_load_yaml_valid_json() {
        let json = r#"{"key": "value"}"#;
        let result = load_yaml(json);
        assert!(result.is_ok());
        let data = result.unwrap();
        assert_eq!(data.get("key").unwrap(), "value");
    }

    #[test]
    fn test_load_yaml_valid_yaml() {
        let yaml = "key: value\n";
        let result = load_yaml(yaml);
        assert!(result.is_ok());
        let data = result.unwrap();
        assert_eq!(data.get("key").unwrap(), "value");
    }

    #[test]
    fn test_compute_hash() {
        let text = "test content";
        let hash = compute_hash(text);
        assert_eq!(hash.len(), 64); // SHA256 produces 64 hex characters
    }

    #[test]
    fn test_load_blueprint_success() {
        let temp_dir = TempDir::new().unwrap();
        let blueprint_path = temp_dir.path().join("blueprint.json");

        let data = create_minimal_blueprint();
        let json = serde_json::to_string_pretty(&data).unwrap();
        fs::write(&blueprint_path, json).unwrap();

        let result = load_blueprint(&blueprint_path);
        assert!(result.is_ok());

        let doc = result.unwrap();
        assert_eq!(doc.model.spec.id, "SPEC-002");
        assert!(!doc.spec_hash.is_empty());
    }

    #[test]
    fn test_load_blueprint_file_not_found() {
        let result = load_blueprint("/nonexistent/path/blueprint.json");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BlueprintValidationError::FileNotFound(_)
        ));
    }

    #[test]
    fn test_load_blueprint_invalid_schema() {
        let temp_dir = TempDir::new().unwrap();
        let blueprint_path = temp_dir.path().join("invalid.json");

        let invalid_data = serde_json::json!({
            "spec": {
                "id": "SPEC-002"
                // Missing required fields
            }
        });

        fs::write(
            &blueprint_path,
            serde_json::to_string_pretty(&invalid_data).unwrap(),
        )
        .unwrap();

        let result = load_blueprint(&blueprint_path);
        assert!(result.is_err());
    }
}
