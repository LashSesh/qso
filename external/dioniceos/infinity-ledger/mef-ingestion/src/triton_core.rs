/*!
 * Triton Core - SPEC-002-compliant normalization
 * Deterministic payload normalization without randomness.
 */

use anyhow::Result;
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Normalized result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedResult {
    pub vector: Vec<f64>,
    pub original_type: String,
    pub data_type: String,
    pub timestamp: String,
    pub metadata: Metadata,
    pub normalized_payload: Value,
}

/// Metadata for normalized result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub size: usize,
    pub hash: String,
}

/// Normalize payload according to SPEC-002
///
/// Rules:
/// - Sort keys
/// - Trim strings
/// - Cast numbers to f64
/// - Convert date strings to ISO-8601
/// - Remove forbidden fields: ["debug", "tmp", "_meta"]
///
/// # Arguments
/// * `payload` - Input dictionary/map
///
/// # Returns
/// Normalized dictionary (deterministic)
pub fn normalize_payload(payload: &Value) -> Value {
    let forbidden_fields = ["debug", "tmp", "_meta"];

    fn clean_dict(d: &Map<String, Value>, forbidden: &[&str]) -> Map<String, Value> {
        let mut cleaned = Map::new();

        // Sort keys for determinism
        let mut keys: Vec<_> = d.keys().collect();
        keys.sort();

        for key in keys {
            // Skip forbidden fields
            if forbidden.contains(&key.as_str()) {
                continue;
            }

            let value = &d[key];

            // Recursively handle nested dicts
            let normalized_value = match value {
                Value::Object(obj) => Value::Object(clean_dict(obj, forbidden)),
                Value::Array(arr) => Value::Array(normalize_list(arr, forbidden)),
                Value::String(s) => {
                    let trimmed = s.trim();
                    if is_date_string(trimmed) {
                        Value::String(convert_to_iso8601(trimmed))
                    } else {
                        Value::String(trimmed.to_string())
                    }
                }
                Value::Number(n) => {
                    // Convert to f64
                    if let Some(f) = n.as_f64() {
                        Value::Number(serde_json::Number::from_f64(f).unwrap_or(n.clone()))
                    } else {
                        value.clone()
                    }
                }
                Value::Bool(_) | Value::Null => value.clone(),
            };

            cleaned.insert(key.clone(), normalized_value);
        }

        cleaned
    }

    fn normalize_list(lst: &[Value], forbidden: &[&str]) -> Vec<Value> {
        lst.iter()
            .map(|item| match item {
                Value::Object(obj) => Value::Object(clean_dict(obj, forbidden)),
                Value::Array(arr) => Value::Array(normalize_list(arr, forbidden)),
                Value::String(s) => {
                    let trimmed = s.trim();
                    if is_date_string(trimmed) {
                        Value::String(convert_to_iso8601(trimmed))
                    } else {
                        Value::String(trimmed.to_string())
                    }
                }
                Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Value::Number(serde_json::Number::from_f64(f).unwrap_or(n.clone()))
                    } else {
                        item.clone()
                    }
                }
                Value::Bool(_) | Value::Null => item.clone(),
            })
            .collect()
    }

    match payload {
        Value::Object(obj) => Value::Object(clean_dict(obj, &forbidden_fields)),
        _ => payload.clone(),
    }
}

/// Check if string represents a date
fn is_date_string(s: &str) -> bool {
    let date_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",                     // YYYY-MM-DD
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", // YYYY-MM-DD HH:MM:SS
        r"^\d{2}/\d{2}/\d{4}$",                     // MM/DD/YYYY or DD/MM/YYYY
        r"^\d{2}\.\d{2}\.\d{4}$",                   // DD.MM.YYYY
    ];

    date_patterns
        .iter()
        .any(|pattern| Regex::new(pattern).unwrap().is_match(s))
}

/// Convert date string to ISO-8601
fn convert_to_iso8601(date_str: &str) -> String {
    // Already ISO-8601
    if date_str.contains('T') && date_str.contains('Z') {
        return date_str.to_string();
    }

    // Try various formats
    let formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
    ];

    for fmt in formats {
        // Try parsing with time
        if fmt.contains("%H") {
            if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, fmt) {
                return format!("{}Z", dt.format("%Y-%m-%dT%H:%M:%S"));
            }
        } else {
            // Try parsing date only
            if let Ok(date) = NaiveDate::parse_from_str(date_str, fmt) {
                return format!("{}T00:00:00Z", date.format("%Y-%m-%d"));
            }
        }
    }

    // Fallback: return unchanged
    date_str.to_string()
}

/// SPEC-002-compliant Triton Core
/// Wrapper for compatibility with existing code
pub struct TritonCore {
    #[allow(dead_code)]
    config: HashMap<String, Value>,
    seed: String,
    vector_dim: usize,
}

impl TritonCore {
    /// Initialize Triton Core
    ///
    /// # Arguments
    /// * `config` - Configuration parameters
    pub fn new(config: HashMap<String, Value>) -> Self {
        let seed = config
            .get("seed")
            .and_then(|v| v.as_str())
            .unwrap_or("MEF_SEED_42")
            .to_string();

        Self {
            config,
            seed,
            vector_dim: 5,
        }
    }

    /// Normalize input data
    ///
    /// # Arguments
    /// * `data` - Input data
    /// * `data_type` - Type hint for data
    ///
    /// # Returns
    /// Normalized data result
    pub fn normalize(&self, data: &Value, data_type: &str) -> Result<NormalizedResult> {
        // Convert to dict if needed
        let payload = if data.is_object() {
            data.clone()
        } else {
            serde_json::json!({
                "data": data,
                "type": data_type
            })
        };

        // SPEC-002 normalization
        let normalized = normalize_payload(&payload);

        // Generate deterministic vector from normalized payload
        let payload_str = serde_json::to_string(&normalized)?;
        let hash_input = format!("{}_{}", payload_str, self.seed);
        let hash_bytes = Sha256::digest(hash_input.as_bytes());

        // Convert to 5D vector
        let mut vector = Vec::with_capacity(self.vector_dim);
        for i in 0..5 {
            // Use 4 bytes per dimension
            let mut byte_slice = [0u8; 4];
            byte_slice.copy_from_slice(&hash_bytes[i * 4..(i + 1) * 4]);
            let value = u32::from_be_bytes(byte_slice) as f64 / (u32::MAX as f64);
            // Scale to [-1, 1]
            vector.push(value * 2.0 - 1.0);
        }

        // Generate deterministic timestamp
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")?;
        let mut seconds_bytes = [0u8; 4];
        seconds_bytes.copy_from_slice(&hash_bytes[20..24]);
        let seconds_offset = u32::from_be_bytes(seconds_bytes) % 86400;
        let timestamp = (base_time + chrono::Duration::seconds(seconds_offset as i64)).to_rfc3339();

        // Compute hash of payload
        let payload_hash = Sha256::digest(payload_str.as_bytes());
        let hash_hex = format!("{:x}", payload_hash);

        Ok(NormalizedResult {
            vector,
            original_type: match data {
                Value::Object(_) => "dict",
                Value::Array(_) => "list",
                Value::String(_) => "str",
                Value::Number(_) => "number",
                Value::Bool(_) => "bool",
                Value::Null => "null",
            }
            .to_string(),
            data_type: data_type.to_string(),
            timestamp,
            metadata: Metadata {
                size: payload_str.len(),
                hash: hash_hex,
            },
            normalized_payload: normalized,
        })
    }
}

impl Default for TritonCore {
    fn default() -> Self {
        let mut config = HashMap::new();
        config.insert("seed".to_string(), Value::String("MEF_SEED_42".to_string()));
        Self::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_normalize_payload_basic() {
        let payload = json!({
            "name": "  Alice  ",
            "age": 30,
            "active": true,
            "debug": "should be removed"
        });

        let normalized = normalize_payload(&payload);

        // Check string trimmed
        assert_eq!(normalized["name"], "Alice");

        // Check number converted to float
        assert_eq!(normalized["age"], 30.0);

        // Check bool preserved
        assert_eq!(normalized["active"], true);

        // Check forbidden field removed
        assert!(normalized.get("debug").is_none());
    }

    #[test]
    fn test_normalize_payload_nested() {
        let payload = json!({
            "user": {
                "name": "  Bob  ",
                "tmp": "should be removed"
            },
            "items": [1, 2, "  test  "]
        });

        let normalized = normalize_payload(&payload);

        // Check nested string trimmed
        assert_eq!(normalized["user"]["name"], "Bob");

        // Check nested forbidden field removed
        assert!(normalized["user"].get("tmp").is_none());

        // Check array item trimmed
        assert_eq!(normalized["items"][2], "test");
    }

    #[test]
    fn test_is_date_string() {
        assert!(is_date_string("2025-01-01"));
        assert!(is_date_string("2025-01-01 12:00:00"));
        assert!(is_date_string("2025-01-01T12:00:00"));
        assert!(is_date_string("01/01/2025"));
        assert!(is_date_string("01.01.2025"));
        assert!(!is_date_string("not a date"));
    }

    #[test]
    fn test_convert_to_iso8601() {
        let result = convert_to_iso8601("2025-01-01");
        assert!(result.starts_with("2025-01-01T"));
        assert!(result.ends_with("Z"));

        let result = convert_to_iso8601("2025-01-01T12:00:00Z");
        assert_eq!(result, "2025-01-01T12:00:00Z");
    }

    #[test]
    fn test_triton_core_normalize() {
        let core = TritonCore::default();
        let data = json!({"test": "data"});

        let result = core.normalize(&data, "raw").unwrap();

        // Check vector has 5 dimensions
        assert_eq!(result.vector.len(), 5);

        // Check all values in [-1, 1]
        for v in &result.vector {
            assert!(*v >= -1.0 && *v <= 1.0);
        }

        // Check metadata
        assert!(result.metadata.size > 0);
        assert_eq!(result.metadata.hash.len(), 64); // SHA256 hex
    }

    #[test]
    fn test_determinism() {
        let core = TritonCore::default();
        let data = json!({"test": "data"});

        let result1 = core.normalize(&data, "raw").unwrap();
        let result2 = core.normalize(&data, "raw").unwrap();

        // Same input should produce same output
        assert_eq!(result1.vector, result2.vector);
        assert_eq!(result1.timestamp, result2.timestamp);
        assert_eq!(result1.metadata.hash, result2.metadata.hash);
    }

    #[test]
    fn test_key_ordering() {
        let payload1 = json!({
            "z": 1,
            "a": 2,
            "m": 3
        });

        let payload2 = json!({
            "a": 2,
            "m": 3,
            "z": 1
        });

        let normalized1 = normalize_payload(&payload1);
        let normalized2 = normalize_payload(&payload2);

        // Keys should be sorted, so serialization should be identical
        let str1 = serde_json::to_string(&normalized1).unwrap();
        let str2 = serde_json::to_string(&normalized2).unwrap();
        assert_eq!(str1, str2);
    }
}
