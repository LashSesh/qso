/*!
 * Acquisition adapters for MEF-Core.
 * Simple fallback adapter used by MEFCorePipeline.
 */

use serde_json::Value;
use std::collections::HashMap;

/// Minimal adapter that wraps raw input into a structured dict
#[derive(Debug, Clone)]
pub struct AcquisitionAdapter {
    source: String,
}

impl AcquisitionAdapter {
    /// Create a new acquisition adapter
    ///
    /// # Arguments
    ///
    /// * `source` - Source identifier (defaults to "generic")
    pub fn new(source: Option<&str>) -> Self {
        Self {
            source: source.unwrap_or("generic").to_string(),
        }
    }

    /// Collect raw input and wrap it into a structured format
    ///
    /// # Arguments
    ///
    /// * `raw_input` - Raw input data as JSON value
    /// * `input_type` - Type of input ("json", "text", or other)
    ///
    /// # Returns
    ///
    /// HashMap with "data" and "metadata" keys
    pub fn collect(&self, raw_input: &Value, input_type: &str) -> HashMap<String, Value> {
        let data = match input_type {
            "json" => {
                if let Value::String(s) = raw_input {
                    serde_json::from_str(s).unwrap_or_else(|_| raw_input.clone())
                } else {
                    raw_input.clone()
                }
            }
            "text" => {
                if let Value::String(s) = raw_input {
                    serde_json::json!({ "text": s.trim() })
                } else {
                    raw_input.clone()
                }
            }
            _ => {
                serde_json::json!({
                    "raw": raw_input,
                    "type": input_type
                })
            }
        };

        let mut result = HashMap::new();
        result.insert("data".to_string(), data);
        result.insert(
            "metadata".to_string(),
            serde_json::json!({ "source": self.source }),
        );
        result
    }
}

impl Default for AcquisitionAdapter {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = AcquisitionAdapter::new(None);
        assert_eq!(adapter.source, "generic");

        let adapter = AcquisitionAdapter::new(Some("custom"));
        assert_eq!(adapter.source, "custom");
    }

    #[test]
    fn test_collect_json() {
        let adapter = AcquisitionAdapter::new(Some("test"));
        let raw = Value::String(r#"{"key": "value"}"#.to_string());
        let result = adapter.collect(&raw, "json");

        assert!(result.contains_key("data"));
        assert!(result.contains_key("metadata"));

        let data = &result["data"];
        assert_eq!(data["key"], "value");

        let metadata = &result["metadata"];
        assert_eq!(metadata["source"], "test");
    }

    #[test]
    fn test_collect_text() {
        let adapter = AcquisitionAdapter::new(None);
        let raw = Value::String("  hello world  ".to_string());
        let result = adapter.collect(&raw, "text");

        let data = &result["data"];
        assert_eq!(data["text"], "hello world");

        let metadata = &result["metadata"];
        assert_eq!(metadata["source"], "generic");
    }

    #[test]
    fn test_collect_other() {
        let adapter = AcquisitionAdapter::new(None);
        let raw = serde_json::json!({"some": "data"});
        let result = adapter.collect(&raw, "custom_type");

        let data = &result["data"];
        assert_eq!(data["raw"], raw);
        assert_eq!(data["type"], "custom_type");
    }

    #[test]
    fn test_default_adapter() {
        let adapter = AcquisitionAdapter::default();
        assert_eq!(adapter.source, "generic");
    }
}
