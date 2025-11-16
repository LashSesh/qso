/*!
 * Common interfaces for vector store benchmark drivers.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/base.py
 */

use std::collections::HashMap;
use thiserror::Error;

/// Type alias for a vector (sequence of floats)
pub type Vector = Vec<f64>;

/// Type alias for upsert items: (id, vector, optional metadata)
pub type UpsertItem = (String, Vector, Option<HashMap<String, serde_json::Value>>);

/// Error raised when a benchmark target cannot be reached or configured
#[derive(Debug, Error)]
#[error("Driver {name} unavailable: {reason}")]
pub struct DriverUnavailable {
    pub name: String,
    pub reason: String,
}

impl DriverUnavailable {
    /// Create a new DriverUnavailable error
    pub fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Return a JSON-serializable payload describing the skip
    pub fn as_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("name".to_string(), serde_json::json!(self.name));
        map.insert("skipped".to_string(), serde_json::json!(true));
        map.insert("reason".to_string(), serde_json::json!(self.reason));
        map
    }
}

/// Abstract interface every benchmark target driver must implement
pub trait VectorStoreDriver: Send + Sync {
    /// Get the driver name
    fn name(&self) -> &str;

    /// Get the configured metric
    fn metric(&self) -> &str;

    /// Connect to the service or initialize the underlying client
    fn connect(&mut self) -> Result<(), anyhow::Error>;

    /// Drop or empty the target namespace/collection/index
    fn clear(&mut self, namespace: &str) -> Result<(), anyhow::Error>;

    /// Insert/update vectors + metadata in batches
    fn upsert(
        &mut self,
        items: Vec<UpsertItem>,
        namespace: &str,
        batch_size: usize,
    ) -> Result<(), anyhow::Error>;

    /// Return `[(id, score)]` with comparable scoring to the metric
    fn search(
        &self,
        query: &Vector,
        k: usize,
        namespace: &str,
    ) -> Result<Vec<(String, f64)>, anyhow::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_unavailable_creation() {
        let error = DriverUnavailable::new("test-driver", "connection failed");
        assert_eq!(error.name, "test-driver");
        assert_eq!(error.reason, "connection failed");
    }

    #[test]
    fn test_driver_unavailable_as_dict() {
        let error = DriverUnavailable::new("test-driver", "connection failed");
        let dict = error.as_dict();

        assert_eq!(dict.get("name").unwrap(), &serde_json::json!("test-driver"));
        assert_eq!(dict.get("skipped").unwrap(), &serde_json::json!(true));
        assert_eq!(
            dict.get("reason").unwrap(),
            &serde_json::json!("connection failed")
        );
    }

    #[test]
    fn test_driver_unavailable_display() {
        let error = DriverUnavailable::new("test-driver", "connection failed");
        let message = format!("{}", error);
        assert!(message.contains("test-driver"));
        assert!(message.contains("connection failed"));
    }
}
