/*!
 * Driver for Weaviate using HTTP API.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/weaviate_driver.py
 *
 * This implementation uses the Weaviate HTTP API directly via reqwest
 * instead of the weaviate-client library to minimize dependencies.
 */

use crate::base::{DriverUnavailable, UpsertItem, Vector, VectorStoreDriver};
use anyhow::{Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Benchmark driver that interacts with a Weaviate cluster via HTTP API
pub struct WeaviateDriver {
    metric: String,
    base_url: String,
    client: Option<reqwest::blocking::Client>,
    dimension: Option<usize>,
}

impl WeaviateDriver {
    /// Create a new Weaviate driver
    pub fn new(metric: Option<&str>) -> Self {
        let metric = metric.unwrap_or("cosine").to_lowercase();
        let base_url = std::env::var("WEAVIATE_URL")
            .unwrap_or_default()
            .trim_end_matches('/')
            .to_string();

        Self {
            metric,
            base_url,
            client: None,
            dimension: None,
        }
    }

    /// Convert a string ID to a deterministic UUID (v5-style)
    fn id_to_uuid(&self, id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(id.as_bytes());
        let hash = hasher.finalize();

        // Format as UUID (8-4-4-4-12 format)
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            hash[0], hash[1], hash[2], hash[3],
            hash[4], hash[5],
            hash[6], hash[7],
            hash[8], hash[9],
            hash[10], hash[11], hash[12], hash[13], hash[14], hash[15]
        )
    }

    fn class_name(&self, namespace: &str) -> String {
        // Sanitize namespace to valid Weaviate class name
        // Must start with uppercase letter, contain only alphanumeric
        let token: String = namespace.chars().filter(|c| c.is_alphanumeric()).collect();

        let token = if token.is_empty() {
            "Namespace".to_string()
        } else if token.chars().next().unwrap().is_alphabetic() {
            token
        } else {
            format!("N{}", token)
        };

        // Capitalize first letter
        let mut chars = token.chars();
        match chars.next() {
            None => "Namespace".to_string(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }

    fn ensure_class(&mut self, namespace: &str, dimension: usize) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        let class_name = self.class_name(namespace);

        // Check if class exists
        let url = format!("{}/v1/schema", self.base_url);
        let response = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .context("Failed to get schema")?;

        if response.status().is_success() {
            let schema: serde_json::Value = response.json().context("Failed to parse schema")?;

            if let Some(classes) = schema.get("classes").and_then(|c| c.as_array()) {
                for class in classes {
                    if let Some(name) = class.get("class").and_then(|n| n.as_str()) {
                        if name == class_name {
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Class doesn't exist, create it
        let distance = match self.metric.as_str() {
            "cosine" => "cosine",
            "ip" => "dot",
            "l2" => "l2-squared",
            _ => "cosine",
        };

        let payload = json!({
            "class": class_name,
            "description": "Benchmark dataset",
            "vectorizer": "none",
            "moduleConfig": {},
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": distance
            }
        });

        let create_url = format!("{}/v1/schema", self.base_url);
        let response = client
            .post(&create_url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .context("Failed to create class")?;

        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to create class: {}", text));
        }

        self.dimension = Some(dimension);
        Ok(())
    }

    fn prepare_vector(&mut self, vector: &Vector) -> Result<Vec<f64>> {
        let array = vector.clone();

        if let Some(dim) = self.dimension {
            if array.len() != dim {
                return Err(anyhow::anyhow!(
                    "dimension mismatch: expected {}, received {}",
                    dim,
                    array.len()
                ));
            }
        } else {
            self.dimension = Some(array.len());
        }

        // Normalize for cosine and ip metrics
        if self.metric == "cosine" || self.metric == "ip" {
            let norm: f64 = array.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                return Ok(array.iter().map(|&v| v / norm).collect());
            }
        }

        Ok(array)
    }
}

impl VectorStoreDriver for WeaviateDriver {
    fn name(&self) -> &str {
        "Weaviate"
    }

    fn metric(&self) -> &str {
        &self.metric
    }

    fn connect(&mut self) -> Result<()> {
        if self.base_url.is_empty() {
            return Err(DriverUnavailable::new("Weaviate", "WEAVIATE_URL not configured").into());
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        // Health check
        let url = format!("{}/v1/schema", self.base_url);
        let response = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send();

        match response {
            Ok(resp) => {
                if !resp.status().is_success() {
                    return Err(DriverUnavailable::new(
                        "Weaviate",
                        format!("unable to reach Weaviate: HTTP {}", resp.status()),
                    )
                    .into());
                }
            }
            Err(e) => {
                return Err(DriverUnavailable::new(
                    "Weaviate",
                    format!("unable to reach Weaviate: {}", e),
                )
                .into());
            }
        }

        self.client = Some(client);
        Ok(())
    }

    fn clear(&mut self, namespace: &str) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before clear()"))?;

        let class_name = self.class_name(namespace);
        let url = format!("{}/v1/schema/{}", self.base_url, class_name);

        let _ = client
            .delete(&url)
            .timeout(std::time::Duration::from_secs(30))
            .send();

        self.dimension = None;
        Ok(())
    }

    fn upsert(&mut self, items: Vec<UpsertItem>, namespace: &str, batch_size: usize) -> Result<()> {
        if self.client.is_none() {
            return Err(anyhow::anyhow!("connect() must be called before upsert()"));
        }

        if items.is_empty() {
            return Ok(());
        }

        if self.dimension.is_none() {
            self.dimension = Some(items[0].1.len());
        }

        let class_name = self.class_name(namespace);
        self.ensure_class(namespace, items[0].1.len())?;

        let client = self.client.clone().unwrap();

        // Batch insert
        let mut batch: Vec<serde_json::Value> = Vec::new();

        for (identifier, vector, _metadata) in items {
            let prepared = self.prepare_vector(&vector)?;
            let uuid = self.id_to_uuid(&identifier);

            let object = json!({
                "class": class_name,
                "id": uuid,
                "vector": prepared,
                "properties": {
                    "original_id": identifier
                }
            });

            batch.push(object);

            if batch.len() >= batch_size {
                // Send batch
                let url = format!("{}/v1/batch/objects", self.base_url);
                let payload = json!({"objects": batch});

                let response = client
                    .post(&url)
                    .json(&payload)
                    .timeout(std::time::Duration::from_secs(60))
                    .send()
                    .context("Failed to batch insert objects")?;

                if !response.status().is_success() {
                    let text = response.text().unwrap_or_default();
                    return Err(anyhow::anyhow!("Batch insert failed: {}", text));
                }

                batch.clear();
            }
        }

        // Send remaining batch
        if !batch.is_empty() {
            let url = format!("{}/v1/batch/objects", self.base_url);
            let payload = json!({"objects": batch});

            let response = client
                .post(&url)
                .json(&payload)
                .timeout(std::time::Duration::from_secs(60))
                .send()
                .context("Failed to batch insert objects")?;

            if !response.status().is_success() {
                let text = response.text().unwrap_or_default();
                return Err(anyhow::anyhow!("Batch insert failed: {}", text));
            }
        }

        Ok(())
    }

    fn search(&self, query: &Vector, k: usize, namespace: &str) -> Result<Vec<(String, f64)>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before search()"))?;

        let class_name = self.class_name(namespace);

        // Prepare query vector
        let mut query_vec = query.clone();
        if self.metric == "cosine" || self.metric == "ip" {
            let norm: f64 = query_vec.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                query_vec = query_vec.iter().map(|&v| v / norm).collect();
            }
        }

        let url = format!("{}/v1/graphql", self.base_url);
        let graphql_query = format!(
            r#"{{ Get {{ {} (limit: {}, nearVector: {{ vector: {:?} }}) {{ _additional {{ id distance }} }} }} }}"#,
            class_name, k, query_vec
        );

        let payload = json!({"query": graphql_query});

        let response = client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .context("Failed to search vectors")?;

        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!("Search failed: {}", text));
        }

        let result: serde_json::Value =
            response.json().context("Failed to parse search response")?;

        let mut hits: Vec<(String, f64)> = Vec::new();

        if let Some(get) = result.get("data").and_then(|d| d.get("Get")) {
            if let Some(class_results) = get.get(&class_name).and_then(|c| c.as_array()) {
                for entry in class_results.iter().take(k) {
                    if let Some(additional) = entry.get("_additional") {
                        if let (Some(id), Some(distance)) = (
                            additional.get("id").and_then(|v| v.as_str()),
                            additional.get("distance").and_then(|v| v.as_f64()),
                        ) {
                            hits.push((id.to_string(), distance));
                        }
                    }
                }
            }
        }

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weaviate_driver_creation() {
        let driver = WeaviateDriver::new(None);
        assert_eq!(driver.name(), "Weaviate");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_weaviate_driver_custom_metric() {
        let driver = WeaviateDriver::new(Some("l2"));
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_weaviate_driver_ip_metric() {
        let driver = WeaviateDriver::new(Some("ip"));
        assert_eq!(driver.metric(), "ip");
    }

    #[test]
    fn test_weaviate_driver_base_url_env() {
        std::env::set_var("WEAVIATE_URL", "http://localhost:8080/");
        let driver = WeaviateDriver::new(None);
        assert_eq!(driver.base_url, "http://localhost:8080");
        std::env::remove_var("WEAVIATE_URL");
    }

    #[test]
    fn test_weaviate_driver_default_url() {
        std::env::remove_var("WEAVIATE_URL");
        let driver = WeaviateDriver::new(None);
        assert_eq!(driver.base_url, "");
    }

    #[test]
    fn test_connect_without_url() {
        std::env::remove_var("WEAVIATE_URL");
        let mut driver = WeaviateDriver::new(None);
        let result = driver.connect();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("WEAVIATE_URL not configured"));
    }

    #[test]
    fn test_upsert_without_connect() {
        let mut driver = WeaviateDriver::new(Some("cosine"));
        let items = vec![("id1".to_string(), vec![1.0, 2.0, 3.0], None)];
        let result = driver.upsert(items, "test", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_without_connect() {
        let driver = WeaviateDriver::new(Some("cosine"));
        let query = vec![1.0, 2.0, 3.0];
        let result = driver.search(&query, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_without_connect() {
        let mut driver = WeaviateDriver::new(Some("cosine"));
        let result = driver.clear("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_class_name_sanitization() {
        let driver = WeaviateDriver::new(None);

        // Normal case
        assert_eq!(driver.class_name("test"), "Test");

        // With special characters
        assert_eq!(driver.class_name("test-collection"), "Testcollection");

        // Starting with number
        assert_eq!(driver.class_name("123test"), "N123test");

        // Empty after sanitization
        assert_eq!(driver.class_name("---"), "Namespace");

        // Already capitalized
        assert_eq!(driver.class_name("MyClass"), "MyClass");
    }

    #[test]
    fn test_distance_metric_mapping() {
        let driver_cosine = WeaviateDriver::new(Some("cosine"));
        assert_eq!(driver_cosine.metric(), "cosine");

        let driver_l2 = WeaviateDriver::new(Some("l2"));
        assert_eq!(driver_l2.metric(), "l2");

        let driver_ip = WeaviateDriver::new(Some("ip"));
        assert_eq!(driver_ip.metric(), "ip");
    }
}
