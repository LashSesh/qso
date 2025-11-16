/*!
 * Driver for Milvus using HTTP API.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/milvus_driver.py
 *
 * This implementation uses the Milvus HTTP API directly via reqwest
 * instead of the pymilvus library to minimize dependencies.
 */

use crate::base::{DriverUnavailable, UpsertItem, Vector, VectorStoreDriver};
use anyhow::{Context, Result};
use serde_json::json;

/// Benchmark driver that interacts with a Milvus deployment via HTTP API
pub struct MilvusDriver {
    metric: String,
    host: String,
    port: String,
    client: Option<reqwest::blocking::Client>,
    dimension: Option<usize>,
}

impl MilvusDriver {
    /// Create a new Milvus driver
    pub fn new(metric: Option<&str>) -> Self {
        let metric = metric.unwrap_or("cosine").to_lowercase();
        let host = std::env::var("MILVUS_HOST").unwrap_or_default();
        let port = std::env::var("MILVUS_PORT").unwrap_or_else(|_| "19530".to_string());

        Self {
            metric,
            host,
            port,
            client: None,
            dimension: None,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    fn ensure_collection(&mut self, namespace: &str, dimension: usize) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        // Check if collection exists
        let url = format!("{}/v1/vector/collections/describe", self.base_url());
        let check_response = client
            .post(&url)
            .json(&json!({"collectionName": namespace}))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .context("Failed to check collection existence")?;

        if check_response.status().is_success() {
            return Ok(());
        }

        // Collection doesn't exist, create it
        let metric_type = self.milvus_metric();

        let create_url = format!("{}/v1/vector/collections/create", self.base_url());
        let payload = json!({
            "collectionName": namespace,
            "dimension": dimension,
            "metricType": metric_type,
            "primaryField": "id",
            "vectorField": "vector"
        });

        let response = client
            .post(&create_url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .context("Failed to create collection")?;

        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to create collection: {}", text));
        }

        Ok(())
    }

    fn flush_batch(&self, namespace: &str, ids: Vec<String>, vectors: Vec<Vec<f64>>) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        let url = format!("{}/v1/vector/insert", self.base_url());

        let data: Vec<serde_json::Value> = ids
            .iter()
            .zip(vectors.iter())
            .map(|(id, vec)| {
                json!({
                    "id": id,
                    "vector": vec
                })
            })
            .collect();

        let payload = json!({
            "collectionName": namespace,
            "data": data
        });

        let response = client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .context("Failed to insert vectors")?;

        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to insert vectors: {}", text));
        }

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

    fn milvus_metric(&self) -> &str {
        match self.metric.as_str() {
            "l2" => "L2",
            "ip" => "IP",
            _ => "COSINE",
        }
    }
}

impl VectorStoreDriver for MilvusDriver {
    fn name(&self) -> &str {
        "Milvus"
    }

    fn metric(&self) -> &str {
        &self.metric
    }

    fn connect(&mut self) -> Result<()> {
        if self.host.is_empty() {
            return Err(DriverUnavailable::new("Milvus", "MILVUS_HOST not configured").into());
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        // Health check
        let url = format!("{}/v1/vector/collections/list", self.base_url());
        let response = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send();

        match response {
            Ok(resp) => {
                if !resp.status().is_success() {
                    return Err(DriverUnavailable::new(
                        "Milvus",
                        format!(
                            "Milvus health check failed at {}:{}. Service may be starting or unhealthy.",
                            self.host, self.port
                        ),
                    )
                    .into());
                }
            }
            Err(e) => {
                return Err(DriverUnavailable::new(
                    "Milvus",
                    format!(
                        "failed to connect to Milvus at {}:{}. Ensure Milvus service is running and healthy. Error: {}",
                        self.host, self.port, e
                    ),
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

        let url = format!("{}/v1/vector/collections/drop", self.base_url());
        let payload = json!({"collectionName": namespace});

        let _ = client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(30))
            .send();

        self.dimension = None;
        Ok(())
    }

    fn upsert(&mut self, items: Vec<UpsertItem>, namespace: &str, batch_size: usize) -> Result<()> {
        if self.client.is_none() {
            return Err(anyhow::anyhow!("connect() must be called before upsert()"));
        }

        let mut pending_ids: Vec<String> = Vec::new();
        let mut pending_vectors: Vec<Vec<f64>> = Vec::new();

        for (identifier, vector, _metadata) in items {
            if self.dimension.is_none() {
                self.dimension = Some(vector.len());
                self.ensure_collection(namespace, vector.len())?;
            }

            let prepared = self.prepare_vector(&vector)?;
            pending_ids.push(identifier);
            pending_vectors.push(prepared);

            if pending_ids.len() >= batch_size {
                self.flush_batch(namespace, pending_ids, pending_vectors)?;
                pending_ids = Vec::new();
                pending_vectors = Vec::new();
            }
        }

        if !pending_ids.is_empty() {
            self.flush_batch(namespace, pending_ids, pending_vectors)?;
        }

        Ok(())
    }

    fn search(&self, query: &Vector, k: usize, namespace: &str) -> Result<Vec<(String, f64)>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before search()"))?;

        // Prepare query vector (need to normalize if needed)
        let mut query_vec = query.clone();
        if self.metric == "cosine" || self.metric == "ip" {
            let norm: f64 = query_vec.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                query_vec = query_vec.iter().map(|&v| v / norm).collect();
            }
        }

        let url = format!("{}/v1/vector/search", self.base_url());
        let payload = json!({
            "collectionName": namespace,
            "vector": query_vec,
            "limit": k,
            "metricType": self.milvus_metric(),
            "outputFields": ["id"]
        });

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

        if let Some(data) = result.get("data").and_then(|d| d.as_array()) {
            for item in data.iter().take(k) {
                if let (Some(id), Some(score)) = (
                    item.get("id").and_then(|v| v.as_str()),
                    item.get("score")
                        .and_then(|v| v.as_f64())
                        .or_else(|| item.get("distance").and_then(|v| v.as_f64())),
                ) {
                    hits.push((id.to_string(), score));
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
    fn test_milvus_driver_creation() {
        let driver = MilvusDriver::new(None);
        assert_eq!(driver.name(), "Milvus");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_milvus_driver_custom_metric() {
        let driver = MilvusDriver::new(Some("l2"));
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_milvus_driver_ip_metric() {
        let driver = MilvusDriver::new(Some("ip"));
        assert_eq!(driver.metric(), "ip");
    }

    #[test]
    fn test_milvus_driver_host_env() {
        std::env::set_var("MILVUS_HOST", "localhost");
        let driver = MilvusDriver::new(None);
        assert_eq!(driver.host, "localhost");
        std::env::remove_var("MILVUS_HOST");
    }

    #[test]
    fn test_milvus_driver_port_env() {
        std::env::set_var("MILVUS_PORT", "19530");
        let driver = MilvusDriver::new(None);
        assert_eq!(driver.port, "19530");
        std::env::remove_var("MILVUS_PORT");
    }

    #[test]
    fn test_milvus_driver_default_port() {
        std::env::remove_var("MILVUS_PORT");
        let driver = MilvusDriver::new(None);
        assert_eq!(driver.port, "19530");
    }

    #[test]
    fn test_connect_without_host() {
        std::env::remove_var("MILVUS_HOST");
        let mut driver = MilvusDriver::new(None);
        let result = driver.connect();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("MILVUS_HOST not configured"));
    }

    #[test]
    fn test_upsert_without_connect() {
        let mut driver = MilvusDriver::new(Some("cosine"));
        let items = vec![("id1".to_string(), vec![1.0, 2.0, 3.0], None)];
        let result = driver.upsert(items, "test", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_without_connect() {
        let driver = MilvusDriver::new(Some("cosine"));
        let query = vec![1.0, 2.0, 3.0];
        let result = driver.search(&query, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_without_connect() {
        let mut driver = MilvusDriver::new(Some("cosine"));
        let result = driver.clear("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_milvus_metric_mapping() {
        let driver_cosine = MilvusDriver::new(Some("cosine"));
        assert_eq!(driver_cosine.milvus_metric(), "COSINE");

        let driver_l2 = MilvusDriver::new(Some("l2"));
        assert_eq!(driver_l2.milvus_metric(), "L2");

        let driver_ip = MilvusDriver::new(Some("ip"));
        assert_eq!(driver_ip.milvus_metric(), "IP");
    }
}
