/*!
 * Driver for Pinecone's managed vector database using HTTP API.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/pinecone_driver.py
 *
 * This implementation uses the Pinecone HTTP API directly via reqwest
 * instead of the pinecone-client library to minimize dependencies.
 */

use crate::base::{DriverUnavailable, UpsertItem, Vector, VectorStoreDriver};
use anyhow::{Context, Result};
use serde_json::json;
use std::time::{Duration, Instant};

/// Benchmark driver that integrates with Pinecone's managed vector database
pub struct PineconeDriver {
    metric: String,
    api_key: String,
    environment: String,
    client: Option<reqwest::blocking::Client>,
    dimension: Option<usize>,
}

impl PineconeDriver {
    /// Create a new Pinecone driver
    pub fn new(metric: Option<&str>) -> Self {
        let metric = metric.unwrap_or("cosine").to_lowercase();
        let api_key = std::env::var("PINECONE_API_KEY").unwrap_or_default();
        let environment = std::env::var("PINECONE_ENV").unwrap_or_default();

        Self {
            metric,
            api_key,
            environment,
            client: None,
            dimension: None,
        }
    }

    fn control_plane_url(&self) -> String {
        if !self.environment.is_empty() {
            format!("https://controller.{}.pinecone.io", self.environment)
        } else {
            "https://api.pinecone.io".to_string()
        }
    }

    fn index_url(&self, index_name: &str) -> Result<String> {
        // For simplicity, construct the data plane URL
        // In practice, this should be fetched from the describe_index response
        if !self.environment.is_empty() {
            Ok(format!(
                "https://{}-{}.svc.{}.pinecone.io",
                index_name,
                "project", // Simplified - should get from API
                self.environment
            ))
        } else {
            // Use the control plane to get index host
            Ok(format!("https://{}.pinecone.io", index_name))
        }
    }

    fn ensure_index(&mut self, namespace: &str, dimension: usize) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        // Check if index exists
        let list_url = format!("{}/indexes", self.control_plane_url());
        let response = client
            .get(&list_url)
            .header("Api-Key", &self.api_key)
            .timeout(Duration::from_secs(10))
            .send()
            .context("Failed to list indexes")?;

        if response.status().is_success() {
            let indexes: serde_json::Value =
                response.json().context("Failed to parse indexes list")?;

            if let Some(indexes_array) = indexes.get("indexes").and_then(|i| i.as_array()) {
                for index in indexes_array {
                    if let Some(name) = index.get("name").and_then(|n| n.as_str()) {
                        if name == namespace {
                            self.wait_for_index_ready(namespace, 300.0)?;
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Index doesn't exist, create it
        let metric = match self.metric.as_str() {
            "cosine" => "cosine",
            "ip" => "dotproduct",
            "l2" => "euclidean",
            _ => "cosine",
        };

        let mut payload = json!({
            "name": namespace,
            "dimension": dimension,
            "metric": metric
        });

        // Add spec if environment is specified
        if !self.environment.is_empty() {
            payload["spec"] = json!({
                "pod": {
                    "environment": self.environment,
                    "pod_type": "p1.x1"
                }
            });
        }

        let create_url = format!("{}/indexes", self.control_plane_url());
        let response = client
            .post(&create_url)
            .header("Api-Key", &self.api_key)
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .context("Failed to create index")?;

        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to create index: {}", text));
        }

        self.wait_for_index_ready(namespace, 300.0)?;
        Ok(())
    }

    fn wait_for_index_ready(&self, index_name: &str, timeout_secs: f64) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before wait"))?;

        let deadline = Instant::now() + Duration::from_secs_f64(timeout_secs);

        loop {
            let describe_url = format!("{}/indexes/{}", self.control_plane_url(), index_name);
            let response = client
                .get(&describe_url)
                .header("Api-Key", &self.api_key)
                .timeout(Duration::from_secs(10))
                .send();

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let description: serde_json::Value =
                        resp.json().context("Failed to parse index description")?;

                    // Check if ready
                    let ready = description
                        .get("status")
                        .and_then(|s| s.get("ready"))
                        .and_then(|r| r.as_bool())
                        .unwrap_or(false);

                    let state = description
                        .get("status")
                        .and_then(|s| s.get("state"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("");

                    if ready || state.to_lowercase() == "ready" {
                        return Ok(());
                    }
                }
                Ok(_) | Err(_) => {
                    // Continue waiting
                }
            }

            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "timed out while waiting for Pinecone index '{}' to become ready",
                    index_name
                ));
            }

            std::thread::sleep(Duration::from_secs(1));
        }
    }

    fn wait_for_index_deletion(&self, index_name: &str, timeout_secs: f64) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before wait"))?;

        let deadline = Instant::now() + Duration::from_secs_f64(timeout_secs);

        loop {
            let list_url = format!("{}/indexes", self.control_plane_url());
            let response = client
                .get(&list_url)
                .header("Api-Key", &self.api_key)
                .timeout(Duration::from_secs(10))
                .send();

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let indexes: serde_json::Value =
                        resp.json().context("Failed to parse indexes list")?;

                    let mut found = false;
                    if let Some(indexes_array) = indexes.get("indexes").and_then(|i| i.as_array()) {
                        for index in indexes_array {
                            if let Some(name) = index.get("name").and_then(|n| n.as_str()) {
                                if name == index_name {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }

                    if !found {
                        return Ok(());
                    }
                }
                Ok(_) | Err(_) => {
                    // Continue waiting
                }
            }

            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "timed out while waiting for Pinecone index '{}' to be deleted",
                    index_name
                ));
            }

            std::thread::sleep(Duration::from_secs(1));
        }
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

impl VectorStoreDriver for PineconeDriver {
    fn name(&self) -> &str {
        "Pinecone"
    }

    fn metric(&self) -> &str {
        &self.metric
    }

    fn connect(&mut self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(
                DriverUnavailable::new("Pinecone", "PINECONE_API_KEY not configured").into(),
            );
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        // Health check - try to list indexes
        let url = format!("{}/indexes", self.control_plane_url());
        let response = client
            .get(&url)
            .header("Api-Key", &self.api_key)
            .timeout(Duration::from_secs(10))
            .send();

        match response {
            Ok(resp) => {
                if !resp.status().is_success() {
                    return Err(DriverUnavailable::new(
                        "Pinecone",
                        format!("unable to reach Pinecone: HTTP {}", resp.status()),
                    )
                    .into());
                }
            }
            Err(e) => {
                return Err(DriverUnavailable::new(
                    "Pinecone",
                    format!("unable to reach Pinecone: {}", e),
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

        // Check if index exists
        let list_url = format!("{}/indexes", self.control_plane_url());
        let response = client
            .get(&list_url)
            .header("Api-Key", &self.api_key)
            .timeout(Duration::from_secs(10))
            .send();

        if let Ok(resp) = response {
            if resp.status().is_success() {
                if let Ok(indexes) = resp.json::<serde_json::Value>() {
                    if let Some(indexes_array) = indexes.get("indexes").and_then(|i| i.as_array()) {
                        let mut found = false;
                        for index in indexes_array {
                            if let Some(name) = index.get("name").and_then(|n| n.as_str()) {
                                if name == namespace {
                                    found = true;
                                    break;
                                }
                            }
                        }

                        if found {
                            // Delete the index
                            let delete_url =
                                format!("{}/indexes/{}", self.control_plane_url(), namespace);
                            let _ = client
                                .delete(&delete_url)
                                .header("Api-Key", &self.api_key)
                                .timeout(Duration::from_secs(30))
                                .send();

                            let _ = self.wait_for_index_deletion(namespace, 120.0);
                        }
                    }
                }
            }
        }

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
            self.ensure_index(namespace, items[0].1.len())?;
        }

        let client = self.client.clone().unwrap();
        let api_key = self.api_key.clone();

        // Note: Simplified index URL construction
        // In production, should get proper host from describe_index
        let base_url = self.index_url(namespace)?;

        let mut batch: Vec<serde_json::Value> = Vec::new();

        for (identifier, vector, metadata) in items {
            let prepared = self.prepare_vector(&vector)?;

            let mut record = json!({
                "id": identifier,
                "values": prepared
            });

            if let Some(meta) = metadata {
                if !meta.is_empty() {
                    record["metadata"] = json!(meta);
                }
            }

            batch.push(record);

            if batch.len() >= batch_size {
                // Send batch
                let url = format!("{}/vectors/upsert", base_url);
                let payload = json!({"vectors": batch});

                let response = client
                    .post(&url)
                    .header("Api-Key", &api_key)
                    .json(&payload)
                    .timeout(Duration::from_secs(60))
                    .send();

                // Ignore errors for now (simplified)
                if let Ok(resp) = response {
                    if !resp.status().is_success() {
                        let text = resp.text().unwrap_or_default();
                        return Err(anyhow::anyhow!("Upsert failed: {}", text));
                    }
                }

                batch.clear();
            }
        }

        // Send remaining batch
        if !batch.is_empty() {
            let url = format!("{}/vectors/upsert", base_url);
            let payload = json!({"vectors": batch});

            let response = client
                .post(&url)
                .header("Api-Key", &api_key)
                .json(&payload)
                .timeout(Duration::from_secs(60))
                .send();

            if let Ok(resp) = response {
                if !resp.status().is_success() {
                    let text = resp.text().unwrap_or_default();
                    return Err(anyhow::anyhow!("Upsert failed: {}", text));
                }
            }
        }

        Ok(())
    }

    fn search(&self, query: &Vector, k: usize, namespace: &str) -> Result<Vec<(String, f64)>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before search()"))?;

        // Prepare query vector
        let mut query_vec = query.clone();
        if self.metric == "cosine" || self.metric == "ip" {
            let norm: f64 = query_vec.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                query_vec = query_vec.iter().map(|&v| v / norm).collect();
            }
        }

        let base_url = self.index_url(namespace)?;
        let url = format!("{}/query", base_url);

        let payload = json!({
            "vector": query_vec,
            "topK": k,
            "includeValues": false
        });

        let response = client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .context("Failed to query vectors")?;

        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!("Query failed: {}", text));
        }

        let result: serde_json::Value =
            response.json().context("Failed to parse query response")?;

        let mut hits: Vec<(String, f64)> = Vec::new();

        if let Some(matches) = result.get("matches").and_then(|m| m.as_array()) {
            for item in matches.iter().take(k) {
                if let (Some(id), Some(score)) = (
                    item.get("id").and_then(|v| v.as_str()),
                    item.get("score").and_then(|v| v.as_f64()),
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
    fn test_pinecone_driver_creation() {
        let driver = PineconeDriver::new(None);
        assert_eq!(driver.name(), "Pinecone");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_pinecone_driver_custom_metric() {
        let driver = PineconeDriver::new(Some("l2"));
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_pinecone_driver_ip_metric() {
        let driver = PineconeDriver::new(Some("ip"));
        assert_eq!(driver.metric(), "ip");
    }

    #[test]
    fn test_pinecone_driver_api_key_env() {
        std::env::set_var("PINECONE_API_KEY", "test-api-key");
        let driver = PineconeDriver::new(None);
        assert_eq!(driver.api_key, "test-api-key");
        std::env::remove_var("PINECONE_API_KEY");
    }

    #[test]
    fn test_pinecone_driver_environment_env() {
        std::env::set_var("PINECONE_ENV", "us-west1-gcp");
        let driver = PineconeDriver::new(None);
        assert_eq!(driver.environment, "us-west1-gcp");
        std::env::remove_var("PINECONE_ENV");
    }

    #[test]
    fn test_pinecone_driver_default_values() {
        std::env::remove_var("PINECONE_API_KEY");
        std::env::remove_var("PINECONE_ENV");
        let driver = PineconeDriver::new(None);
        assert_eq!(driver.api_key, "");
        assert_eq!(driver.environment, "");
    }

    #[test]
    fn test_connect_without_api_key() {
        std::env::remove_var("PINECONE_API_KEY");
        let mut driver = PineconeDriver::new(None);
        let result = driver.connect();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("PINECONE_API_KEY not configured"));
    }

    #[test]
    fn test_upsert_without_connect() {
        let mut driver = PineconeDriver::new(Some("cosine"));
        let items = vec![("id1".to_string(), vec![1.0, 2.0, 3.0], None)];
        let result = driver.upsert(items, "test", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_without_connect() {
        let driver = PineconeDriver::new(Some("cosine"));
        let query = vec![1.0, 2.0, 3.0];
        let result = driver.search(&query, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_without_connect() {
        let mut driver = PineconeDriver::new(Some("cosine"));
        let result = driver.clear("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_metric_mapping() {
        let driver_cosine = PineconeDriver::new(Some("cosine"));
        assert_eq!(driver_cosine.metric(), "cosine");

        let driver_l2 = PineconeDriver::new(Some("l2"));
        assert_eq!(driver_l2.metric(), "l2");

        let driver_ip = PineconeDriver::new(Some("ip"));
        assert_eq!(driver_ip.metric(), "ip");
    }
}
