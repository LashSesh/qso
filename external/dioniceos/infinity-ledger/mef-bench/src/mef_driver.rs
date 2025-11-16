/*!
 * Driver that exercises the MEF HTTP API for apples-to-apples comparisons.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/mef_driver.py
 */

use crate::base::{DriverUnavailable, UpsertItem, Vector, VectorStoreDriver};
use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashMap;

/// Driver that talks to the MEF REST API used by the existing bench
pub struct MEFDriver {
    metric: String,
    base_url: String,
    client: Option<reqwest::blocking::Client>,
}

impl MEFDriver {
    /// Create a new MEF driver
    pub fn new(metric: Option<&str>) -> Self {
        let metric = metric.unwrap_or("cosine").to_lowercase();
        let base_url = std::env::var("MEF_BASE_URL")
            .or_else(|_| std::env::var("QUALITY_BASE_URL"))
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let base_url = base_url.trim_end_matches('/').to_string();

        Self {
            metric,
            base_url,
            client: None,
        }
    }

    /// Flush a batch of vectors to the MEF API
    fn flush_batch(
        &self,
        namespace: &str,
        batch: &[HashMap<String, serde_json::Value>],
    ) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        let url = format!("{}/collections/{}/upsert", self.base_url, namespace);
        let payload = json!({
            "vectors": batch,
        });

        let response = client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .context("Failed to send upsert request")?;

        response
            .error_for_status()
            .context("Upsert request failed")?;

        Ok(())
    }
}

impl VectorStoreDriver for MEFDriver {
    fn name(&self) -> &str {
        "MEF"
    }

    fn metric(&self) -> &str {
        &self.metric
    }

    fn connect(&mut self) -> Result<(), anyhow::Error> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client")?;

        let health_url = format!("{}/healthz", self.base_url);
        let response = client
            .get(&health_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .map_err(|e| {
                DriverUnavailable::new("MEF", format!("failed to contact {}: {}", health_url, e))
            })?;

        if response.status().as_u16() >= 500 {
            return Err(DriverUnavailable::new(
                "MEF",
                format!("service unhealthy: HTTP {}", response.status()),
            )
            .into());
        }

        self.client = Some(client);
        Ok(())
    }

    fn clear(&mut self, namespace: &str) -> Result<(), anyhow::Error> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before clear()"))?;

        let url = format!("{}/collections/{}/upsert", self.base_url, namespace);
        let payload = json!({
            "vectors": [],
        });

        let response = client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .context("Failed to send clear request")?;

        let status = response.status().as_u16();
        if status == 200 || status == 204 || status == 404 {
            return Ok(());
        }

        if status >= 500 {
            return Err(anyhow::anyhow!(
                "failed to clear namespace {}: HTTP {}",
                namespace,
                status
            ));
        }

        response
            .error_for_status()
            .context("Clear namespace failed")?;

        Ok(())
    }

    fn upsert(
        &mut self,
        items: Vec<UpsertItem>,
        namespace: &str,
        batch_size: usize,
    ) -> Result<(), anyhow::Error> {
        if self.client.is_none() {
            return Err(anyhow::anyhow!("connect() must be called before upsert()"));
        }

        let mut batch: Vec<HashMap<String, serde_json::Value>> = Vec::new();

        for (identifier, vector, metadata) in items {
            let mut payload = HashMap::new();
            payload.insert("id".to_string(), json!(identifier));
            payload.insert("vector".to_string(), json!(vector.to_vec()));
            payload.insert("epoch".to_string(), json!(1)); // Default epoch for benchmarking

            if let Some(meta) = metadata {
                payload.insert("metadata".to_string(), json!(meta));
            }

            batch.push(payload);

            if batch.len() >= batch_size {
                self.flush_batch(namespace, &batch)?;
                batch.clear();
            }
        }

        if !batch.is_empty() {
            self.flush_batch(namespace, &batch)?;
        }

        Ok(())
    }

    fn search(
        &self,
        query: &Vector,
        k: usize,
        namespace: &str,
    ) -> Result<Vec<(String, f64)>, anyhow::Error> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before search()"))?;

        let url = format!("{}/search", self.base_url);
        let payload = json!({
            "collection": namespace,
            "query_vector": query.to_vec(),
            "top_k": k as i32,
            "mode": "ann",
            "solve": false,
            "membership_proof": false,
            "pipeline_proof": false,
        });

        let response = client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .context("Failed to send search request")?;

        let response = response
            .error_for_status()
            .context("Search request failed")?;

        let body: serde_json::Value = response.json().context("Failed to parse search response")?;
        let results = body
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'results' field"))?;

        let mut hits = Vec::new();
        for entry in results {
            if let Some(id) = entry.get("id").and_then(|v| v.as_str()) {
                let score = entry.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                hits.push((id.to_string(), score));
            }
        }

        // Truncate to k results
        hits.truncate(k);

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mef_driver_creation() {
        let driver = MEFDriver::new(None);
        assert_eq!(driver.name(), "MEF");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_mef_driver_custom_metric() {
        let driver = MEFDriver::new(Some("l2"));
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_mef_driver_base_url_env() {
        std::env::set_var("MEF_BASE_URL", "http://example.com:9000");
        let driver = MEFDriver::new(None);
        assert_eq!(driver.base_url, "http://example.com:9000");
        std::env::remove_var("MEF_BASE_URL");
    }

    #[test]
    fn test_mef_driver_default_url() {
        std::env::remove_var("MEF_BASE_URL");
        std::env::remove_var("QUALITY_BASE_URL");
        let driver = MEFDriver::new(None);
        assert_eq!(driver.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_upsert_without_connect() {
        let mut driver = MEFDriver::new(None);
        let items = vec![("id1".to_string(), vec![1.0, 2.0, 3.0], None)];
        let result = driver.upsert(items, "test", 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("connect()"));
    }

    #[test]
    fn test_search_without_connect() {
        let driver = MEFDriver::new(None);
        let query = vec![1.0, 2.0, 3.0];
        let result = driver.search(&query, 10, "test");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("connect()"));
    }
}
