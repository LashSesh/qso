/*!
 * Driver for Elasticsearch/OpenSearch dense vector kNN APIs.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/elastic_driver.py
 */

use crate::base::{DriverUnavailable, UpsertItem, Vector, VectorStoreDriver};
use anyhow::{Context, Result};
use serde_json::json;

/// Benchmark driver targeting an Elasticsearch-compatible endpoint
pub struct ElasticDriver {
    metric: String,
    base_url: String,
    client: Option<reqwest::blocking::Client>,
    dimension: Option<usize>,
}

impl ElasticDriver {
    /// Create a new Elasticsearch driver
    pub fn new(metric: Option<&str>) -> Self {
        let metric = metric.unwrap_or("cosine").to_lowercase();
        let base_url = std::env::var("ELASTIC_URL")
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

    fn ensure_index(&mut self, namespace: &str, dimension: usize) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        let similarity = match self.metric.as_str() {
            "cosine" => "cosine",
            "ip" => "dot_product",
            "l2" => "l2_norm",
            _ => "cosine",
        };

        let payload = json!({
            "settings": {
                "index": {
                    "knn": true,
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": true,
                        "similarity": similarity,
                    }
                }
            },
        });

        let url = format!("{}/{}", self.base_url, namespace);
        let response = client
            .put(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .context("Failed to create index")?;

        let status = response.status().as_u16();
        if status == 200 || status == 201 {
            return Ok(());
        }

        if status == 400 {
            let text = response.text().unwrap_or_default();
            if text.contains("resource_already_exists_exception") {
                return Ok(());
            }
            return Err(anyhow::anyhow!("Failed to create index: {}", text));
        }

        if status == 404 {
            return Err(DriverUnavailable::new(
                "Elastic",
                "Elasticsearch server rejected index creation".to_string(),
            )
            .into());
        }

        Err(anyhow::anyhow!("Failed to create index: HTTP {}", status))
    }

    fn flush_bulk(&self, lines: Vec<String>) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before upsert()"))?;

        let body = lines.join("\n") + "\n";
        let url = format!("{}/_bulk", self.base_url);

        let response = client
            .post(&url)
            .header("Content-Type", "application/x-ndjson")
            .body(body)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .context("Failed to send bulk request")?;

        response.error_for_status_ref()?;

        let payload: serde_json::Value =
            response.json().context("Failed to parse bulk response")?;

        if payload
            .get("errors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return Err(anyhow::anyhow!("bulk ingest reported errors"));
        }

        Ok(())
    }

    fn prepare_vector(&mut self, vector: &Vector) -> Result<Vec<f32>> {
        let array: Vec<f32> = vector.iter().map(|&v| v as f32).collect();

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

        // Normalize for cosine/ip metrics
        if self.metric == "cosine" || self.metric == "ip" {
            let norm: f32 = array.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                return Ok(array.iter().map(|&v| v / norm).collect());
            }
        }

        Ok(array)
    }
}

impl VectorStoreDriver for ElasticDriver {
    fn name(&self) -> &str {
        "Elastic"
    }

    fn metric(&self) -> &str {
        &self.metric
    }

    fn connect(&mut self) -> Result<()> {
        if self.base_url.is_empty() {
            return Err(DriverUnavailable::new(
                "Elastic",
                "ELASTIC_URL not configured".to_string(),
            )
            .into());
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client")?;

        let health_url = format!("{}/_cluster/health", self.base_url);
        let response = client
            .get(&health_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .map_err(|e| {
                DriverUnavailable::new("Elastic", format!("unable to reach Elasticsearch: {}", e))
            })?;

        if response.status().as_u16() >= 500 {
            return Err(DriverUnavailable::new(
                "Elastic",
                format!("cluster unhealthy: HTTP {}", response.status()),
            )
            .into());
        }

        self.client = Some(client);
        Ok(())
    }

    fn clear(&mut self, namespace: &str) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before clear()"))?;

        let url = format!("{}/{}", self.base_url, namespace);
        let response = client
            .delete(&url)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .context("Failed to delete index")?;

        let status = response.status().as_u16();
        if status != 200 && status != 202 && status != 204 && status != 404 {
            response.error_for_status()?;
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

        // Set dimension from first item
        if self.dimension.is_none() {
            self.dimension = Some(items[0].1.len());
            self.ensure_index(namespace, items[0].1.len())?;
        }

        let mut bulk_lines: Vec<String> = Vec::new();

        for (identifier, vector, _metadata) in items {
            let prepared = self.prepare_vector(&vector)?;

            let action = json!({
                "index": {
                    "_index": namespace,
                    "_id": identifier
                }
            });

            let doc = json!({
                "vector": prepared
            });

            bulk_lines.push(serde_json::to_string(&action)?);
            bulk_lines.push(serde_json::to_string(&doc)?);

            if bulk_lines.len() / 2 >= batch_size {
                self.flush_bulk(bulk_lines.clone())?;
                bulk_lines.clear();
            }
        }

        if !bulk_lines.is_empty() {
            self.flush_bulk(bulk_lines)?;
        }

        Ok(())
    }

    fn search(&self, query: &Vector, k: usize, namespace: &str) -> Result<Vec<(String, f64)>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("connect() must be called before search()"))?;

        // Prepare query vector (need mutable self for prepare_vector, so clone the vector)
        let mut temp_driver = ElasticDriver {
            metric: self.metric.clone(),
            base_url: self.base_url.clone(),
            client: self.client.clone(),
            dimension: self.dimension,
        };
        let prepared = temp_driver.prepare_vector(query)?;

        let num_candidates = (k * 4).max(k);

        let search_body = json!({
            "size": k,
            "knn": {
                "field": "vector",
                "query_vector": prepared,
                "k": k,
                "num_candidates": num_candidates,
            },
            "_source": false,
        });

        let url = format!("{}/{}/_search", self.base_url, namespace);
        let response = client
            .get(&url)
            .json(&search_body)
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .context("Failed to search")?;

        response.error_for_status_ref()?;

        let payload: serde_json::Value =
            response.json().context("Failed to parse search response")?;

        let hits_payload = payload
            .get("hits")
            .and_then(|h| h.get("hits"))
            .and_then(|h| h.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid search response format"))?;

        let mut hits: Vec<(String, f64)> = Vec::new();
        for entry in hits_payload {
            if let (Some(id), Some(score)) = (
                entry.get("_id").and_then(|v| v.as_str()),
                entry.get("_score").and_then(|v| v.as_f64()),
            ) {
                hits.push((id.to_string(), score));
            }
        }

        Ok(hits.into_iter().take(k).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_driver_creation() {
        let driver = ElasticDriver::new(None);
        assert_eq!(driver.name(), "Elastic");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_elastic_driver_custom_metric() {
        let driver = ElasticDriver::new(Some("l2"));
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_elastic_driver_base_url_env() {
        std::env::set_var("ELASTIC_URL", "http://localhost:9200/");
        let driver = ElasticDriver::new(None);
        assert_eq!(driver.base_url, "http://localhost:9200");
        std::env::remove_var("ELASTIC_URL");
    }

    #[test]
    fn test_elastic_driver_default_url() {
        std::env::remove_var("ELASTIC_URL");
        let driver = ElasticDriver::new(None);
        assert_eq!(driver.base_url, "");
    }

    #[test]
    fn test_connect_without_url() {
        let mut driver = ElasticDriver::new(None);
        let result = driver.connect();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("ELASTIC_URL not configured"));
    }

    #[test]
    fn test_upsert_without_connect() {
        let mut driver = ElasticDriver::new(Some("cosine"));
        let items = vec![("id1".to_string(), vec![1.0, 2.0, 3.0], None)];
        let result = driver.upsert(items, "test", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_without_connect() {
        let driver = ElasticDriver::new(Some("cosine"));
        let query = vec![1.0, 2.0, 3.0];
        let result = driver.search(&query, 10, "test");
        assert!(result.is_err());
    }
}
