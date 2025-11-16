/*!
 * Benchmark runner that measures search latency with progress reporting.
 *
 * Migrated from MEF-Core_v1.0/tests/bench/bench_runner.py
 */

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::datasets::{build_spiral_corpus, Record};

/// Timeout configuration for benchmark HTTP requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutSettings {
    pub connect: f64,
    pub read: f64,
    pub bulk_operation: f64,
}

impl Default for TimeoutSettings {
    fn default() -> Self {
        Self {
            connect: 30.0,
            read: 60.0,
            bulk_operation: 120.0,
        }
    }
}

/// Retry strategy configuration for benchmark HTTP requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrySettings {
    pub max_attempts: u32,
    pub backoff_factor: f64,
    pub status_forcelist: Vec<u16>,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            backoff_factor: 2.0,
            status_forcelist: vec![429, 500, 502, 503, 504],
        }
    }
}

/// Batch sizing configuration for bulk ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSettings {
    pub size: usize,
    pub adaptive: bool,
    pub min_size: usize,
    pub max_size: usize,
}

impl Default for BatchSettings {
    fn default() -> Self {
        Self {
            size: 2000,
            adaptive: true,
            min_size: 500,
            max_size: 5000,
        }
    }
}

impl BatchSettings {
    pub fn clamp(&self, value: usize) -> usize {
        value.max(self.min_size).min(self.max_size)
    }
}

/// Loaded benchmark configuration derived from assets/bench/bench_config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub collection: String,
    pub points: usize,
    pub queries: usize,
    pub k: usize,
    pub warmup: usize,
    #[serde(default)]
    pub timeouts: TimeoutSettings,
    #[serde(default)]
    pub retry: RetrySettings,
    #[serde(default)]
    pub batch: BatchSettings,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            collection: "spiral".to_string(),
            points: 100000,
            queries: 200,
            k: 10,
            warmup: 100,
            timeouts: TimeoutSettings::default(),
            retry: RetrySettings::default(),
            batch: BatchSettings::default(),
        }
    }
}

/// Benchmark report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub status: String,
    pub generated_at: String,
    pub collection: String,
    pub points_tested: usize,
    pub indexed_points: usize,
    pub queries: usize,
    pub successful_queries: usize,
    pub failures: usize,
    pub latency_ms: LatencyMetrics,
    pub metric: String,
    pub stage_durations_ms: HashMap<String, f64>,
    pub index_status: serde_json::Value,
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    #[allow(dead_code)]
    base_url: String,
    #[allow(dead_code)]
    client: Client,
    progress_log: PathBuf,
    report_path: PathBuf,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig, base_url: String, assets_dir: &Path) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(config.timeouts.read))
            .connect_timeout(Duration::from_secs_f64(config.timeouts.connect))
            .build()
            .context("Failed to build HTTP client")?;

        let progress_log = assets_dir.join("progress.log");
        let report_path = assets_dir.join("bench_report.json");

        Ok(Self {
            config,
            base_url,
            client,
            progress_log,
            report_path,
        })
    }

    /// Load configuration from file
    pub fn load_config(path: &Path) -> Result<BenchmarkConfig> {
        if !path.exists() {
            return Ok(BenchmarkConfig::default());
        }

        let content = fs::read_to_string(path).context("Failed to read benchmark config")?;
        let config: BenchmarkConfig =
            serde_json::from_str(&content).context("Failed to parse benchmark config")?;

        Ok(config)
    }

    /// Log progress message
    fn log_progress(&self, message: &str) -> Result<()> {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let line = format!("[{}] {}", timestamp, message);

        println!("{}", line);

        if let Some(parent) = self.progress_log.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.progress_log)?;

        use std::io::Write;
        writeln!(file, "{}", line)?;

        Ok(())
    }

    /// Write benchmark report
    fn write_report(&self, report: &BenchmarkReport) -> Result<()> {
        if let Some(parent) = self.report_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(report).context("Failed to serialize report")?;
        fs::write(&self.report_path, json).context("Failed to write report")?;

        Ok(())
    }

    /// Calculate percentiles from samples
    #[allow(dead_code)]
    fn percentiles(values: &[f64]) -> LatencyMetrics {
        if values.is_empty() {
            return LatencyMetrics::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        LatencyMetrics {
            p50: Self::percentile(&sorted, 0.50),
            p95: Self::percentile(&sorted, 0.95),
            p99: Self::percentile(&sorted, 0.99),
        }
    }

    #[allow(dead_code)]
    fn percentile(values: &[f64], percent: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        if values.len() == 1 {
            return values[0];
        }

        let position = percent * (values.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = (lower + 1).min(values.len() - 1);
        let weight = position - lower as f64;

        values[lower] * (1.0 - weight) + values[upper] * weight
    }

    /// Run the complete benchmark
    pub fn run(&mut self) -> Result<BenchmarkReport> {
        self.log_progress(&format!(
            "bench configuration: points={}, k={}, q={}, batch={}, timeout={}s",
            self.config.points,
            self.config.k,
            self.config.queries,
            self.config.batch.size,
            self.config.timeouts.bulk_operation,
        ))?;

        let mut report = BenchmarkReport {
            status: "running".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            collection: self.config.collection.clone(),
            points_tested: self.config.points,
            indexed_points: 0,
            queries: 0,
            successful_queries: 0,
            failures: 0,
            latency_ms: LatencyMetrics::default(),
            metric: "cosine".to_string(),
            stage_durations_ms: HashMap::new(),
            index_status: serde_json::Value::Null,
        };
        self.write_report(&report)?;

        // Generate dataset
        let (ids, vectors) = build_spiral_corpus(self.config.points, 123);
        let _records: Vec<Record> = crate::iter_records(&ids, &vectors).collect();

        // TODO: Implement ingestion, indexing, and query phases
        // This is a minimal stub for now

        report.status = "ok".to_string();
        report.generated_at = chrono::Utc::now().to_rfc3339();
        self.write_report(&report)?;

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.collection, "spiral");
        assert_eq!(config.points, 100000);
        assert_eq!(config.queries, 200);
        assert_eq!(config.k, 10);
        assert_eq!(config.warmup, 100);
    }

    #[test]
    fn test_timeout_settings_default() {
        let settings = TimeoutSettings::default();
        assert_eq!(settings.connect, 30.0);
        assert_eq!(settings.read, 60.0);
        assert_eq!(settings.bulk_operation, 120.0);
    }

    #[test]
    fn test_batch_settings_clamp() {
        let settings = BatchSettings {
            size: 2000,
            adaptive: true,
            min_size: 500,
            max_size: 5000,
        };

        assert_eq!(settings.clamp(100), 500);
        assert_eq!(settings.clamp(3000), 3000);
        assert_eq!(settings.clamp(10000), 5000);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(BenchmarkRunner::percentile(&values, 0.5), 3.0);

        let values = vec![1.0, 2.0];
        assert!((BenchmarkRunner::percentile(&values, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_percentiles() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let metrics = BenchmarkRunner::percentiles(&values);

        assert!((metrics.p50 - 5.5).abs() < 1e-6);
        assert!(metrics.p95 > 9.0);
        assert!(metrics.p99 > 9.5);
    }

    #[test]
    fn test_percentiles_empty() {
        let values: Vec<f64> = vec![];
        let metrics = BenchmarkRunner::percentiles(&values);

        assert_eq!(metrics.p50, 0.0);
        assert_eq!(metrics.p95, 0.0);
        assert_eq!(metrics.p99, 0.0);
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = BenchmarkConfig::default();
        let base_url = "http://localhost:8080".to_string();

        let runner = BenchmarkRunner::new(config, base_url, temp_dir.path());
        assert!(runner.is_ok());
    }

    #[test]
    fn test_load_config_default() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("nonexistent.json");

        let config = BenchmarkRunner::load_config(&config_path).unwrap();
        assert_eq!(config.collection, "spiral");
    }

    #[test]
    fn test_load_config_from_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");

        let custom_config = BenchmarkConfig {
            collection: "test".to_string(),
            points: 1000,
            queries: 50,
            k: 5,
            warmup: 10,
            timeouts: TimeoutSettings::default(),
            retry: RetrySettings::default(),
            batch: BatchSettings::default(),
        };

        let json = serde_json::to_string_pretty(&custom_config).unwrap();
        fs::write(&config_path, json).unwrap();

        let loaded = BenchmarkRunner::load_config(&config_path).unwrap();
        assert_eq!(loaded.collection, "test");
        assert_eq!(loaded.points, 1000);
    }
}
