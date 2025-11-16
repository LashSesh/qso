/// Configuration management for API server
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Server port
    pub port: u16,

    /// API token for authentication
    pub api_token: String,

    /// Whether authentication is required
    pub auth_required: bool,

    /// Store directory path
    pub store_path: PathBuf,

    /// Ledger directory path
    pub ledger_path: PathBuf,

    /// Logs directory path
    pub logs_path: PathBuf,

    /// Default seed value
    pub seed: String,

    /// Epsilon for PI calculations
    pub eps_pi: f64,

    /// Quality collection name
    pub quality_collection: String,

    /// Quality metric type
    pub quality_metric: String,

    /// Metrics window size
    pub metrics_window: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let mef_home = home.join("mef");

        Self {
            port: 8000,
            api_token: "infinity-ledger-token".to_string(),
            auth_required: true,
            store_path: mef_home.join("store"),
            ledger_path: mef_home.join("ledger"),
            logs_path: mef_home.join("logs"),
            seed: "MEF_SEED_42".to_string(),
            eps_pi: 0.001,
            quality_collection: "spiral".to_string(),
            quality_metric: "cosine".to_string(),
            metrics_window: 10000,
        }
    }
}

impl ApiConfig {
    /// Load configuration from environment variables and config file
    pub fn load() -> Result<Self> {
        let mut config = Self::default();

        // Override from environment variables
        if let Ok(port) = env::var("MEF_API_PORT") {
            config.port = port.parse().context("Invalid MEF_API_PORT")?;
        }

        if let Ok(token) = env::var("MEF_API_TOKEN") {
            config.api_token = token;
        }

        if let Ok(auth) = env::var("AUTH_TOKEN_REQUIRED") {
            config.auth_required = Self::parse_bool(&auth);
        }

        if let Ok(store) = env::var("MEF_STORE_DIR") {
            config.store_path = PathBuf::from(store);
        }

        if let Ok(ledger) = env::var("MEF_LEDGER_DIR") {
            config.ledger_path = PathBuf::from(ledger);
        }

        if let Ok(logs) = env::var("MEF_LOGS_DIR") {
            config.logs_path = PathBuf::from(logs);
        }

        if let Ok(seed) = env::var("MEF_SEED") {
            config.seed = seed;
        }

        if let Ok(eps) = env::var("MEF_EPS_PI") {
            config.eps_pi = eps.parse().context("Invalid MEF_EPS_PI")?;
        }

        if let Ok(collection) = env::var("QUALITY_COLLECTION") {
            config.quality_collection = collection;
        }

        if let Ok(metric) = env::var("QUALITY_METRIC") {
            config.quality_metric = metric;
        }

        if let Ok(window) = env::var("METRICS_WINDOW") {
            config.metrics_window = window.parse().context("Invalid METRICS_WINDOW")?;
        }

        // Try to load from config file if it exists
        let config_path = Path::new("config.yaml");
        if config_path.exists() {
            Self::load_from_file(&mut config, config_path)?;
        }

        // Ensure directories exist
        for path in [&config.store_path, &config.ledger_path, &config.logs_path] {
            fs::create_dir_all(path)
                .with_context(|| format!("Failed to create directory: {:?}", path))?;
        }

        Ok(config)
    }

    fn load_from_file(config: &mut ApiConfig, path: &Path) -> Result<()> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;

        let yaml_config: serde_yaml::Value = serde_yaml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {:?}", path))?;

        // Extract values from YAML if present
        if let Some(yaml_map) = yaml_config.as_mapping() {
            if let Some(seed) = yaml_map.get("seed").and_then(|v| v.as_str()) {
                config.seed = seed.to_string();
            }

            if let Some(port) = yaml_map.get("port").and_then(|v| v.as_u64()) {
                config.port = port as u16;
            }
        }

        Ok(())
    }

    fn parse_bool(value: &str) -> bool {
        matches!(
            value.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }
}
