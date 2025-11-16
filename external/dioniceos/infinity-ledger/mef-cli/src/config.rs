/// Configuration management for CLI
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub api_url: String,
    pub seed: String,
    pub spiral: SpiralConfig,
    pub solvecoagula: SolveCoagulaConfig,
    pub store_dir: PathBuf,
    pub ledger_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiralConfig {
    pub r: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub k: i32,
    pub step: f64,
}

impl Default for SpiralConfig {
    fn default() -> Self {
        Self {
            r: 1.0,
            a: 0.05,
            b: 0.2,
            c: 0.2,
            k: 2,
            step: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveCoagulaConfig {
    pub lambda: f64,
    pub eps: f64,
    pub max_iter: usize,
}

impl Default for SolveCoagulaConfig {
    fn default() -> Self {
        Self {
            lambda: 0.8,
            eps: 1e-6,
            max_iter: 1000,
        }
    }
}

impl CliConfig {
    /// Load configuration from file or use defaults
    pub fn load(config_path: &Path, api_url: &str) -> Result<Self> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let default_store = home.join("mef").join("store");
        let default_ledger = home.join("mef").join("ledger");

        if config_path.exists() {
            // Load from file
            let contents = fs::read_to_string(config_path)
                .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

            let config_data: HashMap<String, serde_yaml::Value> =
                serde_yaml::from_str(&contents)
                    .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;

            let spiral = if let Some(spiral_val) = config_data.get("spiral") {
                serde_yaml::from_value(spiral_val.clone()).unwrap_or_default()
            } else {
                SpiralConfig::default()
            };

            let solvecoagula = if let Some(sc_val) = config_data.get("solvecoagula") {
                serde_yaml::from_value(sc_val.clone()).unwrap_or_default()
            } else {
                SolveCoagulaConfig::default()
            };

            let seed = config_data
                .get("seed")
                .and_then(|v| v.as_str())
                .unwrap_or("MEF_SEED_42")
                .to_string();

            Ok(Self {
                api_url: api_url.to_string(),
                seed,
                spiral,
                solvecoagula,
                store_dir: std::env::var("MEF_STORE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or(default_store),
                ledger_dir: std::env::var("MEF_LEDGER_DIR")
                    .map(PathBuf::from)
                    .unwrap_or(default_ledger),
            })
        } else {
            eprintln!(
                "Warning: Config file {:?} not found, using defaults",
                config_path
            );

            Ok(Self {
                api_url: api_url.to_string(),
                seed: "MEF_SEED_42".to_string(),
                spiral: SpiralConfig::default(),
                solvecoagula: SolveCoagulaConfig::default(),
                store_dir: std::env::var("MEF_STORE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or(default_store),
                ledger_dir: std::env::var("MEF_LEDGER_DIR")
                    .map(PathBuf::from)
                    .unwrap_or(default_ledger),
            })
        }
    }
}
