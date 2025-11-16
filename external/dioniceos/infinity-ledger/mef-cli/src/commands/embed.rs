use crate::config::CliConfig;
/// Embed command - SPEC-002 Spiral embedding
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    snapshot_id: String,
    phase: f64,
    por: String,
}

pub fn execute(
    config: &CliConfig,
    input_file: PathBuf,
    _seed: &str,
    output_path: &str,
) -> Result<()> {
    // Load data
    let data = if input_file.extension().and_then(|s| s.to_str()) == Some("json") {
        let contents = fs::read_to_string(&input_file)
            .with_context(|| format!("Failed to read file: {:?}", input_file))?;
        serde_json::from_str::<serde_json::Value>(&contents)
            .with_context(|| "Failed to parse JSON file")?
    } else {
        let contents = fs::read_to_string(&input_file)
            .with_context(|| format!("Failed to read file: {:?}", input_file))?;
        serde_json::json!({ "data": contents })
    };

    // API call
    let api_url = &config.api_url;
    let client = reqwest::blocking::Client::new();

    let response = client
        .post(format!("{}/acquisition", api_url))
        .json(&data)
        .send()
        .context("Failed to send request to API")?;

    if response.status().is_success() {
        let result: EmbedResponse = response.json()?;
        println!("✓ Snapshot erstellt: {}", result.snapshot_id);
        println!("  Phase: {}", result.phase);
        println!("  PoR: {}", result.por);
        println!("  Gespeichert: {}", output_path);
        Ok(())
    } else {
        let error_text = response.text()?;
        eprintln!("Fehler: {}", error_text);
        std::process::exit(1);
    }
}
