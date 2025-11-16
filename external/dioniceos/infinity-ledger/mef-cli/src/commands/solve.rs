use crate::config::CliConfig;
/// Solve command - SPEC-002 fixpoint calculation
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
struct SolveResponse {
    tic_id: String,
    status: String,
    steps: usize,
}

pub fn execute(config: &CliConfig, snapshot: &str, output_file: Option<PathBuf>) -> Result<()> {
    let api_url = &config.api_url;

    let response = reqwest::blocking::Client::new()
        .post(format!("{}/solve?snapshot_id={}", api_url, snapshot))
        .send()
        .context("Failed to send request to API")?;

    if response.status().is_success() {
        let result: SolveResponse = response.json()?;
        println!("✓ TIC erstellt: {}", result.tic_id);
        println!("  Status: {}", result.status);
        println!("  Schritte: {}", result.steps);

        if let Some(output_path) = output_file {
            let json_str = serde_json::to_string_pretty(&result)?;
            fs::write(&output_path, json_str)
                .with_context(|| format!("Failed to write to {:?}", output_path))?;
            println!("  Gespeichert: {:?}", output_path);
        }
        Ok(())
    } else {
        let error_text = response.text()?;
        eprintln!("Fehler: {}", error_text);
        std::process::exit(1);
    }
}
