use crate::config::CliConfig;
/// Ledger command - SPEC-002 ledger operations
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct AppendRequest {
    tic_id: String,
    snapshot_id: String,
}

#[derive(Debug, Deserialize)]
struct AppendResponse {
    index: usize,
    hash: String,
    previous_hash: String,
}

pub fn append(config: &CliConfig, tic: &str, snapshot: &str) -> Result<()> {
    let api_url = &config.api_url;
    let client = reqwest::blocking::Client::new();

    let response = client
        .post(format!("{}/ledger", api_url))
        .query(&[("tic_id", tic), ("snapshot_id", snapshot)])
        .send()
        .context("Failed to send request to API")?;

    if response.status().is_success() {
        let result: AppendResponse = response.json()?;
        println!("✓ Block hinzugefügt: #{}", result.index);
        println!("  Hash: {}", result.hash);
        println!("  Previous: {}", result.previous_hash);
        Ok(())
    } else {
        let error_text = response.text()?;
        eprintln!("Fehler: {}", error_text);
        std::process::exit(1);
    }
}

pub fn verify(_config: &CliConfig) -> Result<()> {
    // Local ledger verification
    println!("Local ledger verification not yet fully implemented");
    eprintln!("Error: Requires full MEF ledger implementation");
    std::process::exit(1);
}
