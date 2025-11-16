use crate::config::CliConfig;
/// Audit command - audit ledger integrity
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct AuditRequest {
    start_index: usize,
    export: bool,
}

#[derive(Debug, Deserialize)]
struct AuditResponse {
    chain_valid: bool,
    statistics: ChainStatistics,
    export_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChainStatistics {
    total_blocks: usize,
    total_size_mb: f64,
}

pub fn execute(config: &CliConfig, start: usize, export: bool, local: bool) -> Result<()> {
    if local {
        println!("Local audit not yet fully implemented");
        println!("Start: {}", start);
        println!("Export: {}", export);
        eprintln!("Error: Local audit requires full MEF ledger implementation");
        std::process::exit(1);
    } else {
        // Remote API
        let api_url = &config.api_url;
        let client = reqwest::blocking::Client::new();

        let request = AuditRequest {
            start_index: start,
            export,
        };

        let response = client
            .post(format!("{}/audit", api_url))
            .json(&request)
            .send()
            .context("Failed to send request to API")?;

        if response.status().is_success() {
            let result: AuditResponse = response.json()?;
            println!(
                "Chain valid: {}",
                if result.chain_valid { "✓" } else { "✗" }
            );
            println!("Total blocks: {}", result.statistics.total_blocks);
            println!("Chain size: {:.2} MB", result.statistics.total_size_mb);

            if export {
                if let Some(path) = result.export_path {
                    println!("✓ Audit trail exported to: {}", path);
                }
            }
            Ok(())
        } else {
            let error_text = response.text()?;
            eprintln!("Error: {}", error_text);
            std::process::exit(1);
        }
    }
}
