use crate::config::CliConfig;
/// Process command - process snapshots through Solve-Coagula
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct ProcessRequest {
    snapshot_id: String,
    auto_commit: bool,
}

#[derive(Debug, Deserialize)]
struct ProcessResponse {
    tic_id: String,
    fixpoint_converged: bool,
    iterations: usize,
    should_commit: bool,
    committed: bool,
    block_index: Option<usize>,
}

pub fn execute(config: &CliConfig, snapshot_id: &str, commit: bool, local: bool) -> Result<()> {
    if local {
        println!("Local processing not yet fully implemented");
        println!("Snapshot: {}", snapshot_id);
        println!("Commit: {}", commit);
        eprintln!("Error: Local processing requires full MEF pipeline implementation");
        std::process::exit(1);
    } else {
        // Remote API processing
        let api_url = &config.api_url;
        let client = reqwest::blocking::Client::new();

        let request = ProcessRequest {
            snapshot_id: snapshot_id.to_string(),
            auto_commit: commit,
        };

        let response = client
            .post(format!("{}/process", api_url))
            .json(&request)
            .send()
            .context("Failed to send request to API")?;

        if response.status().is_success() {
            let result: ProcessResponse = response.json()?;
            println!("✓ TIC created: {}", result.tic_id);
            println!("  Converged: {}", result.fixpoint_converged);
            println!("  Iterations: {}", result.iterations);
            println!("  Should commit: {}", result.should_commit);
            if result.committed {
                if let Some(idx) = result.block_index {
                    println!("✓ Committed to ledger: Block #{}", idx);
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
