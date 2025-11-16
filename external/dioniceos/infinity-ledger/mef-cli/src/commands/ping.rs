use crate::config::CliConfig;
/// Ping command - test API server connectivity
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PingResponse {
    version: String,
    seed: String,
    timestamp: String,
}

pub fn execute(config: &CliConfig) -> Result<()> {
    let api_url = &config.api_url;

    match reqwest::blocking::get(format!("{}/ping", api_url)) {
        Ok(response) if response.status().is_success() => {
            let data: PingResponse = response.json()?;
            println!("✓ API server is operational");
            println!("  Version: {}", data.version);
            println!("  Seed: {}", data.seed);
            println!("  Timestamp: {}", data.timestamp);
            Ok(())
        }
        Ok(response) => {
            eprintln!("✗ API server error: {}", response.status());
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("✗ Cannot connect to API server at {}", api_url);
            std::process::exit(1);
        }
    }
}
