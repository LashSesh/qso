use crate::config::CliConfig;
/// Export command - export system data
use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub fn execute(config: &CliConfig, format: &str, output: Option<PathBuf>) -> Result<()> {
    let api_url = &config.api_url;

    let response = reqwest::blocking::get(format!("{}/export/{}", api_url, format))
        .context("Failed to send request to API")?;

    if response.status().is_success() {
        if let Some(output_path) = output {
            if format == "json" {
                let json_data: serde_json::Value = response.json()?;
                let json_str = serde_json::to_string_pretty(&json_data)?;
                fs::write(&output_path, json_str)
                    .with_context(|| format!("Failed to write to {:?}", output_path))?;
            } else {
                let bytes = response.bytes()?;
                fs::write(&output_path, bytes)
                    .with_context(|| format!("Failed to write to {:?}", output_path))?;
            }
            println!("✓ Exported to: {:?}", output_path);
        } else if format == "json" {
            let json_data: serde_json::Value = response.json()?;
            println!("{}", serde_json::to_string_pretty(&json_data)?);
        } else {
            let text = response.text()?;
            println!("{}", text);
        }
        Ok(())
    } else {
        let error_text = response.text()?;
        eprintln!("Error: {}", error_text);
        std::process::exit(1);
    }
}
