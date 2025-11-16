use crate::config::CliConfig;
/// Ingest command - ingest files into MEF-Core system
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct IngestRequest {
    data: serde_json::Value,
    data_type: String,
    seed: String,
}

#[derive(Debug, Deserialize)]
struct IngestResponse {
    snapshot_id: String,
    phase: f64,
    por: String,
}

pub fn execute(
    config: &CliConfig,
    file_path: PathBuf,
    data_type: &str,
    seed: &str,
    local: bool,
) -> Result<()> {
    // Read file
    let data = if data_type == "binary" {
        let bytes = fs::read(&file_path)
            .with_context(|| format!("Failed to read file: {:?}", file_path))?;
        serde_json::Value::String(base64::encode(&bytes))
    } else {
        let contents = fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read file: {:?}", file_path))?;

        if data_type == "json" {
            serde_json::from_str(&contents).with_context(|| "Failed to parse JSON file")?
        } else {
            serde_json::Value::String(contents)
        }
    };

    if local {
        // Local processing
        println!("Local processing not yet fully implemented");
        println!("File: {:?}", file_path);
        println!("Type: {}", data_type);
        println!("Seed: {}", seed);
        eprintln!("Error: Local processing requires full MEF pipeline implementation");
        std::process::exit(1);
    } else {
        // Remote API processing
        let api_url = &config.api_url;
        let client = reqwest::blocking::Client::new();

        let request = IngestRequest {
            data,
            data_type: data_type.to_string(),
            seed: seed.to_string(),
        };

        let response = client
            .post(format!("{}/ingest", api_url))
            .json(&request)
            .send()
            .context("Failed to send request to API")?;

        if response.status().is_success() {
            let result: IngestResponse = response.json()?;
            println!("✓ Snapshot created: {}", result.snapshot_id);
            println!("  Phase: {:.4}", result.phase);
            println!("  PoR: {}", result.por);
            Ok(())
        } else {
            let error_text = response.text()?;
            eprintln!("Error: {}", error_text);
            std::process::exit(1);
        }
    }
}

// Base64 encoding helper (simple implementation)
mod base64 {
    pub fn encode(data: &[u8]) -> String {
        use std::fmt::Write;
        let mut result = String::new();
        for chunk in data.chunks(3) {
            let mut buf = [0u8; 3];
            buf[..chunk.len()].copy_from_slice(chunk);
            let b64_chunk = [
                (buf[0] >> 2) & 0x3F,
                ((buf[0] & 0x03) << 4) | ((buf[1] >> 4) & 0x0F),
                ((buf[1] & 0x0F) << 2) | ((buf[2] >> 6) & 0x03),
                buf[2] & 0x3F,
            ];
            for &b in &b64_chunk[..std::cmp::min(chunk.len() + 1, 4)] {
                write!(&mut result, "{}", BASE64_CHARS[b as usize]).unwrap();
            }
        }
        // Pad with '='
        while !result.len().is_multiple_of(4) {
            result.push('=');
        }
        result
    }

    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}
