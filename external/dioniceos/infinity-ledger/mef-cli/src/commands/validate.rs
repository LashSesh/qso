use crate::config::CliConfig;
/// Validate command - validate snapshots using Proof-of-Resonance
use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ValidateResponse {
    snapshot_id: String,
    overall_valid: bool,
    resonance: ResonanceMetrics,
    stability: StabilityMetrics,
}

#[derive(Debug, Deserialize)]
struct ResonanceMetrics {
    fft_resonance: f64,
    claimed_resonance: f64,
    deviation: f64,
    spectral_gap: f64,
}

#[derive(Debug, Deserialize)]
struct StabilityMetrics {
    computed: f64,
    valid: bool,
}

pub fn execute(config: &CliConfig, snapshot_id: &str, local: bool) -> Result<()> {
    if local {
        println!("Local validation not yet fully implemented");
        println!("Snapshot: {}", snapshot_id);
        eprintln!("Error: Local validation requires full MEF validation implementation");
        std::process::exit(1);
    } else {
        // Remote API
        let api_url = &config.api_url;

        let response = reqwest::blocking::Client::new()
            .post(format!("{}/validate/snapshot/{}", api_url, snapshot_id))
            .send()
            .context("Failed to send request to API")?;

        if response.status().is_success() {
            let report: ValidateResponse = response.json()?;
            println!("Snapshot: {}", report.snapshot_id);
            println!("Valid: {}", if report.overall_valid { "✓" } else { "✗" });
            println!("Resonance:");
            println!("  FFT: {:.4}", report.resonance.fft_resonance);
            println!("  Claimed: {:.4}", report.resonance.claimed_resonance);
            println!("  Deviation: {:.4}", report.resonance.deviation);
            println!("  Spectral gap: {:.4}", report.resonance.spectral_gap);
            println!("Stability:");
            println!("  Computed: {:.4}", report.stability.computed);
            println!(
                "  Valid: {}",
                if report.stability.valid { "✓" } else { "✗" }
            );
            Ok(())
        } else {
            let error_text = response.text()?;
            eprintln!("Error: {}", error_text);
            std::process::exit(1);
        }
    }
}
