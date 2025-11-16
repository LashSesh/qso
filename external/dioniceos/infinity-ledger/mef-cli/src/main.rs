/// MEF-Core Command Line Interface
/// CLI for interacting with the MEF-Core system.
///
/// Migrated from: MEF-Core_v1.0/src/cli/mef.py
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;
mod config;

use config::CliConfig;

/// Default configuration constants
const DEFAULT_CONFIG: &str = "config.yaml";
const DEFAULT_API_URL: &str = "http://localhost:8000";

#[derive(Parser, Debug)]
#[command(name = "mef")]
#[command(about = "MEF-Core Command Line Interface", long_about = None)]
#[command(version)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, env = "MEF_CONFIG", default_value = DEFAULT_CONFIG)]
    config: PathBuf,

    /// API server URL
    #[arg(long, env = "MEF_API_URL", default_value = DEFAULT_API_URL)]
    api_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Ingest a file into the MEF-Core system
    Ingest {
        /// File path to ingest
        file_path: PathBuf,

        /// Data type
        #[arg(short = 't', long, value_parser = ["text", "json", "numeric", "binary", "raw"], default_value = "raw")]
        data_type: String,

        /// Deterministic seed
        #[arg(short, long, default_value = "MEF_SEED_42")]
        seed: String,

        /// Use local processing (instead of API)
        #[arg(long)]
        local: bool,
    },

    /// Process a snapshot through Solve-Coagula to create TIC
    Process {
        /// Snapshot ID to process
        snapshot_id: String,

        /// Auto-commit to ledger
        #[arg(long)]
        commit: bool,

        /// Use local processing (instead of API)
        #[arg(long)]
        local: bool,
    },

    /// Audit the MEF ledger integrity
    Audit {
        /// Start block index
        #[arg(short, long, default_value = "0")]
        start: usize,

        /// Export audit trail
        #[arg(short, long)]
        export: bool,

        /// Use local processing (instead of API)
        #[arg(long)]
        local: bool,
    },

    /// Validate a snapshot using Proof-of-Resonance
    Validate {
        /// Snapshot ID to validate
        snapshot_id: String,

        /// Use local processing (instead of API)
        #[arg(long)]
        local: bool,
    },

    /// Export system data
    Export {
        /// Export format
        #[arg(short, long, value_parser = ["json", "audit"], default_value = "json")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// SPEC-002: Embed command for Spiral embedding
    Embed {
        /// Input file
        #[arg(long = "in")]
        input_file: PathBuf,

        /// Deterministic seed
        #[arg(short, long, default_value = "MEF_SEED_42")]
        seed: String,

        /// Output path
        #[arg(long = "out", default_value = "store")]
        output_path: String,
    },

    /// SPEC-002: Solve command for fixpoint calculation
    Solve {
        /// Snapshot ID
        #[arg(long)]
        snapshot: String,

        /// Output TIC file
        #[arg(long = "out")]
        output_file: Option<PathBuf>,
    },

    /// SPEC-002: Ledger commands
    Ledger {
        #[command(subcommand)]
        subcommand: LedgerCommands,
    },

    /// Ping the API server
    Ping,
}

#[derive(Subcommand, Debug)]
enum LedgerCommands {
    /// Append a block to the ledger
    Append {
        /// TIC ID
        #[arg(long)]
        tic: String,

        /// Snapshot ID
        #[arg(long)]
        snapshot: String,
    },

    /// Verify ledger integrity
    Verify,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let config =
        CliConfig::load(&cli.config, &cli.api_url).context("Failed to load configuration")?;

    // Execute command
    match cli.command {
        Commands::Ingest {
            file_path,
            data_type,
            seed,
            local,
        } => commands::ingest::execute(&config, file_path, &data_type, &seed, local),

        Commands::Process {
            snapshot_id,
            commit,
            local,
        } => commands::process::execute(&config, &snapshot_id, commit, local),

        Commands::Audit {
            start,
            export,
            local,
        } => commands::audit::execute(&config, start, export, local),

        Commands::Validate { snapshot_id, local } => {
            commands::validate::execute(&config, &snapshot_id, local)
        }

        Commands::Export { format, output } => commands::export::execute(&config, &format, output),

        Commands::Embed {
            input_file,
            seed,
            output_path,
        } => commands::embed::execute(&config, input_file, &seed, &output_path),

        Commands::Solve {
            snapshot,
            output_file,
        } => commands::solve::execute(&config, &snapshot, output_file),

        Commands::Ledger { subcommand } => match subcommand {
            LedgerCommands::Append { tic, snapshot } => {
                commands::ledger::append(&config, &tic, &snapshot)
            }
            LedgerCommands::Verify => commands::ledger::verify(&config),
        },

        Commands::Ping => commands::ping::execute(&config),
    }
}
