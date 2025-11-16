//! MEF-Core Ledger Module
//!
//! Hash-chained immutable ledger for TICs with deterministic hashing.
//!
//! Migrated from: MEF-Core_v1.0/src/ledger/

pub mod mef_block;

pub use mef_block::{
    BlockSummary, ChainStatistics, CompactTic, LedgerIndex, LedgerMetadata, MEFLedger, MefBlock,
    TimeRange,
};
