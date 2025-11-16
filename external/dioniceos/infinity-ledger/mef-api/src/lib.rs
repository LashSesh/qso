/// MEF-Core API Server
/// Migrated from: MEF-Core_v1.0/src/api/server.py
///
/// FastAPI → Axum migration with identical API contract
pub mod config;
pub mod error;
pub mod models;
pub mod routes;
pub mod state;

pub use config::ApiConfig;
pub use error::{ApiError, Result};
pub use state::AppState;
