/// MEF-Core API Server - Main Entry Point
/// Migrated from: MEF-Core_v1.0/src/api/server.py
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use mef_api::{routes, ApiConfig, AppState};
use mef_knowledge::{ExtensionConfig, ExtensionPipeline};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mef_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = ApiConfig::load()?;
    tracing::info!("Configuration loaded: {:?}", config);

    // Initialize application state
    let state = AppState::new(config.clone()).await?;
    tracing::info!("Application state initialized");

    // Build router
    let mut app = Router::new()
        .merge(routes::health::router())
        .merge(routes::ingest::router())
        .merge(routes::process::router())
        .merge(routes::ledger::router())
        .merge(routes::vector::router())
        .merge(routes::coupling::router())
        .merge(routes::tic::router())
        .merge(routes::index::router())
        .merge(routes::system::router())
        .merge(routes::commit::router())
        .merge(routes::zk::router())
        .merge(routes::domain::router())
        .merge(routes::metatron::router())
        .merge(routes::merkaba::router())
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    // Optionally load and mount extension routes
    if let Ok(ext_config) = ExtensionConfig::load_from_env() {
        let pipeline = ExtensionPipeline::new(ext_config.mef.extension.clone());
        if pipeline.is_enabled() {
            tracing::info!("Extension enabled, mounting extension routes");
            let ext_state = routes::extension::ExtensionState {
                pipeline: Arc::new(tokio::sync::Mutex::new(pipeline)),
            };
            app = app.merge(routes::extension::router(ext_state));
        } else {
            tracing::info!("Extension configuration loaded but all features disabled");
        }
    } else {
        tracing::info!("Extension configuration not found or invalid, skipping extension routes");
    }

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
