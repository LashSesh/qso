//! Metatron Telemetry Server
//!
//! HTTP API and web dashboard for Q⊗DASH

use metatron_telemetry::{api, config::Config, state::AppState};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "metatron_telemetry=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::load().unwrap_or_else(|_| {
        tracing::warn!("Could not load config file, using defaults");
        Config::default()
    });

    tracing::info!("Starting Metatron Telemetry Server");
    tracing::info!("Configuration: {:?}", config);

    // Create application state
    let state = AppState::new();

    // Initialize with some demo history
    init_demo_data(&state).await;

    // Create router
    let app = api::create_router(state, &config.static_dir);

    // Bind and serve
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("Dashboard available at http://{}/", addr);
    tracing::info!("API endpoints at http://{}/api/", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Initialize with demo data
async fn init_demo_data(state: &AppState) {
    // Add some historical data points
    for i in 0..50 {
        let progress = i as f64 / 50.0;

        state
            .update_status(
                0.75 + progress * 0.15,  // psi: 0.75 -> 0.90
                0.80 + progress * 0.10,  // rho: 0.80 -> 0.90
                0.70 + progress * 0.15,  // omega: 0.70 -> 0.85
                "VQE".to_string(),
            )
            .await;
    }

    tracing::info!("Initialized with demo historical data");
}
