//! API routes configuration

use super::handlers;
use crate::state::AppState;
use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};

/// Create the main application router
pub fn create_router(state: AppState, static_dir: &str) -> Router {
    // API routes
    let api_routes = Router::new()
        .route("/status", get(handlers::get_status))
        .route("/jobs", get(handlers::get_jobs))
        .route("/jobs/:id", get(handlers::get_job))
        .route("/history", get(handlers::get_history))
        .route("/control/start_calibration", post(handlers::start_calibration))
        .route("/health", get(handlers::health_check))
        .with_state(state);

    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Main router
    Router::new()
        .nest("/api", api_routes)
        .nest_service("/", ServeDir::new(static_dir))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}
