/// Processing endpoints - Solve-Coagula and TIC creation
use axum::{
    extract::{Path, State},
    routing::post,
    Json, Router,
};
use chrono::Utc;
use ndarray::Array1;
use serde_json::json;

use crate::{error::ApiError, models::*, AppState, Result};
use mef_solvecoagula::{SolveCoagula, SolveCoagulaConfig};
use mef_spiral::SpiralSnapshot;
use mef_tic::{TICConfig, TICCrystallizer};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/process", post(process))
        .route("/solve", post(solve))
        .route("/validate/snapshot/:id", post(validate))
}

/// Process a snapshot through Solve-Coagula to create a TIC
async fn process(
    State(state): State<AppState>,
    Json(request): Json<ProcessRequest>,
) -> Result<Json<ProcessResponse>> {
    // Create spiral snapshot handler to load
    let spiral = SpiralSnapshot::new(
        state.spiral_config.as_ref().clone(),
        state.store_path.as_ref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create spiral snapshot: {}", e)))?;

    // Load snapshot
    let snapshot = spiral
        .load_snapshot(&request.snapshot_id)
        .map_err(|e| ApiError::NotFound(format!("Failed to load snapshot: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Snapshot {} not found", request.snapshot_id)))?;

    // Create Solve-Coagula operator
    let sc_config = SolveCoagulaConfig::default();
    let solver = SolveCoagula::new(sc_config)
        .map_err(|e| ApiError::Internal(format!("Failed to create SolveCoagula: {}", e)))?;

    // Process the snapshot - convert coordinates to Array1
    let coords_array = Array1::from_vec(snapshot.coordinates.clone());
    let (fixpoint, info) = solver
        .iterate_to_fixpoint(&coords_array, true)
        .map_err(|e| ApiError::Processing(format!("Solve-Coagula failed: {}", e)))?;

    // Create TIC from result
    let tic_config = TICConfig::default();
    let crystallizer = TICCrystallizer::new(tic_config, state.store_path.as_ref())
        .map_err(|e| ApiError::Internal(format!("Failed to create TICCrystallizer: {}", e)))?;

    // Prepare convergence info and snapshot data as JSON
    let convergence_json = json!({
        "converged": info.converged,
        "iterations": info.iterations,
        "final_delta": info.final_delta,
    });

    let snapshot_json = json!({
        "id": snapshot.id,
        "phase": snapshot.phase,
        "coordinates": snapshot.coordinates,
    });

    let tic = crystallizer
        .create_tic(
            &fixpoint,
            &snapshot.id,
            &snapshot.seed,
            &convergence_json,
            &snapshot_json,
        )
        .map_err(|e| ApiError::Processing(format!("TIC creation failed: {}", e)))?;

    let tic_id = tic.tic_id.clone();

    // If commit is requested, append to ledger
    if request.commit.unwrap_or(false) {
        // TODO: Implement ledger append with proper TIC/snapshot data
        tracing::info!("Commit requested for TIC: {}", tic_id);
    }

    Ok(Json(ProcessResponse {
        tic_id,
        converged: info.converged,
        iterations: info.iterations,
        final_eigenvalue: fixpoint[0], // Use first component as representative
        timestamp: Utc::now().to_rfc3339(),
    }))
}

/// Solve endpoint - alternative processing method
async fn solve(
    State(state): State<AppState>,
    Json(request): Json<SolveRequest>,
) -> Result<Json<SolveResponse>> {
    // Create spiral snapshot handler to load
    let spiral = SpiralSnapshot::new(
        state.spiral_config.as_ref().clone(),
        state.store_path.as_ref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create spiral snapshot: {}", e)))?;

    // Load snapshot
    let snapshot = spiral
        .load_snapshot(&request.snapshot_id)
        .map_err(|e| ApiError::NotFound(format!("Failed to load snapshot: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Snapshot {} not found", request.snapshot_id)))?;

    // Create Solve-Coagula operator
    let sc_config = SolveCoagulaConfig::default();
    let solver = SolveCoagula::new(sc_config)
        .map_err(|e| ApiError::Internal(format!("Failed to create SolveCoagula: {}", e)))?;

    // Process the snapshot
    let coords_array = Array1::from_vec(snapshot.coordinates.clone());
    let (fixpoint, info) = solver
        .iterate_to_fixpoint(&coords_array, true)
        .map_err(|e| ApiError::Processing(format!("Solve-Coagula failed: {}", e)))?;

    // Create TIC
    let tic_config = TICConfig::default();
    let crystallizer = TICCrystallizer::new(tic_config, state.store_path.as_ref())
        .map_err(|e| ApiError::Internal(format!("Failed to create TICCrystallizer: {}", e)))?;

    let convergence_json = json!({
        "converged": info.converged,
        "iterations": info.iterations,
        "final_delta": info.final_delta,
    });

    let snapshot_json = json!({
        "id": snapshot.id,
        "phase": snapshot.phase,
        "coordinates": snapshot.coordinates,
    });

    let tic = crystallizer
        .create_tic(
            &fixpoint,
            &snapshot.id,
            &snapshot.seed,
            &convergence_json,
            &snapshot_json,
        )
        .map_err(|e| ApiError::Processing(format!("TIC creation failed: {}", e)))?;

    Ok(Json(SolveResponse {
        tic_id: tic.tic_id.clone(),
        status: if info.converged {
            "converged"
        } else {
            "max_iterations"
        }
        .to_string(),
        steps: info.iterations,
    }))
}

/// Validate a snapshot using Proof-of-Resonance
async fn validate(
    State(state): State<AppState>,
    Path(snapshot_id): Path<String>,
) -> Result<Json<ValidateResponse>> {
    // Create spiral snapshot handler to load
    let spiral = SpiralSnapshot::new(
        state.spiral_config.as_ref().clone(),
        state.store_path.as_ref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create spiral snapshot: {}", e)))?;

    // Load snapshot
    let snapshot = spiral
        .load_snapshot(&snapshot_id)
        .map_err(|e| ApiError::NotFound(format!("Failed to load snapshot: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Snapshot {} not found", snapshot_id)))?;

    // Parse PoR value from metrics
    let por: f64 = snapshot.metrics.por.parse().unwrap_or(0.0);
    let valid = por > 0.5; // Threshold for validity

    Ok(Json(ValidateResponse {
        snapshot_id,
        overall_valid: valid,
        resonance: ResonanceMetrics {
            fft_resonance: snapshot.metrics.resonance,
            spectral_gap: 0.0, // TODO: Implement
            phase_coherence: snapshot.phase,
        },
        stability: StabilityMetrics {
            convergence_rate: snapshot.metrics.stability,
            entropy: 0.0, // TODO: Implement
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_solve_basic() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        // Create spiral snapshot handler
        let spiral = SpiralSnapshot::new(
            state.spiral_config.as_ref().clone(),
            state.store_path.as_ref(),
        )
        .unwrap();

        // Create and store a test snapshot
        let data = serde_json::json!({"test": "data"});
        let snapshot = spiral.create_snapshot(&data, "test_seed", None).unwrap();

        let _snapshot_path = spiral.save_snapshot(&snapshot).unwrap();
        let snapshot_id = snapshot.id.clone();

        let request = SolveRequest { snapshot_id };

        let result = solve(State(state), Json(request)).await;
        assert!(result.is_ok());
    }
}
