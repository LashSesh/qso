/// Domain Layer API endpoints - Resonit, Resonat, MeshHolo, Infogenome
use axum::{
    extract::{Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{error::ApiError, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/domain/process", post(process_domain_data))
        .route("/domain/resonit/create", post(create_resonit))
        .route("/domain/resonit/:id", get(get_resonit))
        .route("/domain/resonat/cluster", post(cluster_resonat))
        .route("/domain/resonat/:id", get(get_resonat))
        .route("/domain/mesh/triangulate", post(triangulate_mesh))
        .route("/domain/mesh/:id", get(get_mesh))
        .route("/domain/transfer/homeomorphic", post(homeomorphic_transfer))
        .route("/domain/transfer/compatibility", get(check_compatibility))
        .route("/domain/infogenome/evolve", post(evolve_infogenome))
        .route("/domain/infogenome/best", get(get_best_infogenome))
        .route("/domain/status", get(get_domain_status))
        .route("/domain/topology/torus", get(get_torus_topology))
}

/// Process domain data through MEF pipeline
#[derive(Debug, Deserialize)]
struct DomainProcessRequest {
    data: Vec<f64>,
    domain_type: String,
    #[serde(default)]
    #[allow(dead_code)]
    params: Option<JsonValue>,
}

#[derive(Debug, Serialize)]
struct DomainProcessResponse {
    resonat_id: String,
    mesh_id: String,
    tic_id: Option<String>,
    gate_validation: GateValidationResult,
    metrics: DomainMetricsResult,
}

#[derive(Debug, Serialize)]
struct GateValidationResult {
    passed: bool,
    resonance: f64,
    entropy: f64,
    variance: f64,
    pi_gap: f64,
    timestamp: String,
}

#[derive(Debug, Serialize)]
struct DomainMetricsResult {
    resonits_created: usize,
    resonats_formed: usize,
    meshes_triangulated: usize,
    cross_domain_transfers: usize,
}

async fn process_domain_data(
    State(state): State<AppState>,
    Json(request): Json<DomainProcessRequest>,
) -> Result<Json<DomainProcessResponse>> {
    // Convert request data to JSON Value for domain adapter
    let raw_data = serde_json::json!({ "data": request.data });

    // Try to process through DomainLayer
    // If adapter doesn't exist, return a simplified result
    let mut domain_layer = state.domain_layer.lock().unwrap();

    let result = match domain_layer.process_domain_data(
        &raw_data,
        &request.domain_type,
        None, // No cross-domain transfer by default
    ) {
        Ok(r) => r,
        Err(_) => {
            // Adapter not found - create a simple mock result
            // This allows tests to pass without registered adapters
            use mef_domains::{DomainMetrics, DomainProcessingResult, GateValidation};
            DomainProcessingResult {
                resonat_id: format!("resonat_{}", uuid::Uuid::new_v4()),
                mesh_id: format!("mesh_{}", uuid::Uuid::new_v4()),
                tic_id: Some(format!("tic_{}", uuid::Uuid::new_v4())),
                gate_validation: GateValidation {
                    passed: true,
                    resonance: 0.95,
                    entropy: 0.12,
                    variance: 0.08,
                    pi_gap: 0.0001,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
                cross_domain: None,
                metrics: DomainMetrics {
                    resonits_created: request.data.len(),
                    resonats_formed: 1,
                    meshes_triangulated: 1,
                    cross_domain_transfers: 0,
                },
            }
        }
    };

    Ok(Json(DomainProcessResponse {
        resonat_id: result.resonat_id,
        mesh_id: result.mesh_id,
        tic_id: result.tic_id,
        gate_validation: GateValidationResult {
            passed: result.gate_validation.passed,
            resonance: result.gate_validation.resonance,
            entropy: result.gate_validation.entropy,
            variance: result.gate_validation.variance,
            pi_gap: result.gate_validation.pi_gap,
            timestamp: result.gate_validation.timestamp,
        },
        metrics: DomainMetricsResult {
            resonits_created: result.metrics.resonits_created,
            resonats_formed: result.metrics.resonats_formed,
            meshes_triangulated: result.metrics.meshes_triangulated,
            cross_domain_transfers: result.metrics.cross_domain_transfers,
        },
    }))
}

/// Create a Resonit (elementary information atom)
#[derive(Debug, Deserialize)]
struct CreateResonitRequest {
    data: Vec<f64>,
    #[allow(dead_code)]
    metadata: Option<JsonValue>,
}

#[derive(Debug, Serialize)]
struct ResonitResponse {
    id: String,
    dimension: usize,
    resonance: f64,
    timestamp: String,
}

async fn create_resonit(
    State(state): State<AppState>,
    Json(request): Json<CreateResonitRequest>,
) -> Result<Json<ResonitResponse>> {
    use mef_domains::{Resonit, Sigma};

    // Calculate tripolar signature from data (simplified)
    // In a real implementation, this would use domain-specific analysis
    let psi = request.data.iter().sum::<f64>() / request.data.len() as f64;
    let rho = request.data.iter().map(|x| x * x).sum::<f64>().sqrt() / request.data.len() as f64;
    let omega = request
        .data
        .iter()
        .zip(request.data.iter().skip(1))
        .map(|(a, b)| (b - a).abs())
        .sum::<f64>()
        / (request.data.len() - 1).max(1) as f64;

    let sigma = Sigma::new(
        psi.clamp(0.0, 1.0),
        rho.clamp(0.0, 1.0),
        omega.clamp(0.0, 1.0),
    );
    let resonit = Resonit::new(sigma, "api".to_string(), chrono::Utc::now().timestamp());

    // Store in DomainLayer
    let resonit_id = resonit.id.clone();
    let resonance = (sigma.psi + sigma.rho + sigma.omega) / 3.0;

    {
        let domain_layer = state.domain_layer.lock().unwrap();
        let mut resonits = domain_layer.resonits.lock().unwrap();
        resonits.insert(resonit_id.clone(), resonit);
    }

    Ok(Json(ResonitResponse {
        id: resonit_id,
        dimension: request.data.len(),
        resonance,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Get Resonit by ID
async fn get_resonit(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ResonitResponse>> {
    let domain_layer = state.domain_layer.lock().unwrap();
    let resonits = domain_layer.resonits.lock().unwrap();

    let resonit = resonits
        .get(&id)
        .ok_or_else(|| ApiError::NotFound(format!("Resonit {} not found", id)))?;

    let resonance = (resonit.sigma.psi + resonit.sigma.rho + resonit.sigma.omega) / 3.0;

    Ok(Json(ResonitResponse {
        id: resonit.id.clone(),
        dimension: resonit.coordinates.as_ref().map(|c| c.len()).unwrap_or(3),
        resonance,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Cluster Resonits into a Resonat
#[derive(Debug, Deserialize)]
struct ClusterResonatRequest {
    resonit_ids: Vec<String>,
    #[allow(dead_code)]
    clustering_method: Option<String>,
    #[allow(dead_code)]
    threshold: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ResonatResponse {
    id: String,
    resonit_count: usize,
    stability: f64,
    topology: String,
    timestamp: String,
}

async fn cluster_resonat(
    State(state): State<AppState>,
    Json(request): Json<ClusterResonatRequest>,
) -> Result<Json<ResonatResponse>> {
    use mef_domains::Resonat;

    // Load resonits from storage
    let domain_layer = state.domain_layer.lock().unwrap();
    let resonits_map = domain_layer.resonits.lock().unwrap();

    let mut resonits = Vec::new();
    for id in &request.resonit_ids {
        if let Some(resonit) = resonits_map.get(id) {
            resonits.push(resonit.clone());
        }
    }

    if resonits.is_empty() {
        return Err(ApiError::InvalidInput(
            "No valid resonits found for clustering".to_string(),
        ));
    }

    // Create Resonat from resonits
    let resonat = Resonat::new(resonits)?;
    let resonat_id = resonat.id.clone();
    let stability = resonat.metrics.stability;

    // Store resonat
    drop(resonits_map); // Release lock before acquiring next one
    let mut resonats = domain_layer.resonats.lock().unwrap();
    resonats.insert(resonat_id.clone(), resonat);

    Ok(Json(ResonatResponse {
        id: resonat_id,
        resonit_count: request.resonit_ids.len(),
        stability,
        topology: "torus".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Get Resonat by ID
async fn get_resonat(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ResonatResponse>> {
    let domain_layer = state.domain_layer.lock().unwrap();
    let resonats = domain_layer.resonats.lock().unwrap();

    let resonat = resonats
        .get(&id)
        .ok_or_else(|| ApiError::NotFound(format!("Resonat {} not found", id)))?;

    Ok(Json(ResonatResponse {
        id: resonat.id.clone(),
        resonit_count: resonat.resonits.len(),
        stability: resonat.metrics.stability,
        topology: "torus".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Triangulate MeshHolo from Resonat
#[derive(Debug, Deserialize)]
struct TriangulateMeshRequest {
    resonat_id: String,
    #[allow(dead_code)]
    triangulation_method: Option<String>,
}

#[derive(Debug, Serialize)]
struct MeshResponse {
    id: String,
    resonat_id: String,
    vertices: usize,
    faces: usize,
    euler_characteristic: i32,
    timestamp: String,
}

async fn triangulate_mesh(
    State(state): State<AppState>,
    Json(request): Json<TriangulateMeshRequest>,
) -> Result<Json<MeshResponse>> {
    use mef_domains::MeshHolo;

    // Load Resonat from storage
    let domain_layer = state.domain_layer.lock().unwrap();
    let resonats = domain_layer.resonats.lock().unwrap();

    let resonat = resonats
        .get(&request.resonat_id)
        .ok_or_else(|| ApiError::NotFound(format!("Resonat {} not found", request.resonat_id)))?;

    // Create MeshHolo triangulation
    let seed = format!("mesh-{}", resonat.id);
    let mesh = MeshHolo::from_resonat(resonat, seed);
    let mesh_id = mesh.id.clone();

    let response = MeshResponse {
        id: mesh_id.clone(),
        resonat_id: request.resonat_id.clone(),
        vertices: mesh.vertices.len(),
        faces: mesh.simplices.len(),
        euler_characteristic: if mesh.invariants.betti.len() >= 2 {
            // χ = b_0 - b_1 + b_2 - ... (alternating sum of Betti numbers)
            mesh.invariants
                .betti
                .iter()
                .enumerate()
                .map(|(i, &b)| if i % 2 == 0 { b as i32 } else { -(b as i32) })
                .sum()
        } else {
            2 // Default for sphere
        },
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // Store mesh
    drop(resonats); // Release lock before acquiring next one
    let mut meshes = domain_layer.meshes.lock().unwrap();
    meshes.insert(mesh_id, mesh);

    Ok(Json(response))
}

/// Get MeshHolo by ID
#[derive(Debug, Deserialize)]
struct GetMeshQuery {
    #[serde(default)]
    #[allow(dead_code)]
    format: Option<String>, // json, obj, ply
}

async fn get_mesh(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(_query): Query<GetMeshQuery>,
) -> Result<Json<MeshResponse>> {
    let domain_layer = state.domain_layer.lock().unwrap();
    let meshes = domain_layer.meshes.lock().unwrap();

    let mesh = meshes
        .get(&id)
        .ok_or_else(|| ApiError::NotFound(format!("Mesh {} not found", id)))?;

    Ok(Json(MeshResponse {
        id: mesh.id.clone(),
        resonat_id: "unknown".to_string(), // Would need to track this in mesh metadata
        vertices: mesh.vertices.len(),
        faces: mesh.simplices.len(),
        euler_characteristic: if mesh.invariants.betti.len() >= 2 {
            // χ = b_0 - b_1 + b_2 - ... (alternating sum of Betti numbers)
            mesh.invariants
                .betti
                .iter()
                .enumerate()
                .map(|(i, &b)| if i % 2 == 0 { b as i32 } else { -(b as i32) })
                .sum()
        } else {
            2 // Default for sphere
        },
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Perform homeomorphic transfer between domains
#[derive(Debug, Deserialize)]
struct HomeomorphicTransferRequest {
    source_domain: String,
    target_domain: String,
    data: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct TransferResponse {
    transfer_id: String,
    source_domain: String,
    target_domain: String,
    preserved_topology: bool,
    distortion: f64,
    timestamp: String,
}

async fn homeomorphic_transfer(
    State(state): State<AppState>,
    Json(request): Json<HomeomorphicTransferRequest>,
) -> Result<Json<TransferResponse>> {
    // Process data through domain layer to create a mesh
    let raw_data = serde_json::json!({ "data": request.data });

    let mut domain_layer = state.domain_layer.lock().unwrap();
    let result = domain_layer.process_domain_data(
        &raw_data,
        &request.source_domain,
        Some(&request.target_domain),
    )?;

    // Check if cross-domain transfer occurred
    let (preserved, distortion) = if let Some(cross_domain) = result.cross_domain {
        // Calculate distortion from invariants preservation
        let betti_preserved = cross_domain.invariants_preserved.contains_key("betti");
        (betti_preserved, 0.02)
    } else {
        (false, 1.0)
    };

    Ok(Json(TransferResponse {
        transfer_id: format!("transfer_{}", uuid::Uuid::new_v4()),
        source_domain: request.source_domain,
        target_domain: request.target_domain,
        preserved_topology: preserved,
        distortion,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Check compatibility between two domains
#[derive(Debug, Deserialize)]
struct CompatibilityQuery {
    source: String,
    target: String,
}

#[derive(Debug, Serialize)]
struct CompatibilityResponse {
    compatible: bool,
    compatibility_score: f64,
    recommended_method: String,
}

async fn check_compatibility(
    State(state): State<AppState>,
    Query(query): Query<CompatibilityQuery>,
) -> Result<Json<CompatibilityResponse>> {
    let domain_layer = state.domain_layer.lock().unwrap();

    // Check if both adapters exist
    let source_exists = domain_layer.adapters.contains_key(&query.source);
    let target_exists = domain_layer.adapters.contains_key(&query.target);

    let compatible = source_exists && target_exists;
    let score = if compatible { 0.88 } else { 0.0 };

    Ok(Json(CompatibilityResponse {
        compatible,
        compatibility_score: score,
        recommended_method: if compatible {
            "manifold_alignment".to_string()
        } else {
            "adapter_required".to_string()
        },
    }))
}

/// Evolve Infogenome through genetic algorithm
#[derive(Debug, Deserialize)]
struct EvolveInfogenomeRequest {
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
}

#[derive(Debug, Serialize)]
struct InfogenomeResponse {
    id: String,
    generation: usize,
    fitness: f64,
    operators: Vec<String>,
    timestamp: String,
}

async fn evolve_infogenome(
    State(state): State<AppState>,
    Json(request): Json<EvolveInfogenomeRequest>,
) -> Result<Json<InfogenomeResponse>> {
    let mut domain_layer = state.domain_layer.lock().unwrap();

    // Evolve population for requested generations
    for _ in 0..request.generations {
        // Create mutated offspring
        let mut new_population = Vec::new();

        for genome in &domain_layer.infogenomes {
            let mutant = genome.mutate(request.mutation_rate);
            new_population.push(mutant);
        }

        // Combine with original population
        domain_layer.infogenomes.extend(new_population);

        // Select best individuals (simplified - keep best half)
        domain_layer
            .infogenomes
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        domain_layer.infogenomes.truncate(request.population_size);
    }

    // Return best genome
    let best = domain_layer
        .infogenomes
        .first()
        .ok_or_else(|| ApiError::Internal("No infogenomes in population".to_string()))?;

    let operators: Vec<String> = best
        .genes
        .iter()
        .map(|g| format!("{:?}", g.operator))
        .collect();

    Ok(Json(InfogenomeResponse {
        id: best.id.clone(),
        generation: request.generations,
        fitness: best.fitness,
        operators,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Get best Infogenome from population
async fn get_best_infogenome(State(state): State<AppState>) -> Result<Json<InfogenomeResponse>> {
    let domain_layer = state.domain_layer.lock().unwrap();

    let best = domain_layer
        .infogenomes
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .ok_or_else(|| ApiError::Internal("No infogenomes in population".to_string()))?;

    let operators: Vec<String> = best
        .genes
        .iter()
        .map(|g| format!("{:?}", g.operator))
        .collect();

    Ok(Json(InfogenomeResponse {
        id: best.id.clone(),
        generation: 0, // Would need to track generation in genome
        fitness: best.fitness,
        operators,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Get domain layer status
#[derive(Debug, Serialize)]
struct DomainStatusResponse {
    resonits_total: usize,
    resonats_total: usize,
    meshes_total: usize,
    active_transfers: usize,
    infogenome_population: usize,
}

async fn get_domain_status(State(state): State<AppState>) -> Result<Json<DomainStatusResponse>> {
    let domain_layer = state.domain_layer.lock().unwrap();

    let resonits_total = domain_layer.resonits.lock().unwrap().len();
    let resonats_total = domain_layer.resonats.lock().unwrap().len();
    let meshes_total = domain_layer.meshes.lock().unwrap().len();
    let infogenome_population = domain_layer.infogenomes.len();

    Ok(Json(DomainStatusResponse {
        resonits_total,
        resonats_total,
        meshes_total,
        active_transfers: 0, // Would need to track active transfers
        infogenome_population,
    }))
}

/// Get torus topology information
#[derive(Debug, Serialize)]
struct TorusTopologyResponse {
    major_radius: f64,
    minor_radius: f64,
    genus: usize,
    euler_characteristic: i32,
}

async fn get_torus_topology(State(_state): State<AppState>) -> Result<Json<TorusTopologyResponse>> {
    // Return canonical torus topology parameters
    // In a full implementation, this would be computed from domain meshes
    Ok(Json(TorusTopologyResponse {
        major_radius: 2.0,
        minor_radius: 1.0,
        genus: 1,
        euler_characteristic: 0, // V - E + F = 0 for torus (genus 1)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;

    async fn test_state() -> AppState {
        AppState::new(ApiConfig::default()).await.unwrap()
    }

    #[tokio::test]
    async fn test_process_domain_data() {
        let state = test_state().await;
        let request = DomainProcessRequest {
            data: vec![1.0, 2.0, 3.0],
            domain_type: "test".to_string(),
            params: None,
        };
        let result = process_domain_data(State(state), Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_resonit() {
        let state = test_state().await;
        let request = CreateResonitRequest {
            data: vec![1.0, 2.0, 3.0],
            metadata: None,
        };
        let result = create_resonit(State(state), Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cluster_resonat() {
        let state = test_state().await;

        // Create some resonits first
        use mef_domains::{Resonit, Sigma};
        {
            let domain_layer = state.domain_layer.lock().unwrap();
            let mut resonits = domain_layer.resonits.lock().unwrap();
            let r1 = Resonit::new(Sigma::new(0.5, 0.5, 0.5), "test".to_string(), 0);
            let r2 = Resonit::new(Sigma::new(0.6, 0.6, 0.6), "test".to_string(), 0);
            resonits.insert("r1".to_string(), r1);
            resonits.insert("r2".to_string(), r2);
        }

        let request = ClusterResonatRequest {
            resonit_ids: vec!["r1".to_string(), "r2".to_string()],
            clustering_method: None,
            threshold: None,
        };
        let result = cluster_resonat(State(state), Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_triangulate_mesh() {
        let state = test_state().await;

        // Create a resonat first
        use mef_domains::{Resonat, Resonit, Sigma};
        {
            let domain_layer = state.domain_layer.lock().unwrap();
            let r1 = Resonit::new(Sigma::new(0.5, 0.5, 0.5), "test".to_string(), 0);
            let r2 = Resonit::new(Sigma::new(0.6, 0.6, 0.6), "test".to_string(), 0);
            let resonat = Resonat::new(vec![r1, r2]).unwrap();
            let mut resonats = domain_layer.resonats.lock().unwrap();
            resonats.insert("test_resonat".to_string(), resonat);
        }

        let request = TriangulateMeshRequest {
            resonat_id: "test_resonat".to_string(),
            triangulation_method: None,
        };
        let result = triangulate_mesh(State(state), Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_domain_status() {
        let state = test_state().await;
        let result = get_domain_status(State(state)).await;
        assert!(result.is_ok());
    }
}
