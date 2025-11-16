/*!
 * DomainLayer - Main orchestrator for domain-specific transformations
 *
 * Integrates with MEF-Core pipeline, Metatron Router, and gates to provide
 * end-to-end domain data processing with topological validation.
 */

use crate::adapter::DomainAdapter;
use crate::infogenome::Infogenome;
use crate::meshholo::MeshHolo;
use crate::resonat::Resonat;
use crate::resonit::Resonit;
use anyhow::{anyhow, Result};
use mef_core::{MEFCore, MandorlaField};
use mef_topology::MetatronRouter;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Main Domain-Layer orchestrator
///
/// Manages domain-specific transformations, Resonit/Resonat clustering,
/// MeshHolo triangulation, and cross-domain homeomorphism.
pub struct DomainLayer {
    /// MEF-Core pipeline for TIC processing
    pub mef_pipeline: Arc<MEFCore>,

    /// Metatron Router for operator transformations
    pub metatron_router: Arc<Mutex<MetatronRouter>>,

    /// Storage path for domain-specific data
    pub storage_path: PathBuf,

    /// Registry of domain adapters
    pub adapters: HashMap<String, Box<dyn DomainAdapter>>,

    /// Resonit storage
    pub resonits: Arc<Mutex<HashMap<String, Resonit>>>,

    /// Resonat storage
    pub resonats: Arc<Mutex<HashMap<String, Resonat>>>,

    /// MeshHolo triangulations
    pub meshes: Arc<Mutex<HashMap<String, MeshHolo>>>,

    /// Infogenome population
    pub infogenomes: Vec<Infogenome>,

    /// Mandorla field for gate validation
    pub mandorla: MandorlaField,

    /// Metrics tracking
    pub metrics: Arc<Mutex<DomainMetrics>>,
}

/// Domain processing metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DomainMetrics {
    pub resonits_created: usize,
    pub resonats_formed: usize,
    pub meshes_triangulated: usize,
    pub cross_domain_transfers: usize,
}

/// Domain processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainProcessingResult {
    pub resonat_id: String,
    pub mesh_id: String,
    pub tic_id: Option<String>,
    pub gate_validation: GateValidation,
    pub cross_domain: Option<CrossDomainResult>,
    pub metrics: DomainMetrics,
}

/// Gate validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateValidation {
    pub passed: bool,
    pub resonance: f64,
    pub entropy: f64,
    pub variance: f64,
    pub pi_gap: f64,
    pub timestamp: String,
}

/// Cross-domain transfer result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainResult {
    pub source_domain: String,
    pub target_domain: String,
    pub source_mesh_id: String,
    pub transformed_vertices: usize,
    pub invariants_preserved: HashMap<String, Value>,
}

impl DomainLayer {
    /// Create a new DomainLayer with default configuration
    pub fn new(
        mef_pipeline: Arc<MEFCore>,
        metatron_router: Arc<Mutex<MetatronRouter>>,
        storage_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&storage_path)?;

        // Initialize Mandorla field
        let mandorla = MandorlaField::new(0.985, 0.5, 0.5);

        // Initialize infogenomes
        let infogenomes = Self::initialize_infogenomes();

        Ok(Self {
            mef_pipeline,
            metatron_router,
            storage_path,
            adapters: HashMap::new(),
            resonits: Arc::new(Mutex::new(HashMap::new())),
            resonats: Arc::new(Mutex::new(HashMap::new())),
            meshes: Arc::new(Mutex::new(HashMap::new())),
            infogenomes,
            mandorla,
            metrics: Arc::new(Mutex::new(DomainMetrics::default())),
        })
    }

    /// Register a domain adapter
    pub fn register_adapter(&mut self, adapter: Box<dyn DomainAdapter>) {
        let name = adapter.domain_name().to_string();
        self.adapters.insert(name, adapter);
    }

    /// Initialize population of Infogenomes
    fn initialize_infogenomes() -> Vec<Infogenome> {
        let mut infogenomes = Vec::new();

        // Create base genome
        let base = Infogenome::base();
        infogenomes.push(base.clone());

        // Create variations through mutation
        for _ in 0..4 {
            let mutant = base.mutate(0.3);
            infogenomes.push(mutant);
        }

        infogenomes
    }

    /// Process domain-specific data through the complete pipeline
    pub fn process_domain_data(
        &mut self,
        raw_data: &Value,
        domain: &str,
        target_domain: Option<&str>,
    ) -> Result<DomainProcessingResult> {
        // Step 1: Transform to Resonits
        let adapter = self
            .adapters
            .get(domain)
            .ok_or_else(|| anyhow!("No adapter for domain: {}", domain))?;

        let resonits = adapter.transform(raw_data)?;

        // Store resonits
        {
            let mut resonits_map = self.resonits.lock().unwrap();
            for resonit in &resonits {
                resonits_map.insert(resonit.id.clone(), resonit.clone());
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.resonits_created += resonits.len();
        }

        // Step 2: Cluster into Resonat
        let resonat = self.cluster_resonits(resonits)?;
        let resonat_id = resonat.id.clone();

        // Store resonat
        {
            let mut resonats_map = self.resonats.lock().unwrap();
            resonats_map.insert(resonat_id.clone(), resonat.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.resonats_formed += 1;
        }

        // Step 3: Create MeshHolo triangulation
        let mesh = self.triangulate_resonat(&resonat)?;
        let mesh_id = mesh.id.clone();

        // Store mesh
        {
            let mut meshes_map = self.meshes.lock().unwrap();
            meshes_map.insert(mesh_id.clone(), mesh.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.meshes_triangulated += 1;
        }

        // Step 4: Apply Infogenome transformations
        let transformed_state = self.apply_infogenome(&resonat, &mesh)?;

        // Step 5: Validate through Mandorla gate
        let gate_validation = self.validate_mandorla(&transformed_state, &mesh)?;

        // Step 6: Create domain-enhanced TIC (optional, requires full MEF pipeline)
        let tic_id = None; // Placeholder for now

        // Step 7: Optional cross-domain transfer
        let cross_domain = if let Some(target) = target_domain {
            if target != domain {
                let result = self.homeomorphic_transfer(&mesh, domain, target)?;

                // Update metrics
                {
                    let mut metrics = self.metrics.lock().unwrap();
                    metrics.cross_domain_transfers += 1;
                }

                Some(result)
            } else {
                None
            }
        } else {
            None
        };

        // Get current metrics
        let metrics = {
            let metrics = self.metrics.lock().unwrap();
            metrics.clone()
        };

        Ok(DomainProcessingResult {
            resonat_id,
            mesh_id,
            tic_id,
            gate_validation,
            cross_domain,
            metrics,
        })
    }

    /// Cluster Resonits into a coherent Resonat
    fn cluster_resonits(&self, resonits: Vec<Resonit>) -> Result<Resonat> {
        Resonat::new(resonits)
    }

    /// Create MeshHolo triangulation from Resonat
    fn triangulate_resonat(&self, resonat: &Resonat) -> Result<MeshHolo> {
        // Use a seed for reproducibility
        let seed = format!("domain-{}", resonat.id);
        Ok(MeshHolo::from_resonat(resonat, seed))
    }

    /// Apply Infogenome transformations to Resonat
    fn apply_infogenome(&mut self, resonat: &Resonat, _mesh: &MeshHolo) -> Result<Vec<f64>> {
        // Select best Infogenome based on fitness
        let best_genome = self
            .infogenomes
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .ok_or_else(|| anyhow!("No infogenomes available"))?;

        // Create initial state from Resonat centroid
        let mut state = resonat
            .centroid
            .clone()
            .ok_or_else(|| anyhow!("Resonat has no centroid"))?;

        // Pad to Metatron dimensions (13)
        while state.len() < 13 {
            state.push(0.0);
        }
        state.truncate(13);

        // Apply genes in sequence through Metatron router
        let mut router = self.metatron_router.lock().unwrap();

        for gene in &best_genome.genes {
            let prev_state = state.clone();

            // Apply operator through router
            // Note: Full operator application would require building a RouteSpec
            // For now, we'll apply a simple transformation
            let transformed = router.transform(&state, None);
            state = transformed.output_vector;

            // Weight the transformation
            for i in 0..state.len() {
                state[i] = gene.weight * state[i] + (1.0 - gene.weight) * prev_state[i];
            }
        }

        // Update genome fitness (simplified - would need actual quality metric)
        let quality = 0.5; // Placeholder
        let idx = self
            .infogenomes
            .iter()
            .position(|g| g.id == best_genome.id)
            .unwrap();
        self.infogenomes[idx].update_fitness(quality);

        Ok(state)
    }

    /// Validate transformation through Mandorla gate
    fn validate_mandorla(&mut self, state: &[f64], mesh: &MeshHolo) -> Result<GateValidation> {
        // Clear and populate Mandorla field
        self.mandorla.clear_inputs();

        // Add state at different scales
        let base_projection: Vec<f64> = state.iter().take(5).copied().collect();
        for scale in [0.5, 1.0, 2.0] {
            let scaled: Vec<f64> = base_projection.iter().map(|&x| x * scale).collect();
            self.mandorla.add_input(Array1::from_vec(scaled));
        }

        // Calculate gate metrics
        let resonance = self.mandorla.calc_resonance();
        let entropy = self.mandorla.calc_entropy();
        let variance = self.mandorla.calc_variance();

        // Get spectral gap from mesh
        let pi_gap = mesh.invariants.lambda_gap;

        // Determine if gate passes
        let passed = resonance > 0.5 && pi_gap < 0.1 && entropy < 2.0;

        Ok(GateValidation {
            passed,
            resonance,
            entropy,
            variance,
            pi_gap,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Perform homeomorphic transfer between domains
    fn homeomorphic_transfer(
        &self,
        mesh: &MeshHolo,
        source_domain: &str,
        target_domain: &str,
    ) -> Result<CrossDomainResult> {
        // Check if target adapter exists
        if !self.adapters.contains_key(target_domain) {
            return Err(anyhow!(
                "Target domain adapter not found: {}",
                target_domain
            ));
        }

        // Get Metatron embedding of source mesh
        let _embedding = mesh.to_metatron_embedding();

        // Apply homeomorphism through Metatron topology
        // (simplified - full implementation would use domain-specific transformations)
        let transformed_vertices = mesh.vertices.len();

        // Preserve topological invariants
        let mut invariants = HashMap::new();
        invariants.insert(
            "betti".to_string(),
            Value::Array(
                mesh.invariants
                    .betti
                    .iter()
                    .map(|&b| Value::Number(b.into()))
                    .collect(),
            ),
        );
        invariants.insert(
            "persistence".to_string(),
            Value::Number(
                serde_json::Number::from_f64(mesh.invariants.persistence)
                    .unwrap_or_else(|| 0.into()),
            ),
        );

        Ok(CrossDomainResult {
            source_domain: source_domain.to_string(),
            target_domain: target_domain.to_string(),
            source_mesh_id: mesh.id.clone(),
            transformed_vertices,
            invariants_preserved: invariants,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{SignalDomainAdapter, TextDomainAdapter};
    use mef_core::MEFCoreConfig;

    fn create_test_domain_layer() -> DomainLayer {
        let mef_pipeline =
            Arc::new(MEFCore::new("test-seed", Some(MEFCoreConfig::default())).unwrap());
        let metatron_router = Arc::new(Mutex::new(MetatronRouter::new("/tmp/test_router")));

        let mut layer =
            DomainLayer::new(mef_pipeline, metatron_router, "/tmp/test_domains").unwrap();

        // Register adapters
        layer.register_adapter(Box::new(TextDomainAdapter::new()));
        layer.register_adapter(Box::new(SignalDomainAdapter::new()));

        layer
    }

    #[test]
    fn test_domain_layer_creation() {
        let layer = create_test_domain_layer();
        assert_eq!(layer.adapters.len(), 2);
        assert!(layer.adapters.contains_key("text"));
        assert!(layer.adapters.contains_key("signal"));
    }

    #[test]
    fn test_initialize_infogenomes() {
        let infogenomes = DomainLayer::initialize_infogenomes();
        assert_eq!(infogenomes.len(), 5); // 1 base + 4 mutants
    }

    #[test]
    fn test_process_text_data() {
        let mut layer = create_test_domain_layer();

        let data = Value::String("Hello world. This is a test.".to_string());
        let result = layer.process_domain_data(&data, "text", None);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.resonat_id.is_empty());
        assert!(!result.mesh_id.is_empty());
    }

    #[test]
    fn test_process_signal_data() {
        let mut layer = create_test_domain_layer();

        let data = Value::Array((0..150).map(|i| Value::from(i as f64)).collect());
        let result = layer.process_domain_data(&data, "signal", None);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.resonat_id.is_empty());
    }

    #[test]
    fn test_cross_domain_transfer() {
        let mut layer = create_test_domain_layer();

        let data = Value::String("Test data".to_string());
        let result = layer.process_domain_data(&data, "text", Some("signal"));

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.cross_domain.is_some());

        let cross = result.cross_domain.unwrap();
        assert_eq!(cross.source_domain, "text");
        assert_eq!(cross.target_domain, "signal");
    }

    #[test]
    fn test_metrics_tracking() {
        let mut layer = create_test_domain_layer();

        let data = Value::String("Test".to_string());
        let _ = layer.process_domain_data(&data, "text", None);

        let metrics = layer.metrics.lock().unwrap();
        assert!(metrics.resonits_created > 0);
        assert_eq!(metrics.resonats_formed, 1);
        assert_eq!(metrics.meshes_triangulated, 1);
    }

    #[test]
    fn test_unknown_domain() {
        let mut layer = create_test_domain_layer();

        let data = Value::String("Test".to_string());
        let result = layer.process_domain_data(&data, "unknown", None);

        assert!(result.is_err());
    }
}
