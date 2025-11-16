/// Application state for API server
use anyhow::Result;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::ApiConfig;
use mef_core::gates::merkaba_gate::MerkabaGate;
use mef_core::MEFCore;
use mef_coupling::SpiralCouplingEngine;
use mef_domains::DomainLayer;
use mef_ledger::MEFLedger;
use mef_spiral::SpiralConfig;
use mef_topology::MetatronRouter;
use mef_vector_db::IndexManager;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ApiConfig>,
    pub spiral_config: Arc<SpiralConfig>,
    pub store_path: Arc<PathBuf>,
    pub ledger: Arc<Mutex<MEFLedger>>,
    pub index_manager: Arc<Mutex<IndexManager>>,
    pub coupling_engine: Arc<Mutex<SpiralCouplingEngine>>,
    pub metatron_router: Arc<Mutex<MetatronRouter>>,
    pub merkaba_gate: Arc<Mutex<MerkabaGate>>,
    pub domain_layer: Arc<Mutex<DomainLayer>>,
}

impl AppState {
    /// Create new application state with initialized components
    pub async fn new(config: ApiConfig) -> Result<Self> {
        // Initialize spiral configuration
        let spiral_config = SpiralConfig::default();
        let store_path = config.store_path.clone();

        // Initialize ledger
        let ledger = MEFLedger::new(&config.ledger_path)?;

        // Initialize index manager
        let index_manager = IndexManager::new(Some(store_path.join("vector_db")))?;

        // Initialize coupling engine with proper parameters
        // SpiralCouplingEngine::new(base_path, params, resonance, eps_pi, zk_mu)
        let coupling_engine = SpiralCouplingEngine::new(
            Some(store_path.join("coupling")),
            None, // Use default SpiralParameters
            None, // Use default ResonanceMetric
            0.02, // eps_pi - default epsilon for delta_pi calculations
            0.1,  // zk_mu - zero-knowledge threshold
        )?;

        // Initialize Metatron Router
        let metatron_router = MetatronRouter::new(store_path.join("metatron"));

        // Initialize Merkaba Gate
        let merkaba_gate = MerkabaGate::new(store_path.join("merkaba_audit.jsonl"));

        // Initialize MEF-Core pipeline for domain layer
        let mef_pipeline = Arc::new(MEFCore::new("api-domain-seed", None)?);

        // Initialize Domain Layer
        let domain_layer = DomainLayer::new(
            mef_pipeline,
            Arc::new(Mutex::new(metatron_router.clone())),
            store_path.join("domains"),
        )?;

        Ok(Self {
            config: Arc::new(config),
            spiral_config: Arc::new(spiral_config),
            store_path: Arc::new(store_path),
            ledger: Arc::new(Mutex::new(ledger)),
            index_manager: Arc::new(Mutex::new(index_manager)),
            coupling_engine: Arc::new(Mutex::new(coupling_engine)),
            metatron_router: Arc::new(Mutex::new(metatron_router)),
            merkaba_gate: Arc::new(Mutex::new(merkaba_gate)),
            domain_layer: Arc::new(Mutex::new(domain_layer)),
        })
    }
}
