//! Common types for the unified system

use core_5d::{State5D, SystemParameters};
use mef_schemas::{GateDecision, KnowledgeObject, RouteSpec, SpectralSignature};

use crate::adapters::resonance_adapter::ProofOfResonanceData;

/// Configuration for gate evaluation thresholds
///
/// Controls the Merkaba Gate decision logic:
/// FIRE ⟺ (PoR = valid) ∧ (ΔPI ≤ epsilon) ∧ (Φ ≥ phi_threshold) ∧ (ΔV < 0)
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Path invariance threshold (epsilon)
    /// States with delta_pi ≤ epsilon are considered stable
    /// Default: 0.1
    pub epsilon: f64,

    /// Alignment threshold (phi_threshold)
    /// States with phi ≥ phi_threshold are considered aligned
    /// Default: 0.5
    pub phi_threshold: f64,

    /// Resonance field strength
    /// Used in ConstantResonanceField for PoR computation
    /// Default: 0.8
    pub resonance_strength: f64,
}

impl GateConfig {
    /// Create a new gate configuration with custom parameters
    pub fn new(epsilon: f64, phi_threshold: f64, resonance_strength: f64) -> Self {
        Self {
            epsilon,
            phi_threshold,
            resonance_strength,
        }
    }

    /// Create a strict gate configuration (harder to FIRE)
    pub fn strict() -> Self {
        Self {
            epsilon: 0.05,          // Tighter path invariance
            phi_threshold: 0.7,     // Higher alignment required
            resonance_strength: 0.9, // Stronger resonance
        }
    }

    /// Create a relaxed gate configuration (easier to FIRE)
    pub fn relaxed() -> Self {
        Self {
            epsilon: 0.2,           // Looser path invariance
            phi_threshold: 0.3,     // Lower alignment required
            resonance_strength: 0.6, // Weaker resonance
        }
    }
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            phi_threshold: 0.5,
            resonance_strength: 0.8,
        }
    }
}

/// Input for cognitive processing
///
/// Encapsulates all parameters needed to run the complete APOLLYON → MEF pipeline
#[derive(Clone)]
pub struct CognitiveInput {
    /// Initial 5D state for APOLLYON integration
    pub initial_state: State5D,

    /// System parameters for APOLLYON dynamics
    pub parameters: SystemParameters,

    /// Final integration time
    pub t_final: f64,

    /// TIC identifier for MEF storage
    pub tic_id: String,

    /// Seed for route selection
    pub seed: String,

    /// HD-style seed derivation path (e.g., "MEF/domain/stage/0001")
    pub seed_path: String,
}

/// Output from cognitive processing
///
/// Contains all results from the unified APOLLYON + MEF pipeline
#[derive(Debug)]
pub struct CognitiveOutput {
    /// Final 5D trajectory from APOLLYON integration
    pub trajectory: Vec<State5D>,

    /// Spectral signature computed from trajectory
    pub spectral_signature: SpectralSignature,

    /// Selected MEF route
    pub route: RouteSpec,

    /// Proof-of-Resonance data
    pub proof: ProofOfResonanceData,

    /// Gate decision (FIRE or HOLD)
    pub gate_decision: GateDecision,

    /// Knowledge object (if created)
    pub knowledge: Option<KnowledgeObject>,
}

/// Batch processing result
///
/// Contains results and any errors from batch processing
#[derive(Debug)]
pub struct BatchResult {
    /// Successfully processed outputs
    pub successes: Vec<CognitiveOutput>,

    /// Failed inputs with error messages
    pub failures: Vec<(usize, String)>,

    /// Total processing time in seconds
    pub total_time: f64,

    /// Average processing time per item in seconds
    pub avg_time: f64,
}

impl BatchResult {
    /// Create a new batch result
    pub fn new(
        successes: Vec<CognitiveOutput>,
        failures: Vec<(usize, String)>,
        total_time: f64,
    ) -> Self {
        let total_count = successes.len() + failures.len();
        let avg_time = if total_count > 0 {
            total_time / total_count as f64
        } else {
            0.0
        };

        Self {
            successes,
            failures,
            total_time,
            avg_time,
        }
    }

    /// Get the number of successful processing
    pub fn success_count(&self) -> usize {
        self.successes.len()
    }

    /// Get the number of failed processing
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Get the total number of inputs processed
    pub fn total_count(&self) -> usize {
        self.success_count() + self.failure_count()
    }

    /// Check if all inputs were processed successfully
    pub fn all_succeeded(&self) -> bool {
        self.failures.is_empty()
    }

    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_count() == 0 {
            0.0
        } else {
            (self.success_count() as f64 / self.total_count() as f64) * 100.0
        }
    }
}
