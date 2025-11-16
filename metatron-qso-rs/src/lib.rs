//! Metatron Quantum State Operator - Rust Reimagination
//!
//! This crate provides a modular, idiomatic Rust implementation of the Metatron
//! Quantum State Operator (QSO). It fuses graph-theoretic quantum dynamics with
//! Dynamic Tripolar Logic (DTL) resonator networks in a 13-dimensional hybrid
//! information framework inspired by the Metatron Cube. The design emphasizes
//! memory safety, composability, and high-performance numerics.

pub mod advanced_algorithms;
pub mod dtl;
pub mod graph;
pub mod hamiltonian;
pub mod params;
pub mod qso;
pub mod quantum;
pub mod quantum_walk;
pub mod symmetry_codes;
pub mod vqa;

pub use crate::dtl::{network::DTLResonatorNetwork, operations::DTLOperations, state::DTLState};
pub use crate::graph::metatron::MetatronGraph;
pub use crate::hamiltonian::{MetatronHamiltonian, SpectrumInfo};
pub use crate::params::QSOParameters;
pub use crate::qso::QuantumStateOperator;
pub use crate::quantum::{METATRON_DIMENSION, operator::QuantumOperator, state::QuantumState};

/// Prelude exporting the most commonly used types.
pub mod prelude {
    pub use crate::QuantumStateOperator;
    pub use crate::advanced_algorithms::{
        GroverSearchResult, MetatronGraphML, MetatronGroverSearch, MultiGroverSearchResult,
        PlatonicBosonSampling, PlatonicInterferenceAnalysis, QGNN,
    };
    pub use crate::dtl::{
        network::DTLResonatorNetwork, operations::DTLOperations, state::DTLState,
    };
    pub use crate::graph::metatron::MetatronGraph;
    pub use crate::hamiltonian::{MetatronHamiltonian, SpectrumInfo};
    pub use crate::params::QSOParameters;
    pub use crate::quantum::{METATRON_DIMENSION, operator::QuantumOperator, state::QuantumState};
    pub use crate::quantum_walk::{
        BenchmarkMetadata, QuantumWalkBenchmarkSuite, QuantumWalkBenchmarker,
        continuous::{ContinuousTimeQuantumWalk, SpectralPropagator},
        krylov::{KrylovEvolution, KrylovProjection, LanczosResult},
        scattering::{DensityOfStates, ScatteringAnalysis, ScatteringChannel},
    };
    pub use crate::symmetry_codes::MetatronCode;
    pub use crate::vqa::{
        ansatz::{Ansatz, AnsatzType, EfficientSU2Ansatz, HardwareEfficientAnsatz, MetatronAnsatz},
        cost_function::{CostFunction, GradientMethod},
        optimizer::{OptimizationResult, Optimizer, OptimizerConfig, OptimizerType},
        qaoa::{QAOA, QAOABuilder, QAOAConfig, QAOAResult},
        vqc::{VQC, VQCBuilder, VQCConfig, VQCResult},
        vqe::{VQE, VQEBuilder, VQEConfig, VQEResult},
    };
}
