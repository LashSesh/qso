//! # Metatron Quantum State Operator (QSO)
//!
//! A high-performance quantum computing framework built around the sacred geometry
//! of the Metatron Cube — a 13-dimensional quantum system with 78 edges representing
//! all five Platonic solids.
//!
//! ## Overview
//!
//! The Metatron QSO provides:
//!
//! - **Quantum Graph Dynamics**: State evolution on 13-node Metatron geometry
//! - **Variational Quantum Algorithms**: VQE, QAOA, VQC with multiple ansätze
//! - **Quantum Walks**: Continuous-time quantum walks with Krylov methods
//! - **Dynamic Tripolar Logic (DTL)**: 58.5% information advantage over binary
//! - **Topological Codes**: Symmetry-protected quantum error correction
//!
//! ## Quick Start
//!
//! ```rust
//! use metatron_qso::prelude::*;
//!
//! // Create the Metatron graph
//! let graph = MetatronGraph::new();
//!
//! // Create the Hamiltonian
//! let params = QSOParameters::default();
//! let hamiltonian = MetatronHamiltonian::new(&graph, &params);
//!
//! // Initialize quantum state on central node
//! let initial_state = QuantumState::basis_state(0)?;
//!
//! // Run quantum walk
//! let qw = ContinuousTimeQuantumWalk::new(&hamiltonian);
//! let evolved = qw.evolve(&initial_state, 1.0);
//!
//! // Check probability distribution
//! let probs = evolved.probabilities();
//! println!("Probability at node 0: {:.4}", probs[0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Architecture
//!
//! The library is organized into focused modules:
//!
//! - [`graph`] - Metatron Cube geometry and graph structures
//! - [`quantum`] - Quantum states, operators, and dimensions
//! - [`hamiltonian`] - Graph Hamiltonians and spectral decomposition
//! - [`quantum_walk`] - Quantum walk algorithms (feature: `walks`)
//! - [`vqa`] - Variational Quantum Algorithms (feature: `vqa`)
//! - [`dtl`] - Dynamic Tripolar Logic (feature: `dtl`)
//! - `symmetry_codes` - Topological error correction (feature: `codes`)
//! - `advanced_algorithms` - Grover search, Boson sampling (feature: `advanced`)
//!
//! ## Features
//!
//! Control which components are compiled via Cargo features:
//!
//! ```toml
//! [dependencies]
//! metatron-qso-rs = { version = "0.1", features = ["walks", "vqa"] }
//! ```
//!
//! Available features:
//! - `walks` (default) - Quantum walk algorithms
//! - `vqa` (default) - VQE, QAOA, VQC
//! - `dtl` (default) - Dynamic Tripolar Logic
//! - `codes` - Topological codes
//! - `advanced` - Advanced algorithms (Grover, Boson sampling)
//!
//! ## Graph Structure
//!
//! The Metatron Cube consists of:
//! - **13 nodes**: 1 center + 6 hexagon vertices + 6 cube vertices
//! - **78 edges**: Fully connected subgraphs
//! - **Code distance**: d ≥ 6 for topological error correction
//!
//! ```text
//!        Hexagon Layer (nodes 1-6)
//!              /|\
//!             / | \
//!            /  |  \
//!           0-------0  ← Central node (0)
//!            \  |  /
//!             \ | /
//!              \|/
//!        Cube Layer (nodes 7-12)
//! ```
//!
//! ## Performance
//!
//! Benchmarks on Intel i7-12700K:
//! - Quantum Walk: 31,941 ops/sec
//! - VQE Convergence: ~150 iterations to E₀ = -12.9997
//! - QAOA MaxCut: 99.74% approximation ratio (depth p=3)
//!
//! ## Examples
//!
//! See the [`examples/`](https://github.com/LashSesh/qso/tree/main/metatron-qso-rs/examples) directory:
//! - `quantum_walk_basic.rs` - Simple quantum walk demo
//! - `qaoa_maxcut_basic.rs` - MaxCut optimization
//! - `vqa_demo.rs` - Complete VQA workflow
//!
//! ## References
//!
//! - [Architecture Documentation](https://github.com/LashSesh/qso/blob/main/metatron-qso-rs/docs/ARCHITECTURE.md)
//! - [Developer Guide](https://github.com/LashSesh/qso/blob/main/metatron-qso-rs/docs/RUST_CORE_GUIDE.md)
//! - [VQA Implementation](https://github.com/LashSesh/qso/blob/main/VQA_IMPLEMENTATION_GUIDE.md)

// Core modules (always available)
pub mod graph;
pub mod hamiltonian;
pub mod params;
pub mod qso;
pub mod quantum;

// Feature-gated modules
#[cfg(feature = "walks")]
pub mod quantum_walk;

#[cfg(feature = "vqa")]
pub mod vqa;

#[cfg(feature = "dtl")]
pub mod dtl;

#[cfg(feature = "codes")]
pub mod symmetry_codes;

#[cfg(feature = "advanced")]
pub mod advanced_algorithms;

// High-level toolkits
pub mod optimizer;
pub mod quantum_walk_toolkit;

// Core re-exports (always available)
pub use crate::graph::metatron::MetatronGraph;
pub use crate::hamiltonian::{MetatronHamiltonian, SpectrumInfo};
pub use crate::params::QSOParameters;
pub use crate::qso::QuantumStateOperator;
pub use crate::quantum::{METATRON_DIMENSION, operator::QuantumOperator, state::QuantumState};

// Feature-gated re-exports
#[cfg(feature = "dtl")]
pub use crate::dtl::{network::DTLResonatorNetwork, operations::DTLOperations, state::DTLState};

/// Prelude module for convenient imports.
///
/// Import everything you need with:
/// ```
/// use metatron_qso::prelude::*;
/// ```
pub mod prelude {
    // Core types (always available)
    pub use crate::graph::metatron::MetatronGraph;
    pub use crate::hamiltonian::{MetatronHamiltonian, SpectrumInfo};
    pub use crate::params::QSOParameters;
    pub use crate::qso::QuantumStateOperator;
    pub use crate::quantum::{METATRON_DIMENSION, operator::QuantumOperator, state::QuantumState};

    // DTL (feature: dtl)
    #[cfg(feature = "dtl")]
    pub use crate::dtl::{
        network::DTLResonatorNetwork, operations::DTLOperations, state::DTLState,
    };

    // Quantum Walks (feature: walks)
    #[cfg(feature = "walks")]
    pub use crate::quantum_walk::{
        BenchmarkMetadata, QuantumWalkBenchmarkSuite, QuantumWalkBenchmarker,
        continuous::{ContinuousTimeQuantumWalk, SpectralPropagator},
        krylov::{KrylovEvolution, KrylovProjection, LanczosResult},
        scattering::{DensityOfStates, ScatteringAnalysis, ScatteringChannel},
    };

    // VQA (feature: vqa)
    #[cfg(feature = "vqa")]
    pub use crate::vqa::{
        ansatz::{Ansatz, AnsatzType, EfficientSU2Ansatz, HardwareEfficientAnsatz, MetatronAnsatz},
        cost_function::{CostFunction, GradientMethod},
        optimizer::{OptimizationResult, Optimizer, OptimizerConfig, OptimizerType},
        qaoa::{QAOA, QAOABuilder, QAOAConfig, QAOAResult},
        vqc::{VQC, VQCBuilder, VQCConfig, VQCResult},
        vqe::{VQE, VQEBuilder, VQEConfig, VQEResult},
    };

    // Symmetry Codes (feature: codes)
    #[cfg(feature = "codes")]
    pub use crate::symmetry_codes::MetatronCode;

    // Advanced Algorithms (feature: advanced)
    #[cfg(feature = "advanced")]
    pub use crate::advanced_algorithms::{
        GroverSearchResult, MetatronGraphML, MetatronGroverSearch, MultiGroverSearchResult,
        PlatonicBosonSampling, PlatonicInterferenceAnalysis, QGNN,
    };
}
