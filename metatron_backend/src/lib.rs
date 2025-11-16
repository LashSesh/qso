//! # Metatron Backend - Quantum Provider Abstraction
//!
//! This crate provides a unified interface for executing quantum circuits across
//! multiple backend providers (local simulator, IBM Quantum, etc.).
//!
//! ## Design Philosophy
//!
//! - **Provider Agnostic**: Write algorithm code once, run on any backend
//! - **Safe by Default**: QPU access is explicit and opt-in
//! - **Future Proof**: Easy to add new providers (Azure, IonQ, etc.)
//! - **Zero Overhead**: Local simulator has minimal abstraction cost
//!
//! ## Example
//!
//! ```rust
//! use metatron_backend::{QuantumBackend, LocalSimulatorBackend, MetatronCircuit};
//!
//! // Create a backend
//! let backend = LocalSimulatorBackend::new();
//!
//! // Create a simple circuit
//! let circuit = MetatronCircuit::new(2)
//!     .h(0)
//!     .cnot(0, 1);
//!
//! // Execute
//! let result = backend.run_circuit(&circuit, 1000).unwrap();
//! println!("Counts: {:?}", result.counts);
//! ```

pub mod circuit;
pub mod backends;
pub mod registry;

pub use circuit::{MetatronCircuit, MeasurementResult, Gate, GateType};
pub use backends::{
    QuantumBackend, BackendCapabilities,
    local::LocalSimulatorBackend,
};

#[cfg(feature = "ibm")]
pub use backends::ibm::{IbmQuantumBackend, IbmConfig, IbmMode};

pub use registry::{BackendRegistry, BackendMode};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        QuantumBackend, BackendCapabilities,
        MetatronCircuit, MeasurementResult,
        LocalSimulatorBackend,
        BackendRegistry, BackendMode,
    };

    #[cfg(feature = "ibm")]
    pub use crate::{IbmQuantumBackend, IbmConfig, IbmMode};
}
