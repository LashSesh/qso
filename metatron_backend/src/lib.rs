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

pub mod backends;
pub mod circuit;
pub mod registry;

pub use backends::{local::LocalSimulatorBackend, BackendCapabilities, QuantumBackend};
pub use circuit::{Gate, GateType, MeasurementResult, MetatronCircuit};

#[cfg(feature = "ibm")]
pub use backends::ibm::{IbmConfig, IbmMode, IbmQuantumBackend};

pub use registry::{BackendMode, BackendRegistry};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        BackendCapabilities, BackendMode, BackendRegistry, LocalSimulatorBackend,
        MeasurementResult, MetatronCircuit, QuantumBackend,
    };

    #[cfg(feature = "ibm")]
    pub use crate::{IbmConfig, IbmMode, IbmQuantumBackend};
}
