//! Error type shared by every module of the crate.

use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, QsoError>;

/// All recoverable errors produced by `metatron_qso`.
///
/// Programming errors that can only arise from violating a documented
/// invariant inside the crate are `debug_assert!`ed instead; everything that
/// depends on user input is reported through this type.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum QsoError {
    /// A qubit index was outside `0..n_qubits`.
    #[error("qubit index {qubit} out of range for a {n_qubits}-qubit register")]
    QubitOutOfRange { qubit: usize, n_qubits: usize },

    /// The same qubit was used twice in a multi-qubit gate.
    #[error("gate {gate} acts on qubit {qubit} more than once")]
    DuplicateQubit { gate: &'static str, qubit: usize },

    /// Two objects that must share a dimension did not.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A state vector length was not a power of two.
    #[error("length {0} is not a power of two")]
    NotPowerOfTwo(usize),

    /// The register is too large for the requested representation.
    #[error("{what} supports at most {max} qubits, requested {requested}")]
    TooManyQubits {
        what: &'static str,
        max: usize,
        requested: usize,
    },

    /// A vector that must be normalised had (near-)zero norm.
    #[error("state has zero norm")]
    ZeroNorm,

    /// A circuit was evaluated with the wrong number of parameters.
    #[error("circuit expects {expected} parameters, got {actual}")]
    ParameterCount { expected: usize, actual: usize },

    /// A numerical argument was outside its valid domain.
    #[error("invalid argument `{name}`: {reason}")]
    InvalidArgument { name: &'static str, reason: String },

    /// A graph operation referenced a node that does not exist.
    #[error("node {node} out of range for a graph with {n_nodes} nodes")]
    NodeOutOfRange { node: usize, n_nodes: usize },

    /// An edge could not be added (self loop, duplicate, non-finite weight...).
    #[error("invalid edge ({u}, {v}): {reason}")]
    InvalidEdge { u: usize, v: usize, reason: String },

    /// A textual description (Pauli string, QASM...) could not be parsed.
    #[error("parse error: {0}")]
    Parse(String),

    /// An iterative method did not reach its tolerance.
    #[error("{method} did not converge within {iterations} iterations (residual {residual:.3e})")]
    NoConvergence {
        method: &'static str,
        iterations: usize,
        residual: f64,
    },
}

impl QsoError {
    pub(crate) fn invalid(name: &'static str, reason: impl Into<String>) -> Self {
        QsoError::InvalidArgument {
            name,
            reason: reason.into(),
        }
    }
}
