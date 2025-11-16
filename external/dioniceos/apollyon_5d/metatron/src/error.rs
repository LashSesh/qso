use thiserror::Error;

/// Result type used throughout the Metatron engine.
pub type EngineResult<T> = Result<T, EngineError>;

/// Error variants produced by the Metatron engine when validation fails or
/// dependent systems return failures.
#[derive(Debug, Error)]
pub enum EngineError {
    /// A graph node index was requested that does not exist in the canonical cube.
    #[error("node index {index} out of bounds for cube of size {len}")]
    InvalidNodeIndex { index: usize, len: usize },

    /// A node label could not be found in the cube definition.
    #[error("unknown node label '{label}'")]
    UnknownNodeLabel { label: String },

    /// A permutation vector was malformed.
    #[error("permutation must contain each index exactly once")]
    InvalidPermutation,

    /// An operation expected vectors with matching dimensionality.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// The engine was configured with invalid parameters.
    #[error("engine misconfiguration: {0}")]
    Misconfigured(String),

    /// Wrapper for serde serialisation/deserialisation problems.
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}
