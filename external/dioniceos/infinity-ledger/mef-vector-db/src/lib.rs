/*!
 * MEF Vector DB - Vector database abstraction and proof registry
 *
 * This module provides:
 * - Index management and persistence
 * - Merkle-tree based proof registry
 * - Vector database provider abstraction
 * - S3-backed manifest storage
 */

mod index_manager;
mod manifest_store;
mod proof_registry;
mod providers;

pub use index_manager::{CollectionState as IndexCollectionState, IndexManager, VectorRecord};
pub use manifest_store::{
    CollectionState as ManifestCollectionState, Manifest, ManifestStore, PersistenceConfig,
};
pub use proof_registry::{CollectionState, MembershipProof, ProofError, ProofRegistry};
pub use providers::{
    get_provider, get_providers, HNSWProvider, IVFPQProvider, IndexProvider, ProviderRegistry,
};

// Type aliases for NumPy compatibility
/// Float32 type (equivalent to np.float32)
pub type F32 = f32;

/// Unsigned 32-bit integer type (equivalent to np.uint32)
pub type U32 = u32;
