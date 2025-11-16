/*!
 * MEF Domains - Domain-Layer Extension for MEF-Core
 *
 * Implements Resonit/Resonat structures, MeshHolo triangulation,
 * and cross-domain homeomorphism with Metatron Cube integration.
 *
 * This layer enables domain-specific transformations while maintaining
 * the domain-agnostic nature of the core pipeline.
 */

pub mod adapter;
pub mod domain_layer;
pub mod infogenome;
pub mod meshholo;
pub mod resonat;
pub mod resonit;
pub mod xswap;

// Re-export main types
pub use adapter::{DomainAdapter, SignalDomainAdapter, TextDomainAdapter};
pub use domain_layer::{
    CrossDomainResult, DomainLayer, DomainMetrics, DomainProcessingResult, GateValidation,
};
pub use infogenome::{Infogene, Infogenome};
pub use meshholo::{EdgeData, MeshHolo, TopologicalInvariants, VertexData};
pub use resonat::{Resonat, ResonatMetrics};
pub use resonit::{Resonit, Sigma};
pub use xswap::{AlignmentArtifacts, Xswap};
