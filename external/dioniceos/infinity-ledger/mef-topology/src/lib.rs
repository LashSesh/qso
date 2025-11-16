/*!
 * MEF Topology Module
 *
 * Implements the Metatron Router for topological routing through the 13-node
 * Metatron Cube topology. Provides deterministic operator routing through
 * the S7 permutation space (5040 paths).
 */

pub mod metatron_router;

pub use metatron_router::{
    ConvergenceStep, MetatronRouter, OperatorType, ResonanceMetrics, RouteSpec,
    TransformationResult,
};
