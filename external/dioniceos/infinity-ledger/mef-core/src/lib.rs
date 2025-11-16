/*!
 * MEF-Core Utilities
 *
 * Core utility modules providing foundational functionality across the MEF workspace.
 */

pub mod cube;
pub mod field_vector;
pub mod gabriel_cell;
pub mod gates;
pub mod geometry;
pub mod graph;
pub mod mandorla;
pub mod mef_pipeline;
pub mod qdash_agent;
pub mod qlogic;
pub mod quantum;
pub mod resonance_tensor;
pub mod spiral_memory;
pub mod symmetries;

pub use cube::{EdgeInfo, MetatronCube, NodeInfo, OperatorInfo};
pub use field_vector::FieldVector;
pub use gabriel_cell::{couple_cells, neighbor_feedback, GabrielCell};
pub use gates::{
    validate_gate_event, GateChecks, GateDecision, GateEvent, MerkabaGate, TICCandidate,
};
pub use geometry::{
    canonical_edges, canonical_nodes, complete_canonical_edges, find_node, get_metatron_edges,
    get_metatron_nodes, Node,
};
pub use graph::MetatronCubeGraph;
pub use mandorla::MandorlaField;
pub use mef_pipeline::{MEFCore, MEFCoreConfig, ProcessingResult};
pub use qdash_agent::{QDASHAgent, QDASHResult};
pub use qlogic::{
    EntropyAnalyzer, QLOGICOscillatorCore, QLogicEngine, QLogicStepResult, SpectralGrammar,
};
pub use quantum::{QuantumOperator, QuantumState};
pub use resonance_tensor::ResonanceTensorField;
pub use spiral_memory::SpiralMemory;
pub use symmetries::{
    apply_permutation_to_adjacency, generate_alternating_group, generate_c6_subgroup,
    generate_d6_subgroup, generate_s7_permutations, generate_symmetric_group, hexagon_reflection,
    hexagon_rotation, permutation_matrix, permutation_to_matrix,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
