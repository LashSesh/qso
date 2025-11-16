// Geometry module - Metatron Cube structure and operations

pub mod cube;
pub mod geometry;
pub mod graph;
pub mod operators;

pub use cube::MetatronCube;
pub use geometry::{
    canonical_edges, canonical_nodes, complete_canonical_edges, find_node, full_edge_list,
    get_metatron_edges, get_metatron_nodes, Node,
};
pub use graph::MetatronCubeGraph;
pub use operators::{
    apply_permutation_to_adjacency, generate_c6_subgroup, generate_d6_subgroup, hexagon_reflection,
    hexagon_rotation, permutation_matrix, permutation_to_matrix,
};
