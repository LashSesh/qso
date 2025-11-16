/*!
 * Geometry Module - Metatron Cube Canonical Definitions
 *
 * This module provides the canonical geometric definition of the Metatron Cube
 * as described in the blueprint. It defines the 13 nodes (one center, six
 * hexagon vertices and six cube corners) and their explicit 3D coordinates.
 *
 * The aim of this module is twofold:
 *
 * 1. Provide a single source of truth for the canonical node list used by
 *    graph-based modules.
 * 2. Offer simple utility functions such as computing pairwise distances or
 *    looking up nodes by label or index.
 *
 * The node coordinates follow the convention laid out in the "Complete
 * Canonical Node Table" of the blueprint. The six cube corners deliberately
 * omit the two negative-negative combinations to match the 13-node structure
 * used throughout the prototype.
 */

use ndarray::Array1;

/// Represents a single node in the Metatron Cube
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// 1-based index of the node as defined in the canonical list
    pub index: usize,
    /// Symbolic name of the node (e.g. "C", "H1", "Q1")
    pub label: String,
    /// High-level category of the node ("center", "hexagon" or "cube")
    pub node_type: String,
    /// Cartesian coordinates of the node in ℝ³
    pub coords: (f64, f64, f64),
}

impl Node {
    /// Create a new Node
    pub fn new(index: usize, label: &str, node_type: &str, coords: (f64, f64, f64)) -> Self {
        Self {
            index,
            label: label.to_string(),
            node_type: node_type.to_string(),
            coords,
        }
    }

    /// Return the coordinates as an ndarray Array1
    pub fn as_array(&self) -> Array1<f64> {
        Array1::from_vec(vec![self.coords.0, self.coords.1, self.coords.2])
    }

    /// Compute the Euclidean distance to another node
    pub fn distance_to(&self, other: &Node) -> f64 {
        let dx = self.coords.0 - other.coords.0;
        let dy = self.coords.1 - other.coords.1;
        let dz = self.coords.2 - other.coords.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Return the canonical list of Metatron Cube nodes
///
/// The 13 nodes are defined exactly as in Table 2 of the blueprint.
/// Node indices start at 1. For convenience, the coordinates are given as
/// plain Rust tuples, but all arithmetic is performed using ndarray.
pub fn canonical_nodes() -> Vec<Node> {
    let sqrt3 = 3.0_f64.sqrt();

    vec![
        // Center
        Node::new(1, "C", "center", (0.0, 0.0, 0.0)),
        // Hexagon vertices around the center in the xy-plane
        Node::new(2, "H1", "hexagon", (1.0, 0.0, 0.0)),
        Node::new(3, "H2", "hexagon", (0.5, sqrt3 / 2.0, 0.0)),
        Node::new(4, "H3", "hexagon", (-0.5, sqrt3 / 2.0, 0.0)),
        Node::new(5, "H4", "hexagon", (-1.0, 0.0, 0.0)),
        Node::new(6, "H5", "hexagon", (-0.5, -sqrt3 / 2.0, 0.0)),
        Node::new(7, "H6", "hexagon", (0.5, -sqrt3 / 2.0, 0.0)),
        // Cube corners (a subset of the full cube; missing two negative-negative
        // corners by design)
        Node::new(8, "Q1", "cube", (0.5, 0.5, 0.5)),
        Node::new(9, "Q2", "cube", (0.5, 0.5, -0.5)),
        Node::new(10, "Q3", "cube", (0.5, -0.5, 0.5)),
        Node::new(11, "Q4", "cube", (0.5, -0.5, -0.5)),
        Node::new(12, "Q5", "cube", (-0.5, 0.5, 0.5)),
        Node::new(13, "Q6", "cube", (-0.5, 0.5, -0.5)),
    ]
}

/// Return the canonical list of Metatron Cube nodes
///
/// Alias for backwards compatibility with earlier versions of the prototype
/// and existing unit tests.
pub fn get_metatron_nodes() -> Vec<Node> {
    canonical_nodes()
}

/// Return the canonical edge list for the Metatron Cube
///
/// The returned list contains pairs of node indices (1-based) representing
/// undirected edges. It follows the partial enumeration in the blueprint.
/// This covers the center-hexagon edges, the hexagon cycle, and a selection
/// of cube edges and diagonals.
pub fn canonical_edges() -> Vec<(usize, usize)> {
    vec![
        // Base edges from the blueprint: center to hexagon
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        // Hexagon cycle
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 2),
        // Cube edges (one face): Q1-Q2-Q4-Q3-Q1
        (8, 9),
        (9, 11),
        (11, 10),
        (10, 8),
        // Additional cube cross-edges and diagonals
        (8, 12),
        (9, 13),
        (10, 12),
        (11, 13),
        (12, 13),
        // Face diagonals
        (8, 10),
        (9, 11),
    ]
}

/// Return the exhaustive edge list for the Metatron Cube
///
/// While canonical_edges returns a minimal subset of edges for clarity,
/// the full Metatron Cube embeds all lines connecting the 13 nodes.
/// This function enumerates all C(13, 2) = 78 undirected edges, yielding
/// a complete graph on 13 nodes (K_13).
pub fn complete_canonical_edges() -> Vec<(usize, usize)> {
    full_edge_list(13)
}

/// Return the canonical edge list for the Metatron Cube
///
/// Alias for backwards compatibility. If full is true, returns the
/// exhaustive edge list via complete_canonical_edges.
pub fn get_metatron_edges(full: bool) -> Vec<(usize, usize)> {
    if full {
        complete_canonical_edges()
    } else {
        canonical_edges()
    }
}

/// Return the complete set of undirected edges for n nodes
///
/// This convenience function generates all pairs (i, j) with 1 <= i < j <= n.
/// It can be used to build a fully connected Metatron Cube graph when a
/// maximal connectivity is desired.
pub fn full_edge_list(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            edges.push((i, j));
        }
    }
    edges
}

/// Find a node by its label or index
///
/// Exactly one of label or index must be provided. If a node is not found,
/// None is returned.
pub fn find_node(nodes: &[Node], label: Option<&str>, index: Option<usize>) -> Option<Node> {
    if (label.is_none()) == (index.is_none()) {
        return None; // Must specify exactly one
    }

    for node in nodes {
        if let Some(lbl) = label {
            if node.label == lbl {
                return Some(node.clone());
            }
        }
        if let Some(idx) = index {
            if node.index == idx {
                return Some(node.clone());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_nodes_count() {
        let nodes = canonical_nodes();
        assert_eq!(nodes.len(), 13);
    }

    #[test]
    fn test_center_node() {
        let nodes = canonical_nodes();
        let center = &nodes[0];
        assert_eq!(center.index, 1);
        assert_eq!(center.label, "C");
        assert_eq!(center.node_type, "center");
        assert_eq!(center.coords, (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_hexagon_nodes() {
        let nodes = canonical_nodes();
        let hexagon_nodes: Vec<&Node> = nodes.iter().filter(|n| n.node_type == "hexagon").collect();
        assert_eq!(hexagon_nodes.len(), 6);
    }

    #[test]
    fn test_cube_nodes() {
        let nodes = canonical_nodes();
        let cube_nodes: Vec<&Node> = nodes.iter().filter(|n| n.node_type == "cube").collect();
        assert_eq!(cube_nodes.len(), 6);
    }

    #[test]
    fn test_node_distance() {
        let nodes = canonical_nodes();
        let center = &nodes[0]; // C
        let h1 = &nodes[1]; // H1

        // Distance from center to H1 should be 1.0
        let dist = center.distance_to(h1);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonical_edges_count() {
        let edges = canonical_edges();
        assert_eq!(edges.len(), 23);
    }

    #[test]
    fn test_complete_edges_count() {
        let edges = complete_canonical_edges();
        // C(13, 2) = 13 * 12 / 2 = 78
        assert_eq!(edges.len(), 78);
    }

    #[test]
    fn test_find_node_by_label() {
        let nodes = canonical_nodes();
        let node = find_node(&nodes, Some("C"), None);
        assert!(node.is_some());
        assert_eq!(node.unwrap().index, 1);
    }

    #[test]
    fn test_find_node_by_index() {
        let nodes = canonical_nodes();
        let node = find_node(&nodes, None, Some(2));
        assert!(node.is_some());
        assert_eq!(node.unwrap().label, "H1");
    }

    #[test]
    fn test_find_node_invalid() {
        let nodes = canonical_nodes();
        // Both None - invalid
        let node = find_node(&nodes, None, None);
        assert!(node.is_none());

        // Both Some - invalid
        let node = find_node(&nodes, Some("C"), Some(1));
        assert!(node.is_none());
    }

    #[test]
    fn test_get_metatron_nodes_alias() {
        let nodes1 = canonical_nodes();
        let nodes2 = get_metatron_nodes();
        assert_eq!(nodes1.len(), nodes2.len());
    }

    #[test]
    fn test_get_metatron_edges_alias() {
        let edges1 = canonical_edges();
        let edges2 = get_metatron_edges(false);
        assert_eq!(edges1.len(), edges2.len());

        let edges3 = complete_canonical_edges();
        let edges4 = get_metatron_edges(true);
        assert_eq!(edges3.len(), edges4.len());
    }

    #[test]
    fn test_full_edge_list() {
        let edges = full_edge_list(5);
        // C(5, 2) = 5 * 4 / 2 = 10
        assert_eq!(edges.len(), 10);

        // Check first and last edges
        assert_eq!(edges[0], (1, 2));
        assert_eq!(edges[edges.len() - 1], (4, 5));
    }

    #[test]
    fn test_node_as_array() {
        let nodes = canonical_nodes();
        let h1 = &nodes[1]; // H1 at (1.0, 0.0, 0.0)
        let arr = h1.as_array();
        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 1.0).abs() < 1e-10);
        assert!((arr[1] - 0.0).abs() < 1e-10);
        assert!((arr[2] - 0.0).abs() < 1e-10);
    }
}
