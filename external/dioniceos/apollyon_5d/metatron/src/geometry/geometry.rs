use serde::Serialize;

use crate::error::{EngineError, EngineResult};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Node {
    pub index: usize,
    pub label: &'static str,
    pub node_type: &'static str,
    pub coords: [f64; 3],
}

impl Node {
    pub fn as_array(&self) -> [f64; 3] {
        self.coords
    }

    pub fn distance_to(&self, other: &Node) -> f64 {
        let diff = [
            self.coords[0] - other.coords[0],
            self.coords[1] - other.coords[1],
            self.coords[2] - other.coords[2],
        ];
        (diff[0].powi(2) + diff[1].powi(2) + diff[2].powi(2)).sqrt()
    }
}

pub fn canonical_nodes() -> Vec<Node> {
    let sqrt3 = 3f64.sqrt();
    vec![
        Node {
            index: 1,
            label: "C",
            node_type: "center",
            coords: [0.0, 0.0, 0.0],
        },
        Node {
            index: 2,
            label: "H1",
            node_type: "hexagon",
            coords: [1.0, 0.0, 0.0],
        },
        Node {
            index: 3,
            label: "H2",
            node_type: "hexagon",
            coords: [0.5, sqrt3 / 2.0, 0.0],
        },
        Node {
            index: 4,
            label: "H3",
            node_type: "hexagon",
            coords: [-0.5, sqrt3 / 2.0, 0.0],
        },
        Node {
            index: 5,
            label: "H4",
            node_type: "hexagon",
            coords: [-1.0, 0.0, 0.0],
        },
        Node {
            index: 6,
            label: "H5",
            node_type: "hexagon",
            coords: [-0.5, -sqrt3 / 2.0, 0.0],
        },
        Node {
            index: 7,
            label: "H6",
            node_type: "hexagon",
            coords: [0.5, -sqrt3 / 2.0, 0.0],
        },
        Node {
            index: 8,
            label: "Q1",
            node_type: "cube",
            coords: [0.5, 0.5, 0.5],
        },
        Node {
            index: 9,
            label: "Q2",
            node_type: "cube",
            coords: [0.5, 0.5, -0.5],
        },
        Node {
            index: 10,
            label: "Q3",
            node_type: "cube",
            coords: [0.5, -0.5, 0.5],
        },
        Node {
            index: 11,
            label: "Q4",
            node_type: "cube",
            coords: [0.5, -0.5, -0.5],
        },
        Node {
            index: 12,
            label: "Q5",
            node_type: "cube",
            coords: [-0.5, 0.5, 0.5],
        },
        Node {
            index: 13,
            label: "Q6",
            node_type: "cube",
            coords: [-0.5, 0.5, -0.5],
        },
    ]
}

pub fn get_metatron_nodes() -> Vec<Node> {
    canonical_nodes()
}

pub fn canonical_edges() -> Vec<(usize, usize)> {
    vec![
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 2),
        (8, 9),
        (9, 11),
        (11, 10),
        (10, 8),
        (8, 12),
        (9, 13),
        (10, 12),
        (11, 13),
        (12, 13),
        (8, 10),
        (9, 11),
    ]
}

pub fn complete_canonical_edges() -> Vec<(usize, usize)> {
    full_edge_list(13)
}

pub fn full_edge_list(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            edges.push((i, j));
        }
    }
    edges
}

pub fn get_metatron_edges(full: bool) -> Vec<(usize, usize)> {
    if full {
        complete_canonical_edges()
    } else {
        canonical_edges()
    }
}

pub fn find_node<'a>(
    nodes: &'a [Node],
    label: Option<&str>,
    index: Option<usize>,
) -> EngineResult<&'a Node> {
    match (label, index) {
        (Some(_), Some(_)) | (None, None) => Err(EngineError::Misconfigured(
            "Specify exactly one of label or index".into(),
        )),
        (Some(lbl), None) => {
            nodes
                .iter()
                .find(|n| n.label == lbl)
                .ok_or_else(|| EngineError::UnknownNodeLabel {
                    label: lbl.to_string(),
                })
        }
        (None, Some(idx)) => {
            nodes
                .iter()
                .find(|n| n.index == idx)
                .ok_or_else(|| EngineError::InvalidNodeIndex {
                    index: idx,
                    len: nodes.len(),
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn node_count() {
        assert_eq!(canonical_nodes().len(), 13);
    }

    #[test]
    fn distances() {
        let nodes = canonical_nodes();
        let d = nodes[0].distance_to(&nodes[1]);
        assert_abs_diff_eq!(d, 1.0, epsilon = 1e-8);
    }
}
