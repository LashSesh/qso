use std::collections::HashMap;

use crate::error::EngineResult;
use crate::geometry::graph::MetatronCubeGraph;
use crate::geometry::operators::{generate_c6_subgroup, generate_d6_subgroup, permutation_matrix};
use crate::geometry::{canonical_edges, canonical_nodes, Node};

#[derive(Debug, Clone)]
pub struct MetatronCube {
    pub graph: MetatronCubeGraph,
    pub operators: HashMap<String, Vec<usize>>,
    solid_sets: HashMap<&'static str, Vec<Vec<usize>>>,
}

impl MetatronCube {
    pub fn new(full_edges: bool) -> EngineResult<Self> {
        let nodes = canonical_nodes();
        let edges = if full_edges {
            crate::geometry::complete_canonical_edges()
        } else {
            canonical_edges()
        };
        let graph = MetatronCubeGraph::with_edges(Some(nodes.clone()), Some(edges));
        let mut cube = Self {
            graph,
            operators: HashMap::new(),
            solid_sets: default_solid_sets(),
        };
        cube.register_basic_groups()?;
        Ok(cube)
    }

    fn register_basic_groups(&mut self) -> EngineResult<()> {
        for (idx, perm) in generate_c6_subgroup().into_iter().enumerate() {
            self.operators
                .insert(format!("C6_rot_{}", idx * 60), extend_permutation(&perm));
        }
        let reflections = generate_d6_subgroup()?;
        for (idx, perm) in reflections.into_iter().enumerate().skip(6) {
            self.operators
                .insert(format!("D6_ref_H{}", idx - 4), extend_permutation(&perm));
        }
        Ok(())
    }

    pub fn register_operator(&mut self, name: &str, permutation: Vec<usize>) {
        self.operators.insert(name.to_string(), permutation);
    }

    pub fn apply_operator(&self, name: &str) -> EngineResult<Option<MetatronCubeGraph>> {
        let sigma = match self.operators.get(name) {
            Some(s) => s,
            None => return Ok(None),
        };
        let matrix = permutation_matrix(sigma, self.graph.nodes.len());
        Ok(Some(self.graph.apply_permutation_matrix(&matrix)?))
    }

    pub fn list_nodes_by_type(&self, node_type: &str) -> Vec<&Node> {
        self.graph
            .nodes
            .iter()
            .filter(|node| node.node_type == node_type)
            .collect()
    }

    pub fn solid_members(&self, solid: &str) -> Option<Vec<&Node>> {
        let sets = self.solid_sets.get(solid)?;
        let mut nodes = Vec::new();
        for subset in sets {
            for idx in subset {
                if let Some(node) = self.graph.nodes.iter().find(|n| n.index == *idx) {
                    nodes.push(node);
                }
            }
        }
        Some(nodes)
    }
}

fn extend_permutation(perm7: &[usize]) -> Vec<usize> {
    let mut sigma = perm7.to_vec();
    if sigma.len() < 13 {
        for idx in (sigma.len() + 1)..=13 {
            sigma.push(idx);
        }
    }
    sigma
}

fn default_solid_sets() -> HashMap<&'static str, Vec<Vec<usize>>> {
    let mut map = HashMap::new();
    map.insert("tetrahedron", vec![vec![2, 4, 6, 8], vec![3, 5, 7, 9]]);
    map.insert("cube", vec![vec![8, 9, 10, 11, 12, 13]]);
    map.insert("octahedron", vec![vec![2, 3, 4, 5, 6, 7]]);
    map.insert(
        "icosahedron",
        vec![vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
    );
    map.insert(
        "dodecahedron",
        vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]],
    );
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solids() {
        let cube = MetatronCube::new(false).unwrap();
        let nodes = cube.solid_members("cube").unwrap();
        assert!(!nodes.is_empty());
    }
}
