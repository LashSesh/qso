use serde::Serialize;
use serde_json::Value;

use crate::error::EngineResult;
use crate::geometry::graph::MetatronCubeGraph;
use crate::geometry::operators::permutation_matrix;

#[derive(Serialize)]
struct NodeRecord {
    index: usize,
    label: &'static str,
    #[serde(rename = "type")]
    node_type: &'static str,
    coords: [f64; 3],
}

#[derive(Serialize)]
struct EdgeRecord {
    source: usize,
    target: usize,
}

pub fn export_nodes_json(graph: &MetatronCubeGraph) -> EngineResult<String> {
    let nodes: Vec<NodeRecord> = graph
        .nodes
        .iter()
        .map(|n| NodeRecord {
            index: n.index,
            label: n.label,
            node_type: n.node_type,
            coords: n.coords,
        })
        .collect();
    Ok(serde_json::to_string_pretty(&nodes)?)
}

pub fn export_edges_json(graph: &MetatronCubeGraph) -> EngineResult<String> {
    let edges: Vec<EdgeRecord> = graph
        .get_edge_list()
        .into_iter()
        .map(|(i, j)| EdgeRecord {
            source: i,
            target: j,
        })
        .collect();
    Ok(serde_json::to_string_pretty(&edges)?)
}

pub fn export_adjacency_json(graph: &MetatronCubeGraph) -> EngineResult<String> {
    let matrix = graph.get_adjacency_matrix();
    Ok(serde_json::to_string_pretty(&matrix)?)
}

pub fn export_group_json(perms: &[Vec<usize>]) -> EngineResult<String> {
    let data: Vec<Vec<usize>> = perms.iter().map(|p| p.clone()).collect();
    Ok(serde_json::to_string_pretty(&data)?)
}

pub fn export_matrices_json(perms: &[Vec<usize>], size: usize) -> EngineResult<String> {
    let mats: Vec<Vec<Vec<f64>>> = perms.iter().map(|p| permutation_matrix(p, size)).collect();
    Ok(serde_json::to_string_pretty(&mats)?)
}

pub fn parse_json(data: &str) -> EngineResult<Value> {
    Ok(serde_json::from_str(data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_nodes() {
        let graph = MetatronCubeGraph::new();
        let js = export_nodes_json(&graph).unwrap();
        let value: Value = serde_json::from_str(&js).unwrap();
        assert!(value.is_array());
        assert_eq!(value.as_array().unwrap().len(), 13);
    }
}
