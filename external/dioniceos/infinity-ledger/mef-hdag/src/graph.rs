/*!
 * HDAG (Hyperdimensional Directed Acyclic Graph) implementation.
 * Couples linear event time with spiral phases while maintaining path invariance.
 */

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// HDAG metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HDAGMetadata {
    pub created: String,
    pub last_updated: String,
    pub version: String,
}

/// HDAG node structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HDAGNode {
    pub id: String,
    pub snapshot_id: String,
    pub phase: f64,
    pub time: String,
}

/// HDAG edge structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HDAGEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub cause: String,
}

/// Graph structure for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphData {
    nodes: HashMap<String, HDAGNode>,
    edges: HashMap<String, HDAGEdge>,
    metadata: HDAGMetadata,
}

/// Path invariance verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInvarianceResult {
    pub paths: Vec<Vec<String>>,
    pub weights: Vec<f64>,
    pub mean_weight: f64,
    pub std_weight: f64,
    pub invariant: bool,
}

/// HDAG - Hyperdimensional Directed Acyclic Graph
/// Couples linear time with spiral phases while maintaining path invariance
pub struct HDAG {
    #[allow(dead_code)]
    store_path: PathBuf,
    hdag_file: PathBuf,
    graph: GraphData,
    topological_order_cache: Option<Vec<String>>,
    path_invariants_cache: HashMap<String, PathInvarianceResult>,
}

impl HDAG {
    /// Initialize HDAG with storage path
    ///
    /// # Arguments
    /// * `store_path` - Storage directory path
    pub fn new(store_path: impl AsRef<Path>) -> Result<Self> {
        let store_path = store_path.as_ref().to_path_buf();
        fs::create_dir_all(&store_path)?;

        let hdag_file = store_path.join("hdag.json");
        let graph = Self::load_graph(&hdag_file)?;

        Ok(Self {
            store_path,
            hdag_file,
            graph,
            topological_order_cache: None,
            path_invariants_cache: HashMap::new(),
        })
    }

    /// Load HDAG from disk
    fn load_graph(hdag_file: &Path) -> Result<GraphData> {
        if hdag_file.exists() {
            let contents = fs::read_to_string(hdag_file)?;
            let graph: GraphData = serde_json::from_str(&contents)?;
            Ok(graph)
        } else {
            Ok(GraphData {
                nodes: HashMap::new(),
                edges: HashMap::new(),
                metadata: HDAGMetadata {
                    created: Utc::now().to_rfc3339(),
                    last_updated: Utc::now().to_rfc3339(),
                    version: "1.0.0".to_string(),
                },
            })
        }
    }

    /// Save HDAG to disk
    fn save_graph(&mut self) -> Result<()> {
        self.graph.metadata.last_updated = Utc::now().to_rfc3339();
        let contents = serde_json::to_string_pretty(&self.graph)?;
        fs::write(&self.hdag_file, contents)?;

        // Invalidate caches
        self.topological_order_cache = None;
        self.path_invariants_cache.clear();

        Ok(())
    }

    /// Create a new HDAG node
    ///
    /// # Arguments
    /// * `snapshot_id` - Associated snapshot UUID
    /// * `phase` - Spiral phase parameter
    /// * `timestamp` - Optional ISO timestamp (auto-generated if None)
    /// * `node_id` - Optional node ID (auto-generated if None)
    ///
    /// # Returns
    /// Node ID
    pub fn create_node(
        &mut self,
        snapshot_id: &str,
        phase: f64,
        timestamp: Option<String>,
        node_id: Option<String>,
    ) -> Result<String> {
        let node_id = node_id.unwrap_or_else(|| format!("N-{}", snapshot_id));

        let timestamp = timestamp.unwrap_or_else(|| Utc::now().to_rfc3339());

        let node = HDAGNode {
            id: node_id.clone(),
            snapshot_id: snapshot_id.to_string(),
            phase,
            time: timestamp,
        };

        self.graph.nodes.insert(node_id.clone(), node);
        self.save_graph()?;

        Ok(node_id)
    }

    /// Ensure node exists for snapshot
    fn ensure_node(&mut self, snapshot: &serde_json::Value) -> Result<String> {
        let node_id = snapshot
            .get("hdag_node")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(ref id) = node_id {
            if self.graph.nodes.contains_key(id) {
                return Ok(id.clone());
            }
        }

        let snapshot_id = snapshot
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Snapshot missing id"))?;
        let phase = snapshot
            .get("phase")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("Snapshot missing phase"))?;
        let timestamp = snapshot
            .get("timestamp")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.create_node(snapshot_id, phase, timestamp, node_id)
    }

    /// Create an edge between nodes
    ///
    /// # Arguments
    /// * `from_node` - Source node ID
    /// * `to_node` - Target node ID
    /// * `weight` - Edge weight (typically Phi value)
    /// * `cause` - Transformation cause
    ///
    /// # Returns
    /// Edge ID or None if edge would create cycle
    pub fn create_edge(
        &mut self,
        from_node: &str,
        to_node: &str,
        weight: f64,
        cause: &str,
    ) -> Result<Option<String>> {
        // Check nodes exist
        if !self.graph.nodes.contains_key(from_node) {
            return Err(anyhow!("Node not found: {}", from_node));
        }
        if !self.graph.nodes.contains_key(to_node) {
            return Err(anyhow!("Node not found: {}", to_node));
        }

        // Check temporal ordering
        let time_from = DateTime::parse_from_rfc3339(&self.graph.nodes[from_node].time)?;
        let time_to = DateTime::parse_from_rfc3339(&self.graph.nodes[to_node].time)?;

        if time_from >= time_to {
            return Err(anyhow!(
                "Invalid temporal ordering: {} >= {}",
                time_from,
                time_to
            ));
        }

        // Check for cycle
        if self.would_create_cycle(from_node, to_node) {
            return Ok(None);
        }

        let edge_id = format!("E-{}", Uuid::new_v4());

        let edge = HDAGEdge {
            id: edge_id.clone(),
            from: from_node.to_string(),
            to: to_node.to_string(),
            weight,
            cause: cause.to_string(),
        };

        self.graph.edges.insert(edge_id.clone(), edge);
        self.save_graph()?;

        Ok(Some(edge_id))
    }

    /// Check if adding edge would create a cycle
    fn would_create_cycle(&self, from_node: &str, to_node: &str) -> bool {
        // DFS to check if we can reach from_node from to_node
        let mut visited = HashSet::new();
        let mut stack = vec![to_node.to_string()];

        while let Some(current) = stack.pop() {
            if current == from_node {
                return true;
            }

            if visited.contains(&current) {
                continue;
            }

            visited.insert(current.clone());

            // Find outgoing edges
            for edge in self.graph.edges.values() {
                if edge.from == current {
                    stack.push(edge.to.clone());
                }
            }
        }

        false
    }

    /// Get topological ordering of nodes using Kahn's algorithm
    ///
    /// # Returns
    /// List of node IDs in topological order (empty if cycle detected)
    pub fn get_topological_order(&mut self) -> Vec<String> {
        if let Some(ref cache) = self.topological_order_cache {
            return cache.clone();
        }

        // Build adjacency list and in-degree map
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        for node_id in self.graph.nodes.keys() {
            adj_list.insert(node_id.clone(), Vec::new());
            in_degree.insert(node_id.clone(), 0);
        }

        for edge in self.graph.edges.values() {
            adj_list.get_mut(&edge.from).unwrap().push(edge.to.clone());
            *in_degree.get_mut(&edge.to).unwrap() += 1;
        }

        // Kahn's algorithm
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(node, _)| node.clone())
            .collect();

        let mut topo_order = Vec::new();

        while let Some(node) = queue.pop_front() {
            topo_order.push(node.clone());

            if let Some(neighbors) = adj_list.get(&node) {
                for neighbor in neighbors {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        if topo_order.len() != self.graph.nodes.len() {
            // Cycle detected
            return Vec::new();
        }

        self.topological_order_cache = Some(topo_order.clone());
        topo_order
    }

    /// Compute order parameter Phi between two nodes
    ///
    /// # Arguments
    /// * `node1_id` - First node ID
    /// * `node2_id` - Second node ID
    ///
    /// # Returns
    /// Phi value
    pub fn compute_phi(&self, node1_id: &str, node2_id: &str) -> f64 {
        let node1 = match self.graph.nodes.get(node1_id) {
            Some(n) => n,
            None => return 0.0,
        };
        let node2 = match self.graph.nodes.get(node2_id) {
            Some(n) => n,
            None => return 0.0,
        };

        // Phi based on phase difference and time difference
        let phase_diff = (node2.phase - node1.phase).abs();

        let time1 = DateTime::parse_from_rfc3339(&node1.time).unwrap();
        let time2 = DateTime::parse_from_rfc3339(&node2.time).unwrap();
        let time_diff = (time2 - time1).num_seconds().abs() as f64;

        // Normalize
        let phase_factor = (-phase_diff / (2.0 * std::f64::consts::PI)).exp();
        let time_factor = 1.0 / (1.0 + time_diff / 3600.0); // Hour scale

        phase_factor * time_factor
    }

    /// Update HDAG with new snapshot transition
    ///
    /// # Arguments
    /// * `snapshot_i` - Earlier snapshot
    /// * `snapshot_j` - Later snapshot
    ///
    /// # Returns
    /// Edge ID or None if update failed
    pub fn update_hdag(
        &mut self,
        snapshot_i: &serde_json::Value,
        snapshot_j: &serde_json::Value,
    ) -> Result<Option<String>> {
        // Create or find nodes first
        let node_i_id = self.ensure_node(snapshot_i)?;
        let node_j_id = self.ensure_node(snapshot_j)?;

        // Verify temporal ordering
        let time_i = snapshot_i
            .get("timestamp")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing timestamp in snapshot_i"))?;
        let time_j = snapshot_j
            .get("timestamp")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing timestamp in snapshot_j"))?;

        let time_i = DateTime::parse_from_rfc3339(time_i)?;
        let time_j = DateTime::parse_from_rfc3339(time_j)?;

        if time_i >= time_j {
            return Ok(None);
        }

        // Compute weight
        let weight = self.compute_phi(&node_i_id, &node_j_id);

        // Create edge
        self.create_edge(&node_i_id, &node_j_id, weight, "transform")
    }

    /// Verify path invariance between two nodes
    ///
    /// # Arguments
    /// * `start_node` - Starting node ID
    /// * `end_node` - Ending node ID
    ///
    /// # Returns
    /// Path invariance result
    pub fn verify_path_invariance(&self, start_node: &str, end_node: &str) -> PathInvarianceResult {
        // Find all paths from start to end
        let paths = self.find_all_paths(start_node, end_node, 10);

        if paths.is_empty() {
            return PathInvarianceResult {
                paths: Vec::new(),
                weights: Vec::new(),
                mean_weight: 0.0,
                std_weight: 0.0,
                invariant: false,
            };
        }

        // Compute path weights
        let path_weights: Vec<f64> = paths
            .iter()
            .map(|path| self.compute_path_weight(path))
            .collect();

        // Check invariance (all paths should have similar weight)
        let mean_weight = path_weights.iter().sum::<f64>() / path_weights.len() as f64;
        let variance: f64 = path_weights
            .iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>()
            / path_weights.len() as f64;
        let std_weight = variance.sqrt();

        // Invariance criterion: low coefficient of variation
        let is_invariant = std_weight / (mean_weight + 1e-10) < 0.1;

        PathInvarianceResult {
            paths,
            weights: path_weights,
            mean_weight,
            std_weight,
            invariant: is_invariant,
        }
    }

    /// Find all paths between two nodes (limited)
    fn find_all_paths(&self, start: &str, end: &str, max_paths: usize) -> Vec<Vec<String>> {
        if start == end {
            return vec![vec![start.to_string()]];
        }

        let mut paths = Vec::new();
        let mut stack = vec![(start.to_string(), vec![start.to_string()])];

        while let Some((node, path)) = stack.pop() {
            if paths.len() >= max_paths {
                break;
            }

            // Find outgoing edges
            for edge in self.graph.edges.values() {
                if edge.from == node {
                    let next_node = &edge.to;

                    if !path.contains(next_node) {
                        // Avoid cycles
                        let mut new_path = path.clone();
                        new_path.push(next_node.clone());

                        if next_node == end {
                            paths.push(new_path);
                        } else {
                            stack.push((next_node.clone(), new_path));
                        }
                    }
                }
            }
        }

        paths
    }

    /// Compute aggregate weight of a path
    fn compute_path_weight(&self, path: &[String]) -> f64 {
        if path.len() < 2 {
            return 0.0;
        }

        let mut total_weight = 0.0;

        for i in 0..path.len() - 1 {
            // Find edge between consecutive nodes
            for edge in self.graph.edges.values() {
                if edge.from == path[i] && edge.to == path[i + 1] {
                    total_weight += edge.weight;
                    break;
                }
            }
        }

        total_weight
    }

    /// Get all ancestor nodes
    ///
    /// # Arguments
    /// * `node_id` - Target node ID
    ///
    /// # Returns
    /// Set of ancestor node IDs
    pub fn get_node_ancestors(&self, node_id: &str) -> HashSet<String> {
        let mut ancestors = HashSet::new();
        let mut stack = vec![node_id.to_string()];

        while let Some(current) = stack.pop() {
            for edge in self.graph.edges.values() {
                if edge.to == current && !ancestors.contains(&edge.from) {
                    ancestors.insert(edge.from.clone());
                    stack.push(edge.from.clone());
                }
            }
        }

        ancestors
    }

    /// Get all descendant nodes
    ///
    /// # Arguments
    /// * `node_id` - Source node ID
    ///
    /// # Returns
    /// Set of descendant node IDs
    pub fn get_node_descendants(&self, node_id: &str) -> HashSet<String> {
        let mut descendants = HashSet::new();
        let mut stack = vec![node_id.to_string()];

        while let Some(current) = stack.pop() {
            for edge in self.graph.edges.values() {
                if edge.from == current && !descendants.contains(&edge.to) {
                    descendants.insert(edge.to.clone());
                    stack.push(edge.to.clone());
                }
            }
        }

        descendants
    }

    /// Get HDAG statistics
    ///
    /// # Returns
    /// Dictionary with graph metrics
    pub fn get_statistics(&mut self) -> serde_json::Value {
        let num_nodes = self.graph.nodes.len();
        let num_edges = self.graph.edges.len();

        // Compute connectivity
        let connectivity = if num_nodes > 0 {
            let max_edges = num_nodes * (num_nodes - 1) / 2;
            if max_edges > 0 {
                num_edges as f64 / max_edges as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Check for cycles
        let topo_order = self.get_topological_order();
        let is_acyclic = topo_order.len() == num_nodes;

        serde_json::json!({
            "nodes": num_nodes,
            "edges": num_edges,
            "connectivity": connectivity,
            "is_acyclic": is_acyclic,
            "metadata": self.graph.metadata
        })
    }

    /// Export graph in specified format
    ///
    /// # Arguments
    /// * `format` - Export format ("json" or "dot")
    ///
    /// # Returns
    /// Exported graph string
    pub fn export_graph(&self, format: &str) -> Result<String> {
        match format {
            "json" => Ok(serde_json::to_string_pretty(&self.graph)?),
            "dot" => {
                let mut lines = vec!["digraph HDAG {".to_string()];

                // Add nodes
                for (node_id, node) in &self.graph.nodes {
                    let label = format!("{}\\nφ={:.2}", node_id, node.phase);
                    lines.push(format!("  \"{}\" [label=\"{}\"];", node_id, label));
                }

                // Add edges
                for edge in self.graph.edges.values() {
                    let weight_label = format!("{:.2}", edge.weight);
                    lines.push(format!(
                        "  \"{}\" -> \"{}\" [label=\"{}\"];",
                        edge.from, edge.to, weight_label
                    ));
                }

                lines.push("}".to_string());
                Ok(lines.join("\n"))
            }
            _ => Err(anyhow!("Unsupported format: {}", format)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_create_node() {
        let temp_dir = env::temp_dir().join("test_hdag_create_node");
        let mut hdag = HDAG::new(&temp_dir).unwrap();

        let node_id = hdag.create_node("snap-001", 1.5, None, None).unwrap();
        assert!(node_id.starts_with("N-snap-001"));

        // Verify node exists
        assert!(hdag.graph.nodes.contains_key(&node_id));
        let node = &hdag.graph.nodes[&node_id];
        assert_eq!(node.snapshot_id, "snap-001");
        assert_eq!(node.phase, 1.5);
    }

    #[test]
    fn test_create_edge() {
        let temp_dir = env::temp_dir().join("test_hdag_create_edge");
        let mut hdag = HDAG::new(&temp_dir).unwrap();

        // Create two nodes with different timestamps
        let node1 = hdag
            .create_node(
                "snap-001",
                1.0,
                Some("2025-01-01T00:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node2 = hdag
            .create_node(
                "snap-002",
                2.0,
                Some("2025-01-01T01:00:00Z".to_string()),
                None,
            )
            .unwrap();

        // Create edge
        let edge_id = hdag.create_edge(&node1, &node2, 0.5, "transform").unwrap();
        assert!(edge_id.is_some());

        let edge_id = edge_id.unwrap();
        assert!(hdag.graph.edges.contains_key(&edge_id));
    }

    #[test]
    fn test_cycle_detection() {
        let temp_dir = env::temp_dir().join("test_hdag_cycle");
        let mut hdag = HDAG::new(&temp_dir).unwrap();

        let node1 = hdag
            .create_node(
                "snap-001",
                1.0,
                Some("2025-01-01T00:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node2 = hdag
            .create_node(
                "snap-002",
                2.0,
                Some("2025-01-01T01:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node3 = hdag
            .create_node(
                "snap-003",
                3.0,
                Some("2025-01-01T02:00:00Z".to_string()),
                None,
            )
            .unwrap();

        // Create edges: 1 -> 2 -> 3
        hdag.create_edge(&node1, &node2, 0.5, "transform").unwrap();
        hdag.create_edge(&node2, &node3, 0.5, "transform").unwrap();

        // Try to create edge 3 -> 1 (would create cycle)
        // This should fail due to temporal ordering
        let result = hdag.create_edge(&node3, &node1, 0.5, "transform");
        assert!(result.is_err());
    }

    #[test]
    fn test_topological_order() {
        let temp_dir = env::temp_dir().join("test_hdag_topo");
        let mut hdag = HDAG::new(&temp_dir).unwrap();

        let node1 = hdag
            .create_node(
                "snap-001",
                1.0,
                Some("2025-01-01T00:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node2 = hdag
            .create_node(
                "snap-002",
                2.0,
                Some("2025-01-01T01:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node3 = hdag
            .create_node(
                "snap-003",
                3.0,
                Some("2025-01-01T02:00:00Z".to_string()),
                None,
            )
            .unwrap();

        hdag.create_edge(&node1, &node2, 0.5, "transform").unwrap();
        hdag.create_edge(&node2, &node3, 0.5, "transform").unwrap();

        let topo_order = hdag.get_topological_order();
        assert_eq!(topo_order.len(), 3);

        // node1 should come before node2, node2 before node3
        let pos1 = topo_order.iter().position(|n| n == &node1).unwrap();
        let pos2 = topo_order.iter().position(|n| n == &node2).unwrap();
        let pos3 = topo_order.iter().position(|n| n == &node3).unwrap();
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_compute_phi() {
        let temp_dir = env::temp_dir().join("test_hdag_phi");
        let mut hdag = HDAG::new(&temp_dir).unwrap();

        let node1 = hdag
            .create_node(
                "snap-001",
                0.0,
                Some("2025-01-01T00:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node2 = hdag
            .create_node(
                "snap-002",
                1.0,
                Some("2025-01-01T01:00:00Z".to_string()),
                None,
            )
            .unwrap();

        let phi = hdag.compute_phi(&node1, &node2);
        assert!(phi > 0.0);
        assert!(phi <= 1.0);
    }

    #[test]
    fn test_statistics() {
        let temp_dir = env::temp_dir().join("test_hdag_stats");
        fs::remove_dir_all(&temp_dir).ok(); // Clean up old test data
        let mut hdag = HDAG::new(&temp_dir).unwrap();

        let node1 = hdag
            .create_node(
                "snap-001",
                1.0,
                Some("2025-01-01T00:00:00Z".to_string()),
                None,
            )
            .unwrap();
        let node2 = hdag
            .create_node(
                "snap-002",
                2.0,
                Some("2025-01-01T01:00:00Z".to_string()),
                None,
            )
            .unwrap();

        hdag.create_edge(&node1, &node2, 0.5, "transform").unwrap();

        let stats = hdag.get_statistics();
        assert_eq!(stats["nodes"], 2);
        assert_eq!(stats["edges"], 1);
        assert_eq!(stats["is_acyclic"], true);
    }
}
