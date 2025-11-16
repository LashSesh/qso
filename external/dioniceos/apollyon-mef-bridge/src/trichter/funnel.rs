//! Funnel Graph - 4D kinetic compressor
//!
//! Manages the graph structure with nodes and edges,
//! implements Hebbian learning, decay, and split/merge/prune operations

use super::policies::PolicyParams;
use super::types::{FunnelEdge, FunnelNode, GuidanceVector, State5D};
use std::collections::HashMap;

/// Funnel Graph structure
#[derive(Debug)]
pub struct FunnelGraph {
    pub nodes: HashMap<usize, FunnelNode>,
    pub edges: Vec<FunnelEdge>,
    next_id: usize,
    current_time: f64,
}

impl FunnelGraph {
    /// Create new empty Funnel graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_id: 0,
            current_time: 0.0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, state: State5D) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, FunnelNode::new(id, state, self.current_time));
        id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if self.nodes.contains_key(&from) && self.nodes.contains_key(&to) {
            self.edges.push(FunnelEdge::new(from, to));
        }
    }

    /// Advect states with guidance vector
    ///
    /// Updates node states based on guidance field
    pub fn advect(&mut self, guidance: GuidanceVector, dt: f64, params: &PolicyParams) {
        // Update all node positions
        for node in self.nodes.values_mut() {
            node.state.x += guidance.vx * dt;
            node.state.y += guidance.vy * dt;
            node.state.z += guidance.vz * dt;
            node.state.psi += guidance.vpsi * dt;
        }

        // Update time
        self.current_time += dt;

        // Apply Hebbian learning and decay
        self.update_weights(params);

        // Apply structural changes
        self.split_nodes(params);
        self.merge_nodes(params);
        self.prune_edges(params);
    }

    /// Update edge weights using Hebbian learning
    ///
    /// w_ij(t+1) = w_ij(t) + α_hebb·phase_lock(i,j)·co_use(i,j) - decay·Δt
    fn update_weights(&mut self, params: &PolicyParams) {
        for edge in self.edges.iter_mut() {
            if let (Some(from_node), Some(to_node)) = 
                (self.nodes.get(&edge.from), self.nodes.get(&edge.to)) {
                
                // Compute phase lock based on omega coherence
                let phase_diff = (from_node.state.omega - to_node.state.omega).abs();
                edge.phase_lock = (-phase_diff).exp();

                // Co-use based on mass product
                let co_use = (from_node.mass * to_node.mass).sqrt();

                // Hebbian update
                let hebb_update = params.alpha_hebb * edge.phase_lock * co_use;
                
                // Apply update with decay
                edge.weight += hebb_update - params.decay;
                
                // Clamp to non-negative
                edge.weight = edge.weight.max(0.0);
            }
        }
    }

    /// Split high-mass, high-variance nodes
    ///
    /// If mass(i) > θ_split ∧ variance(i) high ⇒ split i into (i_a, i_b)
    fn split_nodes(&mut self, params: &PolicyParams) {
        let mut nodes_to_split = Vec::new();

        for node in self.nodes.values() {
            if node.mass > params.theta_split && node.variance > 0.5 {
                nodes_to_split.push((node.id, node.state, node.mass, node.variance));
            }
        }

        for (node_id, state, mass, variance) in nodes_to_split {
            // Create two new nodes with perturbed positions
            let mut state_a = state;
            let mut state_b = state;
            
            state_a.x += variance * 0.1;
            state_b.x -= variance * 0.1;

            let id_a = self.add_node(state_a);
            let id_b = self.add_node(state_b);

            // Transfer half the mass to each
            if let Some(node_a) = self.nodes.get_mut(&id_a) {
                node_a.mass = mass * 0.5;
                node_a.variance = variance * 0.7;
            }
            if let Some(node_b) = self.nodes.get_mut(&id_b) {
                node_b.mass = mass * 0.5;
                node_b.variance = variance * 0.7;
            }

            // Transfer edges
            let edges_to_copy: Vec<_> = self.edges.iter()
                .filter(|e| e.from == node_id || e.to == node_id)
                .cloned()
                .collect();

            for edge in edges_to_copy {
                if edge.from == node_id {
                    self.add_edge(id_a, edge.to);
                    self.add_edge(id_b, edge.to);
                }
                if edge.to == node_id {
                    self.add_edge(edge.from, id_a);
                    self.add_edge(edge.from, id_b);
                }
            }

            // Remove original node
            self.nodes.remove(&node_id);
            self.edges.retain(|e| e.from != node_id && e.to != node_id);
        }
    }

    /// Merge low-mass nearby nodes
    ///
    /// If mass(i), mass(j) < θ_merge ∧ dist(i,j) small ⇒ i ⊕ j
    fn merge_nodes(&mut self, params: &PolicyParams) {
        let mut merged = std::collections::HashSet::new();
        let node_ids: Vec<_> = self.nodes.keys().copied().collect();

        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let id_i = node_ids[i];
                let id_j = node_ids[j];

                if merged.contains(&id_i) || merged.contains(&id_j) {
                    continue;
                }

                // Clone data we need before borrowing mutably
                let merge_data = if let (Some(node_i), Some(node_j)) = 
                    (self.nodes.get(&id_i), self.nodes.get(&id_j)) {
                    
                    if node_i.mass < params.theta_merge && node_j.mass < params.theta_merge {
                        let dx = node_i.state.x - node_j.state.x;
                        let dy = node_i.state.y - node_j.state.y;
                        let dz = node_i.state.z - node_j.state.z;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        if dist_sq < 1.0 {
                            Some((
                                node_i.state.x, node_i.state.y, node_i.state.z,
                                node_j.state.x, node_j.state.y, node_j.state.z,
                                node_i.mass, node_j.mass,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((xi, yi, zi, xj, yj, zj, mass_i, mass_j)) = merge_data {
                    // Merge j into i
                    if let Some(node_i_mut) = self.nodes.get_mut(&id_i) {
                        let total_mass = mass_i + mass_j;
                        
                        // Weighted average of positions
                        node_i_mut.state.x = (xi * mass_i + xj * mass_j) / total_mass;
                        node_i_mut.state.y = (yi * mass_i + yj * mass_j) / total_mass;
                        node_i_mut.state.z = (zi * mass_i + zj * mass_j) / total_mass;
                        node_i_mut.mass = total_mass;
                    }

                    // Transfer edges from j to i
                    for edge in self.edges.iter_mut() {
                        if edge.from == id_j {
                            edge.from = id_i;
                        }
                        if edge.to == id_j {
                            edge.to = id_i;
                        }
                    }

                    // Remove node j
                    self.nodes.remove(&id_j);
                    merged.insert(id_j);
                }
            }
        }

        // Remove self-loops created by merging
        self.edges.retain(|e| e.from != e.to);
    }

    /// Prune low-weight edges
    ///
    /// If w_ij < θ_prune ⇒ remove edge
    fn prune_edges(&mut self, params: &PolicyParams) {
        self.edges.retain(|e| e.weight >= params.theta_prune);
    }

    /// Get current node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get current edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get current density (nodes per unit volume)
    pub fn density(&self) -> f64 {
        // Simplified: just return node count
        // In practice, would compute based on spatial extent
        self.nodes.len() as f64
    }
}

impl Default for FunnelGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funnel_creation() {
        let funnel = FunnelGraph::new();
        assert_eq!(funnel.node_count(), 0);
        assert_eq!(funnel.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut funnel = FunnelGraph::new();
        let state = State5D::zero();
        let id = funnel.add_node(state);
        
        assert_eq!(id, 0);
        assert_eq!(funnel.node_count(), 1);
    }

    #[test]
    fn test_add_edge() {
        let mut funnel = FunnelGraph::new();
        let id1 = funnel.add_node(State5D::zero());
        let id2 = funnel.add_node(State5D::zero());
        
        funnel.add_edge(id1, id2);
        assert_eq!(funnel.edge_count(), 1);
    }

    #[test]
    fn test_advect_updates_positions() {
        let mut funnel = FunnelGraph::new();
        let state = State5D::zero();
        funnel.add_node(state);
        
        let guidance = GuidanceVector::new(1.0, 0.0, 0.0, 0.0);
        let params = PolicyParams::explore();
        
        funnel.advect(guidance, 1.0, &params);
        
        let node = funnel.nodes.get(&0).unwrap();
        assert_eq!(node.state.x, 1.0);
    }

    #[test]
    fn test_prune_edges() {
        let mut funnel = FunnelGraph::new();
        let id1 = funnel.add_node(State5D::zero());
        let id2 = funnel.add_node(State5D::zero());
        
        funnel.add_edge(id1, id2);
        funnel.edges[0].weight = 0.001;
        
        let params = PolicyParams::exploit(); // Has higher prune threshold
        funnel.prune_edges(&params);
        
        assert_eq!(funnel.edge_count(), 0);
    }
}
