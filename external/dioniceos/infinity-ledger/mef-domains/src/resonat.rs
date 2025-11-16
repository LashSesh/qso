/*!
 * Resonat - Cluster of Resonits forming topologically stable structure
 *
 * Resonats are validated through Betti vectors and persistence metrics,
 * ensuring they represent coherent information gestalts.
 */

use crate::resonit::Resonit;
use anyhow::Result;
use petgraph::algo::connected_components;
use petgraph::graph::Graph;
use petgraph::Undirected;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Cluster of Resonits forming topologically stable structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resonat {
    /// Unique identifier
    pub id: String,
    /// Constituent Resonits
    pub resonits: Vec<Resonit>,
    /// Topological and stability metrics
    pub metrics: ResonatMetrics,
    /// Optional centroid in sigma space
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid: Option<Vec<f64>>,
    /// Optional topology description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<HashMap<String, serde_json::Value>>,
}

/// Metrics for Resonat validation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResonatMetrics {
    /// Betti numbers [β0, β1, β2, ...]
    pub betti: Vec<usize>,
    /// Topological persistence score [0, 1]
    pub persistence: f64,
    /// Structural stability [0, 1]
    pub stability: f64,
    /// Number of constituent Resonits
    pub size: usize,
}

impl Resonat {
    /// Create a new Resonat from Resonits
    pub fn new(resonits: Vec<Resonit>) -> Result<Self> {
        if resonits.is_empty() {
            anyhow::bail!("Cannot create Resonat from empty Resonit list");
        }

        let centroid = Self::calculate_centroid(&resonits);
        let betti = Self::calculate_betti_vectors(&resonits);
        let persistence = Self::calculate_persistence(&betti);
        let stability = Self::calculate_stability(&resonits);

        let metrics = ResonatMetrics {
            betti,
            persistence,
            stability,
            size: resonits.len(),
        };

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            resonits,
            metrics,
            centroid: Some(centroid),
            topology: None,
        })
    }

    /// Calculate centroid in sigma space
    fn calculate_centroid(resonits: &[Resonit]) -> Vec<f64> {
        let n = resonits.len() as f64;
        let mut centroid = vec![0.0, 0.0, 0.0];

        for resonit in resonits {
            let v = resonit.to_vector();
            for (i, val) in v.iter().enumerate() {
                centroid[i] += val / n;
            }
        }

        centroid
    }

    /// Calculate Betti numbers for the Resonat topology
    ///
    /// Returns:
    ///     List of Betti numbers [β0, β1, β2, ...]
    pub fn calculate_betti_vectors(resonits: &[Resonit]) -> Vec<usize> {
        if resonits.len() < 2 {
            return vec![1, 0, 0];
        }

        // Create graph from Resonit connections
        let mut graph: Graph<usize, f64, Undirected> = Graph::new_undirected();
        let mut nodes = Vec::new();

        // Add nodes
        for i in 0..resonits.len() {
            nodes.push(graph.add_node(i));
        }

        // Add edges based on resonance threshold
        const THRESHOLD: f64 = 0.3;
        for i in 0..resonits.len() {
            for j in (i + 1)..resonits.len() {
                let resonance = resonits[i].resonance_with(&resonits[j]);
                if resonance > THRESHOLD {
                    graph.add_edge(nodes[i], nodes[j], resonance);
                }
            }
        }

        // Calculate β0 (number of connected components)
        let beta_0 = connected_components(&graph);

        // Simplified β1 calculation (number of independent cycles)
        // For a simple connected graph: β1 = edges - nodes + components
        let edges = graph.edge_count();
        let nodes_count = graph.node_count();
        let beta_1 = if edges >= nodes_count {
            edges.saturating_sub(nodes_count).saturating_add(beta_0)
        } else {
            0
        };

        // β2 and higher would require more sophisticated homology calculations
        // For simplicity, we set β2 = 0
        let beta_2 = 0;

        vec![beta_0, beta_1, beta_2]
    }

    /// Calculate topological persistence score
    ///
    /// Returns:
    ///     Persistence score in [0, 1]
    fn calculate_persistence(betti: &[usize]) -> f64 {
        if betti.len() < 3 {
            return 0.5;
        }

        // Persistence based on stability of Betti numbers
        // Higher β0 (more components) reduces persistence
        // Moderate β1 (some cycles) is good
        // High β2 (voids) reduces persistence

        let mut persistence = 1.0 / (1.0 + betti[0] as f64); // Fewer components is better
        persistence *= (1.0 + 0.5 * betti[1] as f64) / (1.0 + betti[1] as f64); // Some cycles OK
        persistence *= 1.0 / (1.0 + betti[2] as f64); // Fewer voids is better

        persistence.clamp(0.0, 1.0)
    }

    /// Calculate stability using resonance variance
    fn calculate_stability(resonits: &[Resonit]) -> f64 {
        if resonits.len() < 2 {
            return 1.0;
        }

        let mut resonances = Vec::new();
        for i in 0..resonits.len() {
            for j in (i + 1)..resonits.len() {
                resonances.push(resonits[i].resonance_with(&resonits[j]));
            }
        }

        if resonances.is_empty() {
            return 1.0;
        }

        // Calculate variance
        let mean: f64 = resonances.iter().sum::<f64>() / resonances.len() as f64;
        let variance: f64 =
            resonances.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / resonances.len() as f64;

        // Stability is inverse of variance (low variance = high stability)
        1.0 - variance.sqrt().min(1.0)
    }

    /// Calculate persistence score (public accessor)
    pub fn persistence_score(&self) -> f64 {
        self.metrics.persistence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resonit::Sigma;

    #[test]
    fn test_resonat_creation() {
        let sigma1 = Sigma::new(0.5, 0.5, 0.5);
        let sigma2 = Sigma::new(0.6, 0.6, 0.6);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();

        assert_eq!(resonat.resonits.len(), 2);
        assert_eq!(resonat.metrics.size, 2);
        assert!(resonat.centroid.is_some());
    }

    #[test]
    fn test_empty_resonat() {
        let result = Resonat::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_resonit() {
        let sigma = Sigma::new(0.5, 0.5, 0.5);
        let r = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r]).unwrap();

        assert_eq!(resonat.metrics.betti, vec![1, 0, 0]);
        assert_eq!(resonat.metrics.size, 1);
    }

    #[test]
    fn test_centroid_calculation() {
        let sigma1 = Sigma::new(0.0, 0.0, 0.0);
        let sigma2 = Sigma::new(1.0, 1.0, 1.0);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();

        let centroid = resonat.centroid.unwrap();
        assert!((centroid[0] - 0.5).abs() < 1e-6);
        assert!((centroid[1] - 0.5).abs() < 1e-6);
        assert!((centroid[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_betti_vectors_connected() {
        // Create resonits with high mutual resonance
        let sigma = Sigma::new(0.5, 0.5, 0.5);
        let r1 = Resonit::new(sigma, "test".to_string(), 0);
        let r2 = Resonit::new(sigma, "test".to_string(), 0);
        let r3 = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2, r3]).unwrap();

        // Should be connected (β0 = 1)
        assert_eq!(resonat.metrics.betti[0], 1);
    }

    #[test]
    fn test_persistence_score() {
        let sigma = Sigma::new(0.5, 0.5, 0.5);
        let r1 = Resonit::new(sigma, "test".to_string(), 0);
        let r2 = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();

        let persistence = resonat.persistence_score();
        assert!((0.0..=1.0).contains(&persistence));
    }

    #[test]
    fn test_serialization() {
        let sigma1 = Sigma::new(0.5, 0.5, 0.5);
        let sigma2 = Sigma::new(0.6, 0.6, 0.6);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();

        let json = serde_json::to_string(&resonat).unwrap();
        let deserialized: Resonat = serde_json::from_str(&json).unwrap();

        assert_eq!(resonat.resonits.len(), deserialized.resonits.len());
        assert_eq!(resonat.metrics.size, deserialized.metrics.size);
    }
}
