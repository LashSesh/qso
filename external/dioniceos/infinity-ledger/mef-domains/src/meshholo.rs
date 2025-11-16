/*!
 * MeshHolo - Holographic triangulation of information space
 *
 * Leverages Metatron Cube topology for enhanced triangulation
 * and topological invariant calculations.
 */

use crate::resonat::Resonat;
use mef_core::geometry::canonical_nodes;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Holographic triangulation of information space using Metatron topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshHolo {
    /// Unique identifier
    pub id: String,
    /// Deterministic seed
    pub seed: String,
    /// Vertex data with coordinates and sigma values
    pub vertices: Vec<VertexData>,
    /// Edge list with weights (v1_id, v2_id, weight)
    pub edges: Vec<EdgeData>,
    /// Triangular/tetrahedral simplices
    pub simplices: Vec<Vec<String>>,
    /// Topological invariants
    pub invariants: TopologicalInvariants,
    /// Vertex to Metatron node mapping (1-13)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metatron_mapping: Option<HashMap<String, usize>>,
    /// Provenance information
    #[serde(default)]
    pub provenance: HashMap<String, serde_json::Value>,
    /// Proof data
    #[serde(default)]
    pub proof: HashMap<String, serde_json::Value>,
}

/// Vertex data in MeshHolo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexData {
    /// Vertex identifier
    pub id: String,
    /// Spherical coordinate theta
    pub theta: f64,
    /// Spherical coordinate chi
    pub chi: f64,
    /// Optional phi coordinate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phi: Option<f64>,
    /// Sigma values
    pub sigma: HashMap<String, f64>,
}

/// Edge data in MeshHolo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    /// Source vertex ID
    pub v1: String,
    /// Target vertex ID
    pub v2: String,
    /// Edge weight
    pub weight: f64,
}

/// Topological invariants for MeshHolo
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopologicalInvariants {
    /// Betti numbers
    pub betti: Vec<usize>,
    /// Spectral gap λ_2 - λ_1
    pub lambda_gap: f64,
    /// Persistence score
    pub persistence: f64,
}

impl MeshHolo {
    /// Create a new MeshHolo from Resonat
    pub fn from_resonat(resonat: &Resonat, seed: String) -> Self {
        let vertices = Self::create_vertices(&resonat.resonits);
        let edges = Self::create_edges(&vertices, resonat);
        let simplices = Vec::new(); // Would be filled by Delaunay triangulation

        let invariants = TopologicalInvariants {
            betti: resonat.metrics.betti.clone(),
            lambda_gap: Self::calculate_spectral_gap(&edges, &vertices),
            persistence: resonat.metrics.persistence,
        };

        let mut provenance = HashMap::new();
        provenance.insert(
            "source".to_string(),
            serde_json::Value::String("domain_layer".to_string()),
        );
        provenance.insert(
            "resonat_id".to_string(),
            serde_json::Value::String(resonat.id.clone()),
        );

        let metatron_mapping = Some(Self::map_to_metatron(&vertices));

        Self {
            id: Uuid::new_v4().to_string(),
            seed,
            vertices,
            edges,
            simplices,
            invariants,
            metatron_mapping,
            provenance,
            proof: HashMap::new(),
        }
    }

    /// Create vertices from Resonits
    fn create_vertices(resonits: &[crate::resonit::Resonit]) -> Vec<VertexData> {
        resonits
            .iter()
            .map(|resonit| {
                let sigma_vec = resonit.to_vector();
                let norm =
                    (sigma_vec[0].powi(2) + sigma_vec[1].powi(2) + sigma_vec[2].powi(2)).sqrt();

                let theta = sigma_vec[1].atan2(sigma_vec[0]);
                let chi = if norm > 1e-10 {
                    (sigma_vec[2] / norm).clamp(-1.0, 1.0).acos()
                } else {
                    0.0
                };

                let mut sigma_map = HashMap::new();
                sigma_map.insert("psi".to_string(), resonit.sigma.psi);
                sigma_map.insert("rho".to_string(), resonit.sigma.rho);
                sigma_map.insert("omega".to_string(), resonit.sigma.omega);

                VertexData {
                    id: resonit.id.clone(),
                    theta,
                    chi,
                    phi: None,
                    sigma: sigma_map,
                }
            })
            .collect()
    }

    /// Create edges from vertices using resonance
    fn create_edges(vertices: &[VertexData], resonat: &Resonat) -> Vec<EdgeData> {
        let mut edges = Vec::new();

        for i in 0..resonat.resonits.len() {
            for j in (i + 1)..resonat.resonits.len() {
                let weight = resonat.resonits[i].resonance_with(&resonat.resonits[j]);

                // Only add edges with significant resonance
                if weight > 0.1 {
                    edges.push(EdgeData {
                        v1: vertices[i].id.clone(),
                        v2: vertices[j].id.clone(),
                        weight,
                    });
                }
            }
        }

        edges
    }

    /// Calculate spectral gap of the triangulation graph
    fn calculate_spectral_gap(edges: &[EdgeData], vertices: &[VertexData]) -> f64 {
        if vertices.len() < 2 {
            return 0.0;
        }

        let n = vertices.len();
        let mut vertex_idx = HashMap::new();
        for (i, v) in vertices.iter().enumerate() {
            vertex_idx.insert(v.id.clone(), i);
        }

        // Build adjacency matrix
        let mut adj = Array2::<f64>::zeros((n, n));
        for edge in edges {
            if let (Some(&i), Some(&j)) = (vertex_idx.get(&edge.v1), vertex_idx.get(&edge.v2)) {
                adj[[i, j]] = edge.weight;
                adj[[j, i]] = edge.weight;
            }
        }

        // Calculate degree matrix and Laplacian
        let mut lap = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let degree: f64 = adj.row(i).sum();
            lap[[i, i]] = degree;
            for j in 0..n {
                lap[[i, j]] -= adj[[i, j]];
            }
        }

        // Calculate eigenvalues using nalgebra
        use nalgebra::DMatrix;
        let lap_vec: Vec<f64> = lap.iter().cloned().collect();
        let nalg_matrix = DMatrix::from_row_slice(n, n, &lap_vec);

        let eigs = nalg_matrix.symmetric_eigenvalues();
        let mut eigenvalues: Vec<f64> = eigs.iter().cloned().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if eigenvalues.len() > 1 {
            eigenvalues[1] - eigenvalues[0]
        } else {
            0.0
        }
    }

    /// Map vertices to closest Metatron nodes
    fn map_to_metatron(vertices: &[VertexData]) -> HashMap<String, usize> {
        let mut mapping = HashMap::new();
        let nodes = canonical_nodes();

        for vertex in vertices {
            let v_coords = [vertex.theta, vertex.chi, 0.0];

            let mut min_dist = f64::INFINITY;
            let mut closest_node = 1;

            for node in &nodes {
                let node_coords = [node.coords.0, node.coords.1, 0.0];

                let dist: f64 = v_coords
                    .iter()
                    .zip(node_coords.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    closest_node = node.index;
                }
            }

            mapping.insert(vertex.id.clone(), closest_node);
        }

        mapping
    }

    /// Embed MeshHolo vertices into Metatron Cube topology
    pub fn to_metatron_embedding(&self) -> Array2<f64> {
        let n_vertices = self.vertices.len();
        let mut embedding = Array2::<f64>::zeros((n_vertices, 13));

        let nodes = canonical_nodes();

        for (i, vertex) in self.vertices.iter().enumerate() {
            let v_coords = [vertex.theta, vertex.chi, vertex.phi.unwrap_or(0.0)];

            for (j, node) in nodes.iter().enumerate() {
                // Calculate affinity based on geometric distance
                let node_coords = [node.coords.0, node.coords.1, node.coords.2];

                let distance: f64 = v_coords
                    .iter()
                    .zip(node_coords.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                embedding[[i, j]] = (-distance).exp();
            }

            // Normalize embedding
            let row_sum: f64 = embedding.row(i).sum();
            if row_sum > 1e-10 {
                for j in 0..13 {
                    embedding[[i, j]] /= row_sum;
                }
            }
        }

        embedding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resonit::{Resonit, Sigma};

    #[test]
    fn test_meshholo_creation() {
        let sigma1 = Sigma::new(0.5, 0.5, 0.5);
        let sigma2 = Sigma::new(0.6, 0.6, 0.6);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "test-seed".to_string());

        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.seed, "test-seed");
        assert!(mesh.metatron_mapping.is_some());
    }

    #[test]
    fn test_vertex_creation() {
        let sigma = Sigma::new(1.0, 0.0, 0.0);
        let r = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "seed".to_string());

        assert_eq!(mesh.vertices.len(), 1);
        assert!(mesh.vertices[0].sigma.contains_key("psi"));
        assert!(mesh.vertices[0].sigma.contains_key("rho"));
        assert!(mesh.vertices[0].sigma.contains_key("omega"));
    }

    #[test]
    fn test_edge_creation() {
        let sigma1 = Sigma::new(0.5, 0.5, 0.5);
        let sigma2 = Sigma::new(0.5, 0.5, 0.5);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "seed".to_string());

        // High resonance should create edges
        assert!(!mesh.edges.is_empty());
    }

    #[test]
    fn test_spectral_gap() {
        let sigma = Sigma::new(0.5, 0.5, 0.5);
        let r1 = Resonit::new(sigma, "test".to_string(), 0);
        let r2 = Resonit::new(sigma, "test".to_string(), 0);
        let r3 = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2, r3]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "seed".to_string());

        assert!(mesh.invariants.lambda_gap >= 0.0);
    }

    #[test]
    fn test_metatron_mapping() {
        let sigma = Sigma::new(0.5, 0.5, 0.5);
        let r = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r.clone()]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "seed".to_string());

        let mapping = mesh.metatron_mapping.unwrap();
        assert!(mapping.contains_key(&r.id));

        let node_idx = mapping.get(&r.id).unwrap();
        assert!(*node_idx >= 1 && *node_idx <= 13);
    }

    #[test]
    fn test_metatron_embedding() {
        let sigma1 = Sigma::new(0.5, 0.5, 0.5);
        let sigma2 = Sigma::new(0.6, 0.6, 0.6);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r1, r2]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "seed".to_string());

        let embedding = mesh.to_metatron_embedding();
        assert_eq!(embedding.shape(), &[2, 13]);

        // Check normalization
        for i in 0..2 {
            let row_sum: f64 = embedding.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_serialization() {
        let sigma = Sigma::new(0.5, 0.5, 0.5);
        let r = Resonit::new(sigma, "test".to_string(), 0);

        let resonat = Resonat::new(vec![r]).unwrap();
        let mesh = MeshHolo::from_resonat(&resonat, "seed".to_string());

        let json = serde_json::to_string(&mesh).unwrap();
        let deserialized: MeshHolo = serde_json::from_str(&json).unwrap();

        assert_eq!(mesh.id, deserialized.id);
        assert_eq!(mesh.seed, deserialized.seed);
        assert_eq!(mesh.vertices.len(), deserialized.vertices.len());
    }
}
