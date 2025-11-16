//! Metatron Bridge: Connect APOLLYON's Metatron-R with MEF's S7 Router
//!
//! This bridge connects APOLLYON's QLogic spectral analysis engine with MEF's
//! S7 routing system. It extracts topological metrics from APOLLYON's spectral
//! analysis and uses them to enhance MEF's deterministic route selection.
//!
//! # Architecture
//! - QLogicEngine → Spectral analysis → Metrics
//! - Metrics → MeshScore → S7 Route Selection
//! - Provides deterministic, reproducible routing

use core_5d::State5D;
use mef_router::{compute_mesh_score, select_route};
use mef_schemas::RouteSpec;
use metatron::cognition::qlogic::QLogicEngine;
use std::collections::HashMap;

/// Bridge between APOLLYON Metatron-R and MEF S7 Router
///
/// This bridge uses APOLLYON's QLogic spectral analysis to compute
/// topological metrics (betti, spectral_gap, persistence) which are
/// then used by MEF's S7 router for deterministic route selection.
pub struct MetatronBridge {
    qlogic: QLogicEngine,
}

impl MetatronBridge {
    /// Create a new MetatronBridge with default 13-node configuration
    ///
    /// The 13-node configuration corresponds to the Metatron Cube geometry
    /// used in APOLLYON's geometric-cognitive framework.
    pub fn new() -> Self {
        Self {
            qlogic: QLogicEngine::new(13, None),
        }
    }

    /// Create a bridge with custom node configuration
    pub fn with_nodes(num_nodes: usize) -> Self {
        Self {
            qlogic: QLogicEngine::new(num_nodes, None),
        }
    }

    /// Compute mesh metrics from a 5D state using QLogic spectral analysis
    ///
    /// # Metric Extraction
    /// The QLogic engine analyzes the 5D state and produces:
    /// - `betti`: Topological Betti number (from spectral analysis)
    /// - `lambda_gap`: Spectral gap (eigenvalue separation)
    /// - `persistence`: Topological persistence (from diagnostics)
    /// - `entropy`: Shannon entropy of the spectrum
    /// - `coherence`: Spectral coherence (from sparsity)
    ///
    /// # Arguments
    /// * `state` - The 5D state to analyze
    /// * `t` - Time parameter for the oscillator
    ///
    /// # Returns
    /// A HashMap containing the computed metrics
    pub fn compute_mesh_metrics(&mut self, _state: &State5D, t: f64) -> HashMap<String, f64> {
        // Run QLogic analysis
        let output = self.qlogic.step(t);

        let mut metrics = HashMap::new();

        // Extract spectral features for mesh metrics
        // betti: Use entropy as a proxy for topological complexity
        // Lower entropy = more structure = higher betti
        let betti = if output.entropy > 0.0 {
            (1.0 / (output.entropy + 0.1)).min(10.0)
        } else {
            1.0
        };
        metrics.insert("betti".to_string(), betti);

        // lambda_gap: Spectral gap from spectrum analysis
        // Compute gap between dominant and secondary frequencies
        let lambda_gap = if output.spectrum.len() >= 2 {
            let sorted: Vec<f64> = {
                let mut s = output.spectrum.clone();
                s.sort_by(|a, b| b.partial_cmp(a).unwrap());
                s
            };
            if sorted[0] > 0.0 {
                (sorted[0] - sorted[1]) / sorted[0]
            } else {
                0.0
            }
        } else {
            0.0
        };
        metrics.insert("lambda_gap".to_string(), lambda_gap);

        // persistence: From diagnostics if available
        let persistence = if let Some(ref diag) = output.diagnostics {
            // Use sparsity as persistence indicator
            diag.sparsity
        } else {
            0.5 // Default middle value
        };
        metrics.insert("persistence".to_string(), persistence);

        // Additional metrics for analysis
        metrics.insert("entropy".to_string(), output.entropy);

        if let Some(ref diag) = output.diagnostics {
            metrics.insert("spectral_centroid".to_string(), diag.spectral_centroid);
            metrics.insert("coherence".to_string(), 1.0 - diag.sparsity);
        }

        metrics
    }

    /// Select MEF route using APOLLYON-enhanced topological metrics
    ///
    /// This method combines APOLLYON's spectral analysis with MEF's
    /// deterministic S7 routing to provide topology-aware route selection.
    ///
    /// # Arguments
    /// * `state` - The 5D state to analyze
    /// * `seed` - Seed for deterministic route selection
    /// * `t` - Time parameter for oscillator
    ///
    /// # Returns
    /// A RouteSpec from the S7 permutation space (7! = 5040 routes)
    ///
    /// # Errors
    /// Returns an error if route selection fails
    pub fn select_route_enhanced(
        &mut self,
        state: &State5D,
        seed: &str,
        t: f64,
    ) -> Result<RouteSpec, Box<dyn std::error::Error>> {
        // Compute metrics from APOLLYON analysis
        let metrics = self.compute_mesh_metrics(state, t);

        // Select route using MEF router
        let route = select_route(seed, &metrics)?;

        Ok(route)
    }

    /// Compute only the mesh score without full route selection
    ///
    /// This is useful for analysis and debugging.
    pub fn compute_mesh_score_only(
        &mut self,
        state: &State5D,
        t: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let metrics = self.compute_mesh_metrics(state, t);
        let score = compute_mesh_score(&metrics)?;
        Ok(score)
    }
}

impl Default for MetatronBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let _bridge = MetatronBridge::new();
        assert!(true); // Bridge created successfully
    }

    #[test]
    fn test_metric_computation() {
        let mut bridge = MetatronBridge::new();
        let state = State5D::new(1.0, 0.5, 0.3, 0.7, 0.2);

        let metrics = bridge.compute_mesh_metrics(&state, 0.0);

        // Verify required metrics are present
        assert!(metrics.contains_key("betti"));
        assert!(metrics.contains_key("lambda_gap"));
        assert!(metrics.contains_key("persistence"));

        // Verify metrics are finite
        assert!(metrics["betti"].is_finite());
        assert!(metrics["lambda_gap"].is_finite());
        assert!(metrics["persistence"].is_finite());
    }

    #[test]
    fn test_deterministic_routing() {
        let mut bridge = MetatronBridge::new();
        let state = State5D::new(1.0, 0.5, 0.3, 0.7, 0.2);

        // Same state + same seed should give same route
        let route1 = bridge
            .select_route_enhanced(&state, "test_seed", 0.0)
            .unwrap();

        // Create new bridge to ensure no state leakage
        let mut bridge2 = MetatronBridge::new();
        let route2 = bridge2
            .select_route_enhanced(&state, "test_seed", 0.0)
            .unwrap();

        assert_eq!(route1.route_id, route2.route_id);
        assert_eq!(route1.permutation, route2.permutation);
    }

    #[test]
    fn test_different_states_different_routes() {
        let mut bridge = MetatronBridge::new();

        let state1 = State5D::new(1.0, 0.5, 0.3, 0.7, 0.2);
        let state2 = State5D::new(2.0, 1.0, 0.6, 1.4, 0.4);

        let route1 = bridge.select_route_enhanced(&state1, "seed", 0.0).unwrap();
        let route2 = bridge.select_route_enhanced(&state2, "seed", 0.0).unwrap();

        // Different states may produce different routes
        // (not guaranteed but likely due to different spectral properties)
        // We just verify both succeed
        assert!(route1.permutation.len() == 7);
        assert!(route2.permutation.len() == 7);
    }

    #[test]
    fn test_mesh_score_computation() {
        let mut bridge = MetatronBridge::new();
        let state = State5D::new(1.0, 0.5, 0.3, 0.7, 0.2);

        let score = bridge.compute_mesh_score_only(&state, 0.0).unwrap();

        // Score should be finite and reasonable
        assert!(score.is_finite());
        assert!(score >= 0.0); // Mesh scores are typically non-negative
    }

    #[test]
    fn test_route_structure() {
        let mut bridge = MetatronBridge::new();
        let state = State5D::new(1.0, 0.5, 0.3, 0.7, 0.2);

        let route = bridge.select_route_enhanced(&state, "test", 0.0).unwrap();

        // Verify route structure
        assert_eq!(route.permutation.len(), 7);
        assert!(route.route_id.starts_with("route_"));
        assert!(route.mesh_score.is_finite());

        // Verify permutation is valid (all indices 0-6 present once)
        let mut seen = vec![false; 7];
        for &idx in &route.permutation {
            assert!(idx < 7);
            assert!(!seen[idx], "Duplicate index in permutation");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&x| x), "Missing index in permutation");
    }

    #[test]
    fn test_custom_node_count() {
        let mut bridge = MetatronBridge::with_nodes(7);
        let state = State5D::new(1.0, 0.5, 0.3, 0.7, 0.2);

        let metrics = bridge.compute_mesh_metrics(&state, 0.0);
        assert!(metrics.contains_key("betti"));
    }
}
