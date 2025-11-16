//! RouteSpec - S7 route specification with 7-slot permutation

use serde::{Deserialize, Serialize};

/// Operator slot enumeration for S7 permutations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorSlot {
    /// Deep Knowledge operator
    DK,
    /// Swarm operator  
    SW,
    /// Path Invariance operator
    PI,
    /// Witness operator
    WT,
    /// Reserved slot 1
    RES1,
    /// Adapter slot
    ADAPTER,
    /// Reserved slot 2
    RES2,
}

/// RouteSpec represents a permutation of the 7 operators in S7 space
/// The permutation space has 7! = 5040 possible routes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteSpec {
    /// The route identifier (content-addressed)
    pub route_id: String,

    /// The 7-slot permutation of operators [0..7)
    pub permutation: Vec<usize>,

    /// Mesh metrics: betti number, spectral gap, persistence
    pub mesh_score: f64,

    /// Operator slot permutation (extended version)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma: Option<Vec<u8>>,

    /// Named operator permutation (extended version)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permutation_slots: Option<Vec<OperatorSlot>>,
}

impl RouteSpec {
    /// Create a new RouteSpec with validation
    pub fn new(route_id: String, permutation: Vec<usize>, mesh_score: f64) -> crate::Result<Self> {
        // Validate permutation is a valid 7-element permutation
        if permutation.len() != 7 {
            return Err(crate::SchemaError::InvalidRoute(format!(
                "Permutation must have exactly 7 elements, got {}",
                permutation.len()
            )));
        }

        // Check all elements are unique and in range [0..7)
        let mut seen = vec![false; 7];
        for &idx in &permutation {
            if idx >= 7 {
                return Err(crate::SchemaError::InvalidRoute(format!(
                    "Invalid permutation index: {}",
                    idx
                )));
            }
            if seen[idx] {
                return Err(crate::SchemaError::InvalidRoute(format!(
                    "Duplicate permutation index: {}",
                    idx
                )));
            }
            seen[idx] = true;
        }

        Ok(Self {
            route_id,
            permutation,
            mesh_score,
            sigma: None,
            permutation_slots: None,
        })
    }

    /// Create a new RouteSpec with extended fields
    pub fn new_extended(
        route_id: String,
        sigma: Vec<u8>,
        permutation_slots: Vec<OperatorSlot>,
        mesh_score: f64,
    ) -> Self {
        // Convert sigma to permutation (0-based indexing)
        let permutation: Vec<usize> = sigma.iter().map(|&x| (x - 1) as usize).collect();

        Self {
            route_id,
            permutation,
            mesh_score,
            sigma: Some(sigma),
            permutation_slots: Some(permutation_slots),
        }
    }

    /// Validate the route specification
    pub fn validate(&self) -> crate::Result<()> {
        // Validate permutation
        if self.permutation.len() != 7 {
            return Err(crate::SchemaError::InvalidRoute(format!(
                "Permutation must have exactly 7 elements, got {}",
                self.permutation.len()
            )));
        }

        // Validate sigma if present
        if let Some(ref sigma) = self.sigma {
            if sigma.len() != 7 {
                return Err(crate::SchemaError::InvalidRoute(format!(
                    "Sigma must have exactly 7 elements, got {}",
                    sigma.len()
                )));
            }
            // Validate range [1..8)
            for &val in sigma {
                if val < 1 || val > 7 {
                    return Err(crate::SchemaError::InvalidRoute(format!(
                        "Sigma values must be in [1..8), got {}",
                        val
                    )));
                }
            }
        }

        // Validate permutation_slots if present
        if let Some(ref slots) = self.permutation_slots {
            if slots.len() != 7 {
                return Err(crate::SchemaError::InvalidRoute(format!(
                    "Permutation slots must have exactly 7 elements, got {}",
                    slots.len()
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_route_spec() {
        let route = RouteSpec::new("route_001".to_string(), vec![0, 1, 2, 3, 4, 5, 6], 0.75);
        assert!(route.is_ok());
    }

    #[test]
    fn test_invalid_permutation_length() {
        let route = RouteSpec::new("route_002".to_string(), vec![0, 1, 2], 0.5);
        assert!(route.is_err());
    }

    #[test]
    fn test_duplicate_index() {
        let route = RouteSpec::new("route_003".to_string(), vec![0, 1, 2, 2, 4, 5, 6], 0.5);
        assert!(route.is_err());
    }
}
