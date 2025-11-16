//! State Adapter: Bidirectional conversion between APOLLYON State5D and MEF Spiral coordinates
//!
//! This adapter provides lossless conversion between the two systems' 5D representations:
//! - APOLLYON State5D: Uses nalgebra SVector<f64, 5>
//! - MEF Spiral: Uses Vec<f64> with 5 coordinates
//!
//! The mapping is 1:1:
//! - D1 (index 0): x coordinate (spatial)
//! - D2 (index 1): y coordinate (spatial)
//! - D3 (index 2): z coordinate (spatial)
//! - D4 (index 3): ψ (psi) - semantic weight
//! - D5 (index 4): ω (omega) - temporal phase

use core_5d::State5D;

/// Adapter for converting between APOLLYON 5D states and MEF Spiral coordinates
pub struct StateAdapter;

impl StateAdapter {
    /// Convert APOLLYON State5D to MEF Spiral coordinates (Vec<f64>)
    ///
    /// # Perfect 1:1 Mapping
    /// - D1 → coords[0] (x)
    /// - D2 → coords[1] (y)
    /// - D3 → coords[2] (z)
    /// - D4 → coords[3] (ψ - semantic weight)
    /// - D5 → coords[4] (ω - temporal phase)
    ///
    /// # Example
    /// ```
    /// use apollyon_mef_bridge::StateAdapter;
    /// use core_5d::State5D;
    ///
    /// let apollon_state = State5D::new(1.0, 2.0, 3.0, 0.5, 0.7);
    /// let mef_coords = StateAdapter::apollyon_to_mef(&apollon_state);
    /// assert_eq!(mef_coords.len(), 5);
    /// assert_eq!(mef_coords[0], 1.0);  // x
    /// assert_eq!(mef_coords[3], 0.5);  // psi
    /// ```
    pub fn apollyon_to_mef(apollon: &State5D) -> Vec<f64> {
        vec![
            apollon.get(0), // x
            apollon.get(1), // y
            apollon.get(2), // z
            apollon.get(3), // psi (semantic weight)
            apollon.get(4), // omega (temporal phase)
        ]
    }

    /// Convert MEF Spiral coordinates to APOLLYON State5D
    ///
    /// # Panics
    /// Panics if `mef_coords` does not have exactly 5 elements or if any value is not finite
    pub fn mef_to_apollyon(mef_coords: &[f64]) -> State5D {
        assert_eq!(
            mef_coords.len(),
            5,
            "MEF coordinates must have exactly 5 elements"
        );

        State5D::new(
            mef_coords[0], // x
            mef_coords[1], // y
            mef_coords[2], // z
            mef_coords[3], // psi
            mef_coords[4], // omega
        )
    }

    /// Validate perfect roundtrip conversion
    ///
    /// Converts APOLLYON → MEF → APOLLYON and checks if the result matches
    /// the original within epsilon (1e-10).
    ///
    /// # Returns
    /// `true` if roundtrip error < 1e-10 for all components
    pub fn validate_roundtrip(original: &State5D) -> bool {
        let mef_coords = Self::apollyon_to_mef(original);
        let roundtrip = Self::mef_to_apollyon(&mef_coords);

        const EPSILON: f64 = 1e-10;

        for i in 0..5 {
            let error = (original.get(i) - roundtrip.get(i)).abs();
            if error >= EPSILON {
                return false;
            }
        }

        true
    }

    /// Compute the maximum error across all components in a roundtrip conversion
    pub fn roundtrip_error(original: &State5D) -> f64 {
        let mef_coords = Self::apollyon_to_mef(original);
        let roundtrip = Self::mef_to_apollyon(&mef_coords);

        (0..5)
            .map(|i| (original.get(i) - roundtrip.get(i)).abs())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apollyon_to_mef() {
        let state = State5D::new(1.0, 2.0, 3.0, 0.5, 0.7);
        let coords = StateAdapter::apollyon_to_mef(&state);

        assert_eq!(coords.len(), 5);
        assert_eq!(coords[0], 1.0);
        assert_eq!(coords[1], 2.0);
        assert_eq!(coords[2], 3.0);
        assert_eq!(coords[3], 0.5);
        assert_eq!(coords[4], 0.7);
    }

    #[test]
    fn test_mef_to_apollyon() {
        let coords = vec![1.0, 2.0, 3.0, 0.5, 0.7];
        let state = StateAdapter::mef_to_apollyon(&coords);

        assert_eq!(state.get(0), 1.0);
        assert_eq!(state.get(1), 2.0);
        assert_eq!(state.get(2), 3.0);
        assert_eq!(state.get(3), 0.5);
        assert_eq!(state.get(4), 0.7);
    }

    #[test]
    fn test_perfect_roundtrip() {
        let original = State5D::new(1.0, 2.0, 3.0, 0.5, 0.7);
        assert!(StateAdapter::validate_roundtrip(&original));

        let error = StateAdapter::roundtrip_error(&original);
        assert!(error < 1e-10, "Roundtrip error {} exceeds threshold", error);
    }

    #[test]
    fn test_multiple_roundtrips() {
        let test_cases = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-5.0, 3.2, 1.8, 0.42, 2.1],
            [100.0, -50.0, 25.5, 0.99, 0.01],
        ];

        for components in test_cases {
            let state = State5D::new(
                components[0],
                components[1],
                components[2],
                components[3],
                components[4],
            );
            assert!(
                StateAdapter::validate_roundtrip(&state),
                "Roundtrip failed for {:?}",
                components
            );
        }
    }

    #[test]
    fn test_zero_state() {
        let zero = State5D::zero();
        assert!(StateAdapter::validate_roundtrip(&zero));
    }

    #[test]
    #[should_panic(expected = "MEF coordinates must have exactly 5 elements")]
    fn test_invalid_mef_coords_length() {
        let coords = vec![1.0, 2.0, 3.0]; // Only 3 elements
        StateAdapter::mef_to_apollyon(&coords);
    }

    #[test]
    fn test_conversion_preserves_structure() {
        let state = State5D::new(1.5, 2.5, 3.5, 0.75, 0.25);
        let coords = StateAdapter::apollyon_to_mef(&state);
        let back = StateAdapter::mef_to_apollyon(&coords);

        // Verify each component individually
        assert!((state.get(0) - back.get(0)).abs() < 1e-15);
        assert!((state.get(1) - back.get(1)).abs() < 1e-15);
        assert!((state.get(2) - back.get(2)).abs() < 1e-15);
        assert!((state.get(3) - back.get(3)).abs() < 1e-15);
        assert!((state.get(4) - back.get(4)).abs() < 1e-15);
    }
}
