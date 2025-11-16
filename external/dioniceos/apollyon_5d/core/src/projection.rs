//! Visualization and Projection
//!
//! Section 6 - dimension reduction from 5D to 2D/3D for visualization.

use crate::state::State5D;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// 2D point for visualization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }
}

/// 3D point for visualization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D { x, y, z }
    }
}

/// Projection methods for dimension reduction
#[derive(Debug, Clone, Copy)]
pub enum ProjectionMethod {
    /// Orthogonal projection onto specified dimensions
    Orthogonal(usize, usize),
    /// Isometric projection
    Isometric,
    /// PCA projection (computed from data)
    PCA,
}

/// Projector for reducing 5D states to 2D
pub struct Projector {
    pub method: ProjectionMethod,
    /// PCA components (if using PCA)
    pca_components: Option<[[f64; 5]; 2]>,
}

impl Projector {
    /// Create a new projector
    pub fn new(method: ProjectionMethod) -> Self {
        Projector {
            method,
            pca_components: None,
        }
    }

    /// Create orthogonal projector for two dimensions
    pub fn orthogonal(dim1: usize, dim2: usize) -> Self {
        Projector::new(ProjectionMethod::Orthogonal(dim1, dim2))
    }

    /// Create isometric projector
    pub fn isometric() -> Self {
        Projector::new(ProjectionMethod::Isometric)
    }

    /// Fit PCA to trajectory data
    pub fn fit_pca(&mut self, states: &[State5D]) {
        if states.len() < 2 {
            return;
        }

        // Compute mean
        let mut mean = State5D::zero();
        for state in states {
            mean = mean + *state;
        }
        mean = mean.scale(1.0 / states.len() as f64);

        // Build data matrix (centered)
        let n = states.len();
        let mut data = DMatrix::zeros(n, 5);
        for (i, state) in states.iter().enumerate() {
            let centered = state.sub(&mean);
            for j in 0..5 {
                data[(i, j)] = centered.get(j);
            }
        }

        // Compute covariance matrix
        let cov = data.transpose() * data.clone() * (1.0 / (n - 1) as f64);

        // Compute eigenvectors (PCA components)
        let eigen = cov.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Find two principal components (largest eigenvalues)
        let mut indices: Vec<usize> = (0..5).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let pc1_idx = indices[0];
        let pc2_idx = indices[1];

        let mut pc1 = [0.0; 5];
        let mut pc2 = [0.0; 5];
        for i in 0..5 {
            pc1[i] = eigenvectors[(i, pc1_idx)];
            pc2[i] = eigenvectors[(i, pc2_idx)];
        }

        self.pca_components = Some([pc1, pc2]);
    }

    /// Project a single state to 2D
    pub fn project(&self, state: &State5D) -> Point2D {
        match self.method {
            ProjectionMethod::Orthogonal(dim1, dim2) => {
                Point2D::new(state.get(dim1), state.get(dim2))
            }
            ProjectionMethod::Isometric => {
                // Isometric projection: weighted sum of dimensions
                let x = 0.866 * state.get(0) - 0.866 * state.get(1);
                let y = 0.5 * state.get(0) + 0.5 * state.get(1) - state.get(2);
                Point2D::new(x, y)
            }
            ProjectionMethod::PCA => {
                if let Some(components) = &self.pca_components {
                    let x = (0..5).map(|i| components[0][i] * state.get(i)).sum();
                    let y = (0..5).map(|i| components[1][i] * state.get(i)).sum();
                    Point2D::new(x, y)
                } else {
                    // Fallback to orthogonal
                    Point2D::new(state.get(0), state.get(1))
                }
            }
        }
    }

    /// Project multiple states
    pub fn project_many(&self, states: &[State5D]) -> Vec<Point2D> {
        states.iter().map(|s| self.project(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonal_projection() {
        let proj = Projector::orthogonal(0, 1);
        let state = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let point = proj.project(&state);

        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
    }

    #[test]
    fn test_isometric_projection() {
        let proj = Projector::isometric();
        let state = State5D::zero();
        let point = proj.project(&state);

        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
    }
}
