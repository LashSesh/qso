//! Core type definitions for the 4D-Trichter system

use serde::{Deserialize, Serialize};

/// 4D state in process space (x, y, z, ψ)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct State4D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub psi: f64, // semantic weight/resonance
}

impl State4D {
    pub fn new(x: f64, y: f64, z: f64, psi: f64) -> Self {
        Self { x, y, z, psi }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn as_array(&self) -> [f64; 4] {
        [self.x, self.y, self.z, self.psi]
    }

    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }
}

/// 5D state in full space (x, y, z, ψ, ω)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct State5D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub psi: f64,   // semantic weight/resonance
    pub omega: f64, // temporal phase/oscillation
}

impl State5D {
    pub fn new(x: f64, y: f64, z: f64, psi: f64, omega: f64) -> Self {
        Self { x, y, z, psi, omega }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn as_array(&self) -> [f64; 5] {
        [self.x, self.y, self.z, self.psi, self.omega]
    }

    pub fn from_array(arr: [f64; 5]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3], arr[4])
    }

    pub fn norm(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + 
         self.psi.powi(2) + self.omega.powi(2)).sqrt()
    }
}

/// Guidance vector field in 4D
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GuidanceVector {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub vpsi: f64,
}

impl GuidanceVector {
    pub fn new(vx: f64, vy: f64, vz: f64, vpsi: f64) -> Self {
        Self { vx, vy, vz, vpsi }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Node in the Funnel graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelNode {
    pub id: usize,
    pub state: State5D,
    pub mass: f64,        // accumulated evidence
    pub variance: f64,    // local uncertainty
    pub t_born: f64,      // creation time
}

impl FunnelNode {
    pub fn new(id: usize, state: State5D, t: f64) -> Self {
        Self {
            id,
            state,
            mass: 1.0,
            variance: 0.1,
            t_born: t,
        }
    }
}

/// Edge in the Funnel graph with Hebbian parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub decay: f64,
    pub phase_lock: f64, // coherence binding ∈ [0,1]
}

impl FunnelEdge {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            weight: 0.1,
            decay: 0.01,
            phase_lock: 0.0,
        }
    }
}

/// Resonance and morphodynamic fields from Hyperbion layer
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HyperbionFields {
    pub phi: f64,  // Phase/Resonance field Φ(x,t)
    pub mu: f64,   // Morphodynamic growth/damping field μ(x,t)
}

impl HyperbionFields {
    pub fn new(phi: f64, mu: f64) -> Self {
        Self { phi, mu }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Hash for proof artifacts (local only, no network)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofHash(pub [u8; 32]);

impl ProofHash {
    pub fn new(data: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        Self(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state4d_creation() {
        let s = State4D::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(s.x, 1.0);
        assert_eq!(s.psi, 4.0);
    }

    #[test]
    fn test_state5d_norm() {
        let s = State5D::new(3.0, 4.0, 0.0, 0.0, 0.0);
        assert_eq!(s.norm(), 5.0);
    }

    #[test]
    fn test_array_conversion() {
        let s4 = State4D::new(1.0, 2.0, 3.0, 4.0);
        let arr = s4.as_array();
        let s4_back = State4D::from_array(arr);
        assert_eq!(s4, s4_back);
    }
}
