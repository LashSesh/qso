//! Metatron's Cube and the Fruit of Life.
//!
//! The Fruit of Life consists of 13 equal circles: one in the centre, six
//! touching it (inner ring) and six more touching those along the same six
//! rays (outer ring). With the circle diameter as unit length the centres
//! are the origin, the inner hexagon at radius 1 and the outer hexagon at
//! radius 2. **Metatron's Cube** is the figure obtained by joining every pair
//! of these 13 centres with a straight line: 78 = C(13, 2) lines.
//!
//! As a graph Metatron's Cube is therefore the complete graph K₁₃. All of
//! its geometric richness lives in the *embedding* (line lengths and
//! directions), not in the adjacency structure; its automorphism group is
//! the full symmetric group S₁₃ of order 13!. The geometric structure is
//! recovered by keeping only lines up to a given length: the Fruit-of-Life
//! contact graph (tangent circles, length 1) has exactly the dihedral
//! symmetry D₆ of order 12. [`distance_graph`] exposes the whole family.
//!
//! A note on a common claim: K₁₃ contains every graph on ≤ 13 vertices as a
//! subgraph, so it trivially "contains" the tetrahedron (4), octahedron (6),
//! cube (8) and icosahedron (12) graphs — but not the dodecahedron, which has
//! 20 vertices. The 2-D drawing shows *projections* of Platonic solids.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_3};

use serde::{Deserialize, Serialize};

use super::Graph;

/// Number of circle centres in the Fruit of Life.
pub const NODES: usize = 13;

/// Position class of a centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ring {
    Centre,
    Inner,
    Outer,
}

/// Ring membership of node `u` (0 centre, 1–6 inner, 7–12 outer).
pub fn ring(u: usize) -> Ring {
    match u {
        0 => Ring::Centre,
        1..=6 => Ring::Inner,
        _ => Ring::Outer,
    }
}

/// Coordinates of the 13 centres (z = 0). Node `k ∈ 1..=6` lies at angle
/// `90° + (k−1)·60°` on the inner ring, node `k + 6` on the same ray of the
/// outer ring.
pub fn positions() -> Vec<[f64; 3]> {
    let mut pos = vec![[0.0, 0.0, 0.0]];
    for radius in [1.0, 2.0] {
        for k in 0..6 {
            let a = FRAC_PI_2 + k as f64 * FRAC_PI_3;
            pos.push([radius * a.cos(), radius * a.sin(), 0.0]);
        }
    }
    pos
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Graph of all lines of Metatron's Cube with length ≤ `max_length`
/// (in units of the circle diameter).
pub fn distance_graph(max_length: f64) -> Graph {
    let pos = positions();
    let mut g = Graph::new(NODES).with_name(format!("metatron(<= {max_length})"));
    for u in 0..NODES {
        for v in u + 1..NODES {
            if distance(&pos[u], &pos[v]) <= max_length + 1e-9 {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    g.with_positions(pos).expect("13 positions")
}

/// Metatron's Cube: all 78 lines between the 13 centres (the complete graph
/// K₁₃ with the Fruit-of-Life embedding).
pub fn metatron_cube() -> Graph {
    distance_graph(f64::INFINITY).with_name("Metatron's Cube (K13)")
}

/// Contact graph of the Fruit of Life: centres of tangent circles are
/// adjacent (18 edges, symmetry group D₆).
pub fn fruit_of_life() -> Graph {
    distance_graph(1.0).with_name("Fruit of Life (contact graph)")
}

/// Distinct line lengths of Metatron's Cube with their multiplicities,
/// ascending. The multiplicities sum to 78.
pub fn line_lengths() -> Vec<(f64, usize)> {
    let pos = positions();
    let mut lengths: Vec<f64> = Vec::new();
    for u in 0..NODES {
        for v in u + 1..NODES {
            lengths.push(distance(&pos[u], &pos[v]));
        }
    }
    lengths.sort_by(f64::total_cmp);
    let mut out: Vec<(f64, usize)> = Vec::new();
    for l in lengths {
        match out.last_mut() {
            Some((last, count)) if (l - *last).abs() < 1e-9 => *count += 1,
            _ => out.push((l, 1)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metatron_cube_is_complete_graph_k13() {
        let g = metatron_cube();
        assert_eq!(g.num_nodes(), 13);
        assert_eq!(g.num_edges(), 78);
        assert_eq!(g.regular_degree(), Some(12));
    }

    #[test]
    fn line_lengths_are_the_classical_ones() {
        let lengths = line_lengths();
        let total: usize = lengths.iter().map(|&(_, c)| c).sum();
        assert_eq!(total, 78);
        // Shortest lines are the 18 contacts of tangent circles.
        assert!((lengths[0].0 - 1.0).abs() < 1e-12);
        assert_eq!(lengths[0].1, 18);
        // Seven distinct lengths, ascending: 1, √3, 2, √7, 3, 2√3, 4 with
        // multiplicities 18, 18, 15, 12, 6, 6, 3.
        let expected = [
            (1.0, 18),
            (3f64.sqrt(), 18),
            (2.0, 15),
            (7f64.sqrt(), 12),
            (3.0, 6),
            (2.0 * 3f64.sqrt(), 6),
            (4.0, 3),
        ];
        assert_eq!(lengths.len(), expected.len());
        for ((l, c), (el, ec)) in lengths.iter().zip(expected) {
            assert!((l - el).abs() < 1e-12 && *c == ec, "{l} x{c}");
        }
    }

    #[test]
    fn fruit_of_life_contact_graph() {
        let g = fruit_of_life();
        assert_eq!(g.num_edges(), 18);
        assert_eq!(g.degree(0), 6);
        assert!((1..=6).all(|u| g.degree(u) == 4));
        assert!((7..=12).all(|u| g.degree(u) == 1));
        assert!(g.is_connected());
    }
}
