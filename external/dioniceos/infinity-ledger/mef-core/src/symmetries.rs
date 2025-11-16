/*!
 * Symmetries Module - Group-Theoretic Utilities for Metatron Cube
 *
 * This module contains group-theoretic utilities for the Metatron Cube. In
 * addition to generating all S7 permutations and constructing permutation
 * matrices (as defined in the blueprint), it provides convenience functions
 * for important subgroups such as the cyclic hexagon rotations (C₆), the
 * dihedral group on the hexagon (D₆), and alternating or symmetric groups
 * on arbitrary subsets of node indices.
 *
 * The permutation functions return vectors of 1-based indices. For permutation
 * matrices acting on the full 13×13 adjacency matrix, use `permutation_matrix`.
 */

use ndarray::Array2;
use std::collections::HashMap;

/// Hexagon node angle mapping (node index -> angle in degrees)
fn hex_angles() -> HashMap<usize, f64> {
    let mut angles = HashMap::new();
    angles.insert(2, 0.0);
    angles.insert(3, 60.0);
    angles.insert(4, 120.0);
    angles.insert(5, 180.0);
    angles.insert(6, 240.0);
    angles.insert(7, 300.0);
    angles
}

/// Generate all 5040 permutations for the 7 nodes: \[1,2,3,4,5,6,7\]
/// (Index 1 = Center, 2-7 = Hexagon)
pub fn generate_s7_permutations() -> Vec<Vec<usize>> {
    let nodes: Vec<usize> = (1..=7).collect();
    let mut result = Vec::new();
    permute_helper(&nodes, 0, &mut result);
    result
}

// Helper function for generating permutations
fn permute_helper(arr: &[usize], start: usize, result: &mut Vec<Vec<usize>>) {
    if start == arr.len() {
        result.push(arr.to_vec());
        return;
    }

    let mut arr = arr.to_vec();
    for i in start..arr.len() {
        arr.swap(start, i);
        permute_helper(&arr, start + 1, result);
        arr.swap(start, i);
    }
}

/// Construct a permutation matrix for a given permutation (legacy name)
///
/// # Arguments
///
/// * `sigma` - A permutation of a subset of indices 1..13
///
/// # Returns
///
/// A 13×13 permutation matrix
pub fn permutation_to_matrix(sigma: &[usize]) -> Array2<f64> {
    permutation_matrix(sigma, 13)
}

/// Construct a full size×size permutation matrix for a given permutation
///
/// # Arguments
///
/// * `sigma` - A permutation of a subset of indices 1..size. Entries not
///   included in sigma are assumed to map to themselves.
/// * `size` - The dimension of the permutation matrix. Defaults to 13.
///
/// # Returns
///
/// A binary matrix P of shape (size, size) such that P[i-1,j-1] = 1 if
/// the permutation maps i to j.
pub fn permutation_matrix(sigma: &[usize], size: usize) -> Array2<f64> {
    let mut p = Array2::eye(size);

    for (src_pos, &tgt_index) in sigma.iter().enumerate() {
        let i = src_pos + 1;
        // Clear the row
        for col in 0..size {
            p[[i - 1, col]] = 0.0;
        }
        // Set the target column
        p[[i - 1, tgt_index - 1]] = 1.0;
    }

    p
}

/// Apply a permutation matrix to an adjacency matrix
///
/// Returns: A' = P @ A @ P^T
pub fn apply_permutation_to_adjacency(a: &Array2<f64>, p: &Array2<f64>) -> Array2<f64> {
    p.dot(a).dot(&p.t())
}

/// Return the permutation corresponding to a k×60° rotation of the hexagon
///
/// The centre node (1) remains fixed. Only nodes 2–7 are rotated. The
/// returned tuple has length 7 and can be used with `permutation_to_matrix`.
///
/// # Arguments
///
/// * `k` - Number of 60° steps to rotate by. Values are taken modulo 6.
///
/// # Returns
///
/// A permutation of (1..7) where node 1 maps to itself and nodes 2..7 are
/// cyclically shifted.
pub fn hexagon_rotation(k: i32) -> Vec<usize> {
    let k = k.rem_euclid(6); // Handle negative k
    let mut mapping = vec![1]; // Centre stays fixed

    for i in 0..6 {
        // Compute index in 2..7 after rotation
        let new_index = 2 + ((i + k as usize) % 6);
        mapping.push(new_index);
    }

    mapping
}

/// Return the reflection permutation across the axis through centre and axis_node
///
/// Reflection axes are defined by node indices 2–7. The centre node (1) stays
/// fixed; cube nodes (8–13) are unaffected and thus not included in this
/// 7-tuple representation. The reflection of each hexagon node is computed by
/// mirroring its angle around the axis angle.
///
/// # Arguments
///
/// * `axis_node` - Index of the hexagon node (2–7) defining the reflection axis.
///
/// # Returns
///
/// A permutation of (1..7) describing the reflection.
pub fn hexagon_reflection(axis_node: usize) -> Vec<usize> {
    let angles = hex_angles();

    if !angles.contains_key(&axis_node) {
        panic!("axis_node must be one of the hexagon nodes 2..7");
    }

    let axis_angle = angles[&axis_node];

    // Precompute inverse mapping from angle to node index
    let angle_to_node: HashMap<i32, usize> = angles
        .iter()
        .map(|(&k, &v)| ((v.round() as i32), k))
        .collect();

    let mut mapping = vec![1]; // Centre stays fixed

    for node in 2..=7 {
        let ang = angles[&node];
        let new_ang = (2.0 * axis_angle - ang) % 360.0;
        // Round to nearest integer angle (multiple of 60)
        let new_ang_round = ((new_ang / 60.0).round() * 60.0) as i32 % 360;
        let new_ang_round = if new_ang_round < 0 {
            new_ang_round + 360
        } else {
            new_ang_round
        };
        mapping.push(angle_to_node[&new_ang_round]);
    }

    mapping
}

/// Generate the 6 rotations of the hexagon (cyclic group C6)
pub fn generate_c6_subgroup() -> Vec<Vec<usize>> {
    (0..6).map(hexagon_rotation).collect()
}

/// Generate the 12 elements of the dihedral group D6 acting on the hexagon
///
/// D6 consists of the 6 rotations and 6 reflections around axes through
/// centre and one hexagon vertex.
pub fn generate_d6_subgroup() -> Vec<Vec<usize>> {
    let mut result = generate_c6_subgroup();
    let reflections: Vec<Vec<usize>> = (2..=7).map(hexagon_reflection).collect();
    result.extend(reflections);
    result
}

/// Determine whether a permutation is even
///
/// Uses the parity of inversion count: a permutation is even if the
/// number of inversions is even.
fn is_even_permutation(seq: &[usize]) -> bool {
    let mut inv_count = 0;
    for i in 0..seq.len() {
        for j in (i + 1)..seq.len() {
            if seq[i] > seq[j] {
                inv_count += 1;
            }
        }
    }
    inv_count % 2 == 0
}

/// Extend a permutation on a subset to a full permutation of 1..total_n
///
/// # Arguments
///
/// * `partial` - A permutation of the values in `subset`
/// * `subset` - The original indices being permuted. `partial` must contain
///   the same elements as `subset`.
/// * `total_n` - The total number of nodes (default 13)
///
/// # Returns
///
/// A permutation vector of length `total_n` where indices in `subset` are
/// permuted according to `partial` and all others map to themselves.
fn extend_partial_permutation(partial: &[usize], subset: &[usize], total_n: usize) -> Vec<usize> {
    let partial_set: std::collections::HashSet<_> = partial.iter().collect();
    let subset_set: std::collections::HashSet<_> = subset.iter().collect();

    if partial_set != subset_set {
        panic!("partial must contain exactly the elements of subset");
    }

    // Build mapping from original to new labels within subset
    let mapping: HashMap<usize, usize> = subset
        .iter()
        .zip(partial.iter())
        .map(|(&old, &new)| (old, new))
        .collect();

    (1..=total_n)
        .map(|i| *mapping.get(&i).unwrap_or(&i))
        .collect()
}

/// Generate the full symmetric group on a subset of node indices
///
/// # Arguments
///
/// * `subset` - The node indices to permute (e.g. vec![8, 9, 10, 12] for four cube nodes)
/// * `total_n` - Total number of nodes in the graph. Defaults to 13.
///
/// # Returns
///
/// A list of permutation vectors of length `total_n`
pub fn generate_symmetric_group(subset: &[usize], total_n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut subset_copy = subset.to_vec();
    let original_subset = subset.to_vec();
    permute_and_extend(&mut subset_copy, 0, &original_subset, total_n, &mut result);
    result
}

// Helper to generate permutations and extend them
fn permute_and_extend(
    arr: &mut [usize],
    start: usize,
    original_subset: &[usize],
    total_n: usize,
    result: &mut Vec<Vec<usize>>,
) {
    if start == arr.len() {
        result.push(extend_partial_permutation(arr, original_subset, total_n));
        return;
    }

    for i in start..arr.len() {
        arr.swap(start, i);
        permute_and_extend(arr, start + 1, original_subset, total_n, result);
        arr.swap(start, i);
    }
}

/// Generate the alternating (even) permutations on a subset of node indices
///
/// # Arguments
///
/// * `subset` - The node indices to permute
/// * `total_n` - Total number of nodes. Defaults to 13.
///
/// # Returns
///
/// A list of even permutation vectors of length `total_n`
pub fn generate_alternating_group(subset: &[usize], total_n: usize) -> Vec<Vec<usize>> {
    let all_perms = generate_symmetric_group(subset, total_n);
    all_perms
        .into_iter()
        .filter(|p| is_even_permutation(&p[..subset.len()]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_s7_permutations_count() {
        let perms = generate_s7_permutations();
        assert_eq!(perms.len(), 5040); // 7! = 5040
    }

    #[test]
    fn test_permutation_matrix_identity() {
        let sigma = vec![1, 2, 3, 4, 5, 6, 7];
        let p = permutation_matrix(&sigma, 13);

        // Identity permutation should give identity matrix
        for i in 0..13 {
            for j in 0..13 {
                if i == j {
                    assert_eq!(p[[i, j]], 1.0);
                } else {
                    assert_eq!(p[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_permutation_matrix_swap() {
        let sigma = vec![1, 3, 2]; // Swap nodes 2 and 3
        let p = permutation_matrix(&sigma, 13);

        // Check that rows 1 and 2 are swapped
        assert_eq!(p[[1, 2]], 1.0);
        assert_eq!(p[[2, 1]], 1.0);
        assert_eq!(p[[1, 1]], 0.0);
        assert_eq!(p[[2, 2]], 0.0);
    }

    #[test]
    fn test_hexagon_rotation_identity() {
        let rot = hexagon_rotation(0);
        assert_eq!(rot, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_hexagon_rotation_once() {
        let rot = hexagon_rotation(1);
        assert_eq!(rot, vec![1, 3, 4, 5, 6, 7, 2]); // Rotate 60°
    }

    #[test]
    fn test_hexagon_rotation_full_cycle() {
        let rot = hexagon_rotation(6);
        assert_eq!(rot, hexagon_rotation(0)); // Full cycle
    }

    #[test]
    fn test_hexagon_reflection() {
        let refl = hexagon_reflection(2);
        // Reflection through node 2 (0°)
        // Node 2 stays, nodes symmetric around 0° are swapped
        assert_eq!(refl[0], 1); // Centre stays
        assert_eq!(refl[1], 2); // Node 2 (0°) stays on axis
    }

    #[test]
    fn test_generate_c6_subgroup_count() {
        let c6 = generate_c6_subgroup();
        assert_eq!(c6.len(), 6); // C6 has 6 elements
    }

    #[test]
    fn test_generate_d6_subgroup_count() {
        let d6 = generate_d6_subgroup();
        assert_eq!(d6.len(), 12); // D6 has 12 elements (6 rotations + 6 reflections)
    }

    #[test]
    fn test_is_even_permutation_identity() {
        let perm = vec![1, 2, 3, 4, 5];
        assert!(is_even_permutation(&perm)); // Identity is even
    }

    #[test]
    fn test_is_even_permutation_single_swap() {
        let perm = vec![2, 1, 3, 4, 5];
        assert!(!is_even_permutation(&perm)); // Single transposition is odd
    }

    #[test]
    fn test_extend_partial_permutation() {
        let partial = vec![3, 2, 4]; // Permutation on nodes 2, 3, 4
        let subset = vec![2, 3, 4];
        let extended = extend_partial_permutation(&partial, &subset, 7);

        assert_eq!(extended[0], 1); // Node 1 unchanged
        assert_eq!(extended[1], 3); // Node 2 -> 3
        assert_eq!(extended[2], 2); // Node 3 -> 2
        assert_eq!(extended[3], 4); // Node 4 -> 4
        assert_eq!(extended[4], 5); // Node 5 unchanged
    }

    #[test]
    fn test_generate_symmetric_group_small() {
        let subset = vec![2, 3];
        let group = generate_symmetric_group(&subset, 5);
        assert_eq!(group.len(), 2); // S2 has 2 elements

        // Check both permutations exist
        let perm1 = vec![1, 2, 3, 4, 5];
        let perm2 = vec![1, 3, 2, 4, 5];
        assert!(group.contains(&perm1));
        assert!(group.contains(&perm2));
    }

    #[test]
    fn test_generate_alternating_group_small() {
        let subset = vec![2, 3, 4];
        let group = generate_alternating_group(&subset, 5);
        assert_eq!(group.len(), 3); // A3 has 3 elements (half of S3)
    }

    #[test]
    fn test_apply_permutation_to_adjacency() {
        // Create a simple adjacency matrix
        let mut a = Array2::zeros((3, 3));
        a[[0, 1]] = 1.0;
        a[[1, 0]] = 1.0;

        // Create identity permutation
        let p = Array2::eye(3);
        let a_prime = apply_permutation_to_adjacency(&a, &p);

        // Should be unchanged
        assert_eq!(a_prime, a);
    }
}
