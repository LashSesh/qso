use crate::error::{EngineError, EngineResult};
use crate::geometry::Node;

pub fn generate_s7_permutations() -> Vec<Vec<usize>> {
    let elements: Vec<usize> = (1..=7).collect();
    permutations_list(&elements)
}

pub fn permutation_to_matrix(sigma: &[usize]) -> Vec<Vec<f64>> {
    let size = 13;
    let mut matrix = vec![vec![0.0; size]; size];
    for (src, &tgt) in sigma.iter().enumerate() {
        matrix[src][tgt - 1] = 1.0;
    }
    for idx in sigma.len()..size {
        matrix[idx][idx] = 1.0;
    }
    matrix
}

pub fn permutation_matrix(sigma: &[usize], size: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; size]; size];
    for (row, &target) in sigma.iter().enumerate() {
        matrix[row][target - 1] = 1.0;
    }
    for idx in sigma.len()..size {
        matrix[idx][idx] = 1.0;
    }
    matrix
}

pub fn apply_permutation_to_adjacency(adjacency: &[Vec<f64>], p: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = adjacency.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                for l in 0..n {
                    sum += p[i][k] * adjacency[k][l] * p[j][l];
                }
            }
            result[i][j] = sum;
        }
    }
    result
}

pub fn permute_node_labels(nodes: &[Node], sigma: &[usize]) -> Vec<Node> {
    let mut result = Vec::with_capacity(nodes.len());
    for (idx, &src) in sigma.iter().enumerate() {
        if idx < nodes.len() {
            result.push(nodes[src - 1].clone());
        }
    }
    if sigma.len() < nodes.len() {
        for node in &nodes[sigma.len()..] {
            result.push(node.clone());
        }
    }
    result
}

pub fn hexagon_rotation(k: i32) -> Vec<usize> {
    let mut mapping = vec![1];
    let shift = ((k % 6) + 6) % 6;
    for i in 0..6 {
        let new_index = 2 + ((i + shift as usize) % 6);
        mapping.push(new_index);
    }
    mapping
}

pub fn hexagon_reflection(axis_node: usize) -> EngineResult<Vec<usize>> {
    const ANGLES: [(usize, f64); 6] = [
        (2, 0.0),
        (3, 60.0),
        (4, 120.0),
        (5, 180.0),
        (6, 240.0),
        (7, 300.0),
    ];
    if !ANGLES.iter().any(|&(node, _)| node == axis_node) {
        return Err(EngineError::Misconfigured(
            "axis_node must be between 2 and 7".into(),
        ));
    }
    let axis_angle = ANGLES.iter().find(|&&(n, _)| n == axis_node).unwrap().1;
    let mut mapping = vec![1];
    for &(_, angle) in &ANGLES {
        let mirrored = (2.0 * axis_angle - angle).rem_euclid(360.0);
        let rounded = (mirrored / 60.0).round() * 60.0 % 360.0;
        let target = ANGLES
            .iter()
            .find(|&&(_, ang)| (ang - rounded).abs() < 1e-6)
            .map(|&(n, _)| n)
            .unwrap();
        mapping.push(target);
    }
    Ok(mapping)
}

pub fn generate_c6_subgroup() -> Vec<Vec<usize>> {
    (0..6).map(hexagon_rotation).collect()
}

pub fn generate_d6_subgroup() -> EngineResult<Vec<Vec<usize>>> {
    let mut group = generate_c6_subgroup();
    for axis in 2..=7 {
        group.push(hexagon_reflection(axis)?);
    }
    Ok(group)
}

fn is_even_permutation(seq: &[usize]) -> bool {
    let mut inv = 0;
    for i in 0..seq.len() {
        for j in (i + 1)..seq.len() {
            if seq[i] > seq[j] {
                inv += 1;
            }
        }
    }
    inv % 2 == 0
}

fn extend_partial_permutation(
    partial: &[usize],
    subset: &[usize],
    total_n: usize,
) -> EngineResult<Vec<usize>> {
    let mut p = partial.to_vec();
    p.sort();
    let mut s = subset.to_vec();
    s.sort();
    if p != s {
        return Err(EngineError::InvalidPermutation);
    }
    let mapping: std::collections::HashMap<usize, usize> = subset
        .iter()
        .copied()
        .zip(partial.iter().copied())
        .collect();
    Ok((1..=total_n)
        .map(|i| *mapping.get(&i).unwrap_or(&i))
        .collect())
}

fn permutations_list(elements: &[usize]) -> Vec<Vec<usize>> {
    if elements.is_empty() {
        return vec![Vec::new()];
    }
    let mut result = Vec::new();
    for (idx, &item) in elements.iter().enumerate() {
        let mut rest = elements.to_vec();
        rest.remove(idx);
        for mut perm in permutations_list(&rest) {
            let mut new_perm = Vec::with_capacity(elements.len());
            new_perm.push(item);
            new_perm.append(&mut perm);
            result.push(new_perm);
        }
    }
    result
}

pub fn generate_symmetric_group(subset: &[usize], total_n: usize) -> EngineResult<Vec<Vec<usize>>> {
    let mut subset_vec = subset.to_vec();
    subset_vec.sort();
    let mut result = Vec::new();
    for perm in permutations_list(&subset_vec) {
        result.push(extend_partial_permutation(&perm, &subset_vec, total_n)?);
    }
    Ok(result)
}

pub fn generate_alternating_group(
    subset: &[usize],
    total_n: usize,
) -> EngineResult<Vec<Vec<usize>>> {
    let mut subset_vec = subset.to_vec();
    subset_vec.sort();
    let mut result = Vec::new();
    for perm in permutations_list(&subset_vec)
        .into_iter()
        .filter(|p| is_even_permutation(p))
    {
        result.push(extend_partial_permutation(&perm, &subset_vec, total_n)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_length() {
        let rot = hexagon_rotation(1);
        assert_eq!(rot.len(), 7);
        assert_eq!(rot[0], 1);
    }
}
