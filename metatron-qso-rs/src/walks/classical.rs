//! Classical random-walk baselines.

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{RMatrix, eigh_real};

/// Continuous-time classical random walk `p(t) = exp(−r·t·L)·e_start` with
/// rate `r` per unit edge weight.
pub fn classical_walk_probabilities(
    graph: &Graph,
    start: usize,
    t: f64,
    rate: f64,
) -> Result<Vec<f64>> {
    let n = graph.num_nodes();
    if start >= n {
        return Err(QsoError::NodeOutOfRange {
            node: start,
            n_nodes: n,
        });
    }
    if graph.edges().iter().any(|&(_, _, w)| w < 0.0) {
        return Err(QsoError::invalid(
            "graph",
            "classical walks need non-negative weights",
        ));
    }
    let (values, vectors) = eigh_real(&graph.laplacian_matrix());
    Ok((0..n)
        .map(|j| {
            (0..n)
                .map(|k| vectors[(j, k)] * vectors[(start, k)] * (-rate * t * values[k]).exp())
                .sum::<f64>()
                .max(0.0)
        })
        .collect())
}

/// Expected number of steps for the discrete-time simple random walk
/// (transition probabilities proportional to edge weights) to first reach
/// `target`, from every node. Unreachable nodes get `f64::INFINITY`.
pub fn classical_hitting_times(graph: &Graph, target: usize) -> Result<Vec<f64>> {
    let n = graph.num_nodes();
    if target >= n {
        return Err(QsoError::NodeOutOfRange {
            node: target,
            n_nodes: n,
        });
    }
    if graph.edges().iter().any(|&(_, _, w)| w <= 0.0) {
        return Err(QsoError::invalid(
            "graph",
            "hitting times need positive weights",
        ));
    }
    let reachable: Vec<bool> = graph
        .distances_from(target)
        .iter()
        .map(Option::is_some)
        .collect();
    let others: Vec<usize> = (0..n).filter(|&u| u != target && reachable[u]).collect();
    let index = |u: usize| others.binary_search(&u).ok();
    let m = others.len();
    let mut a = RMatrix::identity(m, m);
    let b = nalgebra::DVector::from_element(m, 1.0);
    for (row, &u) in others.iter().enumerate() {
        let deg = graph.weighted_degree(u);
        for &(v, w) in graph.neighbors(u) {
            if let Some(col) = index(v) {
                a[(row, col)] -= w / deg;
            }
        }
    }
    let h = a
        .lu()
        .solve(&b)
        .ok_or_else(|| QsoError::invalid("graph", "singular hitting-time system"))?;
    let mut out = vec![f64::INFINITY; n];
    out[target] = 0.0;
    for (row, &u) in others.iter().enumerate() {
        out[u] = h[row];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;

    #[test]
    fn hitting_times_match_closed_forms() {
        // K_n: n − 1 from every other node.
        let h = classical_hitting_times(&generators::complete(13), 0).unwrap();
        assert!(h[1..].iter().all(|&x| (x - 12.0).abs() < 1e-9));
        // C_n: a walker at distance d needs d(n − d) steps.
        let n = 10;
        let h = classical_hitting_times(&generators::cycle(n), 0).unwrap();
        for (u, &x) in h.iter().enumerate() {
            let d = u.min(n - u) as f64;
            assert!((x - d * (n as f64 - d)).abs() < 1e-9);
        }
    }

    #[test]
    fn classical_walk_conserves_probability_and_equilibrates() {
        let g = generators::petersen();
        let p = classical_walk_probabilities(&g, 0, 0.8, 1.0).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let p = classical_walk_probabilities(&g, 0, 100.0, 1.0).unwrap();
        assert!(p.iter().all(|&x| (x - 0.1).abs() < 1e-9));
    }
}
