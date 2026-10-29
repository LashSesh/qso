//! Standard graph families.

use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use super::Graph;
use crate::error::{QsoError, Result};
use crate::rng::Rng;

fn circle_positions(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|k| {
            let a = TAU * k as f64 / n as f64;
            [a.cos(), a.sin(), 0.0]
        })
        .collect()
}

fn with_layout(g: Graph, positions: Vec<[f64; 3]>) -> Graph {
    g.with_positions(positions).expect("one position per node")
}

/// Complete graph `K_n`.
pub fn complete(n: usize) -> Graph {
    let mut g = Graph::new(n).with_name(format!("K{n}"));
    for u in 0..n {
        for v in u + 1..n {
            g.add_edge(u, v).expect("valid edge");
        }
    }
    with_layout(g, circle_positions(n))
}

/// Cycle `C_n` (`n ≥ 3`).
pub fn cycle(n: usize) -> Graph {
    assert!(n >= 3, "a cycle needs at least 3 nodes");
    let mut g = Graph::new(n).with_name(format!("C{n}"));
    for u in 0..n {
        g.add_edge(u, (u + 1) % n).expect("valid edge");
    }
    with_layout(g, circle_positions(n))
}

/// Path `P_n`.
pub fn path(n: usize) -> Graph {
    let mut g = Graph::new(n).with_name(format!("P{n}"));
    for u in 1..n {
        g.add_edge(u - 1, u).expect("valid edge");
    }
    with_layout(g, (0..n).map(|u| [u as f64, 0.0, 0.0]).collect())
}

/// Star `S_n`: node 0 joined to nodes `1..n`.
pub fn star(n: usize) -> Graph {
    let mut g = Graph::new(n).with_name(format!("star({n})"));
    for u in 1..n {
        g.add_edge(0, u).expect("valid edge");
    }
    if n == 0 {
        return g;
    }
    let mut pos = vec![[0.0; 3]];
    pos.extend(circle_positions(n - 1));
    with_layout(g, pos)
}

/// Complete bipartite graph `K_{a,b}`.
pub fn complete_bipartite(a: usize, b: usize) -> Graph {
    let mut g = Graph::new(a + b).with_name(format!("K{a},{b}"));
    for u in 0..a {
        for v in a..a + b {
            g.add_edge(u, v).expect("valid edge");
        }
    }
    g
}

/// `d`-dimensional hypercube `Q_d` (2^d nodes; neighbours differ in one bit).
pub fn hypercube(d: usize) -> Graph {
    assert!(d <= 20, "hypercube dimension too large");
    let n = 1usize << d;
    let mut g = Graph::new(n).with_name(format!("Q{d}"));
    for u in 0..n {
        for k in 0..d {
            let v = u ^ (1 << k);
            if u < v {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    g
}

/// `rows × cols` grid; with `periodic` a torus (each dimension ≥ 3).
pub fn grid_2d(rows: usize, cols: usize, periodic: bool) -> Graph {
    if periodic {
        assert!(
            rows >= 3 && cols >= 3,
            "a torus needs at least 3 nodes per dimension"
        );
    }
    let n = rows * cols;
    let name = if periodic {
        format!("torus({rows}x{cols})")
    } else {
        format!("grid({rows}x{cols})")
    };
    let mut g = Graph::new(n).with_name(name);
    let id = |r: usize, c: usize| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols || periodic {
                g.add_edge(id(r, c), id(r, (c + 1) % cols))
                    .expect("valid edge");
            }
            if r + 1 < rows || periodic {
                g.add_edge(id(r, c), id((r + 1) % rows, c))
                    .expect("valid edge");
            }
        }
    }
    let pos = (0..n)
        .map(|u| [(u % cols) as f64, (u / cols) as f64, 0.0])
        .collect();
    with_layout(g, pos)
}

/// The Petersen graph (10 nodes, 15 edges, |Aut| = 120).
pub fn petersen() -> Graph {
    let mut g = Graph::new(10).with_name("Petersen");
    for k in 0..5 {
        g.add_edge(k, (k + 1) % 5).expect("outer cycle");
        g.add_edge(k, k + 5).expect("spoke");
        g.add_edge(5 + k, 5 + (k + 2) % 5).expect("inner pentagram");
    }
    let mut pos = circle_positions(5);
    pos.extend(
        circle_positions(5)
            .into_iter()
            .map(|[x, y, z]| [0.5 * x, 0.5 * y, z]),
    );
    with_layout(g, pos)
}

/// The Heawood graph: the (3, 6)-cage, i.e. the smallest cubic graph of
/// girth 6 (14 nodes, 21 edges, bipartite, |Aut| = 336). LCF notation
/// `[5, −5]⁷`.
pub fn heawood() -> Graph {
    let mut g = Graph::new(14).with_name("Heawood");
    for k in 0..14 {
        g.add_edge(k, (k + 1) % 14).expect("Hamiltonian cycle");
        if k % 2 == 0 {
            g.add_edge(k, (k + 5) % 14).expect("chord");
        }
    }
    with_layout(g, circle_positions(14))
}

/// Random `d`-regular simple graph on `n` nodes (pairing model with
/// rejection; `n·d` must be even).
pub fn random_regular(n: usize, d: usize, rng: &mut Rng) -> Result<Graph> {
    let stubs_total = n
        .checked_mul(d)
        .ok_or_else(|| QsoError::invalid("n/d", "n·d overflows"))?;
    if !stubs_total.is_multiple_of(2) || d >= n {
        return Err(QsoError::invalid("n/d", "need n·d even and d < n"));
    }
    // Pairings succeed with probability ≈ e^{(1−d²)/4}, independent of n;
    // bounding the work (not only the attempts) keeps hopeless requests
    // from running for minutes. Seeds that succeed within the bound give
    // the same graph as before.
    let attempts = (100_000_000 / stubs_total.max(1)).clamp(10, 10_000);
    'attempt: for _ in 0..attempts {
        let mut stubs: Vec<usize> = (0..n).flat_map(|u| std::iter::repeat_n(u, d)).collect();
        rng.shuffle(&mut stubs);
        let mut g = Graph::new(n).with_name(format!("random_regular({n},{d})"));
        for pair in stubs.chunks(2) {
            if pair[0] == pair[1] || g.has_edge(pair[0], pair[1]) {
                continue 'attempt;
            }
            g.add_edge(pair[0], pair[1]).expect("checked");
        }
        return Ok(g);
    }
    Err(QsoError::invalid(
        "n/d",
        "failed to sample a simple regular graph",
    ))
}

/// Erdős–Rényi graph `G(n, p)`.
pub fn erdos_renyi(n: usize, p: f64, rng: &mut Rng) -> Result<Graph> {
    if !(0.0..=1.0).contains(&p) {
        return Err(QsoError::invalid(
            "p",
            "edge probability must lie in [0, 1]",
        ));
    }
    let mut g = Graph::new(n).with_name(format!("G({n},{p})"));
    for u in 0..n {
        for v in u + 1..n {
            if rng.uniform() < p {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    Ok(g)
}

/// Binomial coefficient `C(n, k)` as `u128`, saturating.
fn binomial(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c: u128 = 1;
    for i in 0..k {
        c = c.saturating_mul((n - i) as u128) / (i + 1) as u128;
    }
    c
}

/// All `k`-element subsets of `0..n` as bit masks, in colex order (the
/// position of a subset is its rank in the combinatorial number system).
fn subsets(n: usize, k: usize) -> Vec<u64> {
    let mut out = Vec::new();
    if k > n {
        return out;
    }
    if k == 0 {
        return vec![0];
    }
    // Gosper's hack enumerates masks with k bits in increasing order,
    // which is colex order.
    let mut x: u64 = (1u64 << k) - 1;
    let limit: u64 = 1u64 << n;
    while x < limit {
        out.push(x);
        let c = x & x.wrapping_neg();
        let r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
    out
}

/// Rank of a subset in colex order: `Σ C(c_i, i + 1)` over its sorted
/// elements `c_0 < c_1 < …`.
fn colex_rank(mut mask: u64) -> usize {
    let mut rank = 0u128;
    let mut i = 0u64;
    while mask != 0 {
        let c = mask.trailing_zeros() as u64;
        rank += binomial(c, i + 1);
        i += 1;
        mask &= mask - 1;
    }
    rank as usize
}

/// Johnson graph `J(n, k)`: the `k`-subsets of `{0, …, n−1}`, adjacent when
/// they share `k − 1` elements (distance-regular; `J(n, 2)` is the
/// triangular graph). Nodes are the subsets in colex order.
pub fn johnson(n: usize, k: usize) -> Result<Graph> {
    if !(1..=62).contains(&n) || k == 0 || k >= n {
        return Err(QsoError::invalid("n/k", "need 0 < k < n ≤ 62"));
    }
    let nodes = binomial(n as u64, k as u64);
    check_generated_size(nodes, nodes * (k * (n - k)) as u128 / 2)?;
    let sets = subsets(n, k);
    let mut g = Graph::new(sets.len()).with_name(format!("J({n},{k})"));
    for (a, &set) in sets.iter().enumerate() {
        // Swap one element out and one in.
        for out in (0..n).filter(|&i| set >> i & 1 == 1) {
            for inn in (0..n).filter(|&i| set >> i & 1 == 0) {
                let b = colex_rank(set & !(1 << out) | 1 << inn);
                if a < b {
                    g.add_edge(a, b).expect("valid edge");
                }
            }
        }
    }
    Ok(g)
}

/// Kneser graph `K(n, k)`: the `k`-subsets of `{0, …, n−1}`, adjacent when
/// disjoint (`K(5, 2)` is the Petersen graph). Nodes in colex order.
pub fn kneser(n: usize, k: usize) -> Result<Graph> {
    if !(1..=62).contains(&n) || k == 0 || 2 * k > n {
        return Err(QsoError::invalid("n/k", "need 0 < 2k ≤ n ≤ 62"));
    }
    let nodes = binomial(n as u64, k as u64);
    let degree = binomial((n - k) as u64, k as u64);
    check_generated_size(nodes, nodes.saturating_mul(degree) / 2)?;
    let sets = subsets(n, k);
    let mut g = Graph::new(sets.len()).with_name(format!("Kneser({n},{k})"));
    let full: u64 = (1u64 << n) - 1;
    for (a, &set) in sets.iter().enumerate() {
        // Neighbours: the k-subsets of the complement.
        let complement = full & !set;
        let free: Vec<usize> = (0..n).filter(|&i| complement >> i & 1 == 1).collect();
        for sub in subsets(free.len(), k) {
            let mask = (0..free.len())
                .filter(|&j| sub >> j & 1 == 1)
                .fold(0u64, |m, j| m | 1 << free[j]);
            let b = colex_rank(mask);
            if a < b {
                g.add_edge(a, b).expect("valid edge");
            }
        }
    }
    Ok(g)
}

/// Hamming graph `H(d, q)`: words of length `d` over `q` symbols, adjacent
/// when they differ in exactly one position (`H(d, 2)` is the hypercube,
/// `H(2, q)` the `q × q` rook's graph). Node `x` has digits `x` in base `q`.
pub fn hamming(d: usize, q: usize) -> Result<Graph> {
    if d == 0 || q < 2 {
        return Err(QsoError::invalid("d/q", "need d ≥ 1 and q ≥ 2"));
    }
    let nodes = u32::try_from(d)
        .ok()
        .and_then(|e| (q as u128).checked_pow(e))
        .unwrap_or(u128::MAX);
    let degree = (d as u128).saturating_mul(q as u128 - 1);
    check_generated_size(nodes, nodes.saturating_mul(degree) / 2)?;
    let n = nodes as usize;
    let mut g = Graph::new(n).with_name(format!("H({d},{q})"));
    for x in 0..n {
        let mut place = 1;
        for _ in 0..d {
            let digit = (x / place) % q;
            for other in digit + 1..q {
                g.add_edge(x, x + (other - digit) * place)
                    .expect("valid edge");
            }
            place *= q;
        }
    }
    Ok(g)
}

/// Rook's graph `K_m □ K_n`: squares of an `m × n` board, adjacent when in
/// the same row or column. `rook(4, 4)` is strongly regular (16, 6, 2, 2),
/// cospectral with but not isomorphic to the Shrikhande graph.
pub fn rook(m: usize, n: usize) -> Result<Graph> {
    if m == 0 || n == 0 {
        return Err(QsoError::invalid("m/n", "board sides must be positive"));
    }
    let nodes = m as u128 * n as u128;
    check_generated_size(nodes, nodes * (m + n - 2) as u128 / 2)?;
    Ok(complete(m)
        .cartesian_product(&complete(n))?
        .with_name(format!("rook({m}x{n})")))
}

/// Circulant graph (Cayley graph of `Z_n`): node `u` is joined to
/// `u ± s (mod n)` for every jump `s` (`1 ≤ s ≤ n/2`). Cycles and complete
/// graphs are circulants; all circulants are diagonalised by the Fourier
/// transform.
pub fn circulant(n: usize, jumps: &[usize]) -> Result<Graph> {
    if n < 2 {
        return Err(QsoError::invalid("n", "need at least 2 nodes"));
    }
    let mut sorted = jumps.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != jumps.len() || sorted.iter().any(|&s| s == 0 || s > n / 2) {
        return Err(QsoError::invalid("jumps", "need distinct jumps in 1..=n/2"));
    }
    check_generated_size(n as u128, n as u128 * jumps.len() as u128)?;
    let mut g = Graph::new(n).with_name(format!("circulant({n};{jumps:?})"));
    for u in 0..n {
        for &s in &sorted {
            let v = (u + s) % n;
            if !g.has_edge(u, v) {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    Ok(with_layout(g, circle_positions(n)))
}

/// Cayley graph of `Z_2^d` ("cubelike" graph): node `x` is joined to
/// `x ⊕ g` for every generator bit mask `g` (`H(d, 2)` uses the unit
/// vectors). All such graphs are diagonalised by the Walsh–Hadamard
/// transform.
pub fn cayley_z2(d: usize, generators: &[u64]) -> Result<Graph> {
    if d == 0 || d > 20 {
        return Err(QsoError::invalid("d", "need 1 ≤ d ≤ 20"));
    }
    let mut sorted = generators.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != generators.len() || sorted.iter().any(|&x| x == 0 || x >> d != 0) {
        return Err(QsoError::invalid(
            "generators",
            "need distinct non-zero masks below 2^d",
        ));
    }
    let n = 1usize << d;
    check_generated_size(n as u128, n as u128 * generators.len() as u128 / 2)?;
    let mut g = Graph::new(n).with_name(format!("cayley_z2({d};{generators:?})"));
    for x in 0..n {
        for &mask in &sorted {
            let y = x ^ mask as usize;
            if x < y {
                g.add_edge(x, y).expect("valid edge");
            }
        }
    }
    Ok(g)
}

/// Paley graph of a prime `q ≡ 1 (mod 4)`: `Z_q`, with `u ~ v` when `u − v`
/// is a non-zero square. Strongly regular with parameters
/// `(q, (q−1)/2, (q−5)/4, (q−1)/4)`; self-complementary.
pub fn paley(q: usize) -> Result<Graph> {
    check_generated_size(q as u128, (q as u128) * (q as u128).saturating_sub(1) / 4)?;
    let prime = q >= 5
        && (2..)
            .take_while(|&d| d <= q / d)
            .all(|d| !q.is_multiple_of(d));
    if !prime || q % 4 != 1 {
        return Err(QsoError::invalid("q", "need a prime q ≡ 1 (mod 4)"));
    }
    let mut square = vec![false; q];
    for x in 1..q {
        square[x * x % q] = true;
    }
    let mut g = Graph::new(q).with_name(format!("Paley({q})"));
    for u in 0..q {
        for v in u + 1..q {
            if square[v - u] {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    Ok(with_layout(g, circle_positions(q)))
}

/// The Shrikhande graph: the Cayley graph of `Z_4 × Z_4` with connection
/// set `±(1,0), ±(0,1), ±(1,1)`. Strongly regular (16, 6, 2, 2) like the
/// 4 × 4 rook's graph, but not isomorphic to it (|Aut| = 192 vs 1152).
pub fn shrikhande() -> Graph {
    let id = |a: usize, b: usize| 4 * (a % 4) + b % 4;
    let mut g = Graph::new(16).with_name("Shrikhande");
    for a in 0..4 {
        for b in 0..4 {
            for (da, db) in [(1, 0), (0, 1), (1, 1)] {
                g.add_edge(id(a, b), id(a + da, b + db))
                    .expect("valid edge");
            }
        }
    }
    g
}

/// Seidel switching of an unweighted graph with respect to the node set
/// `set`: every pair with one node inside and one outside the set changes
/// from adjacent to non-adjacent and back; pairs on the same side keep
/// their state. Switching a regular graph can give a non-isomorphic graph
/// with the same parameters (the Chang graphs, [`chang`]).
pub fn seidel_switching(graph: &Graph, set: &[usize]) -> Result<Graph> {
    if !graph.is_unweighted() {
        return Err(QsoError::invalid(
            "graph",
            "Seidel switching needs an unweighted graph",
        ));
    }
    let n = graph.num_nodes();
    let mut inside = vec![false; n];
    for &u in set {
        if u >= n {
            return Err(QsoError::NodeOutOfRange {
                node: u,
                n_nodes: n,
            });
        }
        inside[u] = true;
    }
    let mut g = Graph::new(n).with_name(format!("{} switched", graph.name()));
    for u in 0..n {
        for v in u + 1..n {
            if graph.has_edge(u, v) != (inside[u] != inside[v]) {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    Ok(g)
}

/// The Chang graphs (`which` = 1, 2, 3): strongly regular (28, 12, 6, 4)
/// like the triangular graph `T(8) = J(8, 2)`, but not isomorphic to it
/// or to each other (Chang, 1959). Each is the Seidel switching of `T(8)`
/// with respect to the edges of `K₈` forming four disjoint edges (1), an
/// 8-cycle (2) or a triangle and a 5-cycle (3). Nodes are the pairs of
/// `{0, …, 7}` in the colex order of [`johnson`].
pub fn chang(which: usize) -> Result<Graph> {
    let edges: &[(usize, usize)] = match which {
        1 => &[(0, 1), (2, 3), (4, 5), (6, 7)],
        2 => &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 0),
        ],
        3 => &[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 3),
        ],
        _ => {
            return Err(QsoError::invalid(
                "which",
                "the Chang graphs are 1, 2 and 3",
            ));
        }
    };
    let set: Vec<usize> = edges
        .iter()
        .map(|&(a, b)| colex_rank(1 << a | 1 << b))
        .collect();
    Ok(seidel_switching(&johnson(8, 2)?, &set)?.with_name(format!("Chang {which}")))
}

/// The five Platonic solids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlatonicSolid {
    Tetrahedron,
    Cube,
    Octahedron,
    Dodecahedron,
    Icosahedron,
}

impl PlatonicSolid {
    pub const ALL: [PlatonicSolid; 5] = [
        PlatonicSolid::Tetrahedron,
        PlatonicSolid::Cube,
        PlatonicSolid::Octahedron,
        PlatonicSolid::Dodecahedron,
        PlatonicSolid::Icosahedron,
    ];

    /// Vertex coordinates (centred at the origin).
    pub fn vertices(self) -> Vec<[f64; 3]> {
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let mut v = Vec::new();
        let signs = [1.0, -1.0];
        match self {
            PlatonicSolid::Tetrahedron => {
                v = vec![
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                ];
            }
            PlatonicSolid::Cube => {
                for x in signs {
                    for y in signs {
                        for z in signs {
                            v.push([x, y, z]);
                        }
                    }
                }
            }
            PlatonicSolid::Octahedron => {
                for s in signs {
                    v.push([s, 0.0, 0.0]);
                    v.push([0.0, s, 0.0]);
                    v.push([0.0, 0.0, s]);
                }
            }
            PlatonicSolid::Icosahedron => {
                for a in signs {
                    for b in signs {
                        v.push([0.0, a, b * phi]);
                        v.push([a, b * phi, 0.0]);
                        v.push([b * phi, 0.0, a]);
                    }
                }
            }
            PlatonicSolid::Dodecahedron => {
                for x in signs {
                    for y in signs {
                        for z in signs {
                            v.push([x, y, z]);
                        }
                    }
                }
                for a in signs {
                    for b in signs {
                        v.push([0.0, a / phi, b * phi]);
                        v.push([a / phi, b * phi, 0.0]);
                        v.push([b * phi, 0.0, a / phi]);
                    }
                }
            }
        }
        v
    }

    pub fn name(self) -> &'static str {
        match self {
            PlatonicSolid::Tetrahedron => "tetrahedron",
            PlatonicSolid::Cube => "cube",
            PlatonicSolid::Octahedron => "octahedron",
            PlatonicSolid::Dodecahedron => "dodecahedron",
            PlatonicSolid::Icosahedron => "icosahedron",
        }
    }
}

/// Graph of the edges of a point set: pairs at the minimal distance.
pub(crate) fn unit_distance_graph(points: &[[f64; 3]], name: &str) -> Graph {
    let dist = |a: &[f64; 3], b: &[f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let n = points.len();
    let mut min = f64::INFINITY;
    for u in 0..n {
        for v in u + 1..n {
            min = min.min(dist(&points[u], &points[v]));
        }
    }
    let mut g = Graph::new(n).with_name(name);
    for u in 0..n {
        for v in u + 1..n {
            if (dist(&points[u], &points[v]) - min).abs() <= 1e-9 * min {
                g.add_edge(u, v).expect("valid edge");
            }
        }
    }
    with_layout(g, points.to_vec())
}

/// Edge graph (1-skeleton) of a Platonic solid.
pub fn platonic(solid: PlatonicSolid) -> Graph {
    unit_distance_graph(&solid.vertices(), solid.name())
}

impl std::str::FromStr for PlatonicSolid {
    type Err = QsoError;

    fn from_str(s: &str) -> Result<Self> {
        PlatonicSolid::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(s.trim()))
            .ok_or_else(|| QsoError::Parse(format!("unknown Platonic solid '{s}'")))
    }
}

/// Largest number of nodes of a graph built by [`from_spec`] (and by the
/// Python constructors).
pub const MAX_GENERATED_NODES: usize = 1 << 20;

/// Largest number of edges (for Erdős–Rényi graphs: of candidate pairs) of
/// a graph built by [`from_spec`] (and by the Python constructors).
pub const MAX_GENERATED_EDGES: usize = 1 << 22;

/// Check a requested graph size against [`MAX_GENERATED_NODES`] and
/// [`MAX_GENERATED_EDGES`] before building it (`u128`, so that products of
/// two `usize` arguments cannot overflow).
pub fn check_generated_size(nodes: u128, edges: u128) -> Result<()> {
    if nodes > MAX_GENERATED_NODES as u128 || edges > MAX_GENERATED_EDGES as u128 {
        return Err(QsoError::invalid(
            "graph",
            format!(
                "{nodes} nodes and {edges} edges exceed the supported \
                 {MAX_GENERATED_NODES} nodes or {MAX_GENERATED_EDGES} edges"
            ),
        ));
    }
    Ok(())
}

/// Build a graph from a short textual specification:
///
/// | Spec | Graph |
/// |---|---|
/// | `metatron` | Metatron's Cube (K₁₃ with Fruit-of-Life embedding) |
/// | `fruit-of-life` | Fruit-of-Life contact graph |
/// | `metatron:<max_length>` | lines of Metatron's Cube up to a length |
/// | `complete:N`, `cycle:N`, `path:N`, `star:N` | standard families |
/// | `hypercube:D`, `petersen`, `heawood` | |
/// | `grid:RxC`, `torus:RxC` | rectangular grid / torus |
/// | `bipartite:A,B` | complete bipartite graph |
/// | `platonic:NAME` | tetrahedron, cube, octahedron, dodecahedron, icosahedron |
/// | `johnson:N,K`, `kneser:N,K`, `hamming:D,Q` | distance-regular families |
/// | `paley:Q`, `shrikhande`, `rook:MxN`, `chang:K` | strongly regular graphs |
/// | `circulant:N,S1,S2,…` | Cayley graph of `Z_N` with jumps `±S_i` |
/// | `random-regular:N,D[,SEED]`, `erdos-renyi:N,P[,SEED]` | random graphs |
///
/// Specifications of graphs with more than [`MAX_GENERATED_NODES`] nodes
/// or [`MAX_GENERATED_EDGES`] edges are rejected before anything is built.
pub fn from_spec(spec: &str) -> Result<Graph> {
    let spec = spec.trim();
    let (kind, args) = match spec.split_once(':') {
        Some((k, a)) => (k.trim().to_ascii_lowercase(), a.trim()),
        None => (spec.to_ascii_lowercase(), ""),
    };
    let bad = || QsoError::Parse(format!("invalid graph specification '{spec}'"));
    let nums = |sep: char| -> Result<Vec<f64>> {
        args.split(sep)
            .map(|x| x.trim().parse::<f64>().map_err(|_| bad()))
            .collect()
    };
    let int = |x: f64| -> Result<usize> {
        if x >= 0.0 && x.fract() == 0.0 && x < 1e9 {
            Ok(x as usize)
        } else {
            Err(bad())
        }
    };
    let one = || -> Result<usize> {
        match nums(',')?.as_slice() {
            [n] => int(*n),
            _ => Err(bad()),
        }
    };
    let size = check_generated_size;
    let pairs = |n: usize| n as u128 * (n as u128).saturating_sub(1) / 2;
    let graph = match kind.as_str() {
        "metatron" | "metatron-cube" if args.is_empty() => super::metatron::metatron_cube(),
        "metatron" => super::metatron::distance_graph(args.parse::<f64>().map_err(|_| bad())?),
        "fruit-of-life" | "fruit_of_life" | "fol" => super::metatron::fruit_of_life(),
        "petersen" => petersen(),
        "heawood" => heawood(),
        "shrikhande" => shrikhande(),
        "chang" => chang(one()?)?,
        "johnson" | "kneser" | "hamming" => match nums(',')?.as_slice() {
            [a, b] => {
                let (a, b) = (int(*a)?, int(*b)?);
                match kind.as_str() {
                    "johnson" => johnson(a, b)?,
                    "kneser" => kneser(a, b)?,
                    _ => hamming(a, b)?,
                }
            }
            _ => return Err(bad()),
        },
        "paley" => paley(one()?)?,
        "rook" => match nums('x')?.as_slice() {
            [m, n] => rook(int(*m)?, int(*n)?)?,
            _ => return Err(bad()),
        },
        "circulant" => {
            let v = nums(',')?;
            let (n, jumps) = v.split_first().ok_or_else(bad)?;
            let jumps = jumps.iter().map(|&j| int(j)).collect::<Result<Vec<_>>>()?;
            circulant(int(*n)?, &jumps)?
        }
        "complete" | "k" => {
            let n = one()?;
            size(n as u128, pairs(n))?;
            complete(n)
        }
        "cycle" | "c" => {
            let n = one()?;
            if n < 3 {
                return Err(bad());
            }
            size(n as u128, n as u128)?;
            cycle(n)
        }
        "path" | "p" => {
            let n = one()?;
            size(n as u128, n as u128)?;
            path(n)
        }
        "star" => {
            let n = one()?;
            size(n as u128, n as u128)?;
            star(n)
        }
        "hypercube" | "q" => {
            let d = one()?;
            if d > 20 {
                return Err(bad());
            }
            hypercube(d)
        }
        "grid" | "torus" => {
            let dims = nums('x')?;
            let [r, c] = dims.as_slice() else {
                return Err(bad());
            };
            let (r, c) = (int(*r)?, int(*c)?);
            let periodic = kind == "torus";
            if periodic && (r < 3 || c < 3) {
                return Err(bad());
            }
            let nodes = r as u128 * c as u128;
            size(nodes, 2 * nodes)?;
            grid_2d(r, c, periodic)
        }
        "bipartite" => match nums(',')?.as_slice() {
            [a, b] => {
                let (a, b) = (int(*a)?, int(*b)?);
                size(a as u128 + b as u128, a as u128 * b as u128)?;
                complete_bipartite(a, b)
            }
            _ => return Err(bad()),
        },
        "platonic" => platonic(args.parse()?),
        "random-regular" => {
            let v = nums(',')?;
            let (n, d, seed) = match v.as_slice() {
                [n, d] => (int(*n)?, int(*d)?, 0),
                [n, d, s] => (int(*n)?, int(*d)?, int(*s)? as u64),
                _ => return Err(bad()),
            };
            size(n as u128, n as u128 * d as u128 / 2)?;
            random_regular(n, d, &mut Rng::seed_from_u64(seed))?
        }
        "erdos-renyi" | "gnp" => {
            let v = nums(',')?;
            let (n, p, seed) = match v.as_slice() {
                [n, p] => (int(*n)?, *p, 0),
                [n, p, s] => (int(*n)?, *p, int(*s)? as u64),
                _ => return Err(bad()),
            };
            size(n as u128, pairs(n))?;
            erdos_renyi(n, p, &mut Rng::seed_from_u64(seed))?
        }
        _ => return Err(bad()),
    };
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_specifications() {
        assert_eq!(from_spec("metatron").unwrap().num_edges(), 78);
        assert_eq!(from_spec("fruit-of-life").unwrap().num_edges(), 18);
        assert_eq!(from_spec("metatron:1.0").unwrap().num_edges(), 18);
        assert_eq!(from_spec("cycle:7").unwrap().num_edges(), 7);
        assert_eq!(from_spec("grid:3x4").unwrap().num_nodes(), 12);
        assert_eq!(from_spec("torus:3x3").unwrap().num_edges(), 18);
        assert_eq!(from_spec("platonic:Icosahedron").unwrap().num_edges(), 30);
        assert_eq!(from_spec("bipartite:2,3").unwrap().num_edges(), 6);
        assert_eq!(
            from_spec("random-regular:10,3,5").unwrap().regular_degree(),
            Some(3)
        );
        assert!(from_spec("erdos-renyi:10,0.5").is_ok());
        for bad in [
            "cycle:2",
            "grid:3",
            "complete:x",
            "nonsense",
            "torus:2x5",
            "platonic:sphere",
            "complete:2.5",
            // Found by the `graph_spec` fuzz target: 2·10¹⁰ edges.
            "K:204204",
            "complete:2898",
            "bipartite:5000,5000",
            "grid:100000x100000",
            "path:1048577",
            "erdos-renyi:3000,0.0",
            "random-regular:1000000,10",
        ] {
            assert!(from_spec(bad).is_err(), "{bad}");
        }
        // The limits leave room for large graphs.
        assert_eq!(from_spec("hypercube:20").unwrap().num_nodes(), 1 << 20);
        assert_eq!(from_spec("complete:2000").unwrap().num_edges(), 1_999_000);
    }

    #[test]
    fn distance_regular_and_strongly_regular_families() {
        // Johnson J(n,k): C(n,k) nodes, degree k(n−k); J(n,2) = T(n) is SRG.
        let j = johnson(6, 3).unwrap();
        assert_eq!((j.num_nodes(), j.regular_degree()), (20, Some(9)));
        assert_eq!(
            johnson(5, 2).unwrap().strongly_regular_parameters(),
            Some((10, 6, 3, 4))
        );
        // Kneser K(5,2) is the Petersen graph.
        let k = kneser(5, 2).unwrap();
        assert!(k.is_isomorphic(&petersen()));
        assert_eq!(kneser(7, 3).unwrap().regular_degree(), Some(4));
        // Hamming H(d,2) is the hypercube; H(2,q) the rook's graph.
        assert!(hamming(4, 2).unwrap().is_isomorphic(&hypercube(4)));
        assert!(hamming(2, 4).unwrap().is_isomorphic(&rook(4, 4).unwrap()));
        assert_eq!(
            hamming(3, 3).unwrap().automorphism_group().order(),
            Some(1296)
        );
        // Paley(q): (q, (q−1)/2, (q−5)/4, (q−1)/4); Paley(5) = C5.
        for q in [5, 13, 17, 29] {
            assert_eq!(
                paley(q).unwrap().strongly_regular_parameters(),
                Some((q, (q - 1) / 2, (q - 5) / 4, (q - 1) / 4)),
                "Paley({q})"
            );
        }
        assert!(paley(5).unwrap().is_isomorphic(&cycle(5)));
        // Shrikhande vs rook 4×4: same parameters, different groups.
        assert_eq!(
            shrikhande().strongly_regular_parameters(),
            Some((16, 6, 2, 2))
        );
        assert_eq!(
            rook(4, 4).unwrap().strongly_regular_parameters(),
            Some((16, 6, 2, 2))
        );
        assert_eq!(shrikhande().automorphism_group().order(), Some(192));
        assert_eq!(rook(4, 4).unwrap().automorphism_group().order(), Some(1152));
        // T(8) and the three Chang graphs: (28, 12, 6, 4), pairwise not
        // isomorphic; their groups have orders 8! and 384, 96 and 360
        // (A. E. Brouwer's page on the Chang graphs).
        let t8 = johnson(8, 2).unwrap();
        let changs: Vec<Graph> = (1..=3).map(|k| chang(k).unwrap()).collect();
        let mut orders = vec![t8.automorphism_group().order()];
        for (k, g) in changs.iter().enumerate() {
            assert_eq!(g.strongly_regular_parameters(), Some((28, 12, 6, 4)));
            assert!(!g.is_isomorphic(&t8), "Chang {}", k + 1);
            for h in &changs[k + 1..] {
                assert!(!g.is_isomorphic(h));
            }
            orders.push(g.automorphism_group().order());
        }
        assert_eq!(orders, [Some(40320), Some(384), Some(96), Some(360)]);
        assert!(chang(0).is_err() && chang(4).is_err());
        assert!(from_spec("chang:2").unwrap().is_isomorphic(&changs[1]));
        // Switching with respect to a set and its complement is the same.
        let set = [0, 3, 5, 9];
        let rest: Vec<usize> = (0..28).filter(|u| !set.contains(u)).collect();
        assert_eq!(
            seidel_switching(&t8, &set).unwrap().edges(),
            seidel_switching(&t8, &rest).unwrap().edges()
        );
        assert!(seidel_switching(&t8, &[28]).is_err());
        assert_eq!(
            petersen().strongly_regular_parameters(),
            Some((10, 3, 0, 1))
        );
        assert_eq!(cycle(6).strongly_regular_parameters(), None);
        assert_eq!(cycle(5).strongly_regular_parameters(), Some((5, 2, 0, 1)));
        // Disjoint triangles (μ = 0) and K3,3 (λ = 0).
        let triangles = Graph::from_edges(6, &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]);
        assert_eq!(
            triangles.unwrap().strongly_regular_parameters(),
            Some((6, 2, 1, 0))
        );
        let k33 = complete_bipartite(3, 3);
        assert_eq!(k33.strongly_regular_parameters(), Some((6, 3, 0, 3)));
        assert_eq!(hypercube(3).strongly_regular_parameters(), None);
        // Circulants and cubelike graphs.
        assert!(circulant(9, &[1]).unwrap().is_isomorphic(&cycle(9)));
        assert!(
            circulant(8, &[1, 2, 3, 4])
                .unwrap()
                .is_isomorphic(&complete(8))
        );
        assert!(
            cayley_z2(3, &[1, 2, 4])
                .unwrap()
                .is_isomorphic(&hypercube(3))
        );
        assert!(circulant(8, &[5]).is_err());
        assert!(circulant(8, &[1, 1]).is_err());
        assert!(paley(7).is_err() && paley(9).is_err());
        assert!(johnson(5, 0).is_err() && kneser(5, 3).is_err() && hamming(0, 2).is_err());
        // Specs.
        assert_eq!(from_spec("johnson:6,3").unwrap().num_nodes(), 20);
        assert_eq!(from_spec("rook:3x4").unwrap().num_edges(), 3 * 4 * 5 / 2);
        assert_eq!(from_spec("circulant:10,1,3").unwrap().num_edges(), 20);
        assert!(from_spec("hamming:30,2").is_err());
        assert!(from_spec("johnson:62,31").is_err());
    }

    #[test]
    fn largest_arguments_are_errors() {
        assert!(hamming(usize::MAX, 2).is_err());
        assert!(circulant(4, &[usize::MAX]).is_err());
        assert!(paley(usize::MAX).is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn huge_arguments_are_errors() {
        assert!(hamming(1 << 40, 3).is_err());
        assert!(hamming(usize::MAX, 2).is_err());
        assert!(circulant(4, &[1 << 63]).is_err());
        assert!(circulant(10, &[1 << 63]).is_err());
        let start = std::time::Instant::now();
        assert!(paley(18_446_744_073_709_551_557).is_err());
        assert!(paley(1_000_000_000_000_000_009).is_err());
        assert!(start.elapsed().as_millis() < 100);
    }

    #[test]
    fn subset_ranking_is_colex() {
        for (n, k) in [(6, 3), (10, 4), (12, 1)] {
            for (rank, &set) in subsets(n, k).iter().enumerate() {
                assert_eq!(colex_rank(set), rank);
                assert_eq!(set.count_ones() as usize, k);
            }
            assert_eq!(subsets(n, k).len() as u128, binomial(n as u64, k as u64));
        }
    }

    #[test]
    fn hopeless_random_regular_requests_fail_quickly() {
        // Pairing succeeds with probability ≈ e^{−(d²−1)/4}: never for d = 60.
        let start = std::time::Instant::now();
        assert!(random_regular(4000, 60, &mut Rng::seed_from_u64(1)).is_err());
        assert!(start.elapsed().as_secs() < 30);
    }

    #[test]
    fn platonic_solids_have_correct_shape() {
        let expected = [(4, 6, 3), (8, 12, 3), (6, 12, 4), (20, 30, 3), (12, 30, 5)];
        for (solid, (v, e, d)) in PlatonicSolid::ALL.into_iter().zip(expected) {
            let g = platonic(solid);
            assert_eq!(g.num_nodes(), v, "{solid:?}");
            assert_eq!(g.num_edges(), e, "{solid:?}");
            assert_eq!(g.regular_degree(), Some(d), "{solid:?}");
            assert!(g.is_connected());
        }
        // The cube graph is the 3-dimensional hypercube.
        assert_eq!(platonic(PlatonicSolid::Cube).triangle_count(), 0);
        assert_eq!(hypercube(3).num_edges(), 12);
    }

    #[test]
    fn random_graphs_are_reproducible_and_valid() {
        let mut a = Rng::seed_from_u64(11);
        let mut b = Rng::seed_from_u64(11);
        let g1 = random_regular(20, 3, &mut a).unwrap();
        let g2 = random_regular(20, 3, &mut b).unwrap();
        assert_eq!(g1, g2);
        assert_eq!(g1.regular_degree(), Some(3));
        assert!(random_regular(5, 3, &mut a).is_err());
        let er = erdos_renyi(30, 0.2, &mut a).unwrap();
        assert!(er.num_edges() > 40 && er.num_edges() < 130);
    }

    #[test]
    fn grids_and_tori() {
        let g = grid_2d(3, 4, false);
        assert_eq!(g.num_edges(), 3 * 3 + 2 * 4);
        let t = grid_2d(3, 4, true);
        assert_eq!(t.num_edges(), 2 * 12);
        assert_eq!(t.regular_degree(), Some(4));
    }
}
