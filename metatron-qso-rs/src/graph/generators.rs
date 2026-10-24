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
