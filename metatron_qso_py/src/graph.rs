//! `Graph` and the graph generators.

use metatron_qso::Rng;
use metatron_qso::graph::generators::{self, PlatonicSolid};
use metatron_qso::graph::{Graph, io, metatron};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{Raise, invalid, pydict, real_array, rmatrix};

/// An edge given as a `(u, v)` or `(u, v, weight)` sequence.
struct EdgeSpec {
    u: usize,
    v: usize,
    weight: Option<f64>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for EdgeSpec {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let items = obj.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        match items.as_slice() {
            [u, v] => Ok(Self {
                u: u.extract()?,
                v: v.extract()?,
                weight: None,
            }),
            [u, v, w] => Ok(Self {
                u: u.extract()?,
                v: v.extract()?,
                weight: Some(w.extract()?),
            }),
            _ => Err(PyTypeError::new_err(
                "edges must be (u, v) or (u, v, weight) sequences",
            )),
        }
    }
}

/// Undirected weighted graph (no self loops or multi-edges) with optional
/// 3-D node positions.
#[pyclass(name = "Graph", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyGraph {
    pub inner: Graph,
}

fn check_nodes(n: usize) -> PyResult<()> {
    if n == 0 {
        return Err(invalid("a graph needs at least one node"));
    }
    generators::check_generated_size(n as u128, 0).raise()
}

/// Number of node pairs `n(n−1)/2`.
fn pairs(n: usize) -> u128 {
    n as u128 * (n as u128).saturating_sub(1) / 2
}

fn wrap(inner: Graph) -> PyGraph {
    PyGraph { inner }
}

#[pymethods]
impl PyGraph {
    /// Empty graph on `n` nodes.
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        check_nodes(n)?;
        Ok(wrap(Graph::new(n)))
    }

    /// Graph from `(u, v)` or `(u, v, w)` edges; `weights` (one per edge)
    /// may be given for unweighted edges.
    #[staticmethod]
    #[pyo3(signature = (n, edges, weights = None))]
    fn from_edges(n: usize, edges: Vec<EdgeSpec>, weights: Option<Vec<f64>>) -> PyResult<Self> {
        check_nodes(n)?;
        if let Some(w) = &weights
            && w.len() != edges.len()
        {
            return Err(invalid(format!(
                "got {} weights for {} edges",
                w.len(),
                edges.len()
            )));
        }
        let mut list = Vec::with_capacity(edges.len());
        for (k, e) in edges.into_iter().enumerate() {
            let w = match (e.weight, &weights) {
                (Some(_), Some(_)) => {
                    return Err(invalid(
                        "give weights either inside the edges or as `weights`",
                    ));
                }
                (Some(w), None) => w,
                (None, Some(ws)) => ws[k],
                (None, None) => 1.0,
            };
            list.push((e.u, e.v, w));
        }
        generators::check_generated_size(n as u128, list.len() as u128).raise()?;
        Graph::from_weighted_edges(n, &list).raise().map(wrap)
    }

    /// Graph from a NetworkX graph (or anything with `nodes()` and
    /// `edges(data=True)`); nodes are numbered in the order of `nodes()`,
    /// weights are read from the edge attribute `weight` (default 1).
    /// NetworkX itself is not needed.
    #[staticmethod]
    #[pyo3(signature = (graph, weight = "weight"))]
    fn from_networkx(graph: &Bound<'_, PyAny>, weight: &str) -> PyResult<Self> {
        let index = PyDict::new(graph.py());
        let mut n = 0usize;
        for node in graph.call_method0("nodes")?.try_iter()? {
            let node = node?;
            if index.contains(&node)? {
                return Err(invalid(format!("node {node} appears twice")));
            }
            index.set_item(node, n)?;
            n += 1;
        }
        check_nodes(n)?;
        let kwargs = PyDict::new(graph.py());
        kwargs.set_item("data", true)?;
        let mut list = Vec::new();
        for item in graph.call_method("edges", (), Some(&kwargs))?.try_iter()? {
            let (u, v, data): (Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>) =
                item?.extract()?;
            let lookup = |x: &Bound<'_, PyAny>| -> PyResult<usize> {
                index
                    .get_item(x)?
                    .ok_or_else(|| invalid(format!("edge endpoint {x} is not a node")))?
                    .extract()
            };
            let w: f64 = match data.call_method1("get", (weight,))? {
                x if x.is_none() => 1.0,
                x => x.extract()?,
            };
            list.push((lookup(&u)?, lookup(&v)?, w));
        }
        generators::check_generated_size(n as u128, list.len() as u128).raise()?;
        let name: String = graph
            .getattr("name")
            .and_then(|x| x.extract())
            .unwrap_or_default();
        let g = Graph::from_weighted_edges(n, &list).raise()?;
        Ok(wrap(if name.is_empty() {
            g
        } else {
            g.with_name(name)
        }))
    }

    /// Graph from an edge list: one `u v [weight]` per line, nodes from 0,
    /// `#` or `%` comments, `# nodes N` for the number of nodes.
    #[staticmethod]
    fn from_edge_list(text: &str) -> PyResult<Self> {
        let g = io::from_edge_list(text).raise()?;
        check_nodes(g.num_nodes())?;
        Ok(wrap(g))
    }

    /// The graph as an edge list (see `from_edge_list`).
    fn to_edge_list(&self) -> String {
        io::to_edge_list(&self.inner)
    }

    /// Graph from a GraphML document (nodes, edges and a `weight` key;
    /// node ids are numbered in order of appearance).
    #[staticmethod]
    fn from_graphml(text: &str) -> PyResult<Self> {
        let g = io::from_graphml(text).raise()?;
        check_nodes(g.num_nodes())?;
        Ok(wrap(g))
    }

    /// The graph as a GraphML document.
    fn to_graphml(&self) -> String {
        io::to_graphml(&self.inner)
    }

    /// Add the edge `{u, v}` with weight `weight` (finite, non-zero).
    #[pyo3(signature = (u, v, weight = 1.0))]
    fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> PyResult<()> {
        self.inner.add_weighted_edge(u, v, weight).raise()?;
        Ok(())
    }

    /// Graph from a specification string such as `"metatron"`,
    /// `"fruit-of-life"`, `"cycle:8"`, `"grid:3x4"`, `"platonic:cube"` or
    /// `"random-regular:12,3,7"` (same syntax as the `qso` CLI).
    #[staticmethod]
    fn from_spec(spec: &str) -> PyResult<Self> {
        let g = generators::from_spec(spec).raise()?;
        check_nodes(g.num_nodes())?;
        Ok(wrap(g))
    }

    /// Metatron's Cube: the complete graph K13 on the 13 Fruit-of-Life
    /// centres.
    #[staticmethod]
    fn metatron_cube() -> Self {
        wrap(metatron::metatron_cube())
    }

    /// Fruit-of-Life contact graph (18 edges, symmetry group D6).
    #[staticmethod]
    fn fruit_of_life() -> Self {
        wrap(metatron::fruit_of_life())
    }

    /// Lines of Metatron's Cube of length ≤ `max_length` (circle
    /// diameters).
    #[staticmethod]
    fn metatron_distance_graph(max_length: f64) -> PyResult<Self> {
        if max_length.is_nan() {
            return Err(invalid("`max_length` must not be NaN"));
        }
        Ok(wrap(metatron::distance_graph(max_length)))
    }

    /// Complete graph K_n.
    #[staticmethod]
    fn complete(n: usize) -> PyResult<Self> {
        check_nodes(n)?;
        generators::check_generated_size(n as u128, pairs(n)).raise()?;
        Ok(wrap(generators::complete(n)))
    }

    /// Cycle C_n (n ≥ 3).
    #[staticmethod]
    fn cycle(n: usize) -> PyResult<Self> {
        if n < 3 {
            return Err(invalid("a cycle needs at least 3 nodes"));
        }
        check_nodes(n)?;
        Ok(wrap(generators::cycle(n)))
    }

    /// Path P_n.
    #[staticmethod]
    fn path(n: usize) -> PyResult<Self> {
        check_nodes(n)?;
        Ok(wrap(generators::path(n)))
    }

    /// Star: node 0 joined to nodes 1..n.
    #[staticmethod]
    fn star(n: usize) -> PyResult<Self> {
        check_nodes(n)?;
        Ok(wrap(generators::star(n)))
    }

    /// Complete bipartite graph K_{a,b}.
    #[staticmethod]
    fn complete_bipartite(a: usize, b: usize) -> PyResult<Self> {
        if a + b == 0 {
            return Err(invalid("a graph needs at least one node"));
        }
        generators::check_generated_size(a as u128 + b as u128, a as u128 * b as u128).raise()?;
        Ok(wrap(generators::complete_bipartite(a, b)))
    }

    /// `d`-dimensional hypercube Q_d (2^d nodes, d ≤ 20).
    #[staticmethod]
    fn hypercube(d: usize) -> PyResult<Self> {
        if d > 20 {
            return Err(invalid("hypercube dimension must be at most 20"));
        }
        Ok(wrap(generators::hypercube(d)))
    }

    /// The Petersen graph.
    #[staticmethod]
    fn petersen() -> Self {
        wrap(generators::petersen())
    }

    /// The Heawood graph, the smallest cubic graph of girth 6.
    #[staticmethod]
    fn heawood() -> Self {
        wrap(generators::heawood())
    }

    /// `rows × cols` grid; a torus if `periodic` (each side ≥ 3).
    #[staticmethod]
    #[pyo3(signature = (rows, cols, periodic = false))]
    fn grid(rows: usize, cols: usize, periodic: bool) -> PyResult<Self> {
        let nodes = rows as u128 * cols as u128;
        if nodes == 0 {
            return Err(invalid("a graph needs at least one node"));
        }
        generators::check_generated_size(nodes, 2 * nodes).raise()?;
        if periodic && (rows < 3 || cols < 3) {
            return Err(invalid("a torus needs at least 3 nodes per dimension"));
        }
        Ok(wrap(generators::grid_2d(rows, cols, periodic)))
    }

    /// Edge graph of a Platonic solid: tetrahedron, cube, octahedron,
    /// dodecahedron or icosahedron.
    #[staticmethod]
    fn platonic(name: &str) -> PyResult<Self> {
        let solid = PlatonicSolid::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(name.trim()))
            .ok_or_else(|| {
                invalid(format!(
                    "unknown Platonic solid '{name}' (expected tetrahedron, cube, octahedron, \
                     dodecahedron or icosahedron)"
                ))
            })?;
        Ok(wrap(generators::platonic(solid)))
    }

    /// Random `d`-regular simple graph (`n·d` even, `d < n`).
    #[staticmethod]
    #[pyo3(signature = (n, d, seed = 0))]
    fn random_regular(py: Python<'_>, n: usize, d: usize, seed: u64) -> PyResult<Self> {
        check_nodes(n)?;
        generators::check_generated_size(n as u128, n as u128 * d as u128 / 2).raise()?;
        py.detach(|| generators::random_regular(n, d, &mut Rng::seed_from_u64(seed)))
            .raise()
            .map(wrap)
    }

    /// Erdős–Rényi graph G(n, p).
    #[staticmethod]
    #[pyo3(signature = (n, p, seed = 0))]
    fn erdos_renyi(py: Python<'_>, n: usize, p: f64, seed: u64) -> PyResult<Self> {
        check_nodes(n)?;
        generators::check_generated_size(n as u128, pairs(n)).raise()?;
        py.detach(|| generators::erdos_renyi(n, p, &mut Rng::seed_from_u64(seed)))
            .raise()
            .map(wrap)
    }

    /// Johnson graph J(n, k): k-subsets of n elements, adjacent when they
    /// share k − 1 elements.
    #[staticmethod]
    fn johnson(n: usize, k: usize) -> PyResult<Self> {
        generators::johnson(n, k).raise().map(wrap)
    }

    /// Kneser graph K(n, k): k-subsets, adjacent when disjoint
    /// (K(5, 2) is the Petersen graph).
    #[staticmethod]
    fn kneser(n: usize, k: usize) -> PyResult<Self> {
        generators::kneser(n, k).raise().map(wrap)
    }

    /// Hamming graph H(d, q): words of length d over q letters, adjacent
    /// when they differ in one letter (H(d, 2) is the hypercube).
    #[staticmethod]
    fn hamming(d: usize, q: usize) -> PyResult<Self> {
        generators::hamming(d, q).raise().map(wrap)
    }

    /// Rook's graph K_m □ K_n (m × n board, adjacent in a row or column).
    #[staticmethod]
    fn rook(m: usize, n: usize) -> PyResult<Self> {
        generators::rook(m, n).raise().map(wrap)
    }

    /// Paley graph of prime order q ≡ 1 (mod 4).
    #[staticmethod]
    fn paley(q: usize) -> PyResult<Self> {
        generators::paley(q).raise().map(wrap)
    }

    /// Circulant graph: node i joined to i ± s (mod n) for every jump s.
    #[staticmethod]
    fn circulant(n: usize, jumps: Vec<usize>) -> PyResult<Self> {
        generators::circulant(n, &jumps).raise().map(wrap)
    }

    /// Cayley graph of Z_2^d: node x joined to x XOR g for every generator g.
    #[staticmethod]
    fn cayley_z2(d: usize, generators: Vec<u64>) -> PyResult<Self> {
        generators::cayley_z2(d, &generators).raise().map(wrap)
    }

    /// The Shrikhande graph: strongly regular (16, 6, 2, 2) like the 4 × 4
    /// rook's graph, but not isomorphic to it.
    #[staticmethod]
    fn shrikhande() -> Self {
        wrap(generators::shrikhande())
    }

    /// Chang graph `which` (1, 2 or 3): strongly regular (28, 12, 6, 4)
    /// like `Graph.johnson(8, 2)`, but not isomorphic to it.
    #[staticmethod]
    fn chang(which: usize) -> PyResult<Self> {
        generators::chang(which).raise().map(wrap)
    }

    /// Seidel switching with respect to the node set `nodes`: pairs with
    /// one node inside and one outside the set swap adjacency (unweighted
    /// graphs).
    fn seidel_switching(&self, nodes: Vec<usize>) -> PyResult<Self> {
        generators::seidel_switching(&self.inner, &nodes).raise().map(wrap)
    }

    /// Cartesian product: node (g, h) has index g·|H| + h.
    fn cartesian_product(&self, other: &PyGraph) -> PyResult<Self> {
        self.inner.cartesian_product(&other.inner).raise().map(wrap)
    }

    /// `(n, k, λ, μ)` if the graph is strongly regular, else `None`.
    fn strongly_regular_parameters(&self, py: Python<'_>) -> Option<(usize, usize, usize, usize)> {
        let g = &self.inner;
        py.detach(|| g.strongly_regular_parameters())
    }

    /// The graph with node `u` renamed `perm[u]`.
    fn relabeled(&self, perm: Vec<usize>) -> PyResult<Self> {
        self.inner.relabeled(&perm).raise().map(wrap)
    }

    /// Canonical labelling: `labels[u]` is the canonical index of node `u`;
    /// isomorphic graphs get identical canonical forms.
    fn canonical_labelling(&self, py: Python<'_>) -> Vec<usize> {
        let g = &self.inner;
        py.detach(|| g.canonical_labelling())
    }

    /// The graph relabelled canonically.
    fn canonical_form(&self, py: Python<'_>) -> Self {
        let g = &self.inner;
        wrap(py.detach(|| g.canonical_form()))
    }

    /// Whether the graphs are isomorphic (edge weights included).
    fn is_isomorphic(&self, py: Python<'_>, other: &PyGraph) -> bool {
        let (g, h) = (&self.inner, &other.inner);
        py.detach(|| g.is_isomorphic(h))
    }

    /// An isomorphism to `other` (node `u` maps to `iso[u]`), or `None`.
    fn isomorphism(&self, py: Python<'_>, other: &PyGraph) -> Option<Vec<usize>> {
        let (g, h) = (&self.inner, &other.inner);
        py.detach(|| g.isomorphism(h))
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Edges `(u, v, weight)` with `u < v`, in insertion order.
    #[getter]
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges().to_vec()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        let g = std::mem::replace(&mut self.inner, Graph::new(0));
        self.inner = g.with_name(name);
    }

    /// Node coordinates as an `(n, 3)` array, or `None`.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .positions()
            .map(|pos| Array2::from_shape_fn((pos.len(), 3), |(r, c)| pos[r][c]).into_pyarray(py))
    }

    /// Node degrees.
    fn degrees(&self) -> Vec<usize> {
        self.inner.degrees()
    }

    /// Sorted neighbours of `u` as `(node, weight)` pairs.
    fn neighbors(&self, u: usize) -> PyResult<Vec<(usize, f64)>> {
        if u >= self.inner.num_nodes() {
            return Err(invalid(format!(
                "node {u} out of range for a graph with {} nodes",
                self.inner.num_nodes()
            )));
        }
        Ok(self.inner.neighbors(u).to_vec())
    }

    /// Dense weighted adjacency matrix.
    fn adjacency_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rmatrix(py, &self.inner.adjacency_matrix())
    }

    /// Dense Laplacian `L = D − A`.
    fn laplacian_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rmatrix(py, &self.inner.laplacian_matrix())
    }

    /// Eigenvalues of the adjacency matrix (ascending).
    fn adjacency_spectrum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let g = &self.inner;
        real_array(py, py.detach(|| g.adjacency_spectrum()))
    }

    /// Eigenvalues of the Laplacian (ascending).
    fn laplacian_spectrum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let g = &self.inner;
        real_array(py, py.detach(|| g.laplacian_spectrum()))
    }

    /// Structural summary as a dict.
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = &self.inner;
        let s = py.detach(|| g.summary());
        Ok(pydict!(py, {
            "name" => s.name,
            "nodes" => s.nodes,
            "edges" => s.edges,
            "min_degree" => s.min_degree,
            "max_degree" => s.max_degree,
            "regular" => s.regular,
            "connected" => s.connected,
            "bipartite" => s.bipartite,
            "diameter" => s.diameter,
            "triangles" => s.triangles,
            "algebraic_connectivity" => s.algebraic_connectivity,
        }))
    }

    /// Order of the automorphism group (`None` if it exceeds 2^128).
    fn automorphism_group_order(&self, py: Python<'_>) -> Option<u128> {
        let g = &self.inner;
        py.detach(|| g.automorphism_group().order())
    }

    /// log10 of the automorphism group order (always finite).
    fn automorphism_log10_order(&self, py: Python<'_>) -> f64 {
        let g = &self.inner;
        py.detach(|| g.automorphism_group().log10_order())
    }

    /// Orbits of the automorphism group (each sorted).
    fn automorphism_orbits(&self, py: Python<'_>) -> Vec<Vec<usize>> {
        let g = &self.inner;
        py.detach(|| g.automorphism_group().orbits())
    }

    /// Strong generating set of the automorphism group; permutation `p`
    /// maps node `u` to `p[u]`.
    fn automorphism_generators(&self, py: Python<'_>) -> Vec<Vec<usize>> {
        let g = &self.inner;
        py.detach(|| g.automorphism_group().generators().to_vec())
    }

    fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Largest hop distance (`None` if disconnected).
    fn diameter(&self) -> Option<usize> {
        self.inner.diameter()
    }

    /// Length of the shortest cycle (`None` for a forest).
    fn girth(&self) -> Option<usize> {
        self.inner.girth()
    }

    /// Number of triangles.
    fn triangle_count(&self) -> usize {
        self.inner.triangle_count()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph(name={:?}, nodes={}, edges={})",
            self.inner.name(),
            self.inner.num_nodes(),
            self.inner.num_edges()
        )
    }
}

/// Distinct line lengths of Metatron's Cube with multiplicities (ascending;
/// the counts sum to 78).
#[pyfunction]
fn metatron_line_lengths() -> Vec<(f64, usize)> {
    metatron::line_lengths()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(metatron_line_lengths, m)?)?;
    Ok(())
}
