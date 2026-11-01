//! Symmetry-reduced walks (quotients by equitable partitions) and state
//! transfer.

use metatron_qso::linalg::RMatrix;
use metatron_qso::walks::{
    PrettyGoodTransfer, Quotient, QuotientSearch, QuotientWalk, StateTransfer,
};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::algorithms::Indices;
use crate::graph::PyGraph;
use crate::walks::{check_time, generator, generator_name};
use crate::{
    ComplexArg, Raise, RealArg, RealMatrixArg, complex_array, complex_vec, invalid, pydict,
    real_array, real_vec, rmatrix,
};

/// Equitable partition of a graph's nodes and its quotient matrix: every
/// node of cell `i` has total edge weight `matrix[i][j]` into cell `j`.
/// A walker started in a cell (a single node, or the uniform superposition
/// over the cell) stays in the span of the cell states, so its dynamics
/// are exact in `num_cells` dimensions however large the graph is.
///
/// Build one from a graph (`with_singletons`, `equitable`, `from_cells`),
/// from a matrix (`from_matrix`), or directly for the distance partitions
/// of hypercubes, Hamming, Johnson, complete and complete multipartite
/// graphs, which are never built (`hypercube(30)` has 2^30 nodes and 31
/// cells).
#[pyclass(name = "Quotient", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyQuotient {
    pub inner: Quotient,
}

fn wrap(inner: Quotient) -> PyQuotient {
    PyQuotient { inner }
}

#[pymethods]
impl PyQuotient {
    /// Coarsest equitable partition refining the node colouring `colours`.
    #[staticmethod]
    fn equitable(py: Python<'_>, graph: &PyGraph, colours: Vec<usize>) -> PyResult<Self> {
        let g = &graph.inner;
        py.detach(|| Quotient::equitable(g, &colours))
            .raise()
            .map(wrap)
    }

    /// Coarsest equitable partition in which each of `nodes` is a cell of
    /// its own (the reduction for a walker started at, or a search for,
    /// those nodes).
    #[staticmethod]
    fn with_singletons(py: Python<'_>, graph: &PyGraph, nodes: Indices) -> PyResult<Self> {
        let nodes = nodes.into_vec();
        let g = &graph.inner;
        py.detach(|| Quotient::with_singletons(g, &nodes))
            .raise()
            .map(wrap)
    }

    /// The partition `cell_of` (node → cell label), checked to be equitable.
    #[staticmethod]
    fn from_cells(py: Python<'_>, graph: &PyGraph, cell_of: Vec<usize>) -> PyResult<Self> {
        let g = &graph.inner;
        py.detach(|| Quotient::from_cells(g, &cell_of))
            .raise()
            .map(wrap)
    }

    /// Quotient from cell sizes and the matrix of weights between cells
    /// (`sizes[i]·matrix[i][j] = sizes[j]·matrix[j][i]` is checked).
    #[staticmethod]
    fn from_matrix(sizes: RealArg<'_>, matrix: RealMatrixArg<'_>) -> PyResult<Self> {
        let view = matrix.as_array();
        let (rows, cols) = view.dim();
        let m = RMatrix::from_fn(rows, cols, |r, c| view[(r, c)]);
        Quotient::from_matrix(real_vec(&sizes), m).raise().map(wrap)
    }

    /// K_n seen from one node: the node and the n − 1 others.
    #[staticmethod]
    fn complete(n: usize) -> PyResult<Self> {
        Quotient::complete(n).raise().map(wrap)
    }

    /// Complete multipartite graph seen from a node of the first part.
    #[staticmethod]
    fn complete_multipartite(parts: Vec<usize>) -> PyResult<Self> {
        Quotient::complete_multipartite(&parts).raise().map(wrap)
    }

    /// Hamming graph H(d, q) by distance from a word (d + 1 cells).
    #[staticmethod]
    fn hamming(d: usize, q: usize) -> PyResult<Self> {
        Quotient::hamming(d, q).raise().map(wrap)
    }

    /// Hypercube Q_d by Hamming weight (d + 1 cells, any d up to about 1000).
    #[staticmethod]
    fn hypercube(d: usize) -> PyResult<Self> {
        Quotient::hypercube(d).raise().map(wrap)
    }

    /// Johnson graph J(n, k) by distance from a k-subset.
    #[staticmethod]
    fn johnson(n: usize, k: usize) -> PyResult<Self> {
        Quotient::johnson(n, k).raise().map(wrap)
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn num_cells(&self) -> usize {
        self.inner.num_cells()
    }

    /// Number of nodes of the graph (a float: implicit families can exceed
    /// every integer type).
    #[getter]
    fn num_nodes(&self) -> f64 {
        self.inner.num_nodes()
    }

    /// Number of nodes in each cell.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.sizes().to_vec())
    }

    /// Weight from any node of cell `i` into cell `j`.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rmatrix(py, self.inner.matrix())
    }

    /// Cell of every node (partitions of an explicit graph only).
    #[getter]
    fn cell_of(&self) -> Option<Vec<usize>> {
        self.inner.cell_of().map(<[usize]>::to_vec)
    }

    /// Weighted degree of the nodes of each cell.
    fn degrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.degrees())
    }

    /// Adjacency matrix on the normalised cell states (symmetric).
    fn adjacency<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rmatrix(py, &self.inner.adjacency())
    }

    /// Laplacian on the normalised cell states (symmetric).
    fn laplacian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rmatrix(py, &self.inner.laplacian())
    }

    /// The uniform superposition over all nodes in the cell basis.
    fn uniform_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.uniform_state())
    }

    /// Per-node probabilities from per-cell ones (explicit graphs only).
    fn node_probabilities<'py>(
        &self,
        py: Python<'py>,
        per_cell: RealArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(real_array(
            py,
            self.inner
                .node_probabilities(&real_vec(&per_cell))
                .raise()?,
        ))
    }

    /// Node amplitudes from cell amplitudes `c` (`c_i/√|C_i|` on every node
    /// of cell `i`; explicit graphs only).
    fn lift<'py>(
        &self,
        py: Python<'py>,
        c: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(
            py,
            self.inner.lift(&complex_vec(&c)).raise()?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Quotient(name={:?}, cells={}, nodes={})",
            self.inner.name(),
            self.inner.num_cells(),
            self.inner.num_nodes()
        )
    }
}

/// Continuous-time walk on a `Quotient`. Starting "at cell c" means the
/// cell state: the node itself for a single-node cell.
#[pyclass(name = "QuotientWalk", module = "metatron_qso", frozen)]
pub struct PyQuotientWalk {
    inner: QuotientWalk,
}

#[pymethods]
impl PyQuotientWalk {
    #[new]
    #[pyo3(signature = (quotient, generator = "laplacian", gamma = 1.0))]
    fn new(py: Python<'_>, quotient: &PyQuotient, generator: &str, gamma: f64) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let q = &quotient.inner;
        let inner = py.detach(|| QuotientWalk::new(q, gen_, gamma)).raise()?;
        Ok(Self { inner })
    }

    /// Probability of each cell at time `t` for the walk started at `start`.
    fn cell_probabilities<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_time("t", t)?;
        Ok(real_array(
            py,
            self.inner.cell_probabilities(start, t).raise()?,
        ))
    }

    /// `⟨C_to| exp(−iHt) |C_from⟩` (the node amplitude for single-node cells).
    fn amplitude(&self, from_cell: usize, to_cell: usize, t: f64) -> PyResult<metatron_qso::C64> {
        check_time("t", t)?;
        self.inner.amplitude(from_cell, to_cell, t).raise()
    }

    /// `exp(−iHt)` applied to amplitudes in the cell basis.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        c0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        check_time("t", t)?;
        Ok(complex_array(
            py,
            self.inner.evolve(&complex_vec(&c0), t).raise()?,
        ))
    }

    /// Limiting (infinite-time-averaged) probability of each cell.
    fn limiting_distribution<'py>(
        &self,
        py: Python<'py>,
        start: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(real_array(
            py,
            self.inner.limiting_distribution(start).raise()?,
        ))
    }

    /// Exact time average of the cell probabilities over `[0, t_max]`.
    fn average_distribution<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_time("t_max", t_max)?;
        Ok(real_array(
            py,
            self.inner.average_distribution(start, t_max).raise()?,
        ))
    }

    /// Eigenvalues of the reduced Hamiltonian (ascending).
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.spectrum().to_vec())
    }

    #[getter]
    fn quotient(&self) -> PyQuotient {
        wrap(self.inner.quotient().clone())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    fn __repr__(&self) -> String {
        format!(
            "QuotientWalk(cells={}, generator={:?}, gamma={})",
            self.inner.quotient().num_cells(),
            generator_name(self.inner.generator()),
            self.inner.gamma()
        )
    }
}

fn search_dict<'py>(
    py: Python<'py>,
    r: &metatron_qso::walks::SearchResult,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(pydict!(py, {
        "gamma" => r.gamma,
        "time" => r.time,
        "success_probability" => r.success_probability,
        "expected_time" => r.expected_time,
    }))
}

/// Spatial search for marked cells of a `Quotient`, from the uniform
/// superposition (a single marked node must be a cell of its own).
#[pyclass(name = "QuotientSearch", module = "metatron_qso", frozen)]
pub struct PyQuotientSearch {
    inner: QuotientSearch,
}

#[pymethods]
impl PyQuotientSearch {
    /// `gamma=None` selects `QuotientSearch.critical_gamma(quotient, first marked cell)`.
    #[new]
    #[pyo3(signature = (quotient, marked_cells, generator = "laplacian", gamma = None))]
    fn new(
        py: Python<'_>,
        quotient: &PyQuotient,
        marked_cells: Indices,
        generator: &str,
        gamma: Option<f64>,
    ) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let marked = marked_cells.into_vec();
        let q = &quotient.inner;
        let inner = py
            .detach(|| {
                let gamma = match gamma {
                    Some(x) => x,
                    None => QuotientSearch::critical_gamma(
                        q,
                        *marked.first().ok_or_else(|| {
                            metatron_qso::QsoError::invalid(
                                "marked",
                                "at least one marked cell required",
                            )
                        })?,
                    )?,
                };
                QuotientSearch::new(q, &marked, gen_, gamma)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// `⟨C|L⁺|C⟩`: the Childs–Goldstone rate for one marked node of a
    /// vertex-transitive graph, from the reduced Laplacian.
    #[staticmethod]
    fn critical_gamma(quotient: &PyQuotient, cell: usize) -> PyResult<f64> {
        QuotientSearch::critical_gamma(&quotient.inner, cell).raise()
    }

    /// Optimise the hopping rate and the measurement time together.
    /// `t_max=None` uses `π√N`.
    #[staticmethod]
    #[pyo3(signature = (quotient, marked_cells, generator = "laplacian", t_max = None, resolution = 400))]
    fn optimize<'py>(
        py: Python<'py>,
        quotient: &PyQuotient,
        marked_cells: Indices,
        generator: &str,
        t_max: Option<f64>,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let gen_ = self::generator(generator)?;
        let marked = marked_cells.into_vec();
        let q = &quotient.inner;
        let t_max = t_max.unwrap_or(std::f64::consts::PI * q.num_nodes().sqrt());
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(invalid("`t_max` must be positive and finite"));
        }
        let r = py
            .detach(|| QuotientSearch::optimize(q, &marked, gen_, t_max, resolution.max(2)))
            .raise()?;
        search_dict(py, &r)
    }

    /// Probability of finding a node of a marked cell at time `t`.
    fn success_probability(&self, t: f64) -> PyResult<f64> {
        check_time("t", t)?;
        Ok(self.inner.success_probability(t))
    }

    /// Maximise the success probability over `[0, t_max]`.
    #[pyo3(signature = (t_max, resolution = 400))]
    fn optimize_time<'py>(
        &self,
        py: Python<'py>,
        t_max: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(invalid("`t_max` must be positive and finite"));
        }
        let s = &self.inner;
        let r = py.detach(|| s.optimize_time(t_max, resolution));
        search_dict(py, &r)
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn marked(&self) -> Vec<usize> {
        self.inner.marked().to_vec()
    }
}

/// State transfer from node `u` to node `v`: perfect state transfer
/// (strong cospectrality and the eigenvalue condition, decided exactly up
/// to floating-point tolerances), amplitudes and the best fidelity within
/// a time window. Runs on the quotient in which `u` is a single cell.
#[pyclass(name = "StateTransfer", module = "metatron_qso", frozen)]
pub struct PyStateTransfer {
    inner: StateTransfer,
}

#[pymethods]
impl PyStateTransfer {
    #[new]
    #[pyo3(signature = (graph, u, v, generator = "laplacian", gamma = 1.0))]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        u: usize,
        v: usize,
        generator: &str,
        gamma: f64,
    ) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let g = &graph.inner;
        let inner = py
            .detach(|| StateTransfer::new(g, gen_, gamma, u, v))
            .raise()?;
        Ok(Self { inner })
    }

    /// Transfer from the single-node cell `from_cell` to cell `to_cell` of
    /// a quotient (for example `Quotient.hypercube(30)`, cells 0 → 30).
    #[staticmethod]
    #[pyo3(signature = (quotient, from_cell, to_cell, generator = "laplacian", gamma = 1.0))]
    fn from_quotient(
        quotient: &PyQuotient,
        from_cell: usize,
        to_cell: usize,
        generator: &str,
        gamma: f64,
    ) -> PyResult<Self> {
        let inner = StateTransfer::from_quotient(
            &quotient.inner,
            self::generator(generator)?,
            gamma,
            from_cell,
            to_cell,
        )
        .raise()?;
        Ok(Self { inner })
    }

    /// `⟨v| exp(−iHt) |u⟩`.
    fn amplitude(&self, t: f64) -> PyResult<metatron_qso::C64> {
        check_time("t", t)?;
        Ok(self.inner.amplitude(t))
    }

    /// `|⟨v| exp(−iHt) |u⟩|²`.
    fn fidelity(&self, t: f64) -> PyResult<f64> {
        check_time("t", t)?;
        Ok(self.inner.fidelity(t))
    }

    /// Highest fidelity over `[0, t_max]`, as `{"time", "fidelity"}`.
    #[pyo3(signature = (t_max, resolution = 2000))]
    fn best_transfer<'py>(
        &self,
        py: Python<'py>,
        t_max: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.inner;
        let r = py.detach(|| s.best_transfer(t_max, resolution)).raise()?;
        Ok(pydict!(py, {"time" => r.time, "fidelity" => r.fidelity}))
    }

    /// Earliest time of perfect state transfer, or `None`.
    #[getter]
    fn perfect_transfer_time(&self) -> Option<f64> {
        self.inner.perfect_transfer_time()
    }

    #[getter]
    fn strongly_cospectral(&self) -> bool {
        self.inner.strongly_cospectral()
    }

    /// Distinct eigenvalues of `H` in the support of `u`.
    #[getter]
    fn support<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.support().to_vec())
    }

    /// Signs `σ_θ` with `E_θ e_v = σ_θ E_θ e_u` on the support, if the
    /// nodes are strongly cospectral.
    #[getter]
    fn signs(&self) -> Option<Vec<i8>> {
        self.inner.signs().map(<[i8]>::to_vec)
    }

    fn __repr__(&self) -> String {
        let py_bool = |b: bool| if b { "True" } else { "False" };
        let time = self
            .inner
            .perfect_transfer_time()
            .map_or("None".to_string(), |t| t.to_string());
        format!(
            "StateTransfer(strongly_cospectral={}, perfect_transfer_time={time})",
            py_bool(self.inner.strongly_cospectral())
        )
    }
}

/// Pretty good state transfer decided exactly (cyclotomic arithmetic and
/// Kronecker's theorem) for paths, cycles and Cayley graphs of abelian
/// groups.
#[pyclass(name = "PrettyGoodTransfer", module = "metatron_qso", frozen)]
pub struct PyPrettyGoodTransfer {
    inner: PrettyGoodTransfer,
}

#[pymethods]
impl PyPrettyGoodTransfer {
    /// Nodes `u` and `v` of the path on `n` nodes.
    #[staticmethod]
    #[pyo3(signature = (n, u, v, generator = "adjacency", gamma = 1.0))]
    fn path(
        py: Python<'_>,
        n: usize,
        u: usize,
        v: usize,
        generator: &str,
        gamma: f64,
    ) -> PyResult<Self> {
        let g = self::generator(generator)?;
        py.detach(|| PrettyGoodTransfer::path(n, g, gamma, u, v))
            .raise()
            .map(|inner| Self { inner })
    }

    /// Nodes `u` and `v` of the cycle on `n` nodes.
    #[staticmethod]
    #[pyo3(signature = (n, u, v, generator = "adjacency", gamma = 1.0))]
    fn cycle(
        py: Python<'_>,
        n: usize,
        u: usize,
        v: usize,
        generator: &str,
        gamma: f64,
    ) -> PyResult<Self> {
        let g = self::generator(generator)?;
        py.detach(|| PrettyGoodTransfer::cycle(n, g, gamma, u, v))
            .raise()
            .map(|inner| Self { inner })
    }

    /// Elements `u` and `v` of the Cayley graph of `Z_{m_1} × … × Z_{m_k}`
    /// with the symmetric connection set `connection`.
    #[staticmethod]
    #[pyo3(signature = (moduli, connection, u, v, generator = "adjacency", gamma = 1.0))]
    fn cayley(
        py: Python<'_>,
        moduli: Vec<usize>,
        connection: Vec<Vec<i64>>,
        u: Vec<usize>,
        v: Vec<usize>,
        generator: &str,
        gamma: f64,
    ) -> PyResult<Self> {
        let g = self::generator(generator)?;
        py.detach(|| PrettyGoodTransfer::cayley(&moduli, &connection, g, gamma, &u, &v))
            .raise()
            .map(|inner| Self { inner })
    }

    /// A graph that is a path or a cycle (any node order).
    #[staticmethod]
    #[pyo3(signature = (graph, u, v, generator = "adjacency", gamma = 1.0))]
    fn from_graph(
        py: Python<'_>,
        graph: &PyGraph,
        u: usize,
        v: usize,
        generator: &str,
        gamma: f64,
    ) -> PyResult<Self> {
        let g = self::generator(generator)?;
        let gr = &graph.inner;
        py.detach(|| PrettyGoodTransfer::from_graph(gr, g, gamma, u, v))
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn strongly_cospectral(&self) -> bool {
        self.inner.strongly_cospectral()
    }

    /// Whether there is pretty good state transfer (exact).
    #[getter]
    fn pretty_good(&self) -> bool {
        self.inner.pretty_good()
    }

    /// Earliest time of perfect state transfer, or `None`.
    #[getter]
    fn perfect_time(&self) -> Option<f64> {
        self.inner.perfect_time()
    }

    /// Dimension over Q of the span of the eigenvalue differences.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// An integer relation among the eigenvalue differences with odd
    /// parity, which rules out PGST, or `None` (also when its coefficients
    /// exceed 64 bits). Coefficient i belongs to `support[i + 1]`.
    #[getter]
    fn obstruction(&self) -> Option<Vec<i64>> {
        self.inner.obstruction().map(<[i64]>::to_vec)
    }

    /// The eigenvalues of the Hamiltonian in the support of `u`, with the
    /// sign σ of `v` on each (0 where the nodes are not strongly
    /// cospectral).
    #[getter]
    fn support(&self) -> Vec<(f64, i8)> {
        self.inner.support()
    }

    /// `|⟨v| exp(−iHt) |u⟩|²`, from the closed-form eigenvectors (any
    /// pair of nodes).
    fn fidelity(&self, t: f64) -> PyResult<f64> {
        check_time("t", t)?;
        Ok(self.inner.fidelity(t))
    }

    /// Highest fidelity over `[0, t_max]`, as `{"time", "fidelity"}`.
    #[pyo3(signature = (t_max, resolution = 2000))]
    fn best_transfer<'py>(
        &self,
        py: Python<'py>,
        t_max: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.inner;
        let r = py.detach(|| s.best_transfer(t_max, resolution)).raise()?;
        Ok(pydict!(py, {"time" => r.time, "fidelity" => r.fidelity}))
    }

    fn __repr__(&self) -> String {
        let d = self.inner.decision();
        format!(
            "PrettyGoodTransfer(strongly_cospectral={}, pretty_good={}, perfect_time={})",
            if d.strongly_cospectral {
                "True"
            } else {
                "False"
            },
            if d.pretty_good { "True" } else { "False" },
            d.perfect_time
                .map_or_else(|| "None".to_string(), |t| t.to_string())
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQuotient>()?;
    m.add_class::<PyQuotientWalk>()?;
    m.add_class::<PyQuotientSearch>()?;
    m.add_class::<PyStateTransfer>()?;
    m.add_class::<PyPrettyGoodTransfer>()?;
    Ok(())
}
