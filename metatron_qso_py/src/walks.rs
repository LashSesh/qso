//! Continuous-time quantum walks, spatial search, dephasing and classical
//! baselines.

use metatron_qso::walks::driven::{DrivenEvolution, DrivenWalk, MagnusOptions, Term};
use metatron_qso::walks::interacting::{
    CoordinateGraph, CoordinateInteractingWalk, GroupInteractingWalk, HypercubeParticles,
    InteractingWalk, Statistics, SymmetricInteractingWalk, certificate_distance,
};
use metatron_qso::walks::{
    ContinuousWalk, DephasingOptimum, DephasingWalk, Generator, SpatialSearch, TransportOptions,
    TransportResult, TransportState, classical_hitting_times, classical_walk_probabilities,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping};

use crate::algorithms::Indices;
use crate::graph::PyGraph;
use crate::{
    ComplexArg, ComplexMatrixArg, Raise, RealArg, cmatrix, complex_array, complex_vec, invalid,
    pydict, real_array, real_vec, rmatrix, square_complex,
};

pub(crate) fn generator(name: &str) -> PyResult<Generator> {
    match name.to_ascii_lowercase().as_str() {
        "laplacian" => Ok(Generator::Laplacian),
        "adjacency" => Ok(Generator::Adjacency),
        _ => Err(invalid(format!(
            "unknown generator '{name}' (expected laplacian or adjacency)"
        ))),
    }
}

pub(crate) fn generator_name(g: Generator) -> &'static str {
    match g {
        Generator::Laplacian => "laplacian",
        Generator::Adjacency => "adjacency",
    }
}

pub(crate) fn check_time(name: &str, t: f64) -> PyResult<()> {
    if t.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("`{name}` must be finite")))
    }
}

/// Continuous-time quantum walk `|ψ(t)⟩ = exp(−iHt)|ψ(0)⟩` with
/// `H = γL` (`"laplacian"`) or `H = −γA` (`"adjacency"`), plus
/// `Σ_j ε_j|j⟩⟨j|` for `site_energies` (one per node, not scaled by `γ`).
///
/// Graphs with up to 2048 nodes are diagonalised on first use; larger
/// graphs use Krylov propagation, and limiting and averaged distributions
/// through the start node's cyclic subspace (small for symmetric graphs);
/// the dense propagator needs at most 2048 nodes.
#[pyclass(name = "ContinuousWalk", module = "metatron_qso", frozen)]
pub struct PyContinuousWalk {
    inner: ContinuousWalk,
}

#[pymethods]
impl PyContinuousWalk {
    #[new]
    #[pyo3(signature = (graph, generator = "laplacian", gamma = 1.0, site_energies = None))]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        generator: &str,
        gamma: f64,
        site_energies: Option<RealArg<'_>>,
    ) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let g = &graph.inner;
        let energies = site_energies.as_ref().map(real_vec);
        let inner = py
            .detach(|| {
                let walk = ContinuousWalk::new(g, gen_, gamma)?;
                match &energies {
                    Some(e) => walk.with_site_energies(e),
                    None => Ok(walk),
                }
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// Node probabilities at time `t` for a walker started at `start`.
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_time("t", t)?;
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.probabilities(start, t)).raise()?,
        ))
    }

    /// `exp(−iHt)|ψ₀⟩` for an arbitrary (not necessarily normalised) vector.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        check_time("t", t)?;
        let psi0 = complex_vec(&psi0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&psi0, t)).raise()?))
    }

    /// Infinite-time-averaged distribution (degenerate eigenspaces grouped).
    fn limiting_distribution<'py>(
        &self,
        py: Python<'py>,
        start: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.limiting_distribution(start)).raise()?,
        ))
    }

    /// Dimension of the smallest invariant subspace containing the start
    /// node: the number of distinct eigenvalues the walker sees.
    fn cyclic_dimension(&self, py: Python<'_>, start: usize) -> PyResult<usize> {
        let w = &self.inner;
        py.detach(|| w.cyclic_dimension(start)).raise()
    }

    /// Exact time average of the node probabilities over `[0, t_max]`.
    fn average_distribution<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_time("t_max", t_max)?;
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.average_distribution(start, t_max)).raise()?,
        ))
    }

    /// Mixing time (Aharonov et al. 2001): smallest `T` on the grid
    /// `t_max·k/steps` from which on the time-averaged distribution stays
    /// within total-variation distance `epsilon` of the limit, up to
    /// `t_max` (`None` if the bound fails at `t_max`).
    #[pyo3(signature = (start, epsilon, t_max, steps = 200))]
    fn mixing_time(
        &self,
        py: Python<'_>,
        start: usize,
        epsilon: f64,
        t_max: f64,
        steps: usize,
    ) -> PyResult<Option<f64>> {
        check_time("t_max", t_max)?;
        if epsilon.is_nan() {
            return Err(invalid("`epsilon` must not be NaN"));
        }
        let w = &self.inner;
        py.detach(|| w.mixing_time(start, epsilon, t_max, steps))
            .raise()
    }

    /// Dense propagator `U(t) = exp(−iHt)`.
    fn propagator<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        check_time("t", t)?;
        let w = &self.inner;
        let u = py.detach(|| w.propagator(t)).raise()?;
        Ok(cmatrix(py, &u))
    }

    /// Eigenvalues of `H` (ascending), or `None` for Krylov-only walks.
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        // The eigendecomposition is computed on first use: without the GIL.
        let w = &self.inner;
        py.detach(|| w.spectrum().map(<[f64]>::to_vec))
            .map(|s| real_array(py, s))
    }

    /// The site energies, or `None` if none were given.
    #[getter]
    fn site_energies<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .site_energies()
            .map(|e| real_array(py, e.to_vec()))
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
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
            "ContinuousWalk(nodes={}, generator={:?}, gamma={})",
            self.inner.num_nodes(),
            generator_name(self.inner.generator()),
            self.inner.gamma()
        )
    }
}

/// Childs–Goldstone spatial search: start in the uniform superposition and
/// evolve under `γL − Σ_w |w⟩⟨w|` (or `−γA − Σ_w |w⟩⟨w|`).
#[pyclass(name = "SpatialSearch", module = "metatron_qso", frozen)]
pub struct PySpatialSearch {
    inner: SpatialSearch,
}

#[pymethods]
impl PySpatialSearch {
    /// `gamma=None` selects `SpatialSearch.critical_gamma(graph)`.
    #[new]
    #[pyo3(signature = (graph, marked, generator = "laplacian", gamma = None))]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        marked: Indices,
        generator: &str,
        gamma: Option<f64>,
    ) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let marked = marked.into_vec();
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                let gamma = match gamma {
                    Some(x) => x,
                    None => SpatialSearch::critical_gamma(g)?,
                };
                SpatialSearch::new(g, &marked, gen_, gamma)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// Childs–Goldstone critical hopping rate `(1/N) Σ_{λ>0} 1/λ` over the
    /// Laplacian spectrum.
    #[staticmethod]
    fn critical_gamma(py: Python<'_>, graph: &PyGraph) -> PyResult<f64> {
        let g = &graph.inner;
        py.detach(|| SpatialSearch::critical_gamma(g)).raise()
    }

    /// Optimise the hopping rate (around the critical rate) and the
    /// measurement time together. `t_max=None` uses `π√N`. On complete
    /// graphs this recovers the exact optimum γ = 1/N with success 1.
    #[staticmethod]
    #[pyo3(signature = (graph, marked, generator = "laplacian", t_max = None, resolution = 400))]
    fn optimize<'py>(
        py: Python<'py>,
        graph: &PyGraph,
        marked: Indices,
        generator: &str,
        t_max: Option<f64>,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let gen_ = self::generator(generator)?;
        let marked = marked.into_vec();
        let g = &graph.inner;
        let t_max = t_max.unwrap_or(std::f64::consts::PI * (g.num_nodes() as f64).sqrt());
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(invalid("`t_max` must be positive and finite"));
        }
        let r = py
            .detach(|| SpatialSearch::optimize(g, &marked, gen_, t_max, resolution.max(2)))
            .raise()?;
        Ok(pydict!(py, {
            "gamma" => r.gamma,
            "time" => r.time,
            "success_probability" => r.success_probability,
            "expected_time" => r.expected_time,
        }))
    }

    /// Probability of measuring a marked node at time `t`.
    fn success_probability(&self, t: f64) -> PyResult<f64> {
        check_time("t", t)?;
        Ok(self.inner.success_probability(t))
    }

    /// Maximise the success probability over `[0, t_max]` (grid scan plus
    /// golden-section refinement).
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
        Ok(pydict!(py, {
            "gamma" => r.gamma,
            "time" => r.time,
            "success_probability" => r.success_probability,
            "expected_time" => r.expected_time,
        }))
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn marked(&self) -> Vec<usize> {
        self.inner.marked().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpatialSearch(marked={:?}, gamma={})",
            self.inner.marked(),
            self.inner.gamma()
        )
    }
}

/// Keyword-only settings of the transport solves.
pub(crate) fn transport_options(
    dark_tolerance: f64,
    tolerance: f64,
    max_iterations: usize,
) -> TransportOptions {
    let mut options = TransportOptions::default();
    options.dark_tolerance = dark_tolerance;
    options.tolerance = tolerance;
    options.max_iterations = max_iterations;
    options
}

/// A transport result as a `dict` (`TransportResult` in the stubs).
pub(crate) fn transport_dict<'py>(
    py: Python<'py>,
    r: TransportResult,
) -> PyResult<Bound<'py, PyDict>> {
    let solves = r
        .solves
        .iter()
        .map(|s| {
            Ok(pydict!(py, {
                "name" => &s.name,
                "iterations" => s.iterations,
                "backward_error" => s.backward_error,
                "condition" => s.condition,
            }))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(pydict!(py, {
        "efficiency" => r.efficiency,
        "arrival_time" => r.arrival_time,
        "trapped" => real_array(py, r.trapped),
        "lost" => r.lost,
        "dark" => r.dark,
        "added_dephasing" => r.added_dephasing,
        "dimension" => r.dimension,
        "slowest_decay" => r.slowest_decay,
        "condition" => r.condition,
        "error" => r.error,
        "solves" => solves,
    }))
}

/// The best rate as a `dict`: the keys of the result at that rate plus
/// `rate`, `interior` and `evaluations` (`DephasingOptimum` in the stubs).
pub(crate) fn optimum_dict<'py>(
    py: Python<'py>,
    o: DephasingOptimum,
) -> PyResult<Bound<'py, PyDict>> {
    let d = transport_dict(py, o.result)?;
    d.set_item("rate", o.rate)?;
    d.set_item("interior", o.interior)?;
    d.set_item("evaluations", o.evaluations)?;
    Ok(d)
}

/// `rows × cols` real matrix from rows (missing entries NaN).
fn rows_array<'py>(py: Python<'py>, rows: &[Vec<f64>], cols: usize) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((rows.len(), cols), |(r, c)| {
        rows[r].get(c).copied().unwrap_or(f64::NAN)
    })
    .into_pyarray(py)
}

/// A dephasing scan as a `dict` of arrays over the rates (`TransportScan`
/// in the stubs); `arrival_time` is NaN where it is not defined.
pub(crate) fn scan_dict<'py>(
    py: Python<'py>,
    rates: Vec<f64>,
    results: Vec<TransportResult>,
    traps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let column = |f: fn(&TransportResult) -> f64| real_array(py, results.iter().map(f).collect());
    let trapped: Vec<Vec<f64>> = results.iter().map(|r| r.trapped.clone()).collect();
    Ok(pydict!(py, {
        "rates" => real_array(py, rates),
        "efficiency" => column(|r| r.efficiency),
        "arrival_time" => column(|r| r.arrival_time.unwrap_or(f64::NAN)),
        "trapped" => rows_array(py, &trapped, traps),
        "lost" => column(|r| r.lost),
        "dark" => column(|r| r.dark),
        "condition" => column(|r| r.condition),
        "error" => column(|r| r.error),
    }))
}

/// A transport trace as a `dict` of arrays over the times
/// (`TransportTrace` in the stubs).
pub(crate) fn trace_dict<'py>(
    py: Python<'py>,
    states: Vec<TransportState>,
    sites: usize,
    traps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let populations: Vec<Vec<f64>> = states.iter().map(|s| s.populations.clone()).collect();
    let trapped: Vec<Vec<f64>> = states.iter().map(|s| s.trapped.clone()).collect();
    Ok(pydict!(py, {
        "times" => real_array(py, states.iter().map(|s| s.time).collect()),
        "populations" => rows_array(py, &populations, sites),
        "trapped" => rows_array(py, &trapped, traps),
        "lost" => real_array(py, states.iter().map(|s| s.lost).collect()),
    }))
}

/// Traps as `(site, rate)` pairs, from a mapping `{site: rate}` or a
/// sequence of pairs.
pub(crate) fn traps_arg(traps: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, f64)>> {
    let shape = || invalid("each trap is a pair (node, rate)");
    let items = match traps.cast::<PyMapping>() {
        Ok(m) => m.items()?.into_any(),
        Err(_) => traps.clone(),
    };
    items
        .try_iter()
        .map_err(|_| shape())?
        .map(|item| {
            let item = item?;
            // Tuples, and other sequences of two (lists).
            let pair = match item.extract::<(usize, f64)>() {
                Ok(pair) => Some(pair),
                Err(_) => item
                    .extract::<Vec<Bound<'_, PyAny>>>()
                    .ok()
                    .filter(|v| v.len() == 2)
                    .and_then(|v| Some((v[0].extract().ok()?, v[1].extract().ok()?))),
            };
            pair.ok_or_else(shape)
        })
        .collect()
}

/// One rate per site: a number (the same for every site) or `sites`
/// values.
pub(crate) fn rates_arg(name: &str, rates: &Bound<'_, PyAny>, sites: usize) -> PyResult<Vec<f64>> {
    if let Ok(v) = rates.extract::<RealArg<'_>>() {
        return Ok(real_vec(&v));
    }
    rates
        .extract::<f64>()
        .map(|r| vec![r; sites])
        .map_err(|_| invalid(format!("`{name}` must be a number or one value per site")))
}

/// Quantum walk with dephasing (Lindblad evolution of the density matrix):
/// `rate` on every node, or one rate per node in `node_dephasing`; with
/// `site_energies`, traps (`traps`: `{node: κ}` or `(node, κ)` pairs, the
/// sink of node `t` filling at `2κ ρ_tt`) and losses (`losses`: `Γ_j ≥ 0`,
/// a number or one per node). Evolution works up to 512 nodes, transport
/// up to 2048.
#[pyclass(name = "DephasingWalk", module = "metatron_qso", frozen)]
pub struct PyDephasingWalk {
    pub(crate) inner: DephasingWalk,
    n: usize,
}

#[pymethods]
impl PyDephasingWalk {
    #[new]
    #[pyo3(signature = (
        graph, generator = "laplacian", gamma = 1.0, rate = 0.0, site_energies = None,
        node_dephasing = None, traps = None, losses = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        generator: &str,
        gamma: f64,
        rate: f64,
        site_energies: Option<RealArg<'_>>,
        node_dephasing: Option<RealArg<'_>>,
        traps: Option<Bound<'_, PyAny>>,
        losses: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if rate != 0.0 && node_dephasing.is_some() {
            return Err(invalid("give either rate or node_dephasing"));
        }
        let gen_ = self::generator(generator)?;
        let (g, n) = (&graph.inner, graph.inner.num_nodes());
        let energies = site_energies.as_ref().map(real_vec);
        let dephasing = node_dephasing.as_ref().map(real_vec);
        let traps = traps.as_ref().map(traps_arg).transpose()?;
        let losses = losses
            .as_ref()
            .map(|l| rates_arg("losses", l, n))
            .transpose()?;
        let inner = py
            .detach(|| -> metatron_qso::Result<DephasingWalk> {
                let mut w = DephasingWalk::new(g, gen_, gamma, rate)?;
                if let Some(e) = &energies {
                    w = w.with_site_energies(e)?;
                }
                if let Some(r) = &dephasing {
                    w = w.with_node_dephasing(r)?;
                }
                if let Some(t) = &traps {
                    w = w.with_traps(t)?;
                }
                if let Some(l) = &losses {
                    w = w.with_losses(l)?;
                }
                Ok(w)
            })
            .raise()?;
        Ok(Self { inner, n })
    }

    /// Node populations at time `t ≥ 0` for a walker started at `start`.
    fn populations<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.populations(start, t)).raise()?,
        ))
    }

    /// Evolve the density matrix `rho0` (`n × n`) for time `t ≥ 0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        rho0: ComplexMatrixArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        let rho0 = self.density("rho0", &rho0)?;
        let w = &self.inner;
        let rho = py.detach(|| w.evolve(&rho0, t)).raise()?;
        Ok(cmatrix(py, &rho))
    }

    /// Efficiency `η` (the population that reaches the traps), mean arrival
    /// time `τ`, losses and dark weight of a walker started at `start`, by
    /// linear solves; needs a trap.
    #[pyo3(signature = (start, *, dark_tolerance = 1e-10, tolerance = 1e-13, max_iterations = 1000))]
    fn transport<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        dark_tolerance: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let o = transport_options(dark_tolerance, tolerance, max_iterations);
        let w = &self.inner;
        transport_dict(py, py.detach(|| w.transport(start, &o)).raise()?)
    }

    /// `transport` from the density matrix `rho0` (Hermitian with trace 1;
    /// positivity is the caller's).
    #[pyo3(signature = (rho0, *, dark_tolerance = 1e-10, tolerance = 1e-13, max_iterations = 1000))]
    fn transport_from<'py>(
        &self,
        py: Python<'py>,
        rho0: ComplexMatrixArg<'_>,
        dark_tolerance: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rho0 = self.density("rho0", &rho0)?;
        let o = transport_options(dark_tolerance, tolerance, max_iterations);
        let w = &self.inner;
        transport_dict(py, py.detach(|| w.transport_from(&rho0, &o)).raise()?)
    }

    /// `transport` with each of `rates` (`≥ 0`) added to every node's
    /// dephasing: arrays over the rates (one Schur form for every positive
    /// rate).
    #[pyo3(signature = (
        start, rates, *, dark_tolerance = 1e-10, tolerance = 1e-13, max_iterations = 1000
    ))]
    fn dephasing_scan<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        rates: RealArg<'_>,
        dark_tolerance: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rates = real_vec(&rates);
        let o = transport_options(dark_tolerance, tolerance, max_iterations);
        let w = &self.inner;
        let results = py.detach(|| w.dephasing_scan(start, &rates, &o)).raise()?;
        scan_dict(py, rates, results, w.traps().len())
    }

    /// The rate in `[low, high]` which, added to every node's dephasing,
    /// gives the highest efficiency, with the result there.
    #[pyo3(signature = (
        start, low = 1e-3, high = 1e3, *, dark_tolerance = 1e-10, tolerance = 1e-13,
        max_iterations = 1000
    ))]
    #[allow(clippy::too_many_arguments)]
    fn optimal_dephasing<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        low: f64,
        high: f64,
        dark_tolerance: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let o = transport_options(dark_tolerance, tolerance, max_iterations);
        let w = &self.inner;
        let best = py
            .detach(|| w.optimal_dephasing(start, low, high, &o))
            .raise()?;
        optimum_dict(py, best)
    }

    /// Orthonormal basis (columns, `n × (n − r)`) of the states that never
    /// reach a trap or a node with more than the smallest loss.
    #[pyo3(signature = (*, dark_tolerance = 1e-10, tolerance = 1e-13, max_iterations = 1000))]
    fn dark_states<'py>(
        &self,
        py: Python<'py>,
        dark_tolerance: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let o = transport_options(dark_tolerance, tolerance, max_iterations);
        let w = &self.inner;
        Ok(rmatrix(py, &py.detach(|| w.dark_states(&o)).raise()?))
    }

    /// Populations, sink populations (per trap) and the lost population at
    /// each of `times` (non-decreasing) from `start`; up to 512 nodes.
    fn transport_trace<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        times: RealArg<'_>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let times = real_vec(&times);
        let w = &self.inner;
        let states = py.detach(|| w.transport_trace(start, &times)).raise()?;
        trace_dict(py, states, self.n, w.traps().len())
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.n
    }

    /// The site energies (zeros unless given).
    #[getter]
    fn site_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.site_energies().to_vec())
    }

    /// The dephasing rate of every node.
    #[getter]
    fn node_dephasing<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.node_dephasing().to_vec())
    }

    /// The loss rate of every node.
    #[getter]
    fn losses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.losses().to_vec())
    }

    /// The traps `(node, κ)` in the order given.
    #[getter]
    fn traps(&self) -> Vec<(usize, f64)> {
        self.inner.traps().to_vec()
    }
}

impl PyDephasingWalk {
    /// An `n × n` density matrix argument.
    fn density(
        &self,
        name: &str,
        rho: &ComplexMatrixArg<'_>,
    ) -> PyResult<metatron_qso::linalg::CMatrix> {
        let (dim, data) = square_complex(name, rho)?;
        if dim != self.n {
            return Err(invalid(format!(
                "{name} must be {n}x{n}, got {dim}x{dim}",
                n = self.n
            )));
        }
        Ok(metatron_qso::linalg::CMatrix::from_row_slice(
            dim, dim, &data,
        ))
    }
}

/// Classical continuous-time random walk `p(t) = exp(−r·t·L)·e_start`.
#[pyfunction(name = "classical_walk_probabilities")]
#[pyo3(signature = (graph, start, t, rate = 1.0))]
fn classical_walk_probabilities_py<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    start: usize,
    t: f64,
    rate: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_time("t", t)?;
    check_time("rate", rate)?;
    let g = &graph.inner;
    Ok(real_array(
        py,
        py.detach(|| classical_walk_probabilities(g, start, t, rate))
            .raise()?,
    ))
}

/// Expected hitting times of `target` for the discrete simple random walk
/// (`inf` for unreachable nodes).
#[pyfunction(name = "classical_hitting_times")]
fn classical_hitting_times_py<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    target: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = &graph.inner;
    Ok(real_array(
        py,
        py.detach(|| classical_hitting_times(g, target)).raise()?,
    ))
}

/// One term of a driven walk: `("laplacian", graph)`, `("adjacency",
/// graph)` (minus the adjacency matrix), `("projector", nodes)`,
/// `("potential", values)` or `("uniform", None)` (the projector onto the
/// uniform superposition).
fn term(n: usize, spec: &Bound<'_, PyAny>) -> PyResult<Term> {
    let (kind, arg): (String, Bound<'_, PyAny>) = spec
        .extract()
        .map_err(|_| invalid("each term is a pair (kind, argument)"))?;
    match kind.to_ascii_lowercase().as_str() {
        "laplacian" | "adjacency" => {
            let g = arg
                .cast::<PyGraph>()
                .map_err(|_| invalid(format!("a {kind} term takes a Graph")))?;
            Ok(Term::graph(&g.borrow().inner, generator(&kind)?))
        }
        "projector" => {
            let nodes: Indices = arg.extract()?;
            Term::projector(n, &nodes.into_vec()).raise()
        }
        "potential" => {
            let values: Vec<f64> = arg.extract()?;
            Term::potential(&values).raise()
        }
        "uniform" => Ok(Term::Uniform),
        _ => Err(invalid(format!(
            "unknown term '{kind}' (expected laplacian, adjacency, projector, potential or uniform)"
        ))),
    }
}

/// A continuous-time walk with a time-dependent Hamiltonian
/// `H(t) = Σ_j c_j(t)·H_j` on `n_nodes` nodes; see `term` kinds in
/// `terms`. `evolve` integrates it with the fourth-order commutator-free
/// Magnus integrator and step-size control.
#[pyclass(name = "DrivenWalk", module = "metatron_qso", frozen)]
pub struct PyDrivenWalk {
    inner: DrivenWalk,
}

#[pymethods]
impl PyDrivenWalk {
    #[new]
    fn new(n_nodes: usize, terms: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let terms = terms
            .iter()
            .map(|t| term(n_nodes, t))
            .collect::<PyResult<Vec<_>>>()?;
        DrivenWalk::new(n_nodes, terms)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    /// Evolve `psi0` from `t0` to `t1`; `coefficients(t)` returns the
    /// coefficients of the terms at time `t`. `tol` bounds the error of the
    /// final state; `fixed_step` switches step-size control off; `max_step`
    /// caps the step (the coefficients are sampled inside steps only:
    /// split evolutions at jumps, and resolve pulses with `max_step`).
    /// Holds the GIL, since it calls `coefficients`. Returns a dict with
    /// `state`, `steps`, `rejected`, `error_estimate` and `forced` (steps
    /// accepted at the minimum step size above the tolerance).
    #[pyo3(signature = (
        psi0, t0, t1, coefficients, tol = 1e-8, initial_step = None, fixed_step = None,
        max_step = None, max_steps = 1_000_000
    ))]
    #[allow(clippy::too_many_arguments)]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        t0: f64,
        t1: f64,
        coefficients: &Bound<'py, PyAny>,
        tol: f64,
        initial_step: Option<f64>,
        fixed_step: Option<f64>,
        max_step: Option<f64>,
        max_steps: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = MagnusOptions {
            tol,
            initial_step,
            fixed_step,
            max_step,
            max_steps,
        };
        let psi0 = complex_vec(&psi0);
        driven_result(py, coefficients, |c| {
            self.inner.evolve(&psi0, t0, t1, c, &options)
        })
    }
}

/// Run a driven evolution whose coefficients come from the Python callable
/// `coefficients`, as the dict `DrivenWalk.evolve` returns.
pub(crate) fn driven_result<'py, E>(
    py: Python<'py>,
    coefficients: &Bound<'py, PyAny>,
    evolve: E,
) -> PyResult<Bound<'py, PyDict>>
where
    E: FnOnce(
        &mut dyn FnMut(f64) -> metatron_qso::Result<Vec<f64>>,
    ) -> metatron_qso::Result<DrivenEvolution>,
{
    let mut error: Option<PyErr> = None;
    let result = evolve(&mut |t| {
        coefficients
            .call1((t,))
            .and_then(|v| v.extract::<Vec<f64>>())
            .map_err(|e| {
                error = Some(e);
                metatron_qso::QsoError::invalid("coefficients", "the callable raised an exception")
            })
    });
    if let Some(e) = error {
        return Err(e);
    }
    let r = result.raise()?;
    Ok(pydict!(py, {
        "state" => complex_array(py, r.state),
        "steps" => r.steps,
        "rejected" => r.rejected,
        "error_estimate" => r.error_estimate,
        "forced" => r.forced,
    }))
}

fn statistics(name: &str) -> PyResult<Statistics> {
    match name.to_ascii_lowercase().as_str() {
        "bosons" => Ok(Statistics::Bosons),
        "fermions" => Ok(Statistics::Fermions),
        "distinguishable" => Ok(Statistics::Distinguishable),
        _ => Err(invalid(format!(
            "unknown statistics '{name}' (expected bosons, fermions or distinguishable)"
        ))),
    }
}

/// Walk of `particles` bosons, fermions or distinguishable particles on a
/// graph: hopping with `H = −γA` (`"adjacency"`, the default: the
/// Bose–Hubbard model) or `H = γL` (`"laplacian"`), on-site interaction
/// `on_site` (U) and interaction `neighbour` (V) between particles on
/// adjacent sites.
///
/// States are vectors over the configurations: multisets of sites for
/// bosons, sets for fermions, the site of each particle for
/// distinguishable particles. Up to 2048 configurations the walk is
/// diagonalised on first use (needed for `propagator`,
/// `transition_probabilities` and `certificate`); above, it propagates by
/// Krylov. Walks beyond 2^22 configurations, 2^25 stored sites or 2^25
/// nonzero Hamiltonian entries are refused.
#[pyclass(name = "InteractingWalk", module = "metatron_qso", frozen)]
pub struct PyInteractingWalk {
    inner: InteractingWalk,
}

#[pymethods]
impl PyInteractingWalk {
    #[new]
    #[pyo3(signature = (
        graph,
        particles,
        statistics = "bosons",
        generator = "adjacency",
        gamma = 1.0,
        on_site = 0.0,
        neighbour = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        particles: usize,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                InteractingWalk::new(g, gen_, gamma, s, particles)?
                    .with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// The sites of configuration `index`.
    fn configuration(&self, index: usize) -> PyResult<Vec<usize>> {
        Ok(self.inner.configuration(index).raise()?.to_vec())
    }

    /// Index of the configuration with particles on `sites` (any order
    /// for bosons and fermions).
    fn index_of(&self, sites: Vec<usize>) -> PyResult<usize> {
        self.inner.index_of(&sites).raise()
    }

    /// The basis state with particles on `sites`.
    fn basis_state<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.basis_state(&sites).raise()?))
    }

    /// `exp(−iHt)|psi0⟩`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let psi0 = complex_vec(&psi0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&psi0, t)).raise()?))
    }

    /// Probabilities of all configurations at time `t`, started with
    /// particles on `sites`.
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<usize>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.probabilities(&sites, t)).raise()?,
        ))
    }

    /// Mean site occupations `⟨n_i⟩` in the state `psi`.
    fn densities<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let psi = complex_vec(&psi);
        Ok(real_array(py, self.inner.densities(&psi).raise()?))
    }

    /// Density correlations `⟨n_i n_j⟩` in the state `psi`.
    fn pair_correlations<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let psi = complex_vec(&psi);
        Ok(rmatrix(py, &self.inner.pair_correlations(&psi).raise()?))
    }

    /// Dense propagator `exp(−iHt)` over the configurations.
    fn propagator<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        let w = &self.inner;
        Ok(cmatrix(py, &py.detach(|| w.propagator(t)).raise()?))
    }

    /// `|⟨a|exp(−iHt)|b⟩|²` for all pairs of configurations.
    fn transition_probabilities<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = &self.inner;
        Ok(rmatrix(
            py,
            &py.detach(|| w.transition_probabilities(t)).raise()?,
        ))
    }

    /// Graph certificate: the sorted transition probabilities at each of
    /// `times`, concatenated. Compare two with
    /// `interacting_certificate_distance` against `certificate_tolerance`.
    fn certificate<'py>(
        &self,
        py: Python<'py>,
        times: RealArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let times = real_vec(&times);
        let w = &self.inner;
        Ok(real_array(py, py.detach(|| w.certificate(&times)).raise()?))
    }

    /// The certificate from one start configuration per orbit of the group
    /// generated by `symmetries` (automorphisms of the graph, for example
    /// `graph.automorphism_generators()`): the same values as
    /// `certificate`, and beyond 2048 configurations (up to 8192) from
    /// Krylov rows.
    fn certificate_by_orbits<'py>(
        &self,
        py: Python<'py>,
        times: RealArg<'_>,
        symmetries: Vec<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let times = real_vec(&times);
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.certificate_by_orbits(&times, &symmetries))
                .raise()?,
        ))
    }

    /// Tolerance for comparing certificates at `times`:
    /// `1e-12 · max(1, max|t| · ‖H‖)` (above 2048 configurations ‖H‖ is
    /// bounded by the largest absolute row sum). Rounding grows with the time;
    /// certificates of isomorphic graphs stay far below it, and a larger
    /// distance shows, up to floating-point error, that two graphs are not
    /// isomorphic.
    fn certificate_tolerance(&self, py: Python<'_>, times: RealArg<'_>) -> PyResult<f64> {
        let times = real_vec(&times);
        let w = &self.inner;
        py.detach(|| w.certificate_tolerance(&times)).raise()
    }

    /// Eigenvalues of the Hamiltonian (ascending), or `None` above 2048
    /// configurations.
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|v| real_array(py, v.to_vec()))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn particles(&self) -> usize {
        self.inner.particles()
    }

    #[getter]
    fn statistics(&self) -> &'static str {
        self.inner.statistics().name()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn on_site(&self) -> f64 {
        self.inner.on_site()
    }

    #[getter]
    fn neighbour(&self) -> f64 {
        self.inner.neighbour()
    }

    fn __repr__(&self) -> String {
        format!(
            "InteractingWalk({} {}, {} configurations, U={}, V={})",
            self.inner.particles(),
            self.inner.statistics().name(),
            self.inner.dimension(),
            self.inner.on_site(),
            self.inner.neighbour()
        )
    }
}

/// An interacting walk on the states invariant under a group of the
/// graph's automorphisms (`symmetries`, permutations of the nodes; for
/// example `graph.stabilizer_generators([...])`): one amplitude per orbit
/// of the group on the configurations. Fermionic configurations that a
/// symmetry maps to minus themselves carry no invariant state. Orbits are
/// found among at most 2^27 configurations. `from_start` reduces by the
/// stabiliser of a start configuration.
#[pyclass(name = "SymmetricInteractingWalk", module = "metatron_qso", frozen)]
pub struct PySymmetricInteractingWalk {
    inner: SymmetricInteractingWalk,
}

#[pymethods]
impl PySymmetricInteractingWalk {
    #[new]
    #[pyo3(signature = (
        graph,
        particles,
        symmetries,
        statistics = "bosons",
        generator = "adjacency",
        gamma = 1.0,
        on_site = 0.0,
        neighbour = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        particles: usize,
        symmetries: Vec<Vec<usize>>,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                SymmetricInteractingWalk::new(g, gen_, gamma, s, particles, &symmetries)?
                    .with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// The walk started with particles on `start`, reduced by the
    /// automorphisms that fix each occupied node.
    #[staticmethod]
    #[pyo3(signature = (
        graph,
        start,
        statistics = "bosons",
        generator = "adjacency",
        gamma = 1.0,
        on_site = 0.0,
        neighbour = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_start(
        py: Python<'_>,
        graph: &PyGraph,
        start: Vec<usize>,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                SymmetricInteractingWalk::from_start(g, gen_, gamma, s, &start)?
                    .with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// The invariant state through the configuration on `sites` (the
    /// configuration itself when the group fixes it).
    fn state_of<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.state_of(&sites).raise()?))
    }

    /// `exp(−iHt)` applied to the orbit amplitudes `x0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        x0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let x0 = complex_vec(&x0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&x0, t)).raise()?))
    }

    /// Probabilities of the orbits.
    fn orbit_probabilities<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        Ok(real_array(py, self.inner.orbit_probabilities(&x).raise()?))
    }

    /// Probability of the configuration on `sites`.
    fn probability_of(&self, x: ComplexArg<'_>, sites: Vec<usize>) -> PyResult<f64> {
        let x = complex_vec(&x);
        self.inner.probability_of(&x, &sites).raise()
    }

    /// Mean site occupations `⟨n_i⟩`.
    fn densities<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        let w = &self.inner;
        Ok(real_array(py, py.detach(|| w.densities(&x)).raise()?))
    }

    /// Density correlations `⟨n_i n_j⟩`.
    fn pair_correlations<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = complex_vec(&x);
        let w = &self.inner;
        Ok(rmatrix(py, &py.detach(|| w.pair_correlations(&x)).raise()?))
    }

    /// The amplitudes on all configurations, in `InteractingWalk`'s order
    /// (at most 2^22 configurations).
    fn lift<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let x = complex_vec(&x);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.lift(&x)).raise()?))
    }

    /// Sites of the representative of orbit `o`.
    fn representative(&self, o: usize) -> PyResult<Vec<usize>> {
        if o >= self.inner.num_orbits() {
            return Err(invalid(format!("orbit {o} of {}", self.inner.num_orbits())));
        }
        Ok(self.inner.orbits().representative(o).to_vec())
    }

    /// Number of configurations in orbit `o`.
    fn orbit_size(&self, o: usize) -> PyResult<usize> {
        if o >= self.inner.num_orbits() {
            return Err(invalid(format!("orbit {o} of {}", self.inner.num_orbits())));
        }
        Ok(self.inner.orbits().size(o))
    }

    /// Eigenvalues of the reduced Hamiltonian, or `None` above 2048 orbits.
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|v| real_array(py, v.to_vec()))
    }

    #[getter]
    fn num_orbits(&self) -> usize {
        self.inner.num_orbits()
    }

    #[getter]
    fn num_configurations(&self) -> usize {
        self.inner.num_configurations()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn particles(&self) -> usize {
        self.inner.particles()
    }

    #[getter]
    fn statistics(&self) -> &'static str {
        self.inner.statistics().name()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn on_site(&self) -> f64 {
        self.inner.on_site()
    }

    #[getter]
    fn neighbour(&self) -> f64 {
        self.inner.neighbour()
    }

    fn __repr__(&self) -> String {
        format!(
            "SymmetricInteractingWalk({} {}, {} orbits of {} configurations, U={}, V={})",
            self.inner.particles(),
            self.inner.statistics().name(),
            self.inner.num_orbits(),
            self.inner.num_configurations(),
            self.inner.on_site(),
            self.inner.neighbour()
        )
    }
}

/// Interacting particles on any graph, reduced by a group of its
/// automorphisms, on the orbits reachable from the start found from
/// minimal images (the configurations one hop from each orbit's
/// representative are imaged; no other configuration is visited). With `whole_group=False` the group is the stabiliser
/// of the start's nodes and the start configuration is the initial state;
/// with `whole_group=True` it is the whole automorphism group, and states
/// are orbit states (`state_of(start)` is the start's).
#[pyclass(name = "GroupInteractingWalk", module = "metatron_qso", frozen)]
pub struct PyGroupInteractingWalk {
    inner: GroupInteractingWalk,
}

#[pymethods]
impl PyGroupInteractingWalk {
    #[new]
    #[pyo3(signature = (graph, start, whole_group = false, statistics = "bosons", generator = "adjacency", gamma = 1.0, on_site = 0.0, neighbour = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        start: Vec<usize>,
        whole_group: bool,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                let walk = if whole_group {
                    let group = g.automorphism_group();
                    GroupInteractingWalk::new(g, gen_, gamma, s, &start, &group)?
                } else {
                    GroupInteractingWalk::from_start(g, gen_, gamma, s, &start)?
                };
                walk.with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// The orbit state of the configuration `sites`.
    fn state_of<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.state_of(&sites).raise()?))
    }

    /// `exp(−iHt)` applied to `x0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        x0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let x0 = complex_vec(&x0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&x0, t)).raise()?))
    }

    fn orbit_probabilities<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        Ok(real_array(py, self.inner.orbit_probabilities(&x).raise()?))
    }

    /// The probability of the configuration `sites`.
    fn probability_of(&self, x: ComplexArg<'_>, sites: Vec<usize>) -> PyResult<f64> {
        let x = complex_vec(&x);
        self.inner.probability_of(&x, &sites).raise()
    }

    /// The density of particles on every node.
    fn densities<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        Ok(real_array(py, self.inner.densities(&x).raise()?))
    }

    fn representative(&self, o: usize) -> PyResult<Vec<usize>> {
        if o >= self.inner.num_orbits() {
            return Err(invalid(format!("orbit {o} of {}", self.inner.num_orbits())));
        }
        Ok(self.inner.representative(o).to_vec())
    }

    fn orbit_size(&self, o: usize) -> PyResult<f64> {
        if o >= self.inner.num_orbits() {
            return Err(invalid(format!("orbit {o} of {}", self.inner.num_orbits())));
        }
        Ok(self.inner.orbit_size(o))
    }

    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|s| real_array(py, s.to_vec()))
    }

    #[getter]
    fn num_orbits(&self) -> usize {
        self.inner.num_orbits()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn log_num_configurations(&self) -> f64 {
        self.inner.log_num_configurations()
    }

    #[getter]
    fn log_group_order(&self) -> f64 {
        self.inner.log_group_order()
    }

    #[getter]
    fn start(&self) -> Vec<usize> {
        self.inner.start().to_vec()
    }

    #[getter]
    fn particles(&self) -> usize {
        self.inner.particles()
    }

    #[getter]
    fn statistics(&self) -> &'static str {
        self.inner.statistics().name()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn on_site(&self) -> f64 {
        self.inner.on_site()
    }

    #[getter]
    fn neighbour(&self) -> f64 {
        self.inner.neighbour()
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupInteractingWalk({} nodes, {} {}, {} orbits, U={}, V={})",
            self.inner.num_nodes(),
            self.inner.particles(),
            self.inner.statistics().name(),
            self.inner.num_orbits(),
            self.inner.on_site(),
            self.inner.neighbour()
        )
    }
}

/// Interacting particles on a Hamming, Johnson or Kneser graph, whose
/// symmetries act on the places of its nodes' words, on the orbits of the
/// start's pointwise stabiliser found from canonical forms (the graph is
/// never built): `hamming(d, q, start)`, `johnson(n, k, start)`,
/// `kneser(n, k, start)`. Nodes are numbered as the graph generators
/// number them (`word`, `node`).
#[pyclass(name = "CoordinateInteractingWalk", module = "metatron_qso", frozen)]
pub struct PyCoordinateInteractingWalk {
    inner: CoordinateInteractingWalk,
}

impl PyCoordinateInteractingWalk {
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        graph: CoordinateGraph,
        start: Vec<u64>,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let inner = py
            .detach(|| {
                CoordinateInteractingWalk::from_start(graph, gen_, gamma, s, &start)?
                    .with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyCoordinateInteractingWalk {
    /// On the Hamming graph `H(d, q)` (`Q_d` for `q = 2`, `K_q` for
    /// `d = 1`), node `Σ x_j q^j`.
    #[staticmethod]
    #[pyo3(signature = (d, q, start, statistics = "bosons", generator = "adjacency", gamma = 1.0, on_site = 0.0, neighbour = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn hamming(
        py: Python<'_>,
        d: usize,
        q: usize,
        start: Vec<u64>,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        Self::build(
            py,
            CoordinateGraph::Hamming { d, q },
            start,
            statistics,
            generator,
            gamma,
            on_site,
            neighbour,
        )
    }

    /// On the Johnson graph `J(n, k)`, nodes in colex order.
    #[staticmethod]
    #[pyo3(signature = (n, k, start, statistics = "bosons", generator = "adjacency", gamma = 1.0, on_site = 0.0, neighbour = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn johnson(
        py: Python<'_>,
        n: usize,
        k: usize,
        start: Vec<u64>,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        Self::build(
            py,
            CoordinateGraph::Johnson { n, k },
            start,
            statistics,
            generator,
            gamma,
            on_site,
            neighbour,
        )
    }

    /// On the Kneser graph `K(n, k)`, nodes in colex order.
    #[staticmethod]
    #[pyo3(signature = (n, k, start, statistics = "bosons", generator = "adjacency", gamma = 1.0, on_site = 0.0, neighbour = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn kneser(
        py: Python<'_>,
        n: usize,
        k: usize,
        start: Vec<u64>,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        Self::build(
            py,
            CoordinateGraph::Kneser { n, k },
            start,
            statistics,
            generator,
            gamma,
            on_site,
            neighbour,
        )
    }

    /// The orbit state of the configuration `sites` (the start's is the
    /// start).
    fn state_of<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<u64>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.state_of(&sites).raise()?))
    }

    /// `exp(−iHt)` applied to `x0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        x0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let x0 = complex_vec(&x0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&x0, t)).raise()?))
    }

    fn orbit_probabilities<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        Ok(real_array(py, self.inner.orbit_probabilities(&x).raise()?))
    }

    /// The probability of the configuration `sites`.
    fn probability_of(&self, x: ComplexArg<'_>, sites: Vec<u64>) -> PyResult<f64> {
        let x = complex_vec(&x);
        self.inner.probability_of(&x, &sites).raise()
    }

    /// Per orbit of nodes: its smallest node among the representatives,
    /// its size and the density on each of its nodes.
    fn node_densities(&self, x: ComplexArg<'_>) -> PyResult<Vec<(u64, f64, f64)>> {
        let x = complex_vec(&x);
        self.inner.node_densities(&x).raise()
    }

    fn representative(&self, o: usize) -> PyResult<Vec<u64>> {
        if o >= self.inner.num_orbits() {
            return Err(invalid(format!("orbit {o} of {}", self.inner.num_orbits())));
        }
        Ok(self.inner.representative(o).to_vec())
    }

    /// The configurations in orbit `o` (a float: they can be many).
    fn orbit_size(&self, o: usize) -> PyResult<f64> {
        if o >= self.inner.num_orbits() {
            return Err(invalid(format!("orbit {o} of {}", self.inner.num_orbits())));
        }
        Ok(self.inner.orbit_size(o))
    }

    /// The word (letters, or bits of the subset) of `node`.
    fn word(&self, node: u64) -> PyResult<Vec<usize>> {
        self.inner.graph().word(node).raise()
    }

    /// The node of a word.
    fn node(&self, word: Vec<usize>) -> PyResult<u64> {
        self.inner.graph().node(&word).raise()
    }

    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|v| real_array(py, v.to_vec()))
    }

    #[getter]
    fn num_orbits(&self) -> usize {
        self.inner.num_orbits()
    }

    #[getter]
    fn num_nodes(&self) -> u64 {
        self.inner.graph().num_nodes()
    }

    /// `ln` of the number of configurations.
    #[getter]
    fn log_num_configurations(&self) -> f64 {
        self.inner.log_num_configurations()
    }

    /// `ln` of the order of the group.
    #[getter]
    fn log_group_order(&self) -> f64 {
        self.inner.log_group_order()
    }

    #[getter]
    fn start(&self) -> Vec<u64> {
        self.inner.start().to_vec()
    }

    #[getter]
    fn particles(&self) -> usize {
        self.inner.particles()
    }

    #[getter]
    fn statistics(&self) -> &'static str {
        self.inner.statistics().name()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn on_site(&self) -> f64 {
        self.inner.on_site()
    }

    #[getter]
    fn neighbour(&self) -> f64 {
        self.inner.neighbour()
    }

    fn __repr__(&self) -> String {
        format!(
            "CoordinateInteractingWalk({:?}, {} {}, {} orbits)",
            self.inner.graph(),
            self.inner.particles(),
            self.inner.statistics().name(),
            self.inner.num_orbits()
        )
    }
}

/// `particles` (1 to 4) interacting particles on the hypercube `Q_d`, on
/// states invariant under the permutations of the coordinates, written down
/// as compositions of `d` without building the graph (`Q_40` with two
/// bosons has 6391 states). `localized()` puts every particle at node 0.
#[pyclass(name = "HypercubeParticles", module = "metatron_qso", frozen)]
pub struct PyHypercubeParticles {
    inner: HypercubeParticles,
}

#[pymethods]
impl PyHypercubeParticles {
    #[new]
    #[pyo3(signature = (
        d,
        particles,
        statistics = "bosons",
        generator = "adjacency",
        gamma = 1.0,
        on_site = 0.0,
        neighbour = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        d: usize,
        particles: usize,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let inner = py
            .detach(|| {
                HypercubeParticles::new(d, s, particles, gen_, gamma)?
                    .with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// Every particle at node `0…0`.
    fn localized<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.localized().raise()?))
    }

    /// The invariant state of the configurations with `counts[p]`
    /// coordinates of pattern `p` (bit `j` of `p`: the coordinate of
    /// particle `j`).
    fn state_of<'py>(
        &self,
        py: Python<'py>,
        counts: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.state_of(&counts).raise()?))
    }

    /// `exp(−iHt)` applied to `x0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        x0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let x0 = complex_vec(&x0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&x0, t)).raise()?))
    }

    /// Probability that every particle is at the node opposite `0…0`.
    fn antipode_probability(&self, x: ComplexArg<'_>) -> PyResult<f64> {
        let x = complex_vec(&x);
        self.inner.antipode_probability(&x).raise()
    }

    /// Expected number of particles at each Hamming weight `0..=d`.
    fn weights<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        Ok(real_array(py, self.inner.weights(&x).raise()?))
    }

    /// Distribution of the Hamming distance between two particles,
    /// averaged over the pairs.
    fn pair_distances<'py>(
        &self,
        py: Python<'py>,
        x: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = complex_vec(&x);
        Ok(real_array(py, self.inner.pair_distances(&x).raise()?))
    }

    /// Eigenvalues of the reduced Hamiltonian, or `None` above 2048 states.
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|v| real_array(py, v.to_vec()))
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn particles(&self) -> usize {
        self.inner.particles()
    }

    #[getter]
    fn statistics(&self) -> &'static str {
        self.inner.statistics().name()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn on_site(&self) -> f64 {
        self.inner.on_site()
    }

    #[getter]
    fn neighbour(&self) -> f64 {
        self.inner.neighbour()
    }

    fn __repr__(&self) -> String {
        format!(
            "HypercubeParticles(Q{}, {} {}, {} states, U={}, V={})",
            self.inner.dimension(),
            self.inner.particles(),
            self.inner.statistics().name(),
            self.inner.num_states(),
            self.inner.on_site(),
            self.inner.neighbour()
        )
    }
}

/// Largest difference between two certificates of
/// `InteractingWalk.certificate`: infinite if their lengths differ, NaN if
/// either holds NaN. Above `InteractingWalk.certificate_tolerance` the
/// graphs are not isomorphic, up to floating-point error.
#[pyfunction(name = "interacting_certificate_distance")]
fn certificate_distance_py(a: RealArg<'_>, b: RealArg<'_>) -> f64 {
    certificate_distance(&real_vec(&a), &real_vec(&b))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContinuousWalk>()?;
    m.add_class::<PyInteractingWalk>()?;
    m.add_class::<PySymmetricInteractingWalk>()?;
    m.add_class::<PyHypercubeParticles>()?;
    m.add_class::<PyCoordinateInteractingWalk>()?;
    m.add_class::<PyGroupInteractingWalk>()?;
    m.add_function(wrap_pyfunction!(certificate_distance_py, m)?)?;
    m.add_class::<PySpatialSearch>()?;
    m.add_class::<PyDephasingWalk>()?;
    m.add_class::<PyDrivenWalk>()?;
    m.add_function(wrap_pyfunction!(classical_walk_probabilities_py, m)?)?;
    m.add_function(wrap_pyfunction!(classical_hitting_times_py, m)?)?;
    Ok(())
}
