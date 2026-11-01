//! Slow noise averaged exactly: Markov processes, fluctuators and Kubo's
//! stochastic Liouville equation.

use metatron_qso::C64;
use metatron_qso::linalg::CMatrix;
use metatron_qso::noise::Spread;
use metatron_qso::noise::slow::{Fluctuators, JointState, MarkovProcess, StochasticLiouville};
use numpy::ndarray::Array3;
use numpy::{AllowTypeChange, IntoPyArray, PyArray3, PyArrayLike3};
use pyo3::prelude::*;

use crate::{ComplexMatrixArg, Raise, invalid, square_complex};

/// Joint density matrices: an array of shape `(joint states, d, d)`.
type JointArg<'py> = PyArrayLike3<'py, C64, AllowTypeChange>;

fn spread(kind: &str, width: f64) -> PyResult<Spread> {
    match kind {
        "gaussian" => Ok(Spread::Gaussian { sigma: width }),
        "uniform" => Ok(Spread::Uniform { half_width: width }),
        other => Err(invalid(format!(
            "unknown spread {other:?} (use \"gaussian\" or \"uniform\")"
        ))),
    }
}

fn pulses_or_empty(pulses: Option<Vec<f64>>) -> Vec<f64> {
    pulses.unwrap_or_default()
}

/// A classical Markov process with finitely many states: a frequency
/// shift (angular frequency) in each state, switching rates
/// `(from, to, rate)` and the probabilities at time 0.
#[pyclass(name = "MarkovProcess", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyMarkovProcess {
    pub inner: MarkovProcess,
}

fn process(p: metatron_qso::Result<MarkovProcess>) -> PyResult<PyMarkovProcess> {
    p.raise().map(|inner| PyMarkovProcess { inner })
}

#[pymethods]
impl PyMarkovProcess {
    #[new]
    fn new(values: Vec<f64>, rates: Vec<(usize, usize, f64)>, initial: Vec<f64>) -> PyResult<Self> {
        process(MarkovProcess::new(values, rates, initial))
    }

    /// Random telegraph noise `±amplitude` flipping at `rate` both ways,
    /// started stationary (as a device's telegraph noise).
    #[staticmethod]
    fn telegraph(amplitude: f64, rate: f64) -> PyResult<Self> {
        process(MarkovProcess::telegraph(amplitude, rate))
    }

    /// Ornstein–Uhlenbeck noise of standard deviation `sigma` and
    /// correlation time `correlation_time`, as the Ehrenfest chain of
    /// `levels − 1` telegraph fluctuators (exponential correlation at every
    /// level count, Gaussian as the levels grow).
    #[staticmethod]
    fn ornstein_uhlenbeck(sigma: f64, correlation_time: f64, levels: usize) -> PyResult<Self> {
        process(MarkovProcess::ornstein_uhlenbeck(
            sigma,
            correlation_time,
            levels,
        ))
    }

    /// Quasi-static noise: a constant drawn from a `"gaussian"` (standard
    /// deviation `width`) or `"uniform"` (half-width `width`) distribution,
    /// on the `nodes` points of its Gauss rule.
    #[staticmethod]
    fn quasi_static(kind: &str, width: f64, nodes: usize) -> PyResult<Self> {
        process(MarkovProcess::quasi_static(spread(kind, width)?, nodes))
    }

    /// The two processes together, independent: their values add.
    fn product(&self, other: &PyMarkovProcess) -> PyResult<Self> {
        process(self.inner.product(&other.inner))
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    #[getter]
    fn rates(&self) -> Vec<(usize, usize, f64)> {
        self.inner.rates().to_vec()
    }

    #[getter]
    fn initial(&self) -> Vec<f64> {
        self.inner.initial().to_vec()
    }

    /// The mean value at time `t`.
    fn mean(&self, t: f64) -> PyResult<f64> {
        self.inner.mean(t).raise()
    }

    /// `⟨ν(0)ν(τ)⟩` from the distribution at time 0.
    fn correlation(&self, tau: f64) -> PyResult<f64> {
        self.inner.correlation(tau).raise()
    }

    /// The coherence `⟨exp(−i∫f ν dt)⟩` of a qubit dephased by the process
    /// over `t`, with ideal π pulses at the times `pulses` (`f` flips sign
    /// at each).
    #[pyo3(signature = (t, pulses = None))]
    fn dephasing(&self, py: Python<'_>, t: f64, pulses: Option<Vec<f64>>) -> PyResult<C64> {
        let pulses = pulses_or_empty(pulses);
        py.detach(|| self.inner.dephasing(&pulses, t)).raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "MarkovProcess(states={}, rates={})",
            self.inner.num_states(),
            self.inner.rates().len()
        )
    }
}

/// Independent processes whose values add up (1/f noise from telegraph
/// fluctuators): their dephasing is the product of their factors.
#[pyclass(name = "Fluctuators", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyFluctuators {
    pub inner: Fluctuators,
}

#[pymethods]
impl PyFluctuators {
    #[new]
    fn new(processes: Vec<PyRef<'_, PyMarkovProcess>>) -> PyResult<Self> {
        Fluctuators::new(processes.iter().map(|p| p.inner.clone()).collect())
            .raise()
            .map(|inner| Self { inner })
    }

    /// 1/f noise: `count` telegraph fluctuators of `amplitude` with rates
    /// spread evenly in their logarithm from `lowest` to `highest`.
    #[staticmethod]
    fn one_over_f(amplitude: f64, lowest: f64, highest: f64, count: usize) -> PyResult<Self> {
        Fluctuators::one_over_f(amplitude, lowest, highest, count)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn processes(&self) -> Vec<PyMarkovProcess> {
        self.inner
            .processes()
            .iter()
            .map(|p| PyMarkovProcess { inner: p.clone() })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.processes().len()
    }

    /// The coherence under all of them (the product of their factors).
    #[pyo3(signature = (t, pulses = None))]
    fn dephasing(&self, py: Python<'_>, t: f64, pulses: Option<Vec<f64>>) -> PyResult<C64> {
        let pulses = pulses_or_empty(pulses);
        py.detach(|| self.inner.dephasing(&pulses, t)).raise()
    }

    /// `⟨ν(0)ν(τ)⟩` of their sum.
    fn correlation(&self, py: Python<'_>, tau: f64) -> PyResult<f64> {
        py.detach(|| self.inner.correlation(tau)).raise()
    }

    /// Their joint process (for a `StochasticLiouville` equation).
    fn combined(&self) -> PyResult<PyMarkovProcess> {
        process(self.inner.combined())
    }

    fn __repr__(&self) -> String {
        format!("Fluctuators(count={})", self.inner.processes().len())
    }
}

fn square_matrix(name: &str, m: &ComplexMatrixArg<'_>) -> PyResult<CMatrix> {
    let (d, data) = square_complex(name, m)?;
    Ok(CMatrix::from_row_slice(d, d, &data))
}

/// Kubo's stochastic Liouville equation: a system of dimension
/// `dimension` under independent `processes`, each given with the
/// Hermitian operator it couples through (`H_x = H + Σ ν_k B_k`), and
/// Lindblad `dissipators` (rates included). States are arrays of shape
/// `(num_joint_states, d, d)`: one density matrix per joint state of the
/// processes, adding up to the state of the system.
#[pyclass(name = "StochasticLiouville", module = "metatron_qso", frozen)]
pub struct PyStochasticLiouville {
    inner: StochasticLiouville,
}

impl PyStochasticLiouville {
    fn joint(&self, state: &JointArg<'_>) -> PyResult<JointState> {
        let view = state.as_array();
        let (j, r, c) = view.dim();
        let d = self.inner.dimension();
        if j != self.inner.num_joint_states() || r != d || c != d {
            return Err(invalid(format!(
                "a state of shape ({}, {d}, {d}), got ({j}, {r}, {c})",
                self.inner.num_joint_states()
            )));
        }
        let blocks: Vec<CMatrix> = (0..j)
            .map(|x| CMatrix::from_fn(d, d, |a, b| view[[x, a, b]]))
            .collect();
        JointState::from_blocks(&blocks).raise()
    }

    fn array<'py>(&self, py: Python<'py>, state: &JointState) -> Bound<'py, PyArray3<C64>> {
        let d = state.dimension();
        let blocks: Vec<CMatrix> = (0..state.num_joint_states())
            .map(|x| state.state(x))
            .collect();
        Array3::from_shape_fn((blocks.len(), d, d), |(x, a, b)| blocks[x][(a, b)]).into_pyarray(py)
    }
}

#[pymethods]
impl PyStochasticLiouville {
    #[new]
    #[pyo3(signature = (dimension, processes, dissipators = None))]
    fn new(
        dimension: usize,
        processes: Vec<(PyRef<'_, PyMarkovProcess>, ComplexMatrixArg<'_>)>,
        dissipators: Option<Vec<ComplexMatrixArg<'_>>>,
    ) -> PyResult<Self> {
        let mut list = Vec::with_capacity(processes.len());
        for (p, b) in &processes {
            list.push((p.inner.clone(), square_matrix("coupling", b)?));
        }
        let mut sle = StochasticLiouville::new(dimension, list).raise()?;
        if let Some(ops) = dissipators {
            let mut l = Vec::with_capacity(ops.len());
            for op in &ops {
                l.push(square_matrix("dissipator", op)?);
            }
            sle = sle.with_dissipators(l).raise()?;
        }
        Ok(Self { inner: sle })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn num_joint_states(&self) -> usize {
        self.inner.num_joint_states()
    }

    /// The system in `rho`, the processes in their distributions at time 0.
    fn start<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
    ) -> PyResult<Bound<'py, PyArray3<C64>>> {
        let rho = square_matrix("rho", &rho)?;
        let state = self.inner.start(&rho).raise()?;
        Ok(self.array(py, &state))
    }

    /// `state` after a time `t` under `hamiltonian`, the noise and the
    /// dissipators.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        state: JointArg<'_>,
        hamiltonian: ComplexMatrixArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray3<C64>>> {
        let state = self.joint(&state)?;
        let h = square_matrix("hamiltonian", &hamiltonian)?;
        let out = py.detach(|| self.inner.evolve(&state, &h, t)).raise()?;
        Ok(self.array(py, &out))
    }

    /// `state` after an ideal pulse `unitary` in every joint state.
    fn pulse<'py>(
        &self,
        py: Python<'py>,
        state: JointArg<'_>,
        unitary: ComplexMatrixArg<'_>,
    ) -> PyResult<Bound<'py, PyArray3<C64>>> {
        let state = self.joint(&state)?;
        let u = square_matrix("unitary", &unitary)?;
        let out = self.inner.pulse(&state, &u).raise()?;
        Ok(self.array(py, &out))
    }

    fn __repr__(&self) -> String {
        format!(
            "StochasticLiouville(dimension={}, joint_states={})",
            self.inner.dimension(),
            self.inner.num_joint_states()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMarkovProcess>()?;
    m.add_class::<PyFluctuators>()?;
    m.add_class::<PyStochasticLiouville>()?;
    Ok(())
}
