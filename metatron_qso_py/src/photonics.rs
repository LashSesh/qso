//! Linear optics: permanents, interferometers and boson sampling.

use metatron_qso::Rng;
use metatron_qso::linalg::CMatrix;
use metatron_qso::linalg::RMatrix;
use metatron_qso::photonics::{self, BosonSampler, GaussianState, ImperfectBosonSampler};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::{
    ComplexArg, ComplexMatrixArg, Raise, RealArg, RealMatrixArg, check_finite, cmatrix,
    complex_vec, invalid, real_array, real_vec, rmatrix,
};

/// Complex matrix from a 2-D array argument.
fn matrix(name: &str, a: &ComplexMatrixArg<'_>) -> PyResult<CMatrix> {
    let view = a.as_array();
    let (rows, cols) = view.dim();
    if view.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(invalid(format!("`{name}` must be finite")));
    }
    Ok(CMatrix::from_fn(rows, cols, |r, c| view[(r, c)]))
}

/// `(patterns, probabilities)` returned by `BosonSampler.distribution`.
type DistributionArrays<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<f64>>);

/// Occupation patterns as an integer matrix (one pattern per row).
fn patterns_array<'py>(
    py: Python<'py>,
    patterns: &[Vec<usize>],
    n_modes: usize,
) -> Bound<'py, PyArray2<i64>> {
    Array2::from_shape_fn((patterns.len(), n_modes), |(k, i)| patterns[k][i] as i64)
        .into_pyarray(py)
}

/// Permanent of a square complex matrix (Glynn's formula, `O(n·2ⁿ)`;
/// dimension at most 40).
#[pyfunction]
fn permanent(py: Python<'_>, matrix: ComplexMatrixArg<'_>) -> PyResult<metatron_qso::C64> {
    let a = self::matrix("matrix", &matrix)?;
    py.detach(|| photonics::permanent(&a)).raise()
}

/// Haar-random `modes × modes` unitary.
#[pyfunction]
#[pyo3(signature = (modes, seed = 0))]
fn haar_unitary<'py>(
    py: Python<'py>,
    modes: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
    let u = py
        .detach(|| photonics::haar_unitary(modes, &mut Rng::seed_from_u64(seed)))
        .raise()?;
    Ok(cmatrix(py, &u))
}

/// Two-mode beam splitter `[[cos θ, −e^{−iφ} sin θ], [e^{iφ} sin θ, cos θ]]`
/// (`theta = π/4` is 50:50).
#[pyfunction]
#[pyo3(signature = (theta, phi = 0.0))]
fn beam_splitter(
    py: Python<'_>,
    theta: f64,
    phi: f64,
) -> PyResult<Bound<'_, PyArray2<metatron_qso::C64>>> {
    check_finite("theta/phi", &[theta, phi])?;
    Ok(cmatrix(py, &photonics::beam_splitter(theta, phi)))
}

/// `n` single photons sent through an `m`-mode interferometer.
///
/// `unitary` is the `m × m` interferometer (input mode `j` → output mode
/// `i` with amplitude `U[i, j]`); `input[j]` is the number of photons
/// entering mode `j`. Probabilities are exact permanents; `sample` draws
/// exact samples (Clifford & Clifford 2018).
#[pyclass(name = "BosonSampler", module = "metatron_qso", frozen)]
pub struct PyBosonSampler {
    inner: BosonSampler,
}

#[pymethods]
impl PyBosonSampler {
    #[new]
    fn new(unitary: ComplexMatrixArg<'_>, input: Vec<usize>) -> PyResult<Self> {
        let u = matrix("unitary", &unitary)?;
        Ok(Self {
            inner: BosonSampler::new(u, &input).raise()?,
        })
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn n_photons(&self) -> usize {
        self.inner.n_photons()
    }

    #[getter]
    fn input(&self) -> Vec<usize> {
        self.inner.input().to_vec()
    }

    #[getter]
    fn unitary<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<metatron_qso::C64>> {
        cmatrix(py, self.inner.unitary())
    }

    /// Transition amplitude to the output occupation `output`.
    fn amplitude(&self, py: Python<'_>, output: Vec<usize>) -> PyResult<metatron_qso::C64> {
        let s = &self.inner;
        py.detach(|| s.amplitude(&output)).raise()
    }

    /// Probability of `output` for indistinguishable photons.
    fn probability(&self, py: Python<'_>, output: Vec<usize>) -> PyResult<f64> {
        let s = &self.inner;
        py.detach(|| s.probability(&output)).raise()
    }

    /// Probability of `output` for distinguishable photons.
    fn probability_distinguishable(&self, py: Python<'_>, output: Vec<usize>) -> PyResult<f64> {
        let s = &self.inner;
        py.detach(|| s.probability_distinguishable(&output)).raise()
    }

    /// Exact output distribution: `(patterns, probabilities)` with one
    /// pattern per row.
    #[pyo3(signature = (distinguishable = false))]
    fn distribution<'py>(
        &self,
        py: Python<'py>,
        distinguishable: bool,
    ) -> PyResult<DistributionArrays<'py>> {
        let s = &self.inner;
        let d = py
            .detach(|| {
                if distinguishable {
                    s.distribution_distinguishable()
                } else {
                    s.distribution()
                }
            })
            .raise()?;
        Ok((
            patterns_array(py, &d.patterns, s.n_modes()),
            real_array(py, d.probabilities),
        ))
    }

    /// `shots` samples as an integer matrix (one output occupation per
    /// row). Indistinguishable photons need at most one photon per input
    /// mode.
    #[pyo3(signature = (shots, seed = 0, distinguishable = false))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
        distinguishable: bool,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let s = &self.inner;
        let samples = py
            .detach(|| {
                let mut rng = Rng::seed_from_u64(seed);
                if distinguishable {
                    s.sample_distinguishable(shots, &mut rng)
                } else {
                    s.sample(shots, &mut rng)
                }
            })
            .raise()?;
        Ok(patterns_array(py, &samples, s.n_modes()))
    }

    fn __repr__(&self) -> String {
        format!(
            "BosonSampler(n_modes={}, n_photons={})",
            self.inner.n_modes(),
            self.inner.n_photons()
        )
    }
}

/// Hafnian of a symmetric complex matrix (`loop=True`: the loop hafnian,
/// with loops weighted by the diagonal); dimension at most 40.
#[pyfunction]
#[pyo3(signature = (matrix, r#loop = false))]
fn hafnian(
    py: Python<'_>,
    matrix: ComplexMatrixArg<'_>,
    r#loop: bool,
) -> PyResult<metatron_qso::C64> {
    let a = self::matrix("matrix", &matrix)?;
    py.detach(|| {
        if r#loop {
            photonics::loop_hafnian(&a)
        } else {
            photonics::hafnian(&a)
        }
    })
    .raise()
}

/// Hafnian of the matrix that repeats row and column `i` `rpt[i]` times
/// (Kan's formula, `Π (rpt_i + 1)` terms); with `mu`, the loop hafnian
/// whose diagonal is `mu` (repeated likewise).
#[pyfunction]
#[pyo3(signature = (matrix, rpt, mu = None))]
fn hafnian_repeated(
    py: Python<'_>,
    matrix: ComplexMatrixArg<'_>,
    rpt: Vec<usize>,
    mu: Option<ComplexArg<'_>>,
) -> PyResult<metatron_qso::C64> {
    let a = self::matrix("matrix", &matrix)?;
    let mu = mu.map(|m| complex_vec(&m));
    py.detach(|| match &mu {
        None => photonics::hafnian_repeated(&a, &rpt),
        Some(mu) => photonics::loop_hafnian_repeated(&a, mu, &rpt),
    })
    .raise()
}

/// Torontonian of a `2N × 2N` matrix in the ordering `(a, a†)`.
#[pyfunction]
fn torontonian(py: Python<'_>, matrix: ComplexMatrixArg<'_>) -> PyResult<metatron_qso::C64> {
    let a = self::matrix("matrix", &matrix)?;
    py.detach(|| photonics::torontonian(&a)).raise()
}

/// A Gaussian state of light: covariance matrix and means of the
/// quadratures `(x₁…x_m, p₁…p_m)` with `ħ = 2` (the vacuum has `cov = I`),
/// as in The Walrus and Strawberry Fields. Operations return the state, so
/// they chain.
#[pyclass(name = "GaussianState", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyGaussianState {
    inner: GaussianState,
}

type GSlf<'py> = PyRefMut<'py, PyGaussianState>;

#[pymethods]
impl PyGaussianState {
    /// The vacuum of `n_modes` modes.
    #[new]
    fn new(n_modes: usize) -> PyResult<Self> {
        GaussianState::vacuum(n_modes)
            .raise()
            .map(|inner| Self { inner })
    }

    /// Thermal states with mean photon numbers `nbar`.
    #[staticmethod]
    fn thermal(nbar: Vec<f64>) -> PyResult<Self> {
        GaussianState::thermal(&nbar)
            .raise()
            .map(|inner| Self { inner })
    }

    /// A state from its covariance matrix and means (checked against the
    /// uncertainty principle).
    #[staticmethod]
    fn from_parts(cov: RealMatrixArg<'_>, means: RealArg<'_>) -> PyResult<Self> {
        let view = cov.as_array();
        let (r, c) = view.dim();
        let cov = RMatrix::from_fn(r, c, |i, j| view[(i, j)]);
        GaussianState::from_parts(cov, real_vec(&means))
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rmatrix(py, self.inner.cov())
    }

    #[getter]
    fn means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.means().to_vec())
    }

    /// Squeeze `mode` by `r` at angle `phi` (`⟨n⟩ = sinh² r` from vacuum).
    #[pyo3(signature = (mode, r, phi = 0.0))]
    fn squeeze(mut slf: GSlf<'_>, mode: usize, r: f64, phi: f64) -> PyResult<GSlf<'_>> {
        slf.inner.squeeze(mode, r, phi).raise()?;
        Ok(slf)
    }

    /// Displace `mode` by the complex amplitude `alpha`.
    fn displace(mut slf: GSlf<'_>, mode: usize, alpha: metatron_qso::C64) -> PyResult<GSlf<'_>> {
        slf.inner.displace(mode, alpha).raise()?;
        Ok(slf)
    }

    /// Phase rotation `a → e^{iθ} a`.
    fn rotate(mut slf: GSlf<'_>, mode: usize, theta: f64) -> PyResult<GSlf<'_>> {
        slf.inner.rotate(mode, theta).raise()?;
        Ok(slf)
    }

    /// The interferometer `a → U a` on `modes` (default: all modes).
    #[pyo3(signature = (unitary, modes = None))]
    fn interferometer<'py>(
        mut slf: GSlf<'py>,
        unitary: ComplexMatrixArg<'_>,
        modes: Option<Vec<usize>>,
    ) -> PyResult<GSlf<'py>> {
        let u = matrix("unitary", &unitary)?;
        let modes = modes.unwrap_or_else(|| (0..slf.inner.n_modes()).collect());
        slf.inner.interferometer(&u, &modes).raise()?;
        Ok(slf)
    }

    /// Loss on `mode` with transmission `eta` (and a thermal environment
    /// with `nbar` photons).
    #[pyo3(signature = (mode, eta, nbar = 0.0))]
    fn loss(mut slf: GSlf<'_>, mode: usize, eta: f64, nbar: f64) -> PyResult<GSlf<'_>> {
        slf.inner.loss(mode, eta, nbar).raise()?;
        Ok(slf)
    }

    /// Mean photon number of every mode.
    fn mean_photon_numbers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.mean_photon_numbers())
    }

    /// The state of `modes` alone.
    fn reduced(&self, modes: Vec<usize>) -> PyResult<Self> {
        self.inner
            .reduced(&modes)
            .raise()
            .map(|inner| Self { inner })
    }

    /// A copy of the state.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Probability of the photon numbers `pattern` (at most 20 photons).
    fn probability(&self, py: Python<'_>, pattern: Vec<usize>) -> PyResult<f64> {
        let s = &self.inner;
        py.detach(|| s.probability(&pattern)).raise()
    }

    /// Probability that exactly the detectors marked in `clicks` click
    /// (threshold detectors; states with zero means).
    fn click_probability(&self, py: Python<'_>, clicks: Vec<bool>) -> PyResult<f64> {
        let s = &self.inner;
        py.detach(|| s.click_probability(&clicks)).raise()
    }

    /// `shots` exact samples of the photon numbers (one row per shot); a
    /// sample with more than `max_photons` photons raises.
    #[pyo3(signature = (shots, max_photons = 20, seed = 0))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        max_photons: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let s = &self.inner;
        let samples = py
            .detach(|| s.sample(shots, max_photons, &mut Rng::seed_from_u64(seed)))
            .raise()?;
        Ok(patterns_array(py, &samples, s.n_modes()))
    }

    /// `shots` exact samples of threshold-detector clicks (0 or 1 per mode).
    #[pyo3(signature = (shots, seed = 0))]
    fn sample_clicks<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let s = &self.inner;
        let samples = py
            .detach(|| s.sample_clicks(shots, &mut Rng::seed_from_u64(seed)))
            .raise()?;
        let counts: Vec<Vec<usize>> = samples
            .iter()
            .map(|row| row.iter().map(|&c| usize::from(c)).collect())
            .collect();
        Ok(patterns_array(py, &counts, s.n_modes()))
    }

    fn __repr__(&self) -> String {
        format!("GaussianState(n_modes={})", self.inner.n_modes())
    }
}

/// A `BosonSampler` with imperfect photons: the Gram matrix of their
/// internal states (`gram`, or `internal_states`, or a common pairwise
/// overlap `indistinguishability` = |⟨ψ_j|ψ_k⟩|²), the transmission of
/// every input mode and the efficiency of every detector.
#[pyclass(name = "ImperfectBosonSampler", module = "metatron_qso", frozen)]
pub struct PyImperfectBosonSampler {
    inner: ImperfectBosonSampler,
}

#[pymethods]
impl PyImperfectBosonSampler {
    #[new]
    #[pyo3(signature = (
        sampler, gram = None, internal_states = None, indistinguishability = None,
        transmission = None, efficiency = None
    ))]
    fn new(
        sampler: &PyBosonSampler,
        gram: Option<ComplexMatrixArg<'_>>,
        internal_states: Option<ComplexMatrixArg<'_>>,
        indistinguishability: Option<f64>,
        transmission: Option<Vec<f64>>,
        efficiency: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let given = [
            gram.is_some(),
            internal_states.is_some(),
            indistinguishability.is_some(),
        ];
        if given.iter().filter(|&&g| g).count() > 1 {
            return Err(invalid(
                "give at most one of gram, internal_states and indistinguishability",
            ));
        }
        let mut s = ImperfectBosonSampler::new(sampler.inner.clone());
        if let Some(g) = gram {
            s = s.with_gram(matrix("gram", &g)?).raise()?;
        }
        if let Some(states) = internal_states {
            let view = states.as_array();
            let rows: Vec<Vec<metatron_qso::C64>> =
                view.rows().into_iter().map(|r| r.to_vec()).collect();
            s = s.with_internal_states(&rows).raise()?;
        }
        if let Some(x) = indistinguishability {
            s = s.with_indistinguishability(x).raise()?;
        }
        if let Some(t) = transmission {
            s = s.with_transmission(&t).raise()?;
        }
        if let Some(e) = efficiency {
            s = s.with_efficiency(&e).raise()?;
        }
        Ok(Self { inner: s })
    }

    /// The Gram matrix of the photons' internal states.
    #[getter]
    fn gram<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<metatron_qso::C64>> {
        cmatrix(py, self.inner.gram())
    }

    /// Probability of detecting `output` (any number of photons).
    fn probability(&self, py: Python<'_>, output: Vec<usize>) -> PyResult<f64> {
        let s = &self.inner;
        py.detach(|| s.probability(&output)).raise()
    }

    /// Exact distribution over all patterns of 0 to n detected photons.
    fn distribution<'py>(&self, py: Python<'py>) -> PyResult<DistributionArrays<'py>> {
        let s = &self.inner;
        let d = py.detach(|| s.distribution()).raise()?;
        Ok((
            patterns_array(py, &d.patterns, s.sampler().n_modes()),
            real_array(py, d.probabilities),
        ))
    }

    /// `shots` exact samples (one row per shot).
    #[pyo3(signature = (shots, seed = 0))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let s = &self.inner;
        let samples = py
            .detach(|| s.sample(shots, &mut Rng::seed_from_u64(seed)))
            .raise()?;
        Ok(patterns_array(py, &samples, s.sampler().n_modes()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ImperfectBosonSampler(n_modes={}, n_photons={})",
            self.inner.sampler().n_modes(),
            self.inner.sampler().n_photons()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBosonSampler>()?;
    m.add_class::<PyGaussianState>()?;
    m.add_class::<PyImperfectBosonSampler>()?;
    m.add_function(wrap_pyfunction!(hafnian, m)?)?;
    m.add_function(wrap_pyfunction!(hafnian_repeated, m)?)?;
    m.add_function(wrap_pyfunction!(torontonian, m)?)?;
    m.add_function(wrap_pyfunction!(permanent, m)?)?;
    m.add_function(wrap_pyfunction!(haar_unitary, m)?)?;
    m.add_function(wrap_pyfunction!(beam_splitter, m)?)?;
    Ok(())
}
