//! Linear optics: permanents, interferometers and boson sampling.

use metatron_qso::Rng;
use metatron_qso::linalg::CMatrix;
use metatron_qso::photonics::{self, BosonSampler};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::{ComplexMatrixArg, Raise, check_finite, cmatrix, invalid, real_array};

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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBosonSampler>()?;
    m.add_function(wrap_pyfunction!(permanent, m)?)?;
    m.add_function(wrap_pyfunction!(haar_unitary, m)?)?;
    m.add_function(wrap_pyfunction!(beam_splitter, m)?)?;
    Ok(())
}
