//! Quantum sensing with NV centres.

use metatron_qso::C64;
use metatron_qso::linalg::CMatrix;
use metatron_qso::spin::{
    FieldNoise, MAX_NV_DIMENSION, Microwave, NV_GYROMAGNETIC_RATIO, NV_ZERO_FIELD_SPLITTING,
    Nitrogen, NitrogenIsotope, NvCenter, NvEvent, cpmg, decay_exponent, filter_function, hahn,
    ramsey, xy8,
};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::{
    ComplexMatrixArg, Raise, RealArg, cmatrix, invalid, real_array, real_vec, square_complex,
};

/// One step of an NV sensing sequence (made by `nv_pulse`, `nv_free`,
/// `ramsey`, `hahn`, `cpmg` and `xy8`).
#[pyclass(name = "NvEvent", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyNvEvent {
    inner: NvEvent,
}

#[pymethods]
impl PyNvEvent {
    /// `"pulse"` or `"free"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            NvEvent::Pulse { .. } => "pulse",
            NvEvent::Free(_) => "free",
        }
    }

    /// The pulse angle (degrees), `None` for free evolution.
    #[getter]
    fn angle(&self) -> Option<f64> {
        match self.inner {
            NvEvent::Pulse { angle, .. } => Some(angle),
            NvEvent::Free(_) => None,
        }
    }

    /// The pulse phase (degrees from x), `None` for free evolution.
    #[getter]
    fn phase(&self) -> Option<f64> {
        match self.inner {
            NvEvent::Pulse { phase, .. } => Some(phase),
            NvEvent::Free(_) => None,
        }
    }

    /// The free time (s), `None` for a pulse.
    #[getter]
    fn duration(&self) -> Option<f64> {
        match self.inner {
            NvEvent::Free(t) => Some(t),
            NvEvent::Pulse { .. } => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            NvEvent::Pulse { angle, phase } => format!("NvEvent(pulse {angle}° at {phase}°)"),
            NvEvent::Free(t) => format!("NvEvent(free {t} s)"),
        }
    }
}

fn events(list: &[PyRef<'_, PyNvEvent>]) -> Vec<NvEvent> {
    list.iter().map(|e| e.inner).collect()
}

fn wrap(events: Vec<NvEvent>) -> Vec<PyNvEvent> {
    events
        .into_iter()
        .map(|inner| PyNvEvent { inner })
        .collect()
}

/// A pulse of `angle` degrees about the axis at `phase` degrees from x on
/// the driven transition.
#[pyfunction]
#[pyo3(signature = (angle, phase = 0.0))]
fn nv_pulse(angle: f64, phase: f64) -> PyNvEvent {
    PyNvEvent {
        inner: NvEvent::Pulse { angle, phase },
    }
}

/// Free evolution for `t` seconds (between pulse centres for finite
/// pulses).
#[pyfunction]
fn nv_free(t: f64) -> PyNvEvent {
    PyNvEvent {
        inner: NvEvent::Free(t),
    }
}

/// Ramsey: `π/2_x − τ − π/2_{−x}`.
#[pyfunction(name = "ramsey")]
fn py_ramsey(tau: f64) -> Vec<PyNvEvent> {
    wrap(ramsey(tau))
}

/// Hahn echo: `π/2_x − τ − π_y − τ − π/2_{−x}`.
#[pyfunction(name = "hahn")]
fn py_hahn(tau: f64) -> Vec<PyNvEvent> {
    wrap(hahn(tau))
}

/// CPMG: `n` π pulses about y spaced `2τ`.
#[pyfunction(name = "cpmg")]
fn py_cpmg(n: usize, tau: f64) -> Vec<PyNvEvent> {
    wrap(cpmg(n, tau))
}

/// XY-8: `n` blocks of eight π pulses (`x y x y y x y x`) spaced `2τ`.
#[pyfunction(name = "xy8")]
fn py_xy8(n: usize, tau: f64) -> Vec<PyNvEvent> {
    wrap(xy8(n, tau))
}

/// Classical field noise `b(t)` (Hz) along the NV axis: telegraph,
/// Ornstein–Uhlenbeck or 1/f (`FieldNoise.telegraph`, `.ornstein_uhlenbeck`,
/// `.pink`).
#[pyclass(name = "FieldNoise", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyFieldNoise {
    inner: FieldNoise,
}

impl PyFieldNoise {
    fn checked(inner: FieldNoise) -> PyResult<Self> {
        inner.validate().raise()?;
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyFieldNoise {
    /// `±amplitude` (Hz) flipping at `rate` per second (Poisson).
    #[staticmethod]
    fn telegraph(amplitude: f64, rate: f64) -> PyResult<Self> {
        Self::checked(FieldNoise::Telegraph { amplitude, rate })
    }

    /// Gaussian with standard deviation `sigma` (Hz) and correlation time
    /// `correlation_time` (s).
    #[staticmethod]
    fn ornstein_uhlenbeck(sigma: f64, correlation_time: f64) -> PyResult<Self> {
        Self::checked(FieldNoise::OrnsteinUhlenbeck {
            sigma,
            correlation_time,
        })
    }

    /// 1/f between `lowest` and `highest` (Hz): `fluctuators`
    /// Ornstein–Uhlenbeck processes of total variance `sigma²`, corner
    /// frequencies evenly spaced in their logarithm.
    #[staticmethod]
    #[pyo3(signature = (sigma, lowest, highest, fluctuators = 40))]
    fn pink(sigma: f64, lowest: f64, highest: f64, fluctuators: usize) -> PyResult<Self> {
        Self::checked(FieldNoise::Pink {
            sigma,
            lowest,
            highest,
            fluctuators,
        })
    }

    /// Whether the noise is Gaussian (the decay `exp(−χ)` is then exact).
    #[getter]
    fn is_gaussian(&self) -> bool {
        self.inner.is_gaussian()
    }

    /// `⟨b(t) b(t + τ)⟩` (Hz²) at each `τ` (s).
    fn correlation<'py>(&self, py: Python<'py>, tau: RealArg<'_>) -> Bound<'py, PyArray1<f64>> {
        real_array(
            py,
            real_vec(&tau)
                .iter()
                .map(|&t| self.inner.correlation(t))
                .collect(),
        )
    }

    /// The two-sided spectrum `S(ω)` (Hz²·s) at each angular frequency `ω`
    /// (rad/s).
    fn spectrum<'py>(&self, py: Python<'py>, omega: RealArg<'_>) -> Bound<'py, PyArray1<f64>> {
        real_array(
            py,
            real_vec(&omega)
                .iter()
                .map(|&w| self.inner.spectrum(w))
                .collect(),
        )
    }

    fn __repr__(&self) -> String {
        format!("FieldNoise({:?})", self.inner)
    }
}

/// The filter function `|Y(ω)|²` (s²) of a sequence of ideal pulses at
/// each angular frequency `ω` (rad/s).
#[pyfunction(name = "filter_function")]
fn py_filter_function<'py>(
    py: Python<'py>,
    events: Vec<PyRef<'_, PyNvEvent>>,
    omega: RealArg<'_>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ev = self::events(&events);
    let values = real_vec(&omega)
        .iter()
        .map(|&w| filter_function(&ev, w))
        .collect::<metatron_qso::Result<Vec<f64>>>()
        .raise()?;
    Ok(real_array(py, values))
}

/// The Gaussian decay exponent `χ` of a sequence of ideal pulses under
/// `noise`: the coherence decays as `exp(−χ)` for Gaussian noise.
#[pyfunction(name = "decay_exponent")]
fn py_decay_exponent(events: Vec<PyRef<'_, PyNvEvent>>, noise: &PyFieldNoise) -> PyResult<f64> {
    decay_exponent(&self::events(&events), &noise.inner).raise()
}

fn nitrogen(name: Option<&str>) -> PyResult<Option<Nitrogen>> {
    match name {
        None => Ok(None),
        Some("14N") => Ok(Some(Nitrogen::n14())),
        Some("15N") => Ok(Some(Nitrogen::n15())),
        Some(other) => Err(invalid(format!(
            "unknown nitrogen '{other}' (use 14N, 15N or None)"
        ))),
    }
}

/// An NV centre in a field `field` (T) along its axis with an optional
/// nitrogen (`"14N"`, `"15N"`) and ¹³C nuclei `(A_∥, A_⊥)` (Hz), in the
/// secular approximation. Pulses drive the transition `0 ↔ transition`
/// (−1 or +1) at `frequency` (Hz; default: on resonance with the bare
/// line), ideally (`rabi=None`) or with a Rabi frequency (Hz) that drives
/// both transitions. The electron starts in `m_s = 0`, the nuclei
/// unpolarised; signals are the `m_s = 0` population.
#[pyclass(name = "NvCenter", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyNvCenter {
    inner: NvCenter,
}

impl PyNvCenter {
    fn drive(&self, frequency: Option<f64>, transition: i8, rabi: Option<f64>) -> Microwave {
        let mut mw = Microwave::resonant(&self.inner, transition, rabi);
        if let Some(f) = frequency {
            mw.frequency = f;
        }
        mw
    }
}

#[pymethods]
impl PyNvCenter {
    #[new]
    #[pyo3(signature = (
        field,
        nitrogen = None,
        carbons = Vec::new(),
        zero_field_splitting = NV_ZERO_FIELD_SPLITTING,
        gyromagnetic_ratio = NV_GYROMAGNETIC_RATIO,
    ))]
    fn new(
        field: f64,
        nitrogen: Option<&str>,
        carbons: Vec<(f64, f64)>,
        zero_field_splitting: f64,
        gyromagnetic_ratio: f64,
    ) -> PyResult<Self> {
        let mut inner = NvCenter::new(field);
        inner.zero_field_splitting = zero_field_splitting;
        inner.gyromagnetic_ratio = gyromagnetic_ratio;
        inner.nitrogen = self::nitrogen(nitrogen)?;
        for (parallel, perpendicular) in carbons {
            inner = inner.with_carbon(parallel, perpendicular);
        }
        inner.validate().raise()?;
        Ok(Self { inner })
    }

    /// The field (T).
    #[getter]
    fn field(&self) -> f64 {
        self.inner.field
    }

    /// Hilbert-space dimension (electron × nitrogen × carbons).
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The Hamiltonian (rad/s) in the frame rotating at `frequency` (Hz).
    #[pyo3(signature = (frequency = 0.0))]
    fn hamiltonian<'py>(
        &self,
        py: Python<'py>,
        frequency: f64,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.hamiltonian(frequency).raise()?))
    }

    /// The optically pumped state: `m_s = 0`, nuclei unpolarised.
    fn initial_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        cmatrix(py, &self.inner.initial_state())
    }

    /// The electron-spin resonance lines `(frequency Hz, strength)`, sorted.
    fn lines(&self) -> PyResult<Vec<(f64, f64)>> {
        self.inner.lines().raise()
    }

    /// Pulsed ODMR: the `m_s = 0` population after a pulse of Rabi
    /// frequency `rabi` (Hz) and `duration` (s) at each frequency (Hz).
    fn odmr<'py>(
        &self,
        py: Python<'py>,
        frequencies: RealArg<'_>,
        rabi: f64,
        duration: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = real_vec(&frequencies);
        let nv = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| nv.odmr(&f, rabi, duration)).raise()?,
        ))
    }

    /// Apply `events` to the density matrix `rho`.
    #[pyo3(
        signature = (rho, events, frequency = None, transition = -1, rabi = None),
        text_signature = "(self, /, rho, events, frequency=None, transition=-1, rabi=None)"
    )]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
        events: Vec<PyRef<'_, PyNvEvent>>,
        frequency: Option<f64>,
        transition: i8,
        rabi: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let (d, data) = square_complex("rho", &rho)?;
        if d != self.inner.dimension() {
            return Err(invalid(format!(
                "a state of this centre is {0}×{0}, got {d}×{d}",
                self.inner.dimension()
            )));
        }
        let rho = CMatrix::from_row_slice(d, d, &data);
        let ev = self::events(&events);
        let mw = self.drive(frequency, transition, rabi);
        let nv = &self.inner;
        Ok(cmatrix(
            py,
            &py.detach(|| nv.evolve(&rho, &ev, &mw)).raise()?,
        ))
    }

    /// The `m_s = 0` population after `events` from the pumped state.
    #[pyo3(
        signature = (events, frequency = None, transition = -1, rabi = None),
        text_signature = "(self, /, events, frequency=None, transition=-1, rabi=None)"
    )]
    fn signal(
        &self,
        py: Python<'_>,
        events: Vec<PyRef<'_, PyNvEvent>>,
        frequency: Option<f64>,
        transition: i8,
        rabi: Option<f64>,
    ) -> PyResult<f64> {
        let ev = self::events(&events);
        let mw = self.drive(frequency, transition, rabi);
        let nv = &self.inner;
        py.detach(|| nv.signal(&ev, &mw)).raise()
    }

    /// The `m_s = 0` population averaged over `trajectories` exact paths
    /// of `noise` (seeded): `(mean, standard error)`.
    #[pyo3(
        signature = (
            events, noise, trajectories, seed = 0, frequency = None, transition = -1, rabi = None,
        ),
        text_signature = "(self, /, events, noise, trajectories, seed=0, frequency=None, transition=-1, rabi=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn signal_with_noise(
        &self,
        py: Python<'_>,
        events: Vec<PyRef<'_, PyNvEvent>>,
        noise: &PyFieldNoise,
        trajectories: usize,
        seed: u64,
        frequency: Option<f64>,
        transition: i8,
        rabi: Option<f64>,
    ) -> PyResult<(f64, f64)> {
        let ev = self::events(&events);
        let mw = self.drive(frequency, transition, rabi);
        let (nv, noise) = (&self.inner, noise.inner);
        py.detach(|| nv.signal_with_noise(&ev, &mw, &noise, trajectories, seed))
            .raise()
    }

    /// The centre as JSON.
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// A centre from JSON (checked).
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: NvCenter::from_json(text).raise()?,
        })
    }

    fn __repr__(&self) -> String {
        let nitrogen = match self.inner.nitrogen.map(|n| n.isotope) {
            Some(NitrogenIsotope::N14) => "14N",
            Some(NitrogenIsotope::N15) => "15N",
            None => "no nitrogen",
        };
        format!(
            "NvCenter({} T, {nitrogen}, {} carbons)",
            self.inner.field,
            self.inner.carbons.len()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNvCenter>()?;
    m.add_class::<PyNvEvent>()?;
    m.add_class::<PyFieldNoise>()?;
    m.add_function(wrap_pyfunction!(nv_pulse, m)?)?;
    m.add_function(wrap_pyfunction!(nv_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_ramsey, m)?)?;
    m.add_function(wrap_pyfunction!(py_hahn, m)?)?;
    m.add_function(wrap_pyfunction!(py_cpmg, m)?)?;
    m.add_function(wrap_pyfunction!(py_xy8, m)?)?;
    m.add_function(wrap_pyfunction!(py_filter_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_decay_exponent, m)?)?;
    m.add("NV_ZERO_FIELD_SPLITTING", NV_ZERO_FIELD_SPLITTING)?;
    m.add("NV_GYROMAGNETIC_RATIO", NV_GYROMAGNETIC_RATIO)?;
    m.add("MAX_NV_DIMENSION", MAX_NV_DIMENSION)?;
    Ok(())
}
