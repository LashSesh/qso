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

/// CPMG: `n ≥ 1` π pulses about y spaced `2τ`.
#[pyfunction(name = "cpmg")]
fn py_cpmg(n: usize, tau: f64) -> PyResult<Vec<PyNvEvent>> {
    if n == 0 {
        return Err(invalid("CPMG needs at least one π pulse"));
    }
    Ok(wrap(cpmg(n, tau)))
}

/// XY-8: `n ≥ 1` blocks of eight π pulses (`x y x y y x y x`) spaced `2τ`.
#[pyfunction(name = "xy8")]
fn py_xy8(n: usize, tau: f64) -> PyResult<Vec<PyNvEvent>> {
    if n == 0 {
        return Err(invalid("XY-8 needs at least one block"));
    }
    Ok(wrap(xy8(n, tau)))
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

/// The filter function `|Y(ω)|²` (s²) of a sequence at each angular
/// frequency `ω` (rad/s), for instantaneous pulses (π/2, free periods and
/// π pulses, π/2; angles modulo 360°).
#[pyfunction(name = "filter_function")]
fn py_filter_function<'py>(
    py: Python<'py>,
    events: Vec<PyRef<'_, PyNvEvent>>,
    omega: RealArg<'_>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ev = self::events(&events);
    let w = real_vec(&omega);
    let values = py
        .detach(|| {
            w.iter()
                .map(|&w| filter_function(&ev, w))
                .collect::<metatron_qso::Result<Vec<f64>>>()
        })
        .raise()?;
    Ok(real_array(py, values))
}

/// The Gaussian decay exponent `χ` of a sequence of instantaneous pulses
/// under `noise`: the coherence decays as `exp(−χ)` for Gaussian noise
/// (for telegraph noise `χ` is only the second-order term).
#[pyfunction(name = "decay_exponent")]
fn py_decay_exponent(
    py: Python<'_>,
    events: Vec<PyRef<'_, PyNvEvent>>,
    noise: &PyFieldNoise,
) -> PyResult<f64> {
    let (ev, noise) = (self::events(&events), noise.inner);
    py.detach(|| decay_exponent(&ev, &noise)).raise()
}

/// `"14N"`, `"15N"`, `None` or `(isotope, A_N, P)` with the hyperfine and
/// quadrupole couplings in Hz.
fn nitrogen(spec: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Nitrogen>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let named = |name: &str| match name {
        "14N" => Ok(Nitrogen::n14()),
        "15N" => Ok(Nitrogen::n15()),
        other => Err(invalid(format!(
            "unknown nitrogen '{other}' (use 14N, 15N, None or (isotope, A_N, P))"
        ))),
    };
    if let Ok(name) = spec.extract::<String>() {
        return named(&name).map(Some);
    }
    let (name, parallel, quadrupole): (String, f64, f64) = spec
        .extract()
        .map_err(|_| invalid("nitrogen is 14N, 15N, None or (isotope, A_N in Hz, P in Hz)"))?;
    let mut n = named(&name)?;
    n.parallel = parallel;
    n.quadrupole = quadrupole;
    Ok(Some(n))
}

fn isotope_name(n: &Nitrogen) -> &'static str {
    match n.isotope {
        NitrogenIsotope::N14 => "14N",
        NitrogenIsotope::N15 => "15N",
    }
}

/// An NV centre in a field `field` (T) along its axis with an optional
/// nitrogen (`"14N"`, `"15N"`, or `(isotope, A_N, P)` in Hz) and ¹³C nuclei
/// `(A_∥, A_⊥)` (Hz), in the secular approximation. Pulses drive the
/// transition `0 ↔ transition` (−1 or +1) at `frequency` (Hz, signed as in
/// `lines`; default: on resonance with the bare line `D + m γ_e B`, which
/// is negative for m = −1 above 102 mT), ideally (`rabi=None`) or with a
/// Rabi frequency (Hz) that drives both transitions with the carrier's
/// phase. The electron starts in `m_s = 0`, the nuclei unpolarised;
/// signals are the `m_s = 0` population.
#[pyclass(name = "NvCenter", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyNvCenter {
    inner: NvCenter,
}

impl PyNvCenter {
    fn drive(
        &self,
        frequency: Option<f64>,
        transition: i64,
        rabi: Option<f64>,
    ) -> PyResult<Microwave> {
        let transition = match transition {
            1 => 1,
            -1 => -1,
            other => return Err(invalid(format!("transition is −1 or +1, got {other}"))),
        };
        let mut mw = Microwave::resonant(&self.inner, transition, rabi);
        if let Some(f) = frequency {
            mw.frequency = f;
        }
        Ok(mw)
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
        nitrogen: Option<&Bound<'_, PyAny>>,
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

    /// The nitrogen as `(isotope, A_N, P)` (Hz), or `None`.
    #[getter]
    fn nitrogen(&self) -> Option<(&'static str, f64, f64)> {
        self.inner
            .nitrogen
            .map(|n| (isotope_name(&n), n.parallel, n.quadrupole))
    }

    /// The ¹³C couplings `(A_∥, A_⊥)` (Hz).
    #[getter]
    fn carbons(&self) -> Vec<(f64, f64)> {
        self.inner
            .carbons
            .iter()
            .map(|c| (c.parallel, c.perpendicular))
            .collect()
    }

    /// The zero-field splitting `D` (Hz).
    #[getter]
    fn zero_field_splitting(&self) -> f64 {
        self.inner.zero_field_splitting
    }

    /// The electron's gyromagnetic ratio (Hz/T).
    #[getter]
    fn gyromagnetic_ratio(&self) -> f64 {
        self.inner.gyromagnetic_ratio
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
    fn initial_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.initial_state().raise()?))
    }

    /// The electron-spin resonance lines `(frequency Hz, strength)` from
    /// `m_s = 0` to `m_s = ±1`, sorted: frequencies signed as the level
    /// differences, strengths averaged over the nuclei (½ per transition),
    /// lines within 1 mHz merged.
    fn lines(&self, py: Python<'_>) -> PyResult<Vec<(f64, f64)>> {
        let nv = &self.inner;
        py.detach(|| nv.lines()).raise()
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

    /// Apply `events` to `rho`: `U ρ U†` (any square matrix of the centre's
    /// dimension).
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
        transition: i64,
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
        let mw = self.drive(frequency, transition, rabi)?;
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
        transition: i64,
        rabi: Option<f64>,
    ) -> PyResult<f64> {
        let ev = self::events(&events);
        let mw = self.drive(frequency, transition, rabi)?;
        let nv = &self.inner;
        py.detach(|| nv.signal(&ev, &mw)).raise()
    }

    /// The `m_s = 0` population averaged over `trajectories` paths of
    /// `noise` (seeded): `(mean, standard error)`, NaN for one trajectory.
    /// The noise is exact between pulses and held at its mean over pieces
    /// of finite pulses.
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
        transition: i64,
        rabi: Option<f64>,
    ) -> PyResult<(f64, f64)> {
        let ev = self::events(&events);
        let mw = self.drive(frequency, transition, rabi)?;
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
