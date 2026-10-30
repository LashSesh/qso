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
use pyo3::types::PyDict;

use metatron_qso::spin::bath::{
    self, BathSpin, Cce, DIAMOND_LATTICE_CONSTANT, MAX_BATH_SPINS, MAX_CLUSTER_ORDER,
    MAX_EXACT_BATH, NATURAL_C13_ABUNDANCE, SpinBath,
};
use metatron_qso::spin::optics::{
    self, CwOdmr, NV_EXCITED_SPLITTING, OpticalNv, Photophysics, PulsedReadout, TwoLevelOdmr,
};

use crate::{
    ComplexMatrixArg, Raise, RealArg, cmatrix, complex_array, invalid, real_array, real_vec,
    square_complex,
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

fn sequences(list: &[Vec<PyRef<'_, PyNvEvent>>]) -> Vec<Vec<NvEvent>> {
    list.iter().map(|s| events(s)).collect()
}

fn transition_of(transition: i64) -> PyResult<i8> {
    match transition {
        1 => Ok(1),
        -1 => Ok(-1),
        other => Err(invalid(format!("transition is −1 or +1, got {other}"))),
    }
}

type SpinTuple = ([f64; 3], [f64; 3]);

/// Energies and eigenvectors.
type Levels<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<C64>>);

/// Means and standard errors, one per sequence.
type MeansAndErrors<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// The ¹³C bath of an NV centre in a field `field` (T) along its axis,
/// seen by the qubit `{0, transition}`: `spins` are `(position,
/// hyperfine)` pairs, the position from the vacancy (m, z along the NV
/// axis) and the secular hyperfine couplings `(A_zx, A_zy, A_zz)` (Hz).
/// `SpinBath.diamond` places ¹³C at random on the lattice. Its coherence
/// after a sequence (a pulse, free periods and π pulses, a pulse, with
/// ideal pulses) comes from the cluster-correlation expansion: clusters
/// of up to `order` spins connected by pairs closer than `cutoff` (m),
/// whose dipolar couplings are kept, each propagated exactly.
#[pyclass(name = "SpinBath", module = "metatron_qso", frozen)]
pub struct PySpinBath {
    inner: SpinBath,
}

#[pymethods]
impl PySpinBath {
    #[new]
    #[pyo3(
        signature = (field, spins, transition = -1),
        text_signature = "(field, spins, transition=-1)"
    )]
    fn new(py: Python<'_>, field: f64, spins: Vec<SpinTuple>, transition: i64) -> PyResult<Self> {
        let spins = spins
            .into_iter()
            .map(|(position, hyperfine)| BathSpin {
                position,
                hyperfine,
            })
            .collect();
        let inner = SpinBath::new(field, transition_of(transition)?, spins);
        py.detach(|| inner.validate()).raise()?;
        Ok(Self { inner })
    }

    /// A random bath: every carbon site within `radius` (m) of the vacancy
    /// holds a ¹³C with probability `abundance` (seeded; each site's draw
    /// depends only on the seed, so baths nest in the radius), with
    /// point-dipole hyperfine couplings; none closer than `min_radius`
    /// (m), where contact terms dominate.
    #[staticmethod]
    #[pyo3(
        signature = (
            field, abundance = NATURAL_C13_ABUNDANCE, radius = 4e-9, seed = 0, transition = -1,
            min_radius = 0.0,
        ),
        text_signature = "(field, abundance=0.0107, radius=4e-09, seed=0, transition=-1, min_radius=0.0)"
    )]
    fn diamond(
        py: Python<'_>,
        field: f64,
        abundance: f64,
        radius: f64,
        seed: u64,
        transition: i64,
        min_radius: f64,
    ) -> PyResult<Self> {
        let transition = transition_of(transition)?;
        let spins = py
            .detach(|| bath::diamond_bath(abundance, radius, seed))
            .raise()?;
        let inner = SpinBath::new(field, transition, spins).without_spins_within(min_radius);
        if !inner.field.is_finite() {
            return Err(invalid("the field must be finite"));
        }
        Ok(Self { inner })
    }

    #[getter]
    fn field(&self) -> f64 {
        self.inner.field
    }

    #[getter]
    fn transition(&self) -> i8 {
        self.inner.transition
    }

    /// The bath without the spins closer than `radius` (m) to the vacancy.
    fn without_spins_within(&self, radius: f64) -> Self {
        Self {
            inner: self.inner.clone().without_spins_within(radius),
        }
    }

    #[getter]
    fn num_spins(&self) -> usize {
        self.inner.spins.len()
    }

    /// The spins as `(position, hyperfine)` pairs.
    #[getter]
    fn spins(&self) -> Vec<SpinTuple> {
        self.inner
            .spins
            .iter()
            .map(|s| (s.position, s.hyperfine))
            .collect()
    }

    /// The connected clusters of up to `order` spins.
    #[pyo3(signature = (order = 2, cutoff = 8e-10))]
    fn clusters(&self, py: Python<'_>, order: usize, cutoff: f64) -> PyResult<Vec<Vec<usize>>> {
        let b = &self.inner;
        py.detach(|| b.clusters(order, cutoff)).raise()
    }

    /// The coherence `L = Tr(U_m ρ U_0†)` of the bath after each sequence.
    #[pyo3(signature = (sequences, order = 2, cutoff = 8e-10))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<Vec<PyRef<'_, PyNvEvent>>>,
        order: usize,
        cutoff: f64,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let (seqs, b) = (self::sequences(&sequences), &self.inner);
        let l = py
            .detach(|| b.coherence(&seqs, &Cce { order, cutoff }))
            .raise()?;
        Ok(complex_array(py, l))
    }

    /// The coherence of the whole bath propagated exactly (at most
    /// `MAX_EXACT_BATH` spins), with the couplings of pairs closer than
    /// `cutoff`.
    #[pyo3(signature = (sequences, cutoff = f64::INFINITY))]
    fn exact_coherence<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<Vec<PyRef<'_, PyNvEvent>>>,
        cutoff: f64,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let (seqs, b) = (self::sequences(&sequences), &self.inner);
        let l = py.detach(|| b.exact_coherence(&seqs, cutoff)).raise()?;
        Ok(complex_array(py, l))
    }

    /// The `m_s = 0` population after each sequence.
    #[pyo3(signature = (sequences, order = 2, cutoff = 8e-10))]
    fn signal<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<Vec<PyRef<'_, PyNvEvent>>>,
        order: usize,
        cutoff: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (seqs, b) = (self::sequences(&sequences), &self.inner);
        let s = py
            .detach(|| b.signal(&seqs, &Cce { order, cutoff }))
            .raise()?;
        Ok(real_array(py, s))
    }

    /// The population with classical field noise as well: `(means,
    /// standard errors)`; exact for Gaussian noise (errors 0, `trajectories`
    /// unused), averaged over `trajectories` paths (seeded) for telegraph
    /// noise (errors NaN for one path).
    #[pyo3(signature = (sequences, noise, trajectories = 1000, seed = 0, order = 2, cutoff = 8e-10))]
    #[allow(clippy::too_many_arguments)]
    fn signal_with_noise<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<Vec<PyRef<'_, PyNvEvent>>>,
        noise: &PyFieldNoise,
        trajectories: usize,
        seed: u64,
        order: usize,
        cutoff: f64,
    ) -> PyResult<MeansAndErrors<'py>> {
        let (seqs, b, noise) = (self::sequences(&sequences), &self.inner, noise.inner);
        let r = py
            .detach(|| {
                b.signal_with_noise(&seqs, &Cce { order, cutoff }, &noise, trajectories, seed)
            })
            .raise()?;
        let (mean, err): (Vec<f64>, Vec<f64>) = r.into_iter().unzip();
        Ok((real_array(py, mean), real_array(py, err)))
    }

    fn __repr__(&self) -> String {
        format!(
            "SpinBath({} T, m = {}, {} spins)",
            self.inner.field,
            self.inner.transition,
            self.inner.spins.len()
        )
    }
}

/// The point-dipole hyperfine couplings `(A_zx, A_zy, A_zz)` (Hz) of a ¹³C
/// at `position` (m from the vacancy, z along the NV axis).
#[pyfunction(name = "point_dipole_hyperfine")]
fn py_point_dipole(position: [f64; 3]) -> PyResult<[f64; 3]> {
    Ok(BathSpin::point_dipole(position).raise()?.hyperfine)
}

/// The carbon sites of diamond within `radius` (m) of the vacancy, in the
/// frame of the centre (`k × 3`).
#[pyfunction(name = "diamond_sites")]
fn py_diamond_sites<'py>(py: Python<'py>, radius: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sites = py.detach(|| bath::diamond_sites(radius)).raise()?;
    let m = metatron_qso::linalg::RMatrix::from_fn(sites.len(), 3, |r, c| sites[r][c]);
    Ok(crate::rmatrix(py, &m))
}

/// The population after each sequence averaged over `configurations`
/// random baths (seeded): `(means, standard errors)`.
#[pyfunction(name = "bath_ensemble_signal")]
#[pyo3(
    signature = (
        field, sequences, configurations, abundance = NATURAL_C13_ABUNDANCE, radius = 4e-9,
        seed = 0, transition = -1, order = 2, cutoff = 8e-10,
    ),
    text_signature = "(field, sequences, configurations, abundance=0.0107, radius=4e-09, seed=0, transition=-1, order=2, cutoff=8e-10)"
)]
#[allow(clippy::too_many_arguments)]
fn py_ensemble_signal<'py>(
    py: Python<'py>,
    field: f64,
    sequences: Vec<Vec<PyRef<'_, PyNvEvent>>>,
    configurations: usize,
    abundance: f64,
    radius: f64,
    seed: u64,
    transition: i64,
    order: usize,
    cutoff: f64,
) -> PyResult<MeansAndErrors<'py>> {
    let (seqs, transition) = (self::sequences(&sequences), transition_of(transition)?);
    let r = py
        .detach(|| {
            bath::ensemble_signal(
                field,
                transition,
                abundance,
                radius,
                configurations,
                seed,
                &seqs,
                &Cce { order, cutoff },
            )
        })
        .raise()?;
    let (mean, err): (Vec<f64>, Vec<f64>) = r.into_iter().unzip();
    Ok((real_array(py, mean), real_array(py, err)))
}

fn cw_dict<'py>(py: Python<'py>, line: CwOdmr) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frequencies", real_array(py, line.frequencies))?;
    d.set_item("rates", real_array(py, line.rates))?;
    d.set_item("off_resonance", line.off_resonance)?;
    d.set_item("contrast", line.contrast)?;
    d.set_item("linewidth", line.linewidth)?;
    d.set_item("slope", line.slope)?;
    d.set_item("sensitivity", line.sensitivity)?;
    Ok(d)
}

fn readout_dict<'py>(py: Python<'py>, r: PulsedReadout) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("readout_contrast", r.readout_contrast())?;
    d.set_item("times", real_array(py, r.times))?;
    d.set_item("bright", real_array(py, r.bright))?;
    d.set_item("dark", real_array(py, r.dark))?;
    d.set_item("window", r.window)?;
    d.set_item("photons_bright", r.photons_bright)?;
    d.set_item("photons_dark", r.photons_dark)?;
    Ok(d)
}

/// The electron spin of an NV centre in its optical cycle: the
/// seven-level model of Tetienne et al. (ground and excited triplets, a
/// metastable singlet) in a field `field` (T, frame of the centre: z
/// along the axis) of any direction, as a Lindblad equation with the
/// rates `(k_r, k_47, k_57, k_71, k_72)` (1/s; default: Tetienne et al.),
/// ground-state dephasing `Γ₂*` and spin-lattice rate `Γ₁` (1/s), and the
/// detected fraction `collection` of the emitted photons. `pump` is the
/// optical excitation rate of each spin-conserving transition (1/s); a
/// microwave of `frequency` (Hz, signed as `lines`) and Rabi frequency
/// `rabi` (Hz) drives the ground state.
#[pyclass(name = "OpticalNv", module = "metatron_qso", frozen)]
pub struct PyOpticalNv {
    inner: OpticalNv,
}

#[pymethods]
impl PyOpticalNv {
    #[new]
    #[pyo3(signature = (
        field = [0.0, 0.0, 0.0], rates = None, dephasing = 2e5, spin_lattice = 1e3,
        collection = 0.01, zero_field_splitting = NV_ZERO_FIELD_SPLITTING,
        excited_splitting = NV_EXCITED_SPLITTING, gyromagnetic_ratio = NV_GYROMAGNETIC_RATIO,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        field: [f64; 3],
        rates: Option<[f64; 5]>,
        dephasing: f64,
        spin_lattice: f64,
        collection: f64,
        zero_field_splitting: f64,
        excited_splitting: f64,
        gyromagnetic_ratio: f64,
    ) -> PyResult<Self> {
        let mut inner = OpticalNv::new(field);
        if let Some(
            [
                radiative,
                isc_zero,
                isc_plus_minus,
                singlet_zero,
                singlet_plus_minus,
            ],
        ) = rates
        {
            inner.rates = Photophysics {
                radiative,
                isc_zero,
                isc_plus_minus,
                singlet_zero,
                singlet_plus_minus,
            };
        }
        inner.dephasing = dephasing;
        inner.spin_lattice = spin_lattice;
        inner.collection = collection;
        inner.zero_field_splitting = zero_field_splitting;
        inner.excited_splitting = excited_splitting;
        inner.gyromagnetic_ratio = gyromagnetic_ratio;
        inner.validate().raise()?;
        Ok(Self { inner })
    }

    #[getter]
    fn field(&self) -> [f64; 3] {
        self.inner.field
    }

    /// `(k_r, k_47, k_57, k_71, k_72)` (1/s).
    #[getter]
    fn rates(&self) -> [f64; 5] {
        let r = self.inner.rates;
        [
            r.radiative,
            r.isc_zero,
            r.isc_plus_minus,
            r.singlet_zero,
            r.singlet_plus_minus,
        ]
    }

    #[getter]
    fn collection(&self) -> f64 {
        self.inner.collection
    }

    /// `H/h` of the ground triplet (Hz), basis `m_s = +1, 0, −1`.
    fn ground_hamiltonian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        cmatrix(py, &self.inner.ground_hamiltonian())
    }

    /// `H/h` of the excited triplet (Hz).
    fn excited_hamiltonian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        cmatrix(py, &self.inner.excited_hamiltonian())
    }

    /// Energies (Hz, ascending) and eigenvectors (columns) of the ground
    /// triplet.
    fn ground_levels<'py>(&self, py: Python<'py>) -> PyResult<Levels<'py>> {
        let (e, v) = self.inner.ground_levels().raise()?;
        Ok((real_array(py, e), cmatrix(py, &v)))
    }

    /// Energies and eigenvectors of the excited triplet.
    fn excited_levels<'py>(&self, py: Python<'py>) -> PyResult<Levels<'py>> {
        let (e, v) = self.inner.excited_levels().raise()?;
        Ok((real_array(py, e), cmatrix(py, &v)))
    }

    /// The two ESR lines `(frequency, strength)`, from the level closest
    /// to `m_s = 0` to those closest to `−1` and `+1` (signed).
    fn lines(&self) -> PyResult<Vec<(f64, f64)>> {
        Ok(self.inner.lines().raise()?.to_vec())
    }

    /// The steady state (7 × 7: ground levels, excited levels, singlet).
    #[pyo3(signature = (pump, frequency = None, rabi = 0.0))]
    fn steady_state<'py>(
        &self,
        py: Python<'py>,
        pump: f64,
        frequency: Option<f64>,
        rabi: f64,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let rho = self
            .inner
            .steady_state(pump, frequency.map(|f| (f, rabi)))
            .raise()?;
        Ok(cmatrix(py, &rho))
    }

    /// The detected photon rate (1/s) in the steady state.
    #[pyo3(signature = (pump, frequency = None, rabi = 0.0))]
    fn fluorescence(&self, pump: f64, frequency: Option<f64>, rabi: f64) -> PyResult<f64> {
        let rho = self
            .inner
            .steady_state(pump, frequency.map(|f| (f, rabi)))
            .raise()?;
        Ok(self.inner.fluorescence(&rho))
    }

    /// The photon rate at each microwave frequency (CW-ODMR).
    fn cw_odmr<'py>(
        &self,
        py: Python<'py>,
        frequencies: RealArg<'_>,
        rabi: f64,
        pump: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (f, nv) = (real_vec(&frequencies), &self.inner);
        Ok(real_array(
            py,
            py.detach(|| nv.cw_odmr(&f, rabi, pump)).raise()?,
        ))
    }

    /// A CW-ODMR line around `centre` over `span` (Hz) with its contrast,
    /// full width, steepest slope and shot-noise sensitivity (T/√Hz).
    #[pyo3(signature = (centre, span, rabi, pump, points = 401))]
    fn cw_line<'py>(
        &self,
        py: Python<'py>,
        centre: f64,
        span: f64,
        rabi: f64,
        pump: f64,
        points: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let nv = &self.inner;
        let line = py
            .detach(|| nv.cw_line(centre, span, points, rabi, pump))
            .raise()?;
        cw_dict(py, line)
    }

    /// The fluorescence after the laser is switched on, from the
    /// laser-polarised state (`bright`) and after an ideal π pulse on the
    /// transition (`dark`), with the best counting window.
    #[pyo3(
        signature = (pump, duration, steps = 1000, transition = -1),
        text_signature = "(self, /, pump, duration, steps=1000, transition=-1)"
    )]
    fn pulsed_readout<'py>(
        &self,
        py: Python<'py>,
        pump: f64,
        duration: f64,
        steps: usize,
        transition: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (nv, t) = (&self.inner, transition_of(transition)?);
        let r = py
            .detach(|| nv.pulsed_readout(pump, duration, steps, t))
            .raise()?;
        readout_dict(py, r)
    }

    /// The population of the level closest to `m_s = 0` after the laser
    /// (`duration` s from the unpolarised state, `inf` for the steady
    /// state) once the excited levels and the singlet have decayed.
    #[pyo3(signature = (pump, duration = f64::INFINITY))]
    fn polarisation(&self, pump: f64, duration: f64) -> PyResult<f64> {
        self.inner.polarisation(pump, duration).raise()
    }

    fn __repr__(&self) -> String {
        format!("OpticalNv(field={:?} T)", self.inner.field)
    }
}

/// The reduced model of Dréau et al. (2011): the ground qubit {0, 1} with
/// optical polarisation `polarisation` (Γ_p), optically induced coherence
/// decay `coherence` (Γ_c), spin-lattice rate `spin_lattice` (Γ₁),
/// dephasing (Γ₂*) and photon rates `bright` (α) and `dark` (β); rates in
/// 1/s, detunings and Rabi frequencies in rad/s.
#[pyclass(name = "TwoLevelOdmr", module = "metatron_qso", frozen)]
pub struct PyTwoLevelOdmr {
    inner: TwoLevelOdmr,
}

#[pymethods]
impl PyTwoLevelOdmr {
    #[new]
    fn new(
        polarisation: f64,
        coherence: f64,
        spin_lattice: f64,
        dephasing: f64,
        bright: f64,
        dark: f64,
    ) -> Self {
        Self {
            inner: TwoLevelOdmr {
                polarisation,
                coherence,
                spin_lattice,
                dephasing,
                bright,
                dark,
            },
        }
    }

    /// The steady-state population of |1⟩ at detuning `ω₀ − ω_m` (rad/s).
    fn population(&self, detuning: f64, rabi: f64) -> PyResult<f64> {
        self.inner.population(detuning, rabi).raise()
    }

    /// The photon rate at a detuning.
    fn rate(&self, detuning: f64, rabi: f64) -> PyResult<f64> {
        self.inner.rate(detuning, rabi).raise()
    }

    /// The line over `span` (rad/s) with its contrast, width (Hz) and
    /// sensitivity.
    #[pyo3(signature = (span, rabi, points = 2001))]
    fn line<'py>(
        &self,
        py: Python<'py>,
        span: f64,
        rabi: f64,
        points: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        cw_dict(py, self.inner.line(span, points, rabi).raise()?)
    }
}

/// `√(τ + overhead)/(2π γ_e τ C |L|)` (T/√Hz): Ramsey magnetometry of a
/// static field with readout contrast `C`.
#[pyfunction(name = "ramsey_sensitivity")]
#[pyo3(signature = (tau, coherence, readout, overhead = 0.0))]
fn py_ramsey_sensitivity(tau: f64, coherence: f64, readout: f64, overhead: f64) -> PyResult<f64> {
    optics::ramsey_sensitivity(tau, coherence, readout, overhead).raise()
}

/// The Hahn echo of total duration `total` for an AC field in its best
/// phase: π/2 times the Ramsey expression.
#[pyfunction(name = "echo_sensitivity")]
#[pyo3(signature = (total, coherence, readout, overhead = 0.0))]
fn py_echo_sensitivity(total: f64, coherence: f64, readout: f64, overhead: f64) -> PyResult<f64> {
    optics::echo_sensitivity(total, coherence, readout, overhead).raise()
}

/// The best sensitivity over times and coherences: `(time, T/√Hz)`.
#[pyfunction(name = "best_sensitivity")]
#[pyo3(signature = (times, coherences, readout, overhead = 0.0, echo = false))]
fn py_best_sensitivity(
    times: Vec<f64>,
    coherences: Vec<f64>,
    readout: f64,
    overhead: f64,
    echo: bool,
) -> PyResult<(f64, f64)> {
    optics::best_sensitivity(&times, &coherences, readout, overhead, echo).raise()
}

/// `[1 + 2(a + b)/(a − b)²]^{−1/2}` for mean photon numbers `a`, `b`.
#[pyfunction(name = "readout_contrast")]
fn py_readout_contrast(a: f64, b: f64) -> f64 {
    optics::readout_contrast(a, b)
}

/// The NV axes of the four orientations in the cubic frame.
#[pyfunction(name = "nv_orientations")]
fn py_nv_orientations() -> [[f64; 3]; 4] {
    optics::nv_orientations()
}

/// The two signed ESR lines of each orientation (Hz) for a field (T, cubic
/// frame).
#[pyfunction(name = "ensemble_lines")]
#[pyo3(signature = (field, zero_field_splitting = NV_ZERO_FIELD_SPLITTING, gyromagnetic_ratio = NV_GYROMAGNETIC_RATIO))]
fn py_ensemble_lines(
    field: [f64; 3],
    zero_field_splitting: f64,
    gyromagnetic_ratio: f64,
) -> PyResult<[[f64; 2]; 4]> {
    optics::ensemble_lines(field, zero_field_splitting, gyromagnetic_ratio).raise()
}

/// The field vector (T, cubic frame, up to its sign) from the lines of the
/// four orientations.
#[pyfunction(name = "field_from_lines")]
#[pyo3(signature = (lines, zero_field_splitting = NV_ZERO_FIELD_SPLITTING, gyromagnetic_ratio = NV_GYROMAGNETIC_RATIO))]
fn py_field_from_lines(
    lines: [[f64; 2]; 4],
    zero_field_splitting: f64,
    gyromagnetic_ratio: f64,
) -> PyResult<[f64; 3]> {
    optics::field_from_lines(lines, zero_field_splitting, gyromagnetic_ratio).raise()
}

/// `(|B_∥|, B_⊥)` (T) from the two signed lines of one orientation.
#[pyfunction(name = "axial_and_transverse")]
#[pyo3(signature = (lines, zero_field_splitting = NV_ZERO_FIELD_SPLITTING, gyromagnetic_ratio = NV_GYROMAGNETIC_RATIO))]
fn py_axial_and_transverse(
    lines: [f64; 2],
    zero_field_splitting: f64,
    gyromagnetic_ratio: f64,
) -> PyResult<(f64, f64)> {
    optics::axial_and_transverse(lines, zero_field_splitting, gyromagnetic_ratio).raise()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpticalNv>()?;
    m.add_class::<PyTwoLevelOdmr>()?;
    m.add_function(wrap_pyfunction!(py_ramsey_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(py_echo_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(py_best_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(py_readout_contrast, m)?)?;
    m.add_function(wrap_pyfunction!(py_nv_orientations, m)?)?;
    m.add_function(wrap_pyfunction!(py_ensemble_lines, m)?)?;
    m.add_function(wrap_pyfunction!(py_field_from_lines, m)?)?;
    m.add_function(wrap_pyfunction!(py_axial_and_transverse, m)?)?;
    m.add("NV_EXCITED_SPLITTING", NV_EXCITED_SPLITTING)?;
    m.add_class::<PySpinBath>()?;
    m.add_function(wrap_pyfunction!(py_point_dipole, m)?)?;
    m.add_function(wrap_pyfunction!(py_diamond_sites, m)?)?;
    m.add_function(wrap_pyfunction!(py_ensemble_signal, m)?)?;
    m.add("NATURAL_C13_ABUNDANCE", NATURAL_C13_ABUNDANCE)?;
    m.add("DIAMOND_LATTICE_CONSTANT", DIAMOND_LATTICE_CONSTANT)?;
    m.add("MAX_BATH_SPINS", MAX_BATH_SPINS)?;
    m.add("MAX_CLUSTER_ORDER", MAX_CLUSTER_ORDER)?;
    m.add("MAX_EXACT_BATH", MAX_EXACT_BATH)?;
    m.add("MAX_BATH_TRAJECTORIES", bath::MAX_BATH_TRAJECTORIES)?;
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
