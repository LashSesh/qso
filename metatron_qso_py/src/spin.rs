//! Nuclear spin dynamics and magnetic resonance.

use metatron_qso::C64;
use metatron_qso::linalg::CMatrix;
use metatron_qso::optimize::Options;
use metatron_qso::spin::{
    Channel, Control, Drive, Event, Exchange, Frame, Fresh, Isotope, Member, Optimized, Profile,
    Segment, Spectrum, SpinSystem, Target, axial_csa, inept, inept_cycle, inversion_recovery, m2s,
    s2m, spin_echo, spin_lock, unitary_bound,
};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::graph::PyGraph;
use crate::{
    ComplexArg, ComplexMatrixArg, Raise, RealArg, cmatrix, complex_vec, invalid, pydict,
    real_array, real_vec, square_complex,
};

fn isotope(name: &str) -> PyResult<Isotope> {
    Isotope::from_name(name).raise()
}

fn frame(name: &str) -> PyResult<Frame> {
    match name {
        "rotating" => Ok(Frame::Rotating),
        "laboratory" | "lab" => Ok(Frame::Laboratory),
        other => Err(invalid(format!(
            "unknown frame '{other}' (use rotating or laboratory)"
        ))),
    }
}

/// A pulse target: an isotope name (`"1H"`) or a spin index (not a
/// bool, which Python would take for 0 or 1).
fn target(t: &Bound<'_, PyAny>) -> PyResult<Target> {
    if t.is_instance_of::<PyBool>() {
        return Err(invalid(
            "a pulse target is an isotope name or a spin index, not a bool",
        ));
    }
    if let Ok(i) = t.extract::<usize>() {
        return Ok(Target::Spin(i));
    }
    if let Ok(name) = t.extract::<String>() {
        return Ok(Target::Isotope(isotope(&name)?));
    }
    Err(invalid("a pulse target is an isotope name or a spin index"))
}

/// One step of a pulse sequence (made by `pulse`, `delay`, `shaped`,
/// `set_field` and `sweep`).
#[pyclass(name = "SpinEvent", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyEvent {
    inner: Event,
}

#[pymethods]
impl PyEvent {
    fn __repr__(&self) -> String {
        format!("SpinEvent({:?})", self.inner)
    }
}

fn events(list: &[PyRef<'_, PyEvent>]) -> Vec<Event> {
    list.iter().map(|e| e.inner.clone()).collect()
}

fn wrap(events: Vec<Event>) -> Vec<PyEvent> {
    events.into_iter().map(|inner| PyEvent { inner }).collect()
}

/// An ideal pulse of `angle` degrees about the axis at `phase` degrees
/// from x, on an isotope (`"1H"`) or one spin (its index).
#[pyfunction(name = "spin_pulse")]
#[pyo3(signature = (target, angle, phase = 0.0))]
fn pulse(target: &Bound<'_, PyAny>, angle: f64, phase: f64) -> PyResult<PyEvent> {
    Ok(PyEvent {
        inner: Event::pulse(self::target(target)?, angle, phase),
    })
}

/// Free evolution for `t` seconds.
#[pyfunction(name = "spin_delay")]
fn delay(t: f64) -> PyEvent {
    PyEvent {
        inner: Event::Delay(t),
    }
}

/// A shaped pulse: segments `(duration s, amplitude Hz, phase degrees)`
/// (rotating frame).
#[pyfunction(name = "spin_shaped")]
fn shaped(target: &Bound<'_, PyAny>, segments: Vec<(f64, f64, f64)>) -> PyResult<PyEvent> {
    Ok(PyEvent {
        inner: Event::Shaped {
            target: self::target(target)?,
            segments: segments
                .into_iter()
                .map(|(duration, amplitude, phase)| Segment {
                    duration,
                    amplitude,
                    phase,
                })
                .collect(),
        },
    })
}

/// Radiofrequency fields on several channels at once (rotating frame):
/// during step `k` (of `step` s) the amplitudes `amplitudes[k][c] = (x,
/// y)` (Hz) on `targets[c]`, as a nested list or an array of shape
/// `(steps, channels, 2)`. The pulses of `SpinSystem.optimize_control`.
#[pyfunction(name = "rf_pulse")]
fn rf_pulse(
    targets: Vec<Bound<'_, PyAny>>,
    step: f64,
    amplitudes: Vec<Vec<[f64; 2]>>,
) -> PyResult<PyEvent> {
    Ok(PyEvent {
        inner: Event::Control {
            targets: targets.iter().map(target).collect::<PyResult<_>>()?,
            step,
            amplitudes,
        },
    })
}

/// Set the field (T) at once.
#[pyfunction(name = "spin_field")]
fn set_field(b: f64) -> PyEvent {
    PyEvent {
        inner: Event::Field(b),
    }
}

/// Change the field linearly to `to` (T) over `duration` s in `steps`
/// constant pieces.
#[pyfunction(name = "spin_sweep")]
#[pyo3(signature = (to, duration, steps = 1000))]
fn sweep(to: f64, duration: f64, steps: usize) -> PyEvent {
    PyEvent {
        inner: Event::Sweep {
            to,
            duration,
            steps,
        },
    }
}

/// INEPT from `source` to `target` for coupling `j` (Hz).
#[pyfunction(name = "inept")]
fn py_inept(source: &str, target: &str, j: f64) -> PyResult<Vec<PyEvent>> {
    Ok(wrap(inept(isotope(source)?, isotope(target)?, j).raise()?))
}

/// INEPT with its two-step phase cycle: `[(events, receiver phase)]`.
#[pyfunction(name = "inept_cycle")]
fn py_inept_cycle(source: &str, target: &str, j: f64) -> PyResult<Vec<(Vec<PyEvent>, f64)>> {
    Ok(inept_cycle(isotope(source)?, isotope(target)?, j)
        .raise()?
        .into_iter()
        .map(|(e, r)| (wrap(e), r))
        .collect())
}

/// Magnetisation-to-singlet conversion (M2S) of a near-equivalent pair:
/// coupling `j` and shift difference `delta_nu` (Hz).
#[pyfunction(name = "m2s")]
fn py_m2s(isotope_name: &str, j: f64, delta_nu: f64) -> PyResult<Vec<PyEvent>> {
    Ok(wrap(m2s(isotope(isotope_name)?, j, delta_nu).raise()?))
}

/// Singlet-to-magnetisation conversion (S2M): M2S reversed.
#[pyfunction(name = "s2m")]
fn py_s2m(isotope_name: &str, j: f64, delta_nu: f64) -> PyResult<Vec<PyEvent>> {
    Ok(wrap(s2m(isotope(isotope_name)?, j, delta_nu).raise()?))
}

/// Continuous irradiation of an isotope at `nu1` Hz for `duration` s (a
/// spin lock).
#[pyfunction(name = "spin_lock")]
#[pyo3(signature = (isotope_name, nu1, duration, phase = 0.0))]
fn py_spin_lock(isotope_name: &str, nu1: f64, duration: f64, phase: f64) -> PyResult<PyEvent> {
    Ok(PyEvent {
        inner: spin_lock(isotope(isotope_name)?, nu1, duration, phase),
    })
}

/// A spin echo `90x – τ – 180y – τ` on an isotope.
#[pyfunction(name = "spin_echo")]
fn py_spin_echo(isotope_name: &str, tau: f64) -> PyResult<Vec<PyEvent>> {
    Ok(wrap(spin_echo(isotope(isotope_name)?, tau)))
}

/// Inversion recovery `180x – t – 90x` on an isotope.
#[pyfunction(name = "inversion_recovery")]
fn py_inversion_recovery(isotope_name: &str, t: f64) -> PyResult<Vec<PyEvent>> {
    Ok(wrap(inversion_recovery(isotope(isotope_name)?, t)))
}

/// Frequencies and values of a spectrum.
type SpectrumArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<C64>>);

/// The Fourier transform of a free induction decay: `(frequencies Hz,
/// values)`, after exponential line broadening `lb` (Hz).
#[pyfunction(name = "fid_spectrum")]
#[pyo3(signature = (fid, dwell, broadening = 0.0))]
fn spectrum<'py>(
    py: Python<'py>,
    fid: ComplexArg<'_>,
    dwell: f64,
    broadening: f64,
) -> PyResult<SpectrumArrays<'py>> {
    let s = Spectrum::from_fid(&complex_vec(&fid), dwell, broadening).raise()?;
    Ok((s.frequencies.into_pyarray(py), s.values.into_pyarray(py)))
}

/// The shielding tensor (ppm) of an axially symmetric anisotropy
/// `Δσ = σ_∥ − σ_⊥` along `axis`.
#[pyfunction(name = "axial_csa")]
fn py_axial_csa<'py>(
    py: Python<'py>,
    anisotropy: f64,
    axis: [f64; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t = axial_csa(anisotropy, axis).raise()?;
    Ok(PyArray2::from_vec2(
        py,
        &t.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
    )?)
}

/// Gyromagnetic ratio (rad s⁻¹ T⁻¹) of an isotope.
#[pyfunction]
fn gyromagnetic_ratio(isotope_name: &str) -> PyResult<f64> {
    Ok(isotope(isotope_name)?.gamma())
}

/// A square complex matrix of any dimension.
fn any_matrix(name: &str, m: &ComplexMatrixArg<'_>) -> PyResult<CMatrix> {
    let (d, data) = square_complex(name, m)?;
    Ok(CMatrix::from_row_slice(d, d, &data))
}

/// The largest `Re Tr(C ρ')` over all `ρ' = UρU†` (Hermitian `ρ` and
/// `C`): the sum of products of their ordered eigenvalues (Glaser et al.
/// 1998).
#[pyfunction(name = "unitary_bound")]
fn py_unitary_bound(rho: ComplexMatrixArg<'_>, target: ComplexMatrixArg<'_>) -> PyResult<f64> {
    unitary_bound(&any_matrix("rho", &rho)?, &any_matrix("target", &target)?).raise()
}

/// An optimal-control problem (GRAPE): controls constant over `steps`
/// equal steps of `duration` s, maximising `Re Tr(C† ρ(T))` from
/// `initial` to `target`. Made by `Control.rf` or `Control.field`.
#[pyclass(name = "Control", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyControl {
    inner: Control,
}

#[pymethods]
impl PyControl {
    /// rf controls on `targets` (isotope names or spin indices): an x and
    /// a y amplitude (Hz) per channel and step, each pair at most `limit`
    /// Hz if given (rotating frame).
    #[staticmethod]
    #[pyo3(signature = (targets, duration, steps, initial, target, limit = None))]
    fn rf(
        targets: Vec<Bound<'_, PyAny>>,
        duration: f64,
        steps: usize,
        initial: ComplexMatrixArg<'_>,
        target: ComplexMatrixArg<'_>,
        limit: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Control::rf(
                targets.iter().map(self::target).collect::<PyResult<_>>()?,
                limit,
                duration,
                steps,
                any_matrix("initial", &initial)?,
                any_matrix("target", &target)?,
            ),
        })
    }

    /// A field profile (T) between `minimum` and `maximum`.
    #[staticmethod]
    fn field(
        minimum: f64,
        maximum: f64,
        duration: f64,
        steps: usize,
        initial: ComplexMatrixArg<'_>,
        target: ComplexMatrixArg<'_>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Control::field(
                minimum,
                maximum,
                duration,
                steps,
                any_matrix("initial", &initial)?,
                any_matrix("target", &target)?,
            ),
        })
    }

    /// Members `(weight, rf scales per channel, offsets per channel in
    /// Hz)`; empty lists mean 1 and 0. Empty: the ideal member alone.
    #[getter]
    fn ensemble(&self) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        self.inner
            .ensemble
            .iter()
            .map(|m| (m.weight, m.scales.clone(), m.offsets.clone()))
            .collect()
    }

    #[setter]
    fn set_ensemble(&mut self, members: Vec<(f64, Vec<f64>, Vec<f64>)>) {
        self.inner.ensemble = members
            .into_iter()
            .map(|(weight, scales, offsets)| Member {
                weight,
                scales,
                offsets,
            })
            .collect();
    }

    /// Weight of `Σ u² Δt` subtracted from the transfer.
    #[getter]
    fn power(&self) -> f64 {
        self.inner.power
    }

    #[setter]
    fn set_power(&mut self, value: f64) {
        self.inner.power = value;
    }

    /// Weight of `Σ (u_{k+1} − u_k)²` subtracted from the transfer.
    #[getter]
    fn smoothness(&self) -> f64 {
        self.inner.smoothness
    }

    #[setter]
    fn set_smoothness(&mut self, value: f64) {
        self.inner.smoothness = value;
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Seeded random controls: amplitudes in a disc of radius `scale` Hz
    /// (at most 0.9 of the limit), or fields uniform in the inner 96 % of
    /// the range (`scale` unused).
    #[pyo3(signature = (seed = 0, scale = 100.0))]
    fn random_start<'py>(
        &self,
        py: Python<'py>,
        seed: u64,
        scale: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !(scale.is_finite() && scale >= 0.0) {
            return Err(invalid("scale must be finite and ≥ 0"));
        }
        Ok(real_array(py, self.inner.random_start(seed, scale)))
    }

    fn __repr__(&self) -> String {
        let what = match &self.inner.drive {
            Drive::Rf { targets, .. } => format!("rf on {} channels", targets.len()),
            Drive::Field { .. } => "field".to_string(),
        };
        format!(
            "Control({what}, {} s in {} steps)",
            self.inner.duration, self.inner.steps
        )
    }
}

/// A field profile for an `Exchange`: the field constant over `steps`
/// steps of `duration` s between `minimum` and `maximum` (T), from
/// `initial` (one state per site) towards the largest `Σ_s Re Tr(C_s†
/// ρ_s(T))`, `C_s = target[s]`, less `smoothness · Σ (ΔB)²`.
#[pyclass(name = "Profile", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyProfile {
    inner: Profile,
}

#[pymethods]
impl PyProfile {
    #[new]
    #[pyo3(signature = (minimum, maximum, duration, steps, initial, target, smoothness = 0.0))]
    fn new(
        minimum: f64,
        maximum: f64,
        duration: f64,
        steps: usize,
        initial: Vec<ComplexMatrixArg<'_>>,
        target: Vec<ComplexMatrixArg<'_>>,
        smoothness: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Profile {
                min: minimum,
                max: maximum,
                duration,
                steps,
                initial: initial
                    .iter()
                    .map(|m| any_matrix("initial", m))
                    .collect::<PyResult<_>>()?,
                target: target
                    .iter()
                    .map(|m| any_matrix("target", m))
                    .collect::<PyResult<_>>()?,
                smoothness,
            },
        })
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps
    }

    fn __repr__(&self) -> String {
        format!(
            "Profile({}..{} T, {} s in {} steps)",
            self.inner.min, self.inner.max, self.inner.duration, self.inner.steps
        )
    }
}

/// The outcome of an optimisation as a dict.
fn optimized<'py>(py: Python<'py>, r: Optimized) -> PyResult<Bound<'py, PyDict>> {
    Ok(pydict!(py, {
        "controls" => real_array(py, r.controls),
        "transfer" => r.transfer,
        "members" => r.members,
        "objective" => r.objective,
        "iterations" => r.iterations,
        "evaluations" => r.evaluations,
        "converged" => r.converged,
        "message" => r.message,
    }))
}

fn options(max_iterations: usize) -> Options {
    Options {
        max_iterations,
        g_tol: 1e-10,
        record_history: false,
        ..Options::default()
    }
}

/// A molecule's nuclear spins: isotopes, shifts (ppm), couplings (Hz),
/// the field (T), the temperature (K) and the frame (`"rotating"` or
/// `"laboratory"`). States are density matrices (NumPy, complex).
#[pyclass(name = "SpinSystem", module = "metatron_qso")]
#[derive(Clone)]
pub struct PySpinSystem {
    inner: SpinSystem,
}

impl PySpinSystem {
    /// A `d × d` matrix of this system: `what` is "a state" or "an
    /// operator".
    fn matrix(&self, what: &str, m: &ComplexMatrixArg<'_>) -> PyResult<CMatrix> {
        let (d, data) = square_complex("rho", m)?;
        if d != self.inner.dimension() {
            return Err(invalid(format!(
                "{what} of this system is {0}×{0}, got {d}×{d}",
                self.inner.dimension()
            )));
        }
        Ok(CMatrix::from_row_slice(d, d, &data))
    }

    fn state(&self, rho: &ComplexMatrixArg<'_>) -> PyResult<CMatrix> {
        self.matrix("a state", rho)
    }

    fn spin(&self, i: usize) -> PyResult<&metatron_qso::spin::Spin> {
        self.inner.spins().get(i).ok_or_else(|| {
            invalid(format!(
                "spin {i} out of range ({} spins)",
                self.inner.spins().len()
            ))
        })
    }
}

#[pymethods]
impl PySpinSystem {
    #[new]
    #[pyo3(signature = (field, temperature = 298.15, frame = "rotating"))]
    fn new(field: f64, temperature: f64, frame: &str) -> PyResult<Self> {
        let inner = SpinSystem::new(0.0)
            .with_field(field)
            .raise()?
            .with_temperature(temperature)
            .raise()?
            .with_frame(self::frame(frame)?);
        Ok(Self { inner })
    }

    /// Add a nucleus (`"1H"`, `"2H"`, `"13C"`, `"15N"`, `"19F"`, `"31P"`)
    /// with a chemical shift (ppm) and optional `T₁`, `T₂` (s); returns its
    /// index.
    #[pyo3(signature = (isotope, shift = 0.0, t1 = None, t2 = None))]
    fn add_spin(
        &mut self,
        isotope: &str,
        shift: f64,
        t1: Option<f64>,
        t2: Option<f64>,
    ) -> PyResult<usize> {
        // On an error the system stays as it was.
        let mut next = self.inner.clone();
        let i = next.add_spin(self::isotope(isotope)?, shift).raise()?;
        next.set_relaxation(i, t1, t2).raise()?;
        self.inner = next;
        Ok(i)
    }

    /// Set the coupling `J` (Hz) between spins `i` and `j` (0 removes it).
    fn couple(&mut self, i: usize, j: usize, coupling: f64) -> PyResult<()> {
        self.inner.couple(i, j, coupling).raise()?;
        Ok(())
    }

    /// Set `T₁` and `T₂` (s) of spin `i` (`None`: none, `T₂` then `2T₁`;
    /// `T₂ ≤ 2T₁`). On an error the spin keeps its times.
    #[pyo3(signature = (i, t1 = None, t2 = None))]
    fn set_relaxation(&mut self, i: usize, t1: Option<f64>, t2: Option<f64>) -> PyResult<()> {
        self.inner.set_relaxation(i, t1, t2).raise()?;
        Ok(())
    }

    /// The same system at another field (T).
    fn with_field(&self, field: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_field(field).raise()?,
        })
    }

    /// Set the position (Å) of spin `i`, for dipolar relaxation.
    fn set_position(&mut self, i: usize, position: [f64; 3]) -> PyResult<()> {
        self.inner.set_position(i, position).raise()?;
        Ok(())
    }

    /// The position (Å) of spin `i`, if set.
    fn position(&self, i: usize) -> PyResult<Option<[f64; 3]>> {
        Ok(self.spin(i)?.position)
    }

    /// Set the shielding tensor (ppm, 3×3, molecular frame) of spin `i`,
    /// or remove it with `None`; its traceless symmetric part relaxes.
    #[pyo3(signature = (i, tensor))]
    fn set_csa(&mut self, i: usize, tensor: Option<[[f64; 3]; 3]>) -> PyResult<()> {
        self.inner.set_csa(i, tensor).raise()?;
        Ok(())
    }

    /// The shielding tensor (ppm) of spin `i`, if set.
    fn csa<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match self.spin(i)?.csa {
            Some(t) => Ok(Some(PyArray2::from_vec2(
                py,
                &t.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
            )?)),
            None => Ok(None),
        }
    }

    /// The same system tumbling isotropically with correlation time
    /// `tau_c` (s), which turns on Redfield relaxation; `None` turns it
    /// off.
    fn with_correlation_time(&self, tau_c: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_correlation_time(tau_c).raise()?,
        })
    }

    #[getter]
    fn correlation_time(&self) -> Option<f64> {
        self.inner.correlation_time()
    }

    /// The dipolar coupling constant `(μ₀/4π) γ_i γ_j ħ / r³` (rad/s).
    fn dipolar_coupling(&self, i: usize, j: usize) -> PyResult<f64> {
        self.inner.dipolar_coupling(i, j).raise()
    }

    /// Relaxation rates (1/s) among observables: `Γ_ij = −Re Tr(O_i†
    /// D(O_j)) / Tr(O_i† O_i)`, `D` the relaxation without `H`. The
    /// diagonal holds auto-relaxation rates (`1/T₁` of `I_z`, `1/T₂` of
    /// `I₊`), the rest cross-relaxation.
    fn relaxation_rates<'py>(
        &self,
        py: Python<'py>,
        ops: Vec<ComplexMatrixArg<'py>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ops = ops
            .iter()
            .map(|o| self.matrix("an operator", o))
            .collect::<PyResult<Vec<CMatrix>>>()?;
        let sys = &self.inner;
        let g = py.detach(|| sys.relaxation_rates(&ops)).raise()?;
        let rows: Vec<Vec<f64>> = (0..g.nrows())
            .map(|r| (0..g.ncols()).map(|c| g[(r, c)]).collect())
            .collect();
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    /// The same system at another temperature (K).
    fn with_temperature(&self, temperature: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_temperature(temperature).raise()?,
        })
    }

    /// The same system in another frame.
    fn with_frame(&self, frame: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_frame(self::frame(frame)?),
        })
    }

    /// The same system with an isotope's carrier at `ppm`.
    fn with_carrier(&self, isotope: &str, ppm: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_carrier(self::isotope(isotope)?, ppm)
                .raise()?,
        })
    }

    #[getter]
    fn field(&self) -> f64 {
        self.inner.field()
    }

    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.temperature()
    }

    #[getter]
    fn frame(&self) -> &'static str {
        match self.inner.frame() {
            Frame::Rotating => "rotating",
            Frame::Laboratory => "laboratory",
        }
    }

    /// The carrier of an isotope (ppm, rotating frame; 0 unless set).
    fn carrier(&self, isotope: &str) -> PyResult<f64> {
        Ok(self.inner.carrier(self::isotope(isotope)?))
    }

    /// The field (T) at the end of a sequence: `evolve` leaves the system's
    /// own field unchanged, so a state left at another field is observed
    /// with `with_field(field_after(events))`.
    fn field_after(&self, events: Vec<PyRef<'_, PyEvent>>) -> f64 {
        self.inner.field_after(&self::events(&events))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// `[(isotope, shift, t1, t2)]`.
    #[getter]
    fn spins(&self) -> Vec<(&'static str, f64, Option<f64>, Option<f64>)> {
        self.inner
            .spins()
            .iter()
            .map(|s| (s.isotope.name(), s.shift, s.t1, s.t2))
            .collect()
    }

    /// `[(i, j, J)]`.
    #[getter]
    fn couplings(&self) -> Vec<(usize, usize, f64)> {
        self.inner.couplings().to_vec()
    }

    /// The coupling network as a graph (weights J in Hz).
    fn coupling_graph(&self) -> PyResult<PyGraph> {
        Ok(PyGraph {
            inner: self.inner.coupling_graph().raise()?,
        })
    }

    /// The free Hamiltonian (rad/s).
    fn hamiltonian<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.hamiltonian().raise()?))
    }

    /// Thermal equilibrium at the field and temperature.
    fn thermal_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.thermal_state().raise()?))
    }

    /// The maximally mixed state.
    fn identity_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        cmatrix(py, &self.inner.identity_state())
    }

    /// Spins `i`, `j` in the singlet with probability `fraction` (1:
    /// parahydrogen, 0.25: normal hydrogen), the rest maximally mixed.
    #[pyo3(signature = (i, j, fraction = 1.0))]
    fn parahydrogen_state<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        j: usize,
        fraction: f64,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(
            py,
            &self.inner.parahydrogen_state(i, j, fraction).raise()?,
        ))
    }

    /// The state averaged over a long, random time of free evolution.
    fn time_average<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let rho = self.state(&rho)?;
        let sys = &self.inner;
        Ok(cmatrix(py, &py.detach(|| sys.time_average(&rho)).raise()?))
    }

    /// Apply a sequence of events.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
        events: Vec<PyRef<'py, PyEvent>>,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let rho = self.state(&rho)?;
        let ev = self::events(&events);
        let sys = &self.inner;
        Ok(cmatrix(py, &py.detach(|| sys.evolve(&rho, &ev)).raise()?))
    }

    /// `I_z`, `I_x`, `I_y` or `I₊` (`which` = "z", "x", "y", "+") of spin
    /// `i`.
    #[pyo3(signature = (i, which = "z"))]
    fn spin_operator<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        which: &str,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let m = match which {
            "z" => self.inner.iz(i),
            "x" => self.inner.ix(i),
            "y" => self.inner.iy(i),
            "+" => self.inner.iplus(i),
            other => {
                return Err(invalid(format!(
                    "unknown component '{other}' (z, x, y or +)"
                )));
            }
        };
        Ok(cmatrix(py, &m.raise()?))
    }

    /// `F_z`, `F_x`, `F_y` or `F₊` of an isotope.
    #[pyo3(signature = (isotope, which = "z"))]
    fn total_operator<'py>(
        &self,
        py: Python<'py>,
        isotope: &str,
        which: &str,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let x = self::isotope(isotope)?;
        let m = match which {
            "z" => self.inner.fz(x),
            "x" => self.inner.fx(x),
            "y" => self.inner.fy(x),
            "+" => self.inner.fplus(x),
            other => {
                return Err(invalid(format!(
                    "unknown component '{other}' (z, x, y or +)"
                )));
            }
        };
        Ok(cmatrix(py, &m.raise()?))
    }

    /// `Tr(ρ O)`.
    fn expectation(&self, rho: ComplexMatrixArg<'_>, op: ComplexMatrixArg<'_>) -> PyResult<C64> {
        let rho = self.state(&rho)?;
        let op = self.matrix("an operator", &op)?;
        self.inner.expectation(&rho, &op).raise()
    }

    /// Polarisation `⟨I_z⟩/I` of spin `i`.
    fn polarization(&self, rho: ComplexMatrixArg<'_>, i: usize) -> PyResult<f64> {
        let rho = self.state(&rho)?;
        self.inner.polarization(&rho, i).raise()
    }

    /// The exact stick spectrum on an isotope: dicts with `frequency`
    /// (Hz), `ppm` and complex `amplitude`.
    fn lines<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
        isotope: &str,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let rho = self.state(&rho)?;
        let x = self::isotope(isotope)?;
        let sys = &self.inner;
        py.detach(|| sys.lines(&rho, x))
            .raise()?
            .into_iter()
            .map(|l| {
                Ok(pydict!(py, {
                    "frequency" => l.frequency,
                    "ppm" => l.ppm,
                    "amplitude" => l.amplitude,
                }))
            })
            .collect()
    }

    /// Free induction decay `Tr(ρ(k·dwell) F₊)` on an isotope.
    fn fid<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
        isotope: &str,
        dwell: f64,
        points: usize,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let rho = self.state(&rho)?;
        let x = self::isotope(isotope)?;
        let sys = &self.inner;
        Ok(py
            .detach(|| sys.fid(&rho, x, dwell, points))
            .raise()?
            .into_pyarray(py))
    }

    /// The FID summed over a phase cycle `[(events, receiver phase °)]`.
    fn phase_cycle<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
        cycle: Vec<(Vec<PyRef<'py, PyEvent>>, f64)>,
        isotope: &str,
        dwell: f64,
        points: usize,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let rho = self.state(&rho)?;
        let x = self::isotope(isotope)?;
        let cycle: Vec<(Vec<Event>, f64)> = cycle.iter().map(|(e, r)| (events(e), *r)).collect();
        let sys = &self.inner;
        Ok(py
            .detach(|| sys.phase_cycle(&rho, &cycle, x, dwell, points))
            .raise()?
            .into_pyarray(py))
    }

    /// Chemical shift (ppm) of a frequency (Hz) on an isotope.
    fn ppm(&self, isotope: &str, frequency: f64) -> PyResult<f64> {
        Ok(self.inner.ppm(self::isotope(isotope)?, frequency))
    }

    /// The transfer `Re Tr(C† ρ(T))` (averaged over the ensemble) of the
    /// controls `u` and its exact gradient.
    fn control_transfer<'py>(
        &self,
        py: Python<'py>,
        control: PyRef<'_, PyControl>,
        u: RealArg<'_>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let u = real_vec(&u);
        let (sys, c) = (&self.inner, &control.inner);
        let (value, grad) = py.detach(|| sys.control_transfer(c, &u)).raise()?;
        Ok((value, real_array(py, grad)))
    }

    /// Optimise the controls with L-BFGS from `start` (default: random of
    /// `scale` Hz, or uniform fields, seeded): a dict with `controls`,
    /// `transfer`, `members`, `objective`, `iterations`, `evaluations`,
    /// `converged` and `message`.
    #[pyo3(signature = (control, start = None, seed = 0, scale = 100.0, max_iterations = 500))]
    fn optimize_control<'py>(
        &self,
        py: Python<'py>,
        control: PyRef<'_, PyControl>,
        start: Option<RealArg<'_>>,
        seed: u64,
        scale: f64,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c = &control.inner;
        let start = match start {
            Some(s) => real_vec(&s),
            None => c.random_start(seed, scale),
        };
        let sys = &self.inner;
        let r = py
            .detach(|| sys.optimize_control(c, &start, &options(max_iterations)))
            .raise()?;
        optimized(py, r)
    }

    /// The events that apply controls `u` in `evolve`.
    fn control_events(
        &self,
        control: PyRef<'_, PyControl>,
        u: RealArg<'_>,
    ) -> PyResult<Vec<PyEvent>> {
        Ok(wrap(
            self.inner
                .control_events(&control.inner, &real_vec(&u))
                .raise()?,
        ))
    }

    /// The system as JSON.
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// A system from JSON (checked).
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SpinSystem::from_json(text).raise()?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SpinSystem({} spins, {} T, {})",
            self.inner.spins().len(),
            self.inner.field(),
            self.frame()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpinSystem>()?;
    m.add_class::<PyEvent>()?;
    m.add_function(wrap_pyfunction!(pulse, m)?)?;
    m.add_function(wrap_pyfunction!(delay, m)?)?;
    m.add_function(wrap_pyfunction!(shaped, m)?)?;
    m.add_function(wrap_pyfunction!(set_field, m)?)?;
    m.add_function(wrap_pyfunction!(sweep, m)?)?;
    m.add_function(wrap_pyfunction!(py_inept, m)?)?;
    m.add_function(wrap_pyfunction!(py_inept_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(py_spin_echo, m)?)?;
    m.add_function(wrap_pyfunction!(py_inversion_recovery, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(gyromagnetic_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_axial_csa, m)?)?;
    m.add_function(wrap_pyfunction!(py_m2s, m)?)?;
    m.add_function(wrap_pyfunction!(py_s2m, m)?)?;
    m.add_function(wrap_pyfunction!(py_spin_lock, m)?)?;
    m.add_class::<PyExchange>()?;
    m.add_class::<PyControl>()?;
    m.add_class::<PyProfile>()?;
    m.add_function(wrap_pyfunction!(rf_pulse, m)?)?;
    m.add_function(wrap_pyfunction!(py_unitary_bound, m)?)?;
    Ok(())
}

/// Chemical exchange between spin systems (Bloch–McConnell in Liouville
/// space): sites with concentrations and first-order channels that carry
/// some spins over and bring others in fresh (thermal or parahydrogen).
/// States are lists of density matrices, one per site, trace 1 each.
#[pyclass(name = "Exchange", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyExchange {
    inner: Exchange,
}

impl PyExchange {
    fn states(&self, states: &[ComplexMatrixArg<'_>]) -> PyResult<Vec<CMatrix>> {
        if states.len() != self.inner.sites().len() {
            return Err(invalid(format!(
                "{} states for {} sites",
                states.len(),
                self.inner.sites().len()
            )));
        }
        states
            .iter()
            .zip(self.inner.sites())
            .map(|(m, site)| {
                let (d, data) = square_complex("states", m)?;
                if d != site.system.dimension() {
                    return Err(invalid(format!(
                        "a state of this site is {0}×{0}, got {d}×{d}",
                        site.system.dimension()
                    )));
                }
                Ok(CMatrix::from_row_slice(d, d, &data))
            })
            .collect()
    }
}

fn matrices<'py>(py: Python<'py>, ms: &[CMatrix]) -> Vec<Bound<'py, PyArray2<C64>>> {
    ms.iter().map(|m| cmatrix(py, m)).collect()
}

#[pymethods]
impl PyExchange {
    #[new]
    fn new() -> Self {
        Self {
            inner: Exchange::new(),
        }
    }

    /// Add a site (a spin system and its concentration); returns its index.
    fn add_site(&mut self, system: PyRef<'_, PySpinSystem>, concentration: f64) -> PyResult<usize> {
        self.inner
            .add_site(system.inner.clone(), concentration)
            .raise()
    }

    /// Add a channel from site `source` to site `target` at `rate` (s⁻¹,
    /// the fraction of the source converted per second), carrying the
    /// spin pairs `carried` `[(spin of source, spin of target)]`; the
    /// target's other spins arrive thermal, or two of them
    /// (`parahydrogen=(i, j)`) in the singlet with probability `fraction`
    /// (default 1).
    #[pyo3(signature = (source, target, rate, carried = Vec::new(), parahydrogen = None, fraction = None))]
    fn add_channel(
        &mut self,
        source: usize,
        target: usize,
        rate: f64,
        carried: Vec<(usize, usize)>,
        parahydrogen: Option<(usize, usize)>,
        fraction: Option<f64>,
    ) -> PyResult<()> {
        let fresh = match (parahydrogen, fraction) {
            (None, None) => Fresh::Thermal,
            (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "fraction: only with parahydrogen=(i, j)",
                ));
            }
            (Some((i, j)), fraction) => Fresh::Parahydrogen {
                spins: [i, j],
                fraction: fraction.unwrap_or(1.0),
            },
        };
        self.inner
            .add_channel(Channel {
                from: source,
                to: target,
                rate,
                carried,
                fresh,
            })
            .raise()?;
        Ok(())
    }

    /// Check the sites, channels and the chemical steady state.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().raise()
    }

    /// `[(SpinSystem, concentration)]`.
    #[getter]
    fn sites(&self) -> Vec<(PySpinSystem, f64)> {
        self.inner
            .sites()
            .iter()
            .map(|s| {
                (
                    PySpinSystem {
                        inner: s.system.clone(),
                    },
                    s.concentration,
                )
            })
            .collect()
    }

    #[getter]
    fn field(&self) -> f64 {
        self.inner.field()
    }

    /// The same exchange at another field (T).
    fn with_field(&self, field: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_field(field).raise()?,
        })
    }

    /// Thermal equilibrium of every site.
    fn thermal_state<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<C64>>>> {
        Ok(matrices(py, &self.inner.thermal_state().raise()?))
    }

    /// Apply pulses on isotopes, delays, fields and sweeps to the states.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        states: Vec<ComplexMatrixArg<'py>>,
        events: Vec<PyRef<'py, PyEvent>>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<C64>>>> {
        let states = self.states(&states)?;
        let ev = self::events(&events);
        let x = &self.inner;
        Ok(matrices(py, &py.detach(|| x.evolve(&states, &ev)).raise()?))
    }

    /// The steady state of exchange and relaxation at the field.
    fn steady_state<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<C64>>>> {
        let x = &self.inner;
        Ok(matrices(py, &py.detach(|| x.steady_state()).raise()?))
    }

    /// The free induction decay summed over the sites weighted by
    /// concentration.
    fn fid<'py>(
        &self,
        py: Python<'py>,
        states: Vec<ComplexMatrixArg<'py>>,
        isotope: &str,
        dwell: f64,
        points: usize,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let states = self.states(&states)?;
        let iso = self::isotope(isotope)?;
        let x = &self.inner;
        Ok(py
            .detach(|| x.fid(&states, iso, dwell, points))
            .raise()?
            .into_pyarray(py))
    }

    /// Polarisation `⟨I_z⟩/I` of a spin of a site.
    fn polarization(
        &self,
        states: Vec<ComplexMatrixArg<'_>>,
        site: usize,
        spin: usize,
    ) -> PyResult<f64> {
        let states = self.states(&states)?;
        self.inner.polarization(&states, site, spin).raise()
    }

    /// `Σ_s Re Tr(C_s† ρ_s(T))` under the field profile `fields` (T per
    /// step) and its gradient.
    fn profile_transfer<'py>(
        &self,
        py: Python<'py>,
        profile: PyRef<'_, PyProfile>,
        fields: RealArg<'_>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let b = real_vec(&fields);
        let (x, p) = (&self.inner, &profile.inner);
        let (value, grad) = py.detach(|| x.profile_transfer(p, &b)).raise()?;
        Ok((value, real_array(py, grad)))
    }

    /// Optimise a field profile with L-BFGS from `start` (fields within
    /// the range): a dict as `SpinSystem.optimize_control` returns.
    #[pyo3(signature = (profile, start, max_iterations = 500))]
    fn optimize_profile<'py>(
        &self,
        py: Python<'py>,
        profile: PyRef<'_, PyProfile>,
        start: RealArg<'_>,
        max_iterations: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let start = real_vec(&start);
        let (x, p) = (&self.inner, &profile.inner);
        let r = py
            .detach(|| x.optimize_profile(p, &start, &options(max_iterations)))
            .raise()?;
        optimized(py, r)
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Exchange::from_json(text).raise()?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Exchange({} sites, {} channels)",
            self.inner.sites().len(),
            self.inner.channels().len()
        )
    }
}
