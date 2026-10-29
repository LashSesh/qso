//! Nuclear spin dynamics and magnetic resonance.

use metatron_qso::C64;
use metatron_qso::linalg::CMatrix;
use metatron_qso::spin::{
    Event, Frame, Isotope, Segment, Spectrum, SpinSystem, Target, inept, inept_cycle,
    inversion_recovery, spin_echo,
};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::PyGraph;
use crate::{
    ComplexArg, ComplexMatrixArg, Raise, cmatrix, complex_vec, invalid, pydict, square_complex,
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

/// A pulse target: an isotope name (`"1H"`) or a spin index.
fn target(t: &Bound<'_, PyAny>) -> PyResult<Target> {
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
    Ok(wrap(inept(isotope(source)?, isotope(target)?, j)))
}

/// INEPT with its two-step phase cycle: `[(events, receiver phase)]`.
#[pyfunction(name = "inept_cycle")]
fn py_inept_cycle(source: &str, target: &str, j: f64) -> PyResult<Vec<(Vec<PyEvent>, f64)>> {
    Ok(inept_cycle(isotope(source)?, isotope(target)?, j)
        .into_iter()
        .map(|(e, r)| (wrap(e), r))
        .collect())
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

/// Gyromagnetic ratio (rad s⁻¹ T⁻¹) of an isotope.
#[pyfunction]
fn gyromagnetic_ratio(isotope_name: &str) -> PyResult<f64> {
    Ok(isotope(isotope_name)?.gamma())
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
    fn state(&self, rho: &ComplexMatrixArg<'_>) -> PyResult<CMatrix> {
        let (d, data) = square_complex("rho", rho)?;
        if d != self.inner.dimension() {
            return Err(invalid(format!(
                "a state of this system is {0}×{0}, got {d}×{d}",
                self.inner.dimension()
            )));
        }
        Ok(CMatrix::from_row_slice(d, d, &data))
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

    /// Set `T₁` and `T₂` (s) of spin `i` (`None`: none, `T₂` then `2T₁`).
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
        Ok(cmatrix(py, &self.inner.time_average(&rho).raise()?))
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
        let op = self.state(&op)?;
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
        self.inner
            .lines(&rho, self::isotope(isotope)?)
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
    Ok(())
}
