//! Radical pairs: spin-selective recombination, field effects and CIDNP.

use metatron_qso::C64;
use metatron_qso::linalg::CMatrix;
use metatron_qso::spin::{
    Birth, Fresh, PairSolution, PairYields, Product, RadicalPair, Recombination,
};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::spin::PySpinSystem;
use crate::{ComplexMatrixArg, Raise, cmatrix, invalid, pydict, square_complex};

/// A birth (or a projector) by name.
fn birth(name: &str) -> PyResult<Birth> {
    match name {
        "singlet" | "S" => Ok(Birth::Singlet),
        "t0" | "T0" | "triplet-zero" => Ok(Birth::TripletZero),
        "t+" | "T+" | "triplet-plus" => Ok(Birth::TripletPlus),
        "t-" | "T-" | "triplet-minus" => Ok(Birth::TripletMinus),
        "triplet" | "T" => Ok(Birth::Triplet),
        "uncorrelated" => Ok(Birth::Uncorrelated),
        other => Err(invalid(format!(
            "unknown birth '{other}' (singlet, t0, t+, t-, triplet or uncorrelated)"
        ))),
    }
}

fn recombination(name: &str) -> PyResult<Recombination> {
    match name {
        "haberkorn" => Ok(Recombination::Haberkorn),
        "jones-maeda-hore" | "jmh" => Ok(Recombination::JonesMaedaHore),
        other => Err(invalid(format!(
            "unknown recombination '{other}' (haberkorn or jones-maeda-hore)"
        ))),
    }
}

fn recombination_name(model: Recombination) -> &'static str {
    match model {
        Recombination::JonesMaedaHore => "jones-maeda-hore",
        _ => "haberkorn",
    }
}

fn product(name: &str) -> PyResult<Product> {
    match name {
        "singlet" => Ok(Product::Singlet),
        "triplet" => Ok(Product::Triplet),
        "escape" => Ok(Product::Escape),
        other => Err(invalid(format!(
            "unknown product '{other}' (singlet, triplet or escape)"
        ))),
    }
}

/// `cidnp` as a NumPy array of nuclei × 3.
fn cidnp_array<'py>(py: Python<'py>, y: &PairYields) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((y.cidnp.len(), 3), |(j, p)| y.cidnp[j][p]).into_pyarray(py)
}

/// The yields as a dict.
fn yields_dict<'py>(
    py: Python<'py>,
    y: &PairYields,
    nuclei: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    Ok(pydict!(py, {
        "field" => y.field,
        "singlet" => y.singlet,
        "triplet" => y.triplet,
        "escape" => y.escape,
        "cidnp" => cidnp_array(py, y),
        "nuclei" => nuclei.to_vec(),
        "backward_error" => y.backward_error,
        "iterations" => y.iterations,
    }))
}

/// Two radicals (the system's two electrons), their nuclei and their
/// reactions: recombination from the singlet at `singlet`, from the
/// triplet at `triplet` and escape at `escape` (s⁻¹), with Haberkorn's
/// term (`"haberkorn"`) or that of Jones, Maeda and Hore
/// (`"jones-maeda-hore"`, `"jmh"`). Electron 1 (radical 1) is the
/// electron of lower index. Births and projectors are named `"singlet"`
/// (`"S"`), `"t0"`, `"t+"`, `"t-"` (also `"triplet-zero"`,
/// `"triplet-plus"`, `"triplet-minus"`, as the command line's JSON writes
/// them), `"triplet"` (`"T"`) and `"uncorrelated"`.
#[pyclass(name = "RadicalPair", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyRadicalPair {
    inner: RadicalPair,
}

#[pymethods]
impl PyRadicalPair {
    #[new]
    #[pyo3(signature = (system, singlet = 0.0, triplet = 0.0, escape = 0.0, recombination = "haberkorn"))]
    fn new(
        system: PyRef<'_, PySpinSystem>,
        singlet: f64,
        triplet: f64,
        escape: f64,
        recombination: &str,
    ) -> PyResult<Self> {
        let model = self::recombination(recombination)?;
        let inner = RadicalPair::new(system.inner.clone())
            .raise()?
            .with_rates(singlet, triplet, escape)
            .raise()?
            .with_recombination(model);
        Ok(Self { inner })
    }

    /// The spin system of the pair (a copy).
    #[getter]
    fn system(&self) -> PySpinSystem {
        PySpinSystem {
            inner: self.inner.system().clone(),
        }
    }

    /// The indices of the two electrons, the lower first.
    #[getter]
    fn electrons(&self) -> (usize, usize) {
        let [a, b] = self.inner.electrons();
        (a, b)
    }

    /// The nuclei in index order: the rows of `cidnp` and the order of the
    /// products' spins.
    #[getter]
    fn nuclei(&self) -> Vec<usize> {
        self.inner.nuclei()
    }

    /// `(k_S, k_T, k_E)` (s⁻¹).
    #[getter]
    fn rates(&self) -> (f64, f64, f64) {
        let [s, t, e] = self.inner.rates();
        (s, t, e)
    }

    /// `"haberkorn"` or `"jones-maeda-hore"`.
    #[getter]
    fn recombination(&self) -> &'static str {
        recombination_name(self.inner.recombination())
    }

    /// The same pair at another field (T).
    fn with_field(&self, field: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_field(field).raise()?,
        })
    }

    /// The electron projector of a birth (`P_S`, `P_T₀`, `P_T₊`, `P_T₋`,
    /// `P_T`, or the identity for `"uncorrelated"`) over the whole space.
    fn projector<'py>(&self, py: Python<'py>, which: &str) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.projector(birth(which)?)))
    }

    /// `projector(birth)/Tr`: the electrons in that state, the nuclei
    /// maximally mixed.
    #[pyo3(signature = (birth = "singlet"))]
    fn birth_state<'py>(
        &self,
        py: Python<'py>,
        birth: &str,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.birth_state(self::birth(birth)?)))
    }

    /// Yields, CIDNP and `X = ∫₀^∞ ρ dt` at the system's field, from
    /// `birth`, or from any start `rho0` (Hermitian, with a positive trace;
    /// normalised to trace 1) when given (`birth` is then left out). The
    /// solve runs without the GIL.
    #[pyo3(signature = (birth = None, rho0 = None))]
    fn solve(
        &self,
        py: Python<'_>,
        birth: Option<&str>,
        rho0: Option<ComplexMatrixArg<'_>>,
    ) -> PyResult<PyPairSolution> {
        let pair = &self.inner;
        let solution = match (birth, rho0) {
            (Some(_), Some(_)) => {
                return Err(invalid("give a birth or a start rho0, not both"));
            }
            (birth, None) => {
                let b = self::birth(birth.unwrap_or("singlet"))?;
                py.detach(|| pair.solve(b)).raise()?
            }
            (None, Some(rho0)) => {
                let (d, data) = square_complex("rho0", &rho0)?;
                let rho0 = CMatrix::from_row_slice(d, d, &data);
                py.detach(|| pair.solve_from(&rho0)).raise()?
            }
        };
        Ok(PyPairSolution {
            inner: solution,
            nuclei: pair.nuclei(),
        })
    }

    /// Yields and CIDNP at each field (T) from `birth`: the MARY curve, as
    /// a dict of arrays over the fields (`cidnp` of shape fields × nuclei
    /// × 3). The solves run without the GIL.
    #[pyo3(signature = (fields, birth = "singlet"))]
    fn field_effect<'py>(
        &self,
        py: Python<'py>,
        fields: Vec<f64>,
        birth: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let b = self::birth(birth)?;
        let pair = &self.inner;
        let all = py.detach(|| pair.field_effect(b, &fields)).raise()?;
        let n = pair.nuclei().len();
        let column = |f: fn(&PairYields) -> f64| -> Vec<f64> { all.iter().map(f).collect() };
        let cidnp = Array3::from_shape_fn((all.len(), n, 3), |(i, j, p)| all[i].cidnp[j][p])
            .into_pyarray(py);
        let iterations: Vec<i64> = all.iter().map(|y| y.iterations as i64).collect();
        Ok(pydict!(py, {
            "field" => column(|y| y.field).into_pyarray(py),
            "singlet" => column(|y| y.singlet).into_pyarray(py),
            "triplet" => column(|y| y.triplet).into_pyarray(py),
            "escape" => column(|y| y.escape).into_pyarray(py),
            "cidnp" => cidnp,
            "nuclei" => pair.nuclei(),
            "backward_error" => column(|y| y.backward_error).into_pyarray(py),
            "iterations" => iterations.into_pyarray(py),
        }))
    }

    /// `ρ(t)` under the pair's equation of motion at the system's field,
    /// from `rho` (`d × d`); the trace falls as the pair reacts. With all
    /// rates zero this is `system.evolve(rho, [delay(time)])`. Runs without
    /// the GIL.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        rho: ComplexMatrixArg<'_>,
        time: f64,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let (d, data) = square_complex("rho", &rho)?;
        let rho = CMatrix::from_row_slice(d, d, &data);
        let pair = &self.inner;
        Ok(cmatrix(py, &py.detach(|| pair.evolve(&rho, time)).raise()?))
    }

    fn __repr__(&self) -> String {
        let [s, t, e] = self.inner.rates();
        format!(
            "RadicalPair(dimension={}, nuclei={}, rates=({s:e}, {t:e}, {e:e}), recombination={:?})",
            self.inner.system().dimension(),
            self.inner.nuclei().len(),
            recombination_name(self.inner.recombination())
        )
    }
}

/// A solved radical pair: its yields, `X = ∫₀^∞ ρ dt` and the nuclear
/// states of its products (`"singlet"`, `"triplet"`, `"escape"`).
#[pyclass(name = "PairSolution", module = "metatron_qso", frozen)]
pub struct PyPairSolution {
    inner: PairSolution,
    /// The pair's nuclei (the rows of `cidnp`).
    nuclei: Vec<usize>,
}

#[pymethods]
impl PyPairSolution {
    /// Yields and polarisations per pair born: `field` (T), `singlet`,
    /// `triplet`, `escape`, `cidnp` (nuclei × 3: `⟨I_z⟩` in the singlet,
    /// triplet and escape products), `nuclei`, `backward_error` and
    /// `iterations` (0: direct Sylvester solves).
    #[getter]
    fn yields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        yields_dict(py, self.inner.yields(), &self.nuclei)
    }

    /// `X = ∫₀^∞ ρ dt` (s).
    fn integral<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        cmatrix(py, self.inner.integral())
    }

    /// The product's nuclear state over the pair's nuclei in their order,
    /// unnormalised: `k_P Tr_e(P_P X)`, of trace the product's yield.
    fn product<'py>(&self, py: Python<'py>, which: &str) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(cmatrix(py, &self.inner.product(self::product(which)?)))
    }

    /// The product's state on `system`, normalised: pair spin `i` becomes
    /// spin `j` of `system` for each `(i, j)` in `spins` (the same isotope,
    /// each spin once); the pair's other spins are traced out and the
    /// other spins of `system` arrive thermal, or two of them
    /// (`parahydrogen=(i, j, fraction)`) in the singlet with probability
    /// `fraction`. The result goes on to `SpinSystem.evolve` or to
    /// `Reactions.add_source`.
    #[pyo3(signature = (which, system, spins = Vec::new(), parahydrogen = None))]
    fn product_on<'py>(
        &self,
        py: Python<'py>,
        which: &str,
        system: PyRef<'_, PySpinSystem>,
        spins: Vec<(usize, usize)>,
        parahydrogen: Option<(usize, usize, f64)>,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let fresh = match parahydrogen {
            None => Fresh::Thermal,
            Some((i, j, fraction)) => Fresh::Parahydrogen {
                spins: [i, j],
                fraction,
            },
        };
        let which = self::product(which)?;
        let target = system.inner.clone();
        drop(system);
        let solution = &self.inner;
        let out = py
            .detach(|| solution.product_on(which, &target, &spins, fresh))
            .raise()?;
        Ok(cmatrix(py, &out))
    }

    fn __repr__(&self) -> String {
        let y = self.inner.yields();
        format!(
            "PairSolution(field={}, singlet={}, triplet={}, escape={})",
            y.field, y.singlet, y.triplet, y.escape
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRadicalPair>()?;
    m.add_class::<PyPairSolution>()?;
    Ok(())
}
