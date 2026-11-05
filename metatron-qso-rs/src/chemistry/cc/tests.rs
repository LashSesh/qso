//! Validation of MP2, CCSD and CCSD(T) (1.62).

use super::*;
use crate::chemistry::integrals::{Basis, Molecule};
use crate::chemistry::scf::{ScfOptions, rhf, rohf, uhf};

const H2O: &str = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692";
const LIH: &str = "Li 0 0 0; H 0 0 1.595";
const N2: &str = "N 0 0 0; N 0 0 1.0977";
const H2: &str = "H 0 0 0; H 0 0 0.735";
const OH: &str = "O 0 0 0; H 0 0 0.97";
const NH2: &str = "N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6";
const O2: &str = "O 0 0 0; O 0 0 1.2075";

type Method = fn(&Molecule, &AoIntegrals, &ScfOptions) -> crate::error::Result<ScfResult>;

fn setup(atoms: &str, basis: Basis, multiplicity: u32, method: Method) -> (ScfResult, AoIntegrals) {
    let mol = Molecule::from_atoms(atoms, 0, multiplicity).unwrap();
    let ints = AoIntegrals::compute(&mol, basis).unwrap();
    let options = ScfOptions {
        energy_tolerance: 1e-12,
        gradient_tolerance: 1e-9,
        ..ScfOptions::default()
    };
    (method(&mol, &ints, &options).unwrap(), ints)
}

/// PySCF 2.14 (`scf.RHF` with conv_tol 1e-13, then `mp.MP2`, `cc.CCSD`
/// with conv_tol 1e-12 and conv_tol_normt 1e-10, `ccsd_t.kernel`,
/// `get_t1_diagnostic`), geometries in bohr: basis, frozen orbitals, MP2,
/// CCSD and (T) correlation energies, T1 diagnostic.
#[rustfmt::skip]
const CLOSED: [(&str, Basis, usize, f64, f64, f64, f64); 20] = [
    (H2O, Basis::Sto3g, 0, -0.03554565164944129, -0.049438563034793256, -6.740968416021857e-05, 0.0043299679793998205),
    (H2O, Basis::Sto3g, 1, -0.03544594119987926, -0.04936022397391921, -6.748571270771238e-05, 0.00483735094111411),
    (H2O, Basis::SixThirtyOneG, 0, -0.12885091722190162, -0.13537949961735596, -0.0009958598248008747, 0.005999755400659754),
    (H2O, Basis::SixThirtyOneG, 1, -0.12781377130373422, -0.1344712679360825, -0.0009849202057073765, 0.006703085707512252),
    (H2O, Basis::CcPvdz, 0, -0.20400356383506027, -0.2133274268678449, -0.003058707400253549, 0.005252942649361242),
    (H2O, Basis::CcPvdz, 1, -0.20166597980719958, -0.2112326591918939, -0.003036490785481939, 0.005888954629811227),
    (LIH, Basis::Sto3g, 0, -0.012864638165834897, -0.020367576193063723, -8.388986126014873e-06, 0.019557231288134254),
    (LIH, Basis::Sto3g, 1, -0.012636673952663395, -0.0201506456447765, 0.0, 0.027465160195115013),
    (LIH, Basis::SixThirtyOneG, 0, -0.012602229243348162, -0.01899578444076132, -9.877301898241776e-06, 0.022792013937959994),
    (LIH, Basis::SixThirtyOneG, 1, -0.01234293815436815, -0.01873309086745776, 0.0, 0.03190275928478577),
    (LIH, Basis::CcPvdz, 0, -0.022814976372534164, -0.03110133737561362, -9.355065495825818e-06, 0.011795963896530632),
    (LIH, Basis::CcPvdz, 1, -0.022445325124699798, -0.030738734752173202, 0.0, 0.016594173096956072),
    (N2, Basis::Sto3g, 0, -0.15409049981051837, -0.1530479190036185, -0.0017153858253626329, 0.0027100269754546523),
    (N2, Basis::Sto3g, 2, -0.15367477059164764, -0.15275293189381994, -0.0017127124759014126, 0.003213268079688575),
    (N2, Basis::SixThirtyOneG, 0, -0.23870056444158427, -0.22775487995457547, -0.0075850320831220615, 0.009966643381905615),
    (N2, Basis::SixThirtyOneG, 2, -0.2364394334145952, -0.22578696883285623, -0.007544350602560488, 0.011798300537373527),
    (N2, Basis::CcPvdz, 0, -0.31059711384565225, -0.31308218781163033, -0.011936386111486605, 0.009894482171585),
    (N2, Basis::CcPvdz, 2, -0.3062970545361032, -0.30926379354807004, -0.011860888994263628, 0.011708996132271035),
    (H2, Basis::SixThirtyOneG, 0, -0.017327020533897335, -0.024804961815597746, 0.0, 0.003944152037048809),
    (H2, Basis::CcPvdz, 0, -0.026326192410096216, -0.03458678767560497, 0.0, 0.005046775052127327),
];

#[test]
fn closed_shells_match_pyscf_in_sto3g() {
    closed_shells_match(|b| b == Basis::Sto3g);
}

#[test]
fn closed_shells_match_pyscf_in_split_valence_and_cc_pvdz() {
    closed_shells_match(|b| b != Basis::Sto3g);
}

fn closed_shells_match(which: impl Fn(Basis) -> bool) {
    for (atoms, basis, frozen, mp2_e, ccsd_e, t_e, t1d) in CLOSED {
        if !which(basis) {
            continue;
        }
        let (hf, ints) = setup(atoms, basis, 1, rhf);
        let name = format!("{atoms} {basis:?} frozen {frozen}");
        let pt = mp2(&hf, &ints, frozen).unwrap();
        assert!(
            (pt.correlation - mp2_e).abs() < 1e-8,
            "{name}: MP2 {} vs {mp2_e}",
            pt.correlation
        );
        assert!((pt.same_spin + pt.opposite_spin - pt.correlation).abs() < 1e-14);
        let cc = ccsd_t(&hf, &ints, frozen, &CcOptions::default()).unwrap();
        assert!(!cc.ccsd.spin_orbitals);
        assert!(
            (cc.ccsd.correlation - ccsd_e).abs() < 1e-8,
            "{name}: CCSD {} vs {ccsd_e}",
            cc.ccsd.correlation
        );
        assert!(
            (cc.triples - t_e).abs() < 1e-8,
            "{name}: (T) {} vs {t_e}",
            cc.triples
        );
        assert!(
            (cc.ccsd.t1_diagnostic - t1d).abs() < 1e-8,
            "{name}: T1 {} vs {t1d}",
            cc.ccsd.t1_diagnostic
        );
        assert!((cc.energy - (hf.energy + ccsd_e + t_e)).abs() < 1e-8);
    }
}

/// PySCF 2.14 (`scf.UHF` or `scf.ROHF`, then `mp.MP2`, `cc.CCSD`,
/// `uccsd_t.kernel`): basis, multiplicity, UHF (else ROHF), frozen
/// orbitals, MP2, CCSD and (T) correlation energies (UHF only for MP2 and
/// (T)).
/// Molecule, basis, multiplicity, UHF, frozen, MP2, CCSD, (T).
type OpenCase = (&'static str, Basis, u32, bool, usize, f64, f64, f64);

#[rustfmt::skip]
const OPEN: [OpenCase; 14] = [
    (OH, Basis::SixThirtyOneG, 2, true, 0, -0.08918054448159576, -0.09882768680936423, -0.0005574953546378659),
    (OH, Basis::SixThirtyOneG, 2, true, 1, -0.08823017877986912, -0.09797235548364035, -0.000548117052090689),
    (OH, Basis::SixThirtyOneG, 2, false, 0, f64::NAN, -0.10014818757154079, f64::NAN),
    (OH, Basis::SixThirtyOneG, 2, false, 1, f64::NAN, -0.09928156732665849, f64::NAN),
    (NH2, Basis::SixThirtyOneGStar, 2, true, 0, -0.1322622470294003, -0.15059244970329425, -0.0016809020104161505),
    (NH2, Basis::SixThirtyOneGStar, 2, true, 1, -0.13049932003867162, -0.1490086409542308, -0.0016594660123317323),
    (NH2, Basis::SixThirtyOneGStar, 2, false, 0, f64::NAN, -0.1547928506871504, f64::NAN),
    (NH2, Basis::SixThirtyOneGStar, 2, false, 1, f64::NAN, -0.1531838573423805, f64::NAN),
    (O2, Basis::CcPvdz, 3, true, 2, -0.34471545234676104, -0.34826316227694787, -0.009709281680736528),
    (O2, Basis::CcPvdz, 3, false, 2, f64::NAN, -0.3667337365624451, f64::NAN),
    (OH, Basis::CcPvdz, 2, true, 0, -0.15100877050166847, -0.16552527943000628, -0.001751849059388646),
    (OH, Basis::CcPvdz, 2, true, 1, -0.14898593290003245, -0.1637028671856848, -0.0017305065190367627),
    (OH, Basis::CcPvdz, 2, false, 0, f64::NAN, -0.16933093170526484, f64::NAN),
    (OH, Basis::CcPvdz, 2, false, 1, f64::NAN, -0.167478082261773, f64::NAN),
];

#[test]
fn open_shells_match_pyscf() {
    for (atoms, basis, mult, unrestricted, frozen, mp2_e, ccsd_e, t_e) in OPEN {
        let method: Method = if unrestricted { uhf } else { rohf };
        let (hf, ints) = setup(atoms, basis, mult, method);
        let name = format!("{atoms} {basis:?} UHF {unrestricted} frozen {frozen}");
        if unrestricted {
            let pt = mp2(&hf, &ints, frozen).unwrap();
            assert!(
                (pt.correlation - mp2_e).abs() < 1e-8,
                "{name}: MP2 {} vs {mp2_e}",
                pt.correlation
            );
            let cc = ccsd_t(&hf, &ints, frozen, &CcOptions::default()).unwrap();
            assert!(cc.ccsd.spin_orbitals);
            assert!(
                (cc.ccsd.correlation - ccsd_e).abs() < 1e-8,
                "{name}: CCSD {} vs {ccsd_e}",
                cc.ccsd.correlation
            );
            assert!(
                (cc.triples - t_e).abs() < 1e-8,
                "{name}: (T) {} vs {t_e}",
                cc.triples
            );
        } else {
            assert!(mp2(&hf, &ints, frozen).is_err());
            assert!(ccsd_t(&hf, &ints, frozen, &CcOptions::default()).is_err());
            let cc = ccsd(&hf, &ints, frozen, &CcOptions::default()).unwrap();
            assert!(
                (cc.correlation - ccsd_e).abs() < 1e-8,
                "{name}: CCSD {} vs {ccsd_e}",
                cc.correlation
            );
        }
    }
}

#[test]
fn spin_orbitals_and_spatial_orbitals_agree_on_closed_shells() {
    for (atoms, basis, frozen) in [
        (H2O, Basis::Sto3g, 0),
        (H2O, Basis::SixThirtyOneG, 1),
        (N2, Basis::Sto3g, 2),
    ] {
        let (hf, ints) = setup(atoms, basis, 1, rhf);
        let options = CcOptions::default();
        let r = Reference::new(&hf, &ints, frozen).unwrap();
        let (closed, a) = run_ccsd(&r, &ints, &options, false).unwrap();
        let (spin, b) = run_ccsd(&r, &ints, &options, true).unwrap();
        assert!(spin.spin_orbitals && !closed.spin_orbitals);
        assert!(
            (closed.correlation - spin.correlation).abs() < 1e-9,
            "{atoms}"
        );
        assert!(
            (closed.t1_diagnostic - spin.t1_diagnostic).abs() < 1e-8,
            "{atoms}"
        );
        let (Solved::Closed(c, ta), Solved::Spin(s, tb)) = (a, b) else {
            panic!("the two forms")
        };
        assert!((c.triples(&ta) - s.triples(&tb)).abs() < 1e-10, "{atoms}");
        // UHF of a closed shell is RHF: the same in spin orbitals.
        let (u, ints_u) = setup(atoms, basis, 1, uhf);
        let cu = ccsd_t(&u, &ints_u, frozen, &options).unwrap();
        assert!(
            (cu.ccsd.correlation - closed.correlation).abs() < 1e-8,
            "{atoms}"
        );
        assert!((cu.triples - c.triples(&ta)).abs() < 1e-8, "{atoms}");
    }
}

#[test]
fn two_electrons_are_exact() {
    // CCSD is full CI for two electrons, and the triples vanish.
    for (atoms, basis, charge) in [
        (H2, Basis::SixThirtyOneG, 0),
        (H2, Basis::CcPvdz, 0),
        ("He 0 0 0; H 0 0 0.774", Basis::SixThirtyOneG, 1),
    ] {
        let mol = Molecule::from_atoms(atoms, charge, 1).unwrap();
        let ints = AoIntegrals::compute(&mol, basis).unwrap();
        let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
        let fci = hf
            .molecular_integrals(&ints, 0, None)
            .unwrap()
            .fci_energies(1)
            .unwrap()[0];
        let cc = ccsd_t(&hf, &ints, 0, &CcOptions::default()).unwrap();
        assert!(
            (cc.ccsd.energy - fci).abs() < 1e-9,
            "{atoms}: {} vs {fci}",
            cc.ccsd.energy
        );
        assert!(cc.triples.abs() < 1e-14, "{}", cc.triples);
        let u = uhf(&mol, &ints, &ScfOptions::default()).unwrap();
        let cu = ccsd(&u, &ints, 0, &CcOptions::default()).unwrap();
        assert!((cu.energy - fci).abs() < 1e-9, "{atoms}");
    }
}

#[test]
fn correlation_is_size_consistent() {
    // Two nitrogen molecules 100 Å apart (STO-3G; no dipoles, whose
    // interaction would show at 10⁻⁷, and quadrupoles at 3·10⁻¹¹): every
    // energy is twice the molecule's.
    let far = "N 0 0 0; N 0 0 1.0977; N 0 0 100; N 0 0 101.0977";
    let (one, ints1) = setup(N2, Basis::Sto3g, 1, rhf);
    let (two, ints2) = setup(far, Basis::Sto3g, 1, rhf);
    assert!((two.energy - 2.0 * one.energy).abs() < 1e-9);
    let options = CcOptions::default();
    let (a, b) = (mp2(&one, &ints1, 2).unwrap(), mp2(&two, &ints2, 4).unwrap());
    assert!((b.correlation - 2.0 * a.correlation).abs() < 1e-9);
    let (a, b) = (
        ccsd_t(&one, &ints1, 2, &options).unwrap(),
        ccsd_t(&two, &ints2, 4, &options).unwrap(),
    );
    assert!(
        (b.ccsd.correlation - 2.0 * a.ccsd.correlation).abs() < 1e-9,
        "{} vs {}",
        b.ccsd.correlation,
        2.0 * a.ccsd.correlation
    );
    assert!((b.triples - 2.0 * a.triples).abs() < 1e-10);
}

#[test]
fn mp2_is_the_energy_of_the_first_amplitudes() {
    // The closed form of MP2 in canonical orbitals against the CCSD energy
    // functional at the MP2 amplitudes (t₁ = 0, t₂ = (ia|jb)/D or
    // ⟨ij||ab⟩/D).
    for (atoms, basis, mult, method, frozen) in [
        (H2O, Basis::SixThirtyOneG, 1, rhf as Method, 1),
        (OH, Basis::SixThirtyOneG, 2, uhf as Method, 0),
        (NH2, Basis::SixThirtyOneGStar, 2, uhf as Method, 1),
    ] {
        let (hf, ints) = setup(atoms, basis, mult, method);
        let pt = mp2(&hf, &ints, frozen).unwrap();
        let r = Reference::new(&hf, &ints, frozen).unwrap();
        let s = Spin::new(&r, &ints).unwrap();
        let t = s.start();
        let (t1, t2) = t.split_at(s.o * s.v);
        assert!(t1.iter().all(|x| x.abs() < 1e-7));
        assert!((s.energy(t1, t2) - pt.correlation).abs() < 1e-10, "{atoms}");
        if r.closed_shell() {
            let c = Closed::new(&r, &ints);
            let t = c.start();
            let (t1, t2) = t.split_at(c.o * c.v);
            assert!((c.energy(t1, t2) - pt.correlation).abs() < 1e-12);
        }
    }
}

#[test]
fn correlated_methods_check_their_input() {
    let (hf, ints) = setup(H2O, Basis::Sto3g, 1, rhf);
    assert!(mp2(&hf, &ints, 6).is_err());
    assert!(ccsd(&hf, &ints, 6, &CcOptions::default()).is_err());
    let other = AoIntegrals::compute(
        &Molecule::from_atoms(H2O, 0, 1).unwrap(),
        Basis::SixThirtyOneG,
    )
    .unwrap();
    assert!(mp2(&hf, &other, 0).is_err());
    for max_iterations in [0, 1] {
        let bad = CcOptions {
            max_iterations,
            ..CcOptions::default()
        };
        assert!(ccsd(&hf, &ints, 0, &bad).is_err());
    }
    let short = CcOptions {
        max_iterations: 2,
        ..CcOptions::default()
    };
    assert!(ccsd(&hf, &ints, 0, &short).is_err());
    // MP2 and (T) need canonical orbitals: a loosely converged
    // Hartree–Fock calculation is refused, CCSD takes it.
    let mol = Molecule::from_atoms(H2O, 0, 1).unwrap();
    let loose = ScfOptions {
        energy_tolerance: 1e-3,
        gradient_tolerance: 1e-2,
        ..ScfOptions::default()
    };
    let rough = rhf(&mol, &ints, &loose).unwrap();
    assert!(Reference::new(&rough, &ints, 0).unwrap().off_diagonal() > CANONICAL_TOLERANCE);
    assert!(mp2(&rough, &ints, 0).is_err());
    assert!(ccsd_t(&rough, &ints, 0, &CcOptions::default()).is_err());
    assert!(ccsd(&rough, &ints, 0, &CcOptions::default()).is_ok());
    // No correlated occupied orbitals: a T1 diagnostic of +0.
    let (h2, ints_h2) = setup(H2, Basis::SixThirtyOneG, 1, rhf);
    for spin in [false, true] {
        let r = Reference::new(&h2, &ints_h2, 1).unwrap();
        let (cc, _) = run_ccsd(&r, &ints_h2, &CcOptions::default(), spin).unwrap();
        assert!(cc.correlation == 0.0 && cc.t1_diagnostic == 0.0);
        assert!(cc.t1_diagnostic.is_sign_positive());
    }
    // Without virtual orbitals nothing correlates.
    let he = Molecule::from_atoms("He 0 0 0", 0, 1).unwrap();
    let ints_he = AoIntegrals::compute(&he, Basis::Sto3g).unwrap();
    let hf_he = rhf(&he, &ints_he, &ScfOptions::default()).unwrap();
    assert_eq!(
        ccsd_t(&hf_he, &ints_he, 0, &CcOptions::default())
            .unwrap()
            .energy,
        hf_he.energy
    );
    assert_eq!(mp2(&hf_he, &ints_he, 0).unwrap().correlation, 0.0);
}

#[test]
fn mp2_parts_match_pyscf() {
    // PySCF 2.14 `e_corr_ss` and `e_corr_os`.
    for (atoms, basis, mult, method, frozen, same, opposite) in [
        (
            H2O,
            Basis::CcPvdz,
            1,
            rhf as Method,
            1,
            -0.05071689385987746,
            -0.1509490859486789,
        ),
        (
            OH,
            Basis::SixThirtyOneG,
            2,
            uhf as Method,
            0,
            -0.020820011569448908,
            -0.06836053289884131,
        ),
    ] {
        let (hf, ints) = setup(atoms, basis, mult, method);
        let pt = mp2(&hf, &ints, frozen).unwrap();
        assert!(
            (pt.same_spin - same).abs() < 1e-9,
            "{atoms}: {} vs {same}",
            pt.same_spin
        );
        assert!(
            (pt.opposite_spin - opposite).abs() < 1e-9,
            "{atoms}: {} vs {opposite}",
            pt.opposite_spin
        );
    }
}

#[test]
fn stretched_bonds_converge() {
    // N₂ at 2.2 Å in 6-31G with two frozen orbitals, where DIIS on the
    // change of each step instead of between successive steps stalled;
    // PySCF 2.14 (CCSD to 10⁻¹²) needs 110 iterations.
    let (hf, ints) = setup("N 0 0 0; N 0 0 2.2", Basis::SixThirtyOneG, 1, rhf);
    assert!((hf.energy - -108.21646279056053).abs() < 1e-9);
    let cc = ccsd(&hf, &ints, 2, &CcOptions::default()).unwrap();
    assert!(
        (cc.correlation - -0.7040644455958951).abs() < 1e-8,
        "{}",
        cc.correlation
    );
}
