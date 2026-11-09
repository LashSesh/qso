use super::*;
use crate::qec::splitting::SplittingOptions;
use crate::qec::{BpMethod, Noise, OsdMethod};
use crate::qec::{
    CircuitNoise, CircuitStep, CssCode, CssSchedule, Decoder, DecoderChoice, ErasureNoise,
    ExtendedOp, MemoryBasis, QecOp, RepeatedCircuit, correlated_error_model,
    css_code_memory_erasure, css_code_memory_erasure_repeated, css_code_memory_with,
    detector_error_model, enumerated_failure_rate, explain_errors, leading_failures,
    logical_error_rate_decoding, logical_error_rate_using, repetition_code_memory,
    repetition_code_memory_erasure, repetition_code_memory_erasure_repeated,
    repetition_code_memory_repeated, sample_detectors, splitting_failure_rate,
    surface_code_memory_erasure, surface_code_memory_erasure_repeated,
    surface_code_memory_repeated, surface_code_memory_with,
};
use crate::rng::Rng;

fn circuit(text: &str) -> DetectorCircuit {
    DetectorCircuit::from_stim(text).unwrap()
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
}

/// The model's errors equal these (detectors, observables, probability),
/// keys exactly and probabilities to `rel`.
fn assert_errors(dem: &DetectorErrorModel, expected: &[(&[usize], u64, f64)], rel: f64) {
    let mut want: Vec<(Vec<usize>, u64, f64)> = expected
        .iter()
        .map(|(d, o, p)| (d.to_vec(), *o, *p))
        .collect();
    want.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
    let got: Vec<(Vec<usize>, u64, f64)> = dem
        .errors
        .iter()
        .map(|e| (e.detectors.clone(), e.observables, e.probability))
        .collect();
    assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
    for (g, w) in got.iter().zip(&want) {
        assert_eq!((&g.0, g.1), (&w.0, w.1), "{got:?} against {want:?}");
        assert!(close(g.2, w.2, rel), "{got:?} against {want:?}");
    }
}

const A: &str = "HERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]";
const G: &str = "H 0\nHERALDED_PAULI_CHANNEL_1(0.01,0.02,0.03,0.04) 0\nH 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]";
const H: &str = "H 0\nCX 0 1\nHERALDED_ERASE(0.2) 0\nCX 0 1\nH 0\nM 0 1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]";
const I: &str = "HERALDED_ERASE(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-2]\nDETECTOR rec[-1]";
const K: &str = "HERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-2] rec[-1]";
const OBS_DET: &str =
    "HERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-1]";
const WIDE: &str = "HERALDED_ERASE(0.1) 0\nCX 0 1 0 2\nM 0 1 2\nDETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]";
const TWO: &str = "HERALDED_ERASE(0.1) 0 1\nM 0 1\nDETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]";

/// The circuit of the cross-validation (`dem_rule.py`), with heralds read
/// by their own detectors, unread, and inside a REPEAT block.
const REPEATED: &str = "R 0 1 2 3 4
M 1 3
REPEAT 6 {
    HERALDED_ERASE(0.1) 0 2
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 4
    DETECTOR rec[-1]
    DEPOLARIZE1(0.01) 0 2 4
    CX 0 1 2 1 2 3 4 3
    HERALDED_ERASE(0.05) 1 3
    DETECTOR rec[-1]
    MR 1 3
    DETECTOR rec[-1] rec[-8]
    DETECTOR rec[-2] rec[-9]
}
M 0 2 4
DETECTOR rec[-2] rec[-3] rec[-5]
DETECTOR rec[-1] rec[-2] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
";

// ---- error models against Stim 1.16 ------------------------------------

#[test]
fn heralded_channels_enter_the_model_by_stims_approximate_rule() {
    // Values of Stim 1.16's detector_error_model(approximate_disjoint_errors=True).
    let model = |t: &str| detector_error_model(&circuit(t), false).unwrap();
    let rel = 1e-14;
    assert_errors(&model(A), &[(&[0], 0, 0.05), (&[0, 1], 0, 0.05)], rel);
    assert_errors(
        &model("HERALDED_ERASE(0.1) !0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]"),
        &[(&[0], 0, 0.05), (&[0, 1], 0, 0.05)],
        rel,
    );
    assert_errors(
        &model("HERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]"),
        &[(&[0, 1], 0, 0.05), (&[1], 0, 0.05)],
        rel,
    );
    // B: the herald unread.
    assert_errors(
        &model("HERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-1]"),
        &[(&[0], 0, 0.05)],
        rel,
    );
    // C, D: merged with other errors as independent ones.
    assert_errors(
        &model("X_ERROR(0.1) 0\nHERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-1]"),
        &[(&[0], 0, 0.14)],
        rel,
    );
    assert_errors(
        &model("HERALDED_ERASE(0.1) 0\nHERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-1]"),
        &[(&[0], 0, 0.095)],
        rel,
    );
    // E, F: probability 1, and outcomes of equal effect summed.
    assert_errors(
        &model(&A.replace("0.1", "1")),
        &[(&[0], 0, 0.5), (&[0, 1], 0, 0.5)],
        rel,
    );
    assert_errors(
        &model(&A.replace(
            "HERALDED_ERASE(0.1)",
            "HERALDED_PAULI_CHANNEL_1(0.3,0.3,0.3,0.1)",
        )),
        &[(&[0], 0, 0.4), (&[0, 1], 0, 0.6)],
        rel,
    );
    assert_errors(&model(G), &[(&[0], 0, 0.03), (&[0, 1], 0, 0.07)], rel);
    assert_errors(
        &model(H),
        &[
            (&[0], 0, 0.05),
            (&[0, 1], 0, 0.05),
            (&[0, 1, 2], 0, 0.05),
            (&[0, 2], 0, 0.05),
        ],
        rel,
    );
    assert_errors(&model(I), &[(&[0], 1, 0.05), (&[], 1, 0.05)], rel);
    // J: duplicate targets, two records.
    let j = circuit("HERALDED_ERASE(0.1) 0 0\nM 0\nDETECTOR rec[-1]");
    assert_eq!(j.n_measurements(), 3);
    assert_errors(
        &detector_error_model(&j, false).unwrap(),
        &[(&[0], 0, 0.095)],
        rel,
    );
    assert_errors(&model(K), &[(&[0], 0, 0.05), (&[0, 1], 0, 0.05)], rel);
    assert_errors(&model(OBS_DET), &[(&[0], 0, 0.05), (&[0], 1, 0.05)], rel);
    assert_errors(
        &model(
            "HERALDED_PAULI_CHANNEL_1(0.1,0.2,0.3,0.4) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
        ),
        &[(&[0, 1], 0, 0.5), (&[1], 0, 0.5)],
        rel,
    );
    assert_errors(
        &model(WIDE),
        &[(&[0], 0, 0.05), (&[0, 1, 2, 3], 0, 0.05)],
        rel,
    );
}

#[test]
fn heralded_channels_decompose_as_stim_without_the_single_error_rule() {
    let parts = |t: &str| -> Vec<(f64, Vec<Vec<usize>>)> {
        correlated_error_model(&circuit(t))
            .unwrap()
            .errors
            .into_iter()
            .map(|e| {
                assert!(e.parts.iter().all(|p| p.1 == 0));
                (e.probability, e.parts.into_iter().map(|p| p.0).collect())
            })
            .collect()
    };
    let check = |t: &str, want: &[(f64, Vec<Vec<usize>>)]| {
        let got = parts(t);
        assert_eq!(got.len(), want.len(), "{t}: {got:?}");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.1, w.1, "{t}: {got:?}");
            assert!(close(g.0, w.0, 1e-14), "{t}: {got:?}");
        }
    };
    check(A, &[(0.05, vec![vec![0]]), (0.05, vec![vec![0], vec![1]])]);
    check(
        "HERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
        &[(0.05, vec![vec![0], vec![1]]), (0.05, vec![vec![1]])],
    );
    for args in ["0,0.1,0,0", "0,0,0.1,0"] {
        check(
            &format!("HERALDED_PAULI_CHANNEL_1({args}) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]"),
            &[(0.1, vec![vec![0], vec![1]])],
        );
    }
    // A Y error stays whole, as before.
    check(
        "Y_ERROR(0.1) 0\nM 0 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
        &[(0.1, vec![vec![0, 1]])],
    );
    check(
        H,
        &[
            (0.05, vec![vec![0]]),
            (0.05, vec![vec![0], vec![1]]),
            (0.05, vec![vec![0], vec![1], vec![2]]),
            (0.05, vec![vec![0], vec![2]]),
        ],
    );
    check(G, &[(0.03, vec![vec![0]]), (0.07, vec![vec![0], vec![1]])]);
}

fn assert_same_model(a: &DetectorErrorModel, b: &DetectorErrorModel, rel: f64) {
    assert_eq!(
        (a.n_detectors, a.n_observables),
        (b.n_detectors, b.n_observables)
    );
    assert_eq!(a.errors.len(), b.errors.len());
    for (x, y) in a.errors.iter().zip(&b.errors) {
        assert_eq!((&x.detectors, x.observables), (&y.detectors, y.observables));
        assert!(close(x.probability, y.probability, rel), "{x:?} {y:?}");
    }
}

fn assert_same_heralded(a: &HeraldedErrorModel, b: &HeraldedErrorModel, rel: f64) {
    assert_same_model(a.base(), b.base(), rel);
    assert_eq!(a.herald_detectors(), b.herald_detectors());
    for h in 0..a.herald_detectors().len() {
        assert!(close(a.herald_probability(h), b.herald_probability(h), rel));
        let (x, y) = (a.outcomes(h), b.outcomes(h));
        assert_eq!(x.len(), y.len());
        for (p, q) in x.iter().zip(y) {
            assert_eq!((&p.detectors, p.observables), (&q.detectors, q.observables));
            assert!(close(p.probability, q.probability, rel));
        }
    }
}

#[test]
fn repeated_blocks_fold_heralded_channels() {
    let r6 = RepeatedCircuit::from_stim(
        "REPEAT 6 {\nHERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n}",
    )
    .unwrap();
    let flat = r6.flatten().unwrap();
    assert_eq!(flat.n_measurements(), 12);
    let folded = r6.detector_error_model(false).unwrap();
    assert_same_model(&folded, &detector_error_model(&flat, false).unwrap(), 1e-14);
    assert_eq!(folded.errors.len(), 12);
    assert!(
        folded
            .errors
            .iter()
            .all(|e| close(e.probability, 0.05, 1e-14))
    );
    let has = |d: &[usize]| folded.errors.iter().any(|e| e.detectors == d);
    assert!(has(&[0]) && has(&[0, 1, 3, 5, 7, 9, 11]) && has(&[2]) && has(&[2, 3, 5, 7, 9, 11]));
    assert_same_heralded(
        &r6.heralded_error_model(false).unwrap(),
        &heralded_error_model(&flat, false).unwrap(),
        1e-14,
    );

    let r = RepeatedCircuit::from_stim(REPEATED).unwrap();
    let flat = r.flatten().unwrap();
    assert_eq!((flat.n_detectors(), flat.n_measurements()), (38, 47));
    assert_eq!((r.n_detectors(), r.n_measurements()), (38, 47));
    let folded = r.detector_error_model(false).unwrap();
    assert_same_model(&folded, &detector_error_model(&flat, false).unwrap(), 1e-12);
    assert_eq!(folded.errors.len(), 72);
    let sum: f64 = folded.errors.iter().map(|e| e.probability).sum();
    assert!((sum - 2.37).abs() < 1e-12, "{sum}");
    let max = folded
        .errors
        .iter()
        .map(|e| e.probability)
        .fold(0.0, f64::max);
    assert!(close(max, 0.05, 1e-14));
    for decompose in [false, true] {
        assert_same_model(
            &r.detector_error_model(decompose).unwrap(),
            &detector_error_model(&flat, decompose).unwrap(),
            1e-12,
        );
    }

    // The heralded split (`rep_parts.py`, `split.py`).
    let m = r.heralded_error_model(false).unwrap();
    assert_same_heralded(&m, &heralded_error_model(&flat, false).unwrap(), 1e-12);
    assert_eq!(m.herald_detectors().len(), 24);
    assert_eq!(&m.herald_detectors()[..6], &[0, 1, 2, 3, 6, 7]);
    assert_eq!(m.base().errors.len(), 24);
    let sum: f64 = m.base().errors.iter().map(|e| e.probability).sum();
    assert!((sum - 0.27).abs() < 1e-12, "{sum}");
    let (mut tenth, mut twentieth) = (0, 0);
    for h in 0..24 {
        assert_eq!(m.outcomes(h).len(), 1);
        assert_eq!(m.outcomes(h)[0].probability, 0.5);
        let p = m.herald_probability(h);
        if close(p, 0.1, 1e-15) {
            tenth += 1;
        } else if close(p, 0.05, 1e-15) {
            twentieth += 1;
        }
    }
    assert_eq!((tenth, twentieth), (18, 6));
}

// ---- the heralded model -------------------------------------------------

#[test]
fn the_heralded_model_splits_off_the_owned_heralds() {
    let split = |t: &str| heralded_error_model(&circuit(t), false).unwrap();
    let part = |p: f64, d: &[usize]| ErrorMechanism {
        probability: p,
        detectors: d.to_vec(),
        observables: 0,
    };
    let a = split(A);
    assert_eq!(a.herald_detectors(), &[0]);
    assert_eq!(a.herald_probability(0), 0.1);
    assert_eq!(a.outcomes(0), &[part(0.5, &[1])]);
    assert!(a.base().errors.is_empty());
    assert_eq!((a.n_detectors(), a.n_observables()), (2, 0));

    let f = split(&A.replace(
        "HERALDED_ERASE(0.1)",
        "HERALDED_PAULI_CHANNEL_1(0.3,0.3,0.3,0.1)",
    ));
    assert!(close(f.herald_probability(0), 1.0, 1e-15));
    assert_eq!(f.outcomes(0).len(), 1);
    assert_eq!(f.outcomes(0)[0].detectors, vec![1]);
    assert!(close(f.outcomes(0)[0].probability, 0.6, 1e-15));
    assert!(f.base().errors.is_empty());

    let g = split(G);
    assert_eq!(g.herald_detectors(), &[0]);
    assert!(close(g.herald_probability(0), 0.1, 1e-15));
    assert_eq!(g.outcomes(0).len(), 1);
    assert_eq!(g.outcomes(0)[0].detectors, vec![1]);
    assert!(close(g.outcomes(0)[0].probability, 0.7, 1e-15));

    let h = split(H);
    assert_eq!(h.herald_probability(0), 0.2);
    assert_eq!(h.outcomes(0), &[part(0.5, &[1]), part(0.5, &[2])]);
    assert!(h.base().errors.is_empty());

    let two = split(TWO);
    assert_eq!(two.herald_detectors(), &[0, 1]);
    assert_eq!(
        (two.herald_probability(0), two.herald_probability(1)),
        (0.1, 0.1)
    );
    assert_eq!(two.outcomes(0), &[part(0.5, &[2])]);
    assert_eq!(two.outcomes(1), &[part(0.5, &[3])]);
    assert!(two.base().errors.is_empty());

    for decompose in [false, true] {
        let w = heralded_error_model(&circuit(WIDE), decompose).unwrap();
        assert_eq!(w.herald_detectors(), &[0]);
        assert_eq!(w.herald_probability(0), 0.1);
        assert_eq!(w.outcomes(0), &[part(0.5, &[1, 2, 3])]);
        assert!(w.base().errors.is_empty());
    }

    // At probability 0 the herald stays listed, with no parts.
    let zero = split(&A.replace("0.1", "0"));
    assert_eq!(zero.herald_detectors(), &[0]);
    assert_eq!(zero.herald_probability(0), 0.0);
    assert!(zero.outcomes(0).is_empty());
    assert!(zero.base().errors.is_empty());

    // Heralds read by an observable or by a detector of several records
    // are not owned: the whole model is the base.
    for t in [I, K, OBS_DET] {
        let m = split(t);
        assert!(m.herald_detectors().is_empty(), "{t}");
        assert_eq!(m.base(), &detector_error_model(&circuit(t), false).unwrap());
    }

    // Conditioning case A.
    let c = a.conditioned(&[0]).unwrap();
    assert_errors(&c, &[(&[1], 0, 0.5)], 0.0);
    assert!(a.conditioned(&[0, 0]).unwrap().errors.is_empty());
    assert!(a.conditioned(&[]).unwrap().errors.is_empty());
    assert!(a.conditioned(&[1]).is_err());
}

#[test]
fn heralded_models_are_checked() {
    let dem = |errors: Vec<ErrorMechanism>| DetectorErrorModel {
        n_detectors: 4,
        n_observables: 1,
        errors,
    };
    let e = |p: f64, d: &[usize]| ErrorMechanism {
        probability: p,
        detectors: d.to_vec(),
        observables: 0,
    };
    let good = HeraldedErrorModel::new(
        dem(vec![e(0.1, &[1, 2])]),
        vec![0, 3],
        vec![0.1, 0.2],
        vec![vec![e(0.5, &[2]), e(0.5, &[1]), e(0.5, &[1])], vec![]],
    )
    .unwrap();
    // Sorted, equal parts merged (½ absorbs).
    assert_eq!(good.outcomes(0), &[e(0.5, &[1]), e(0.5, &[2])]);
    assert_eq!(good.n_detectors(), 4);
    let refused = [
        (vec![3, 0], vec![0.1, 0.1], vec![vec![], vec![]], vec![]),
        (vec![0, 0], vec![0.1, 0.1], vec![vec![], vec![]], vec![]),
        (vec![4], vec![0.1], vec![vec![]], vec![]),
        (vec![0], vec![0.1], vec![vec![]], vec![e(0.1, &[0, 1])]),
        (vec![0], vec![0.1, 0.2], vec![vec![]], vec![]),
        (vec![0], vec![0.1], vec![], vec![]),
        (vec![0], vec![1.5], vec![vec![]], vec![]),
        (vec![0], vec![0.1], vec![vec![e(0.5, &[0])]], vec![]),
        (vec![0], vec![0.1], vec![vec![e(1.5, &[1])]], vec![]),
        (vec![0], vec![0.1], vec![vec![e(0.5, &[2, 1])]], vec![]),
    ];
    for (heralds, probabilities, outcomes, base) in refused {
        assert!(
            HeraldedErrorModel::new(dem(base.clone()), heralds.clone(), probabilities, outcomes)
                .is_err(),
            "{heralds:?} {base:?}"
        );
    }
    let mut beyond = e(0.5, &[1]);
    beyond.observables = 2;
    assert!(HeraldedErrorModel::new(dem(vec![]), vec![0], vec![0.1], vec![vec![beyond]]).is_err());
}

#[test]
fn a_half_absorbs_exactly() {
    let mut rng = Rng::seed_from_u64(166);
    for k in 0..10_000 {
        let p = match k {
            0 => 0.0,
            1 => 1.0,
            _ => rng.uniform(),
        };
        assert_eq!(xor_probability(p, 0.5), 0.5);
        assert_eq!(xor_probability(0.5, p), 0.5);
    }
    assert_eq!(xor_probability(0.0, 0.25), 0.25);
}

// ---- format, builders, sampler ------------------------------------------

#[test]
fn heralded_operations_are_read_and_written_in_stims_format() {
    let mut c = DetectorCircuit::new(3).unwrap();
    c.h(0).unwrap();
    assert_eq!(
        c.heralded_pauli_channel_1([0.01, 0.02, 0.03, 0.04], &[0, 1])
            .unwrap(),
        vec![0, 1]
    );
    let m = c.measure(1, 0.0).unwrap();
    assert_eq!(m, 2);
    assert_eq!(c.heralded_erase(0.25, &[2, 2]).unwrap(), vec![3, 4]);
    c.detector(&[0]).unwrap();
    c.detector(&[1, 3]).unwrap();
    c.observable(0, &[4]).unwrap();
    assert_eq!(c.n_measurements(), 5);
    let text = c.to_stim().unwrap();
    assert!(
        text.contains("HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0 1\n"),
        "{text}"
    );
    assert!(text.contains("HERALDED_ERASE(0.25) 2 2\n"), "{text}");
    assert_eq!(DetectorCircuit::from_stim(&text).unwrap(), c);
    let r = RepeatedCircuit::from_circuit(&c).unwrap();
    assert_eq!(r.flatten().unwrap(), c);
    assert_eq!(
        RepeatedCircuit::from_stim(&r.to_stim().unwrap()).unwrap(),
        r
    );
    let rep = RepeatedCircuit::from_stim(REPEATED).unwrap();
    assert_eq!(
        RepeatedCircuit::from_stim(&rep.to_stim().unwrap()).unwrap(),
        rep
    );
    let flat = rep.flatten().unwrap();
    assert_eq!(
        DetectorCircuit::from_stim(&flat.to_stim().unwrap()).unwrap(),
        flat
    );
    assert_eq!(
        RepeatedCircuit::from_circuit(&flat)
            .unwrap()
            .flatten()
            .unwrap(),
        flat
    );

    // Accepted as by Stim.
    assert_eq!(
        circuit("HERALDED_ERASE(0.1) !0\nM 0"),
        circuit("HERALDED_ERASE(0.1) 0\nM 0")
    );
    let zero = circuit("HERALDED_ERASE(0) 0");
    assert_eq!((zero.n_measurements(), zero.n_steps()), (1, 1));
    assert_eq!(zero.to_stim().unwrap(), "HERALDED_ERASE(0.0) 0\n");
    assert_eq!(circuit("HERALDED_ERASE(-0) 0").n_measurements(), 1);
    let none = circuit("HERALDED_ERASE(0.1)\nH 0");
    assert_eq!((none.n_steps(), none.n_measurements()), (1, 0));
    assert_eq!(
        circuit("HERALDED_PAULI_CHANNEL_1(0.25, 0.25, 0.25, 0.2500000001) 0").n_measurements(),
        1
    );
    // Refused as by Stim (`tol.py`).
    for (bad, what) in [
        ("HERALDED_ERASE 0", "takes 1 probability"),
        ("HERALDED_ERASE(0.1, 0.2) 0", "takes 1 probability"),
        (
            "HERALDED_PAULI_CHANNEL_1(0.1, 0.1, 0.1) 0",
            "takes 4 probabilities",
        ),
        ("M 0\nHERALDED_ERASE(0.1) rec[-1]", "takes qubit targets"),
        ("HERALDED_ERASE(0.1) X0", "X0"),
        ("HERALDED_ERASE(1.0000000001) 0", "probability"),
        ("HERALDED_ERASE(-1e-300) 0", "probability"),
        ("HERALDED_ERASE(nan) 0", "nan"),
        ("HERALDED_ERASE(inf) 0", "inf"),
        (
            "HERALDED_PAULI_CHANNEL_1(0.25, 0.25, 0.25, 0.2500002) 0",
            "add up",
        ),
    ] {
        let err = DetectorCircuit::from_stim(bad).unwrap_err().to_string();
        assert!(err.contains(what), "{bad:?}: {err}");
        assert!(RepeatedCircuit::from_stim(bad).is_err(), "{bad:?}");
    }
    let mut c = DetectorCircuit::new(1).unwrap();
    assert!(c.heralded_erase(0.1, &[1]).is_err());
    assert!(c.heralded_erase(f64::NAN, &[0]).is_err());
    assert!(
        c.heralded_pauli_channel_1([0.5, 0.5, 0.1, 0.0], &[0])
            .is_err()
    );
    assert!(
        c.heralded_pauli_channel_1([-0.1, 0.5, 0.1, 0.0], &[0])
            .is_err()
    );
    assert_eq!(c.n_steps(), 0);
}

#[test]
fn circuits_list_their_extended_operations_in_order() {
    let mut c = DetectorCircuit::new(2).unwrap();
    c.heralded_erase(0.1, &[0]).unwrap();
    c.s(0).unwrap();
    c.noise(Noise::XError(0.1), &[1]).unwrap();
    c.heralded_pauli_channel_1([0.0, 0.1, 0.0, 0.0], &[1])
        .unwrap();
    c.heralded_erase(0.2, &[1]).unwrap();
    let m = c.measure(0, 0.01).unwrap();
    c.detector(&[0]).unwrap();
    c.detector(&[m]).unwrap();
    c.heralded_erase(0.3, &[0]).unwrap();
    c.detector(&[1]).unwrap();
    assert_eq!(c.operations().len(), 6);
    assert_eq!(c.n_steps(), 10);
    assert_eq!(c.steps().len(), 10);
    let kinds: String = c
        .steps()
        .map(|s| match s {
            CircuitStep::Extended(_) => 'E',
            CircuitStep::Qec(QecOp::Detector { .. }) => 'D',
            CircuitStep::Qec(_) => 'q',
        })
        .collect();
    assert_eq!(kinds, "EqqEEqDDED");
    let forward: Vec<_> = c.steps().collect();
    let mut backward: Vec<_> = c.steps().rev().collect();
    backward.reverse();
    assert_eq!(forward, backward);
    // Mixed ends.
    let mut it = c.steps();
    assert!(matches!(it.next(), Some(CircuitStep::Extended(_))));
    assert!(matches!(
        it.next_back(),
        Some(CircuitStep::Qec(QecOp::Detector { .. }))
    ));
    assert_eq!(it.len(), 8);
    assert_eq!(it.count(), 8);
    // Layers break at an extended operation.
    assert_eq!(c.detector_layers(), vec![0, 0, 1]);
    // Without noise: heralds kept at probability 0 in their places.
    let quiet = c.without_noise();
    assert_eq!(quiet.n_measurements(), c.n_measurements());
    assert_eq!(quiet.n_steps(), 9);
    let kinds: String = quiet
        .steps()
        .map(|s| match s {
            CircuitStep::Extended(ExtendedOp::HeraldedErase { probability, .. }) => {
                assert_eq!(*probability, 0.0);
                'E'
            }
            CircuitStep::Extended(ExtendedOp::HeraldedPauliChannel1 { probabilities, .. }) => {
                assert_eq!(*probabilities, [0.0; 4]);
                'P'
            }
            CircuitStep::Qec(QecOp::Detector { .. }) => 'D',
            CircuitStep::Qec(_) => 'q',
        })
        .collect();
    assert_eq!(kinds, "EqPEqDDED");
    assert_eq!(quiet.detector_layers(), c.detector_layers());
    assert!(
        detector_error_model(&quiet, false)
            .unwrap()
            .errors
            .is_empty()
    );
    // Noise of probability 0 is still left out.
    let mut d = DetectorCircuit::new(1).unwrap();
    d.noise(Noise::XError(0.0), &[0]).unwrap();
    assert_eq!(d.n_steps(), 0);
    // Explanations refuse heralded circuits.
    assert!(explain_errors(&c, &[]).is_err());
}

/// The rate `count / shots` is within `z` standard errors of `p`.
fn within(count: usize, shots: usize, p: f64, z: f64) -> bool {
    let rate = count as f64 / shots as f64;
    let sigma = (p * (1.0 - p) / shots as f64).sqrt();
    (rate - p).abs() <= z * sigma.max(1e-12)
}

#[test]
fn the_sampler_draws_herald_and_pauli_together() {
    let shots = 1_000_000;
    let both = |text: &str, seed: u64| -> [usize; 4] {
        let s = sample_detectors(&circuit(text), shots, seed).unwrap();
        let mut counts = [0; 4];
        for shot in 0..shots {
            let k =
                s.fired(0, shot) as usize + 2 * (s.n_detectors() > 1 && s.fired(1, shot)) as usize;
            counts[k] += 1;
        }
        counts
    };
    let rate = both("HERALDED_ERASE(0.2) 0\nDETECTOR rec[-1]", 1);
    assert!(within(rate[1], shots, 0.2, 5.0), "{rate:?}");
    let x = both("X 0\nHERALDED_ERASE(1) 0\nM 0\nDETECTOR rec[-1]", 2);
    assert!(within(x[1], shots, 0.5, 5.0), "{x:?}");
    // Herald (bit 0) and measurement (bit 1): X and Y flip it, I and Z
    // not; never a flip without the herald.
    let c = both(
        "HERALDED_PAULI_CHANNEL_1(0.05, 0.1, 0.15, 0.2) 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
        3,
    );
    assert_eq!(c[2], 0);
    assert!(
        within(c[3], shots, 0.25, 5.0) && within(c[1], shots, 0.25, 5.0),
        "{c:?}"
    );
    let h = both(
        "H 0\nHERALDED_PAULI_CHANNEL_1(0.05, 0.1, 0.15, 0.2) 0\nH 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
        4,
    );
    assert_eq!(h[2], 0);
    assert!(
        within(h[3], shots, 0.35, 5.0) && within(h[1], shots, 0.15, 5.0),
        "{h:?}"
    );
    let a = both(A, 5);
    assert_eq!(a[2], 0);
    assert!(within(a[3], shots, 0.05, 5.0), "{a:?}");
    // The same seed, the same samples; repeated circuits agree.
    let c = circuit(A);
    assert_eq!(
        sample_detectors(&c, 5000, 9).unwrap(),
        sample_detectors(&c, 5000, 9).unwrap()
    );
    let r = RepeatedCircuit::from_stim(REPEATED).unwrap();
    assert_eq!(
        r.sample(3000, 7).unwrap(),
        sample_detectors(&r.flatten().unwrap(), 3000, 7).unwrap()
    );
    // Quiet runs leave every herald at 0: herald detectors are
    // deterministic.
    assert!(crate::qec::frame::check_deterministic(&r).is_ok());
}

// ---- erasure-biased memory experiments ----------------------------------

#[test]
fn erasure_noise_at_share_zero_is_circuit_noise() {
    let noise = CircuitNoise::uniform(0.01);
    let zero = ErasureNoise::uniform(0.01, 0.0).unwrap();
    assert_eq!(zero, ErasureNoise::new(noise, 0.0).unwrap());
    assert!(ErasureNoise::uniform(0.01, 1.5).is_err());
    assert!(ErasureNoise::uniform(0.01, f64::NAN).is_err());
    assert!(ErasureNoise::uniform(0.01, -0.1).is_err());
    let colour = CssCode::color_code(3).unwrap();
    for rounds in [1, 3] {
        assert_eq!(
            repetition_code_memory_erasure(3, rounds, &zero).unwrap(),
            repetition_code_memory(3, rounds, &noise).unwrap()
        );
        assert_eq!(
            repetition_code_memory_erasure_repeated(3, rounds, &zero).unwrap(),
            repetition_code_memory_repeated(3, rounds, &noise).unwrap()
        );
        for basis in [MemoryBasis::Z, MemoryBasis::X] {
            assert_eq!(
                surface_code_memory_erasure(3, rounds, &zero, basis).unwrap(),
                surface_code_memory_with(3, rounds, &noise, basis).unwrap()
            );
            assert_eq!(
                surface_code_memory_erasure_repeated(3, rounds, &zero, basis).unwrap(),
                surface_code_memory_repeated(3, rounds, &noise, basis).unwrap()
            );
        }
        for schedule in [
            CssSchedule::Greedy,
            CssSchedule::Directions([0, 1, 2, 3, 4, 5]),
        ] {
            assert_eq!(
                css_code_memory_erasure(&colour, rounds, &zero, MemoryBasis::Z, schedule).unwrap(),
                css_code_memory_with(&colour, rounds, &noise, MemoryBasis::Z, schedule).unwrap()
            );
        }
    }
}

#[test]
fn erasure_biased_memories_declare_their_heralds_round_by_round() {
    let half = ErasureNoise::uniform(0.01, 0.5).unwrap();
    // Flattening the repeated forms gives the flat circuits.
    assert_eq!(
        repetition_code_memory_erasure_repeated(3, 4, &half)
            .unwrap()
            .flatten()
            .unwrap(),
        repetition_code_memory_erasure(3, 4, &half).unwrap()
    );
    for basis in [MemoryBasis::Z, MemoryBasis::X] {
        assert_eq!(
            surface_code_memory_erasure_repeated(3, 4, &half, basis)
                .unwrap()
                .flatten()
                .unwrap(),
            surface_code_memory_erasure(3, 4, &half, basis).unwrap()
        );
    }
    let colour = CssCode::color_code(3).unwrap();
    let (bb, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    for (code, schedule) in [(&colour, CssSchedule::Greedy), (&bb, CssSchedule::Bicycle)] {
        let flat = css_code_memory_erasure(code, 3, &half, MemoryBasis::Z, schedule).unwrap();
        assert_eq!(
            css_code_memory_erasure_repeated(code, 3, &half, MemoryBasis::Z, schedule)
                .unwrap()
                .flatten()
                .unwrap(),
            flat
        );
        let m = heralded_error_model(&flat, false).unwrap();
        assert!(!m.herald_detectors().is_empty());
    }

    // Repetition, d = 3, r = 3: heralds (2d − 1) + r(7d − 6) + d.
    let c = repetition_code_memory_erasure(3, 3, &half).unwrap();
    assert_eq!((c.n_detectors(), c.n_measurements()), (61, 62));
    let m = heralded_error_model(&c, true).unwrap();
    assert_eq!(m.herald_detectors().len(), 53);
    let layers = c.detector_layers();
    assert_eq!(layers.iter().max(), Some(&3));
    let plain = repetition_code_memory(3, 3, &CircuitNoise::uniform(0.01)).unwrap();
    assert_eq!(plain.detector_layers().iter().max(), Some(&3));
    // The repeated form folds to the same heralded model.
    let r = repetition_code_memory_erasure_repeated(3, 3, &half).unwrap();
    assert_same_heralded(&r.heralded_error_model(true).unwrap(), &m, 1e-12);
    assert_same_heralded(
        &r.heralded_error_model(false).unwrap(),
        &heralded_error_model(&c, false).unwrap(),
        1e-12,
    );

    // Surface, d = 3, r = 3: 17 + 3·81 + 9 heralds in the Z basis.
    let z = surface_code_memory_erasure(3, 3, &half, MemoryBasis::Z).unwrap();
    assert_eq!((z.n_detectors(), z.n_measurements()), (285, 302));
    assert_eq!(
        heralded_error_model(&z, true)
            .unwrap()
            .herald_detectors()
            .len(),
        269
    );
    let x = surface_code_memory_erasure(3, 3, &half, MemoryBasis::X).unwrap();
    assert_eq!(
        heralded_error_model(&x, true)
            .unwrap()
            .herald_detectors()
            .len(),
        287
    );
    let rz = surface_code_memory_erasure_repeated(3, 5, &half, MemoryBasis::Z).unwrap();
    assert_same_heralded(
        &rz.heralded_error_model(true).unwrap(),
        &heralded_error_model(&rz.flatten().unwrap(), true).unwrap(),
        1e-12,
    );

    // Share 1: no Pauli noise remains.
    let all = ErasureNoise::uniform(0.01, 1.0).unwrap();
    let c = surface_code_memory_erasure(3, 2, &all, MemoryBasis::X).unwrap();
    for step in c.steps() {
        match step {
            CircuitStep::Qec(QecOp::Noise { .. }) => panic!("noise left"),
            CircuitStep::Qec(QecOp::Measure { flip, .. }) => assert_eq!(*flip, 0.0),
            _ => {}
        }
    }
    let m = heralded_error_model(&c, false).unwrap();
    assert!(m.base().errors.is_empty());
    // Every herald detector reads one herald, and owns it.
    let heralds = c
        .steps()
        .filter_map(|s| match s {
            CircuitStep::Extended(op) => Some(op.n_records()),
            _ => None,
        })
        .sum::<usize>();
    assert_eq!(m.herald_detectors().len(), heralds);
}

// ---- decoding with heralds ---------------------------------------------

/// The repetition code at code capacity: `d` data qubits, each erased with
/// probability `p` (herald detectors `0..d`), perfect measurements,
/// detectors `Z_i Z_{i+1}` (`d..2d − 1`), observable data 0.
fn repetition_capacity(d: usize, p: f64) -> DetectorCircuit {
    let mut c = DetectorCircuit::new(d).unwrap();
    let data: Vec<usize> = (0..d).collect();
    for r in c.heralded_erase(p, &data).unwrap() {
        c.detector(&[r]).unwrap();
    }
    let m: Vec<usize> = data.iter().map(|&q| c.measure(q, 0.0).unwrap()).collect();
    for i in 0..d - 1 {
        c.detector(&[m[i], m[i + 1]]).unwrap();
    }
    c.observable(0, &[m[0]]).unwrap();
    c
}

fn methods() -> Vec<HeraldedMethod> {
    let osd = |osd: OsdMethod| {
        HeraldedMethod::BpOsd(BpOsdOptions {
            osd,
            ..BpOsdOptions::default()
        })
    };
    vec![
        HeraldedMethod::UnionFind,
        HeraldedMethod::Matching,
        osd(OsdMethod::Zero),
        osd(OsdMethod::CombinationSweep),
    ]
}

#[test]
fn the_repetition_code_with_heralds_fails_with_probability_p_to_the_d_over_two() {
    for d in [3, 5, 7] {
        for p in [0.1, 0.5, 0.9] {
            let c = repetition_capacity(d, p);
            let model = heralded_error_model(&c, true).unwrap();
            assert_eq!(model.herald_detectors(), (0..d).collect::<Vec<_>>());
            for method in methods() {
                let mut dec = ConditionedDecoder::new(&model, method).unwrap();
                let mut failure = 0.0;
                for erased in 0u32..1 << d {
                    let k = erased.count_ones() as i32;
                    let weight = p.powi(k) * (1.0 - p).powi(d as i32 - k) / 2f64.powi(k);
                    let heralds: Vec<usize> = (0..d).filter(|&i| erased >> i & 1 == 1).collect();
                    let mut failing = 0;
                    // Every X pattern on the erased qubits.
                    for x in 0u32..1 << d {
                        if x & !erased != 0 {
                            continue;
                        }
                        let fired: Vec<usize> = (0..d - 1)
                            .filter(|&i| (x >> i ^ x >> (i + 1)) & 1 == 1)
                            .map(|i| d + i)
                            .collect();
                        let predicted = dec.decode_heralded(&fired, &heralds).unwrap();
                        if predicted != u64::from(x & 1) {
                            failure += weight;
                            failing += 1;
                        }
                    }
                    if erased == (1 << d) - 1 {
                        assert_eq!(failing, 1 << (d - 1), "{method:?}");
                    } else {
                        assert_eq!(failing, 0, "{method:?} {erased:b}");
                    }
                }
                let want = p.powi(d as i32) / 2.0;
                assert!(
                    close(failure, want, 1e-12),
                    "{d} {p} {method:?}: {failure} {want}"
                );
            }
        }
    }
}

/// The checks of the rotated surface code of distance `d` (data `(r, c)` as
/// `r·d + c`), as `build_surface` lays them out: (X checks, Z checks).
fn surface_checks(d: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let d = d as i64;
    let (mut xs, mut zs) = (Vec::new(), Vec::new());
    for i in -1..d {
        for j in -1..d {
            let is_x = (i + j).rem_euclid(2) == 0;
            let interior = (0..d - 1).contains(&i) && (0..d - 1).contains(&j);
            let top_bottom = (i == -1 || i == d - 1) && (0..d - 1).contains(&j) && is_x;
            let left_right = (j == -1 || j == d - 1) && (0..d - 1).contains(&i) && !is_x;
            if interior || top_bottom || left_right {
                let mut support = Vec::new();
                for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let (r, c) = (i + dr, j + dc);
                    if (0..d).contains(&r) && (0..d).contains(&c) {
                        support.push((r * d + c) as usize);
                    }
                }
                if is_x {
                    xs.push(support);
                } else {
                    zs.push(support);
                }
            }
        }
    }
    (xs, zs)
}

/// The rotated surface code at code capacity, Z memory: each data qubit
/// erased with probability `p` (herald detectors `0..d²`), measured
/// perfectly; detectors on the Z checks, observable Z on the first row.
fn surface_capacity(d: usize, p: f64) -> DetectorCircuit {
    let n = d * d;
    let (_, zs) = surface_checks(d);
    let mut c = DetectorCircuit::new(n).unwrap();
    let data: Vec<usize> = (0..n).collect();
    for r in c.heralded_erase(p, &data).unwrap() {
        c.detector(&[r]).unwrap();
    }
    let m: Vec<usize> = data.iter().map(|&q| c.measure(q, 0.0).unwrap()).collect();
    for z in &zs {
        c.detector(&z.iter().map(|&q| m[q]).collect::<Vec<_>>())
            .unwrap();
    }
    c.observable(0, &(0..d).map(|col| m[col]).collect::<Vec<_>>())
        .unwrap();
    c
}

fn mask(qubits: &[usize]) -> u128 {
    qubits.iter().fold(0, |m, &q| m | 1 << q)
}

/// Whether `v` is a sum of `rows` over GF(2).
fn in_span(rows: &[u128], v: u128) -> bool {
    let mut basis: Vec<u128> = Vec::new();
    let reduce = |basis: &[u128], mut x: u128| {
        for &b in basis {
            x = x.min(x ^ b);
        }
        x
    };
    for &r in rows {
        let x = reduce(&basis, r);
        if x != 0 {
            basis.push(x);
            basis.sort_unstable_by(|a, b| b.cmp(a));
        }
    }
    reduce(&basis, v) == 0
}

/// Whether the qubits `set` support a logical operator: one that commutes
/// with `checks` and anticommutes with `logical`.
fn supports(set: u128, checks: &[Vec<usize>], logical: &[usize]) -> bool {
    let rows: Vec<u128> = checks.iter().map(|c| mask(c) & set).collect();
    !in_span(&rows, mask(logical) & set)
}

#[test]
fn the_surface_code_with_heralds_at_one_half_fails_exactly_a_quarter() {
    let d = 3;
    let n = d * d;
    let (xs, zs) = surface_checks(d);
    let z_logical: Vec<usize> = (0..d).collect();
    let x_logical: Vec<usize> = (0..d).map(|r| r * d).collect();
    // The cleaning lemma: a set supports X_L exactly when its complement
    // does not support Z_L; 256 of the 512 sets do.
    let all = (1u128 << n) - 1;
    let supporting: Vec<bool> = (0..1u128 << n)
        .map(|e| supports(e, &zs, &z_logical))
        .collect();
    assert_eq!(supporting.iter().filter(|&&s| s).count(), 256);
    for e in 0..1u128 << n {
        assert_eq!(supporting[e as usize], !supports(all & !e, &xs, &x_logical));
    }
    let c = surface_capacity(d, 0.5);
    let model = heralded_error_model(&c, true).unwrap();
    assert_eq!(model.herald_detectors().len(), n);
    for method in methods() {
        let mut dec = ConditionedDecoder::new(&model, method).unwrap();
        let mut total: u64 = 0;
        for e in 0u32..1 << n {
            let heralds: Vec<usize> = (0..n).filter(|&q| e >> q & 1 == 1).collect();
            let mut failing = 0u64;
            for x in 0u32..1 << n {
                if x & !e != 0 {
                    continue;
                }
                let fired: Vec<usize> = zs
                    .iter()
                    .enumerate()
                    .filter(|(_, z)| z.iter().filter(|&&q| x >> q & 1 == 1).count() % 2 == 1)
                    .map(|(k, _)| n + k)
                    .collect();
                let flip = (0..d).filter(|&q| x >> q & 1 == 1).count() as u64 % 2;
                if dec.decode_heralded(&fired, &heralds).unwrap() != flip {
                    failing += 1;
                }
            }
            let size = e.count_ones();
            if supporting[e as usize] {
                assert_eq!(failing, 1 << (size - 1), "{method:?} {e:b}");
            } else {
                assert_eq!(failing, 0, "{method:?} {e:b}");
            }
            total += failing << (n as u32 - size);
        }
        assert_eq!(total, 65_536, "{method:?}");
    }
}

/// Code capacity at p = ½ sampled: |rate − ¼| within 5σ, no failure on
/// shots whose erased set supports no logical, half of the others, and
/// matching corrections of weight 0.
fn check_sampled_capacity(d: usize, shots: usize, seed: u64) {
    let (_, zs) = surface_checks(d);
    let z_logical: Vec<usize> = (0..d).collect();
    let c = surface_capacity(d, 0.5);
    let model = heralded_error_model(&c, true).unwrap();
    let samples = sample_detectors(&c, shots, seed).unwrap();
    for method in [HeraldedMethod::UnionFind, HeraldedMethod::Matching] {
        let mut dec = ConditionedDecoder::new(&model, method).unwrap();
        let (mut failures, mut supporting, mut failing_supporting) = (0, 0, 0);
        for (s, fired) in samples.fired_by_shot().into_iter().enumerate() {
            let (ordinary, heralds) = dec.split(&fired).unwrap();
            let set = mask(&heralds);
            let predicted = if method == HeraldedMethod::Matching {
                let (p, w) = dec.decode_with_weight(&ordinary, &heralds).unwrap();
                assert_eq!(w, 0.0);
                p
            } else {
                Decoder::decode(&mut dec, &fired).unwrap()
            };
            let fail = predicted != samples.observable_flips(s);
            failures += fail as usize;
            if supports(set, &zs, &z_logical) {
                supporting += 1;
                failing_supporting += fail as usize;
            } else {
                assert!(!fail, "d = {d}, shot {s}");
            }
        }
        assert!(
            within(failures, shots, 0.25, 5.0),
            "{d} {method:?}: {failures}"
        );
        assert!(within(failing_supporting, supporting, 0.5, 5.0));
    }
}

#[test]
fn sampled_surface_code_capacity_with_heralds_fails_a_quarter() {
    check_sampled_capacity(5, 20_000, 51);
}

/// The ordinary detectors' compact index of each detector.
fn compact_of(model: &HeraldedErrorModel) -> Vec<usize> {
    let mut out = vec![usize::MAX; model.n_detectors()];
    let mut k = 0;
    for (d, o) in out.iter_mut().enumerate() {
        if model.herald_detectors().binary_search(&d).is_err() {
            *o = k;
            k += 1;
        }
    }
    out
}

#[test]
fn under_pure_erasure_decoders_fail_only_on_logical_supporting_sets() {
    let c = surface_code_memory_erasure(
        3,
        3,
        &ErasureNoise::uniform(0.04, 1.0).unwrap(),
        MemoryBasis::Z,
    )
    .unwrap();
    let shots = 5000;
    let samples = sample_detectors(&c, shots, 166).unwrap();
    for method in [
        HeraldedMethod::UnionFind,
        HeraldedMethod::Matching,
        HeraldedMethod::BpOsd(BpOsdOptions::default()),
    ] {
        let decompose = !matches!(method, HeraldedMethod::BpOsd(_));
        let model = heralded_error_model(&c, decompose).unwrap();
        assert!(model.base().errors.is_empty());
        let compact = compact_of(&model);
        let m = compact.iter().filter(|&&k| k != usize::MAX).count();
        assert!(m < 127);
        let mut dec = ConditionedDecoder::new(&model, method).unwrap();
        let (mut supporting, mut failing) = (0, 0);
        for (s, fired) in samples.fired_by_shot().into_iter().enumerate() {
            let (_, heralds) = dec.split(&fired).unwrap();
            // The parts of the fired heralds as (syndrome, observable).
            let mut rows = Vec::new();
            for d in &heralds {
                let h = model.herald_detectors().binary_search(d).unwrap();
                for part in model.outcomes(h) {
                    let mut v = u128::from(part.observables & 1) << 127;
                    for &x in &part.detectors {
                        v ^= 1 << compact[x];
                    }
                    rows.push(v);
                }
            }
            let fail = Decoder::decode(&mut dec, &fired).unwrap() != samples.observable_flips(s);
            if in_span(&rows, 1 << 127) {
                supporting += 1;
                failing += fail as usize;
            } else {
                assert!(!fail, "{method:?} shot {s}");
            }
        }
        assert!(supporting > 10, "{supporting}");
        assert!(
            within(failing, supporting, 0.5, 5.0),
            "{method:?}: {failing} of {supporting}"
        );
    }
}

/// The conditioned model of a shot in the decoder's compact numbering, its
/// errors the decoded slots in slot order.
fn conditioned_slots(
    model: &HeraldedErrorModel,
    dec: &ConditionedDecoder,
    heralds: &[usize],
) -> DetectorErrorModel {
    let conditioned: HashMap<Signature, f64> = model
        .conditioned(heralds)
        .unwrap()
        .errors
        .into_iter()
        .map(|e| ((e.detectors, e.observables), e.probability))
        .collect();
    let compact = compact_of(model);
    let errors = dec
        .slots()
        .iter()
        .filter(|e| !e.detectors.is_empty())
        .map(|e| {
            let p = conditioned
                .get(&(e.detectors.clone(), e.observables))
                .copied()
                .unwrap_or(0.0);
            ErrorMechanism {
                probability: p,
                detectors: e.detectors.iter().map(|&d| compact[d]).collect(),
                observables: e.observables,
            }
        })
        .collect();
    DetectorErrorModel {
        n_detectors: compact.iter().filter(|&&k| k != usize::MAX).count(),
        n_observables: model.n_observables(),
        errors,
    }
}

#[test]
fn conditioning_sets_and_restores_exactly() {
    let c = surface_code_memory_erasure(
        3,
        3,
        &ErasureNoise::uniform(0.01, 0.9).unwrap(),
        MemoryBasis::Z,
    )
    .unwrap();
    let samples = sample_detectors(&c, 2000, 61).unwrap();
    let shots = samples.fired_by_shot();
    let graphlike = heralded_error_model(&c, true).unwrap();
    let whole = heralded_error_model(&c, false).unwrap();

    // Matching: a fresh decoder of the conditioned slots at the same scale.
    let mut dec = ConditionedDecoder::new(&graphlike, HeraldedMethod::Matching).unwrap();
    let scale = match &dec.inner {
        Inner::Matching(d) => d.scale,
        _ => unreachable!(),
    };
    let start = dec.inner_state();
    let mut heralded_shots = 0;
    for fired in &shots {
        let (ordinary, heralds) = dec.split(fired).unwrap();
        heralded_shots += !heralds.is_empty() as usize;
        let got = dec.decode_with_weight(&ordinary, &heralds).unwrap();
        let compact = compact_of(&graphlike);
        let events: Vec<usize> = ordinary.iter().map(|&d| compact[d]).collect();
        let mut fresh = SparseMatchingDecoder::with_scale(
            MatchingGraph::from_slots(&conditioned_slots(&graphlike, &dec, &heralds)).unwrap(),
            scale,
        );
        let want = if events.is_empty() {
            (0, 0.0)
        } else {
            fresh.decode_with_weight(&events).unwrap()
        };
        assert_eq!(got, want);
    }
    assert!(heralded_shots > 1000, "{heralded_shots}");
    assert_eq!(dec.inner_state(), start);

    // BP+OSD: a fresh decoder of the conditioned columns.
    for options in [
        BpOsdOptions::default(),
        BpOsdOptions {
            osd: OsdMethod::Zero,
            method: BpMethod::ProductSum,
            ..BpOsdOptions::default()
        },
    ] {
        let mut dec = ConditionedDecoder::new(&whole, HeraldedMethod::BpOsd(options)).unwrap();
        let start = dec.inner_state();
        let compact = compact_of(&whole);
        let decoded: Vec<usize> = dec
            .slots()
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.detectors.is_empty())
            .map(|(k, _)| k)
            .collect();
        for fired in shots.iter().take(500) {
            let (ordinary, heralds) = dec.split(fired).unwrap();
            let got = dec.decode_detailed(&ordinary, &heralds).unwrap();
            let events: Vec<usize> = ordinary.iter().map(|&d| compact[d]).collect();
            let mut fresh =
                BpOsdDecoder::with_columns(&conditioned_slots(&whole, &dec, &heralds), options)
                    .unwrap();
            let mut want = fresh.decode_detailed(&events).unwrap();
            for e in &mut want.errors {
                *e = decoded[*e];
            }
            want.errors.sort_unstable();
            assert_eq!(got, want);
        }
        assert_eq!(dec.inner_state(), start);
    }

    // Union-find: without heralds, the decoder of the base model.
    let mut dec = ConditionedDecoder::new(&graphlike, HeraldedMethod::UnionFind).unwrap();
    let start = dec.inner_state();
    let compact = compact_of(&graphlike);
    let base = DetectorErrorModel {
        n_detectors: compact.iter().filter(|&&k| k != usize::MAX).count(),
        n_observables: 1,
        errors: graphlike
            .base()
            .errors
            .iter()
            .map(|e| ErrorMechanism {
                probability: e.probability,
                detectors: e.detectors.iter().map(|&d| compact[d]).collect(),
                observables: e.observables,
            })
            .collect(),
    };
    let mut plain = UnionFindDecoder::new(MatchingGraph::from_dem(&base).unwrap());
    let mut compared = 0;
    for fired in &shots {
        let (ordinary, heralds) = dec.split(fired).unwrap();
        let got = dec.decode_heralded(&ordinary, &heralds).unwrap();
        if heralds.is_empty() && !ordinary.is_empty() {
            let events: Vec<usize> = ordinary.iter().map(|&d| compact[d]).collect();
            assert_eq!(got, plain.decode(&events).unwrap());
            compared += 1;
        }
    }
    assert!(compared > 10, "{compared}");
    assert_eq!(dec.inner_state(), start);
}

#[test]
fn circuits_with_herald_detectors_are_routed_to_the_conditioned_decoder() {
    let c = surface_code_memory_erasure(
        3,
        3,
        &ErasureNoise::uniform(0.01, 0.9).unwrap(),
        MemoryBasis::Z,
    )
    .unwrap();
    for (choice, method, decompose) in [
        (DecoderChoice::UnionFind, HeraldedMethod::UnionFind, true),
        (DecoderChoice::Matching, HeraldedMethod::Matching, true),
        (
            DecoderChoice::BpOsd,
            HeraldedMethod::BpOsd(BpOsdOptions::default()),
            false,
        ),
    ] {
        let r = logical_error_rate_using(&c, 2000, 3, choice).unwrap();
        assert!(r.rate < 0.1, "{choice:?}: {r:?}");
        // The same decoder built by hand, on the same shots.
        let model = heralded_error_model(&c, decompose).unwrap();
        let mut dec = ConditionedDecoder::new(&model, method).unwrap();
        assert_eq!(
            logical_error_rate_decoding(&c, 2000, 3, &mut dec).unwrap(),
            r
        );
    }
    let r = surface_code_memory_erasure_repeated(
        3,
        3,
        &ErasureNoise::uniform(0.01, 0.9).unwrap(),
        MemoryBasis::Z,
    )
    .unwrap();
    let folded =
        crate::qec::logical_error_rate_repeated(&r, 2000, 3, DecoderChoice::Matching).unwrap();
    assert!(folded.rate < 0.1, "{folded:?}");
    let mut dec = ConditionedDecoder::new(
        &r.heralded_error_model(true).unwrap(),
        HeraldedMethod::Matching,
    )
    .unwrap();
    assert_eq!(
        crate::qec::logical_error_rate_repeated_decoding(&r, 2000, 3, &mut dec).unwrap(),
        folded
    );
    for choice in [
        DecoderChoice::CorrelatedMatching,
        DecoderChoice::WindowMatching {
            window: 2,
            commit: 1,
        },
        DecoderChoice::WindowBpOsd {
            window: 2,
            commit: 1,
        },
        DecoderChoice::ParallelWindowMatching {
            commit: 1,
            buffer: 1,
        },
        DecoderChoice::ParallelWindowBpOsd {
            commit: 1,
            buffer: 1,
        },
    ] {
        let err = choice.build_for(&c).err().expect("refused").to_string();
        assert!(err.contains("does not use heralds yet"), "{err}");
        // The decoder is named as a user names it, not as a Rust value.
        assert!(!err.contains('{') && !err.contains("Window"), "{err}");
        assert!(choice.build_for_repeated(&r).is_err());
    }
    // Heralds no detector owns: today's decoders, on the full model.
    let mut unread = repetition_code_memory(3, 2, &CircuitNoise::uniform(0.01)).unwrap();
    unread.heralded_erase(0.05, &[0, 1, 2]).unwrap();
    unread.measure(0, 0.0).unwrap();
    assert!(
        heralded_error_model(&unread, true)
            .unwrap()
            .herald_detectors()
            .is_empty()
    );
    for choice in [DecoderChoice::CorrelatedMatching, DecoderChoice::UnionFind] {
        assert!(logical_error_rate_using(&unread, 500, 1, choice).is_ok());
    }
}

#[test]
fn heralded_circuits_are_refused_where_errors_must_be_independent() {
    let c = repetition_capacity(3, 0.1);
    let err = enumerated_failure_rate(&c, DecoderChoice::UnionFind, 2)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("heralded"), "{err}");
    assert!(leading_failures(&c, DecoderChoice::UnionFind, 1000, 1000).is_err());
    let at = |p: f64| Ok::<_, crate::error::QsoError>(repetition_capacity(3, p));
    let err = splitting_failure_rate(
        &at,
        DecoderChoice::UnionFind,
        0.1,
        0.01,
        &SplittingOptions::default(),
    )
    .err()
    .unwrap()
    .to_string();
    assert!(err.contains("heralded"), "{err}");
    let err = explain_errors(&c, &[]).err().unwrap().to_string();
    assert!(err.contains("heralded"), "{err}");
    let wide = heralded_error_model(&circuit(WIDE), true).unwrap();
    for method in [HeraldedMethod::Matching, HeraldedMethod::UnionFind] {
        let err = ConditionedDecoder::new(&wide, method)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("flips 3 detectors"), "{err}");
    }
    assert!(ConditionedDecoder::new(&wide, HeraldedMethod::BpOsd(BpOsdOptions::default())).is_ok());
    assert!(
        ConditionedDecoder::new(
            &wide,
            HeraldedMethod::BpOsd(BpOsdOptions {
                scaling: 2.0,
                ..BpOsdOptions::default()
            })
        )
        .is_err()
    );
}

#[test]
fn conditioned_decoders_check_their_input() {
    let c = repetition_capacity(3, 0.1);
    let model = heralded_error_model(&c, true).unwrap();
    let mut uf = ConditionedDecoder::new(&model, HeraldedMethod::UnionFind).unwrap();
    assert_eq!(uf.method(), HeraldedMethod::UnionFind);
    assert_eq!(
        (uf.n_detectors(), uf.herald_detectors()),
        (5, &[0, 1, 2][..])
    );
    assert_eq!(uf.slots().len(), 3);
    assert!(uf.slots().iter().all(|e| e.probability == 0.0));
    assert!(uf.decode_heralded(&[0], &[]).is_err());
    assert!(uf.decode_heralded(&[3], &[3]).is_err());
    assert!(uf.decode_heralded(&[5], &[]).is_err());
    assert!(uf.decode_heralded(&[3], &[7]).is_err());
    assert!(uf.decode_with_weight(&[3], &[0]).is_err());
    assert!(uf.decode_detailed(&[3], &[0]).is_err());
    assert!(Decoder::decode(&mut uf, &[9]).is_err());
    // Nothing explains a syndrome without heralds.
    assert!(uf.decode_heralded(&[3], &[]).is_err());
    // No ordinary detector fired: no flip, whatever the heralds.
    assert_eq!(uf.decode_heralded(&[], &[0, 1]).unwrap(), 0);
    assert_eq!(uf.decode_heralded(&[3, 3], &[0]).unwrap(), 0);
    // Listed twice, a herald cancels.
    assert!(uf.decode_heralded(&[3], &[0, 0]).is_err());
    assert_eq!(uf.decode_heralded(&[3], &[0, 0, 0]).unwrap(), 1);
    assert_eq!(Decoder::decode(&mut uf, &[3, 0]).unwrap(), 1);
    let mut m = ConditionedDecoder::new(&model, HeraldedMethod::Matching).unwrap();
    assert!(m.decode_detailed(&[3], &[0]).is_err());
    assert_eq!(m.decode_with_weight(&[3], &[0]).unwrap(), (1, 0.0));
    let mut b =
        ConditionedDecoder::new(&model, HeraldedMethod::BpOsd(BpOsdOptions::default())).unwrap();
    assert!(b.decode_with_weight(&[3], &[0]).is_err());
    let outcome = b.decode_detailed(&[3], &[0]).unwrap();
    assert_eq!((outcome.observables, &outcome.errors[..]), (1, &[0][..]));
}

/// A herald whose X and Z parts have the same effect and probability 1
/// (a pure Y before a Y-basis readout) has a part of probability 0; it
/// must not set the scale of the matching weights (it never happens).
#[test]
fn a_part_that_never_happens_leaves_the_matching_weights_alone() {
    let c = circuit(
        "RY 0 1\nX_ERROR(0.01) 1\nHERALDED_PAULI_CHANNEL_1(0, 0, 0.1, 0) 0\nDETECTOR rec[-1]\nMY 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    );
    let model = heralded_error_model(&c, true).unwrap();
    assert_eq!(model.herald_detectors(), &[0]);
    assert_errors(
        &DetectorErrorModel {
            n_detectors: 3,
            n_observables: 1,
            errors: model.outcomes(0).to_vec(),
        },
        &[(&[1], 0, 0.0)],
        0.0,
    );
    let mut m = ConditionedDecoder::new(&model, HeraldedMethod::Matching).unwrap();
    let length = (0.99f64 / 0.01).ln();
    for heralds in [&[][..], &[0][..]] {
        let (p, w) = m.decode_with_weight(&[2], heralds).unwrap();
        assert_eq!(p, 1);
        assert!(close(w, length, 1e-6), "{w} {length}");
    }
}

/// A repetition code at code capacity whose `d` data qubits each take
/// `X_ERROR(base)` and `HERALDED_PAULI_CHANNEL_1(pauli)` (herald detectors
/// `0..d`, `Z_i Z_{i+1}` as `d..2d − 1`, observable data 0).
fn pauli_capacity(d: usize, base: f64, pauli: &str) -> DetectorCircuit {
    let q: Vec<String> = (0..d).map(|k| k.to_string()).collect();
    let q = q.join(" ");
    let mut text = format!("R {q}\nX_ERROR({base}) {q}\nHERALDED_PAULI_CHANNEL_1({pauli}) {q}\n");
    for k in 0..d {
        text += &format!("DETECTOR rec[-{}]\n", d - k);
    }
    text += &format!("M {q}\n");
    for k in 0..d - 1 {
        text += &format!("DETECTOR rec[-{}] rec[-{}]\n", d - k, d - k - 1);
    }
    text += &format!("OBSERVABLE_INCLUDE(0) rec[-{d}]\n");
    circuit(&text)
}

/// A part more likely than not given its herald (X part (pX + pY)/t > ½)
/// is applied before decoding, as PyMatching does with a negative weight:
/// with `HERALDED_PAULI_CHANNEL_1(0, 0.5, 0, 0)` a fired herald means a
/// certain X, so no method may fail, though the syndrome of every qubit
/// heralded is empty (the review found 3 % of failures at d = 5).
#[test]
fn parts_above_one_half_are_applied_before_decoding() {
    let d = 5;
    let c = pauli_capacity(d, 0.0, "0, 0.5, 0, 0");
    let model = heralded_error_model(&c, true).unwrap();
    assert_errors(
        &DetectorErrorModel {
            n_detectors: c.n_detectors(),
            n_observables: 1,
            errors: model.outcomes(0).to_vec(),
        },
        &[(&[d], 1, 1.0)],
        0.0,
    );
    for method in methods() {
        let mut dec = ConditionedDecoder::new(&model, method).unwrap();
        for x in 0u32..1 << d {
            let heralds: Vec<usize> = (0..d).filter(|&i| x >> i & 1 == 1).collect();
            let fired: Vec<usize> = (0..d - 1)
                .filter(|&i| (x >> i ^ x >> (i + 1)) & 1 == 1)
                .map(|i| d + i)
                .collect();
            let predicted = dec.decode_heralded(&fired, &heralds).unwrap();
            assert_eq!(predicted, u64::from(x & 1), "{method:?} {x:b}");
            // As one list of every fired detector.
            let all: Vec<usize> = heralds.iter().chain(&fired).copied().collect();
            assert_eq!(Decoder::decode(&mut dec, &all).unwrap(), u64::from(x & 1));
        }
    }
    for choice in [
        DecoderChoice::UnionFind,
        DecoderChoice::Matching,
        DecoderChoice::BpOsd,
    ] {
        let r = logical_error_rate_using(&c, 20_000, 3, choice).unwrap();
        assert_eq!(r.errors, 0, "{choice:?}");
    }
}

/// With Pauli noise beside the heralded channel (X_ERROR(0.08) and an X
/// part of 0.9), conditioned matching finds the minimum-weight correction
/// of every herald pattern and syndrome, negative weights included, as an
/// enumeration of all 2⁵ corrections does; BP+OSD's corrections explain the
/// syndrome, and their weights and observables count the applied errors.
#[test]
fn conditioned_matching_with_parts_above_one_half_is_minimum_weight() {
    let d = 5;
    let (base, part) = (0.08, 0.9);
    let c = pauli_capacity(d, base, "0.02, 0.18, 0, 0");
    let graphlike = heralded_error_model(&c, true).unwrap();
    let whole = heralded_error_model(&c, false).unwrap();
    let mut matching = ConditionedDecoder::new(&graphlike, HeraldedMethod::Matching).unwrap();
    let start = matching.inner_state();
    let mut bposd = [OsdMethod::Zero, OsdMethod::CombinationSweep].map(|osd| {
        ConditionedDecoder::new(
            &whole,
            HeraldedMethod::BpOsd(BpOsdOptions {
                osd,
                ..BpOsdOptions::default()
            }),
        )
        .unwrap()
    });
    let length = |p: f64| ((1.0 - p) / p).ln();
    let (mut compared, mut negative) = (0, 0);
    for erased in 0u32..1 << d {
        let heralds: Vec<usize> = (0..d).filter(|&i| erased >> i & 1 == 1).collect();
        let p: Vec<f64> = (0..d)
            .map(|i| {
                if erased >> i & 1 == 1 {
                    xor_probability(base, part)
                } else {
                    base
                }
            })
            .collect();
        for syndrome in 0u32..1 << (d - 1) {
            let fired: Vec<usize> = (0..d - 1)
                .filter(|&i| syndrome >> i & 1 == 1)
                .map(|i| d + i)
                .collect();
            // Every correction with this syndrome, by weight.
            let mut best: Vec<(f64, u64)> = (0u32..1 << d)
                .filter(|e| (0..d - 1).all(|i| (e >> i ^ e >> (i + 1)) & 1 == syndrome >> i & 1))
                .map(|e| {
                    let w: f64 = (0..d)
                        .filter(|&i| e >> i & 1 == 1)
                        .map(|i| length(p[i]))
                        .sum();
                    (w, u64::from(e & 1))
                })
                .collect();
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (got, w) = matching.decode_with_weight(&fired, &heralds).unwrap();
            assert!(
                (w - best[0].0).abs() < 1e-5 * best[0].0.abs().max(1.0),
                "{erased:b} {syndrome:b}: {w} against {best:?}"
            );
            if best[1].0 - best[0].0 > 1e-4 {
                assert_eq!(got, best[0].1, "{erased:b} {syndrome:b}");
                compared += 1;
                negative += (best[0].0 < 0.0) as usize;
            }
            for dec in &mut bposd {
                let o = dec.decode_detailed(&fired, &heralds).unwrap();
                let slots = dec.slots();
                let mut flipped: Vec<usize> = Vec::new();
                let mut observables = 0;
                let mut weight = 0.0;
                for &k in &o.errors {
                    flipped.extend(&slots[k].detectors);
                    observables ^= slots[k].observables;
                    let q = model_probability(&whole, &heralds, &slots[k]);
                    weight += length(q);
                }
                assert_eq!(cancel_pairs(flipped), fired, "{erased:b} {syndrome:b}");
                assert_eq!(o.observables, observables);
                assert!((o.weight - weight).abs() < 1e-9 * weight.abs().max(1.0));
                assert_eq!(dec.decode_heralded(&fired, &heralds).unwrap(), observables);
            }
        }
    }
    assert!(compared > 400 && negative > 100, "{compared} {negative}");
    assert_eq!(matching.inner_state(), start);
}

/// The probability of `e`'s detectors and observables in the shot's model.
fn model_probability(model: &HeraldedErrorModel, heralds: &[usize], e: &ErrorMechanism) -> f64 {
    model
        .conditioned(heralds)
        .unwrap()
        .errors
        .into_iter()
        .find(|m| m.detectors == e.detectors && m.observables == e.observables)
        .map_or(0.0, |m| m.probability)
}

/// A Pauli channel whose X and Z have the same effect (before a Y-basis
/// readout) gives one part at (pX + pZ)/t, the probability that the
/// readout flips given the herald, not the two parts merged as
/// independent errors.
#[test]
fn equal_x_and_z_parts_are_one_part_at_its_exact_probability() {
    for (pauli, want) in [
        ("0, 0.1, 0, 0.1", 1.0),
        ("0.01, 0.03, 0.05, 0.02", 0.05 / 0.11),
    ] {
        let c = circuit(&format!(
            "RY 0\nHERALDED_PAULI_CHANNEL_1({pauli}) 0\nDETECTOR rec[-1]\nMY 0\nDETECTOR rec[-1]"
        ));
        let model = heralded_error_model(&c, true).unwrap();
        assert_eq!(model.herald_detectors(), &[0]);
        let parts = model.outcomes(0);
        assert_eq!(parts.len(), 1, "{parts:?}");
        assert_eq!(parts[0].detectors, [1]);
        assert!(close(parts[0].probability, want, 1e-15), "{parts:?}");
        // The sampler agrees: given the herald, D1 fires with that
        // probability.
        let s = sample_detectors(&c, 200_000, 9).unwrap();
        let (mut heralded, mut both) = (0, 0);
        for fired in s.fired_by_shot() {
            if fired.first() == Some(&0) {
                heralded += 1;
                both += fired.contains(&1) as usize;
            }
        }
        assert!(within(both, heralded, want, 5.0), "{both} of {heralded}");
    }
}

/// A colour code's errors flip three detectors: its decomposed heralded
/// model is refused with the remedy named, and BP+OSD decodes it on the
/// model without decomposition.
#[test]
fn colour_codes_decode_with_heralds_on_the_model_without_decomposition() {
    let c = css_code_memory_erasure(
        &CssCode::color_code(3).unwrap(),
        2,
        &ErasureNoise::uniform(0.01, 0.5).unwrap(),
        MemoryBasis::Z,
        CssSchedule::Greedy,
    )
    .unwrap();
    let err = heralded_error_model(&c, true).unwrap_err().to_string();
    assert!(
        err.contains("BP+OSD, on the model without decomposition"),
        "{err}"
    );
    let model = heralded_error_model(&c, false).unwrap();
    assert!(!model.herald_detectors().is_empty());
    let mut dec =
        ConditionedDecoder::new(&model, HeraldedMethod::BpOsd(BpOsdOptions::default())).unwrap();
    let r = logical_error_rate_decoding(&c, 500, 1, &mut dec).unwrap();
    assert_eq!(
        r,
        logical_error_rate_using(&c, 500, 1, DecoderChoice::BpOsd).unwrap()
    );
}

/// `HERALDED_PAULI_CHANNEL_1`'s probabilities may add up to 1 + 10⁻⁷
/// (Stim's tolerance); the model's errors stay probabilities, so the
/// decoders take it.
#[test]
fn a_pauli_sum_just_over_one_stays_a_probability() {
    let c = circuit("HERALDED_PAULI_CHANNEL_1(0.1, 0.2, 0.3, 0.40000005) 0\nDETECTOR rec[-1]");
    for decompose in [false, true] {
        let dem = detector_error_model(&c, decompose).unwrap();
        assert_errors(&dem, &[(&[0], 0, 1.0)], 0.0);
        assert!(
            UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap())
                .decode(&[0])
                .is_ok()
        );
        assert!(BpOsdDecoder::new(&dem, BpOsdOptions::default()).is_ok());
    }
    let model = heralded_error_model(&c, true).unwrap();
    assert_eq!(model.herald_probability(0), 1.0);
}

#[test]
#[ignore = "measurement: cargo test --release -- --ignored measure_heralds --nocapture"]
fn measure_heralds() {
    use std::time::Instant;
    let peak = || {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM"))
                    .map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
            })
            .unwrap_or_else(|| "?".into())
    };
    let noise = |share| ErasureNoise::uniform(0.01, share).unwrap();
    println!("== heralded models (surface Z, rounds = d, p = 0.01, share 0.9)");
    for d in [5, 7, 11] {
        let c = surface_code_memory_erasure(d, d, &noise(0.9), MemoryBasis::Z).unwrap();
        let t = Instant::now();
        let m = heralded_error_model(&c, true).unwrap();
        println!(
            "d = {d}: {} detectors, {} herald detectors, {} base errors, built in {:.3} s, peak {} kB",
            c.n_detectors(),
            m.herald_detectors().len(),
            m.base().errors.len(),
            t.elapsed().as_secs_f64(),
            peak()
        );
    }
    let t = Instant::now();
    let r = surface_code_memory_erasure_repeated(5, 50_000, &noise(0.9), MemoryBasis::Z).unwrap();
    let m = r.heralded_error_model(true).unwrap();
    println!(
        // 10^5 rounds would hold 2.7·10^7 detectors, beyond a model's 2^24.
        "folded d = 5, 5·10^4 rounds: {} herald detectors, built in {:.3} s, peak {} kB",
        m.herald_detectors().len(),
        t.elapsed().as_secs_f64(),
        peak()
    );
    println!("== sampler (surface Z d = 5, r = 5, p = 0.01)");
    for share in [0.0, 0.9] {
        let c = surface_code_memory_erasure(5, 5, &noise(share), MemoryBasis::Z).unwrap();
        let shots = 100_000;
        let t = Instant::now();
        let s = sample_detectors(&c, shots, 1).unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "share {share}: {} detectors, {:.0} shots/s",
            s.n_detectors(),
            shots as f64 / secs
        );
    }
    println!("== decoding, µs per shot (surface Z, rounds = d, p = 0.01)");
    for d in [5, 7, 9] {
        for share in [0.0, 0.9] {
            let c = surface_code_memory_erasure(d, d, &noise(share), MemoryBasis::Z).unwrap();
            let shots = if d == 9 { 2000 } else { 5000 };
            let samples = sample_detectors(&c, shots, 2).unwrap().fired_by_shot();
            let mut line = format!("d = {d}, share {share}:");
            for method in [
                HeraldedMethod::UnionFind,
                HeraldedMethod::Matching,
                HeraldedMethod::BpOsd(BpOsdOptions::default()),
            ] {
                let decompose = !matches!(method, HeraldedMethod::BpOsd(_));
                if d == 9 && !decompose {
                    continue;
                }
                let model = heralded_error_model(&c, decompose).unwrap();
                let mut dec: Box<dyn Decoder> = if model.herald_detectors().is_empty() {
                    let choice = match method {
                        HeraldedMethod::UnionFind => DecoderChoice::UnionFind,
                        HeraldedMethod::Matching => DecoderChoice::Matching,
                        _ => DecoderChoice::BpOsd,
                    };
                    choice.build_for(&c).unwrap()
                } else {
                    Box::new(ConditionedDecoder::new(&model, method).unwrap())
                };
                let (mut errors, t) = (0, Instant::now());
                for fired in &samples {
                    errors += (!fired.is_empty() && dec.decode(fired).is_err()) as usize;
                }
                let us = t.elapsed().as_secs_f64() * 1e6 / shots as f64;
                let name = match method {
                    HeraldedMethod::UnionFind => "UF",
                    HeraldedMethod::Matching => "matching",
                    _ => "BP+OSD",
                };
                line += &format!(" {name} {us:.1}");
                assert_eq!(errors, 0);
            }
            println!("{line}");
        }
    }
    let c = surface_code_memory_erasure(7, 7, &noise(0.9), MemoryBasis::Z).unwrap();
    let model = heralded_error_model(&c, true).unwrap();
    let dec = ConditionedDecoder::new(&model, HeraldedMethod::Matching).unwrap();
    let parts = dec.herald_parts.len();
    let bytes = dec
        .slots
        .iter()
        .map(|e| 48 + 8 * e.detectors.len())
        .sum::<usize>()
        + parts * 16
        + dec.n_detectors * 8
        + dec.slots.len() * 24;
    println!(
        "ConditionedDecoder (matching, d = 7): {} slots, {parts} herald parts, about {} kB besides the inner decoder",
        dec.slots.len(),
        bytes / 1024
    );
}
