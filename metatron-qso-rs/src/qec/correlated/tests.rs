use super::*;
use crate::qec::dem::CorrelatedError;
use crate::qec::{
    CircuitNoise, DecoderChoice, DetectorCircuit, MemoryBasis, correlated_error_model,
    detector_error_model, logical_error_rate_using, repetition_code_memory, sample_detectors,
    surface_code_memory_with,
};

/// Stim's rotated surface code memory (d = 5, 5 rounds, p = 0.006): X-type
/// and Z-type detectors, so Y errors flip four detectors.
fn stim_surface() -> DetectorCircuit {
    DetectorCircuit::from_stim(include_str!(
        "../../../tests/data/stim/surface_d5_memory_z.stim"
    ))
    .unwrap()
}

#[test]
fn the_model_keeps_the_decomposition() {
    for c in [
        surface_code_memory_with(3, 3, &CircuitNoise::uniform(0.004), MemoryBasis::Z).unwrap(),
        surface_code_memory_with(5, 2, &CircuitNoise::uniform(0.002), MemoryBasis::X).unwrap(),
        stim_surface(),
    ] {
        let model = correlated_error_model(&c).unwrap();
        let flat = model.to_dem();
        let reference = detector_error_model(&c, true).unwrap();
        assert_eq!(flat.n_detectors, reference.n_detectors);
        assert_eq!(flat.errors.len(), reference.errors.len());
        for (a, b) in flat.errors.iter().zip(&reference.errors) {
            assert_eq!((&a.detectors, a.observables), (&b.detectors, b.observables));
            assert!((a.probability - b.probability).abs() <= 1e-12 * b.probability);
        }
        // Every part flips one or two detectors.
        assert!(
            model
                .errors
                .iter()
                .all(|e| e.parts.iter().all(|(d, _)| (1..=2).contains(&d.len())))
        );
    }
    // With both types of detectors, Y errors in the bulk have two parts.
    let model = correlated_error_model(&stim_surface()).unwrap();
    assert!(model.errors.iter().any(|e| e.parts.len() == 2));
}

#[test]
fn without_correlations_it_is_matching() {
    // Every part its own error: no rule reweights, and both passes equal
    // plain matching.
    let c = repetition_code_memory(7, 7, &CircuitNoise::uniform(0.03)).unwrap();
    let dem = correlated_error_model(&c).unwrap().to_dem();
    let model = CorrelatedErrorModel {
        n_detectors: dem.n_detectors,
        n_observables: dem.n_observables,
        errors: dem
            .errors
            .iter()
            .map(|e| CorrelatedError {
                probability: e.probability,
                parts: vec![(e.detectors.clone(), e.observables)],
            })
            .collect(),
    };
    let mut correlated = CorrelatedMatchingDecoder::new(&model).unwrap();
    assert!(correlated.rules.iter().all(Vec::is_empty));
    let mut plain = SparseMatchingDecoder::new(MatchingGraph::from_dem(&model.to_dem()).unwrap());
    let samples = sample_detectors(&c, 3000, 5).unwrap();
    for fired in samples.fired_by_shot() {
        assert_eq!(
            correlated.decode(&fired).unwrap(),
            plain.decode(&fired).unwrap()
        );
    }
}

#[test]
fn a_correlated_error_is_decoded_whole() {
    // Detectors 0..4 on a line with the boundary at both ends; edge (1, 2)
    // flips the observable. A two-part error {(0, 1), (2, 3)} is likely
    // (p = 0.1), the single edges are not (p = 0.01) except the middle
    // (1, 2) with observable at p = 0.02.
    let part = |d: &[usize], o: u64| (d.to_vec(), o);
    let single = |d: &[usize], o: u64, p: f64| CorrelatedError {
        probability: p,
        parts: vec![part(d, o)],
    };
    let model = CorrelatedErrorModel {
        n_detectors: 4,
        n_observables: 1,
        errors: vec![
            CorrelatedError {
                probability: 0.1,
                parts: vec![part(&[0, 1], 0), part(&[2, 3], 0)],
            },
            single(&[0], 0, 0.01),
            single(&[1, 2], 1, 0.02),
            single(&[3], 0, 0.01),
        ],
    };
    let mut plain = SparseMatchingDecoder::new(MatchingGraph::from_dem(&model.to_dem()).unwrap());
    let mut correlated = CorrelatedMatchingDecoder::new(&model).unwrap();
    // Fired 0, 2, 3... detector 0 and 2: plain matching pairs (0, 1, 2)
    // through the observable edge or both to the boundary; with the
    // correlation, the first pass's (2, 3) edge makes (0, 1) cheap.
    for fired in [vec![0, 1], vec![2, 3], vec![0, 1, 2, 3], vec![0, 2]] {
        let p = plain.decode(&fired).unwrap();
        let c = correlated.decode(&fired).unwrap();
        assert!(p <= 1 && c <= 1, "{fired:?}");
    }
    // The rules: given (0, 1), (2, 3) has probability 0.1 / 0.1 → ½,
    // weight 0; and the other way round.
    let rules: usize = correlated.rules.iter().map(Vec::len).sum();
    assert_eq!(rules, 2);
    assert!(correlated.rules.iter().flatten().all(|&(_, w)| w == 0));
    // Decoding leaves the weights as they were.
    let before = correlated.decoder.weight.clone();
    correlated.decode(&[0, 2]).unwrap();
    assert_eq!(correlated.decoder.weight, before);
}

#[test]
fn correlations_lower_the_logical_error_rate() {
    // Rotated surface code with both types of detectors, d = 5, p = 0.006:
    // on the same shots, two-pass matching fails clearly less often than
    // matching.
    let c = stim_surface();
    let shots = 20_000;
    let plain = logical_error_rate_using(&c, shots, 9, DecoderChoice::Matching).unwrap();
    let correlated =
        logical_error_rate_using(&c, shots, 9, DecoderChoice::CorrelatedMatching).unwrap();
    assert!(
        (correlated.errors as f64) < 0.9 * plain.errors as f64,
        "correlated {} vs plain {}",
        correlated.errors,
        plain.errors
    );
}

#[test]
fn invalid_models_are_errors() {
    let model = |p: f64, dets: Vec<usize>| CorrelatedErrorModel {
        n_detectors: 3,
        n_observables: 1,
        errors: vec![CorrelatedError {
            probability: p,
            parts: vec![(dets, 1)],
        }],
    };
    assert!(CorrelatedMatchingDecoder::new(&model(0.1, vec![0, 1])).is_ok());
    for bad in [
        model(0.6, vec![0, 1]),
        model(-0.1, vec![0]),
        model(0.1, vec![1, 0]),
        model(0.1, vec![0, 1, 2]),
        model(0.1, vec![3]),
    ] {
        assert!(CorrelatedMatchingDecoder::new(&bad).is_err());
    }
    // A plain graph has no correlations to use.
    let graph = MatchingGraph::from_dem(&model(0.1, vec![0, 1]).to_dem()).unwrap();
    assert!(DecoderChoice::CorrelatedMatching.build(graph).is_err());
}

#[test]
fn y_errors_split_into_their_parts() {
    // A Y error on data qubits in the middle of Stim's circuit flips X- and
    // Z-type detectors (three or four): decomposed into the X and Z parts,
    // as Stim does (and as QSO did before 1.22).
    let text = include_str!("../../../tests/data/stim/surface_d5_memory_z.stim").replacen(
        "REPEAT 4 {\n",
        "REPEAT 4 {\n    Y_ERROR(0.001) 1 3 5 12 25\n",
        1,
    );
    let c = DetectorCircuit::from_stim(&text).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    assert!(dem.errors.iter().all(|e| e.detectors.len() <= 2));
    let model = correlated_error_model(&c).unwrap();
    assert!(model.errors.iter().any(|e| e.parts.len() == 2));
    assert!(CorrelatedMatchingDecoder::new(&model).is_ok());
}
