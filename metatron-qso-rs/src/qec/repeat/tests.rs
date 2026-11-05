use super::*;
use crate::qec::codes::{
    CircuitNoise, MemoryBasis, repetition_code_memory, repetition_code_memory_repeated,
    surface_code_memory_repeated, surface_code_memory_with,
};
use crate::qec::css::{CssCode, CssSchedule, css_code_memory_repeated, css_code_memory_with};
use crate::qec::dem::{correlated_error_model, detector_error_model};
use crate::qec::frame::sample_detectors;

/// The memory experiments, flat and repeated, of `rounds` rounds.
fn experiments(rounds: usize) -> Vec<(String, DetectorCircuit, RepeatedCircuit)> {
    let noise = CircuitNoise::uniform(2e-3);
    let mut out = vec![(
        "repetition d = 5".to_string(),
        repetition_code_memory(5, rounds, &noise).unwrap(),
        repetition_code_memory_repeated(5, rounds, &noise).unwrap(),
    )];
    for basis in [MemoryBasis::Z, MemoryBasis::X] {
        out.push((
            format!("surface d = 3 {basis:?}"),
            surface_code_memory_with(3, rounds, &noise, basis).unwrap(),
            surface_code_memory_repeated(3, rounds, &noise, basis).unwrap(),
        ));
    }
    let colour = CssCode::color_code(3).unwrap();
    for schedule in [
        CssSchedule::Greedy,
        CssSchedule::Directions([0, 1, 2, 3, 4, 5]),
    ] {
        out.push((
            format!("colour d = 3 {schedule:?}"),
            css_code_memory_with(&colour, rounds, &noise, MemoryBasis::Z, schedule).unwrap(),
            css_code_memory_repeated(&colour, rounds, &noise, MemoryBasis::Z, schedule).unwrap(),
        ));
    }
    let (bb, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    out.push((
        "[[72, 12, 6]] bicycle".to_string(),
        css_code_memory_with(&bb, rounds, &noise, MemoryBasis::X, CssSchedule::Bicycle).unwrap(),
        css_code_memory_repeated(&bb, rounds, &noise, MemoryBasis::X, CssSchedule::Bicycle)
            .unwrap(),
    ));
    out
}

fn assert_same_model(name: &str, ours: &DetectorErrorModel, theirs: &DetectorErrorModel) {
    assert_eq!(ours.n_detectors, theirs.n_detectors, "{name}");
    assert_eq!(ours.n_observables, theirs.n_observables, "{name}");
    assert_eq!(ours.errors.len(), theirs.errors.len(), "{name}");
    for (a, b) in ours.errors.iter().zip(&theirs.errors) {
        assert_eq!(
            (&a.detectors, a.observables),
            (&b.detectors, b.observables),
            "{name}"
        );
        assert!(
            (a.probability - b.probability).abs() <= 1e-13 * b.probability,
            "{name}: {} vs {}",
            a.probability,
            b.probability
        );
    }
}

#[test]
fn repeated_memory_experiments_expand_to_the_flat_ones() {
    for rounds in [1, 2, 3, 7] {
        for (name, flat, repeated) in experiments(rounds) {
            let expanded = repeated.flatten().unwrap();
            assert_eq!(expanded, flat, "{name}, {rounds} rounds");
            assert_eq!(repeated.n_detectors(), flat.n_detectors());
            assert_eq!(repeated.n_measurements(), flat.n_measurements());
            assert_eq!(repeated.n_operations(), flat.operations().len() as u64);
            assert_eq!(repeated.detector_layers(), flat.detector_layers());
            if rounds > 2 {
                assert!(repeated.n_stored_operations() < flat.operations().len());
            }
        }
    }
}

#[test]
fn the_folded_model_equals_the_expanded_one_term_by_term() {
    for rounds in [2, 3, 4, 25] {
        for (name, flat, repeated) in experiments(rounds) {
            let name = format!("{name}, {rounds} rounds");
            for decompose in [false, true] {
                // Colour and bicycle codes are no matching graph: both
                // refuse to decompose them.
                match detector_error_model(&flat, decompose) {
                    Ok(theirs) => {
                        let ours = repeated.detector_error_model(decompose).unwrap();
                        assert_same_model(&name, &ours, &theirs);
                    }
                    Err(_) => assert!(repeated.detector_error_model(decompose).is_err()),
                }
            }
            let Ok(theirs) = correlated_error_model(&flat) else {
                assert!(repeated.correlated_error_model().is_err());
                continue;
            };
            let ours = repeated.correlated_error_model().unwrap();
            assert_eq!(ours.errors.len(), theirs.errors.len(), "{name}");
            for (a, b) in ours.errors.iter().zip(&theirs.errors) {
                assert_eq!(a.parts, b.parts, "{name}");
                assert!((a.probability - b.probability).abs() <= 1e-13 * b.probability);
            }
        }
    }
}

#[test]
fn a_repeated_circuit_samples_the_expanded_one_shot_for_shot() {
    for (name, flat, repeated) in experiments(6) {
        let ours = repeated.sample(300, 9).unwrap();
        let theirs = sample_detectors(&flat, 300, 9).unwrap();
        assert_eq!(ours, theirs, "{name}");
    }
}

/// A Stim circuit with nested blocks, a body that reads records of the
/// iteration before, and a final block.
const NESTED: &str = "\
R 0 1 2 3 4
REPEAT 3 {
    X_ERROR(0.01) 0 1 2
    REPEAT 4 {
        DEPOLARIZE1(0.002) 0 1 2
        CX 0 3 1 3 1 4 2 4
        DEPOLARIZE2(0.003) 0 3 1 4
        MR(0.005) 3 4
        DETECTOR rec[-2] rec[-4]
        DETECTOR rec[-1] rec[-3]
    }
    H 0
    H 0
}
M 0 1 2
DETECTOR rec[-3] rec[-2] rec[-5]
DETECTOR rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
";

#[test]
fn stim_blocks_are_kept_and_expand_as_stim_expands_them() {
    // The first iteration of the inner block reads records from before
    // it: two measurements at the start come first.
    let text = format!("R 3 4\nM 3 4\n{NESTED}");
    let repeated = RepeatedCircuit::from_stim(&text).unwrap();
    let flat = DetectorCircuit::from_stim(&text).unwrap();
    assert_eq!(repeated.flatten().unwrap(), flat);
    assert_eq!(repeated.n_detectors(), 3 * 4 * 2 + 2);
    assert!(repeated.n_stored_operations() < flat.operations().len());
    // Written with its blocks, read back the same.
    let written = repeated.to_stim().unwrap();
    assert!(written.contains("REPEAT 3 {") && written.contains("    REPEAT 4 {"));
    assert_eq!(RepeatedCircuit::from_stim(&written).unwrap(), repeated);
    assert_eq!(DetectorCircuit::from_stim(&written).unwrap(), flat);
    for decompose in [false, true] {
        assert_same_model(
            "nested",
            &repeated.detector_error_model(decompose).unwrap(),
            &detector_error_model(&flat, decompose).unwrap(),
        );
    }
    assert_eq!(
        repeated.sample(200, 3).unwrap(),
        sample_detectors(&flat, 200, 3).unwrap()
    );
}

#[test]
fn a_block_that_never_repeats_its_state_is_passed_iteration_by_iteration() {
    // Qubit 1 is never measured inside the block, and its errors reach the
    // final detector, which no iteration shifts into: the state carries
    // that detector through every iteration.
    let text = "\
R 0 1
M 0
REPEAT 30 {
    X_ERROR(0.01) 0 1
    M 0
    DETECTOR rec[-1] rec[-2]
}
M 1
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
";
    let repeated = RepeatedCircuit::from_stim(text).unwrap();
    let flat = repeated.flatten().unwrap();
    assert_same_model(
        "never repeats",
        &repeated.detector_error_model(false).unwrap(),
        &detector_error_model(&flat, false).unwrap(),
    );
}

#[test]
fn a_hundred_thousand_rounds_fold_after_a_few() {
    let noise = CircuitNoise::uniform(1e-3);
    let start = std::time::Instant::now();
    let c = surface_code_memory_repeated(3, 100_000, &noise, MemoryBasis::Z).unwrap();
    assert_eq!(c.n_detectors(), 4 * 100_001);
    let model = c.detector_error_model(true).unwrap();
    assert!(start.elapsed().as_secs_f64() < crate::time_limit(20.0));
    // The bulk repeats: the errors of round 50 000 are those of round
    // 50 001, shifted by its four detectors.
    let round = |k: usize| -> Vec<(Vec<usize>, u64, u64)> {
        model
            .errors
            .iter()
            .filter(|e| e.detectors[0] / 4 == k)
            .map(|e| {
                (
                    e.detectors.iter().map(|d| d - 4 * k).collect(),
                    e.observables,
                    e.probability.to_bits(),
                )
            })
            .collect()
    };
    let bulk = round(50_000);
    assert!(!bulk.is_empty());
    assert_eq!(round(50_001), bulk);
    assert_eq!(round(3), bulk);
    // And it equals the model of the flat circuit where both fit.
    let short = surface_code_memory_repeated(3, 300, &noise, MemoryBasis::Z).unwrap();
    assert_same_model(
        "300 rounds",
        &short.detector_error_model(true).unwrap(),
        &detector_error_model(&short.flatten().unwrap(), true).unwrap(),
    );
}

#[test]
fn blocks_that_do_nothing_or_repeat_one_error_cost_nothing() {
    // Bodies without operations, nested, are left out.
    let c = RepeatedCircuit::from_stim(
        "R 0\nREPEAT 2147483648 {\nREPEAT 2147483648 {\nTICK\n}\n}\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n",
    )
    .unwrap();
    assert_eq!(c.n_operations(), 4);
    assert_eq!(c.flatten().unwrap().operations().len(), 4);
    assert_eq!(c.sample(10, 1).unwrap().shots(), 10);
    // Two billion iterations whose errors all flip the detector after the
    // block: merged at once.
    let start = std::time::Instant::now();
    let c = RepeatedCircuit::from_stim(
        "R 0\nREPEAT 2000000000 {\nX_ERROR(0.001) 0\n}\nM 0\nDETECTOR rec[-1]\n",
    )
    .unwrap();
    let m = c.detector_error_model(false).unwrap();
    assert_eq!(m.errors.len(), 1);
    assert!((m.errors[0].probability - 0.5).abs() < 1e-12);
    assert!(start.elapsed().as_secs_f64() < crate::time_limit(2.0));
    // As merging every iteration does: a block without detectors, and an
    // error that flips only the observable in a block with detectors.
    for text in [
        "R 0\nREPEAT 300 {\nX_ERROR(0.003) 0\nDEPOLARIZE1(0.4) 0\n}\nM 0\nDETECTOR rec[-1]\n",
        "R 0 1\nMR 1\nREPEAT 300 {\nX_ERROR(0.003) 0\nDEPOLARIZE1(0.002) 1\nMR(0.001) 1\nDETECTOR rec[-1] rec[-2]\n}\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    ] {
        let c = RepeatedCircuit::from_stim(text).unwrap();
        for decompose in [false, true] {
            assert_same_model(
                text,
                &c.detector_error_model(decompose).unwrap(),
                &detector_error_model(&c.flatten().unwrap(), decompose).unwrap(),
            );
        }
    }
}

#[test]
fn repeated_circuits_check_their_input() {
    assert!(RepeatedCircuit::from_stim("REPEAT 0 {\nM 0\n}\n").is_err());
    assert!(RepeatedCircuit::from_stim("M 0\nREPEAT 2 {\nDETECTOR rec[-2]\n}\n").is_err());
    assert!(RepeatedCircuit::from_stim("REPEAT 100000 {\nREPEAT 100000 {\nH 0\n}\n}\n").is_err());
    assert!(RepeatedCircuit::from_stim("REPEAT 3 {\nH 0\n").is_err());
    let noise = CircuitNoise::uniform(1e-3);
    assert!(repetition_code_memory_repeated(3, 0, &noise).is_err());
    assert!(surface_code_memory_repeated(4, 10, &noise, MemoryBasis::Z).is_err());
    let long = repetition_code_memory_repeated(3, 2_000_000, &noise).unwrap();
    assert!(long.flatten().is_err());
    assert!(long.sample(1 << 20, 1).is_err());
    // A detector that is not deterministic is refused by the model and
    // the sampler.
    let random = RepeatedCircuit::from_stim("H 0\nREPEAT 2 {\nM 0\nDETECTOR rec[-1]\n}\n").unwrap();
    assert!(random.detector_error_model(false).is_err());
    assert!(random.sample(10, 1).is_err());
}

#[test]
fn repeated_logical_error_rates_equal_the_expanded_ones() {
    use crate::qec::{DecoderChoice, logical_error_rate_repeated, logical_error_rate_using};
    let noise = CircuitNoise::uniform(4e-3);
    let repeated = surface_code_memory_repeated(3, 20, &noise, MemoryBasis::Z).unwrap();
    let flat = repeated.flatten().unwrap();
    for decoder in [
        DecoderChoice::Matching,
        DecoderChoice::WindowMatching {
            window: 4,
            commit: 2,
        },
        DecoderChoice::ParallelWindowMatching {
            commit: 3,
            buffer: 2,
        },
    ] {
        let ours = logical_error_rate_repeated(&repeated, 3000, 17, decoder).unwrap();
        let theirs = logical_error_rate_using(&flat, 3000, 17, decoder).unwrap();
        assert_eq!(ours, theirs, "{decoder:?}");
        assert!(ours.errors > 10, "{decoder:?}: {}", ours.errors);
    }
}

/// Costs of long experiments (1.61): `cargo test --release --lib
/// measure_long_experiments -- --ignored --nocapture`.
#[test]
#[ignore]
#[cfg(feature = "parallel")]
fn measure_long_experiments() {
    use crate::qec::{ParallelWindowOptions, WindowDecoder, WindowMethod, WindowOptions};
    use std::time::Instant;
    let noise = CircuitNoise::uniform(1e-3);
    // The model: flat circuits by the pass backwards, repeated ones folded.
    // (1.57: 1.3 s for 1000 rounds, 30 s for 4000.)
    for rounds in [1000, 4000] {
        let flat = crate::qec::surface_code_memory(5, rounds, &noise).unwrap();
        let t = Instant::now();
        let m = detector_error_model(&flat, true).unwrap();
        eprintln!(
            "flat d = 5, {rounds} rounds: model {:.2} s, {} errors",
            t.elapsed().as_secs_f64(),
            m.errors.len()
        );
    }
    for rounds in [10_000, 100_000] {
        let t = Instant::now();
        let c = surface_code_memory_repeated(5, rounds, &noise, MemoryBasis::Z).unwrap();
        let m = c.detector_error_model(true).unwrap();
        let bytes: usize = m.errors.iter().map(|e| 48 + 8 * e.detectors.len()).sum();
        eprintln!(
            "repeated d = 5, {rounds} rounds: {} operations stored of {}, model {:.2} s, {} errors ({:.0} per round, about {:.0} MB)",
            c.n_stored_operations(),
            c.n_operations(),
            t.elapsed().as_secs_f64(),
            m.errors.len(),
            m.errors.len() as f64 / rounds as f64,
            bytes as f64 / 1e6
        );
    }
    // Decoding per round and core: sliding windows of ten rounds against
    // parallel windows committing five with buffers of five.
    for rounds in [10_000, 100_000] {
        let c = surface_code_memory_repeated(5, rounds, &noise, MemoryBasis::Z).unwrap();
        let m = c.detector_error_model(true).unwrap();
        let layers = c.detector_layers();
        let shots = if rounds == 10_000 { 64 } else { 8 };
        let t = Instant::now();
        let samples = c.sample(shots, 5).unwrap();
        let sampled = t.elapsed().as_secs_f64();
        let fired = samples.fired_by_shot();
        let t = Instant::now();
        let mut sliding = WindowDecoder::new(
            &m,
            &layers,
            WindowOptions {
                window: 10,
                commit: 5,
            },
            WindowMethod::Matching,
        )
        .unwrap();
        let built = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut fails = 0;
        let mut predictions = Vec::new();
        for (s, f) in fired.iter().enumerate() {
            let p = sliding.decode(f).unwrap();
            predictions.push(p);
            fails += (p != samples.observable_flips(s)) as usize;
        }
        let per = t.elapsed().as_secs_f64() / (shots * rounds) as f64;
        eprintln!(
            "d = 5, {rounds} rounds, {shots} shots: sampled {:.2} µs per round and shot, sliding windows built {built:.2} s, decoded {:.2} µs per round and shot, {fails} failures",
            sampled / (shots * rounds) as f64 * 1e6,
            per * 1e6
        );
        let mut parallel = WindowDecoder::parallel(
            &m,
            &layers,
            ParallelWindowOptions::new(5, 5),
            WindowMethod::Matching,
        )
        .unwrap();
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let t = Instant::now();
            let mut fails = 0;
            pool.install(|| {
                for (s, f) in fired.iter().enumerate() {
                    fails += (parallel.decode(f).unwrap() != samples.observable_flips(s)) as usize;
                }
            });
            let per = t.elapsed().as_secs_f64() / (shots * rounds) as f64;
            eprintln!(
                "  parallel windows on {threads} threads: {:.2} µs per round and shot, {fails} failures",
                per * 1e6
            );
        }
    }
}
