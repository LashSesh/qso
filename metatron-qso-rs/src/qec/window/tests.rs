//! Validation of decoding in windows (1.57).

use super::*;
use crate::qec::dem::{DetectorErrorModel, ErrorMechanism};
use crate::qec::{
    BpOsdDecoder, BpOsdOptions, CircuitNoise, CssCode, DecoderChoice, DetectorCircuit, MemoryBasis,
    css_code_memory, detector_error_model, logical_error_rate_using, repetition_code_memory,
    sample_detectors, surface_code_memory,
};

/// Detection events of `shots` shots and their observable flips.
fn shots(circuit: &DetectorCircuit, shots: usize, seed: u64) -> Vec<(Vec<usize>, u64)> {
    let s = sample_detectors(circuit, shots, seed).unwrap();
    s.fired_by_shot()
        .into_iter()
        .enumerate()
        .map(|(k, f)| (f, s.observable_flips(k)))
        .collect()
}

fn matching_windows(circuit: &DetectorCircuit, window: usize, commit: usize) -> WindowDecoder {
    WindowDecoder::new(
        &detector_error_model(circuit, true).unwrap(),
        &circuit.detector_layers(),
        WindowOptions { window, commit },
        WindowMethod::Matching,
    )
    .unwrap()
}

#[test]
fn layers_are_the_rounds() {
    let c = surface_code_memory(3, 5, &CircuitNoise::uniform(1e-3)).unwrap();
    let layers = c.detector_layers();
    // Four Z checks per round, five rounds, then the data measurement.
    assert_eq!(layers.len(), c.n_detectors());
    assert_eq!(layers.len(), 4 * 6);
    for (d, &l) in layers.iter().enumerate() {
        assert_eq!(l, d / 4);
    }
    let r = repetition_code_memory(5, 7, &CircuitNoise::uniform(1e-3)).unwrap();
    assert_eq!(*r.detector_layers().last().unwrap(), 7);
}

#[test]
fn a_window_over_everything_is_the_whole_decoding() {
    let noise = CircuitNoise::uniform(4e-3);
    let c = surface_code_memory(5, 6, &noise).unwrap();
    let mut whole = DecoderChoice::Matching.build_for(&c).unwrap();
    let mut windows = matching_windows(&c, 7, 3);
    assert_eq!(windows.n_windows(), 1);
    let events = shots(&c, 400, 3);
    for (fired, _) in &events {
        assert_eq!(windows.decode(fired).unwrap(), whole.decode(fired).unwrap());
    }
    // BP+OSD on a bicycle code (two rounds, one window).
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    let c = css_code_memory(&code, 2, &CircuitNoise::uniform(2e-3), MemoryBasis::Z).unwrap();
    let dem = detector_error_model(&c, false).unwrap();
    let mut whole = BpOsdDecoder::new(&dem, BpOsdOptions::default()).unwrap();
    let mut windows = WindowDecoder::new(
        &dem,
        &c.detector_layers(),
        WindowOptions::new(3),
        WindowMethod::BpOsd(BpOsdOptions::default()),
    )
    .unwrap();
    assert_eq!(windows.n_windows(), 1);
    for (fired, _) in shots(&c, 100, 5) {
        assert_eq!(
            windows.decode(&fired).unwrap(),
            whole.decode(&fired).unwrap()
        );
    }
}

#[test]
fn streaming_equals_the_batch() {
    let c = surface_code_memory(3, 12, &CircuitNoise::uniform(6e-3)).unwrap();
    let layers = c.detector_layers();
    let mut windows = matching_windows(&c, 4, 2);
    assert!(windows.n_windows() > 4);
    for (fired, _) in shots(&c, 300, 9) {
        let batch = windows.decode(&fired).unwrap();
        let n_layers = windows.n_layers();
        let mut stream = windows.stream();
        let mut committed = Vec::new();
        for l in 0..n_layers {
            let in_layer: Vec<usize> = fired.iter().copied().filter(|&d| layers[d] == l).collect();
            committed.push(stream.push_layer(&in_layer).unwrap());
        }
        assert!(stream.push_layer(&[]).is_err());
        assert_eq!(stream.finish().unwrap(), batch);
        // Commits arrive as windows complete, never later than the end.
        assert_eq!(*committed.last().unwrap(), batch);
    }
}

#[test]
fn windows_share_decoders_and_errors_are_checked() {
    let c = repetition_code_memory(5, 200, &CircuitNoise::uniform(1e-2)).unwrap();
    let w = matching_windows(&c, 6, 3);
    // First window, the bulk, and the last ones.
    assert!(w.n_windows() > 60);
    assert_eq!(w.n_decoders(), 3);
    let dem = detector_error_model(&c, true).unwrap();
    let layers = c.detector_layers();
    let bad = |window, commit| {
        WindowDecoder::new(
            &dem,
            &layers,
            WindowOptions { window, commit },
            WindowMethod::Matching,
        )
        .is_err()
    };
    assert!(bad(0, 0) && bad(3, 4) && bad(3, 0));
    let mut decreasing = layers.clone();
    decreasing.swap(0, 10);
    assert!(
        WindowDecoder::new(
            &dem,
            &decreasing,
            WindowOptions::new(4),
            WindowMethod::Matching
        )
        .is_err()
    );
    assert!(
        WindowDecoder::new(
            &dem,
            &layers[1..],
            WindowOptions::new(4),
            WindowMethod::Matching
        )
        .is_err()
    );
    let mut w = matching_windows(&c, 6, 3);
    assert!(w.decode(&[c.n_detectors()]).is_err());
    let mut s = w.stream();
    assert!(s.push_layer(&[c.n_detectors() - 1]).is_err());
    assert!(s.finish().is_err());
}

#[test]
fn logical_error_rates_approach_the_whole_decoding_as_the_window_grows() {
    // Distance 5, 40 rounds: a window of one round has no buffer and
    // commits blindly; with a buffer of the distance it decodes as the
    // whole experiment does.
    let c = surface_code_memory(5, 40, &CircuitNoise::uniform(5e-3)).unwrap();
    let (n, seed) = (3000, 17);
    let whole = logical_error_rate_using(&c, n, seed, DecoderChoice::Matching).unwrap();
    let mut rates = Vec::new();
    for (window, commit) in [(1, 1), (2, 1), (4, 2), (10, 5)] {
        let r = logical_error_rate_using(
            &c,
            n,
            seed,
            DecoderChoice::WindowMatching { window, commit },
        )
        .unwrap();
        rates.push(r.errors);
    }
    assert!(
        rates[0] > 2 * whole.errors + 20,
        "{rates:?} vs {}",
        whole.errors
    );
    assert!(rates[0] > rates[2] && rates[1] > rates[3], "{rates:?}");
    // A buffer of the distance: within the sampling error.
    let diff = (rates[3] as f64 - whole.errors as f64).abs();
    assert!(
        diff <= 3.0 * (whole.errors as f64).sqrt() + 3.0,
        "{rates:?} vs {}",
        whole.errors
    );
}

#[test]
fn bposd_in_windows_follows_the_whole_decoding() {
    // A bicycle code of distance 6 over 6 rounds, BP+OSD in windows of
    // three rounds committing two, against BP+OSD on the whole model.
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    let c = css_code_memory(&code, 6, &CircuitNoise::uniform(2e-3), MemoryBasis::Z).unwrap();
    let (n, seed) = (300, 4);
    let whole = logical_error_rate_using(&c, n, seed, DecoderChoice::BpOsd).unwrap();
    let windowed = logical_error_rate_using(
        &c,
        n,
        seed,
        DecoderChoice::WindowBpOsd {
            window: 3,
            commit: 2,
        },
    )
    .unwrap();
    let diff = (windowed.errors as f64 - whole.errors as f64).abs();
    assert!(
        diff <= 3.0 * (whole.errors.max(windowed.errors) as f64).sqrt() + 3.0,
        "{} vs {}",
        windowed.errors,
        whole.errors
    );
}

/// Decoding time per round against the rounds (`cargo test --release --
/// --ignored measure_windows --nocapture`).
#[test]
#[ignore]
fn measure_windows() {
    let noise = CircuitNoise::uniform(5e-3);
    for rounds in [100usize, 1000, 4000] {
        let c = surface_code_memory(5, rounds, &noise).unwrap();
        let start = std::time::Instant::now();
        let dem = detector_error_model(&c, true).unwrap();
        let model = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let mut w = WindowDecoder::new(
            &dem,
            &c.detector_layers(),
            WindowOptions {
                window: 10,
                commit: 5,
            },
            WindowMethod::Matching,
        )
        .unwrap();
        let built = start.elapsed().as_secs_f64();
        let events = shots(&c, 200, 1);
        let start = std::time::Instant::now();
        let mut wrong = 0;
        for (fired, obs) in &events {
            if w.decode(fired).unwrap() != *obs {
                wrong += 1;
            }
        }
        let windowed = start.elapsed().as_secs_f64();
        let mut whole = DecoderChoice::Matching.build_for(&c).unwrap();
        let start = std::time::Instant::now();
        let mut wrong_whole = 0;
        for (fired, obs) in &events {
            if whole.decode(fired).unwrap() != *obs {
                wrong_whole += 1;
            }
        }
        let full = start.elapsed().as_secs_f64();
        println!(
            "surface d=5, {rounds} rounds: model in {model:.1} s, {} windows, {} decoders, built in {built:.2} s; \
             {:.1} µs per round windowed ({wrong} wrong of 200), {:.1} µs whole ({wrong_whole})",
            w.n_windows(),
            w.n_decoders(),
            windowed / 200.0 / rounds as f64 * 1e6,
            full / 200.0 / rounds as f64 * 1e6,
        );
    }
}

/// Logical error rates against the window, for matching on the surface
/// code and BP+OSD on the [[72,12,6]] bicycle code (`cargo test --release
/// -- --ignored measure_rates --nocapture`).
#[test]
#[ignore]
fn measure_rates() {
    let c = surface_code_memory(5, 40, &CircuitNoise::uniform(5e-3)).unwrap();
    let (n, seed) = (20_000, 17);
    let whole = logical_error_rate_using(&c, n, seed, DecoderChoice::Matching).unwrap();
    println!(
        "surface d=5, 40 rounds, p=5e-3: whole {} of {n}",
        whole.errors
    );
    for (window, commit) in [(1, 1), (2, 1), (3, 1), (4, 2), (6, 3), (10, 5), (20, 10)] {
        let r = logical_error_rate_using(
            &c,
            n,
            seed,
            DecoderChoice::WindowMatching { window, commit },
        )
        .unwrap();
        println!("  window {window}, commit {commit}: {} of {n}", r.errors);
    }
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    for rounds in [6usize, 12, 24, 48] {
        let c =
            css_code_memory(&code, rounds, &CircuitNoise::uniform(2e-3), MemoryBasis::Z).unwrap();
        let n = 2000;
        let start = std::time::Instant::now();
        let windowed = logical_error_rate_using(
            &c,
            n,
            4,
            DecoderChoice::WindowBpOsd {
                window: 6,
                commit: 3,
            },
        )
        .unwrap();
        let tw = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let whole = logical_error_rate_using(&c, n, 4, DecoderChoice::BpOsd).unwrap();
        let t = start.elapsed().as_secs_f64();
        println!(
            "bicycle [[72,12,6]], {rounds} rounds, p=2e-3: windows (6, 3) {} of {n} in {tw:.1} s; whole {} in {t:.1} s",
            windowed.errors, whole.errors
        );
    }
}

fn model(n: usize, errors: &[(f64, &[usize], u64)]) -> DetectorErrorModel {
    DetectorErrorModel {
        n_detectors: n,
        n_observables: 1,
        errors: errors
            .iter()
            .map(|&(p, d, o)| ErrorMechanism {
                probability: p,
                detectors: d.to_vec(),
                observables: o,
            })
            .collect(),
    }
}

#[test]
fn a_boundary_only_in_the_past_falls_back() {
    // Found by the review: detector 0 reaches the boundary, detector 1
    // only through detector 0. The window of layer 1 alone cannot explain
    // an event at 1; its fallback takes the error back to layer 0.
    let dem = model(2, &[(0.1, &[0], 0), (0.05, &[0, 1], 1)]);
    let mut w =
        WindowDecoder::new(&dem, &[0, 1], WindowOptions::new(1), WindowMethod::Matching).unwrap();
    assert_eq!(w.decode(&[1]).unwrap(), 1);
    // Reset errors only: their boundary lies at the start.
    let noise = CircuitNoise {
        after_reset: 3e-2,
        ..CircuitNoise::default()
    };
    for c in [
        repetition_code_memory(5, 10, &noise).unwrap(),
        surface_code_memory(3, 10, &noise).unwrap(),
    ] {
        // Every window decodes; with a buffer of four rounds as the whole
        // decoding does (repetition 71, 9, 6, 0, 0 failures against 0;
        // surface 138, 49, 40, 34, 35 against 34).
        let whole = logical_error_rate_using(&c, 2000, 1, DecoderChoice::Matching).unwrap();
        let mut errors = Vec::new();
        for (window, commit) in [(1, 1), (2, 1), (3, 2), (4, 2), (6, 3)] {
            let r = logical_error_rate_using(
                &c,
                2000,
                1,
                DecoderChoice::WindowMatching { window, commit },
            )
            .unwrap();
            errors.push(r.errors);
        }
        assert!(
            errors[0] > errors[1] && errors[1] >= errors[2],
            "{errors:?}"
        );
        let tolerance = 3 * (whole.errors as f64).sqrt() as usize + 3;
        assert!(
            errors[3..].iter().all(|&e| e <= whole.errors + tolerance),
            "{errors:?} vs {}",
            whole.errors
        );
    }
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    let c = css_code_memory(&code, 4, &noise, MemoryBasis::Z).unwrap();
    logical_error_rate_using(
        &c,
        200,
        1,
        DecoderChoice::WindowBpOsd {
            window: 2,
            commit: 1,
        },
    )
    .unwrap();
}

#[test]
fn a_failed_window_voids_the_stream() {
    // Detector 1 has no error: no correction explains an event there.
    let dem = model(2, &[(0.1, &[0], 0)]);
    let mut w =
        WindowDecoder::new(&dem, &[0, 1], WindowOptions::new(1), WindowMethod::Matching).unwrap();
    assert!(w.decode(&[1]).is_err());
    let mut s = w.new_stream();
    w.push_layer(&mut s, &[]).unwrap();
    assert!(w.push_layer(&mut s, &[1]).is_err());
    assert!(w.finish(&s).is_err());
    assert!(w.push_layer(&mut s, &[]).is_err());
}

#[test]
fn matching_windows_refuse_hyperedges() {
    // An error of three detectors would hide its third beyond every
    // window.
    let dem = model(
        3,
        &[
            (0.1, &[0, 1, 2], 1),
            (0.1, &[0], 0),
            (0.1, &[1], 0),
            (0.1, &[2], 0),
        ],
    );
    assert!(
        WindowDecoder::new(
            &dem,
            &[0, 0, 1],
            WindowOptions::new(1),
            WindowMethod::Matching
        )
        .is_err()
    );
    assert!(
        WindowDecoder::new(
            &dem,
            &[0, 0, 1],
            WindowOptions::new(1),
            WindowMethod::BpOsd(BpOsdOptions::default())
        )
        .is_ok()
    );
}

#[test]
fn bposd_streams_equal_the_batch() {
    let c = crate::qec::css_code_memory(
        &CssCode::color_code(5).unwrap(),
        6,
        &CircuitNoise::uniform(3e-3),
        MemoryBasis::X,
    )
    .unwrap();
    let layers = c.detector_layers();
    let mut w = WindowDecoder::new(
        &detector_error_model(&c, false).unwrap(),
        &layers,
        WindowOptions {
            window: 3,
            commit: 2,
        },
        WindowMethod::BpOsd(BpOsdOptions::default()),
    )
    .unwrap();
    for (fired, _) in shots(&c, 100, 2) {
        let batch = w.decode(&fired).unwrap();
        let mut s = w.new_stream();
        for l in 0..w.n_layers() {
            let in_layer: Vec<usize> = fired.iter().copied().filter(|&d| layers[d] == l).collect();
            w.push_layer(&mut s, &in_layer).unwrap();
        }
        assert_eq!(w.finish(&s).unwrap(), batch);
    }
}

// ---- parallel windows (1.61) ------------------------------------------------

fn parallel_matching(circuit: &DetectorCircuit, commit: usize, buffer: usize) -> WindowDecoder {
    WindowDecoder::parallel(
        &detector_error_model(circuit, true).unwrap(),
        &circuit.detector_layers(),
        ParallelWindowOptions::new(commit, buffer),
        WindowMethod::Matching,
    )
    .unwrap()
}

#[test]
fn parallel_windows_over_everything_are_the_whole_decoding() {
    let c = surface_code_memory(5, 6, &CircuitNoise::uniform(4e-3)).unwrap();
    let mut whole = DecoderChoice::Matching.build_for(&c).unwrap();
    let mut windows = parallel_matching(&c, 7, 2);
    assert_eq!(windows.n_windows(), 1);
    for (fired, _) in shots(&c, 400, 3) {
        assert_eq!(
            windows.decode(&fired).unwrap(),
            whole.decode(&fired).unwrap()
        );
    }
}

#[test]
fn the_parallel_layout_alternates_windows_and_seams() {
    // 21 layers (20 rounds and the final one), commit 3, buffer 2, errors
    // spanning one layer: first-pass windows commit 0..3, 6..9, 12..15,
    // 18..21, the seams 3..6, 9..12 and 15..18 between them.
    let c = surface_code_memory(3, 20, &CircuitNoise::uniform(1e-3)).unwrap();
    let w = parallel_matching(&c, 3, 2);
    assert_eq!(w.span, 1);
    let layout: Vec<(Kind, usize, usize, usize)> = w
        .windows
        .iter()
        .map(|v| {
            (
                v.kind,
                v.start,
                w.layer_of(v.commit_start),
                w.layer_of(v.commit),
            )
        })
        .collect();
    use Kind::{First, Seam};
    assert_eq!(
        layout,
        vec![
            (First, 0, 0, 3),
            (First, 4, 6, 9),
            (Seam, 3, 3, 6),
            (First, 10, 12, 15),
            (Seam, 9, 9, 12),
            (First, 16, 18, 21),
            (Seam, 15, 15, 18),
        ]
    );
    // Windows of the bulk share decoders: the first, the bulk and the
    // last of the first pass, and the seams.
    assert!(w.n_decoders() <= 4, "{}", w.n_decoders());
    assert_eq!(w.parallel_options(), Some(ParallelWindowOptions::new(3, 2)));
    assert!(
        WindowDecoder::parallel(
            &detector_error_model(&c, true).unwrap(),
            &c.detector_layers(),
            ParallelWindowOptions::new(0, 2),
            WindowMethod::Matching,
        )
        .is_err()
    );
}

#[test]
fn parallel_windows_stream_as_they_decode_in_parallel() {
    let c = surface_code_memory(3, 30, &CircuitNoise::uniform(6e-3)).unwrap();
    let layers = c.detector_layers();
    let mut windows = parallel_matching(&c, 3, 2);
    for (fired, _) in shots(&c, 300, 11) {
        let batch = windows.decode(&fired).unwrap();
        let n_layers = windows.n_layers();
        let mut stream = windows.stream();
        for l in 0..n_layers {
            let in_layer: Vec<usize> = fired.iter().copied().filter(|&d| layers[d] == l).collect();
            stream.push_layer(&in_layer).unwrap();
        }
        assert_eq!(stream.finish().unwrap(), batch);
    }
}

#[test]
fn parallel_windows_fail_as_often_as_sliding_ones() {
    // Distance 5, 40 rounds, as 1.57 measured sliding windows: with a
    // buffer of the distance both approach the whole decoding.
    let c = surface_code_memory(5, 40, &CircuitNoise::uniform(5e-3)).unwrap();
    let mut whole = DecoderChoice::Matching.build_for(&c).unwrap();
    let mut sliding = matching_windows(&c, 10, 5);
    let mut parallel = parallel_matching(&c, 5, 5);
    let (mut w, mut s, mut p) = (0, 0, 0);
    let events = shots(&c, 4000, 21);
    for (fired, actual) in &events {
        w += (whole.decode(fired).unwrap() != *actual) as usize;
        s += (sliding.decode(fired).unwrap() != *actual) as usize;
        p += (parallel.decode(fired).unwrap() != *actual) as usize;
    }
    // About 440 failures each (2197 of 20 000 in 1.57); three standard
    // errors of the difference.
    let sigma = (2.0 * w as f64).sqrt();
    assert!((p as f64 - w as f64).abs() < 3.0 * sigma, "{p} vs {w}");
    assert!((p as f64 - s as f64).abs() < 3.0 * sigma, "{p} vs {s}");
}

#[test]
fn a_stream_of_another_layout_is_refused() {
    let c = surface_code_memory(3, 6, &CircuitNoise::uniform(1e-3)).unwrap();
    let mut sliding = matching_windows(&c, 4, 2);
    let mut parallel = parallel_matching(&c, 2, 1);
    let state = sliding.new_stream();
    assert!(parallel.push_layer(&mut state.clone(), &[]).is_err());
    let state = parallel.new_stream();
    assert!(sliding.push_layer(&mut state.clone(), &[]).is_err());
}

#[test]
fn parallel_bp_osd_windows_decode_a_bicycle_code() {
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    let c = css_code_memory(&code, 12, &CircuitNoise::uniform(2e-3), MemoryBasis::Z).unwrap();
    let dem = detector_error_model(&c, false).unwrap();
    let options = BpOsdOptions::default();
    let mut whole = BpOsdDecoder::new(&dem, options).unwrap();
    let mut parallel = WindowDecoder::parallel(
        &dem,
        &c.detector_layers(),
        ParallelWindowOptions::new(2, 2),
        WindowMethod::BpOsd(options),
    )
    .unwrap();
    assert!(parallel.n_windows() > 2);
    let (mut w, mut p) = (0, 0);
    for (fired, actual) in shots(&c, 300, 4) {
        w += (whole.decode(&fired).unwrap() != actual) as usize;
        p += (parallel.decode(&fired).unwrap() != actual) as usize;
    }
    assert!(
        p <= w + 3 * (w.max(1) as f64).sqrt() as usize + 3,
        "{p} vs {w}"
    );
}
