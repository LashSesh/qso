//! Splitting against enumeration and sampling.

use super::*;
use crate::qec::{
    CircuitNoise, logical_error_rate_using, repetition_code_memory, surface_code_memory,
};

fn repetition(d: usize) -> impl Fn(f64) -> Result<DetectorCircuit> + Sync {
    move |p| repetition_code_memory(d, 2, &CircuitNoise::uniform(p))
}

fn surface(rounds: usize) -> impl Fn(f64) -> Result<DetectorCircuit> + Sync {
    move |p| surface_code_memory(3, rounds, &CircuitNoise::uniform(p))
}

fn options(seed: u64) -> SplittingOptions {
    SplittingOptions {
        start_shots: 40_000,
        chains: 8,
        steps: 4000,
        burn_in: 1000,
        seed,
    }
}

#[test]
fn enumeration_brackets_sampling() {
    // The repetition code of distance 3 over two rounds (15 errors) in
    // full, and the surface code of distance 3 over three rounds (55
    // errors) to five errors, against 10⁶ sampled shots of the circuit.
    for (circuit, weight) in [
        (repetition(3)(0.04).unwrap(), 15),
        (surface(3)(0.01).unwrap(), 5),
    ] {
        let e = enumerated_failure_rate(&circuit, DecoderChoice::Matching, weight).unwrap();
        let s = logical_error_rate_using(&circuit, 1_000_000, 3, DecoderChoice::Matching).unwrap();
        let low = e.rate - 4.0 * s.standard_error;
        let high = e.rate + e.remainder + 4.0 * s.standard_error;
        assert!(
            s.rate >= low && s.rate <= high,
            "{} vs [{}, {}]",
            s.rate,
            e.rate,
            e.rate + e.remainder
        );
        if weight == 15 {
            assert_eq!(e.remainder, 0.0);
            assert_eq!(e.configurations, 1 << 15);
        }
    }
}

#[test]
fn splitting_follows_the_enumerated_rate_far_below_sampling() {
    // From 3 % down to 10⁻⁶ (repetition code, failure rate about 10⁻¹¹)
    // and 10⁻⁴ (surface code, about 10⁻⁷): the enumeration's remainder is
    // below 10⁻⁶ of the rate there.
    for (name, at, max_weight, target) in [
        (
            "repetition",
            &repetition(3) as &(dyn Fn(f64) -> Result<DetectorCircuit> + Sync),
            15,
            1e-6,
        ),
        ("surface", &surface(3), 5, 1e-4),
    ] {
        let exact =
            enumerated_failure_rate(&at(target).unwrap(), DecoderChoice::Matching, max_weight)
                .unwrap();
        assert!(exact.remainder < 1e-6 * exact.rate, "{name}: {exact:?}");
        let est =
            splitting_failure_rate(at, DecoderChoice::Matching, 0.03, target, &options(1)).unwrap();
        let z = (est.rate.ln() - exact.rate.ln()) / est.log_standard_error;
        eprintln!(
            "{name}: {:.4e} ± {:.1} % vs {:.4e} ({z:.2} σ), {} rates, {} decodes",
            est.rate,
            100.0 * est.log_standard_error,
            exact.rate,
            est.rates.len(),
            est.decodes
        );
        assert!(z.abs() < 4.0, "{name}: {z}");
        assert!(
            est.log_standard_error < 0.2,
            "{name}: {}",
            est.log_standard_error
        );
    }
}

#[test]
fn splitting_equals_direct_sampling_where_that_reaches() {
    // Surface code over three rounds from 2 % to 0.5 %: 4·10⁶ shots of the
    // circuit at the target.
    let at = surface(3);
    let est =
        splitting_failure_rate(&at, DecoderChoice::Matching, 0.02, 0.005, &options(2)).unwrap();
    let s = logical_error_rate_using(&at(0.005).unwrap(), 4_000_000, 9, DecoderChoice::Matching)
        .unwrap();
    let sigma = (est.standard_error.powi(2) + s.standard_error.powi(2)).sqrt();
    assert!(
        (est.rate - s.rate).abs() < 4.0 * sigma,
        "{} vs {} ± {sigma}",
        est.rate,
        s.rate
    );
}

#[test]
fn the_standard_error_covers_the_reference_in_repeated_runs() {
    // Twenty runs of the repetition code from 3 % to 10⁻⁴ with smaller
    // chains: the deviations from the exact rate in standard errors have a
    // spread near one, and most lie within two.
    let at = repetition(3);
    let exact = enumerated_failure_rate(&at(1e-4).unwrap(), DecoderChoice::Matching, 15).unwrap();
    let runs = 20;
    let z: Vec<f64> = (0..runs)
        .map(|seed| {
            let o = SplittingOptions {
                start_shots: 20_000,
                chains: 8,
                steps: 1000,
                burn_in: 300,
                seed,
            };
            let est = splitting_failure_rate(&at, DecoderChoice::Matching, 0.03, 1e-4, &o).unwrap();
            (est.rate.ln() - exact.rate.ln()) / est.log_standard_error
        })
        .collect();
    let inside = z.iter().filter(|z| z.abs() < 2.0).count();
    let spread = (z.iter().map(|z| z * z).sum::<f64>() / runs as f64).sqrt();
    eprintln!("z: {z:.2?}, spread {spread:.2}");
    assert!(inside >= 16, "{inside} of {runs}");
    assert!(spread < 1.5, "{spread}");
}

#[test]
fn splitting_follows_enumeration_at_rates_of_ten_to_the_minus_nine() {
    // The surface code of distance 3 over three rounds at 10⁻⁶ (about
    // 8·10⁻¹⁰), all configurations to five errors.
    let at = surface(3);
    let exact = enumerated_failure_rate(&at(1e-6).unwrap(), DecoderChoice::Matching, 5).unwrap();
    assert!(exact.remainder < 1e-9 * exact.rate);
    let est =
        splitting_failure_rate(&at, DecoderChoice::Matching, 0.01, 1e-6, &options(4)).unwrap();
    let z = (est.rate.ln() - exact.rate.ln()) / est.log_standard_error;
    assert!(z.abs() < 4.0, "{} vs {}: {z}", est.rate, exact.rate);
}

#[test]
fn the_leading_coefficient_counts_the_failing_configurations_of_least_weight() {
    // Enumerated failing configurations of ⌈d/2⌉ errors against the
    // subsets of the logical errors of at most d + 1 errors.
    for (name, circuit) in [
        ("repetition 3", repetition(3)(1e-3).unwrap()),
        ("repetition 5", repetition(5)(1e-3).unwrap()),
        ("surface 3", surface(3)(1e-3).unwrap()),
        (
            "surface 3, data only",
            surface_code_memory(3, 1, &CircuitNoise::data_only(1e-3)).unwrap(),
        ),
    ] {
        let lead = leading_failures(&circuit, DecoderChoice::Matching, 1 << 30, 1 << 30).unwrap();
        assert!(lead.exhaustive);
        // From the subsets of the logical errors alone, the same here.
        let subsets = leading_failures(&circuit, DecoderChoice::Matching, 1 << 30, 0).unwrap();
        assert!(!subsets.exhaustive);
        assert_eq!(subsets.configurations, lead.configurations, "{name}");
        let exact =
            enumerated_failure_rate(&circuit, DecoderChoice::Matching, lead.weight).unwrap();
        eprintln!("{name}: {lead:?}, enumerated {:?}", exact.failing);
        for k in 0..lead.weight {
            assert_eq!(exact.failing[k], 0, "{name}: weight {k}");
        }
        assert_eq!(
            exact.failing[lead.weight] as usize, lead.configurations,
            "{name}"
        );
        let w = exact.by_weight[lead.weight];
        assert!(
            (lead.probability - w).abs() < 1e-12 * w,
            "{name}: {} vs {w}",
            lead.probability
        );
        // The leading term leaves out Π(1 − p) of the other errors.
        let dem = detector_error_model(&circuit, false).unwrap();
        let none: f64 = dem.errors.iter().map(|e| 1.0 - e.probability).product();
        let ratio = w / (none * lead.leading_term);
        assert!((ratio - 1.0).abs() < 0.05, "{name}: {ratio}");
    }
}

#[test]
fn unequal_weights_and_bposd_need_the_exhaustive_count() {
    // Measurement errors 10⁴ times likelier than gate errors: two errors
    // fail the repetition code of distance 5, not three. And BP+OSD on
    // [[72, 12, 6]] with data noise fails on configurations outside the
    // logical errors of six errors: the subsets count fewer.
    let noisy = CircuitNoise {
        after_gate2: 1e-5,
        measurement: 0.05,
        ..CircuitNoise::default()
    };
    let rep5 = repetition_code_memory(5, 3, &noisy).unwrap();
    let lead = leading_failures(&rep5, DecoderChoice::Matching, 1 << 30, 1 << 30).unwrap();
    let exact = enumerated_failure_rate(&rep5, DecoderChoice::Matching, 3).unwrap();
    assert!(lead.exhaustive && lead.weight == 2, "{lead:?}");
    assert_eq!(lead.configurations as u64, exact.failing[2]);
    let (code, _) = crate::qec::CssCode::bivariate_bicycle_code(72).unwrap();
    let bb = crate::qec::css_code_memory(
        &code,
        1,
        &CircuitNoise::data_only(1e-3),
        crate::qec::MemoryBasis::Z,
    )
    .unwrap();
    let all = leading_failures(&bb, DecoderChoice::BpOsd, 1 << 30, 1 << 30).unwrap();
    let subsets = leading_failures(&bb, DecoderChoice::BpOsd, 1 << 30, 0).unwrap();
    let exact = enumerated_failure_rate(&bb, DecoderChoice::BpOsd, all.weight).unwrap();
    assert_eq!(all.configurations as u64, exact.failing[all.weight]);
    assert!(
        subsets.configurations < all.configurations,
        "{subsets:?} vs {all:?}"
    );
}

#[test]
fn inputs_are_refused() {
    let at = repetition(3);
    let o = SplittingOptions::default();
    assert!(splitting_failure_rate(&at, DecoderChoice::Matching, 1e-3, 1e-2, &o).is_err());
    let one = SplittingOptions { chains: 1, ..o };
    assert!(splitting_failure_rate(&at, DecoderChoice::Matching, 0.03, 1e-3, &one).is_err());
    let huge = SplittingOptions {
        steps: usize::MAX / 64,
        ..o
    };
    assert!(splitting_failure_rate(&at, DecoderChoice::Matching, 0.03, 1e-3, &huge).is_err());
    // Weight zero: the empty configuration alone.
    let zero = enumerated_failure_rate(&at(0.01).unwrap(), DecoderChoice::Matching, 0).unwrap();
    assert_eq!((zero.configurations, zero.rate), (1, 0.0));
    // No failure at the start.
    let few = SplittingOptions {
        start_shots: 10,
        ..o
    };
    let err = splitting_failure_rate(&at, DecoderChoice::Matching, 1e-5, 1e-6, &few).unwrap_err();
    assert!(err.to_string().contains("no failure"), "{err}");
    // A circuit whose errors change with the rate.
    let changing =
        |p: f64| repetition_code_memory(if p > 0.01 { 3 } else { 5 }, 2, &CircuitNoise::uniform(p));
    let err =
        splitting_failure_rate(&changing, DecoderChoice::Matching, 0.03, 1e-3, &o).unwrap_err();
    assert!(err.to_string().contains("other errors"), "{err}");
    // The start alone.
    let same = splitting_failure_rate(&at, DecoderChoice::Matching, 0.05, 0.05, &o).unwrap();
    assert!(same.rates.is_empty() && same.rate == same.start.rate);
    // Enumeration beyond the limit.
    let big = surface(3)(0.01).unwrap();
    assert!(enumerated_failure_rate(&big, DecoderChoice::Matching, 40).is_err());
}

/// Surface codes of distance 3 to 7 with circuit noise, their leading
/// terms, and a bivariate bicycle code under BP+OSD, for the documentation
/// (`cargo test --release -- --ignored measure_splitting --nocapture`).
#[test]
#[ignore]
fn measure_splitting() {
    let run = |name: &str,
               at: &(dyn Fn(f64) -> Result<DetectorCircuit> + Sync),
               decoder: DecoderChoice,
               start: f64,
               target: f64,
               o: &SplittingOptions| {
        let clock = std::time::Instant::now();
        let est = splitting_failure_rate(at, decoder, start, target, o).unwrap();
        eprintln!(
            "{name}: p = {target:e}: {:.3e} ± {:.1} % ({} rates from {start}, start {:.3e}, {} \
             decodes, {:.1} s)",
            est.rate,
            100.0 * est.log_standard_error,
            est.rates.len(),
            est.start.rate,
            est.decodes,
            clock.elapsed().as_secs_f64()
        );
        est
    };
    let o = SplittingOptions::default();
    for d in [3, 5, 7] {
        let at = move |p: f64| surface_code_memory(d, d, &CircuitNoise::uniform(p));
        for target in [1e-3, 1e-4] {
            run(
                &format!("surface d = {d}"),
                &at,
                DecoderChoice::Matching,
                0.01,
                target,
                &o,
            );
        }
    }
    for d in [3, 5] {
        let p = 1e-4;
        let circuit = surface_code_memory(d, d, &CircuitNoise::uniform(p)).unwrap();
        let clock = std::time::Instant::now();
        let lead = leading_failures(&circuit, DecoderChoice::Matching, 1 << 34, 0).unwrap();
        eprintln!(
            "surface d = {d}, p = {p:e}: leading {} configurations of {} errors among {} logical \
             errors, term {:.3e}, probability {:.3e} ({:.1} s)",
            lead.configurations,
            lead.weight,
            lead.logicals,
            lead.leading_term,
            lead.probability,
            clock.elapsed().as_secs_f64()
        );
    }
    let (code, _) = crate::qec::CssCode::bivariate_bicycle_code(72).unwrap();
    let at = move |p: f64| {
        crate::qec::css_code_memory(
            &code,
            1,
            &CircuitNoise::data_only(p),
            crate::qec::MemoryBasis::Z,
        )
    };
    let fewer = SplittingOptions {
        start_shots: 20_000,
        ..o
    };
    for target in [1e-2, 1e-3] {
        run(
            "[[72,12,6]], data only, BP+OSD",
            &at,
            DecoderChoice::BpOsd,
            0.06,
            target,
            &fewer,
        );
    }
    // [[72, 12, 6]] at 10⁻³ and 10⁻⁵ against enumeration to four errors.
    for target in [1e-3, 1e-5] {
        let clock = std::time::Instant::now();
        let e = enumerated_failure_rate(&at(target).unwrap(), DecoderChoice::BpOsd, 4).unwrap();
        eprintln!(
            "[[72,12,6]], p = {target:e}: enumerated [{:.4e}, {:.4e}] ({:.1} s)",
            e.rate,
            e.rate + e.remainder,
            clock.elapsed().as_secs_f64()
        );
    }
    run(
        "[[72,12,6]], data only, BP+OSD",
        &at,
        DecoderChoice::BpOsd,
        0.06,
        1e-5,
        &fewer,
    );
    // Eight seeds of the surface code of distance 5 at 10⁻⁴, with the
    // default chains and with shorter ones.
    let d5 = |p: f64| surface_code_memory(5, 5, &CircuitNoise::uniform(p));
    for (steps, burn_in) in [(20_000, 5000), (5000, 1000)] {
        let estimates: Vec<(f64, f64)> = (0..8)
            .map(|seed| {
                let o = SplittingOptions {
                    steps,
                    burn_in,
                    seed,
                    ..o
                };
                let e =
                    splitting_failure_rate(&d5, DecoderChoice::Matching, 0.01, 1e-4, &o).unwrap();
                (e.rate, e.log_standard_error)
            })
            .collect();
        let logs: Vec<f64> = estimates.iter().map(|e| e.0.ln()).collect();
        let mean = logs.iter().sum::<f64>() / 8.0;
        let spread = (logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / 7.0).sqrt();
        let stated = estimates.iter().map(|e| e.1).sum::<f64>() / 8.0;
        eprintln!(
            "surface d = 5, p = 1e-4, {steps} steps after {burn_in}: {:.3e} to {:.3e}, spread \
             {:.1} % against {:.1} % stated",
            estimates.iter().map(|e| e.0).fold(f64::INFINITY, f64::min),
            estimates.iter().map(|e| e.0).fold(0.0, f64::max),
            100.0 * spread,
            100.0 * stated
        );
    }
    // Direct sampling where it reaches.
    for (name, circuit, decoder, shots) in [
        (
            "surface d = 3, p = 1e-3",
            surface_code_memory(3, 3, &CircuitNoise::uniform(1e-3)).unwrap(),
            DecoderChoice::Matching,
            2_000_000,
        ),
        (
            "surface d = 5, p = 1e-3",
            surface_code_memory(5, 5, &CircuitNoise::uniform(1e-3)).unwrap(),
            DecoderChoice::Matching,
            2_000_000,
        ),
        (
            "[[72,12,6]], p = 1e-2",
            at(1e-2).unwrap(),
            DecoderChoice::BpOsd,
            200_000,
        ),
    ] {
        let clock = std::time::Instant::now();
        let r = logical_error_rate_using(&circuit, shots, 5, decoder).unwrap();
        eprintln!(
            "{name}: sampled {:.3e} ± {:.1} % ({shots} shots, {:.1} s)",
            r.rate,
            100.0 * r.standard_error / r.rate,
            clock.elapsed().as_secs_f64()
        );
    }
}
