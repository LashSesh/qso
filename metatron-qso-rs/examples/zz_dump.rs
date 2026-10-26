use metatron_qso::qec::*;
use std::fmt::Write;
fn main() -> metatron_qso::Result<()> {
    let dir = std::env::args().nth(1).unwrap();
    let cases: Vec<(&str, DetectorCircuit)> = vec![
        (
            "rep5",
            repetition_code_memory(5, 4, &CircuitNoise::uniform(0.02))?,
        ),
        (
            "surf3",
            surface_code_memory(3, 3, &CircuitNoise::uniform(0.01))?,
        ),
        (
            "surf5",
            surface_code_memory(5, 5, &CircuitNoise::uniform(0.003))?,
        ),
        (
            "surf5odd",
            surface_code_memory(
                5,
                2,
                &CircuitNoise {
                    after_gate1: 0.004,
                    after_gate2: 0.01,
                    measurement: 0.02,
                    after_reset: 0.003,
                    before_round: 0.005,
                },
            )?,
        ),
    ];
    for (name, c) in cases {
        std::fs::write(format!("{dir}/{name}.stim"), c.to_stim()?).unwrap();
        for (tag, dec) in [("full", false), ("dec", true)] {
            let dem = detector_error_model(&c, dec)?;
            let mut s = String::new();
            for e in &dem.errors {
                writeln!(
                    s,
                    "{:.17e} {:?} {}",
                    e.probability, e.detectors, e.observables
                )
                .unwrap();
            }
            std::fs::write(format!("{dir}/{name}.{tag}.dem"), s).unwrap();
        }
        let shots = 1_000_000;
        let smp = sample_detectors(&c, shots, 11)?;
        let rates: Vec<String> = (0..c.n_detectors())
            .map(|d| format!("{}", smp.count(d) as f64 / shots as f64))
            .collect();
        std::fs::write(format!("{dir}/{name}.rates"), rates.join("\n")).unwrap();
        let r = logical_error_rate(&c, 200_000, 5)?;
        std::fs::write(
            format!("{dir}/{name}.ler"),
            format!("{} {}", r.rate, r.standard_error),
        )
        .unwrap();
    }
    Ok(())
}
