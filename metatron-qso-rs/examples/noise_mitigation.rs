//! Noise at scale and error mitigation on a toy device.
//!
//! Run with `cargo run --release --example noise_mitigation`. The device
//! (`examples/devices/toy_device.json`) has per-qubit T1/T2 and readout
//! errors, depolarising and ZZ errors on its CX gates and small
//! over-rotations. The example prepares a 5-qubit GHZ state and estimates
//! `H = Σ Z_i Z_{i+1}` (noiseless value 4):
//!
//! 1. exactly with density matrices and with quantum trajectories;
//! 2. from shots, raw and with readout mitigation, zero-noise
//!    extrapolation and both;
//! 3. with probabilistic error cancellation on a Pauli-noise version of the
//!    device, and from a classical shadow;
//! 4. with trajectories on a 20-qubit GHZ state, beyond density matrices.

use metatron_qso::Result;
use metatron_qso::backend::DeviceBackend;
use metatron_qso::estimation::{ClassicalShadow, Estimate, EstimatorOptions, estimate};
use metatron_qso::mitigation::ReadoutCalibration;
use metatron_qso::mitigation::pec::pec_expectation;
use metatron_qso::mitigation::zne::{Extrapolation, Folding, zne};
use metatron_qso::noise::{
    ChannelSpec, DensityMatrix, DeviceModel, GateRule, QubitProperties, TrajectorySimulator,
};
use metatron_qso::quantum::{Circuit, PauliSum};

fn ghz(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 0..n - 1 {
        c.cx(q, q + 1);
    }
    c
}

fn chain(n: usize) -> PauliSum {
    let terms: Vec<String> = (0..n - 1).map(|q| format!("Z{q} Z{}", q + 1)).collect();
    terms.join(" + ").parse().expect("valid observable")
}

fn row(label: &str, e: &Estimate, exact: f64) {
    println!(
        "| {label} | {:.4} ± {:.4} | {:+.4} |",
        e.value,
        e.std_error,
        e.value - exact
    );
}

fn main() -> Result<()> {
    let device = DeviceModel::from_json(include_str!("devices/toy_device.json"))?;
    let n = 5;
    let c = ghz(n);
    let h = chain(n);
    let ideal = (n - 1) as f64;

    let mut rho = DensityMatrix::zero(n)?;
    rho.apply_device(&c, &[], &device)?;
    let noisy = rho.expectation(&h)?;
    let trajectories =
        TrajectorySimulator::new(device.clone(), 4000, 1)?.expectation(&c, &[], &h)?;

    println!(
        "Device {:?}: 5-qubit GHZ state, H = Σ Z_i Z_(i+1), noiseless ⟨H⟩ = {ideal}\n",
        device.name
    );
    println!("| method | ⟨H⟩ | error |");
    println!("|---|---:|---:|");
    row(
        "density matrix (exact, no readout error)",
        &Estimate::exact(noisy),
        ideal,
    );
    row("4000 trajectories", &trajectories, ideal);

    // Shots on the device include its readout errors.
    let backend = DeviceBackend::density_matrix(device.clone())?;
    let shots = 200_000;
    let options = EstimatorOptions::new(shots).with_seed(2);
    let raw = estimate(&backend, &c, &[], &h, &options)?.estimate;
    row("shots, raw", &raw, ideal);
    let calibration = ReadoutCalibration::tensored(&backend, n, shots, 3)?;
    let mitigated_options = options.clone().with_readout_mitigation(calibration);
    let readout = estimate(&backend, &c, &[], &h, &mitigated_options)?.estimate;
    row("shots, readout mitigated", &readout, ideal);
    for (label, opts) in [
        ("shots, ZNE (1, 3, 5)", &options),
        ("shots, readout + ZNE", &mitigated_options),
    ] {
        let mut k = 10;
        let r = zne(
            &c,
            &[1.0, 3.0, 5.0],
            Folding::Global,
            Extrapolation::Richardson,
            |folded| {
                k += 1;
                Ok(estimate(&backend, folded, &[], &h, &opts.clone().with_seed(k))?.estimate)
            },
        )?;
        row(label, &r.estimate, ideal);
    }

    // PEC needs Pauli noise: the device's CX errors as Pauli channels.
    let pauli = DeviceModel::new("toy, Pauli noise")
        .with_rule(
            GateRule::for_gate("cx")
                .with_depolarizing(0.008)
                .with_channel(ChannelSpec::Pauli([("ZZ".to_string(), 0.002)].into())),
        )
        .with_rule(GateRule::any().with_num_qubits(1).with_depolarizing(0.0005));
    let mut rho = DensityMatrix::zero(n)?;
    rho.apply_device(&c, &[], &pauli)?;
    let pec = pec_expectation(&c, &[], &pauli, &h, 20_000, 4)?;
    row(
        "Pauli noise, density matrix",
        &Estimate::exact(rho.expectation(&h)?),
        ideal,
    );
    row(
        &format!("Pauli noise, PEC (overhead {:.3})", pec.overhead),
        &pec.estimate,
        ideal,
    );

    let shadow = ClassicalShadow::collect(&backend, &c, &[], 20_000, 5)?;
    row(
        "classical shadow, 20 000 snapshots",
        &shadow.estimate(&h)?,
        ideal,
    );

    // Beyond density matrices: 20 qubits with trajectories.
    let big = 20;
    let wide = DeviceModel::new("uniform")
        .with_default_qubit(QubitProperties::new().with_t1_t2(100.0, 80.0))
        .with_rule(
            GateRule::for_gate("cx")
                .with_duration(0.3)
                .with_depolarizing(0.008),
        )
        .with_rule(
            GateRule::any()
                .with_duration(0.035)
                .with_depolarizing(0.0005),
        );
    let start = std::time::Instant::now();
    let e = TrajectorySimulator::new(wide, 32, 6)?.expectation(&ghz(big), &[], &chain(big))?;
    println!(
        "\n20-qubit GHZ state on a uniform device: ⟨H⟩ = {:.3} ± {:.3} (noiseless {}), 32 trajectories in {:.1} s",
        e.value,
        e.std_error,
        big - 1,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
