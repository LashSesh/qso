//! Pauli twirling of two-qubit Clifford gates.
//!
//! A random Pauli `P` before a Clifford gate `G` and `G P G†` after it
//! leave the ideal circuit unchanged (up to a global phase). Averaged over
//! the random Paulis, the noise after `G` acts as its *Pauli twirl*
//! ([`Channel::twirled`](crate::noise::Channel::twirled)): coherent errors
//! and amplitude damping become stochastic Pauli noise, which
//! probabilistic error cancellation can invert and which degrades results
//! predictably. Twirled gates: `cx`, `cy`, `cz` and `swap`.
//!
//! The inserted Paulis are separate single-qubit gates. On hardware they
//! are merged into neighbouring single-qubit gates; in simulation a device
//! rule for one-qubit gates adds its noise to them.
//!
//! ```
//! use metatron_qso::mitigation::twirl::twirl;
//! use metatron_qso::quantum::{Circuit, StateVector};
//! use metatron_qso::Rng;
//!
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1).rz(1, 0.4).cz(0, 1);
//! let t = twirl(&c, &mut Rng::seed_from_u64(1));
//! let a = StateVector::from_circuit(&c, &[]).unwrap();
//! let b = StateVector::from_circuit(&t, &[]).unwrap();
//! assert!((a.fidelity(&b).unwrap() - 1.0).abs() < 1e-12);
//! ```

use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::quantum::{Circuit, GateKind, Instruction};
use crate::rng::Rng;

/// A two-qubit Pauli in symplectic form: `(x_a, z_a, x_b, z_b)`.
type Symplectic = [bool; 4];

/// Conjugation `P ↦ G P G†` of a Clifford gate in symplectic form.
type Conjugation = fn(Symplectic) -> Symplectic;

/// The twirlable gates: `(first qubit, second qubit, conjugation)`.
fn clifford(inst: &Instruction) -> Option<(usize, usize, Conjugation)> {
    fn cx([xa, za, xb, zb]: Symplectic) -> Symplectic {
        [xa, za ^ zb, xb ^ xa, zb]
    }
    fn cz([xa, za, xb, zb]: Symplectic) -> Symplectic {
        [xa, za ^ xb, xb, zb ^ xa]
    }
    fn cy([xa, za, xb, zb]: Symplectic) -> Symplectic {
        // CY = S_b · CX · S_b†; S conjugation maps z_b ↦ z_b ⊕ x_b.
        let [xa, za, xb, zb] = cx([xa, za, xb, zb ^ xb]);
        [xa, za, xb, zb ^ xb]
    }
    fn swap([xa, za, xb, zb]: Symplectic) -> Symplectic {
        [xb, zb, xa, za]
    }
    match (inst.kind, inst.controls.as_slice(), inst.targets.as_slice()) {
        (GateKind::X, [c], [t]) => Some((*c, *t, cx)),
        (GateKind::Y, [c], [t]) => Some((*c, *t, cy)),
        (GateKind::Z, [c], [t]) => Some((*c, *t, cz)),
        (GateKind::Swap, [], [a, b]) => Some((*a, *b, swap)),
        _ => None,
    }
}

fn push_pauli(c: &mut Circuit, q: usize, x: bool, z: bool) {
    match (x, z) {
        (true, false) => {
            c.x(q);
        }
        (true, true) => {
            c.y(q);
        }
        (false, true) => {
            c.z(q);
        }
        (false, false) => {}
    }
}

/// Twirl with Pauli indices from `choose` (0..16: bits `x_a z_a x_b z_b`).
fn twirl_with(circuit: &Circuit, mut choose: impl FnMut() -> usize) -> Circuit {
    let mut out = Circuit::new(circuit.n_qubits());
    out.reserve_parameters(circuit.num_parameters());
    for inst in circuit.instructions() {
        match clifford(inst) {
            Some((a, b, conjugate)) => {
                let k = choose();
                let p = [k & 8 != 0, k & 4 != 0, k & 2 != 0, k & 1 != 0];
                let after = conjugate(p);
                push_pauli(&mut out, a, p[0], p[1]);
                push_pauli(&mut out, b, p[2], p[3]);
                out.try_push(inst.clone()).expect("valid instruction");
                push_pauli(&mut out, a, after[0], after[1]);
                push_pauli(&mut out, b, after[2], after[3]);
            }
            None => {
                out.try_push(inst.clone()).expect("valid instruction");
            }
        }
    }
    out
}

/// `circuit` with a random Pauli frame around every `cx`, `cy`, `cz` and
/// `swap`; the ideal circuit is unchanged up to a global phase.
pub fn twirl(circuit: &Circuit, rng: &mut Rng) -> Circuit {
    twirl_with(circuit, || rng.below(16) as usize)
}

/// Average of `execute` over `count` independently twirled copies of
/// `circuit`. The standard error is the spread over the copies, which
/// includes the statistical error of each evaluation.
pub fn twirled_average<F>(
    circuit: &Circuit,
    count: usize,
    seed: u64,
    mut execute: F,
) -> Result<Estimate>
where
    F: FnMut(&Circuit) -> Result<Estimate>,
{
    if count == 0 {
        return Err(QsoError::invalid(
            "count",
            "need at least one twirled circuit",
        ));
    }
    let base = Rng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(count);
    let mut samples = 0;
    for k in 0..count {
        let e = execute(&twirl(circuit, &mut base.fork(k as u64)))?;
        values.push(e.value);
        samples += e.samples;
    }
    let mut e = Estimate::from_samples(&values);
    e.samples = samples.max(count);
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::{ChannelSpec, DensityMatrix, DeviceModel, GateRule};
    use crate::quantum::{PauliSum, StateVector};

    #[test]
    fn twirled_circuits_are_the_ideal_circuit() {
        let mut c = Circuit::new(3);
        let t = c.new_parameter();
        c.h(0)
            .cx(0, 1)
            .ry(2, t)
            .cy(2, 0)
            .cz(1, 2)
            .swap(0, 2)
            .crz(0, 1, 0.3)
            .cx(2, 1);
        let ideal = StateVector::from_circuit(&c, &[0.8]).unwrap();
        let mut rng = Rng::seed_from_u64(2);
        for _ in 0..50 {
            let tw = twirl(&c, &mut rng);
            let psi = StateVector::from_circuit(&tw, &[0.8]).unwrap();
            assert!((psi.fidelity(&ideal).unwrap() - 1.0).abs() < 1e-12);
        }
        // Every frame is valid, not only random ones.
        for k in 0..16 {
            let tw = twirl_with(&c, || k);
            let psi = StateVector::from_circuit(&tw, &[0.8]).unwrap();
            assert!((psi.fidelity(&ideal).unwrap() - 1.0).abs() < 1e-12, "{k}");
        }
    }

    #[test]
    fn twirling_averages_noise_to_its_pauli_twirl() {
        let mut c = Circuit::new(2);
        c.ry(0, 0.9).h(1).cx(0, 1).rx(1, 0.4);
        let theta: [f64; 0] = [];
        for spec in [
            ChannelSpec::Rotation {
                pauli: "ZZ".into(),
                angle: 0.4,
            },
            ChannelSpec::AmplitudeDamping(0.2),
        ] {
            let device = DeviceModel::new("cx")
                .with_rule(GateRule::for_gate("cx").with_channel(spec.clone()));
            // Exact average over all 16 frames.
            let mut avg = DensityMatrix::zero(2).unwrap();
            let mut first = true;
            for k in 0..16 {
                let mut rho = DensityMatrix::zero(2).unwrap();
                rho.apply_device(&twirl_with(&c, || k), &theta, &device)
                    .unwrap();
                if first {
                    avg = rho.clone();
                    avg.scale(1.0 / 16.0);
                    first = false;
                } else {
                    avg.add_scaled(&rho, 1.0 / 16.0);
                }
            }
            // The same circuit with the twirled noise (both qubits for a
            // one-qubit channel: the twirl of a product is the product of
            // the twirls).
            let twirled = spec.to_channel().unwrap().twirled();
            let device = DeviceModel::new("twirled")
                .with_rule(GateRule::for_gate("cx").with_channel(ChannelSpec::from(&twirled)));
            let mut expected = DensityMatrix::zero(2).unwrap();
            expected.apply_device(&c, &theta, &device).unwrap();
            let diff = (avg.to_matrix() - expected.to_matrix())
                .iter()
                .map(|z| z.norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-14, "{spec:?}: {diff}");
        }
    }

    #[test]
    fn random_twirls_estimate_the_twirled_value() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0);
        let zz = ChannelSpec::Rotation {
            pauli: "ZZ".into(),
            angle: 0.5,
        };
        let device = DeviceModel::new("coherent")
            .with_rule(GateRule::for_gate("cx").with_channel(zz.clone()));
        let h: PauliSum = "Z0".parse().unwrap();
        let exact = |dev: &DeviceModel, circ: &Circuit| -> f64 {
            let mut rho = DensityMatrix::zero(2).unwrap();
            rho.apply_device(circ, &[], dev).unwrap();
            rho.expectation(&h).unwrap()
        };
        let twirled_device = DeviceModel::new("twirled").with_rule(
            GateRule::for_gate("cx")
                .with_channel(ChannelSpec::from(&zz.to_channel().unwrap().twirled())),
        );
        let target = exact(&twirled_device, &c);
        let coherent = exact(&device, &c);
        assert!((target - coherent).abs() > 0.05);
        let e = twirled_average(&c, 2000, 9, |t| Ok(Estimate::exact(exact(&device, t)))).unwrap();
        assert!(
            (e.value - target).abs() < 4.0 * e.std_error,
            "{} vs {target}",
            e.value
        );
        assert!(twirled_average(&c, 0, 9, |_| Ok(Estimate::exact(0.0))).is_err());
        let ideal = StateVector::from_circuit(&c, &[]).unwrap();
        assert!((h.expectation(&ideal).unwrap() - 0.3f64.cos()).abs() < 1e-12);
    }
}
