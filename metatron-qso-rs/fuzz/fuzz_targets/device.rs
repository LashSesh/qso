//! Device descriptions (JSON): no input may panic; every accepted device
//! must survive a JSON round trip unchanged and give a valid density matrix
//! (unit trace, Hermitian, finite) on a small fixed circuit.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::noise::{DensityMatrix, DeviceModel};
use metatron_qso::quantum::Circuit;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(device) = DeviceModel::from_json(text) else {
        return;
    };
    let again = DeviceModel::from_json(&device.to_json()).expect("round trip");
    assert_eq!(again, device);
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).rz(1, 0.3).crx(1, 2, 0.7).swap(0, 2).sx(1);
    let mut rho = DensityMatrix::zero(3).expect("3 qubits");
    if rho.apply_device(&c, &[], &device).is_err() {
        return;
    }
    assert!((rho.trace() - 1.0).abs() < 1e-9, "trace {}", rho.trace());
    let m = rho.to_matrix();
    assert!(m.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    assert!((&m - m.adjoint()).iter().all(|z| z.norm() < 1e-9));
    assert!(rho.probabilities().iter().all(|&p| p >= 0.0));
});
