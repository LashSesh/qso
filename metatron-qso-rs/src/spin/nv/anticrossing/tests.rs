//! The NV centre with its nuclei near the anticrossings.

use super::*;

#[test]
#[ignore]
fn explore() {
    let pump = 0.1 * 65.9e6;
    for &b in &[
        0.005, 0.02, 0.035, 0.044, 0.048, 0.05, 0.051, 0.052, 0.055, 0.07, 0.1,
    ] {
        let nv = NuclearOpticalNv::new([0.0, 0.0, b], NuclearSpin::nitrogen_15());
        let s = nv.steady_state(pump, None).unwrap();
        eprintln!(
            "15N B = {b:.3}: P = {:.4}, F = {:.4e}",
            s.polarisation[0], s.fluorescence
        );
    }
    for &b in &[
        0.098, 0.1, 0.101, 0.102, 0.1022, 0.1024, 0.1026, 0.103, 0.105,
    ] {
        let nv = NuclearOpticalNv::new([0.0, 0.0, b], NuclearSpin::nitrogen_14());
        let s = nv.steady_state(pump, None).unwrap();
        eprintln!(
            "14N B = {b:.4}: P = {:.4}, F = {:.6e}",
            s.polarisation[0], s.fluorescence
        );
    }
}
