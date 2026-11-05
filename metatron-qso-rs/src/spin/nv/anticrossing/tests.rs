//! The NV centre with its nuclei near the anticrossings.

use super::*;

#[test]
#[ignore]
fn explore() {
    let pump = 0.1 * 65.9e6;
    for k72 in [0.7e6, 0.3e6, 0.1e6, 0.0] {
        let mut nv = NuclearOpticalNv::new([0.0, 0.0, 0.0507], NuclearSpin::nitrogen_15());
        nv.electron.rates.singlet_plus_minus = k72;
        let s = nv.steady_state(pump, None).unwrap();
        let nn = 2;
        let pop = |e: usize| {
            (0..nn)
                .map(|m| s.rho[(e * nn + m, e * nn + m)].re)
                .sum::<f64>()
        };
        let g: f64 = (0..3).map(pop).sum();
        eprintln!(
            "k72 = {k72:.1e}: P = {:.4}, ground m_s = +1, 0, -1: {:.4} {:.4} {:.4}",
            s.polarisation[0],
            pop(0) / g,
            pop(1) / g,
            pop(2) / g
        );
    }
    for k in 0..=40 {
        let b = 0.1022 + 0.00001 * k as f64;
        let nv = NuclearOpticalNv::new([0.0, 0.0, b], NuclearSpin::nitrogen_14());
        let s = nv.steady_state(pump, None).unwrap();
        let mut tilted = nv.clone();
        tilted.electron.field = [1e-4, 0.0, b];
        let t = tilted.steady_state(pump, None).unwrap();
        eprintln!(
            "14N B = {:.5}: F = {:.6e}, P = {:.4}; tilted 0.1 mT: F = {:.6e}",
            b, s.fluorescence, s.polarisation[0], t.fluorescence
        );
    }
}
