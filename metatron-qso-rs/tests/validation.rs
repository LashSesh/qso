//! Validation suite: every quantitative claim in the README is reproduced
//! here through the public API. If a number in the documentation changes,
//! a test in this file must change with it.

use std::f64::consts::PI;

use metatron_qso::algorithms::qaoa::optimize_layerwise;
use metatron_qso::prelude::*;
use metatron_qso::walks::classical_hitting_times;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn metatron_cube_is_k13_with_symmetric_group_s13() {
    let cube = metatron::metatron_cube();
    assert_eq!((cube.num_nodes(), cube.num_edges()), (13, 78));
    assert_eq!(cube.automorphism_group().order(), Some(6_227_020_800));
    let spectrum = cube.laplacian_spectrum();
    assert!(close(spectrum[0], 0.0, 1e-12));
    assert!(spectrum[1..].iter().all(|&x| close(x, 13.0, 1e-10)));
}

#[test]
fn fruit_of_life_has_hexagonal_symmetry() {
    let fol = metatron::fruit_of_life();
    assert_eq!((fol.num_nodes(), fol.num_edges()), (13, 18));
    let group = fol.automorphism_group();
    assert_eq!(group.order(), Some(12));
    assert_eq!(group.orbits().len(), 3);
}

#[test]
fn platonic_solid_symmetry_groups() {
    use metatron_qso::graph::generators::{PlatonicSolid, platonic};
    let orders: Vec<u128> = PlatonicSolid::ALL
        .into_iter()
        .map(|s| platonic(s).automorphism_group().order().unwrap())
        .collect();
    assert_eq!(orders, vec![24, 48, 48, 120, 120]);
}

#[test]
fn ctqw_on_metatron_cube_matches_closed_form() {
    let walk = ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0).unwrap();
    let limit = walk.limiting_distribution(0).unwrap();
    assert!(close(limit[0], 145.0 / 169.0, 1e-12));
    assert!(limit[1..].iter().all(|&p| close(p, 2.0 / 169.0, 1e-12)));
}

#[test]
fn spatial_search_on_metatron_cube_is_exact() {
    let s = SpatialSearch::new(
        &metatron::metatron_cube(),
        &[3],
        Generator::Laplacian,
        1.0 / 13.0,
    )
    .unwrap();
    assert!(close(
        s.success_probability(PI * 13f64.sqrt() / 2.0),
        1.0,
        1e-10
    ));
    // Classical random walk needs 12 steps on average.
    let h = classical_hitting_times(&metatron::metatron_cube(), 3).unwrap();
    assert!(close(h[0], 12.0, 1e-9));
}

#[test]
fn vqe_h2_reaches_full_ci_energy() {
    let h2 = models::h2_sto3g();
    let r = Vqe::new(
        h2.hamiltonian.clone(),
        ansatz::real_amplitudes(2, 1, Entanglement::Linear),
    )
    .unwrap()
    .with_restarts(4)
    .run()
    .unwrap();
    assert!(close(r.energy + h2.nuclear_repulsion, -1.137306, 1e-6));
}

#[test]
fn qaoa_reference_ratios() {
    let ring = MaxCutQaoa::new(&generators::cycle(10), 1)
        .unwrap()
        .optimize(&QaoaConfig::default())
        .unwrap();
    assert!(close(ring.approximation_ratio.unwrap(), 0.75, 1e-7));
    let cube = MaxCutQaoa::new(&generators::hypercube(3), 1)
        .unwrap()
        .optimize(&QaoaConfig::default())
        .unwrap();
    assert!(close(cube.approximation_ratio.unwrap(), 0.692450, 2e-6));
}

#[test]
fn qaoa_on_metatron_cube_converges_with_depth() {
    let results =
        optimize_layerwise(&metatron::metatron_cube(), 3, &QaoaConfig::default()).unwrap();
    assert_eq!(results[0].max_cut, 42.0);
    let ratios: Vec<f64> = results
        .iter()
        .map(|r| r.approximation_ratio.unwrap())
        .collect();
    assert!(ratios.windows(2).all(|w| w[1] >= w[0] - 1e-12));
    assert!(ratios[2] > 0.9999, "{ratios:?}");
}

#[test]
fn grover_follows_the_sine_law() {
    let r = grover::grover_search(10, &[123], None).unwrap();
    assert_eq!(r.iterations, 25);
    assert!(close(
        r.success_probability,
        r.theoretical_success_probability,
        1e-12
    ));
    assert!(r.success_probability > 0.999);
}

#[test]
fn qasm_round_trip_and_noise() {
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2).rz(2, 0.3).ry(1, -0.2);
    let parsed = from_qasm2(&to_qasm2(&c, &[], true).unwrap()).unwrap();
    let a = StateVector::from_circuit(&c, &[]).unwrap();
    let b = StateVector::from_circuit(&parsed, &[]).unwrap();
    assert!(close(a.fidelity(&b).unwrap(), 1.0, 1e-12));

    let noisy = DensityMatrixBackend::new(NoiseModel::depolarizing(0.01, 0.05).unwrap());
    let zz: PauliSum = "Z0 Z1".parse().unwrap();
    let mut bell = Circuit::new(2);
    bell.h(0).cx(0, 1);
    let v = estimate_expectation(&noisy, &bell, &[], &zz, 20_000, 1).unwrap();
    assert!(v < 1.0 && v > 0.8);
}

#[test]
fn qaoa_ring_of_disagrees_reaches_proven_optimum_at_every_depth() {
    // Mbeng, Fazio & Santoro (arXiv:1906.08948): on a ring with n ≥ 2p + 2
    // nodes, depth-p QAOA attains exactly (2p + 1)/(2p + 2).
    let results = optimize_layerwise(&generators::cycle(10), 4, &QaoaConfig::default()).unwrap();
    for r in &results {
        let expected = (2 * r.p + 1) as f64 / (2 * r.p + 2) as f64;
        assert!(
            close(r.approximation_ratio.unwrap(), expected, 1e-6),
            "p={}: {r:?}",
            r.p
        );
    }
}
