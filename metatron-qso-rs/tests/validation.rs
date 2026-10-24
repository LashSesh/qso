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

#[test]
fn qaoa_p2_on_large_girth_cubic_graphs_reaches_the_tree_value() {
    // For cubic graphs of girth > 2p + 1 every edge sees a tree, so the
    // optimal per-edge expectation is a universal constant: 0.6925 at p = 1
    // and 0.7559 at p = 2 (Wurtz & Love, Phys. Rev. A 103, 042612, 2021).
    // The Heawood graph (girth 6) is bipartite, so C_max = |E| and the
    // approximation ratio equals that per-edge value.
    let config = QaoaConfig {
        restarts: 16,
        ..Default::default()
    };
    let r = MaxCutQaoa::new(&generators::heawood(), 2)
        .unwrap()
        .optimize(&config)
        .unwrap();
    assert!(close(r.max_cut, 21.0, 0.0));
    assert!(close(r.approximation_ratio.unwrap(), 0.7559, 5e-5), "{r:?}");
}

#[test]
fn hypercube_q30_transfers_perfectly_in_31_dimensions() {
    use metatron_qso::walks::{Quotient, QuotientWalk, StateTransfer};
    let q = Quotient::hypercube(30).unwrap();
    assert_eq!((q.num_cells(), q.num_nodes()), (31, 2f64.powi(30)));
    let walk = QuotientWalk::new(&q, Generator::Laplacian, 1.0).unwrap();
    let p = walk.cell_probabilities(0, PI / 2.0).unwrap();
    assert!(close(p[30], 1.0, 1e-15));
    let transfer = StateTransfer::from_quotient(&q, Generator::Laplacian, 1.0, 0, 30).unwrap();
    assert!(close(transfer.perfect_transfer_time().unwrap(), PI / 2.0, 1e-9));
}

#[test]
fn perfect_state_transfer_on_paths() {
    use metatron_qso::walks::StateTransfer;
    let ends = |n: usize| {
        StateTransfer::new(&generators::path(n), Generator::Adjacency, 1.0, 0, n - 1)
            .unwrap()
            .perfect_transfer_time()
    };
    assert!(close(ends(2).unwrap(), PI / 2.0, 1e-9));
    assert!(close(ends(3).unwrap(), PI / 2f64.sqrt(), 1e-9));
    assert!((4..=12).all(|n| ends(n).is_none()));
}

#[test]
fn hadamard_walk_spreads_at_konnos_rate() {
    use metatron_qso::walks::LineWalk;
    let steps = 1000;
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let p = LineWalk::hadamard()
        .distribution(steps, [C64::new(s, 0.0), C64::new(0.0, s)])
        .unwrap();
    let x = |k: usize| k as f64 - steps as f64;
    let second: f64 = p.iter().enumerate().map(|(k, q)| q * x(k).powi(2)).sum();
    let mean: f64 = p.iter().enumerate().map(|(k, q)| q * x(k)).sum();
    let sigma = (second - mean * mean).sqrt() / steps as f64;
    assert!(close(sigma, 0.54120, 5e-6));
    assert!(close((1.0 - s).sqrt(), 0.54120, 5e-6));
}

#[test]
fn szegedy_hitting_time_on_cycles_is_a_quarter_of_the_length() {
    use metatron_qso::walks::SzegedyWalk;
    for n in [16usize, 32, 64, 128] {
        let w = SzegedyWalk::new(&generators::cycle(n), &[0]).unwrap();
        assert_eq!(w.hitting_time(10 * n), Some(n / 4));
        let h = classical_hitting_times(&generators::cycle(n), 0).unwrap();
        let classical: f64 = h.iter().sum::<f64>() / n as f64;
        assert!(close(classical, (n * n - 1) as f64 / 6.0, 1e-6));
    }
}

#[test]
fn shrikhande_and_rook_graph_are_cospectral_but_not_isomorphic() {
    let (s, r) = (generators::shrikhande(), generators::rook(4, 4).unwrap());
    assert_eq!(s.strongly_regular_parameters(), Some((16, 6, 2, 2)));
    assert_eq!(r.strongly_regular_parameters(), Some((16, 6, 2, 2)));
    assert!(!s.is_isomorphic(&r));
    assert_eq!(s.automorphism_group().order(), Some(192));
    assert_eq!(r.automorphism_group().order(), Some(1152));
}
