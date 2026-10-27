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
    assert!(close(p[30], 1.0, 1e-14), "{}", p[30]);
    let transfer = StateTransfer::from_quotient(&q, Generator::Laplacian, 1.0, 0, 30).unwrap();
    assert!(close(
        transfer.perfect_transfer_time().unwrap(),
        PI / 2.0,
        1e-9
    ));
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

/// The toy device of `examples/devices/toy_device.json`.
fn toy_device() -> metatron_qso::noise::DeviceModel {
    metatron_qso::noise::DeviceModel::from_json(include_str!("../examples/devices/toy_device.json"))
        .unwrap()
}

fn ghz(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 0..n - 1 {
        c.cx(q, q + 1);
    }
    c
}

fn zz_chain(n: usize) -> PauliSum {
    let terms: Vec<String> = (0..n - 1).map(|q| format!("Z{q} Z{}", q + 1)).collect();
    terms.join(" + ").parse().unwrap()
}

#[test]
fn trajectories_reproduce_the_noisy_density_matrix() {
    use metatron_qso::noise::{DensityMatrix, TrajectorySimulator};
    let (device, c, h) = (toy_device(), ghz(5), zz_chain(5));
    let mut rho = DensityMatrix::zero(5).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let exact = rho.expectation(&h).unwrap();
    let e = TrajectorySimulator::new(device, 4000, 1)
        .unwrap()
        .expectation(&c, &[], &h)
        .unwrap();
    assert!((e.value - exact).abs() < 4.0 * e.std_error);
    assert!(e.std_error < 0.02);
}

#[test]
fn readout_mitigation_and_zne_recover_the_noiseless_value() {
    use metatron_qso::backend::DeviceBackend;
    use metatron_qso::estimation::{EstimatorOptions, estimate};
    use metatron_qso::mitigation::ReadoutCalibration;
    use metatron_qso::mitigation::zne::{Extrapolation, Folding, zne};
    let (device, c, h) = (toy_device(), ghz(5), zz_chain(5));
    let backend = DeviceBackend::density_matrix(device).unwrap();
    let options = EstimatorOptions::new(200_000).with_seed(2);
    let raw = estimate(&backend, &c, &[], &h, &options).unwrap().estimate;
    assert!(4.0 - raw.value > 0.5, "raw {}", raw.value);
    let cal = ReadoutCalibration::tensored(&backend, 5, 200_000, 3).unwrap();
    let options = options.with_readout_mitigation(cal);
    let mut k = 10;
    let r = zne(
        &c,
        &[1.0, 3.0, 5.0],
        Folding::Global,
        Extrapolation::Richardson,
        |folded| {
            k += 1;
            Ok(estimate(&backend, folded, &[], &h, &options.clone().with_seed(k))?.estimate)
        },
    )
    .unwrap();
    assert!(
        (r.estimate.value - 4.0).abs() < 4.0 * r.estimate.std_error + 0.01,
        "{:?}",
        r.estimate
    );
    assert!(r.estimate.std_error < 0.02);
}

#[test]
fn zero_noise_extrapolation_removes_depolarising_error() {
    use metatron_qso::estimation::Estimate;
    use metatron_qso::mitigation::zne::{Extrapolation, Folding, zne};
    use metatron_qso::noise::{DensityMatrix, DeviceModel, GateRule};
    let device = DeviceModel::new("dep")
        .with_rule(GateRule::any().with_num_qubits(2).with_depolarizing(0.01))
        .with_rule(GateRule::any().with_depolarizing(0.002));
    let mut c = Circuit::new(3);
    c.h(0)
        .ry(1, 0.9)
        .cx(0, 1)
        .rz(2, 0.4)
        .cx(1, 2)
        .rx(0, 0.3)
        .cx(0, 2)
        .ry(2, -0.8);
    let h: PauliSum = "Z0 Z1 + 0.5 X2 - 0.3 Z1".parse().unwrap();
    let ideal = h
        .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
        .unwrap();
    let run = |f: &Circuit| {
        let mut rho = DensityMatrix::zero(3)?;
        rho.apply_device(f, &[], &device)?;
        Ok(Estimate::exact(rho.expectation(&h)?))
    };
    let noisy = run(&c).unwrap().value;
    let r = zne(
        &c,
        &[1.0, 3.0, 5.0],
        Folding::Global,
        Extrapolation::Richardson,
        run,
    )
    .unwrap();
    // The error shrinks by more than a factor of 100 (about 380).
    assert!(100.0 * (r.estimate.value - ideal).abs() < (noisy - ideal).abs());
}

#[test]
fn probabilistic_error_cancellation_is_unbiased() {
    use metatron_qso::mitigation::pec::pec_expectation;
    use metatron_qso::noise::{ChannelSpec, DensityMatrix, DeviceModel, GateRule};
    let device = DeviceModel::new("pauli")
        .with_rule(
            GateRule::for_gate("cx")
                .with_depolarizing(0.008)
                .with_channel(ChannelSpec::Pauli([("ZZ".to_string(), 0.002)].into())),
        )
        .with_rule(GateRule::any().with_num_qubits(1).with_depolarizing(0.0005));
    let (c, h) = (ghz(5), zz_chain(5));
    let mut rho = DensityMatrix::zero(5).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    assert!(4.0 - rho.expectation(&h).unwrap() > 0.05);
    let r = pec_expectation(&c, &[], &device, &h, 20_000, 4).unwrap();
    assert!(
        (r.estimate.value - 4.0).abs() < 4.0 * r.estimate.std_error,
        "{:?}",
        r.estimate
    );
    assert!(r.overhead > 1.05 && r.overhead < 1.15);
}

#[test]
fn classical_shadow_variance_is_three_to_the_weight() {
    use metatron_qso::estimation::ClassicalShadow;
    let psi = StateVector::from_circuit(&ghz(4), &[]).unwrap();
    let snapshots = 60_000;
    let shadow = ClassicalShadow::from_state(&psi, snapshots, 3).unwrap();
    for (label, k) in [("Z0 Z1", 2), ("X0 X1 X2 X3", 4)] {
        let h: PauliSum = label.parse().unwrap();
        let e = shadow.estimate(&h).unwrap();
        assert!((e.value - 1.0).abs() < 4.0 * e.std_error, "{label}");
        let variance = e.std_error.powi(2) * snapshots as f64;
        let bound = 3f64.powi(k) - 1.0;
        let tol = 4.0 * ((3f64.powi(k) - 1.0) / snapshots as f64).sqrt();
        assert!(
            (variance / bound - 1.0).abs() < tol,
            "{label}: {variance} vs {bound}"
        );
    }
}

/// Full-CI energies from `examples/molecules/reference.json` (PySCF 2.14,
/// STO-3G, lowest roots of the neutral molecule with Sz = 0).
const FCI_REFERENCE: [(&str, &str, &[f64]); 3] = [
    (
        "H2",
        include_str!("../examples/molecules/h2_sto3g.fcidump"),
        &[
            -1.1373060357534004,
            -0.5246155553643471,
            -0.16275315579588379,
            0.49505774161810945,
        ],
    ),
    (
        "LiH",
        include_str!("../examples/molecules/lih_sto3g.fcidump"),
        &[
            -7.882401932290221,
            -7.766418475107613,
            -7.749216186506998,
            -7.716454011441431,
            -7.716454011441427,
            -7.696951987758109,
        ],
    ),
    (
        "BeH2",
        include_str!("../examples/molecules/beh2_sto3g.fcidump"),
        &[
            -15.595182356661681,
            -15.33255843934909,
            -15.33255843934906,
            -15.328335603778719,
            -15.328335603778708,
            -15.303332939179398,
        ],
    ),
];

#[test]
fn fermion_mappings_give_identical_spectra_equal_to_full_ci() {
    use metatron_qso::chemistry::{Mapping, MolecularIntegrals, QubitEncoding};
    for (name, fcidump, reference) in FCI_REFERENCE {
        let mol = MolecularIntegrals::from_fcidump(fcidump).unwrap();
        let dim = mol.determinants().unwrap().len();
        let spectra: Vec<Vec<f64>> = [
            QubitEncoding::new(Mapping::JordanWigner),
            QubitEncoding::new(Mapping::Parity),
            QubitEncoding::new(Mapping::BravyiKitaev),
            QubitEncoding::parity_reduced(),
        ]
        .into_iter()
        .map(|e| mol.fci_energies_with(e, dim).unwrap())
        .collect();
        // The whole spectrum of the determinant space agrees between mappings.
        for other in &spectra[1..] {
            for (a, b) in spectra[0].iter().zip(other) {
                assert!(close(*a, *b, 1e-9), "{name}: {a} vs {b}");
            }
        }
        // The lowest roots are PySCF's full-CI energies.
        for (k, e) in reference.iter().enumerate() {
            assert!(
                close(spectra[0][k], *e, 1e-8),
                "{name} root {k}: {} vs {e}",
                spectra[0][k]
            );
        }
    }
}

#[test]
fn uccsd_and_adapt_vqe_reach_full_ci_for_lih() {
    use metatron_qso::chemistry::{
        AdaptOptions, MolecularIntegrals, QubitEncoding, adapt_vqe_molecule, uccsd,
    };
    let mol = MolecularIntegrals::from_fcidump(FCI_REFERENCE[1].1).unwrap();
    let fci = FCI_REFERENCE[1].2[0];
    let encoding = QubitEncoding::parity_reduced();
    let h = mol.qubit_hamiltonian(encoding).unwrap();
    assert_eq!(h.min_qubits(), 10);
    // UCCSD from the Hartree–Fock point (all parameters zero).
    let ansatz = uccsd(&mol, encoding).unwrap();
    let n = ansatz.num_parameters();
    let res = Vqe::new(h, ansatz)
        .unwrap()
        .with_initial_parameters(vec![0.0; n])
        .unwrap()
        .run()
        .unwrap();
    // Variational, and far within chemical accuracy (1.6 mHa).
    assert!(
        res.energy >= fci - 1e-9 && res.energy - fci < 1e-4,
        "UCCSD {}",
        res.energy
    );
    let adapt = adapt_vqe_molecule(&mol, encoding, false, &AdaptOptions::default()).unwrap();
    assert!(adapt.converged);
    assert!(adapt.operators.len() < n);
    assert!(
        adapt.energy >= fci - 1e-9 && adapt.energy - fci < 1e-4,
        "ADAPT {}",
        adapt.energy
    );
    // Energies decrease monotonically.
    for w in adapt.energies.windows(2) {
        assert!(w[1] <= w[0] + 1e-9);
    }
}

#[test]
fn product_formulas_converge_with_their_order_below_the_bounds() {
    use metatron_qso::algorithms::time_evolution::{trotter_circuit, trotter_error_bound};
    use metatron_qso::linalg::expm_hermitian;
    let n = 4;
    let h = models::heisenberg_xxz(n, 1.0, 0.5, false);
    let exact = expm_hermitian(&h.to_dense(n).unwrap(), 1.0);
    let error = |steps: usize, order: usize| {
        let c = trotter_circuit(&h, n, 1.0, steps, order).unwrap();
        let mut u = metatron_qso::linalg::CMatrix::zeros(1 << n, 1 << n);
        for col in 0..1 << n {
            let mut psi = StateVector::basis(n, col).unwrap();
            psi.apply_circuit(&c, &[]).unwrap();
            for (row, a) in psi.amplitudes().iter().enumerate() {
                u[(row, col)] = *a;
            }
        }
        (u - &exact).singular_values().max()
    };
    for (order, expected) in [(1usize, 2.0), (2, 4.0), (4, 16.0)] {
        let (e16, e32) = (error(16, order), error(32, order));
        assert!(e16 <= trotter_error_bound(&h, 1.0, 16, order).unwrap());
        assert!(e32 <= trotter_error_bound(&h, 1.0, 32, order).unwrap());
        // Doubling the steps divides the error by 2^order.
        assert!(
            (e16 / e32 / expected - 1.0).abs() < 0.05,
            "order {order}: {}",
            e16 / e32
        );
    }
}

#[test]
fn amplitude_estimation_error_falls_as_one_over_the_oracle_calls() {
    use metatron_qso::algorithms::amplitude_estimation::{AmplitudeProblem, maximum_likelihood};
    let a = 0.3;
    let problem = AmplitudeProblem::bernoulli(a).unwrap();
    let rmse = |levels: usize, shots: usize| {
        let mut sq = 0.0;
        let mut calls = 0;
        for seed in 0..200 {
            let r = maximum_likelihood(&problem, levels, shots, 0.95, seed).unwrap();
            sq += (r.estimate - a).powi(2);
            calls = r.oracle_calls;
        }
        (calls as f64, (sq / 200.0).sqrt())
    };
    let slope = |points: Vec<(f64, f64)>| {
        let n = points.len() as f64;
        let mx = points.iter().map(|p| p.0.ln()).sum::<f64>() / n;
        let my = points.iter().map(|p| p.1.ln()).sum::<f64>() / n;
        let sxy: f64 = points
            .iter()
            .map(|p| (p.0.ln() - mx) * (p.1.ln() - my))
            .sum();
        let sxx: f64 = points.iter().map(|p| (p.0.ln() - mx).powi(2)).sum();
        sxy / sxx
    };
    let quantum = slope((3..9).map(|levels| rmse(levels, 100)).collect());
    let classical = slope([100, 400, 1600, 6400].iter().map(|&s| rmse(1, s)).collect());
    assert!((quantum + 1.0).abs() < 0.05, "{quantum}");
    assert!((classical + 0.5).abs() < 0.05, "{classical}");
}

#[test]
fn vqd_finds_every_level_of_h2() {
    use metatron_qso::algorithms::ansatz::{Entanglement, efficient_su2};
    use metatron_qso::algorithms::excited::{ExcitedStateOptions, vqd};
    use metatron_qso::chemistry::{MolecularIntegrals, QubitEncoding};
    let mol = MolecularIntegrals::from_fcidump(FCI_REFERENCE[0].1).unwrap();
    let h = mol
        .qubit_hamiltonian(QubitEncoding::parity_reduced())
        .unwrap();
    let r = vqd(
        &h,
        &efficient_su2(2, 2, Entanglement::Linear),
        4,
        &ExcitedStateOptions::default().with_restarts(6).with_seed(2),
    )
    .unwrap();
    for (e, reference) in r.energies.iter().zip(FCI_REFERENCE[0].2) {
        assert!(close(*e, *reference, 1e-6), "{e} vs {reference}");
    }
}

#[test]
fn stabilizer_states_at_scale() {
    use metatron_qso::graph::generators::from_spec;
    use metatron_qso::stabilizer::{StabilizerState, local_complementation};
    // GHZ state on 10 000 qubits: every ZZ correlation is +1, single Z
    // expectations vanish, and X^⊗n stabilizes it.
    let n = 10_000;
    let mut ghz = StabilizerState::zero(n).unwrap();
    ghz.h(0).unwrap();
    for q in 1..n {
        ghz.cx(q - 1, q).unwrap();
    }
    assert_eq!(ghz.expectation_str("Z0 Z9999").unwrap(), 1.0);
    assert_eq!(ghz.expectation_str("Z4321").unwrap(), 0.0);
    let all_x: String = (0..n).map(|q| format!("X{q} ")).collect();
    assert_eq!(ghz.expectation_str(&all_x).unwrap(), 1.0);
    // Graph-state entanglement equals the cut-rank over GF(2): the
    // hypercube Q10 split into two Q9 halves is joined by a perfect
    // matching (rank 512); a cycle split into two arcs by two edges.
    let q10 = StabilizerState::graph_state(&from_spec("hypercube:10").unwrap()).unwrap();
    assert_eq!(
        q10.entanglement_entropy(&(0..512).collect::<Vec<_>>())
            .unwrap(),
        512
    );
    let c1000 = StabilizerState::graph_state(&from_spec("cycle:1000").unwrap()).unwrap();
    assert_eq!(
        c1000
            .entanglement_entropy(&(0..500).collect::<Vec<_>>())
            .unwrap(),
        2
    );
    // Metatron's Cube (K13) is a GHZ state up to local Cliffords: local
    // complementation at a node leaves the star.
    let k13 = from_spec("metatron").unwrap();
    let star = local_complementation(&k13, 0).unwrap();
    assert_eq!((star.num_edges(), star.degree(0)), (12, 12));
    let mut s = StabilizerState::graph_state(&k13).unwrap();
    s.local_complement(&k13, 0).unwrap();
    assert!(s.same_state(&StabilizerState::graph_state(&star).unwrap()));
    // Every bipartition of the K13 graph state carries exactly one ebit.
    assert_eq!(
        StabilizerState::graph_state(&k13)
            .unwrap()
            .entanglement_entropy(&[0, 1, 2, 3, 4, 5])
            .unwrap(),
        1
    );
}

#[test]
fn surface_codes_suppress_logical_errors_below_threshold() {
    use metatron_qso::qec::{
        CircuitNoise, logical_error_rate, repetition_code_memory, surface_code_memory,
    };
    let surface = |d: usize, p: f64, shots: usize| {
        logical_error_rate(
            &surface_code_memory(d, d, &CircuitNoise::uniform(p)).unwrap(),
            shots,
            7,
        )
        .unwrap()
    };
    // Rotated surface code, d rounds, circuit-level depolarizing noise
    // p = 0.001 on every gate, reset, measurement and idle data qubit:
    // the union-find decoder's logical error rate falls with the distance
    // (10⁶ shots, seed 7).
    let below: Vec<_> = [3, 5, 7]
        .iter()
        .map(|&d| surface(d, 0.001, 1_000_000))
        .collect();
    assert!((7.5e-4..9.5e-4).contains(&below[0].rate), "{:?}", below[0]);
    assert!((1.15e-4..1.7e-4).contains(&below[1].rate), "{:?}", below[1]);
    assert!((1.2e-5..3.2e-5).contains(&below[2].rate), "{:?}", below[2]);
    for w in below.windows(2) {
        assert!(w[1].rate + 3.0 * w[1].standard_error < w[0].rate - 3.0 * w[0].standard_error);
    }
    // Above the threshold (p = 0.01) larger codes fail more often.
    let above: Vec<_> = [3, 5, 7]
        .iter()
        .map(|&d| surface(d, 0.01, 20_000))
        .collect();
    for w in above.windows(2) {
        assert!(w[0].rate + 3.0 * w[0].standard_error < w[1].rate - 3.0 * w[1].standard_error);
    }
    assert!((0.055..0.07).contains(&above[0].rate), "{:?}", above[0]);
    // The repetition code at p = 0.01 (bit flips only matter).
    let rep: Vec<f64> = [3, 5, 9]
        .iter()
        .map(|&d| {
            let c = repetition_code_memory(d, d, &CircuitNoise::uniform(0.01)).unwrap();
            logical_error_rate(&c, 100_000, 7).unwrap().rate
        })
        .collect();
    assert!(rep[0] > 3.0 * rep[1] && rep[1] > 3.0 * rep[2], "{rep:?}");
}

#[test]
fn union_find_decodes_the_repetition_code_by_majority_vote() {
    use metatron_qso::qec::{
        CircuitNoise, MatchingGraph, UnionFindDecoder, detector_error_model, repetition_code_memory,
    };
    // One round with bit flips on the data only: every error pattern of
    // d bits is enumerated, and the decoder fails exactly when more than
    // half of the bits flipped.
    for d in [3usize, 5, 7, 9] {
        let c = repetition_code_memory(d, 1, &CircuitNoise::data_only(0.3)).unwrap();
        let dem = detector_error_model(&c, true).unwrap();
        assert_eq!(dem.errors.len(), d);
        let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
        let mut failures_by_weight = vec![0usize; d + 1];
        for pattern in 0u32..(1 << d) {
            let mut fired = std::collections::BTreeSet::new();
            let mut flips = 0;
            for (k, e) in dem.errors.iter().enumerate() {
                if (pattern >> k) & 1 == 1 {
                    for &det in &e.detectors {
                        if !fired.remove(&det) {
                            fired.insert(det);
                        }
                    }
                    flips ^= e.observables;
                }
            }
            let fired: Vec<usize> = fired.into_iter().collect();
            let predicted = if fired.is_empty() {
                0
            } else {
                decoder.decode(&fired).unwrap()
            };
            if predicted != flips {
                failures_by_weight[pattern.count_ones() as usize] += 1;
            }
        }
        // Weight k fails for all C(d, k) patterns if k > d/2, never otherwise.
        for (k, &f) in failures_by_weight.iter().enumerate() {
            let binom = (0..k).fold(1usize, |b, i| b * (d - i) / (i + 1));
            assert_eq!(f, if 2 * k > d { binom } else { 0 }, "d = {d}, weight {k}");
        }
    }
}

#[test]
fn matrix_product_states_follow_free_fermions_and_qaoa_rings() {
    use metatron_qso::C64;
    use metatron_qso::algorithms::ansatz::qaoa_maxcut_circuit;
    use metatron_qso::mps::{Mps, MpsOptions};
    // A Trotterized XX chain on 100 qubits from the Néel state. Each gate
    // exp(−iθ(XX + YY)/2) on (j, j + 1) is a hopping rotation of the
    // Jordan–Wigner fermions, so the circuit's single-particle unitary U
    // is the product of 2 × 2 rotations, and ⟨Z_j⟩ = 1 − 2 Σ_{k∈S} |U_jk|²
    // and ⟨Z_i Z_j⟩ (Wick) are exact for the circuit, Trotter error and all.
    let n = 100;
    let theta: f64 = 0.25;
    let steps = 6;
    let mut c = Circuit::new(n);
    for q in (0..n).step_by(2) {
        c.x(q);
    }
    let mut u = vec![vec![C64::new(0.0, 0.0); n]; n];
    for (i, row) in u.iter_mut().enumerate() {
        row[i] = C64::new(1.0, 0.0);
    }
    let (cs, sn) = (theta.cos(), theta.sin());
    for _ in 0..steps {
        for start in [0, 1] {
            for j in (start..n - 1).step_by(2) {
                c.rxx(j, j + 1, theta).ryy(j, j + 1, theta);
                // Rows j, j + 1 of U ← [[cos, −i sin], [−i sin, cos]]·rows.
                let (top, bottom) = u.split_at_mut(j + 1);
                for (a, b) in top[j].iter_mut().zip(bottom[0].iter_mut()) {
                    let (x, y) = (*a, *b);
                    *a = x * cs + y * C64::new(0.0, -sn);
                    *b = x * C64::new(0.0, -sn) + y * cs;
                }
            }
        }
    }
    let occupied: Vec<usize> = (0..n).step_by(2).collect();
    let corr =
        |i: usize, j: usize| -> C64 { occupied.iter().map(|&k| u[i][k].conj() * u[j][k]).sum() };
    let opts = MpsOptions {
        max_bond: 256,
        cutoff: 1e-16,
    };
    let mut mps = Mps::from_circuit(&c, &[], opts).unwrap();
    // Bond dimension 52 suffices; only rounding-level weight is dropped.
    assert!(mps.max_bond_dimension() < 64);
    assert!(mps.fidelity_lower_bound().unwrap() > 1.0 - 1e-7);
    for j in [0, 1, 37, 50, 98, 99] {
        let nj = corr(j, j).re;
        let z = mps.expectation_str(&format!("Z{j}")).unwrap();
        assert!(
            close(z, 1.0 - 2.0 * nj, 1e-8),
            "site {j}: {z} vs {}",
            1.0 - 2.0 * nj
        );
    }
    for (i, j) in [(10, 11), (49, 50), (20, 23), (0, 99)] {
        let (ni, nj) = (corr(i, i).re, corr(j, j).re);
        let ninj = ni * nj - corr(i, j).norm_sqr();
        let zz = 1.0 - 2.0 * ni - 2.0 * nj + 4.0 * ninj;
        let got = mps.expectation_str(&format!("Z{i} Z{j}")).unwrap();
        assert!(close(got, zz, 1e-8), "({i}, {j}): {got} vs {zz}");
    }

    // QAOA on a ring of 100 nodes with the angles optimized on a ring of
    // 10: every edge sees the same light cone for n ≥ 2p + 2, so the ratio
    // is the proven optimum (2p + 1)/(2p + 2) (Mbeng, Fazio & Santoro).
    let results = optimize_layerwise(&generators::cycle(10), 3, &QaoaConfig::default()).unwrap();
    let ring = generators::cycle(n);
    for r in &results {
        let mut theta = r.gammas.clone();
        theta.extend(&r.betas);
        let circuit = qaoa_maxcut_circuit(&ring, r.p).unwrap();
        let mut s = Mps::from_circuit(&circuit, &theta, MpsOptions::default()).unwrap();
        let cut: f64 = ring
            .edges()
            .iter()
            .map(|&(a, b, _)| (1.0 - s.expectation_str(&format!("Z{a} Z{b}")).unwrap()) / 2.0)
            .sum();
        let expected = (2 * r.p + 1) as f64 / (2 * r.p + 2) as f64;
        assert!(
            close(cut / n as f64, expected, 1e-6),
            "p = {}: {}",
            r.p,
            cut / n as f64
        );
        assert!(s.fidelity_lower_bound().unwrap() > 1.0 - 1e-9);
    }
}

#[test]
fn learned_noise_recovers_a_known_device_and_cancels_it() {
    use metatron_qso::noise::learning::{LearningOptions, learn_layer, pec_layered};
    use metatron_qso::noise::{ChannelSpec, DensityMatrix, DeviceModel, GateRule, QubitProperties};
    // cx: two-qubit depolarising (p = 0.012) and a coherent Z rotation of
    // the control (θ = 0.12), with readout errors. Twirled, the noise of
    // each cx is sparse Pauli–Lindblad: λ = −ln(1 − 16p/15)/16 on all 15
    // Paulis of its pair, plus −ln(cos θ)/2 on Z of the control. Both
    // terms commute with the cx, so its fidelities are symmetric, which
    // learning assumes (an error that cx moves, like ZZ, is learnable only
    // as a product with its image).
    let (p, theta) = (0.012, 0.12);
    let device = DeviceModel::new("learnable")
        .with_rule(GateRule::for_gate("cx").with_depolarizing(p).with_channel(
            ChannelSpec::Rotation {
                pauli: "ZI".into(),
                angle: theta,
            },
        ))
        .with_default_qubit(QubitProperties::new().with_readout(0.02, 0.04))
        .with_idle_noise(false);
    let depolarising = -(1.0 - 16.0 * p / 15.0).ln() / 16.0;
    let phase = -theta.cos().ln() / 2.0;
    let mut layer = Circuit::new(4);
    layer.cx(0, 1).cx(2, 3);
    let learned = learn_layer(&layer, &device, &LearningOptions::default()).unwrap();
    let model = &learned.model;
    assert_eq!(model.generators.len(), 12 + 27);
    for ((g, rate), se) in model
        .generators
        .iter()
        .zip(&model.rates)
        .zip(&learned.std_errors)
    {
        let qubits = g.x_mask() | g.z_mask();
        let on_a_pair = qubits & !0b0011 == 0 || qubits & !0b1100 == 0;
        let control_z = g.x_mask() == 0 && (qubits == 0b0001 || qubits == 0b0100);
        let expected = match (on_a_pair, control_z) {
            (true, true) => depolarising + phase,
            (true, false) => depolarising,
            _ => 0.0,
        };
        assert!(
            (rate - expected).abs() < 4.0 * se,
            "{g}: {rate} ± {se}, expected {expected}"
        );
    }
    // Error cancellation with the learned model on three layers between
    // random rotations.
    let mut rng = Rng::seed_from_u64(17);
    let blocks: Vec<Circuit> = (0..4)
        .map(|_| {
            let mut b = Circuit::new(4);
            for q in 0..4 {
                b.ry(q, rng.uniform_range(-3.0, 3.0))
                    .rz(q, rng.uniform_range(-3.0, 3.0));
            }
            b
        })
        .collect();
    let mut joined = Circuit::new(4);
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 {
            joined.append(&layer).unwrap();
        }
        joined.append(b).unwrap();
    }
    let h = zz_chain(4);
    let exact = DensityMatrix::from_circuit(&joined, &[])
        .unwrap()
        .expectation(&h)
        .unwrap();
    let mut rho = DensityMatrix::zero(4).unwrap();
    rho.apply_device(&joined, &[], &device).unwrap();
    let noisy = rho.expectation(&h).unwrap();
    let r = pec_layered(&blocks, &layer, model, &device, &h, 4000, 5).unwrap();
    let se = r.estimate.std_error;
    assert!(
        (r.estimate.value - exact).abs() < 4.0 * se,
        "{:?} vs {exact}",
        r.estimate
    );
    assert!(
        (noisy - exact).abs() > 5.0 * se,
        "noisy {noisy}, exact {exact}, se {se}"
    );
}
