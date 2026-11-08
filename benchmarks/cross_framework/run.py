"""Cross-framework comparison: correctness and speed of QSO against Qiskit
Aer, Qulacs, Cirq, PennyLane Lightning and (for permanents) The Walrus.

    python run.py --out results.json            # full run (~15 min, 4 cores)
    python run.py --out results.json --quick    # smoke test

Correctness compares final state vectors, energies, gradients and
permanents. Speed is measured in interleaved rounds: in every round each
framework runs every configuration in a fresh process, in rotated order,
so that bursts of load on a shared machine hit all frameworks alike.
Reported are the median over all timed runs and the minimum (load can only
add time). Circuit construction and transpilation are not timed.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frameworks as fw  # noqa: E402

PACKAGES = [
    "metatron-qso", "qiskit", "qiskit-aer", "qulacs", "cirq-core", "pennylane",
    "pennylane-lightning", "thewalrus", "numpy", "projectq", "stim", "pymatching", "pyscf",
    "ldpc", "qutip", "pycce", "sympy", "qiskit-addon-obp", "qiskit-ibm-runtime",
]

# ----------------------------------------------------------------------------
# Worker: one configuration in a fresh process.


WARMUP_SECONDS = 0.25


def worker(lib: str, kind: str, n: int, reps: int, warmup: int) -> list[float]:
    # "1@500": one layer, each timed run after a pause of 500 µs (the first
    # simulation after an idle period: an optimiser step, Python code).
    kind, _, pause = kind.partition("@")
    pause_s = int(pause) * 1e-6 if pause else 0.0
    if kind == "permanent":
        rng = np.random.default_rng(n)
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        if lib == "qso":
            import metatron_qso as q

            run = lambda: q.permanent(a)  # noqa: E731
        else:
            from thewalrus import perm

            run = lambda: perm(a)  # noqa: E731
    elif kind == "gradient":
        ops, n_params = fw.ansatz_ops(n, 4)
        terms = fw.tfim_terms(n)
        theta = np.linspace(0.0, 1.0, n_params)
        if lib == "qso":
            import metatron_qso as q

            c = fw.qso_circuit(n, ops, n_params)
            h = fw.qso_pauli_sum(terms)
            run = lambda: q.expectation_and_gradient(c, h, theta)  # noqa: E731
        elif lib == "pennylane":
            f = fw.pennylane_value_and_grad_fn(n, ops, terms)
            run = lambda: f(theta)  # noqa: E731
        else:
            f = fw.qulacs_value_and_grad_fn(n, ops, terms, n_params)
            run = lambda: f(theta)  # noqa: E731
    else:
        ops = fw.layer_ops(n, int(kind))
        if lib in ("qso", "qso-single"):
            import metatron_qso as q

            c = fw.qso_circuit(n, ops)
            precision = "single" if lib == "qso-single" else "double"
            run = lambda: q.simulate(c, precision=precision)  # noqa: E731
        elif lib == "qulacs":
            from qulacs import QuantumState

            c = fw.qulacs_circuit(n, ops)

            def run():
                s = QuantumState(n)
                c.update_quantum_state(s)

        elif lib in ("aer", "aer-single"):
            sim, tqc = fw.aer_prepare(n, ops, "single" if lib == "aer-single" else "double")
            run = lambda: sim.run(tqc).result()  # noqa: E731
        elif lib == "pennylane":
            run = fw.pennylane_state_fn(n, ops)
        elif lib == "cirq":
            import cirq

            circuit, qubits = fw.cirq_circuit(n, ops)
            sim = cirq.Simulator(dtype=np.complex128)
            run = lambda: sim.simulate(circuit, qubit_order=list(reversed(qubits)))  # noqa: E731
        else:
            raise ValueError(lib)
    # Warm up for a minimum time as well as a minimum count: on virtual
    # machines, thread pools whose workers sleep when idle (rayon) reach full
    # speed only after tens of milliseconds of sustained work, while OpenMP
    # workers spin and are fast at once (see BENCHMARKS.md).
    end = time.perf_counter() + WARMUP_SECONDS
    done = 0
    while done < warmup or time.perf_counter() < end:
        run()
        done += 1
    times = []
    for _ in range(reps):
        if pause_s:
            time.sleep(pause_s)
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    return times


# ----------------------------------------------------------------------------
# Correctness


def q_single_state(n: int, ops: list):
    import metatron_qso as q

    return q.simulate(fw.qso_circuit(n, ops), precision="single").amplitudes


def correctness(quick: bool) -> dict:
    rng = np.random.default_rng(20261024)
    runners = {
        "qiskit-aer": fw.aer_state,
        "qulacs": fw.qulacs_state,
        "cirq": fw.cirq_state,
        "pennylane-lightning": fw.pennylane_state,
    }
    circuits = [fw.random_ops(8, 60, rng) for _ in range(5 if quick else 20)]
    circuits.append(fw.layer_ops(10, 2))
    max_diff = {name: 0.0 for name in runners}
    for ops in circuits:
        n = 1 + max(max(qs) for _, qs, _ in ops)
        ref = fw.qso_state(n, ops)
        for name, run in runners.items():
            max_diff[name] = max(max_diff[name], float(np.max(np.abs(run(n, ops) - ref))))

    # Noisy circuits: density matrices against Qiskit Aer's density-matrix
    # method with the same channels after every gate.
    noise_models = [
        {"p1": 0.01, "p2": 0.03},
        {"p1": 0.002, "p2": 0.02, "gamma": 0.05},
        {"p1": 0.2, "p2": 0.5, "gamma": 0.3},
    ]
    noisy_circuits = [fw.random_ops(5, 30, rng) for _ in range(2 if quick else 6)]
    density_diff = 0.0
    for noise in noise_models:
        for ops in noisy_circuits:
            n = 1 + max(max(qs) for _, qs, _ in ops)
            density_diff = max(
                density_diff,
                float(np.max(np.abs(fw.qso_density(n, ops, noise) - fw.aer_density(n, ops, noise)))),
            )

    # Energies and gradients on the 10-qubit TFIM ansatz.
    import metatron_qso as q

    n = 10
    ops, n_params = fw.ansatz_ops(n, 4)
    terms = fw.tfim_terms(n)
    theta = rng.uniform(-1, 1, n_params)
    value, grad = q.expectation_and_gradient(fw.qso_circuit(n, ops, n_params), fw.qso_pauli_sum(terms), theta)
    gradients = {}
    for name, make in {
        "pennylane-lightning (adjoint)": lambda: fw.pennylane_value_and_grad_fn(n, ops, terms),
        "qulacs (backprop)": lambda: fw.qulacs_value_and_grad_fn(n, ops, terms, n_params),
    }.items():
        v, g = make()(theta)
        gradients[name] = {"value_diff": abs(v - value), "max_gradient_diff": float(np.max(np.abs(g - grad)))}

    # Permanents against The Walrus.
    from thewalrus import perm

    perm_err = 0.0
    for size in range(1, 13):
        a = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        ours, theirs = q.permanent(a), perm(a)
        perm_err = max(perm_err, abs(ours - theirs) / abs(theirs))
    # Single precision: QSO's StateVector32 and Aer's single-precision
    # simulator against QSO's double-precision state (directions; norms
    # drift by rounding).
    single = {"qso": 0.0, "qiskit-aer": 0.0}
    for ops in circuits:
        n = 1 + max(max(qs) for _, qs, _ in ops)
        ref = fw.qso_state(n, ops)
        ours = np.asarray(q_single_state(n, ops), dtype=np.complex128)
        theirs = np.asarray(fw.aer_state(n, ops, "single"), dtype=np.complex128)
        for name, v in (("qso", ours), ("qiskit-aer", theirs)):
            f = abs(np.vdot(ref, v)) ** 2 / (np.vdot(v, v).real * np.vdot(ref, ref).real)
            single[name] = max(single[name], 1 - f)
    return {
        "single_precision_infidelity_vs_double": single,
        "random_circuits": len(circuits) - 1,
        "qubits": 8,
        "gates_per_circuit": 60,
        "max_amplitude_difference_vs_qso": max_diff,
        "tfim10_energy": value,
        "gradients_vs_qso": gradients,
        "permanent_max_relative_difference_vs_thewalrus": perm_err,
        "gaussian_vs_thewalrus": gaussian_vs_thewalrus(rng),
        "noisy_circuits": {
            "circuits": len(noisy_circuits),
            "qubits": 5,
            "gates_per_circuit": 30,
            "noise_models": noise_models,
            "max_density_matrix_difference_vs_qiskit_aer": density_diff,
        },
        "qec_vs_stim": qec_vs_stim(quick),
        "repeated_vs_stim": repeated_vs_stim(quick),
        "bposd_vs_ldpc": bposd_vs_ldpc(quick),
        "correlated_and_distance_vs_stim": correlated_and_distance_vs_stim(quick),
        "chemistry_vs_pyscf": chemistry_vs_pyscf(),
        "spins_vs_qutip": spins_vs_qutip(),
        "redfield_vs_qutip": redfield_vs_qutip(),
        "bath_vs_pycce": bath_vs_pycce(),
        "generalised_bath_vs_pycce": generalised_bath_vs_pycce(),
        "transport_vs_qutip": transport_in_child(quick),
        "pauli_propagation": pauli_propagation_vs_references(quick),
    }


def qec_vs_stim(quick: bool) -> dict:
    """Detector error models and detection-event rates against Stim: of
    QSO's memory experiments (on their export) and of Stim's generated
    circuits (imported into QSO, against the originals); QSO's matching
    decoder against PyMatching shot by shot on the same error model and
    syndromes; and QSO's union-find logical error rate next to PyMatching's
    (minimum-weight matching)."""
    import pymatching
    import stim

    import metatron_qso as q

    shots = 100_000 if quick else 1_000_000
    cases = {
        "repetition d=5, 4 rounds, p=0.02": q.repetition_code_memory(5, 4, 0.02),
        "surface d=3, 3 rounds, p=0.01": q.surface_code_memory(3, 3, 0.01),
        "surface d=5, 5 rounds, p=0.003": q.surface_code_memory(5, 5, 0.003),
        "surface X basis d=5, 5 rounds, p=0.003": q.surface_code_memory(5, 5, 0.003, basis="x"),
        "surface d=5, 2 rounds, mixed noise": q.surface_code_memory(
            5,
            2,
            q.CircuitNoise(
                after_gate1=0.004, after_gate2=0.01, measurement=0.02, after_reset=0.003, before_round=0.005
            ),
        ),
    }

    def merge(model: dict, key: tuple, p: float) -> None:
        e = model.get(key, 0.0)
        model[key] = e * (1 - p) + p * (1 - e)

    # Stim's own generated circuits, imported; Stim runs the originals.
    originals = {}
    generated = {
        "stim repetition_code:memory d=5": ("repetition_code:memory", 5),
        "stim rotated_memory_x d=5": ("surface_code:rotated_memory_x", 5),
        "stim rotated_memory_z d=5": ("surface_code:rotated_memory_z", 5),
    }
    for name, (task, d) in generated.items():
        sc = stim.Circuit.generated(
            task,
            distance=d,
            rounds=d,
            after_clifford_depolarization=0.003,
            before_round_data_depolarization=0.002,
            before_measure_flip_probability=0.004,
            after_reset_flip_probability=0.003,
        )
        cases[name] = q.DetectorCircuit.from_stim(str(sc))
        originals[name] = sc

    out = {}
    for name, c in cases.items():
        sc = originals.get(name) or stim.Circuit(c.to_stim())
        theirs: dict = {}
        for inst in sc.detector_error_model(decompose_errors=False).flattened():
            if inst.type != "error":
                continue
            targets = inst.targets_copy()
            dets = tuple(sorted(t.val for t in targets if t.is_relative_detector_id()))
            obs = sum(1 << t.val for t in targets if t.is_logical_observable_id())
            merge(theirs, (dets, obs), inst.args_copy()[0])
        ours: dict = {}
        for p, dets, obs in c.detector_error_model().errors:
            merge(ours, (tuple(dets), obs), p)
        keys = set(theirs) | set(ours)
        dem_diff = max(abs(theirs.get(k, 0.0) - ours.get(k, 0.0)) / max(theirs.get(k, 0.0), ours.get(k, 0.0)) for k in keys)
        det, _ = c.sample(shots, seed=1)
        ref = sc.compile_detector_sampler(seed=2).sample(shots)
        a, b = det.mean(axis=0), ref.mean(axis=0)
        sigma = np.sqrt(np.maximum(a * (1 - a) + b * (1 - b), 1e-300) / shots)
        matching = pymatching.Matching.from_detector_error_model(sc.detector_error_model(decompose_errors=True))
        d2, o2 = sc.compile_detector_sampler(seed=3).sample(shots, separate_observables=True)
        mwpm = float(np.mean(np.any(matching.decode_batch(d2) != o2, axis=1)))
        # The same syndromes through both matching decoders, on QSO's
        # decomposed model (parallel edges keep the lighter, as in QSO).
        dem = c.detector_error_model(decompose=True)
        reference = pymatching.Matching()
        for p, dets, obs in dem.errors:
            faults = {k for k in range(64) if obs >> k & 1}
            weight = math.log((1 - p) / p) if p < 0.5 else 0.0
            if len(dets) == 1:
                reference.add_boundary_edge(
                    dets[0], fault_ids=faults, weight=weight, error_probability=p, merge_strategy="smallest-weight"
                )
            elif len(dets) == 2:
                reference.add_edge(
                    dets[0], dets[1], fault_ids=faults, weight=weight, error_probability=p, merge_strategy="smallest-weight"
                )
        bits = reference.decode_batch(det)
        reference_flips = np.zeros(shots, dtype=np.uint64)
        for k in range(bits.shape[1]):
            reference_flips |= bits[:, k].astype(np.uint64) << np.uint64(k)
        qso_flips = q.MatchingDecoder(dem).decode_batch(det)
        sparse_flips = q.SparseMatchingDecoder(dem).decode_batch(det)
        # Correction weights shot by shot: the first 2000 shots and every
        # shot whose prediction differs from PyMatching's.
        sparse = q.SparseMatchingDecoder(dem)
        weight_diff = 0.0
        differing = np.flatnonzero(sparse_flips != reference_flips)
        for s in sorted(set(range(min(2000, shots))) | set(differing.tolist())):
            row = det[s]
            _, ours_w = sparse.decode_with_weight(list(np.flatnonzero(row)))
            _, theirs_w = reference.decode(row, return_weight=True)
            weight_diff = max(weight_diff, abs(ours_w - theirs_w) / max(1.0, theirs_w))
        out[name] = {
            "detectors": c.n_detectors,
            "error_mechanisms": len(ours),
            "error_mechanisms_only_one_side": sum(1 for k in keys if k not in theirs or k not in ours),
            "max_relative_probability_difference": dem_diff,
            "max_detector_rate_z_score": float(np.max(np.abs(a - b) / sigma)),
            "shots": shots,
            "logical_error_rate_union_find": q.logical_error_rate(c, shots, seed=4).rate,
            "logical_error_rate_matching": q.logical_error_rate(c, shots, seed=4, decoder="matching").rate,
            "logical_error_rate_pymatching": mwpm,
            "matching_disagreement_with_pymatching": float(np.mean(qso_flips != reference_flips)),
            "sparse_matching_disagreement_with_pymatching": float(np.mean(sparse_flips != reference_flips)),
            "sparse_matching_max_relative_weight_difference": weight_diff,
        }
    out["decoding_speed"] = decoding_speed(quick)
    return out


def repeated_vs_stim(quick: bool) -> dict:
    """Long experiments with REPEAT blocks (1.61): Stim's generated memory
    experiments of thousands of rounds, read by QSO with their blocks kept;
    the folded detector error models against Stim's term by term, the
    detection-event rates, and the logical error rate of QSO's parallel
    windows next to PyMatching on the whole experiment."""
    import pymatching
    import stim

    import metatron_qso as q

    out = {}
    for task, d, rounds in [
        ("surface_code:rotated_memory_z", 5, 1000 if quick else 10_000),
        ("repetition_code:memory", 7, 2000 if quick else 20_000),
    ]:
        sc = stim.Circuit.generated(
            task,
            distance=d,
            rounds=rounds,
            after_clifford_depolarization=0.003,
            before_round_data_depolarization=0.002,
            before_measure_flip_probability=0.004,
            after_reset_flip_probability=0.003,
        )
        c = q.RepeatedCircuit.from_stim(str(sc))
        t = time.perf_counter()
        model = c.detector_error_model()
        ours_time = time.perf_counter() - t
        t = time.perf_counter()
        stim_dem = sc.detector_error_model(decompose_errors=False)
        stim_time = time.perf_counter() - t
        theirs: dict = {}
        for inst in stim_dem.flattened():
            if inst.type != "error":
                continue
            targets = inst.targets_copy()
            dets = tuple(sorted(t.val for t in targets if t.is_relative_detector_id()))
            obs = sum(1 << t.val for t in targets if t.is_logical_observable_id())
            e = theirs.get((dets, obs), 0.0)
            p = inst.args_copy()[0]
            theirs[(dets, obs)] = e * (1 - p) + p * (1 - e)
        ours = {(tuple(dets), obs): p for p, dets, obs in model.errors}
        keys = set(theirs) | set(ours)
        diff = max(abs(theirs.get(k, 0.0) - ours.get(k, 0.0)) / max(theirs.get(k, 0.0), ours.get(k, 0.0)) for k in keys)
        # Shots × detectors stay below the 2³⁰ entries of an array.
        shots = 2000
        det, _ = c.sample(shots, seed=1)
        ref = sc.compile_detector_sampler(seed=2).sample(shots)
        a, b = det.mean(axis=0), ref.mean(axis=0)
        sigma = np.sqrt(np.maximum(a * (1 - a) + b * (1 - b), 1e-300) / shots)
        matching = pymatching.Matching.from_detector_error_model(sc.detector_error_model(decompose_errors=True))
        d2, o2 = sc.compile_detector_sampler(seed=3).sample(shots, separate_observables=True)
        mwpm = float(np.mean(np.any(matching.decode_batch(d2) != o2, axis=1)))
        parallel = q.logical_error_rate(c, shots, seed=4, decoder="parallel_window_matching", commit=d, buffer=d)
        out[f"{task} d={d}, {rounds} rounds"] = {
            "stored_operations": c.n_stored_operations,
            "expanded_operations": c.n_operations,
            "detectors": c.n_detectors,
            "error_mechanisms": len(ours),
            "error_mechanisms_only_one_side": sum(1 for k in keys if k not in theirs or k not in ours),
            "max_relative_probability_difference": diff,
            "model_seconds_qso": ours_time,
            "model_seconds_stim": stim_time,
            "max_detector_rate_z_score": float(np.max(np.abs(a - b) / sigma)),
            "shots": shots,
            "logical_error_rate_parallel_windows": parallel.rate,
            "logical_error_rate_pymatching": mwpm,
        }
    return out


def decoding_speed(quick: bool) -> dict:
    """Microseconds per shot of QSO's sparse matching decoder and of
    PyMatching on the same error model and syndromes (one thread each;
    after a warm-up, medians of interleaved repetitions)."""
    import pymatching

    import metatron_qso as q

    shots = 2000 if quick else 20_000
    out = {}
    for d, p in [(11, 0.003), (17, 0.001), (17, 0.005), (25, 0.001), (25, 0.005)]:
        c = q.surface_code_memory(d, d, p)
        dem = c.detector_error_model(decompose=True)
        reference = pymatching.Matching()
        for pr, dets, obs in dem.errors:
            faults = {k for k in range(64) if obs >> k & 1}
            weight = math.log((1 - pr) / pr) if pr < 0.5 else 0.0
            if len(dets) == 1:
                reference.add_boundary_edge(dets[0], fault_ids=faults, weight=weight, error_probability=pr, merge_strategy="smallest-weight")
            elif len(dets) == 2:
                reference.add_edge(dets[0], dets[1], fault_ids=faults, weight=weight, error_probability=pr, merge_strategy="smallest-weight")
        det, _ = c.sample(shots, seed=7)
        decoder = q.SparseMatchingDecoder(dem)
        # Warm-up, then interleaved repetitions; medians.
        decoder.decode_batch(det[:200])
        reference.decode_batch(det[:200])
        ours_times, theirs_times = [], []
        for _ in range(3 if quick else 5):
            start = time.perf_counter()
            decoder.decode_batch(det)
            ours_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            reference.decode_batch(det)
            theirs_times.append(time.perf_counter() - start)
        ours, theirs = statistics.median(ours_times), statistics.median(theirs_times)
        out[f"surface d={d}, {d} rounds, p={p}"] = {
            "detectors": c.n_detectors,
            "shots": shots,
            "qso_us_per_shot": 1e6 * ours / shots,
            "pymatching_us_per_shot": 1e6 * theirs / shots,
        }
    return out


# Largest accepted deviations for --check (all are rounding-level).
def bposd_vs_ldpc(quick: bool) -> dict:
    """QSO's BP+OSD decoder against the `ldpc` package (Roffe), shot by
    shot with the same options: on colour and bivariate bicycle codes
    with independent bit flips (every prior equal), and on circuit-level
    memory experiments of QSO's CSS circuits through Stim's detector error
    model (columns in Stim's order; the priors differ, and `ldpc` scores
    OSD candidates by Σ ln(1/p) where QSO uses Σ ln((1 − p)/p)). Also the
    logical error rate of QSO's own pipeline next to `ldpc`'s."""
    import stim
    from ldpc import BpOsdDecoder as LdpcBpOsd

    import metatron_qso as q

    rng = np.random.default_rng(20261027)
    shots = 1000 if quick else 5000
    variants = {
        "min-sum 0.625": ({}, {"bp_method": "minimum_sum", "ms_scaling_factor": 0.625}),
        "min-sum adaptive": ({"scaling": 0.0}, {"bp_method": "minimum_sum", "ms_scaling_factor": 0.0}),
        "product-sum": ({"method": "product_sum"}, {"bp_method": "product_sum"}),
    }
    out: dict = {"capacity": {}, "circuit": {}}
    codes = {
        "colour d=7": q.CssCode.color_code(7),
        "colour d=9": q.CssCode.color_code(9),
        "[[72, 12, 6]]": q.CssCode.bivariate_bicycle_code(72)[0],
        "[[144, 12, 12]]": q.CssCode.bivariate_bicycle_code(144)[0],
    }
    p = 0.05
    for name, code in codes.items():
        h = np.zeros((len(code.z_checks), code.n), dtype=np.uint8)
        for i, c in enumerate(code.z_checks):
            h[i, c] = 1
        dem = q.DetectorErrorModel(h.shape[0], [(p, list(np.flatnonzero(h[:, j])), 0) for j in range(code.n)])
        errors = (rng.random((shots, code.n)) < p).astype(np.uint8)
        syndromes = errors @ h.T % 2
        for label, (ours_kw, theirs_kw) in variants.items():
            ours = q.BpOsdDecoder(dem, **ours_kw)
            theirs = LdpcBpOsd(h, error_rate=p, max_iter=50, osd_method="osd_cs", osd_order=7, **theirs_kw)
            same = converged_same = 0
            t_ours = t_theirs = 0.0
            for s in syndromes:
                fired = list(np.flatnonzero(s))
                start = time.perf_counter()
                o = ours.decode_detailed(fired)
                t_ours += time.perf_counter() - start
                start = time.perf_counter()
                d = theirs.decode(s)
                t_theirs += time.perf_counter() - start
                x = np.zeros(code.n, dtype=np.uint8)
                x[o.errors] = 1
                same += bool(np.array_equal(x, d))
                converged_same += o.converged == bool(theirs.converge)
            out["capacity"][f"{name}, {label}"] = {
                "shots": shots,
                "correction_disagreement": 1 - same / shots,
                "convergence_disagreement": 1 - converged_same / shots,
                # One call per shot from Python, both decoders.
                "qso_us_per_shot": 1e6 * t_ours / shots,
                "ldpc_us_per_shot": 1e6 * t_theirs / shots,
            }
    # The Steane code (colour code d = 3): an error on the qubit of all
    # three checks fires every detector; both decoders converge in BP to
    # four errors (a logical error) with scaled min-sum.
    steane = q.CssCode.color_code(3)
    h = np.zeros((3, 7), dtype=np.uint8)
    for i, c in enumerate(steane.z_checks):
        h[i, c] = 1
    shared = int(np.flatnonzero(h.all(axis=0))[0])
    dem = q.DetectorErrorModel(3, [(0.01, list(np.flatnonzero(h[:, j])), 0) for j in range(7)])
    o = q.BpOsdDecoder(dem).decode_detailed([0, 1, 2])
    theirs = LdpcBpOsd(h, error_rate=0.01, max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625,
                       osd_method="osd_cs", osd_order=7)
    d = theirs.decode(np.ones(3, dtype=np.uint8))
    out["steane_shared_qubit"] = {
        "qso": o.errors,
        "ldpc": [int(j) for j in np.flatnonzero(d)],
        "qso_converged": o.converged,
        "ldpc_converged": bool(theirs.converge),
        "shared_qubit": shared,
    }
    # Exhaustive OSD after one BP iteration on random models with equal
    # priors: the same subsets in the same order.
    same = total = 0
    for _ in range(20 if quick else 100):
        m_, n_ = 8, 30
        h = (rng.random((m_, n_)) < 0.25).astype(np.uint8)
        h[rng.integers(m_), :] |= h.sum(axis=0) == 0
        dem = q.DetectorErrorModel(m_, [(0.05, list(np.flatnonzero(h[:, j])), 0) for j in range(n_)])
        ours = q.BpOsdDecoder(dem, max_iterations=1, osd="exhaustive", osd_order=4)
        theirs = LdpcBpOsd(h, error_rate=0.05, max_iter=1, bp_method="minimum_sum", ms_scaling_factor=0.625,
                           osd_method="osd_e", osd_order=4)
        for _ in range(20):
            s = h @ (rng.random(n_) < 0.15).astype(np.uint8) % 2
            if not s.any():
                continue
            x = np.zeros(n_, dtype=np.uint8)
            x[ours.decode_detailed(list(np.flatnonzero(s))).errors] = 1
            same += bool(np.array_equal(x, theirs.decode(s)))
            total += 1
    out["exhaustive_random_models"] = {"shots": total, "correction_disagreement": 1 - same / total}
    circuit_cases = {
        "colour d=5, 5 rounds, p=0.003": (q.CssCode.color_code(5), 5, 0.003),
        "[[72, 12, 6]], 6 rounds, p=0.002": (q.CssCode.bivariate_bicycle_code(72)[0], 6, 0.002),
    }
    cshots = 1000 if quick else 4000
    for name, (code, rounds, pc) in circuit_cases.items():
        c = q.css_code_memory(code, rounds, pc)
        sc = stim.Circuit(c.to_stim())
        cols = []
        for inst in sc.detector_error_model(decompose_errors=False).flattened():
            if inst.type != "error":
                continue
            t = inst.targets_copy()
            dets = sorted(x.val for x in t if x.is_relative_detector_id())
            if dets:
                obs = sum(1 << x.val for x in t if x.is_logical_observable_id())
                cols.append((inst.args_copy()[0], dets, obs))
        # QSO's own model of the circuit against Stim's.
        ours_model = {(tuple(d), o): pp for pp, d, o in c.detector_error_model().errors if d}
        theirs_model = {(tuple(d), o): pp for pp, d, o in cols}
        keys = set(ours_model) | set(theirs_model)
        model_diff = max(
            abs(ours_model.get(k, 0.0) - theirs_model.get(k, 0.0)) / max(ours_model.get(k, 0.0), theirs_model.get(k, 0.0))
            for k in keys
        )
        nd = sc.num_detectors
        h = np.zeros((nd, len(cols)), dtype=np.uint8)
        for j, (_, dets, _) in enumerate(cols):
            h[dets, j] = 1
        col_obs = np.array([o for _, _, o in cols], dtype=np.uint64)
        theirs = LdpcBpOsd(
            h, channel_probs=[pp for pp, _, _ in cols], max_iter=50, bp_method="minimum_sum",
            ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=7,
        )
        ours = q.BpOsdDecoder(q.DetectorErrorModel(nd, cols))
        det, flips = sc.compile_detector_sampler(seed=5).sample(cshots, separate_observables=True)
        actual = (flips.astype(np.uint64) << np.arange(flips.shape[1], dtype=np.uint64)).sum(axis=1)
        a = ours.decode_batch(det)
        b = np.array(
            [np.bitwise_xor.reduce(col_obs[theirs.decode(r.astype(np.uint8)).astype(bool)], initial=np.uint64(0)) for r in det],
            dtype=np.uint64,
        )
        out["circuit"][name] = {
            "shots": cshots,
            "error_mechanisms": len(cols),
            "max_relative_probability_difference_vs_stim": model_diff,
            "prediction_disagreement": float(np.mean(a != b)),
            "logical_error_rate_qso_on_stim_model": float(np.mean(a != actual)),
            "logical_error_rate_ldpc": float(np.mean(b != actual)),
            "logical_error_rate_qso_pipeline": q.logical_error_rate(c, cshots, seed=6, decoder="bposd").rate,
        }
    return out


def stim_parts(dem) -> dict:
    """A Stim detector error model's errors as {sorted parts: probability},
    each part (detectors, observable mask)."""
    out: dict = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        parts = [[[], 0]]
        for t in inst.targets_copy():
            if t.is_separator():
                parts.append([[], 0])
            elif t.is_relative_detector_id():
                parts[-1][0].append(t.val)
            elif t.is_logical_observable_id():
                parts[-1][1] |= 1 << t.val
        key = tuple(sorted((tuple(sorted(d)), o) for d, o in parts if d or o))
        e = out.get(key, 0.0)
        out[key] = e * (1 - p) + p * (1 - e)
    return out


def correlated_and_distance_vs_stim(quick: bool) -> dict:
    """QSO's decomposition of Stim's generated circuits against Stim's
    decomposed models, error by error with their parts; two-pass matching
    against PyMatching with `enable_correlations=True`, shot by shot on the
    same model (Stim's) and syndromes; circuit distances against Stim's
    `shortest_graphlike_error` (graph-like, exact on both sides) and
    `search_for_undetectable_logical_errors`, a heuristic search with
    truncation and exhaustive (so exact) without, against QSO's
    `circuit_distance` bounds and `exact_circuit_distance`."""
    import pymatching
    import stim

    import metatron_qso as q

    shots = 50_000 if quick else 500_000
    noise = dict(
        after_clifford_depolarization=0.006,
        before_round_data_depolarization=0.006,
        before_measure_flip_probability=0.006,
        after_reset_flip_probability=0.006,
    )
    cases = {
        "stim rotated_memory_z d=5": stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, **noise),
        "stim rotated_memory_x d=5": stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, **noise),
        "stim unrotated_memory_z d=3": stim.Circuit.generated("surface_code:unrotated_memory_z", distance=3, rounds=3, **noise),
        "stim repetition_code:memory d=7": stim.Circuit.generated("repetition_code:memory", distance=7, rounds=7, **noise),
    }
    # Y errors on data qubits mid-circuit flip X- and Z-type detectors.
    with_y = str(cases["stim rotated_memory_z d=5"]).replace(
        "REPEAT 4 {\n", "REPEAT 4 {\n    Y_ERROR(0.002) 1 3 5 12 25\n", 1
    )
    cases["stim rotated_memory_z d=5 with Y errors"] = stim.Circuit(with_y)
    out = {}
    for name, sc in cases.items():
        qc = q.DetectorCircuit.from_stim(str(sc))
        dem = sc.detector_error_model(decompose_errors=True)
        theirs = stim_parts(dem)
        ours: dict = {}
        for p, parts in qc.correlated_error_model().errors:
            key = tuple(sorted((tuple(d), o) for d, o in parts))
            e = ours.get(key, 0.0)
            ours[key] = e * (1 - p) + p * (1 - e)
        keys = set(theirs) | set(ours)
        both = [k for k in keys if k in theirs and k in ours]
        # The same syndromes through both two-pass decoders, on Stim's model.
        errors = [(p, [(list(d), o) for d, o in key]) for key, p in theirs.items()]
        model = q.CorrelatedErrorModel(dem.num_detectors, errors, dem.num_observables)
        det, obs = sc.compile_detector_sampler(seed=5).sample(shots, separate_observables=True)
        truth = np.zeros(shots, dtype=np.uint64)
        for k in range(obs.shape[1]):
            truth |= obs[:, k].astype(np.uint64) << np.uint64(k)
        def masks(bits: np.ndarray) -> np.ndarray:
            m = np.zeros(shots, dtype=np.uint64)
            for k in range(bits.shape[1]):
                m |= bits[:, k].astype(np.uint64) << np.uint64(k)
            return m
        correlated = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        plain = pymatching.Matching.from_detector_error_model(dem)
        reference = masks(correlated.decode_batch(det, enable_correlations=True))
        reference_plain = masks(plain.decode_batch(det))
        mine = q.CorrelatedMatchingDecoder(model).decode_batch(det)
        own = q.CorrelatedMatchingDecoder(qc.correlated_error_model()).decode_batch(det)
        # Distances.
        graphlike = len(sc.shortest_graphlike_error())
        searched = len(
            sc.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=4,
                dont_explore_edges_with_degree_above=4,
                dont_explore_edges_increasing_symptom_degree=False,
            )
        )
        exact = q.circuit_distance(qc.detector_error_model())
        out[name] = {
            "error_mechanisms": len(theirs),
            "decomposed_errors_only_one_side": len(keys) - len(both),
            "decomposed_max_relative_probability_difference": max(
                (abs(theirs[k] - ours[k]) / theirs[k] for k in both), default=0.0
            ),
            "shots": shots,
            "logical_error_rate_pymatching": float(np.mean(reference_plain != truth)),
            "logical_error_rate_pymatching_correlated": float(np.mean(reference != truth)),
            "logical_error_rate_qso_correlated": float(np.mean(mine != truth)),
            "logical_error_rate_qso_correlated_own_model": float(np.mean(own != truth)),
            "correlated_disagreement_with_pymatching": float(np.mean(mine != reference)),
            "graphlike_distance_stim": graphlike,
            "graphlike_distance_qso": q.graphlike_distance(qc.detector_error_model(decompose=True)).distance,
            "searched_distance_stim": searched,
            "circuit_distance_qso": [exact.lower_bound, exact.upper_bound],
            **exact_distances(sc, qc),
        }
    # Errors that flip three or more detectors: QSO's colour and bicycle
    # codes through Stim, and Stim's colour code (its C_XYZ gates as
    # S_DAG then H, the same Clifford up to sign) through QSO.
    hyper = {
        "qso colour d=5": q.css_code_memory(q.CssCode.color_code(5), 2, 0.001),
        "qso bicycle [[72, 12, 6]] depth-8 cycle": q.css_code_memory(
            q.CssCode.bivariate_bicycle_code(72)[0], 1, 0.001, schedule="bicycle"
        ),
    }
    for d in (5, 7):
        text = str(stim.Circuit.generated("color_code:memory_xyz", distance=d, rounds=3, **noise).flattened())
        text = "\n".join(
            f"S_DAG {line[6:]}\nH {line[6:]}" if line.startswith("C_XYZ ") else line
            for line in text.splitlines()
        )
        hyper[f"stim color_code:memory_xyz d={d}"] = q.DetectorCircuit.from_stim(text)
    for name, qc in hyper.items():
        sc = stim.Circuit(qc.to_stim())
        exact = q.circuit_distance(qc.detector_error_model())
        out[name] = {
            "searched_distance_stim": len(
                sc.search_for_undetectable_logical_errors(
                    dont_explore_detection_event_sets_with_size_above=6,
                    dont_explore_edges_with_degree_above=6,
                    dont_explore_edges_increasing_symptom_degree=False,
                )
            ),
            "circuit_distance_qso": [exact.lower_bound, exact.upper_bound],
            **exact_distances(sc, qc),
        }
    return out


def exact_distances(sc, qc) -> dict:
    """Stim's search with no truncation (exhaustive: the circuit distance)
    and QSO's `exact_circuit_distance`, with their times; QSO again on the
    orbits of the circuit's symmetries (1.47)."""
    import metatron_qso as q

    started = time.time()
    exhaustive = len(
        sc.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=sc.num_detectors,
            dont_explore_edges_with_degree_above=sc.num_detectors,
            dont_explore_edges_increasing_symptom_degree=False,
        )
    )
    middle = time.time()
    model = qc.detector_error_model()
    exact = q.exact_circuit_distance(model, max_seconds=600.0)
    plain_done = time.time()
    symmetric = q.exact_circuit_distance(model, max_seconds=600.0, symmetry=True)
    return {
        "exhaustive_distance_stim": exhaustive,
        "exhaustive_distance_stim_seconds": middle - started,
        "exact_distance_qso": [exact.lower_bound, exact.upper_bound],
        "exact_distance_qso_seconds": plain_done - middle,
        "exact_distance_qso_symmetric": [symmetric.lower_bound, symmetric.upper_bound],
        "exact_distance_qso_symmetric_seconds": time.time() - plain_done,
        "symmetry_generators": len(q.error_model_symmetries(model)),
    }


def gaussian_vs_thewalrus(rng) -> dict:
    """Hafnians, torontonians and Gaussian boson sampling against The Walrus."""
    import metatron_qso as q
    import thewalrus
    import thewalrus.quantum as tq
    from thewalrus.symplectic import expand, interferometer, loss, squeezing
    from thewalrus import threshold_detection_prob

    haf_err = 0.0
    for size in range(2, 21, 2):
        a = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        a = a + a.T
        for loop in (False, True):
            ours, theirs = q.hafnian(a, loop=loop), thewalrus.hafnian(a, loop=loop)
            haf_err = max(haf_err, abs(ours - theirs) / abs(theirs))
    b = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    b = b + b.T
    mu = rng.normal(size=3) + 1j * rng.normal(size=3)
    rep_err = max(
        abs(q.hafnian_repeated(b, [4, 2, 2]) - thewalrus.hafnian_repeated(b, [4, 2, 2]))
        / abs(thewalrus.hafnian_repeated(b, [4, 2, 2])),
        abs(q.hafnian_repeated(b, [3, 2, 2], mu) - thewalrus.hafnian_repeated(b, [3, 2, 2], mu=mu, loop=True))
        / abs(thewalrus.hafnian_repeated(b, [3, 2, 2], mu=mu, loop=True)),
    )
    # Four squeezed modes through a Haar-random interferometer, displaced,
    # one mode lossy.
    n = 4
    params = [(0.5, 0.1), (0.3, 1.0), (0.6, 0.4), (0.2, 2.0)]
    s = np.eye(2 * n)
    for i, (r, phi) in enumerate(params):
        s = expand(squeezing(r, phi), i, n) @ s
    u = q.haar_unitary(n, seed=7)
    s = interferometer(u) @ s
    cov, means = s @ s.T, np.zeros(2 * n)
    means[1], means[n + 2] = 0.4, -0.3
    means, cov = loss(means, cov, 0.7, 3)
    state = q.GaussianState(n)
    for i, (r, phi) in enumerate(params):
        state.squeeze(i, r, phi)
    state.interferometer(u)
    state = q.GaussianState.from_parts(state.cov, np.array([0, 0.4, 0, 0, 0, 0, -0.3, 0]))
    state.loss(3, 0.7)
    cov_err = float(np.max(np.abs(state.cov - cov)))
    table = tq.probabilities(means, cov, cutoff=5)
    prob_err = max(abs(state.probability(list(idx)) - table[idx]) for idx in np.ndindex(*table.shape))
    zero = q.GaussianState.from_parts(state.cov, np.zeros(2 * n))
    click_err = max(
        abs(zero.click_probability([bool(c) for c in pat]) - threshold_detection_prob(np.zeros(2 * n), state.cov, np.array(pat)).real)
        for pat in np.ndindex(*(2,) * n)
    )
    o = np.eye(2 * n) - np.linalg.inv(tq.Qmat(cov))
    tor_err = abs(q.torontonian(o) - thewalrus.tor(o)) / abs(thewalrus.tor(o))
    # Relative accuracy at higher photon numbers: 20 photons from 10
    # squeezed modes (pure), 12 photons from 6 lossy modes (mixed).
    rel_err = 0.0
    for modes, lossy, patterns in [
        (10, False, [[2] * 10, [4, 0, 4, 0, 2, 2, 0, 4, 2, 2]]),
        (6, True, [[2] * 6, [4, 0, 4, 0, 2, 2], [6, 0, 0, 6, 0, 0]]),
    ]:
        big = q.GaussianState(modes)
        for i in range(modes):
            big.squeeze(i, 0.9)
        big.interferometer(q.haar_unitary(modes, seed=modes))
        if lossy:
            for i in range(modes):
                big.loss(i, 0.9)
        cov_big, zeros = np.array(big.cov), np.zeros(2 * modes)
        for pat in patterns:
            if lossy:
                theirs = tq.density_matrix_element(zeros, cov_big, pat, pat).real
            else:
                theirs = abs(tq.pure_state_amplitude(zeros, cov_big, pat)) ** 2
            rel_err = max(rel_err, abs(big.probability(pat) - theirs) / theirs)
    return {
        "hafnian_max_relative_difference": haf_err,
        "repeated_hafnian_max_relative_difference": rep_err,
        "torontonian_relative_difference": tor_err,
        "covariance_max_difference": cov_err,
        "gbs_probability_max_difference": prob_err,
        "click_probability_max_difference": click_err,
        "gbs_probability_max_relative_difference": rel_err,
    }


def chemistry_vs_pyscf() -> dict:
    """Gaussian integrals and Hartree–Fock against PySCF, on the same
    geometries (QSO's positions in bohr, so that the two codes' bohr
    constants do not matter)."""
    import metatron_qso as q
    from pyscf import gto, mcscf, scf

    symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
               "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]

    def pyscf_molecule(m, basis, cart=False):
        atoms = [(symbols[z - 1], tuple(r)) for z, r in zip(m.charges, m.positions)]
        return gto.M(atom=atoms, basis=basis, unit="Bohr", cart=cart, charge=m.charge, spin=m.multiplicity - 1, verbose=0)

    water = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
    ammonia = "N 0.1 -0.05 0.12; H 0.9377 -0.3816 0.0; H -0.4689 -0.1 0.8121; H -0.4689 0.7 -0.8121"
    integral_err = 0.0
    cases = [(a, b, "spherical") for a in (water, ammonia, "C 0 0 0; O 0 0 1.128", "F 0 0 0; F 0.3 0.2 1.41")
             for b in ("sto-3g", "6-31g", "6-31g*", "cc-pvdz")]
    cases += [
        (water, "6-31g*", "cartesian"),
        (ammonia, "cc-pvdz", "cartesian"),
        ("H 0 0 0; Cl 0 0 1.2746", "cc-pvdz", "spherical"),
        ("S 0 0 0.1030; H 0 0.9616 -0.8239; H 0 -0.9616 -0.8239", "6-31g*", "spherical"),
        ("Si 0 0 0; H 0.86 0.86 0.86; H -0.86 -0.86 0.86; H -0.86 0.86 -0.86; H 0.86 -0.86 -0.86", "sto-3g", "spherical"),
        ("Na 0 0 0; F 0 0 1.93", "6-31g", "spherical"),
    ]
    for atoms, basis, harmonics in cases:
        m = q.Molecule(atoms)
        mol = pyscf_molecule(m, basis, cart=harmonics == "cartesian")
        ours = q.AoIntegrals(m, basis, harmonics)
        # PySCF's Cartesian d (and higher) components are not normalised to
        # one; QSO's are.
        scale = 1 / np.sqrt(np.diag(mol.intor("int1e_ovlp")))
        two = np.outer(scale, scale)
        for name, theirs in (
            ("overlap", mol.intor("int1e_ovlp") * two),
            ("kinetic", mol.intor("int1e_kin") * two),
            ("nuclear", mol.intor("int1e_nuc") * two),
            ("eri", mol.intor("int2e") * np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)),
        ):
            integral_err = max(integral_err, float(np.max(np.abs(getattr(ours, name) - theirs))))
    energy_err = 0.0
    for atoms, charge, mult, basis in (
        (water, 0, 1, "6-31g"),
        ("N 0 0 0; N 0 0 1.0977", 0, 1, "sto-3g"),
        ("Be 0 0 0; H 0 0 1.326; H 0 0 -1.326", 0, 1, "6-31g"),
        ("O 0 0 0; H 0 0 0.97", 0, 2, "6-31g"),
        ("C 0 0 0.1; H 0 0.99 -0.5; H 0 -0.99 -0.5", 0, 3, "6-31g"),
        ("N 0 0 0; O 0 0 1.151", 0, 2, "sto-3g"),
        (water, 0, 1, "cc-pvdz"),
        (ammonia, 0, 1, "6-31g*"),
        ("H 0 0 0; Cl 0 0 1.2746", 0, 1, "cc-pvdz"),
        ("S 0 0 0.1030; H 0 0.9616 -0.8239; H 0 -0.9616 -0.8239", 0, 1, "cc-pvdz"),
        ("O 0 0 0; O 0 0 1.2075", 0, 3, "cc-pvdz"),
    ):
        m = q.Molecule(atoms, charge, mult)
        mol = pyscf_molecule(m, basis)
        mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
        mf.conv_tol = 1e-12
        theirs = mf.kernel()
        ours = q.hartree_fock(m, basis, unrestricted=mult > 1)
        energy_err = max(energy_err, abs(ours.energy - theirs))
        # CASCI with one frozen orbital and 6 active (n − 2 electrons fit
        # up to 14 electrons).
        if mult == 1 and mol.nelectron <= 14:
            cas = mcscf.CASCI(mf, 6, mol.nelectron - 2)
            cas.fcisolver.conv_tol = 1e-12
            theirs_cas = cas.kernel()[0]
            ours_cas = ours.molecular_integrals(frozen=1, active=6).fci_energies()[0]
            energy_err = max(energy_err, abs(ours_cas - theirs_cas))
    # Restricted open shells and CASCI in their orbitals.
    for atoms, mult, basis, frozen, active in (
        ("O 0 0 0; O 0 0 1.2075", 3, "cc-pvdz", 2, 8),
        ("N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6", 2, "6-31g*", 2, 6),
        ("O 0 0 0; H 0 0 0.97", 2, "cc-pvdz", 1, 6),
    ):
        m = q.Molecule(atoms, 0, mult)
        mol = pyscf_molecule(m, basis)
        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-12
        theirs = mf.kernel()
        ours = q.hartree_fock(m, basis, restricted_open=True)
        energy_err = max(energy_err, abs(ours.energy - theirs))
        n_active = mol.nelectron - 2 * frozen
        na = (n_active + mult - 1) // 2
        cas = mcscf.CASCI(mf, active, (na, n_active - na))
        cas.fcisolver.conv_tol = 1e-12
        theirs_cas = cas.kernel()[0]
        ours_cas = ours.molecular_integrals(frozen=frozen, active=active).fci_energies()[0]
        energy_err = max(energy_err, abs(ours_cas - theirs_cas))
    # MP2, CCSD and (T) on molecules the unit tests do not use (1.62):
    # RHF closed shells, UHF and ROHF open shells, with frozen cores.
    from pyscf import cc, mp
    from pyscf.cc import ccsd_t, uccsd_t

    correlation_err = 0.0
    for atoms, mult, basis, kind, frozen in (
        (water, 1, "6-31g*", "rhf", 1),
        (ammonia, 1, "cc-pvdz", "rhf", 1),
        ("C 0 0 0; O 0 0 1.128", 1, "cc-pvdz", "rhf", 2),
        ("H 0 0 0; F 0 0 0.917", 1, "6-31g*", "rhf", 0),
        ("H 0 0 0; Cl 0 0 1.2746", 1, "6-31g", "rhf", 5),
        ("C 0 0 0.1; H 0 0.99 -0.5; H 0 -0.99 -0.5", 3, "6-31g", "uhf", 1),
        ("N 0 0 0; O 0 0 1.151", 2, "6-31g", "uhf", 2),
        ("C 0 0 0.1; H 0 0.99 -0.5; H 0 -0.99 -0.5", 3, "6-31g", "rohf", 1),
        ("N 0 0 0; O 0 0 1.151", 2, "6-31g", "rohf", 2),
    ):
        m = q.Molecule(atoms, 0, mult)
        mol = pyscf_molecule(m, basis)
        mf = {"rhf": scf.RHF, "uhf": scf.UHF, "rohf": scf.ROHF}[kind](mol)
        # PySCF's UHF of NO needs over a hundred cycles to reach the
        # gradient; MP2 moves with the orbitals at first order (10⁻⁸ Ha at
        # PySCF's default 50 cycles), the CC energies hardly.
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 1000
        mf.kernel()
        assert mf.converged, (atoms, kind)
        ours = q.hartree_fock(
            m, basis, unrestricted=kind == "uhf", restricted_open=kind == "rohf",
            energy_tolerance=1e-12, gradient_tolerance=1e-10,
        )
        energy_err = max(energy_err, abs(ours.energy - mf.e_tot))
        pyscf_cc = cc.CCSD(mf).set(frozen=frozen or None, conv_tol=1e-12, conv_tol_normt=1e-10, max_cycle=200)
        pyscf_cc.kernel()
        assert pyscf_cc.converged, (atoms, kind)
        if kind == "rohf":
            mine = ours.ccsd(frozen)
            correlation_err = max(correlation_err, abs(mine["correlation"] - pyscf_cc.e_corr))
            continue
        pt = mp.MP2(mf).set(frozen=frozen or None).kernel()[0]
        triples = (ccsd_t if kind == "rhf" else uccsd_t).kernel(pyscf_cc, pyscf_cc.ao2mo(), verbose=0)
        mine_t = ours.ccsd_t(frozen)
        correlation_err = max(
            correlation_err,
            abs(ours.mp2(frozen)["correlation"] - pt),
            abs(mine_t["ccsd"]["correlation"] - pyscf_cc.e_corr),
            abs(mine_t["triples"] - triples),
        )
    return {
        "integral_max_difference": integral_err,
        "energy_max_difference": energy_err,
        "correlation_max_difference": correlation_err,
    }

TOLERANCES = {
    "state": 1e-12,
    "gradient": 1e-12,
    "permanent": 1e-10,
    # QSO sums 2^{n/2} terms with alternating signs: up to 7·10⁻¹² relative
    # at 20 × 20 (The Walrus: 10⁻¹⁵ against exact integer hafnians); the
    # same for 20-photon probabilities.
    "hafnian": 1e-10,
    "gaussian_probability": 1e-12,
    "density_matrix": 1e-12,
    # Single precision: 1 − F of the direction, measured 10⁻¹² for both
    # simulators on 60-gate circuits.
    "single_precision": 1e-9,
    "error_model": 1e-12,
    # Detection-event rates are sampled: |z| over up to 72 detectors.
    "detector_rate_z": 5.0,
    # Two exact matching decoders differ only on ties and on rounded
    # weights.
    "matching_disagreement": 1e-3,
    # Both decoders round ln((1 − p)/p) to integers (QSO: the largest edge
    # at 2²⁰).
    "matching_weight": 1e-5,
    # BP+OSD against ldpc: with equal priors the same arithmetic (min-sum
    # exactly; product-sum differs where tanh saturates); on circuit
    # models the OSD scores differ (Σ ln(1/p) against Σ ln((1 − p)/p)).
    "bposd_min_sum_disagreement": 1e-3,
    "bposd_product_sum_disagreement": 1e-2,
    "bposd_circuit_disagreement": 1e-2,
    # Two-pass matching: the first pass's shortest paths may break ties
    # differently from PyMatching's, so a few shots reweight other edges.
    "correlated_disagreement": 1e-3,
    # Same geometries in bohr: integrals to rounding, energies to the
    # convergence of both SCF codes.
    "integral": 1e-12,
    "scf_energy": 1e-9,
    # MP2, CCSD and (T) correlation energies: both CCSD codes converged to
    # 10⁻¹⁰ in the energy (1.62's plan asks for 10⁻⁸ hartree).
    "correlation_energy": 1e-8,
    # QAOA energy and gradient at 22–26 qubits, relative to |⟨C⟩|: double
    # precision against each other to rounding of 2^n-term sums; QSO's
    # single precision against double (measured up to 1.1·10⁻⁸ for the
    # energy and 1.3·10⁻⁶ for the gradient).
    "qaoa_double": 1e-10,
    "spin_expectation": 1e-9,
    "spin_mesolve": 1e-3,
    "spin_hamiltonian": 1e-13,
    # The coherence of an NV centre's ¹³C bath by the cluster-correlation
    # expansion: PyCCE keeps the full spin-1 electron with D = 2.87 GHz,
    # QSO the secular Hamiltonian, which differ by about (A/D)·A (measured
    # 7·10⁻⁶ on a natural bath of 182 spins at 50 mT).
    "redfield_rate": 1e-12,
    "bath_coherence": 2e-4,
    # The generalised expansion: both propagate the same Hamiltonian in the
    # laboratory frame, where the electron's phase (GHz over milliseconds,
    # ωt ≈ 4·10⁷) rounds to ε·ωt ≈ 10⁻⁸ per step (measured 5·10⁻⁸).
    "generalised_bath_coherence": 1e-6,
    "redfield_expectation": 1e-7,
    "qaoa_single": 1e-5,
    # Transport to a trap (1.63), efficiency absolute, mean arrival time
    # relative, against exact rationals and closed forms (SymPy; measured
    # 2·10⁻¹⁴ and 4·10⁻¹⁴) and QuTiP's liouvillian solved by sparse LU
    # (measured 1.7·10⁻¹³ on FMO without dephasing, condition 9·10⁴, and
    # 3·10⁻¹⁴); QSO's GMRES stops at its own residual tolerance.
    "transport_efficiency": 1e-11,
    "transport_arrival": 1e-10,
    # QuTiP's mesolve at atol 1e-13, rtol 1e-11, stopped when less than
    # 10⁻¹² is left in the system: measured 6.7·10⁻¹³ in η and 6.8·10⁻¹¹
    # in τ.
    "transport_mesolve": 1e-9,
    # The dimer's dephasing rate of highest efficiency (exactly 39/20), a
    # golden-section search on a flat maximum: measured 1.3·10⁻⁷.
    "transport_best_rate": 1e-6,
    # Pauli propagation (1.64): values against Aer's state vector of the
    # light cone (2^26 amplitudes at 7 steps; Aer and obp agree to
    # 1.8·10⁻¹⁴); the Heisenberg operator string for string against
    # qiskit-addon-obp, which trims |c| ≤ 10⁻¹⁵ after every gate (measured
    # 6.3·10⁻¹⁵ at θ_h = 0.3 and 5 steps, 134 744 strings each); noisy
    # values against obp and Aer's density matrix (Aer against an
    # independent NumPy density matrix: 1.5·10⁻¹⁵).
    "propagation_value": 1e-10,
    "propagation_coefficient": 1e-13,
    "propagation_noisy": 1e-12,
}


def spins_vs_qutip() -> dict:
    """Spin dynamics with relaxation against QuTiP's master-equation solver.

    Everything on the QuTiP side is built here from the documented
    formulas with QuTiP's own spin operators (``jmat``, ``tensor``): the
    Hamiltonian of the frame, the thermal state, parahydrogen, the ideal
    pulses and the Lindblad operators (T1 towards thermal equilibrium, pure
    dephasing from T2). QSO runs each sequence; on the QuTiP side its
    delays and the observation are propagated twice: exactly, by the
    exponential (SciPy) of QuTiP's ``liouvillian`` applied to the deviation
    from 1/d with L(1/d) as an augmented column, and by ``mesolve``.
    Compared are the Hamiltonians and, over time, every spin's I_z and each
    isotope's F_x, F_y, relative to the largest of them (at room
    temperature they are parts in 10^5). ``mesolve`` is held to a looser
    tolerance: its adaptive steps lose about 10^-6 of the deviation over
    10^4 rad of precession, and 10^-4 in the laboratory frame at 9.4 T.
    """
    import scipy.linalg
    import metatron_qso as q
    import qutip as qt
    from metatron_qso import spin

    hbar, kb = 1.054571817e-34, 1.380649e-23
    gamma = {iso: q.spin.gyromagnetic_ratio(iso) for iso in ("1H", "13C", "15N")}
    two_pi = 2 * math.pi
    options = {"atol": 1e-15, "rtol": 1e-12, "nsteps": 10**8}

    def op(spins, i, which):
        return qt.tensor([qt.jmat(0.5, which) if k == i else qt.qeye(2) for k in range(len(spins))])

    def hamiltonian(field, spins, couplings, frame):
        h = 0
        for i, (iso, shift, _, _) in enumerate(spins):
            factor = (1 + shift * 1e-6) if frame == "laboratory" else shift * 1e-6
            h += -gamma[iso] * field * factor * op(spins, i, "z")
        for i, j, jh in couplings:
            if frame == "laboratory" or spins[i][0] == spins[j][0]:
                h += two_pi * jh * sum(op(spins, i, w) * op(spins, j, w) for w in "xyz")
            else:
                h += two_pi * jh * op(spins, i, "z") * op(spins, j, "z")
        return h

    def boltzmann(field, temperature, iso, shift):
        return hbar * gamma[iso] * field * (1 + shift * 1e-6) / (kb * temperature)

    def collapse(field, temperature, spins):
        ops = []
        for i, (iso, shift, t1, t2) in enumerate(spins):
            r1 = 0.0 if t1 is None else 1 / t1
            if r1 > 0:
                up = r1 / (1 + math.exp(-boltzmann(field, temperature, iso, shift)))
                ops += [math.sqrt(r1 - up) * op(spins, i, "-"), math.sqrt(up) * op(spins, i, "+")]
            if t2 is not None and 1 / t2 - r1 / 2 > 0:
                ops.append(math.sqrt(2 * (1 / t2 - r1 / 2)) * op(spins, i, "z"))
        return ops

    def initial(field, temperature, spins, kind):
        if kind == "thermal":
            factors = []
            for iso, shift, _, _ in spins:
                x = boltzmann(field, temperature, iso, shift)
                w = np.array([math.exp(x / 2), math.exp(-x / 2)])
                factors.append(qt.Qobj(np.diag(w / w.sum())))
            return qt.tensor(factors)
        d = 2 ** len(spins)
        dot = sum(op(spins, 0, w) * op(spins, 1, w) for w in "xyz")
        return qt.tensor([qt.qeye(2)] * len(spins)) / d - 4 / d * dot

    def exact(liouvillian, rho, t):
        """exp(Lt) on the deviation from 1/d, with L(1/d) as an augmented column."""
        d = rho.shape[0]
        n = d * d
        trace = rho.tr()
        mixed = (np.eye(d) / d).reshape(-1, order="F")
        aug = np.zeros((n + 1, n + 1), dtype=complex)
        aug[:n, :n] = liouvillian * t
        aug[:n, n] = liouvillian @ mixed * t * trace
        p = scipy.linalg.expm(aug)
        dev = rho.full().reshape(-1, order="F") - mixed * trace
        out = p[:n, :n] @ dev + p[:n, n] + mixed * trace
        return qt.Qobj(out.reshape(d, d, order="F"), dims=rho.dims)

    def run_qutip(case, integrate):
        field, temperature, spins, couplings, frame, kind, sequence, duration = case
        h = hamiltonian(field, spins, couplings, frame)
        c_ops = collapse(field, temperature, spins)
        liouvillian = qt.liouvillian(h, c_ops).full()
        rho = initial(field, temperature, spins, kind)
        for step in sequence:
            if step[0] == "pulse":
                _, iso, angle, phase = step
                g = sum(
                    math.cos(math.radians(phase)) * op(spins, i, "x") + math.sin(math.radians(phase)) * op(spins, i, "y")
                    for i, s in enumerate(spins)
                    if s[0] == iso
                )
                u = (-1j * math.radians(angle) * g).expm()
                rho = u * rho * u.dag()
            elif integrate:
                rho = qt.mesolve(h, rho, [0.0, step[1]], c_ops, options=options).states[-1]
            else:
                rho = exact(liouvillian, rho, step[1])
        ops = [op(spins, i, "z") for i in range(len(spins))]
        for iso in sorted({s[0] for s in spins}):
            ops += [sum(op(spins, i, w) for i, s in enumerate(spins) if s[0] == iso) for w in "xy"]
        times = np.linspace(0.0, duration, 11)
        if integrate:
            return h, times, qt.mesolve(h, rho, times, c_ops, e_ops=ops, options=options).expect
        states = [exact(liouvillian, rho, t) for t in times]
        return h, times, [[qt.expect(o, r) for r in states] for o in ops]

    def run_qso(case, times):
        field, temperature, spins, couplings, frame, kind, sequence, _ = case
        sys = q.SpinSystem(field, temperature=temperature, frame=frame)
        for iso, shift, t1, t2 in spins:
            sys.add_spin(iso, shift, t1=t1, t2=t2)
        for i, j, jh in couplings:
            sys.couple(i, j, jh)
        rho = sys.thermal_state() if kind == "thermal" else sys.parahydrogen_state(0, 1)
        events = [spin.pulse(s[1], s[2], s[3]) if s[0] == "pulse" else spin.delay(s[1]) for s in sequence]
        rho = sys.evolve(rho, events)
        ops = [sys.spin_operator(i, "z") for i in range(len(spins))]
        for iso in sorted({s[0] for s in spins}):
            ops += [sys.total_operator(iso, "x"), sys.total_operator(iso, "y")]
        values = [[sys.expectation(sys.evolve(rho, [spin.delay(float(t))]), o).real for t in times] for o in ops]
        return sys.hamiltonian(), values

    tau = 1 / (4 * 140.0)
    inept = [
        ("pulse", "1H", 90.0, 0.0), ("delay", tau), ("pulse", "1H", 180.0, 0.0), ("pulse", "13C", 180.0, 0.0),
        ("delay", tau), ("pulse", "1H", 90.0, 90.0), ("pulse", "13C", 90.0, 0.0),
    ]
    cases = {
        "three protons, T1/T2, 1 T, 0.05 K": (
            1.0, 0.05, [("1H", 0.0, 0.4, 0.1), ("1H", 2.0, 0.6, 0.15), ("1H", 5.0, 0.8, 0.2)],
            [(0, 1, 6.0), (1, 2, -3.0)], "rotating", "thermal", [("pulse", "1H", 70.0, 20.0)], 0.2,
        ),
        "1H-1H-13C-15N after INEPT, T1/T2, 9.4 T, 298 K, 1 s": (
            9.4, 298.15, [("1H", 1.0, 0.5, 0.2), ("1H", 3.0, 0.7, 0.3), ("13C", 40.0, 1.5, 0.5), ("15N", 120.0, 2.0, 0.6)],
            [(0, 1, 7.0), (0, 2, 140.0), (1, 2, 5.0), (2, 3, -12.0)], "rotating", "thermal", inept, 1.0,
        ),
        "five protons, T1/T2, 9.4 T, 298 K": (
            9.4, 298.15, [("1H", float(k), 0.5 + 0.1 * k, 0.2) for k in range(5)],
            [(k, k + 1, 3.0 + k) for k in range(4)], "rotating", "thermal", [("pulse", "1H", 60.0, 0.0)], 0.2,
        ),
        "two protons, laboratory frame, 9.4 T, 298 K": (
            9.4, 298.15, [("1H", 1.0, 0.8, 0.3), ("1H", 3.0, 1.2, 0.4)],
            [(0, 1, 7.0)], "laboratory", "thermal", [("pulse", "1H", 90.0, 0.0)], 1e-3,
        ),
        "SABRE-SHEATH, 1H-1H-15N, 0.14 uT, laboratory frame": (
            1.4e-7, 298.15, [("1H", -22.0, None, None), ("1H", -22.0, None, None), ("15N", 250.0, None, None)],
            [(0, 1, -7.0), (0, 2, -2.0)], "laboratory", "parahydrogen", [], 2.0,
        ),
    }
    def relative(values, expect):
        scale = max(np.abs(np.asarray(e, dtype=float)).max() for e in expect)
        diff = max(float(np.abs(np.asarray(v) - np.asarray(e, dtype=float)).max()) for v, e in zip(values, expect))
        return diff / scale, float(scale)

    out = {}
    worst, worst_mesolve, worst_h = 0.0, 0.0, 0.0
    for name, case in cases.items():
        h_ref, times, expect = run_qutip(case, integrate=False)
        _, _, integrated = run_qutip(case, integrate=True)
        h, values = run_qso(case, times)
        h_diff = float(np.abs(h - h_ref.full()).max() / np.abs(h_ref.full()).max())
        diff, scale = relative(values, expect)
        diff_mesolve, _ = relative(values, integrated)
        worst, worst_mesolve, worst_h = max(worst, diff), max(worst_mesolve, diff_mesolve), max(worst_h, h_diff)
        out[name] = {
            "spins": len(case[2]),
            "largest_expectation": scale,
            "max_relative_expectation_difference": diff,
            "max_relative_expectation_difference_mesolve": diff_mesolve,
            "hamiltonian_relative_difference": h_diff,
        }
    out["max_relative_expectation_difference"] = worst
    out["max_relative_expectation_difference_mesolve"] = worst_mesolve
    out["max_hamiltonian_relative_difference"] = worst_h
    return out


def import_pycce():
    """PyCCE 1.1.1 (its last release) on current NumPy, SciPy and pandas:
    aliases for the names they removed, nothing else changed."""
    import pandas as pd
    import scipy.sparse._sputils as new
    import scipy.sparse.sputils as old

    for name, alias in (("unicode_", "str_"), ("float_", "float64"), ("complex_", "complex128")):
        if not hasattr(np, name):
            setattr(np, name, getattr(np, alias))
    old.isintlike = new.isintlike
    read_csv = pd.read_csv

    def compat(*args, **kw):
        if kw.pop("delim_whitespace", False):
            kw["sep"] = r"\s+"
        return read_csv(*args, **kw)

    pd.read_csv = compat
    try:
        import pycce
    finally:
        pd.read_csv = read_csv
    return pycce


def bath_vs_pycce() -> dict:
    """The Hahn-echo coherence of an NV centre's ¹³C bath against PyCCE
    (Onizhuk and Galli 2021), conventional CCE to second order with the
    same pair cutoff: six ¹³C close together with every pair kept, and a
    random natural bath within 3 nm (pairs within 0.8 nm), at 50 mT. Both
    take QSO's positions and point-dipole hyperfine couplings (the secular
    row A_z·, kHz in PyCCE); PyCCE propagates the electron as a spin 1 with
    D = 2.87 GHz, QSO in the secular approximation."""
    import warnings

    from metatron_qso import spin

    warnings.filterwarnings("ignore", module="pycce")
    pycce = import_pycce()
    field = 0.05
    compact = np.array(
        [[0.6, 0.2, 0.4], [0.9, 0.1, 0.1], [0.5, 0.6, 0.0], [0.8, 0.5, 0.5], [1.4, -0.3, 0.9], [-0.8, 0.9, 1.1]]
    ) * 1e-9
    baths = {
        "6 13C, all pairs": (
            spin.SpinBath(field, [(list(p), spin.point_dipole_hyperfine(list(p))) for p in compact]),
            math.inf,
        ),
        "natural bath within 3 nm (seed 3), pairs within 0.8 nm": (
            spin.SpinBath.diamond(field, radius=3e-9, seed=3),
            8e-10,
        ),
    }
    times = np.linspace(0.05e-3, 1.5e-3, 10)
    out = {}
    for name, (bath, cutoff) in baths.items():
        start = time.perf_counter()
        ours = bath.coherence([spin.hahn(t / 2) for t in times], order=2, cutoff=cutoff)
        ours_time = time.perf_counter() - start
        spins = bath.spins
        atoms = pycce.BathArray((len(spins),), names=["13C"] * len(spins))
        atoms["xyz"] = np.array([p for p, _ in spins]) * 1e10
        for i, (_, hf) in enumerate(spins):
            a = np.zeros((3, 3))
            a[2, :] = np.array(hf) / 1e3
            a[:, 2] = a[2, :]
            atoms["A"][i] = a
        r_dipole = 1e3 if math.isinf(cutoff) else cutoff * 1e10
        calc = pycce.Simulator(
            spin=1, position=[0, 0, 0], alpha=[0, 1, 0], beta=[0, 0, 1],
            magnetic_field=[0, 0, field * 1e4], bath=atoms, r_bath=1e3, r_dipole=r_dipole,
            order=2, pulses=1, D=2.87e6, gyro=-17608.597050,
        )
        start = time.perf_counter()
        theirs = np.asarray(calc.compute(times * 1e3, method="cce", quantity="coherence"))
        their_time = time.perf_counter() - start
        out[name] = {
            "spins": len(spins),
            "max_coherence_difference": float(np.abs(ours - theirs).max()),
            "qso_seconds": ours_time,
            "pycce_seconds": their_time,
        }
    out["max_coherence_difference"] = max(r["max_coherence_difference"] for r in out.values())
    return out


def generalised_bath_vs_pycce() -> dict:
    """The generalised cluster-correlation expansion (the electron, a spin 1
    with D = 2.87 GHz, in every cluster; full point-dipole hyperfine
    tensors; fields of any direction) against PyCCE's ``gcce``, both with
    ideal pulses about axes fixed in the laboratory frame (QSO's
    ``pulses="laboratory"``), the Hahn-echo coherence relative to the
    electron alone. PyCCE takes the qubit as the eigenvectors of the
    electron's Hamiltonian, the electron's gyromagnetic ratio of QSO and,
    for ¹³C, QSO's 6.728284·10⁷ rad/s/T (its own differs by 2·10⁻⁷)."""
    import warnings

    from metatron_qso import spin

    warnings.filterwarnings("ignore", module="pycce")
    pycce = import_pycce()
    d, gamma_e = 2.87e9, 28.025e9
    s = 1 / math.sqrt(2)
    sx = s * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex)
    sy = s * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], complex)
    sz = np.diag([1.0, 0.0, -1.0]).astype(complex)

    def qubit(field):
        # The electron's eigenvectors closest to m_s = 0 and -1 (basis +1, 0, -1).
        w, v = np.linalg.eigh(d * sz @ sz + gamma_e * (field[0] * sx + field[1] * sy + field[2] * sz))
        a = int(np.argmax(abs(v[1, :])))
        b = max((j for j in range(3) if j != a), key=lambda j: abs(v[2, j]))
        return v[:, a], v[:, b]

    compact = np.array(
        [[0.6, 0.2, 0.4], [0.9, 0.1, 0.1], [0.5, 0.6, 0.0], [0.8, 0.5, 0.5], [1.4, -0.3, 0.9], [-0.8, 0.9, 1.1]]
    ) * 1e-9
    tilt = lambda b, angle: (b * math.sin(angle), 0.0, b * math.cos(angle))  # noqa: E731
    cases = {
        "6 13C, all pairs, 50 mT along the axis, order 3": (
            spin.GeneralisedBath((0.0, 0.0, 0.05), [(list(p), spin.point_dipole_tensor(list(p))) for p in compact]),
            math.inf,
            3,
        ),
        "6 13C, all pairs, 5 mT at 0.3 rad, order 3": (
            spin.GeneralisedBath(tilt(0.005, 0.3), [(list(p), spin.point_dipole_tensor(list(p))) for p in compact]),
            math.inf,
            3,
        ),
        "natural bath within 2.5 nm (seed 3), pairs within 0.8 nm, 20 mT at 1 rad, order 2": (
            spin.GeneralisedBath.diamond(tilt(0.02, 1.0), radius=2.5e-9, seed=3, min_radius=5e-10),
            8e-10,
            2,
        ),
    }
    times = np.linspace(0.05e-3, 1.5e-3, 10)
    out = {}
    for name, (bath, cutoff, order) in cases.items():
        start = time.perf_counter()
        ours = bath.coherence([spin.hahn(t / 2) for t in times], order=order, cutoff=cutoff, pulses="laboratory")
        ours_time = time.perf_counter() - start
        spins = bath.spins
        atoms = pycce.BathArray((len(spins),), names=["13C"] * len(spins))
        atoms.add_type("13C", 0.5, 6.728284)
        atoms["xyz"] = np.array([p for p, _ in spins]) * 1e10
        for i, (_, tensor) in enumerate(spins):
            atoms["A"][i] = np.array(tensor) / 1e3
        alpha, beta = qubit(bath.field)
        calc = pycce.Simulator(
            spin=1, position=[0, 0, 0], alpha=alpha, beta=beta,
            magnetic_field=list(np.array(bath.field) * 1e4), bath=atoms, r_bath=1e3,
            r_dipole=1e3 if math.isinf(cutoff) else cutoff * 1e10,
            order=order, pulses=1, D=d / 1e3, gyro=-2 * math.pi * gamma_e / 1e7,
        )
        start = time.perf_counter()
        theirs = np.asarray(calc.compute(times * 1e3, method="gcce", quantity="coherence"))
        their_time = time.perf_counter() - start
        out[name] = {
            "spins": len(spins),
            # QSO's coherence is ⟨β′|ρ|α′⟩, PyCCE's ⟨α′|ρ|β′⟩.
            "max_coherence_difference": float(np.abs(ours - np.conj(theirs)).max()),
            "qso_seconds": ours_time,
            "pycce_seconds": their_time,
        }
    out["max_coherence_difference"] = max(r["max_coherence_difference"] for r in out.values())
    return out


def redfield_vs_qutip() -> dict:
    """Redfield relaxation against QuTiP's Bloch–Redfield tensor.

    The bath operators of isotropic tumbling are built here from an exact
    numerical average over orientations (Gauss–Legendre in cos β, uniform
    in the other Euler angles) of the dipolar and shielding tensors, not
    from the closed form QSO uses; QuTiP's ``bloch_redfield_tensor``
    (non-secular) turns them into the relaxation superoperator. Compared,
    in the laboratory frame: the relaxation rates among I_z, I_+ and
    two-spin orders, and expectations after exact propagation by the
    exponential of QuTiP's tensor. The temperature is 10^9 K, as QuTiP's
    symmetric spectra relax to the identity.
    """
    import metatron_qso as q
    import qutip as qt
    import scipy.linalg
    from metatron_qso import spin

    def rotations(n=8):
        nodes, weights = np.polynomial.legendre.leggauss(n)
        rz = lambda x: np.array([[math.cos(x), -math.sin(x), 0], [math.sin(x), math.cos(x), 0], [0, 0, 1]])
        out = []
        for a in np.arange(n) * 2 * np.pi / n:
            for cb, wb in zip(nodes, weights):
                b = math.acos(cb)
                ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
                for g in np.arange(n) * 2 * np.pi / n:
                    out.append((rz(a) @ ry @ rz(g), wb / (2 * n * n)))
        return out

    grid = rotations()
    gamma = {iso: q.spin.gyromagnetic_ratio(iso) for iso in ("1H", "13C", "15N")}

    def run(field, tau, spins, csa, couplings, times):
        n = len(spins)
        sys = q.SpinSystem(field, temperature=1e9, frame="laboratory")
        for k, (iso, shift, pos) in enumerate(spins):
            sys.add_spin(iso, shift)
            sys.set_position(k, pos)
        for k, tensor in csa.items():
            sys.set_csa(k, np.asarray(tensor))
        for i, j, jh in couplings:
            sys.couple(i, j, jh)
        sys = sys.with_correlation_time(tau)

        def op(i, which):
            return qt.tensor([qt.jmat(0.5, which) if k == i else qt.qeye(2) for k in range(n)])

        h = sum(-gamma[iso] * field * (1 + shift * 1e-6) * op(i, "z") for i, (iso, shift, _) in enumerate(spins))
        for i, j, jh in couplings:
            h += 2 * math.pi * jh * sum(op(i, w) * op(j, w) for w in "xyz")
        interactions = []
        for i in range(n):
            for j in range(i + 1, n):
                r = np.subtract(spins[j][2], spins[i][2]) * 1e-10
                dist = np.linalg.norm(r)
                b = 1.00000000055e-7 * gamma[spins[i][0]] * gamma[spins[j][0]] * 1.054571817e-34 / dist**3
                u = r / dist
                interactions.append((b * (np.eye(3) - 3 * np.outer(u, u)), lambda a, c, i=i, j=j: op(i, "xyz"[a]) * op(j, "xyz"[c])))
        for k, tensor in csa.items():
            t = np.asarray(tensor)
            aniso = (t + t.T) / 2 - np.trace(t) / 3 * np.eye(3)
            w = gamma[spins[k][0]] * field * 1e-6
            interactions.append((w * aniso, lambda a, c, k=k: op(k, "xyz"[a]) if c == 2 else 0 * op(k, "z")))
        keys = [(k, a, c) for k in range(len(interactions)) for a in range(3) for c in range(3)]
        cov = np.zeros((len(keys), len(keys)))
        for rot, w in grid:
            v = np.array([(rot @ interactions[k][0] @ rot.T)[a, c] for k, a, c in keys])
            cov += w * np.outer(v, v)
        values, vectors = np.linalg.eigh(cov)
        spectrum = lambda w: 2 * tau / (1 + (w * tau) ** 2)
        a_ops = []
        for lam, vec in zip(values, vectors.T):
            if lam > 1e-12 * values.max():
                a = sum(x * interactions[k][1](a, c) for x, (k, a, c) in zip(vec, keys) if abs(x) > 1e-14)
                a_ops.append((math.sqrt(lam) * a, spectrum))
        full = qt.bloch_redfield_tensor(h, a_ops, sec_cutoff=-1, fock_basis=True).full()
        dissipation = full - qt.liouvillian(h).full()
        vec = lambda m: np.asarray(m).reshape(-1, order="F")
        ops = [op(i, "z") for i in range(n)] + [op(i, "+") for i in range(n)]
        ops += [2 * op(0, "z") * op(i, "z") for i in range(1, n)]
        rates = sys.relaxation_rates([o.full() for o in ops])
        want = np.array([[-(vec(oi.full()).conj() @ dissipation @ vec(oj.full())).real / (oi.dag() * oi).tr().real for oj in ops] for oi in ops])
        rate_diff = float(np.abs(rates - want).max() / np.abs(want).max())
        d = 2**n
        rho0 = np.eye(d) / d + (0.1 * op(0, "z") + 0.05 * op(1, "x") + 0.03 * 2 * op(1, "z") * op(0, "z")).full()
        observed = [op(i, "z") for i in range(n)] + [op(1, "x"), op(0, "y")]
        diff = 0.0
        for t in times:
            ref = (scipy.linalg.expm(full * t) @ vec(rho0)).reshape(d, d, order="F")
            got = sys.evolve(rho0, [spin.delay(float(t))])
            for o in observed:
                diff = max(diff, abs(np.trace(got @ o.full()).real - np.trace(ref @ o.full()).real))
        return {"spins": n, "max_relative_rate_difference": rate_diff, "max_expectation_difference": diff / 0.1}

    tetrahedral = [(0.0, 0.0, 0.0)] + [
        (1.09 * x, 1.09 * y, 1.09 * z)
        for x, y, z in [(0.943, 0.0, -0.333), (-0.471, 0.816, -0.333), (-0.471, -0.816, -0.333)]
    ]
    cases = {
        "13C-1H-1H with CSA, 1 T, 1 ns": (
            1.0, 1e-9, [("13C", 60.0, (0.0, 0.0, 0.0)), ("1H", 2.0, (1.09, 0.0, 0.0)), ("1H", 3.5, (-0.36, 1.03, 0.0))],
            {0: [[-40.0, 12.0, 0.0], [12.0, 15.0, -8.0], [0.0, -8.0, 25.0]]}, [(0, 1, 140.0), (0, 2, 135.0), (1, 2, -12.0)],
            [0.01, 0.1, 0.3],
        ),
        "13CH3 with CSA, 1 T, 0.3 ns": (
            1.0, 3e-10, [("13C", 25.0, tetrahedral[0])] + [("1H", 1.0 + 0.2 * k, p) for k, p in enumerate(tetrahedral[1:])],
            {0: [[-20.0, 0.0, 0.0], [0.0, -20.0, 0.0], [0.0, 0.0, 40.0]]}, [(0, k, 125.0) for k in (1, 2, 3)],
            [0.05, 0.5],
        ),
        "15N-1H amide with CSA (TROSY cross-correlation), 0.5 T, 10 ns": (
            0.5, 1e-8, [("15N", 120.0, (0.0, 0.0, 0.0)), ("1H", 8.3, (1.02, 0.0, 0.0))],
            {0: [[110.0, 0.0, 0.0], [0.0, -55.0, 0.0], [0.0, 0.0, -55.0]]}, [(0, 1, -92.0)],
            [0.01, 0.1],
        ),
    }
    out = {name: run(*case) for name, case in cases.items()}
    out["max_relative_rate_difference"] = max(r["max_relative_rate_difference"] for r in out.values())
    out["max_expectation_difference"] = max(r["max_expectation_difference"] for r in out.values() if isinstance(r, dict))
    return out


def node_class(u: int) -> str:
    """The class of node u of K_N in transport_vs_qutip: start 0, trap 1, the rest."""
    return "s" if u == 0 else ("t" if u == 1 else "r")


def qso_transport(H, start, traps, loss, dephasing, kind="dense", **extra) -> dict:
    """QSO's transport to a trap (1.63) for one case of transport_vs_qutip.

    `kind` picks the path: "dense" (`DephasingWalk` on the graph whose −A
    carries H's off-diagonal and whose site energies are H's diagonal),
    "orbital" (K_N with the stabiliser of nodes 0 and 1: `extra` holds N
    and the class parameters) or "hypercube" (`extra` holds d and the
    energies and losses by Hamming weight; trap at the antipode, κ = 1).
    """
    import metatron_qso as q

    if kind == "orbital":  # K_N; Python's edge cap allows N ≤ 2896
        N, p = extra["N"], extra["params"]
        per_node = lambda key: [float(p[key][node_class(u)]) for u in range(N)]  # noqa: E731
        sym = [[0, 1, 3, 2] + list(range(4, N)), [0, 1] + list(range(3, N)) + [2]]
        w = q.SymmetricOpenWalk(q.Graph.complete(N), "adjacency", -float(p["J"]), symmetries=sym,
                                site_energies=per_node("eps"), traps=[(1, float(p["kappa"]))],
                                losses=per_node("Gamma"), node_dephasing=per_node("gamma"))
        return w.transport(0)
    if kind == "hypercube":
        d = extra["d"]
        w = q.HypercubeOpenWalk(d, 1.0, site_energies=extra["energies"], losses=extra["losses"], traps=[(d, 1.0)])
        return w.transport()
    return transport_walk(H, traps, loss, dephasing).transport(start)


def transport_walk(H, traps, loss, dephasing):
    """QSO's `DephasingWalk` with Hamiltonian H (real symmetric)."""
    import metatron_qso as q

    H = np.asarray(H, dtype=float)
    n = len(H)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if H[i, j] != 0.0]
    g = q.Graph.from_edges(n, edges, weights=[-float(H[i, j]) for i, j in edges])  # H = −A off the diagonal
    return q.DephasingWalk(g, "adjacency", 1.0, site_energies=[float(e) for e in np.diag(H)],
                           traps=list(traps.items()), losses=[float(v) for v in loss],
                           node_dephasing=[float(v) for v in dephasing])


def transport_vs_qutip(quick: bool) -> dict:
    """Transport to a trap (1.63) against QuTiP and SymPy.

    The model: H_eff = H − iΣ_t κ_t P_t − (i/2)Σ_j Γ_j P_j and dephasing
    √γ_j P_j; a trap feeds its sink at 2κ_t ρ_tt, a node loses Γ_j ρ_jj;
    η = 2Σ_t κ_t X_tt and τ = 2Σ_t κ_t Y_tt/η with X = ∫ρ dt, Y = ∫tρ dt.
    QuTiP gets the system plus a sink and a loss level and the collapse
    operators √(2κ_t)|sink⟩⟨t|, √Γ_j|loss⟩⟨j|, √γ_j|j⟩⟨j|, built here. Two
    QuTiP routes: ``liouvillian`` restricted to the system block (the sink
    and loss levels only receive) with two sparse solves, L X = −ρ0 and
    L Y = −X; and ``mesolve`` until less than 10⁻¹² of the population is
    left in the system (the horizon from the slowest decay rate, checked
    by SciPy's exponential of the same block), with an accumulator level
    whose row, added to QuTiP's Liouvillian, integrates t·dp_sink/dt along
    with the state: η = p_sink(T), τ = acc(T)/η. SymPy gives exact rationals:
    the detuned dimer (η = 32000/38881, τ = 139260/38881 at dephasing 1;
    highest efficiency 800/963 at rate 39/20), and K_N on the ten orbitals
    of the stabiliser of start and trap (orbits on ordered node pairs ss,
    tt, st, ts, sr, rs, tr, rt, rr, rq), with site energies, losses and
    dephasing constant on {s}, {t} and the rest, the reduced generator
    built from the orbit sums and checked against SymPy's solve of the full
    Liouvillian at N = 4–6. Closed forms on K_N without loss (Caruso et
    al.): η = 1/(N − 1) without dephasing, 1/(N − 2) with dephasing at one
    other node, 1 with dephasing at every node. Compared: QSO's dense
    (`DephasingWalk`), orbital (`SymmetricOpenWalk`, K_64 and in the full
    run K_1024) and hypercube (`HypercubeOpenWalk`, trap at the antipode,
    against QuTiP on the full cube) solves, and its best dephasing rate.

    Runs in a child process pinned to one thread (`--transport-only`):
    QuTiP's data layer starts OpenMP threads on these tiny matrices (the
    quick run, unpinned on an idle 4-core machine, used 19 s of CPU in
    7.6 s; pinned, 7.5 s in 7.3 s), which compete with any other load.
    """
    import qutip as qt
    import scipy.linalg
    import scipy.sparse
    import scipy.sparse.linalg
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix

    started = time.perf_counter()
    R = sp.Rational
    # QuTiP 5 normalises mesolve's states by default: the system loses
    # population into the sink, so that is switched off.
    opts = {"atol": 1e-13, "rtol": 1e-11, "nsteps": 10**8, "normalize_output": False}

    def levels(H, start, traps, loss, dephasing, extra=0):
        n = len(H)
        d = n + 2 + extra
        ket = [qt.basis(d, j) for j in range(d)]
        h = np.zeros((d, d), complex)
        h[:n, :n] = H
        c_ops = [math.sqrt(2 * k) * ket[n] * ket[t].dag() for t, k in traps.items()]
        c_ops += [math.sqrt(g) * ket[n + 1] * ket[j].dag() for j, g in enumerate(loss) if g > 0]
        c_ops += [math.sqrt(g) * ket[j] * ket[j].dag() for j, g in enumerate(dephasing) if g > 0]
        return qt.Qobj(h), c_ops, ket[start] * ket[start].dag(), ket, d

    def liouville(H, start, traps, loss, dephasing):
        """QuTiP's liouvillian, system block: eta, tau, the block (dense)."""
        h, c_ops, rho0, ket, d = levels(H, start, traps, loss, dephasing)
        n = len(H)
        full = qt.liouvillian(h, c_ops).to("csr").data.as_scipy().tocsr()
        idx = np.array([i + d * j for j in range(n) for i in range(n)])
        block = full[idx][:, idx].tocsc()
        lu = scipy.sparse.linalg.splu(block)
        b = np.zeros(n * n, complex)
        b[start + n * start] = 1
        x = lu.solve(-b)
        y = lu.solve(-x)
        # QuTiP's own sink row (a list index: SciPy 1.18 sparse arrays give
        # a 1-d row for a scalar index).
        sink = full[[n + d * n]][:, idx]
        eta = float((sink @ x)[0].real)
        return eta, float((sink @ y)[0].real) / eta, block.toarray(), b

    def horizon(block, b, remain=1e-12):
        n = int(round(math.sqrt(len(b))))
        tr = lambda v: float(np.trace(v.reshape((n, n), order="F")).real)  # noqa: E731
        left = lambda T: tr(scipy.linalg.expm(block * T) @ b)  # noqa: E731
        T = math.log(1 / remain) / float(-np.linalg.eigvals(block).real.max())
        while left(T) >= remain:
            T *= 1.1
        return T, left(T)

    def mesolve(H, start, traps, loss, dephasing, T):
        h, c_ops, rho0, ket, d = levels(H, start, traps, loss, dephasing, extra=1)
        n = len(H)
        L = qt.liouvillian(h, c_ops)
        acc = scipy.sparse.lil_matrix((d * d, d * d), dtype=complex)
        for t, k in traps.items():
            acc[(n + 2) * (d + 1), t * (d + 1)] = 2 * k  # d acc/dt = t · Σ 2κ ρ_tt
        gen = qt.QobjEvo([L, [qt.Qobj(acc.tocsr(), dims=L.dims), lambda t: t]])
        P = lambda j: ket[j] * ket[j].dag()  # noqa: E731
        stiff = max(list(dephasing) + [0.0]) * T > 1e5 or T > 2000
        res = qt.mesolve(gen, rho0, [0.0, T], e_ops=[P(n), P(n + 2)],
                         options={**opts, "method": "bdf" if stiff else "adams"})
        ps, a = (float(np.asarray(e)[-1].real) for e in res.expect)
        return ps, a / ps

    def exact_solve(M, b):
        A, B = DomainMatrix.from_Matrix(M).unify(DomainMatrix.from_Matrix(b))
        A = A.to_field()
        return A.lu_solve(B.convert_to(A.domain)).to_Matrix()

    def exact_full(H, start, traps, loss, dephasing):
        """SymPy on the full Liouvillian (rational parameters)."""
        S = sp.nsimplify
        H = sp.Matrix(H).applyfunc(S)
        traps = {t: S(k) for t, k in traps.items()}
        loss, dephasing = [S(v) for v in loss], [S(v) for v in dephasing]
        n = H.shape[0]
        a = [loss[j] / 2 + traps.get(j, 0) for j in range(n)]
        L = sp.zeros(n * n, n * n)
        for u in range(n):
            for v in range(n):
                r = u + n * v
                for w in range(n):
                    L[r, w + n * v] += -sp.I * H[u, w]
                    L[r, u + n * w] += sp.I * H[w, v]
                L[r, r] += -(a[u] + a[v]) - (0 if u == v else (dephasing[u] + dephasing[v]) / 2)
        b = sp.zeros(n * n, 1)
        b[start + n * start] = 1
        x = exact_solve(L, -b)
        y = exact_solve(L, -x)
        eta = sum(2 * k * x[t + n * t] for t, k in traps.items())
        return sp.nsimplify(eta), sp.nsimplify(sum(2 * k * y[t + n * t] for t, k in traps.items()) / eta)

    ORB = ["ss", "tt", "st", "ts", "sr", "rs", "tr", "rt", "rr", "rq"]

    def exact_orbitals(m, p):
        """K_N, N = m + 2, on the ten orbitals: (L X)_uv = −i(ε_u − ε_v)x_uv
        − iJ(S_col(v) − S_row(u)) − (a_u + a_v + [u≠v](γ_u + γ_v)/2) x_uv,
        S_col, S_row the column and row sums of X written in orbitals."""
        S = sp.nsimplify
        p = {key: ({c: S(v) for c, v in val.items()} if isinstance(val, dict) else S(val)) for key, val in p.items()}
        k = {o: i for i, o in enumerate(ORB)}

        def row(**c):
            r = [0] * 10
            for o, v in c.items():
                r[k[o]] += v
            return r

        col = {"s": row(ss=1, ts=1, rs=m), "t": row(st=1, tt=1, rt=m), "r": row(sr=1, tr=1, rr=1, rq=m - 1)}
        rw = {"s": row(ss=1, st=1, sr=m), "t": row(ts=1, tt=1, tr=m), "r": row(rs=1, rt=1, rr=1, rq=m - 1)}
        a = {c: p["Gamma"][c] / 2 + (p["kappa"] if c == "t" else 0) for c in "str"}
        M = sp.zeros(10, 10)
        for o in ORB:
            u, v, i = o[0], ("r" if o[1] == "q" else o[1]), k[o]
            diag = o in ("ss", "tt", "rr")
            if not diag and o != "rq":
                M[i, i] += -sp.I * (p["eps"][u] - p["eps"][v])
            for j in range(10):
                M[i, j] += -sp.I * p["J"] * (col[v][j] - rw[u][j])
            M[i, i] += -(a[u] + a[v]) - (0 if diag else (p["gamma"][u] + p["gamma"][v]) / 2)
        b = sp.zeros(10, 1)
        b[0] = 1
        x = exact_solve(M, -b)
        y = exact_solve(M, -x)
        eta = 2 * p["kappa"] * x[1]
        return sp.nsimplify(eta), sp.nsimplify(2 * p["kappa"] * y[1] / eta)

    def complete(N, p):
        cls = [node_class(u) for u in range(N)]
        H = float(p["J"]) * (np.ones((N, N)) - np.eye(N)) + np.diag([float(p["eps"][c]) for c in cls])
        loss = [float(p["Gamma"][c]) for c in cls]
        deph = [float(p["gamma"][c]) for c in cls]
        return H, 0, {1: float(p["kappa"])}, loss, deph

    diffs = {"exact": [0.0, 0.0], "liouvillian": [0.0, 0.0], "mesolve": [0.0, 0.0]}
    out: dict = {}

    def record(name, ref, eta, tau, got):
        de = abs(got["efficiency"] - eta)
        dt = abs(got["arrival_time"] / tau - 1)
        diffs[ref][0] = max(diffs[ref][0], de)
        diffs[ref][1] = max(diffs[ref][1], dt)
        out.setdefault(name, {})[ref] = {"efficiency": eta, "arrival_time": tau,
                                         "efficiency_difference": de, "relative_arrival_difference": dt}

    # 1. The detuned dimer (units of the hopping): exact, liouvillian, mesolve.
    dimer = np.array([[3.0, 1.0], [1.0, 0.0]])
    for g in ([0, 1, 100] if quick else [0, 1, 3, 10, 100]):
        eta_x, tau_x = exact_full(sp.Matrix([[3, 1], [1, 0]]), 0, {1: 1}, [R(1, 20)] * 2, [g, g])
        args = (dimer, 0, {1: 1.0}, [0.05, 0.05], [float(g)] * 2)
        got = qso_transport(*args)
        name = f"dimer, dephasing {g}"
        record(name, "exact", float(eta_x), float(tau_x), got)
        eta_l, tau_l, block, b = liouville(*args)
        record(name, "liouvillian", eta_l, tau_l, got)
        T, left = horizon(block, b)
        eta_m, tau_m = mesolve(*args, T)
        record(name, "mesolve", eta_m, tau_m, got)
        out[name].update({"exact": str(eta_x), "exact_arrival_time": str(tau_x), "horizon": T, "left": left})
    best = transport_walk(dimer, {1: 1.0}, [0.05, 0.05], [0.0, 0.0]).optimal_dephasing(0, 0.01, 100.0)
    out["dimer best rate"] = {"qso": best["rate"], "qso_efficiency": best["efficiency"], "exact": "39/20",
                              "efficiency_at_best": "800/963"}
    best_diff = abs(best["rate"] / 1.95 - 1)

    # 2. K_N on the ten orbitals (with loss: regular), against SymPy: QSO's
    # dense solve up to 64 nodes, its orbital solve at 64 (and 1024).
    params = {
        "uniform": {"J": -1, "kappa": 1, "eps": dict(s=0, t=0, r=0), "Gamma": dict(s=R(1, 20), t=R(1, 20), r=R(1, 20)),
                    "gamma": dict(s=1, t=1, r=1)},
        "classes": {"J": -1, "kappa": R(3, 2), "eps": dict(s=R(1, 2), t=-1, r=R(1, 4)),
                    "Gamma": dict(s=R(1, 10), t=R(1, 50), r=R(1, 20)), "gamma": dict(s=R(1, 2), t=2, r=1)},
    }
    reduction = 0.0
    for label, p in params.items():
        for N in ([4, 7, 64] if quick else [4, 5, 6, 7, 8, 9, 64, 1024]):
            eta_x, tau_x = exact_orbitals(N - 2, p)
            if N <= 6:  # the reduction against SymPy on the full Liouvillian
                cls = [node_class(u) for u in range(N)]
                Hs = sp.Matrix([[p["eps"][cls[u]] if u == v else p["J"] for v in range(N)] for u in range(N)])
                full = exact_full(Hs, 0, {1: p["kappa"]}, [p["Gamma"][c] for c in cls], [p["gamma"][c] for c in cls])
                reduction = max(reduction, float(abs(full[0] - eta_x)), float(abs(full[1] - tau_x)))
            if N >= 64:
                name = f"K{N} {label}, orbitals"
                record(name, "exact", float(eta_x), float(tau_x),
                       qso_transport(None, 0, {1: p["kappa"]}, None, None, kind="orbital", N=N, params=p))
                out[name]["exact_efficiency"] = str(eta_x)
            if N > 64:
                continue
            args = complete(N, p)
            got = qso_transport(*args)
            name = f"K{N} {label}"
            record(name, "exact", float(eta_x), float(tau_x), got)
            if N == 5:
                eta_l, tau_l, block, b = liouville(*args)
                record(name, "liouvillian", eta_l, tau_l, got)
                T, _ = horizon(block, b)
                eta_m, tau_m = mesolve(*args, T)
                record(name, "mesolve", eta_m, tau_m, got)
            out[name]["exact_efficiency"] = str(eta_x)

    # 3. K_N without loss: closed forms (Caruso et al.) and SymPy's bright
    # subspace: τ = (N² − 2N + 3)/(2(N − 1)) without dephasing (κ = 1),
    # N(N + 2)/4 with dephasing 1 everywhere; the rest is dark.
    closed = 0.0
    for N in ([5] if quick else range(4, 10)):
        H = -(np.ones((N, N)) - np.eye(N))
        for deph, eta, tau in [
            ([0.0] * N, 1 / (N - 1), (N * N - 2 * N + 3) / (2 * (N - 1))),
            ([0.0, 0.0, 1.0] + [0.0] * (N - 3), 1 / (N - 2), None),
            ([1.0] * N, 1.0, N * (N + 2) / 4),
        ]:
            got = qso_transport(H, 0, {1: 1.0}, [0.0] * N, deph)
            closed = max(closed, abs(got["efficiency"] - eta), abs(got["arrival_time"] / tau - 1) if tau else 0.0,
                         abs(got["dark"] - (1 - eta)))
    out["complete graphs without loss"] = {"max_difference_from_closed_forms": closed}
    diffs["exact"][0] = max(diffs["exact"][0], closed)

    # 4. Random weighted graphs with site energies (seeded).
    for seed in ([1, 2] if quick else range(1, 9)):
        rng = np.random.default_rng(seed)
        n = 5 + (seed - 1) // 2
        while True:  # connected
            A = np.triu((rng.random((n, n)) < 0.6).astype(float), 1)
            A = A + A.T
            if np.all(scipy.linalg.expm(A)[0] > 0):
                break
        W = np.triu(A * rng.uniform(0.3, 1.5, (n, n)), 1)
        H = -(W + W.T) + np.diag(rng.uniform(-1.0, 1.0, n))
        traps = {n - 1: float(rng.uniform(0.5, 2.0))}
        if seed % 2 == 0:
            traps[n - 2] = float(rng.uniform(0.2, 1.0))
        loss = list(rng.uniform(0.0, 0.1, n))
        deph = list(rng.uniform(0.0, 2.0, n) * (rng.random(n) < 0.7))
        args = (H, 0, traps, loss, deph)
        got = qso_transport(*args)
        eta_l, tau_l, block, b = liouville(*args)
        name = f"random graph {seed} ({n} nodes)"
        record(name, "liouvillian", eta_l, tau_l, got)
        T, _ = horizon(block, b)
        eta_m, tau_m = mesolve(*args, T)
        record(name, "mesolve", eta_m, tau_m, got)

    # 5. FMO (Adolphs & Renger 2006, C. tepidum, as tabulated by Sarovar et
    # al. 2010 and Kreisbeck et al. 2011), ps: trap BChl 3 (κ = 1/ps),
    # loss 1/ns, uniform dephasing; start BChl 1. A tutorial input.
    fmo = np.array([
        [12410.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9], [-87.7, 12530.0, 30.8, 8.2, 0.7, 11.8, 4.3],
        [5.5, 30.8, 12210.0, -53.5, -2.2, -9.6, 6.0], [-5.9, 8.2, -53.5, 12320.0, -70.7, -17.0, -63.3],
        [6.7, 0.7, -2.2, -70.7, 12480.0, 81.1, -1.3], [-13.7, 11.8, -9.6, -17.0, 81.1, 12630.0, 39.7],
        [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440.0]])
    cm = 2 * math.pi * 2.99792458e-2  # 1/cm in rad/ps
    H = (fmo - np.trace(fmo) / 7 * np.eye(7)) * cm
    for g in ([10.0] if quick else [0.0, 1.0, 10.0, 300 * cm, 100.0]):
        args = (H, 0, {2: 1.0}, [1e-3] * 7, [g] * 7)
        got = qso_transport(*args)
        eta_l, tau_l, block, b = liouville(*args)
        name = f"FMO, dephasing {g:.4g}/ps"
        record(name, "liouvillian", eta_l, tau_l, got)
        out[name]["condition"] = got["condition"]
        if g > 0:  # without dephasing the horizon is 2.6·10⁴ ps: no mesolve
            T, _ = horizon(block, b)
            eta_m, tau_m = mesolve(*args, T)
            record(name, "mesolve", eta_m, tau_m, got)
            out[name]["horizon_ps"] = T

    # 6. Hypercube, trap at the antipode, no dephasing: QSO's solve on the
    # orbitals against QuTiP on the full cube (energies, losses by weight).
    for d in ([3] if quick else [3, 4, 5]):
        n = 1 << d
        A = np.zeros((n, n))
        for u in range(n):
            for k in range(d):
                A[u, u ^ (1 << k)] = 1.0
        w = np.array([bin(u).count("1") for u in range(n)])
        eps = 0.3 * np.arange(d + 1) - 0.1 * np.arange(d + 1) ** 2
        loss = 0.02 + 0.01 * np.arange(d + 1)
        args = (-A + np.diag(eps[w]), 0, {n - 1: 1.0}, list(loss[w]), [0.0] * n)
        eta_l, tau_l, _, _ = liouville(*args)
        record(f"hypercube Q{d}", "liouvillian", eta_l, tau_l,
               qso_transport(*args, kind="hypercube", d=d, energies=list(eps), losses=list(loss)))

    return {
        **out,
        "max_efficiency_difference_exact": diffs["exact"][0],
        "max_relative_arrival_difference_exact": diffs["exact"][1],
        "max_efficiency_difference_liouvillian": diffs["liouvillian"][0],
        "max_relative_arrival_difference_liouvillian": diffs["liouvillian"][1],
        "max_efficiency_difference_mesolve": diffs["mesolve"][0],
        "max_relative_arrival_difference_mesolve": diffs["mesolve"][1],
        "orbital_reduction_difference_exact": reduction,
        "best_rate_relative_difference": best_diff,
        "seconds": time.perf_counter() - started,
    }


def transport_in_child(quick: bool) -> dict:
    """transport_vs_qutip in a child process pinned to one thread (pinning
    this process would skew the other frameworks' timings)."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    cmd = [sys.executable, str(Path(__file__)), "--transport-only"] + (["--quick"] if quick else [])
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[1:])} failed:\n{res.stderr}")
    return json.loads(res.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------------
# Pauli propagation (1.64): Stim, Qiskit Aer and qiskit-addon-obp on the
# kicked Ising circuits of Kim et al. on heavy_hex(7) (frameworks.py).

# Pauli–Lindblad layers scaled by these factors G (in order) after every RZZ
# layer: noiseless (0), amplified, inverted (PEC's limit), and amplified then
# inverted, which must give the noiseless value.
PROPAGATION_SCALES = ([0.0], [1.0], [1.5], [2.0], [-1.0], [1.0, -1.0])


def kim_noise_model(n: int, colours: list, kind: str) -> list:
    """Sparse Pauli–Lindblad noise after the RZZ layer of each colour, as a
    list per colour of (qubits, Qiskit labels, rates). "uniform": 10⁻³ on
    X, Y and Z of every qubit, 2·10⁻⁴ on XX and 5·10⁻⁴ on ZZ of every edge
    of the layer; "seeded": X, Y and Z on every qubit (rates uniform in
    [10⁻⁴, 2·10⁻³]) and the nine weight-2 Paulis on every edge of the layer
    (uniform in [10⁻⁴, 10⁻³]), seed 1640. A Qiskit label's last letter
    acts on the first qubit."""
    rng = np.random.default_rng(1640)
    one = ["X", "Y", "Z"]
    two = [a + b for a in "XYZ" for b in "XYZ"]
    model = []
    for edges in colours:
        if kind == "uniform":
            terms = [((q,), one, np.full(3, 1e-3)) for q in range(n)]
            terms += [(e, ["XX", "ZZ"], np.array([2e-4, 5e-4])) for e in edges]
        else:
            terms = [((q,), one, rng.uniform(1e-4, 2e-3, 3)) for q in range(n)]
            terms += [(e, two, rng.uniform(1e-4, 1e-3, 9)) for e in edges]
        model.append(terms)
    return model


def kim_timeline(n: int, colours: list, steps: int, theta_h: float, scales=()) -> list:
    """The kicked Ising circuit in time order: ("gates", operations) per
    layer and, after the RZZ layer of colour k, ("noise", k, G) for each G
    in `scales`."""
    out = []
    for i, ops in enumerate(fw.kicked_ising_layers(n, colours, steps, theta_h)):
        out.append(("gates", ops))
        k = i % (1 + len(colours)) - 1
        if k >= 0:
            out += [("noise", k, G) for G in scales]
    return out


def qso_kicked_ising(n: int, timeline: list, model=None, **options) -> tuple[dict, dict, float]:
    """⟨Z_{n//2}⟩ by metatron_qso.propagate (OpenQASM per gate layer,
    (rates, factor) per noise layer): the result, the Heisenberg operator
    as {label: coefficient} and the seconds of the call."""
    import metatron_qso as q

    items = []
    for item in timeline:
        if item[0] == "gates":
            items.append(fw.qasm2(n, item[1]))
            continue
        _, k, G = item
        rates = [
            (float(r), fw.pauli_label((qb, lab[len(qs) - 1 - pos]) for pos, qb in enumerate(qs)))
            for qs, labels, rr in model[k]
            for lab, r in zip(labels, rr)
        ]
        items.append((rates, float(G)))
    t = time.perf_counter()
    r = q.propagate(items, f"Z{n // 2}", n_qubits=n, return_operator=True, **options)
    seconds = time.perf_counter() - t
    return r, fw.qso_operator_dict(r["operator"]), seconds


def obp_kicked_ising(n: int, timeline: list, model=None) -> tuple[dict, float]:
    """The Heisenberg operator of Z_{n//2} by qiskit-addon-obp's
    backpropagate, one slice per layer, with Pauli–Lindblad layers as
    PauliLindbladErrorInstruction. obp splits every gate with
    SparsePauliOp.from_operator, which drops components below
    max(SparsePauliOp.atol, rtol) = 10⁻⁵ whatever the operator budget, so
    both are lowered to 10⁻¹⁶ for the call; the budget trims |c| ≤ 10⁻¹⁵
    after every gate (with no trimming, rounding residues of 10⁻¹⁷ are
    never removed and the 5-step operator grows towards 6·10⁷ strings;
    trimming at 10⁻¹⁴ loses 14 of the 134 744 strings at θ_h = 0.3 and 5
    steps and moves others by up to 1.3·10⁻¹³ against a long-double
    propagation, which QSO matches to 1.6·10⁻¹⁶)."""
    import warnings

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import PauliList, SparsePauliOp

    with warnings.catch_warnings():
        # obp 0.3 imports PauliLindbladError from its pre-0.48 location.
        warnings.simplefilter("ignore", DeprecationWarning)
        from qiskit_addon_obp import backpropagate
        from qiskit_addon_obp.utils.noise import PauliLindbladErrorInstruction
        from qiskit_addon_obp.utils.simplify import OperatorBudget
        from qiskit_ibm_runtime.results import PauliLindbladError

    slices = []
    for item in timeline:
        if item[0] == "gates":
            slices.append(fw.qiskit_circuit(n, item[1], save=False))
            continue
        _, k, G = item
        qc = QuantumCircuit(n)
        for qs, labels, rates in model[k]:
            qc.append(PauliLindbladErrorInstruction(PauliLindbladError(PauliList(labels), G * rates)), list(qs))
        slices.append(qc)
    old = SparsePauliOp.atol, SparsePauliOp.rtol
    SparsePauliOp.atol = SparsePauliOp.rtol = 1e-16
    try:
        obs = SparsePauliOp.from_sparse_list([("Z", [n // 2], 1.0)], n)
        t = time.perf_counter()
        out, rest, _ = backpropagate(obs, slices, operator_budget=OperatorBudget(atol=1e-15))
        seconds = time.perf_counter() - t
    finally:
        SparsePauliOp.atol, SparsePauliOp.rtol = old
    assert len(rest) == 0, "qiskit-addon-obp left slices unabsorbed"
    return fw.qiskit_operator_dict(out), seconds


def aer_kicked_ising(n: int, timeline: list, model=None) -> tuple[float, int, float]:
    """⟨Z_{n//2}⟩ by Qiskit Aer on the light cone (fw.reduce_to_light_cone;
    Pauli–Lindblad generators restricted to it, exact because their parts
    outside commute with every string inside): a state vector without
    noise, else a density matrix with qiskit_aer.noise.PauliLindbladError
    (which refuses negative rates). Returns the value, the qubits of the
    cone and the seconds of the run."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import PauliLindbladError

    gates = [op for item in timeline if item[0] == "gates" for op in item[1]]
    kept, support = fw.reduce_to_light_cone(gates, n // 2)
    kept = set(kept)
    idx = {q: i for i, q in enumerate(support)}
    qc = QuantumCircuit(len(support))
    g = 0
    noisy = False
    for item in timeline:
        if item[0] == "gates":
            for name, qs, ps in item[1]:
                if g in kept:
                    getattr(qc, name)(*ps, *[idx[q] for q in qs])
                g += 1
            continue
        _, k, G = item
        noisy = True
        for qs, labels, rates in model[k]:
            inside = [p for p, q in enumerate(qs) if q in idx]
            if not inside:
                continue
            gens = [Pauli("".join(lab[len(qs) - 1 - p] for p in reversed(inside))) for lab in labels]
            qc.append(PauliLindbladError(gens, G * rates), [idx[qs[p]] for p in inside])
    qc.save_expectation_value(Pauli("Z"), [idx[n // 2]])
    sim = AerSimulator(method="density_matrix" if noisy else "statevector")
    tqc = transpile(qc, sim, optimization_level=0)
    t = time.perf_counter()
    r = sim.run(tqc).result()
    seconds = time.perf_counter() - t
    return float(np.real(r.data()["expectation_value"])), len(support), seconds


def stim_kicked_ising(n: int, colours: list, steps: int, theta_h: float):
    """The kicked Ising circuit at θ_h = 0 or π/2 as a Stim circuit
    (RX(π/2) = SQRT_X and RZZ(−π/2) = SQRT_ZZ_DAG up to global phases)."""
    import stim

    c = stim.Circuit()
    for _ in range(steps):
        if theta_h != 0.0:
            assert theta_h == np.pi / 2
            c.append("SQRT_X", list(range(n)))
        for edges in colours:
            c.append("SQRT_ZZ_DAG", [q for e in edges for q in e])
    return c


def zero_state_value(d: dict) -> float:
    """⟨0…0| O |0…0⟩ of {label: coefficient}: the strings of I and Z."""
    return sum(c for k, c in d.items() if not any(t[0] in "XY" for t in k.split()))


def propagation_clifford_vs_stim(quick: bool) -> dict:
    """Seeded Clifford brickwork circuits on 10³ and 10⁴ qubits: single Z
    and random dense Pauli strings through QSO and Stim's
    PauliString.before (C†PC), string and sign."""
    import stim

    import metatron_qso as q

    rng = np.random.default_rng(1640)
    one = ["H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "X", "Y", "Z"]
    two = ["CX", "CZ", "CY", "SWAP", "ISWAP"]
    out = {}
    for n, layers in ((1000, 20), (10_000, 10)):
        circuit = stim.Circuit()
        ops = []
        for layer in range(layers):
            kinds = rng.integers(len(one), size=n)
            for k, g in enumerate(one):
                qs = np.flatnonzero(kinds == k).tolist()
                if qs:
                    circuit.append(g, qs)
                    ops += [(g, (qb,)) for qb in qs]
            pairs = {}
            for a in range(layer % 2, n - 1, 2):
                g = two[rng.integers(len(two))]
                qs = (a, a + 1) if rng.random() < 0.5 else (a + 1, a)
                pairs.setdefault(g, []).extend(qs)
                ops.append((g, qs))
            for g, targets in pairs.items():
                circuit.append(g, targets)
        program = q.Program.from_qasm(fw.qasm2(n, [(fw.STIM_TO_QASM[g], qs, ()) for g, qs in ops], fw.QASM_ISWAP))
        worst, unmatched, strings, seconds = 0.0, 0, 0, 0.0
        for k in range(10 if quick else 50):
            if k % 5 == 0:
                p = stim.PauliString(n)
                p[int(rng.integers(n))] = "Z"
            else:
                # stim.PauliString.random draws from Stim's unseeded generator.
                p = stim.PauliString("".join(rng.choice(list("_XYZ"), n)))
            observable = fw.pauli_label((i, ch) for i, ch in enumerate(str(p)[1:]) if ch != "_")
            t = time.perf_counter()
            r = q.propagate(program, observable, n_qubits=n, return_operator=True)
            seconds += time.perf_counter() - t
            c = fw.compare_operators(fw.qso_operator_dict(r["operator"]), fw.stim_operator_dict(p.before(circuit)), 0.0)
            worst = max(worst, c["max_coefficient_difference"])
            unmatched += c["only_qso"] + c["only_reference"]
            strings += 1
        out[f"n={n}"] = {
            "gates": len(ops),
            "strings": strings,
            "max_coefficient_difference": worst,
            "unmatched_strings": unmatched,
            "seconds_qso": seconds,
        }
    return out


def pauli_propagation_vs_references(quick: bool) -> dict:
    """Heisenberg propagation (1.64) of ⟨Z_57⟩ on the 115-qubit kicked Ising
    circuits of Kim et al. on heavy_hex(7) (qubit 57 = n // 2, whose light
    cones have 1, 3, 5, 9, 13, 20 and 26 qubits after 1–7 steps), without
    truncation unless stated: Clifford circuits on 10³ and 10⁴ qubits and
    the Clifford points θ_h = 0 and π/2 against Stim (strings and signs);
    θ_h ∈ {0.3, π/4, 1} against Qiskit Aer's state vector of the light
    cone; the Heisenberg operator string for string against
    qiskit-addon-obp 0.3; truncated runs at 6 and 7 steps against Aer
    within the reported bound (not with --quick); and scaled sparse
    Pauli–Lindblad noise against obp and Aer's density matrix."""
    import stim

    started = time.perf_counter()
    out = {"clifford_vs_stim": propagation_clifford_vs_stim(quick)}
    n, colours = fw.heavy_hex_lattice(7)

    # Clifford points against Stim on all 115 qubits.
    points = {}
    for theta, expect in ((0.0, 1.0), (np.pi / 2, 0.0)):
        rows = []
        for steps in range(1, 11 if quick else 21):
            r, d, seconds = qso_kicked_ising(n, kim_timeline(n, colours, steps, theta))
            p = stim_kicked_ising(n, colours, steps, theta)
            ref = fw.stim_operator_dict(stim.PauliString("_" * (n // 2) + "Z" + "_" * (n - n // 2 - 1)).before(p))
            rows.append({
                "steps": steps,
                "value": r["value"],
                "value_difference": abs(r["value"] - expect),
                "strings": r["terms"],
                "weight": max((len(k.split()) for k in d), default=0),
                "truncation_bound": r["truncation_bound"],
                "seconds_qso": seconds,
                **fw.compare_operators(d, ref, 0.0),
            })
        points[f"theta_h={theta:.6f}"] = rows
    out["kicked_ising_clifford_points"] = points

    # Light cones against Aer's state vector; operators against obp.
    light, aer_value = [], {}
    for theta in (0.3, np.pi / 4, 1.0):
        for steps in range(1, 5 if quick else 6):
            timeline = kim_timeline(n, colours, steps, theta)
            ref, cone, aer_seconds = aer_kicked_ising(n, timeline)
            r, _, seconds = qso_kicked_ising(n, timeline)
            aer_value[(theta, steps)] = ref
            light.append({
                "theta_h": theta, "steps": steps, "light_cone": cone, "strings": r["terms"],
                "peak_strings": r["peak_terms"], "aer": ref, "qso": r["value"],
                "difference": abs(r["value"] - ref), "truncation_bound": r["truncation_bound"],
                "seconds_qso": seconds, "seconds_aer": aer_seconds,
            })
    out["kicked_ising_vs_aer_light_cone"] = light
    strings = []
    for theta in (0.3, 1.0):
        for steps in (3, 4) if quick else (3, 4, 5):
            timeline = kim_timeline(n, colours, steps, theta)
            ref, obp_seconds = obp_kicked_ising(n, timeline)
            r, d, seconds = qso_kicked_ising(n, timeline)
            row = {"theta_h": theta, "steps": steps}
            row.update(fw.compare_operators(d, ref, TOLERANCES["propagation_coefficient"]))
            row.update({"value_difference": abs(r["value"] - zero_state_value(ref)),
                        "seconds_qso": seconds, "seconds_obp": obp_seconds})
            strings.append(row)
    out["kicked_ising_vs_obp"] = strings

    # Truncated runs beyond the exact reach of obp: error within the bound.
    truncated = []
    if not quick:
        theta = 0.3
        for steps in (6, 7):
            timeline = kim_timeline(n, colours, steps, theta)
            ref, cone, aer_seconds = aer_kicked_ising(n, timeline)
            for tau in (1e-3, 1e-4):
                r, _, seconds = qso_kicked_ising(n, timeline, min_coefficient=tau)
                truncated.append({
                    "theta_h": theta, "steps": steps, "light_cone": cone, "min_coefficient": tau,
                    "aer": ref, "qso": r["value"], "error": abs(r["value"] - ref),
                    "truncation_bound": r["truncation_bound"], "strings": r["terms"],
                    "peak_strings": r["peak_terms"], "dropped_strings": r["dropped_terms"],
                    "seconds_qso": seconds, "seconds_aer": aer_seconds,
                })
    out["kicked_ising_truncated"] = truncated

    # Scaled sparse Pauli–Lindblad noise after every RZZ layer.
    noisy = []
    cases = [(1.0, 3)] + ([] if quick else [(np.pi / 4, 4)])
    for kind in ("uniform", "seeded"):
        model = kim_noise_model(n, colours, kind)
        for theta, steps in cases:
            noiseless = aer_value[(theta, steps)]
            for scales in PROPAGATION_SCALES:
                timeline = kim_timeline(n, colours, steps, theta, scales)
                r, d, seconds = qso_kicked_ising(n, timeline, model)
                ref, obp_seconds = obp_kicked_ising(n, timeline, model)
                row = {"model": kind, "theta_h": theta, "steps": steps, "scales": list(scales),
                       "qso": r["value"], "obp": zero_state_value(ref),
                       "obp_difference": abs(r["value"] - zero_state_value(ref))}
                row.update(fw.compare_operators(d, ref, TOLERANCES["propagation_coefficient"]))
                if min(scales) >= 0:
                    aer, _, aer_seconds = aer_kicked_ising(n, timeline, model)
                    row.update({"aer": aer, "aer_difference": abs(r["value"] - aer), "seconds_aer": aer_seconds})
                if sum(scales) == 0.0:
                    # Noise amplified and inverted (or scaled by 0) is no noise.
                    row["noiseless_difference"] = abs(r["value"] - noiseless)
                row.update({"truncation_bound": r["truncation_bound"], "seconds_qso": seconds, "seconds_obp": obp_seconds})
                noisy.append(row)
    out["kicked_ising_noisy"] = noisy
    out["seconds"] = time.perf_counter() - started
    return out


def pauli_propagation_failures(p: dict) -> list[str]:
    """The deviations of pauli_propagation_vs_references beyond
    TOLERANCES."""
    failures = []
    for k, r in p["clifford_vs_stim"].items():
        if r["unmatched_strings"] or r["max_coefficient_difference"] != 0.0:
            failures.append(f"Clifford propagation at {k} differs from Stim's: {r}")
    for k, rows in p["kicked_ising_clifford_points"].items():
        for r in rows:
            if (r["value_difference"] != 0.0 or r["strings"] != 1 or r["only_qso"] or r["only_reference"]
                    or r["max_coefficient_difference"] != 0.0 or not r["truncation_bound"] <= 1e-13):
                failures.append(f"kicked Ising at the Clifford point {k} differs from Stim's: {r}")
    for r in p["kicked_ising_vs_aer_light_cone"]:
        if not r["difference"] <= TOLERANCES["propagation_value"]:
            failures.append(f"kicked Ising differs from Qiskit Aer's light cone: {r}")
    for r in p["kicked_ising_vs_obp"]:
        if (r["only_qso"] or r["only_reference"] or not r["max_coefficient_difference"] <= TOLERANCES["propagation_coefficient"]
                or not r["value_difference"] <= TOLERANCES["propagation_value"]):
            failures.append(f"kicked Ising operator differs from qiskit-addon-obp's: {r}")
    for r in p["kicked_ising_truncated"]:
        # The bound excludes rounding, which the reference has too.
        if not r["error"] <= r["truncation_bound"] + TOLERANCES["propagation_value"]:
            failures.append(f"truncated kicked Ising exceeds its bound: {r}")
    for r in p["kicked_ising_noisy"]:
        worst = max(r["obp_difference"], r.get("aer_difference", 0.0), r.get("noiseless_difference", 0.0))
        if (not worst <= TOLERANCES["propagation_noisy"] or r["only_qso"] or r["only_reference"]
                or not r["max_coefficient_difference"] <= TOLERANCES["propagation_coefficient"]):
            failures.append(f"noisy kicked Ising differs from qiskit-addon-obp or Qiskit Aer: {r}")
    return failures


def check(c: dict) -> list[str]:
    """Deviations in the correctness section that exceed TOLERANCES."""
    failures = []
    for name, d in c["max_amplitude_difference_vs_qso"].items():
        if not d <= TOLERANCES["state"]:
            failures.append(f"final states differ from {name} by {d:.3g}")
    for name, g in c["gradients_vs_qso"].items():
        if not max(g["value_diff"], g["max_gradient_diff"]) <= TOLERANCES["gradient"]:
            failures.append(f"energies or gradients differ from {name}: {g}")
    if not c["permanent_max_relative_difference_vs_thewalrus"] <= TOLERANCES["permanent"]:
        failures.append(f"permanents differ from The Walrus by {c['permanent_max_relative_difference_vs_thewalrus']:.3g}")
    g = c["gaussian_vs_thewalrus"]
    for key in (
        "hafnian_max_relative_difference",
        "repeated_hafnian_max_relative_difference",
        "torontonian_relative_difference",
        "gbs_probability_max_relative_difference",
    ):
        if not g[key] <= TOLERANCES["hafnian"]:
            failures.append(f"{key} against The Walrus is {g[key]:.3g}")
    for key in ("covariance_max_difference", "gbs_probability_max_difference", "click_probability_max_difference"):
        if not g[key] <= TOLERANCES["gaussian_probability"]:
            failures.append(f"{key} against The Walrus is {g[key]:.3g}")
    if not c["single_precision_infidelity_vs_double"]["qso"] <= TOLERANCES["single_precision"]:
        failures.append(f"single-precision states differ: {c['single_precision_infidelity_vs_double']}")
    d = c["noisy_circuits"]["max_density_matrix_difference_vs_qiskit_aer"]
    if not d <= TOLERANCES["density_matrix"]:
        failures.append(f"noisy density matrices differ from Qiskit Aer by {d:.3g}")
    for name, r in c["qec_vs_stim"].items():
        if name == "decoding_speed":
            continue
        if not r["sparse_matching_max_relative_weight_difference"] <= TOLERANCES["matching_weight"]:
            failures.append(f"sparse matching weights on {name} differ from PyMatching's: {r}")
        if not r["sparse_matching_disagreement_with_pymatching"] <= TOLERANCES["matching_disagreement"]:
            failures.append(f"sparse matching predictions on {name} differ from PyMatching's: {r}")
        if r["error_mechanisms_only_one_side"] or not r["max_relative_probability_difference"] <= TOLERANCES["error_model"]:
            failures.append(f"detector error model of {name} differs from Stim's: {r}")
        if not r["max_detector_rate_z_score"] <= TOLERANCES["detector_rate_z"]:
            failures.append(f"detection-event rates of {name} differ from Stim's: {r}")
        if not r["matching_disagreement_with_pymatching"] <= TOLERANCES["matching_disagreement"]:
            failures.append(f"matching predictions on {name} differ from PyMatching's: {r}")
    st = c["bposd_vs_ldpc"]["steane_shared_qubit"]
    if st["qso"] != st["ldpc"] or st["qso_converged"] != st["ldpc_converged"]:
        failures.append(f"BP+OSD on the Steane code's shared qubit differs from ldpc's: {st}")
    ex = c["bposd_vs_ldpc"]["exhaustive_random_models"]
    if not ex["correction_disagreement"] <= TOLERANCES["bposd_min_sum_disagreement"]:
        failures.append(f"exhaustive OSD differs from ldpc's: {ex}")
    for name, r in c["bposd_vs_ldpc"]["capacity"].items():
        key = "bposd_product_sum_disagreement" if "product-sum" in name else "bposd_min_sum_disagreement"
        if not max(r["correction_disagreement"], r["convergence_disagreement"]) <= TOLERANCES[key]:
            failures.append(f"BP+OSD corrections on {name} differ from ldpc's: {r}")
    for name, r in c["bposd_vs_ldpc"]["circuit"].items():
        if not r["max_relative_probability_difference_vs_stim"] <= TOLERANCES["error_model"]:
            failures.append(f"detector error model of {name} differs from Stim's: {r}")
        if not r["prediction_disagreement"] <= TOLERANCES["bposd_circuit_disagreement"]:
            failures.append(f"BP+OSD predictions on {name} differ from ldpc's: {r}")
    for name, r in c["correlated_and_distance_vs_stim"].items():
        lower, upper = r["circuit_distance_qso"]
        if lower != upper or upper > r["searched_distance_stim"]:
            failures.append(f"circuit distance of {name} is not exact or exceeds Stim's search: {r}")
        if r["exact_distance_qso"] != [r["exhaustive_distance_stim"]] * 2:
            failures.append(f"exact circuit distance of {name} differs from Stim's exhaustive search: {r}")
        if r["exact_distance_qso_symmetric"] != [r["exhaustive_distance_stim"]] * 2:
            failures.append(f"circuit distance of {name} on the orbits differs from Stim's exhaustive search: {r}")
        if "graphlike_distance_stim" not in r:
            continue
        if r["graphlike_distance_qso"] != r["graphlike_distance_stim"]:
            failures.append(f"graph-like distance of {name} differs from Stim's: {r}")
        # Stim decomposes a Y error that flips more than two detectors
        # against the whole model's edges, QSO within the channel: both
        # valid, not always the same.
        if "Y errors" not in name and (r["decomposed_errors_only_one_side"] or not r["decomposed_max_relative_probability_difference"] <= TOLERANCES["error_model"]):
            failures.append(f"decomposed error model of {name} differs from Stim's: {r}")
        if not r["correlated_disagreement_with_pymatching"] <= TOLERANCES["correlated_disagreement"]:
            failures.append(f"two-pass matching on {name} differs from PyMatching's: {r}")
    for n, rows in c.get("qaoa_vs_qso", {}).items():
        for lib, r in rows.items():
            tol = TOLERANCES["qaoa_single" if lib.endswith("single") else "qaoa_double"]
            diff = max(r["value_rel_diff"], r.get("gradient_rel_diff", 0.0))
            if not diff <= tol:
                failures.append(f"QAOA energy or gradient of {lib} at {n} qubits differs from QSO's by {diff:.3g}")
    sp = c["spins_vs_qutip"]
    if not sp["max_relative_expectation_difference"] <= TOLERANCES["spin_expectation"]:
        failures.append(f"spin dynamics differ from QuTiP by {sp['max_relative_expectation_difference']:.3g} (relative)")
    if not sp["max_relative_expectation_difference_mesolve"] <= TOLERANCES["spin_mesolve"]:
        failures.append(f"spin dynamics differ from QuTiP's mesolve by {sp['max_relative_expectation_difference_mesolve']:.3g}")
    rf = c["redfield_vs_qutip"]
    if not rf["max_relative_rate_difference"] <= TOLERANCES["redfield_rate"]:
        failures.append(f"Redfield rates differ from QuTiP's Bloch–Redfield tensor by {rf['max_relative_rate_difference']:.3g}")
    if not rf["max_expectation_difference"] <= TOLERANCES["redfield_expectation"]:
        failures.append(f"Redfield dynamics differ from QuTiP's by {rf['max_expectation_difference']:.3g}")
    if not sp["max_hamiltonian_relative_difference"] <= TOLERANCES["spin_hamiltonian"]:
        failures.append(f"spin Hamiltonians differ from QuTiP's by {sp['max_hamiltonian_relative_difference']:.3g}")
    bath = c["bath_vs_pycce"]
    if not bath["max_coherence_difference"] <= TOLERANCES["bath_coherence"]:
        failures.append(f"bath coherences differ from PyCCE by {bath['max_coherence_difference']:.3g}")
    general = c["generalised_bath_vs_pycce"]
    if not general["max_coherence_difference"] <= TOLERANCES["generalised_bath_coherence"]:
        failures.append(
            f"generalised bath coherences differ from PyCCE by {general['max_coherence_difference']:.3g}"
        )
    chem = c["chemistry_vs_pyscf"]
    if not chem["integral_max_difference"] <= TOLERANCES["integral"]:
        failures.append(f"integrals differ from PySCF by {chem['integral_max_difference']:.3g}")
    if not chem["energy_max_difference"] <= TOLERANCES["scf_energy"]:
        failures.append(f"Hartree–Fock or CASCI energies differ from PySCF by {chem['energy_max_difference']:.3g}")
    if not chem["correlation_max_difference"] <= TOLERANCES["correlation_energy"]:
        failures.append(f"MP2, CCSD or (T) energies differ from PySCF by {chem['correlation_max_difference']:.3g}")
    tp = c["transport_vs_qutip"]
    for ref, label in (("exact", "SymPy's exact rationals and closed forms"), ("liouvillian", "QuTiP's liouvillian")):
        if not tp[f"max_efficiency_difference_{ref}"] <= TOLERANCES["transport_efficiency"]:
            failures.append(f"transport efficiencies differ from {label} by {tp[f'max_efficiency_difference_{ref}']:.3g}")
        if not tp[f"max_relative_arrival_difference_{ref}"] <= TOLERANCES["transport_arrival"]:
            failures.append(
                f"mean arrival times differ from {label} by {tp[f'max_relative_arrival_difference_{ref}']:.3g} (relative)"
            )
    worst = max(tp["max_efficiency_difference_mesolve"], tp["max_relative_arrival_difference_mesolve"])
    if not worst <= TOLERANCES["transport_mesolve"]:
        failures.append(f"transport differs from QuTiP's mesolve by {worst:.3g}")
    if tp["orbital_reduction_difference_exact"] != 0.0:
        failures.append("the ten-orbital SymPy reference differs from SymPy on the full Liouvillian")
    if not tp["best_rate_relative_difference"] <= TOLERANCES["transport_best_rate"]:
        failures.append(
            f"the dimer's rate of highest efficiency differs from 39/20 by {tp['best_rate_relative_difference']:.3g} (relative)"
        )
    failures += pauli_propagation_failures(c["pauli_propagation"])
    return failures


# ----------------------------------------------------------------------------
# QAOA at 22–26 qubits: p = 10 on a random 3-regular graph, energy and
# gradient of every framework in its own process (time and peak memory).

QAOA_LIBS = ["qso", "qso-single", "qso-circuit", "qso-circuit-single", "qulacs", "aer"]


def qaoa_problem(n: int, p: int = 10):
    import metatron_qso as q

    g = q.Graph.random_regular(n, 3, seed=1)
    ops, n_params = fw.qaoa_ops(n, g.edges, p)
    theta = 0.1 + 0.01 * np.arange(n_params)
    return g, ops, n_params, theta


def qaoa_worker(lib: str, n: int) -> dict:
    """One energy and gradient (Aer: energy only) of the QAOA problem at
    n qubits, after one untimed call below 24 qubits."""
    import resource

    g, ops, n_params, theta = qaoa_problem(n)
    terms, constant = fw.maxcut_terms(g.edges)
    if lib in ("qso", "qso-single"):
        import metatron_qso as q

        qaoa = q.MaxCutQaoa(g, n_params // 2, precision="single" if lib == "qso-single" else "double")
        run = lambda: qaoa.expectation_and_gradient(theta)  # noqa: E731
    elif lib in ("qso-circuit", "qso-circuit-single"):
        import metatron_qso as q

        c, h = fw.qso_qaoa_circuit(n, ops, n_params), q.maxcut_cost(g)
        precision = "single" if lib.endswith("single") else "double"
        run = lambda: q.expectation_and_gradient(c, h, theta, precision=precision)  # noqa: E731
    elif lib == "qulacs":
        run = lambda f=fw.qulacs_qaoa_value_and_grad_fn(n, ops, terms, constant, n_params): f(theta)  # noqa: E731
    else:
        f = fw.aer_qaoa_energy_fn(n, ops, terms, constant)
        run = lambda: (f(theta), None)  # noqa: E731
    if n < 24:
        run()
    start = time.perf_counter()
    value, grad = run()
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB → GiB
    return {
        "value": float(value),
        "gradient": None if grad is None else [float(x) for x in grad],
        "seconds": seconds,
        "peak_gib": peak,
    }


def qaoa_large(sizes: list[int]) -> dict:
    out = {}
    for n in sizes:
        rows = {}
        for lib in QAOA_LIBS:
            cmd = [sys.executable, str(Path(__file__)), "--qaoa-worker", lib, str(n)]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            rows[lib] = json.loads(res.stdout.strip().splitlines()[-1])
        ref = rows["qso"]
        scale = max(abs(ref["value"]), 1.0)
        for lib, r in rows.items():
            r["value_rel_diff"] = abs(r["value"] - ref["value"]) / scale
            if r["gradient"] is not None:
                r["gradient_rel_diff"] = float(np.max(np.abs(np.subtract(r["gradient"], ref["gradient"])))) / scale
        out[str(n)] = rows
        print(f"qaoa {n}: " + ", ".join(f"{k} {v['seconds']:.2f} s" for k, v in rows.items()), file=sys.stderr)
    return out


# ----------------------------------------------------------------------------
# Interleaved timing


def configurations(quick: bool) -> list[tuple]:
    """(kind, n, libraries, reps per round, warm-up)."""
    # Single precision: QSO's StateVector32 and Aer's single-precision
    # state vector (Qulacs is double precision only; PennyLane Lightning
    # and Cirq run in double precision here).
    state_libs = ["qso", "qulacs", "aer", "pennylane", "qso-single", "aer-single"]
    sizes = [16, 20] if quick else [16, 20, 22, 24]
    out = []
    for n in sizes:
        reps = {16: 15, 20: 7, 22: 4, 24: 3}[n]
        libs = state_libs + (["cirq"] if n <= 20 else [])
        out.append(("1", n, libs, reps, 3 if n <= 16 else 1))
    for n in ([20] if quick else [20, 22]):
        out.append(("10", n, state_libs, 3 if n == 20 else 2, 1))
    # Cold starts: one layer after pauses of 0.5 and 20 ms, next to the same
    # layer warm (16 qubits is timed warm above).
    for n in ([12, 16] if quick else [10, 12, 14, 16, 18]):
        if n not in sizes:
            out.append(("1", n, ["qso", "qulacs", "aer"], 21, 3))
        for pause in (500, 20000):
            out.append((f"1@{pause}", n, ["qso", "qulacs", "aer"], 21, 3))
    out.append(("gradient", 10, ["qso", "pennylane", "qulacs"], 15, 3))
    for n in ([10, 14] if quick else [10, 14, 18, 20, 22, 24]):
        out.append(("permanent", n, ["qso", "thewalrus"], 5 if n <= 20 else 2, 1))
    return out


def timing(rounds: int, quick: bool) -> list[dict]:
    results = {}
    for r in range(rounds):
        for kind, n, libs, reps, warm in configurations(quick):
            k = r % len(libs)
            order = libs[k:] + libs[:k]
            for lib in order:
                cmd = [sys.executable, str(HERE / "run.py"), "--worker", lib, kind, str(n), str(reps), str(warm)]
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
                key = (kind, n, lib)
                if p.returncode != 0:
                    print(f"FAILED {key}: {p.stderr[-800:]}", file=sys.stderr)
                    results.setdefault(key, {"error": p.stderr[-400:]})
                    continue
                times = json.loads(p.stdout.strip().splitlines()[-1])
                results.setdefault(key, {"times": []})["times"].extend(times)
        print(f"round {r + 1}/{rounds} done", file=sys.stderr, flush=True)
    rows = []
    for (kind, n, lib), v in results.items():
        row = {"kind": kind, "n": n, "library": lib}
        if "times" in v and v["times"]:
            row.update(median=statistics.median(v["times"]), minimum=min(v["times"]), runs=len(v["times"]))
        else:
            row["error"] = v.get("error", "no data")
        rows.append(row)
    return rows


def environment() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    cpu = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "packages": versions,
        "machine": os.environ.get("QSO_MACHINE"),
        "commit": os.environ.get("QSO_COMMIT") or os.environ.get("GITHUB_SHA"),
        "date": time.strftime("%Y-%m-%d"),
        "notes": [
            "ProjectQ is not included: version 0.8.0 (2022) does not build on current CPython.",
            "Cirq is timed with its built-in Python simulator, not qsim.",
            "Aer circuits are transpiled at optimization_level=0 so the gate list is unchanged; "
            "Aer's default gate fusion is on.",
        ],
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--qaoa-worker":
        print(json.dumps(qaoa_worker(sys.argv[2], int(sys.argv[3]))))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        lib, kind, n, reps, warm = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        print(json.dumps(worker(lib, kind, n, reps, warm)))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="cross_framework.json")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="exit with status 1 if a correctness deviation exceeds its tolerance")
    ap.add_argument("--no-timing", action="store_true", help="correctness only")
    ap.add_argument("--transport-only", action="store_true",
                    help="print only the transport cross-validation as JSON (correctness() runs it so, "
                    "in a child process pinned to one thread)")
    ap.add_argument("--qaoa", default="22,24",
                    help="qubit counts of the QAOA p = 10 comparison (comma-separated; empty: none)")
    args = ap.parse_args()
    if args.transport_only:
        print(json.dumps(transport_vs_qutip(args.quick)))
        return
    started = time.time()
    report = {"environment": environment()}
    report["correctness"] = correctness(args.quick)
    sizes = [int(x) for x in args.qaoa.split(",") if x] if not args.quick else [16]
    report["correctness"]["qaoa_vs_qso"] = qaoa_large(sizes)
    print("correctness:", json.dumps(report["correctness"]["max_amplitude_difference_vs_qso"]), file=sys.stderr)
    failures = check(report["correctness"])
    report["correctness"]["failures"] = failures
    if not args.no_timing:
        report["timing"] = timing(1 if args.quick else args.rounds, args.quick)
        report["rounds"] = 1 if args.quick else args.rounds
    report["seconds"] = time.time() - started
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    for f in failures:
        print(f"::error title=Cross-validation::{f}", file=sys.stderr)
    if args.check and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
