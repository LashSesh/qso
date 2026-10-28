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
    "ldpc",
]

# ----------------------------------------------------------------------------
# Worker: one configuration in a fresh process.


WARMUP_SECONDS = 0.25


def worker(lib: str, kind: str, n: int, reps: int, warmup: int) -> list[float]:
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
        "bposd_vs_ldpc": bposd_vs_ldpc(quick),
        "correlated_and_distance_vs_stim": correlated_and_distance_vs_stim(quick),
        "chemistry_vs_pyscf": chemistry_vs_pyscf(),
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
    `search_for_undetectable_logical_errors` (a heuristic search)."""
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
        }
    # QSO's colour code (errors flip three detectors) through Stim.
    colour = q.css_code_memory(q.CssCode.color_code(5), 2, 0.001)
    sc = stim.Circuit(colour.to_stim())
    exact = q.circuit_distance(colour.detector_error_model())
    out["qso colour d=5"] = {
        "searched_distance_stim": len(
            sc.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=6,
                dont_explore_edges_with_degree_above=6,
                dont_explore_edges_increasing_symptom_degree=False,
            )
        ),
        "circuit_distance_qso": [exact.lower_bound, exact.upper_bound],
    }
    return out


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

    def pyscf_molecule(m, basis):
        symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        atoms = [(symbols[z - 1], tuple(r)) for z, r in zip(m.charges, m.positions)]
        return gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, charge=m.charge, spin=m.multiplicity - 1, verbose=0)

    water = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
    ammonia = "N 0.1 -0.05 0.12; H 0.9377 -0.3816 0.0; H -0.4689 -0.1 0.8121; H -0.4689 0.7 -0.8121"
    integral_err = 0.0
    for atoms in (water, ammonia, "C 0 0 0; O 0 0 1.128", "F 0 0 0; F 0.3 0.2 1.41"):
        for basis in ("sto-3g", "6-31g"):
            m = q.Molecule(atoms)
            mol = pyscf_molecule(m, basis)
            ours = q.AoIntegrals(m, basis)
            for name, theirs in (
                ("overlap", mol.intor("int1e_ovlp")),
                ("kinetic", mol.intor("int1e_kin")),
                ("nuclear", mol.intor("int1e_nuc")),
                ("eri", mol.intor("int2e")),
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
    ):
        m = q.Molecule(atoms, charge, mult)
        mol = pyscf_molecule(m, basis)
        mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
        mf.conv_tol = 1e-12
        theirs = mf.kernel()
        ours = q.hartree_fock(m, basis, unrestricted=mult > 1)
        energy_err = max(energy_err, abs(ours.energy - theirs))
        if mult == 1:
            cas = mcscf.CASCI(mf, 6, mol.nelectron - 2)
            cas.fcisolver.conv_tol = 1e-12
            theirs_cas = cas.kernel()[0]
            ours_cas = ours.molecular_integrals(frozen=1, active=6).fci_energies()[0]
            energy_err = max(energy_err, abs(ours_cas - theirs_cas))
    return {
        "integral_max_difference": integral_err,
        "energy_max_difference": energy_err,
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
}


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
    chem = c["chemistry_vs_pyscf"]
    if not chem["integral_max_difference"] <= TOLERANCES["integral"]:
        failures.append(f"integrals differ from PySCF by {chem['integral_max_difference']:.3g}")
    if not chem["energy_max_difference"] <= TOLERANCES["scf_energy"]:
        failures.append(f"Hartree–Fock or CASCI energies differ from PySCF by {chem['energy_max_difference']:.3g}")
    return failures


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
    args = ap.parse_args()
    started = time.time()
    report = {"environment": environment()}
    report["correctness"] = correctness(args.quick)
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
