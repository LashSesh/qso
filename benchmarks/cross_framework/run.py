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
    "pennylane-lightning", "thewalrus", "numpy", "projectq", "stim", "pymatching",
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
        if lib == "qso":
            import metatron_qso as q

            c = fw.qso_circuit(n, ops)
            run = lambda: q.simulate(c)  # noqa: E731
        elif lib == "qulacs":
            from qulacs import QuantumState

            c = fw.qulacs_circuit(n, ops)

            def run():
                s = QuantumState(n)
                c.update_quantum_state(s)

        elif lib == "aer":
            sim, tqc = fw.aer_prepare(n, ops)
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
    return {
        "random_circuits": len(circuits) - 1,
        "qubits": 8,
        "gates_per_circuit": 60,
        "max_amplitude_difference_vs_qso": max_diff,
        "tfim10_energy": value,
        "gradients_vs_qso": gradients,
        "permanent_max_relative_difference_vs_thewalrus": perm_err,
        "noisy_circuits": {
            "circuits": len(noisy_circuits),
            "qubits": 5,
            "gates_per_circuit": 30,
            "noise_models": noise_models,
            "max_density_matrix_difference_vs_qiskit_aer": density_diff,
        },
        "qec_vs_stim": qec_vs_stim(quick),
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
        }
    return out


# Largest accepted deviations for --check (all are rounding-level).
TOLERANCES = {
    "state": 1e-12,
    "gradient": 1e-12,
    "permanent": 1e-10,
    "density_matrix": 1e-12,
    "error_model": 1e-12,
    # Detection-event rates are sampled: |z| over up to 72 detectors.
    "detector_rate_z": 5.0,
    # Two exact matching decoders differ only on ties and on rounded
    # weights.
    "matching_disagreement": 1e-3,
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
    d = c["noisy_circuits"]["max_density_matrix_difference_vs_qiskit_aer"]
    if not d <= TOLERANCES["density_matrix"]:
        failures.append(f"noisy density matrices differ from Qiskit Aer by {d:.3g}")
    for name, r in c["qec_vs_stim"].items():
        if r["error_mechanisms_only_one_side"] or not r["max_relative_probability_difference"] <= TOLERANCES["error_model"]:
            failures.append(f"detector error model of {name} differs from Stim's: {r}")
        if not r["max_detector_rate_z_score"] <= TOLERANCES["detector_rate_z"]:
            failures.append(f"detection-event rates of {name} differ from Stim's: {r}")
        if not r["matching_disagreement_with_pymatching"] <= TOLERANCES["matching_disagreement"]:
            failures.append(f"matching predictions on {name} differ from PyMatching's: {r}")
    return failures


# ----------------------------------------------------------------------------
# Interleaved timing


def configurations(quick: bool) -> list[tuple]:
    """(kind, n, libraries, reps per round, warm-up)."""
    state_libs = ["qso", "qulacs", "aer", "pennylane"]
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
