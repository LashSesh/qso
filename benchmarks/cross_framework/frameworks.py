"""Circuit builders and runners for QSO and the reference frameworks.

A circuit is a list of operations ``(gate, qubits, angles)``; ``angles``
holds floats or ``("p", k)`` references to entry ``k`` of a parameter
vector. Conventions shared by every runner (checked by ``run.py``):

* qubit 0 is the least significant bit of a basis index; Cirq and
  PennyLane order qubits the other way, so their qubits are reversed;
* rotations are ``R_P(t) = exp(-i t P / 2)``. Qulacs' ``add_RX/RY/RZ_gate``
  use the opposite sign; its ``RotX/RotY/RotZ`` gates and negated angles
  for parametric gates are used instead.
"""

from __future__ import annotations

import numpy as np

GATES_1Q = ["h", "x", "y", "z", "s", "t"]
GATES_ROT = ["rx", "ry", "rz"]
GATES_2Q = ["cx", "cz", "swap"]
ALL_GATES = GATES_1Q + GATES_ROT + GATES_2Q


def layer_ops(n: int, layers: int = 1) -> list:
    """H and RZ(0.3) on every qubit, then a CX chain — per layer."""
    ops = []
    for _ in range(layers):
        ops += [("h", (q,), ()) for q in range(n)]
        ops += [("rz", (q,), (0.3,)) for q in range(n)]
        ops += [("cx", (q, q + 1), ()) for q in range(n - 1)]
    return ops


def random_ops(n: int, n_gates: int, rng: np.random.Generator) -> list:
    ops = []
    for _ in range(n_gates):
        g = ALL_GATES[rng.integers(len(ALL_GATES))]
        if g in GATES_2Q:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append((g, (int(a), int(b)), ()))
        elif g in GATES_ROT:
            ops.append((g, (int(rng.integers(n)),), (float(rng.uniform(-2 * np.pi, 2 * np.pi)),)))
        else:
            ops.append((g, (int(rng.integers(n)),), ()))
    return ops


def ansatz_ops(n: int, layers: int) -> tuple[list, int]:
    """EfficientSU2-like ansatz: RY and RZ on every qubit, CX chain, repeated,
    with a final rotation layer. Returns the operations and parameter count."""
    ops, k = [], 0
    for layer in range(layers + 1):
        for g in ("ry", "rz"):
            for q in range(n):
                ops.append((g, (q,), (("p", k),)))
                k += 1
        if layer < layers:
            ops += [("cx", (q, q + 1), ()) for q in range(n - 1)]
    return ops, k


def tfim_terms(n: int, j: float = 1.0, h: float = 0.7) -> list:
    """Open-chain transverse-field Ising model as (coefficient, {qubit: Pauli})."""
    terms = [(j, {q: "Z", q + 1: "Z"}) for q in range(n - 1)]
    terms += [(h, {q: "X"}) for q in range(n)]
    return terms


def _angle(a, theta):
    if isinstance(a, tuple):
        return float(theta[a[1]])
    return a


# --------------------------------------------------------------------------- QSO
def qso_circuit(n: int, ops: list, theta=None):
    import metatron_qso as q

    c = q.Circuit(n)
    params = c.new_parameters(int(theta)) if isinstance(theta, int) else None
    for g, qs, ps in ops:
        args = [params[a[1]] if isinstance(a, tuple) and params is not None else _angle(a, theta) for a in ps]
        getattr(c, g)(*qs, *args)
    return c


def qso_state(n: int, ops: list) -> np.ndarray:
    import metatron_qso as q

    return q.simulate(qso_circuit(n, ops)).amplitudes


def qso_pauli_sum(terms: list):
    import metatron_qso as q

    text = " + ".join(
        f"{c}*" + " ".join(f"{p}{qb}" for qb, p in sorted(t.items())) for c, t in terms
    )
    return q.PauliSum(text)


# --------------------------------------------------------------------------- Qiskit Aer
def qiskit_circuit(n: int, ops: list, save: bool = True):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for g, qs, ps in ops:
        getattr(qc, g)(*ps, *qs)
    if save:
        qc.save_statevector()
    return qc


def aer_simulator(fusion: bool = True):
    from qiskit_aer import AerSimulator

    return AerSimulator(method="statevector", fusion_enable=fusion)


def aer_prepare(n: int, ops: list):
    """Transpile once at optimisation level 0 (keeps the gate list intact)."""
    from qiskit import transpile

    sim = aer_simulator()
    tqc = transpile(qiskit_circuit(n, ops), sim, optimization_level=0)
    return sim, tqc


def aer_state(n: int, ops: list) -> np.ndarray:
    sim, tqc = aer_prepare(n, ops)
    return np.asarray(sim.run(tqc).result().get_statevector())


# --------------------------------------------------------------------------- Noise
# A noise specification is a dict with
#   p1     depolarising probability after every single-qubit gate,
#   p2     two-qubit depolarising probability after every two-qubit gate,
#   gamma  amplitude damping after every gate, on each of its qubits (optional).
# QSO's depolarising channels are (1−p)ρ + p/3 Σ PρP (one qubit) and
# (1−p)ρ + p/15 Σ PρP (two qubits); Qiskit's depolarizing_error(λ, n) is
# (1−λ)ρ + λ I/2ⁿ, so λ = 4p/3 and λ = 16p/15 respectively.
NOISY_1Q = GATES_1Q + GATES_ROT
NOISY_2Q = GATES_2Q


def qso_density(n: int, ops: list, noise: dict) -> np.ndarray:
    import metatron_qso as q

    single = [q.Channel.depolarizing(noise["p1"])]
    multi = [q.Channel.depolarizing_2q(noise["p2"])]
    if noise.get("gamma"):
        single.append(q.Channel.amplitude_damping(noise["gamma"]))
        multi.append(q.Channel.amplitude_damping(noise["gamma"]))
    model = q.NoiseModel(single_qubit=single, multi_qubit=multi)
    return q.DensityMatrix.simulate(qso_circuit(n, ops), noise=model).to_matrix()


def aer_density(n: int, ops: list, noise: dict) -> np.ndarray:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

    e1 = depolarizing_error(4 * noise["p1"] / 3, 1)
    e2 = depolarizing_error(16 * noise["p2"] / 15, 2)
    if noise.get("gamma"):
        damping = amplitude_damping_error(noise["gamma"])
        e1 = e1.compose(damping)
        e2 = e2.compose(damping.tensor(damping))
    model = NoiseModel()
    model.add_all_qubit_quantum_error(e1, NOISY_1Q)
    model.add_all_qubit_quantum_error(e2, NOISY_2Q)
    sim = AerSimulator(method="density_matrix", noise_model=model)
    qc = qiskit_circuit(n, ops, save=False)
    qc.save_density_matrix()
    tqc = transpile(qc, sim, optimization_level=0)
    return np.asarray(sim.run(tqc).result().data()["density_matrix"])


# --------------------------------------------------------------------------- Qulacs
_QULACS = {
    "h": "add_H_gate", "x": "add_X_gate", "y": "add_Y_gate", "z": "add_Z_gate",
    "s": "add_S_gate", "t": "add_T_gate", "rx": "add_RotX_gate", "ry": "add_RotY_gate",
    "rz": "add_RotZ_gate", "cx": "add_CNOT_gate", "cz": "add_CZ_gate", "swap": "add_SWAP_gate",
}


def qulacs_circuit(n: int, ops: list):
    from qulacs import QuantumCircuit

    c = QuantumCircuit(n)
    for g, qs, ps in ops:
        getattr(c, _QULACS[g])(*qs, *ps)
    return c


def qulacs_state(n: int, ops: list) -> np.ndarray:
    from qulacs import QuantumState

    s = QuantumState(n)
    qulacs_circuit(n, ops).update_quantum_state(s)
    return s.get_vector()


# --------------------------------------------------------------------------- Cirq
def cirq_circuit(n: int, ops: list):
    import cirq

    qs_ = cirq.LineQubit.range(n)
    table = {
        "h": cirq.H, "x": cirq.X, "y": cirq.Y, "z": cirq.Z, "s": cirq.S, "t": cirq.T,
        "cx": cirq.CNOT, "cz": cirq.CZ, "swap": cirq.SWAP,
    }
    rot = {"rx": cirq.rx, "ry": cirq.ry, "rz": cirq.rz}
    out = cirq.Circuit()
    for g, qs, ps in ops:
        gate = rot[g](ps[0]) if g in rot else table[g]
        out.append(gate.on(*[qs_[i] for i in qs]))
    return out, qs_


def cirq_state(n: int, ops: list) -> np.ndarray:
    import cirq

    circuit, qubits = cirq_circuit(n, ops)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=list(reversed(qubits)))
    return result.final_state_vector


# --------------------------------------------------------------------------- PennyLane
def _pl_apply(n: int, ops: list, theta=None):
    import pennylane as qml

    wire = lambda q: n - 1 - q  # noqa: E731  (PennyLane: wire 0 most significant)
    table = {
        "h": qml.Hadamard, "x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ, "s": qml.S,
        "t": qml.T, "rx": qml.RX, "ry": qml.RY, "rz": qml.RZ, "cx": qml.CNOT, "cz": qml.CZ,
        "swap": qml.SWAP,
    }
    for g, qs, ps in ops:
        args = [theta[a[1]] if isinstance(a, tuple) else a for a in ps]
        table[g](*args, wires=[wire(q) for q in qs])


def pennylane_state_fn(n: int, ops: list, device: str = "lightning.qubit"):
    import pennylane as qml

    dev = qml.device(device, wires=n)

    @qml.qnode(dev)
    def circuit():
        _pl_apply(n, ops)
        return qml.state()

    return circuit


def pennylane_state(n: int, ops: list) -> np.ndarray:
    return np.asarray(pennylane_state_fn(n, ops)())


def pennylane_hamiltonian(n: int, terms: list):
    import pennylane as qml

    paulis = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, observables = [], []
    for c, t in terms:
        obs = None
        for q, p in sorted(t.items()):
            o = paulis[p](n - 1 - q)
            obs = o if obs is None else obs @ o
        coeffs.append(c)
        observables.append(obs)
    return qml.Hamiltonian(coeffs, observables)


def pennylane_value_and_grad_fn(n: int, ops: list, terms: list):
    """Value and adjoint gradient on lightning.qubit."""
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("lightning.qubit", wires=n)
    h = pennylane_hamiltonian(n, terms)

    @qml.qnode(dev, diff_method="adjoint")
    def energy(theta):
        _pl_apply(n, ops, theta)
        return qml.expval(h)

    grad = qml.grad(energy)

    def run(theta):
        t = pnp.array(theta, requires_grad=True)
        return float(energy(t)), np.asarray(grad(t))

    return run


# --------------------------------------------------------------------------- Qulacs gradients
def qulacs_value_and_grad_fn(n: int, ops: list, terms: list, n_params: int):
    """Value and gradient with Qulacs' reverse-mode `backprop`."""
    from qulacs import Observable, ParametricQuantumCircuit, QuantumState

    obs = Observable(n)
    for c, t in terms:
        obs.add_operator(c, " ".join(f"{p} {q}" for q, p in sorted(t.items())))
    circuit = ParametricQuantumCircuit(n)
    index = []  # circuit parameter slot -> our parameter index
    for g, qs, ps in ops:
        if ps and isinstance(ps[0], tuple):
            # Parametric rotations use exp(+i t P/2): store −θ, flip gradient.
            getattr(circuit, {"rx": "add_parametric_RX_gate", "ry": "add_parametric_RY_gate",
                              "rz": "add_parametric_RZ_gate"}[g])(qs[0], 0.0)
            index.append(ps[0][1])
        else:
            getattr(circuit, _QULACS[g])(*qs, *ps)

    def run(theta):
        for slot, k in enumerate(index):
            circuit.set_parameter(slot, -float(theta[k]))
        grads = circuit.backprop(obs)
        state = QuantumState(n)
        circuit.update_quantum_state(state)
        value = obs.get_expectation_value(state)
        g = np.zeros(n_params)
        for slot, k in enumerate(index):
            g[k] -= grads[slot]
        return float(value), g

    return run
