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


def aer_simulator(fusion: bool = True, precision: str = "double"):
    from qiskit_aer import AerSimulator

    return AerSimulator(method="statevector", fusion_enable=fusion, precision=precision)


def aer_prepare(n: int, ops: list, precision: str = "double"):
    """Transpile once at optimisation level 0 (keeps the gate list intact)."""
    from qiskit import transpile

    sim = aer_simulator(precision=precision)
    tqc = transpile(qiskit_circuit(n, ops), sim, optimization_level=0)
    return sim, tqc


def aer_state(n: int, ops: list, precision: str = "double") -> np.ndarray:
    sim, tqc = aer_prepare(n, ops, precision)
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


# --------------------------------------------------------------------------- QAOA
# QAOA for MaxCut with parameters [γ_1…γ_p, β_1…β_p]: |+⟩^n, then per layer
# e^{−iγC} with C = Σ w (1 − Z_a Z_b)/2 (RZZ(−w·γ) per edge, up to a global
# phase) and e^{−iβ Σ X} (RX(2β) per qubit). Angle references are
# ("p", k, scale): scale·θ_k.
def qaoa_ops(n: int, edges: list, p: int) -> tuple[list, int]:
    ops = [("h", (q,), ()) for q in range(n)]
    for layer in range(p):
        ops += [("rzz", (a, b), (("p", layer, -w),)) for a, b, w in edges]
        ops += [("rx", (q,), (("p", p + layer, 2.0),)) for q in range(n)]
    return ops, 2 * p


def maxcut_terms(edges: list) -> tuple[list, float]:
    """C = constant + Σ (−w/2) Z_a Z_b."""
    return [(-w / 2, {a: "Z", b: "Z"}) for a, b, w in edges], sum(w for *_, w in edges) / 2


def qso_qaoa_circuit(n: int, ops: list, n_params: int):
    import metatron_qso as q

    c = q.Circuit(n)
    params = c.new_parameters(n_params)
    for g, qs, ps in ops:
        args = [a[2] * params[a[1]] for a in ps]
        getattr(c, g)(*qs, *args)
    return c


def qulacs_qaoa_value_and_grad_fn(n: int, ops: list, terms: list, constant: float, n_params: int):
    """Value and gradient with Qulacs' `backprop`; parametric rotations
    (RX and the ZZ Pauli rotation) use exp(+i t P/2), so −scale·θ is stored
    and the gradient flipped."""
    from qulacs import Observable, ParametricQuantumCircuit, QuantumState

    obs = Observable(n)
    for c, t in terms:
        obs.add_operator(c, " ".join(f"{p} {q}" for q, p in sorted(t.items())))
    circuit = ParametricQuantumCircuit(n)
    slots = []  # (parameter index, scale) per circuit parameter
    for g, qs, ps in ops:
        if ps:
            _, k, scale = ps[0]
            if g == "rx":
                circuit.add_parametric_RX_gate(qs[0], 0.0)
            else:
                circuit.add_parametric_multi_Pauli_rotation_gate(list(qs), [3, 3], 0.0)
            slots.append((k, scale))
        else:
            getattr(circuit, _QULACS[g])(*qs)

    def run(theta):
        for slot, (k, scale) in enumerate(slots):
            circuit.set_parameter(slot, -scale * float(theta[k]))
        grads = circuit.backprop(obs)
        state = QuantumState(n)
        circuit.update_quantum_state(state)
        value = obs.get_expectation_value(state) + constant
        g = np.zeros(n_params)
        for slot, (k, scale) in enumerate(slots):
            g[k] -= scale * grads[slot]
        return float(value), g

    return run


def aer_qaoa_energy_fn(n: int, ops: list, terms: list, constant: float, precision: str = "double"):
    """⟨C⟩ with Aer's statevector method and `save_expectation_value`
    (Aer has no gradient of its own)."""
    import qiskit_aer  # noqa: F401  (adds save_expectation_value)
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    k_max = 1 + max(a[1] for _, _, ps in ops for a in ps)
    theta = ParameterVector("t", k_max)
    qc = QuantumCircuit(n)
    for g, qs, ps in ops:
        args = [a[2] * theta[a[1]] for a in ps]
        getattr(qc, g)(*args, *qs)
    op = SparsePauliOp.from_sparse_list(
        [("ZZ", sorted(t), c) for c, t in terms] + [("", [], constant)], num_qubits=n
    )
    qc.save_expectation_value(op, list(range(n)))
    sim = aer_simulator(precision=precision)
    tqc = transpile(qc, sim, optimization_level=0)

    def run(values):
        bound = tqc.assign_parameters(list(map(float, values)))
        return float(sim.run(bound).result().data()["expectation_value"].real)

    return run


# --------------------------------------------------------------------------- Pauli propagation
# The kicked Ising circuits of Kim et al., Nature 618, 500 (2023), on Qiskit's
# CouplingMap.from_heavy_hex(7) (115 qubits, 132 edges): one step is
# RX(θ_h) on every qubit, then RZZ(−π/2) on the edges of each of three colour
# classes (a bipartite edge colouring; the ZZ gates commute, so the colouring
# changes slices, never values). Operations use the format above.
KIM_THETA_J = -np.pi / 2

# Stim's Clifford gates as OpenQASM 2 (equal up to a global phase). Qiskit's
# iSWAP decomposition defines `iswap`, which qelib1.inc lacks.
STIM_TO_QASM = {
    "H": "h", "S": "s", "S_DAG": "sdg", "SQRT_X": "sx", "SQRT_X_DAG": "sxdg",
    "X": "x", "Y": "y", "Z": "z", "CX": "cx", "CY": "cy", "CZ": "cz", "SWAP": "swap", "ISWAP": "iswap",
}
QASM_ISWAP = "gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }\n"


def heavy_hex_lattice(distance: int = 7) -> tuple[int, list]:
    """The qubit count and the three colour classes of the edges (each
    sorted, edges as (low, high)) of CouplingMap.from_heavy_hex(distance),
    coloured by rustworkx's graph_bipartite_edge_color."""
    import rustworkx as rx
    from qiskit.transpiler import CouplingMap

    cm = CouplingMap.from_heavy_hex(distance)
    g = cm.graph.to_undirected(multigraph=False)
    colour = rx.graph_bipartite_edge_color(g)
    classes = [[] for _ in range(1 + max(colour.values()))]
    for idx in g.edge_indices():
        a, b = g.get_edge_endpoints_by_index(idx)
        classes[colour[idx]].append((min(a, b), max(a, b)))
    classes = [sorted(c) for c in classes]
    for c in classes:
        qs = [q for e in c for q in e]
        assert len(qs) == len(set(qs)), "a colour class is not a matching"
    return g.num_nodes(), classes


def kicked_ising_layers(n: int, colours: list, steps: int, theta_h: float) -> list:
    """The circuit as a list of layers in time order (an RX layer, then one
    RZZ layer per colour, per step), each a list of operations."""
    out = []
    for _ in range(steps):
        out.append([("rx", (q,), (theta_h,)) for q in range(n)])
        for c in colours:
            out.append([("rzz", e, (KIM_THETA_J,)) for e in c])
    return out


def qasm2(n: int, ops: list, definitions: str = "") -> str:
    """OpenQASM 2 text of `ops` on a register of n qubits (angles written
    with repr, which reads back to the same double)."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', definitions, f"qreg q[{n}];"]
    for g, qs, ps in ops:
        args = f"({', '.join(repr(float(p)) for p in ps)})" if ps else ""
        lines.append(f"{g}{args} " + ", ".join(f"q[{q}]" for q in qs) + ";")
    return "\n".join(lines) + "\n"


def qiskit_slices(n: int, layers: list) -> list:
    """One QuantumCircuit on all n qubits per layer (the slices of
    qiskit-addon-obp, which applies them last to first)."""
    return [qiskit_circuit(n, ops, save=False) for ops in layers]


def reduce_to_light_cone(ops: list, observable_qubit: int) -> tuple[list, list]:
    """Drop every RX or RZZ gate that commutes with the Heisenberg-evolved
    Z on `observable_qubit` at its position, decided from the Pauli letters
    each qubit may carry (an over-approximation, so exact). Returns the
    indices of the kept gates in `ops` (ascending) and the sorted qubits
    they act on (with the observable's qubit)."""
    types = {observable_qubit: {"Z"}}
    kept = []
    for i in range(len(ops) - 1, -1, -1):
        name, qs, _ = ops[i]
        if name == "rx":
            (q,) = qs
            t = types.get(q, set())
            if t & {"Y", "Z"}:
                types[q] = t | {"Y", "Z"}
                kept.append(i)
        elif name == "rzz":
            a, b = qs
            ta, tb = types.get(a, set()), types.get(b, set())
            if (ta | tb) & {"X", "Y"}:
                na, nb = set(ta), set(tb)
                if ta & {"X", "Y"}:
                    na |= {"X", "Y"}
                    nb |= {"Z"}
                if tb & {"X", "Y"}:
                    nb |= {"X", "Y"}
                    na |= {"Z"}
                types[a], types[b] = na, nb
                kept.append(i)
        else:
            raise ValueError(name)
    kept.reverse()
    support = sorted({q for i in kept for q in ops[i][1]} | {observable_qubit})
    return kept, support


def pauli_label(factors) -> str:
    """QSO's PauliSum term syntax ("X0 Z3", qubits ascending, "" for the
    identity) of (qubit, letter) pairs; identity letters are skipped."""
    return " ".join(f"{p}{q}" for q, p in sorted(factors) if p != "I")


def qso_operator_dict(terms: list) -> dict:
    """QSO's (coefficient, Pauli string) pairs as {label: coefficient}."""
    out = {}
    for c, s in terms:
        factors = [(int(t[1:]), t[0].upper()) for t in s.split() if t.upper() != "I"]
        k = pauli_label(factors)
        out[k] = out.get(k, 0.0) + float(c)
    return out


def qiskit_operator_dict(op) -> dict:
    """A SparsePauliOp (as qiskit-addon-obp returns it) as {label: real
    coefficient}, read from paulis.x and .z (indexed by qubit)."""
    assert not np.any(op.paulis.phase), "unexpected phase in SparsePauliOp"
    assert np.max(np.abs(op.coeffs.imag), initial=0.0) < 1e-12
    out = {}
    for x, z, c in zip(op.paulis.x, op.paulis.z, op.coeffs.real):
        k = pauli_label((int(q), "IXZY"[int(x[q]) + 2 * int(z[q])]) for q in np.flatnonzero(x | z))
        out[k] = out.get(k, 0.0) + float(c)
    return out


def stim_operator_dict(p) -> dict:
    """A stim.PauliString of sign ±1 as {label: ±1.0}."""
    x, z = p.to_numpy()
    s = complex(p.sign)
    assert s.imag == 0
    k = pauli_label((int(q), "IXZY"[int(x[q]) + 2 * int(z[q])]) for q in np.flatnonzero(x | z))
    return {k: s.real}


def compare_operators(a: dict, b: dict, atol: float) -> dict:
    """The largest |a − b| over the union of labels (a missing label counts
    as 0) and the labels with |c| > atol present on one side only."""
    worst = 0.0
    only_a = only_b = 0
    for k in set(a) | set(b):
        ca, cb = a.get(k, 0.0), b.get(k, 0.0)
        worst = max(worst, abs(ca - cb))
        if k not in b and abs(ca) > atol:
            only_a += 1
        if k not in a and abs(cb) > atol:
            only_b += 1
    return {"max_coefficient_difference": worst, "only_qso": only_a, "only_reference": only_b,
            "terms_qso": len(a), "terms_reference": len(b)}
