# Pauli propagation of large noisy circuits

State vectors stop at 34 qubits, density matrices at 14, and the tableau
of [Stabilizer simulation](stabilizers.md) takes Clifford gates only.
Many experiments on 100 to 1000 qubits ask for one expectation value
`⟨O⟩` of a circuit that is mostly Clifford, or shallow, or whose
observable sees only a small part of the device: the 127-qubit
kicked-Ising experiment of Kim et al. (Nature 618, 500 (2023)),
Trotterised dynamics on heavy-hex and square lattices, and the
zero-noise extrapolations run on them. `propagate` computes such values
in the Heisenberg picture, on up to 16384 qubits: exactly where the
observable stays sparse, and with a rigorous bound on the error where it
does not.

## The Heisenberg picture and light cones

`⟨O⟩ = Tr(E_L ∘ … ∘ E_1(ρ) O) = Tr(ρ E_1† ∘ … ∘ E_L†(O))`: instead of the
state, the observable is carried backwards through the circuit, from the
last layer to the first, and read in the input state at the end. QSO
holds it as a table of Pauli strings with real coefficients,
`O = Σ c_s P_s`, each string two bit masks of `⌈m/64⌉` words, as in sparse
Pauli dynamics (Begušić, Gray and Chan, Sci. Adv. 10, eadk4321 (2024)).

Only the gates in the observable's backward light cone matter. Before
anything is propagated, the operations are sorted into levels of
operations on disjoint qubits, and an operation that cannot change any
string the observable can have reached at that point is removed. The
remaining qubits are relabelled `0..m`, so a table costs words for the
light cone, not for the register. For `⟨Z₅₇⟩` on Qiskit's 115-qubit
heavy-hex lattice after one to seven kicked-Ising steps the light cone
holds 1, 3, 5, 9, 13, 20 and 26 qubits.

The value is read in `|0…0⟩` (the strings without `X` or `Y`), in a
computational basis state, or in any product state given by its Bloch
vectors `r_q = (⟨X⟩, ⟨Y⟩, ⟨Z⟩)` (`|r_q| ≤ 1`, so mixed qubits are allowed):
each string contributes `c_s Π_q r_q[P_s(q)]`.

## Layers

A propagation takes a list of layers in time order:

- **gates**: a circuit on the register's qubits `0..k` (Rust
  `Layer::Gates`; Python a `Circuit`, a `Program` or OpenQASM 2 or 3
  text, parsed in Rust, so beyond the Python `Circuit`'s 34 qubits;
  final measurements of a program are left out, as by
  `program_circuit` in Rust, and other measurements, resets and
  conditions are refused);
- **a channel** on one to four qubits (`Layer::Channel`; Python
  `(Channel, qubits)`);
- **a sparse Pauli–Lindblad layer** with rates `Σ λ_k P_k` on any qubits,
  every rate multiplied by a factor `G` (`Layer::PauliLindblad`; Python
  `(rates, G)` with rates as text or `(rate, Pauli string)` pairs, or
  `(SparsePauliLindblad, qubits, G)` for a model of
  [Learned noise](mitigation.md#learned-noise) placed on register qubits);
- **a device**: in Python `device=` (a `DeviceModel` or its JSON text)
  turns every circuit item into the device's gates and channels, in Rust
  `device_layers`.

## Gates

- **Clifford gates** (H, S, S†, √X, √X†, X, Y, Z, SWAP, CX, CY, CZ) map a
  string to one string with a sign, by the rules of the stabilizer
  tableau. Rotations by multiples of π/2 become such signed permutations
  too (below).
- **Rotations** `R = exp(−iθG/2)` (RX, RY, RZ, RXX, RYY, RZZ, P, T and T†,
  the last three as Z rotations) leave a string `P` that commutes with
  `G` unchanged and split one that anticommutes into
  `cos θ·P + sin θ·(iGP)`.
- **Every other gate on at most four qubits**, controls included (U,
  controlled rotations and phases, CH, CS, CSX, CCX, CCZ, CSWAP, C3X,
  custom unitaries), acts by its Pauli transfer matrix
  `T_ab = 2^{−k} Re Tr(P_a U† P_b U)`. A gate on five or more qubits is
  refused; decompose it first.

Entries of a table of magnitude at most 10⁻¹⁴ are dropped, and
their magnitude goes into the truncation bound. That turns rotations at
multiples of π/2 (`cos(π/2) = 6·10⁻¹⁷`) into exact signed permutations and
removes the rounding residue of transfer matrices built from matrices.
Tables are computed once per distinct gate and angle.

Gates are applied level by level. A level of Clifford gates, Pauli
channels and Pauli–Lindblad layers maps every string to one string: runs
of such levels rewrite the table in place. Any other level splits every
string into its images at once, and equal images are merged. The results
are bit-identical for any number of threads.

## Noise

- **Channels** on one to four qubits act by the transfer matrices of
  their adjoints, `T_ab = 2^{−k} Re Tr(E(P_a) P_b)`: Pauli and depolarising
  channels as a factor per string, amplitude damping as `Z ↦ (1 − γ)Z + γI`,
  thermal relaxation, coherent rotations and any Kraus channel by their
  matrices.
- **Sparse Pauli–Lindblad layers** with rates `λ_k` on generators `P_k`
  (van den Berg, Minev, Kandala and Temme, Nat. Phys. 19, 1116 (2023))
  multiply a string by `exp(−2G Σ λ_k)` over the generators that
  anticommute with it. Any real factor `G` is allowed: `G > 1` amplifies
  the noise, as probabilistic error amplification (PEA) does, and `G = −1`
  placed after a device's own noise gives the value probabilistic error
  cancellation (PEC) converges to with that model.
- **Device models** of [Noise](noise.md#device-models) compile to their
  gates and channels: gate rules, idle T₁/T₂ relaxation, spectators and
  fixed over-rotations, as `DensityMatrix.simulate(c, device=…)` applies
  them. Each circuit item is compiled on its own, so the device noise of
  a circuit in several pieces is that of the pieces run one after
  another. On this path a device may list up to 16384 qubits (the
  simulators take 34).
- **Readout errors are not applied**, as in a density matrix's
  `expectation`. Mitigate them separately.
- **Slow noise** (quasi-static and telegraph detunings, spreads of
  over-rotations) is refused: it is correlated in time, and its average
  over a circuit is no product of channels.

## Truncation and its bound

Without options the propagation is exact up to rounding, and it fails
with an error once more than `max_terms` strings (2²⁴ by default) are
held at once, the images of a splitting level before they merge
included. Two truncations keep tables small: `min_coefficient` drops
strings with `|c|` below it, and `max_weight` drops strings with more
non-identity factors. Both are applied to the observable and before
every splitting level.

The adjoint of a channel is unital and does not increase the operator
norm (Russo–Dye), and neither does a unitary, so a string `c·P` dropped
anywhere changes `⟨O⟩` by at most `|c|` in every input state. An inverted
Pauli–Lindblad layer (`G < 0`) is no channel: it can increase the norm by
`e^{2|G| Σλ_k}`, summed over the generators that touch the light cone at
that point. The light cone counts the images of table entries snapped
to zero too (the exact map carries them on, through the inverted layers
as well), so that its generators cover everything the bound counts. `truncation_bound` adds up every dropped magnitude, every
table entry snapped to zero, and every string that cancelled exactly
(its coefficient at most `2⁻⁵⁰` of the magnitudes summed into it),
each times the overheads of the inverted layers between it and the start
of the circuit. It bounds `|exact − value|` rigorously for every product
or mixed input state, up to the rounding of the arithmetic that is kept
(about 10⁻¹⁶ per operation and string) and up to the defects of at most
10⁻⁹ that channel and gate validation accept.

The bound is tight for a single dropped string (dropping `Z₀` before an
inverted `X₀` layer with rate 0.1 makes an error of `e^{0.2}`, and the
bound says `e^{0.2}`), but it adds magnitudes, so in deep circuits it is
far above the actual error. On the kicked-Ising circuits below, at
θ_h = 0.3, it is 230 to 6500 times the error; at θ_h = π/4 and 1, where the
observable spreads, it exceeds 1 and says nothing (the values were still
within 2.6·10⁻² of the exact ones).

```python
import metatron_qso as q

# Kicked Ising on a 12 × 12 square lattice (144 qubits): RX(θ) on every
# qubit, then RZZ(−π/2) on every edge, in four layers of disjoint edges.
L = 12
n = L * L
edge_layers = [[], [], [], []]
for r in range(L):
    for c in range(L):
        k = r * L + c
        if c + 1 < L:
            edge_layers[c % 2].append((k, k + 1))
        if r + 1 < L:
            edge_layers[2 + r % 2].append((k, k + L))


def kicked_ising(theta, steps):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for _ in range(steps):
        lines += [f"rx({theta}) q[{k}];" for k in range(n)]
        for edges in edge_layers:
            lines += [f"rzz(-pi/2) q[{a}],q[{b}];" for a, b in edges]
    return "\n".join(lines) + "\n"


centre = f"Z{(L // 2) * L + L // 2}"
for steps in [1, 2, 3]:                       # exact: 2, 34, 9826 strings
    r = q.propagate(kicked_ising(0.3, steps), centre)
    print(steps, r["value"], r["terms"], r["truncation_bound"])
for tau in [1e-3, 1e-4]:                      # four steps, truncated
    r = q.propagate(kicked_ising(0.3, 4), centre, min_coefficient=tau)
    print(tau, r["value"], "±", r["truncation_bound"], r["terms"], r["dropped_terms"])
```

The result is a dict with `value`, `truncation_bound`, `terms` (strings
at the start of the circuit), `peak_terms` (most strings in one table),
`dropped_terms` and, with `return_operator=True`, `operator`: the
Heisenberg observable as `(coefficient, Pauli string)` pairs.

## Noise in Python

```python
import json
import math
import metatron_qso as q

# Closed forms: RY(0.7) then amplitude damping (γ = 0.2) gives
# ⟨Z⟩ = (1 − γ) cos 0.7 + γ; a Bell pair under two-qubit depolarising
# noise (p = 0.06) keeps ⟨XX⟩ = 1 − 16p/15.
r = q.propagate([q.Circuit(1).ry(0, 0.7), (q.Channel.amplitude_damping(0.2), [0])], "Z0")
assert abs(r["value"] - (0.8 * math.cos(0.7) + 0.2)) < 1e-15
r = q.propagate([q.Circuit(2).h(0).cx(0, 1), (q.Channel.depolarizing_2q(0.06), [0, 1])], "X0 X1")
assert abs(r["value"] - 0.936) < 1e-15

# A GHZ state of 60 qubits on a device with idle relaxation (T1, T2 in
# µs) and two-qubit depolarising noise on every cx.
n = 60
ghz = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" + f"qreg q[{n}];\nh q[0];\n"
ghz += "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(n - 1))
device = (q.DeviceModel("chain")
          .with_default_qubit(t1=100.0, t2=80.0)
          .with_rule(gate="cx", duration=0.3, depolarizing=0.01))
print(q.propagate(ghz, "Z0 Z59", device=device)["value"])

# Per-qubit entries beyond the simulators' 34 qubits: pass the device as
# JSON text, which this path reads with its own limit (16384).
spec = json.loads(device.to_json())
spec["qubits"] = [{"t1": 100.0 - k, "t2": 80.0 - k} for k in range(n)]
print(q.propagate(ghz, "Z0 Z59", device=json.dumps(spec))["value"])

# The Heisenberg observable at the start, and a basis state as input
# (qubit 0 on the right): Z0 Z59 becomes Z1 … Z59, X0 becomes Z0 X1.
r = q.propagate(ghz, "Z0 Z59 + 0.5*X0", return_operator=True)
print(r["operator"])
assert q.propagate(ghz, "Z0 Z59", state="1" + "0" * (n - 1))["value"] == -1.0
```

## Probabilistic error amplification and the PEC limit

With a noise model of every layer, the values at several noise factors
`G` and an extrapolation to `G = 0` are zero-noise extrapolation by
probabilistic error amplification, without sampling: loop over `G`, then
pass the values to `extrapolate` (Python) or `zne::extrapolate` (Rust,
with each `PropagationResult::estimate()`, an exact `Estimate`). The
extrapolated value is `Σ w_i v_i` with the weights of the extrapolation,
so its truncation error is at most `Σ |w_i| b_i` for the bounds `b_i` of
the values: Richardson on `G = 1, 1.2, 1.6` has the weights 16, −20 and 5,
which amplify the bounds 41 times.

`G = −1` after the noise gives the value PEC converges to with the
model. It equals the noiseless value only where the model is the
device's noise exactly: on hardware, for twirled Pauli noise on the
mitigated layers.

```python
import metatron_qso as q

# A 100-qubit chain: RY on every qubit, then CX on even and odd pairs;
# after each CX layer a sparse Pauli–Lindblad layer (10⁻³ on X, Y and Z
# of every qubit, 5·10⁻⁴ on ZZ of every pair the layer couples) scaled by G.
n = 100
head = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" + f"qreg q[{n}];\n"


def cx_layer(start):
    return head + "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(start, n - 1, 2))


def ry_layer(angle):
    return head + "".join(f"ry({angle + 0.01 * k}) q[{k}];\n" for k in range(n))


def rates(start):
    terms = [(1e-3, f"{p}{k}") for k in range(n) for p in "XYZ"]
    return terms + [(5e-4, f"Z{k} Z{k + 1}") for k in range(start, n - 1, 2)]


def circuit(g, depth=4):
    layers = []
    for d in range(depth):
        layers.append(ry_layer(0.3 + 0.1 * d))
        for start in (0, 1):
            layers += [cx_layer(start), (rates(start), g)]
    return layers


obs = "Z50 + 0.5*Z49 Z50"
ideal = q.propagate(circuit(0.0), obs)["value"]       # 1 269 486 strings
scales = [1.0, 1.2, 1.6]
runs = [q.propagate(circuit(g), obs) for g in scales]
zne = q.extrapolate(scales, [r["value"] for r in runs], "richardson")
bound = sum(abs(w) * r["truncation_bound"] for w, r in zip([16, -20, 5], runs))
print(ideal, runs[0]["value"], zne.value, bound)   # ideal, noisy, mitigated

# Inverting the model right after it restores the noiseless value.
pec = []
for layer in circuit(1.0):
    pec.append(layer)
    if isinstance(layer, tuple):
        pec.append((layer[0], -1.0))
assert abs(q.propagate(pec, obs)["value"] - ideal) < 1e-12
```

## In Rust

`metatron_qso::propagation` has `propagate(layers, observable, state,
options)`, the operator type `PauliOperator` (Pauli strings on any number
of qubits; `parse`, `from_sum`, `from_lindblad`, `from_terms`, `add_term`,
`terms`; `from_terms` builds many terms at once, while `add_term` is a
sorted insert that costs time in proportion to the operator), `Layer`,
`PropagationOptions` (`with_min_coefficient`, `with_max_weight`,
`with_max_terms`, `with_operator`), `PropagationResult`, `device_layers`,
`device_from_json` and `program_circuit` (the gates of a `Program`, final
measurements left out). The example `pauli_propagation` runs the
kicked-Ising circuits of Kim et al. on the 115-qubit heavy-hex lattice,
exact through five steps, truncated at six and seven, and with scaled
noise.

```rust
use metatron_qso::noise::{Channel, DeviceModel, GateRule, QubitProperties};
use metatron_qso::propagation::{
    Layer, PauliOperator, PropagationOptions, device_layers, propagate,
};
use metatron_qso::quantum::{Circuit, Pauli};

fn main() -> metatron_qso::Result<()> {
    // Trotter steps of a transverse-field Ising chain of 200 qubits:
    // RX on every qubit, then RZZ on even and on odd bonds.
    let n = 200;
    // Sparse Pauli–Lindblad noise after each RZZ layer: 10⁻³ on X, Y and Z
    // of every qubit, 5·10⁻⁴ on ZZ of every bond the layer couples.
    let rates = |start: usize| -> metatron_qso::Result<PauliOperator> {
        let mut r = PauliOperator::new(n)?;
        for k in 0..n {
            for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                r.add_term(1e-3, &[(k, p)])?;
            }
        }
        for k in (start..n - 1).step_by(2) {
            r.add_term(5e-4, &[(k, Pauli::Z), (k + 1, Pauli::Z)])?;
        }
        Ok(r)
    };
    let noisy = |g: f64| -> metatron_qso::Result<Vec<Layer>> {
        let mut layers = Vec::new();
        for _ in 0..6 {
            let mut kick = Circuit::new(n);
            for k in 0..n {
                kick.rx(k, 0.3);
            }
            layers.push(Layer::gates(kick));
            for start in [0, 1] {
                let mut zz = Circuit::new(n);
                for k in (start..n - 1).step_by(2) {
                    zz.rzz(k, k + 1, 0.4);
                }
                layers.push(Layer::gates(zz));
                layers.push(Layer::pauli_lindblad(rates(start)?, g));
            }
        }
        Ok(layers)
    };
    let obs = PauliOperator::parse(n, "Z100 + 0.5*Z99 Z100")?;
    let exact = PropagationOptions::default();
    for g in [0.0, 1.0, 2.0, -1.0] {
        let r = propagate(&noisy(g)?, &obs, None, &exact)?;
        println!("G = {g:>4}: {:.12} ({} strings)", r.value, r.terms);
    }
    // Truncated: strings below 10⁻⁴ are dropped, and the bound covers them.
    let opts = PropagationOptions::default().with_min_coefficient(1e-4);
    let r = propagate(&noisy(1.0)?, &obs, None, &opts)?;
    println!(
        "truncated: {:.12} ± {:.2e}, {} strings, {} dropped",
        r.value, r.truncation_bound, r.terms, r.dropped_terms
    );

    // A circuit on a device: idle T1/T2 and noise on every CX as channels,
    // then amplitude damping on the last qubit.
    let device = DeviceModel::new("chain")
        .with_default_qubit(QubitProperties::new().with_t1_t2(100.0, 80.0))
        .with_rule(GateRule::for_gate("cx").with_duration(0.3).with_depolarizing(0.01));
    let mut ghz = Circuit::new(n);
    ghz.h(0);
    for k in 0..n - 1 {
        ghz.cx(k, k + 1);
    }
    let mut layers = device_layers(&ghz, &[], &device)?;
    layers.push(Layer::channel(Channel::amplitude_damping(0.1)?, &[n - 1]));
    let zz = PauliOperator::parse(n, &format!("Z0 Z{}", n - 1))?;
    let r = propagate(&layers, &zz, None, &exact)?;
    println!("GHZ on {n} qubits with device noise: <Z0 Z{}> = {:.12}", n - 1, r.value);
    Ok(())
}
```

The exact runs hold 208 286 strings; the truncated one is 1.8·10⁻⁴ off
the exact value, within its bound of 0.02. The program takes 2.4 s on one
thread and 0.7 s on four.

## On the command line

`qso propagate FILE --observable "…"` reads an OpenQASM circuit (gates,
optionally followed by final measurements, which are left out: the value
is that of the state before them; a measurement followed by a gate on
its qubit, resets and conditions are refused) and prints the value, the bound and the string
counts (`--json` for the keys `value`, `truncation_bound`, `terms`,
`peak_terms`, `dropped_terms`, `operator`, `qubits` and `layers`).
`--device FILE` runs it with a device's noise (per-qubit entries up to
16384), `--state BITS` starts from a basis state (qubit 0 on the right),
`--min-coefficient`, `--max-weight` and `--max-terms` truncate and limit,
and `--operator` prints the observable at the start of the circuit. On a
100-qubit GHZ circuit followed by RY(0.2) on every qubit:

```text
$ qso propagate ghz100.qasm --observable "Z0 Z99 + 0.5*X50"
<O> = 0.960530497001 (truncation bound 0.000e0)
100 qubits, 1 layer; 6 strings at the start, at most 6 held, 0 dropped

$ cat chain.json
{"name": "chain", "default_qubit": {"t1": 100.0, "t2": 80.0},
 "gates": [{"gate": "cx", "duration": 0.3, "depolarizing": 0.01}]}
$ qso propagate ghz100.qasm --observable "Z0 Z99" --device chain.json
<O> = 0.224540781668 (truncation bound 1.702e-15)
100 qubits, 791 layers; 303 strings at the start, at most 303 held, 0 dropped
```

## How it is checked

- **State vectors and density matrices.** Without truncation, random
  circuits of the whole gate set (2 to 8 qubits, up to depth 20; 12 and 20
  qubits at depth 20 for Clifford circuits with six rotations) equal the
  state vector to 10⁻¹². Every channel kind, random Kraus channels on one to four qubits and compiled devices (idle T₁/T₂,
  spectators, Pauli and Kraus rules, fixed over-rotations; up to 12
  qubits) equal the density matrix to 10⁻¹². Pauli–Lindblad layers in the
  middle of a three-qubit circuit at `G` = 0, 1, 2.5, −0.5 and −1, and `G`
  = 1 then −1, equal a density-matrix reference to 10⁻¹⁴; inverted layers
  are checked against the density matrix of the inverse map too.
- **The bound.** On 24 random noisy circuits of 3 to 7 qubits (216
  truncated runs) with every
  channel kind, `G` ∈ {0, 1, 2, −1}, cut at `min_coefficient` ∈ {10⁻³,
  10⁻², 5·10⁻²} and `max_weight` ∈ {2, 3}, from `|0…0⟩`, random product
  states and mixed states, the error never exceeds the bound; the bound
  is tight for a single dropped string.
- **Clifford circuits against Stim.** The images `C†Z_qC` of a seeded
  Clifford circuit of 10³ qubits and 10⁴ gates equal Stim's
  `PauliString.before` string for string and sign for sign (20 strings in
  the tests), and the tableau's generators on 10⁴ qubits. In the
  cross-validation, circuits with 3·10⁴ gates on 10³ qubits and 1.5·10⁵
  gates on 10⁴ qubits carry 50 strings each, 10 single `Z_q` and 40
  random dense strings.
- **Kicked Ising on heavy-hex(7)**, `⟨Z₅₇⟩` of Qiskit's
  `CouplingMap.from_heavy_hex(7)` (115 qubits), RX(θ_h) on every qubit,
  then RZZ(−π/2) on three colour classes of edges:
  - at θ_h = 0 and π/2 (Clifford) one string for 1 to 20 steps, at π/2
    with the weights and signs of Stim;
  - at θ_h = 0.3, π/4 and 1 through five steps without truncation, equal
    to Qiskit Aer's state vector of the light cone to 7.7·10⁻¹⁵, with 2,
    10, 70, 1620 and 134 744 strings (134 256 at π/4, where 488 strings of
    the fifth step vanish in exact arithmetic; in floating point most
    leave rounding residues, which the cancellation rule removes, and an
    independent propagator without the rule keeps 134 740 strings), and
    string for
    string equal to qiskit-addon-obp 0.3's
    `backpropagate` (coefficients to 6.3·10⁻¹⁵);
  - at six and seven steps (light cones of 20 and 26 qubits), truncated
    at 10⁻³ and 10⁻⁴, within the bound of Aer's light-cone value;
  - with a sparse Pauli–Lindblad model after every RZZ layer (a uniform
    one and a seeded one with all nine two-qubit Paulis per edge), at `G`
    = 0, 1, 1.5, 2, −1 and 1 then −1: equal to qiskit-addon-obp's
    `PauliLindbladErrorInstruction` to 2.1·10⁻¹⁵ and, for `G ≥ 0`, to Aer's
    density matrix of the light cone to 8.3·10⁻¹⁶.
- **Threads.** Values, bounds and operators are compared bit for bit on
  one, two and four threads.

The comparisons with Stim, Aer and qiskit-addon-obp run in the
[cross-validation](validation.md).

## Performance

On four cores (release build), the exact five-step kicked-Ising runs on
heavy-hex(7) (134 744 strings) take 30 ms on one thread and 15 ms on
four, against 4.7 to 5.0 s for qiskit-addon-obp's `backpropagate`. A
Pauli–Lindblad layer after every ZZ layer costs nothing measurable on one
thread (a factor per string); on four threads its extra in-place passes
add up to 7 ms of dispatch over small tables. Peak memory is set by the images of a splitting level before they merge, not
by the table: about 100 bytes per string on light cones of up to 64
qubits when every image is new, 140 on up to 128 qubits and 200 on up to
256, and several times that when many images merge into few strings.
The truncated runs at θ_h = 1 (six steps, `min_coefficient` 10⁻⁴) hold
9.1·10⁶ strings at their peak, take 4.7 s on one thread and 1.6 s on
four, and use 0.83 GB (resident peak of a fresh process).

## Limits

| Limit | Value |
|---|---|
| Register | 16384 qubits |
| Gates and channels | at most four qubits each, controls included |
| Strings held at once | `max_terms`, 2²⁴ by default, at most 2³⁰ (images before merging count) |
| Memory | about 100 bytes per string held on light cones of up to 64 qubits, plus 32 per further 64 qubits; more when many images merge |
| Device models | Markovian only; per-qubit entries up to 16384; readout errors not applied |
| Input states | product states (pure or mixed), `\|0…0⟩` by default |
| Accuracy | exact up to rounding without truncation; with truncation within `truncation_bound`, which is rigorous but adds magnitudes and grows far above the actual error in deep circuits |

Not included: Pauli twirling of noisy layers (the device's noise is
propagated as given), noise learning on non-Clifford layers, and a
Python `Circuit` beyond 34 qubits (larger circuits go in as OpenQASM text
or a `Program`).
