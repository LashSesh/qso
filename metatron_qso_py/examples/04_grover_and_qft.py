"""Grover search, the quantum Fourier transform and phase estimation."""

import math

import numpy as np

import metatron_qso as q

n, marked = 6, [11, 42]
theta = math.asin(math.sqrt(len(marked) / 2**n))
print(f"Grover search, n = {n} qubits, marked = {marked}")
print("  k  simulated   sin^2((2k+1)theta)")
for k in range(8):
    r = q.grover_search(n, marked, iterations=k)
    print(f"  {k}  {r['success_probability']:.9f} {math.sin((2 * k + 1) * theta) ** 2:.9f}")
best = q.grover_search(n, marked)
print(f"  optimal k = {best['iterations']}: P = {best['success_probability']:.6f}, "
      f"{best['gates']} gates")

state = q.simulate(q.grover_circuit(n, marked))
top = sorted(state.sample(1000, seed=0).items(), key=lambda kv: -kv[1])[:2]
print("  1000 shots, most frequent:", ", ".join(f"{b} ({int(b, 2)}) x{c}" for b, c in top))

m = 5
dim = 2**m
dft = np.exp(2j * np.pi * np.outer(np.arange(dim), np.arange(dim)) / dim) / math.sqrt(dim)
circuit = q.qft(m)
err = max(
    np.abs(q.StateVector.basis(m, x).apply(circuit).amplitudes - dft[:, x]).max()
    for x in range(dim)
)
print(f"\nQFT on {m} qubits: {len(circuit)} gates, max deviation from the DFT matrix {err:.1e}")

for phase in (0.3125, 1 / 3):
    u = np.diag([1, np.exp(2j * np.pi * phase)])
    est = q.estimate_phase(6, u, [0, 1])
    print(f"phase estimation (6 bits) of {phase:.6f}: {est['phase']:.6f} with P = {est['probability']:.4f}")
