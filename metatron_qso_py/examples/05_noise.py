"""Density-matrix simulation with gate, relaxation and readout noise."""

import numpy as np

import metatron_qso as q

bell = q.Circuit(2).h(0).cx(0, 1)
ideal = q.simulate(bell)
zz = q.PauliSum("Z0 Z1")
print("Bell state under depolarising noise (p2 = 2*p1):")
print("  p1      purity    fidelity  <Z0 Z1>")
for p in (0.0, 0.01, 0.05, 0.1):
    rho = q.DensityMatrix.simulate(bell, noise=q.NoiseModel.depolarizing(p, 2 * p))
    print(f"  {p:<6}  {rho.purity():.6f}  {rho.fidelity_with_pure(ideal):.6f}  {rho.expectation(zz):.6f}")

# Energy relaxation: |1> decays as exp(-t/T1); coherences of |+> as exp(-t/T2).
t1, t2 = 50.0, 70.0
print(f"\nThermal relaxation, T1 = {t1}, T2 = {t2}:")
for t in (0.0, 10.0, 50.0):
    excited = q.DensityMatrix.simulate(q.Circuit(1).x(0))
    excited.apply_channel(q.Channel.thermal_relaxation(t1, t2, t), [0])
    plus = q.DensityMatrix.from_pure(q.StateVector.plus(1))
    plus.apply_channel(q.Channel.thermal_relaxation(t1, t2, t), [0])
    print(
        f"  t = {t:5.1f}: P(1) = {excited.probabilities()[1]:.4f} "
        f"(exp(-t/T1) = {np.exp(-t / t1):.4f}), |rho01| = {abs(plus.to_matrix()[0, 1]):.4f} "
        f"(exp(-t/T2)/2 = {np.exp(-t / t2) / 2:.4f})"
    )

# Shot-based energy of the H2 ground state with and without noise.
h2 = q.h2_sto3g()
ansatz = q.real_amplitudes(2, 1)
theta = q.run_vqe(h2["hamiltonian"], ansatz, restarts=2)["parameters"]
model = q.NoiseModel(
    single_qubit=[q.Channel.depolarizing(0.001)],
    multi_qubit=[q.Channel.depolarizing_2q(0.01)],
    readout=(0.01, 0.02),
)
print(f"\nH2 energy from 100k shots per measurement group (nuclear repulsion included):")
for label, noise in (("noiseless", None), ("noisy", model)):
    e = q.estimate_expectation(ansatz, h2["hamiltonian"], theta, shots=100_000, seed=3, noise=noise)
    print(f"  {label:9s} {e + h2['nuclear_repulsion']:.5f} Ha")
print(f"  exact     {h2['reference_total_energy']:.5f} Ha")
