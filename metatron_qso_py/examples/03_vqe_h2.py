"""Variational ground states: H2 in STO-3G and a transverse-field Ising chain."""

import numpy as np

import metatron_qso as q

h2 = q.h2_sto3g()
ansatz = q.real_amplitudes(2, 1)
result = q.run_vqe(h2["hamiltonian"], ansatz, restarts=4, seed=0)
total = result["energy"] + h2["nuclear_repulsion"]
print(h2["name"])
print(f"  VQE total energy   {total:.8f} Ha")
print(f"  full-CI reference  {h2['reference_total_energy']:.6f} Ha")
print(f"  error              {abs(total - h2['reference_total_energy']):.1e} Ha "
      f"({result['iterations']} iterations, {result['evaluations']} evaluations)")

shots = q.estimate_expectation(ansatz, h2["hamiltonian"], result["parameters"], shots=100_000, seed=1)
print(f"  100k-shot estimate {shots + h2['nuclear_repulsion']:.5f} Ha")

# Gradients: adjoint differentiation equals the parameter-shift rule.
theta = result["parameters"] + 0.1
_, g_adj = q.expectation_and_gradient(ansatz, h2["hamiltonian"], theta, method="adjoint")
_, g_ps = q.expectation_and_gradient(ansatz, h2["hamiltonian"], theta, method="parameter_shift")
print(f"  max |adjoint - parameter shift| = {np.abs(g_adj - g_ps).max():.1e}")

n = 6
tfim = q.transverse_field_ising(n, j=1.0, h=1.0, periodic=True)
exact = tfim.ground_state_energy()
vqe = q.run_vqe(tfim, q.real_amplitudes(n, 6), restarts=8, seed=0)
print(f"\nTFIM ({n} sites, critical point h = J)")
print(f"  exact (Lanczos) {exact:.6f}")
print(f"  VQE             {vqe['energy']:.6f}  (relative error {abs(vqe['energy'] / exact - 1):.1e})")

print("\nOptimised H2 circuit as OpenQASM 2.0:")
print(ansatz.to_qasm(result["parameters"], measure=True))
