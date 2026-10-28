# VQE and QAOA

Two examples: `vqe_h2.rs` compares optimisers on H₂ and measures the
result with finite shots and with noise; `qaoa_maxcut.rs` runs QAOA for
MaxCut on five graphs with layer-wise initialisation.

## The model

`models::h2_sto3g()` holds the Hamiltonian of H₂ (STO-3G, 0.735 Å) on two
qubits as a `PauliSum`, and the nuclear repulsion separately. Its exact
ground energy comes from diagonalisation (`ground_state_energy(2)`, the
argument is the number of qubits).

{{example vqe_h2.rs:model}}

## Optimisers

`Vqe` takes a Hamiltonian and a parametrised circuit. Gradient-based
methods (L-BFGS, Adam) use exact adjoint gradients; Nelder–Mead and SPSA
use energies only. Each run starts from four random points.

{{example vqe_h2.rs:methods}}

## Shots and noise

On hardware the energy is estimated from measurements.
`estimate_expectation` measures each group of qubit-wise commuting Pauli
terms with the given number of shots, so the estimate converges to the
exact energy like one over the square root of the shots. With the
`DensityMatrixBackend` and a depolarising noise model (0.1 % per
single-qubit gate, 1 % per two-qubit gate) the estimate is biased.

{{example vqe_h2.rs:shots}}

{{example vqe_h2.rs:noise}}

Output of `cargo run --release --example vqe_h2`:

{{output vqe_h2}}

## QAOA with layer-wise initialisation

`optimize_layerwise` optimises QAOA at depth 1, then starts each deeper
level from the angles of the previous one, interpolated (the INTERP
strategy of Zhou et al.). For every depth it reports the approximation
ratio, the probability of measuring an optimal cut and the most likely
bitstring.

{{example qaoa_maxcut.rs}}

{{output qaoa_maxcut}}
