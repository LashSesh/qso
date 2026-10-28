# Noisy devices and error mitigation

`metatron-qso-rs/examples/noise_mitigation.rs` prepares a 5-qubit GHZ
state on a toy device and estimates H = Σ Z_i Z_{i+1}, whose noiseless
value is 4, in every way QSO offers: exactly, from shots, with readout
mitigation, zero-noise extrapolation, probabilistic error cancellation and
classical shadows.

## The device and some helpers

The device (`examples/devices/toy_device.json`) has per-qubit T1 and T2
times and readout errors, depolarising and ZZ errors on its CX gates and
small over-rotations; [Noise](../noise.md) describes the JSON format. The
helpers build the GHZ circuit, the observable, and one row of the result
table.

{{example noise_mitigation.rs:helpers}}

## Exact values

A `DensityMatrix` evolves under the device's noise exactly. Quantum
trajectories sample pure states instead; 4000 of them estimate the same
value with a standard error. Neither includes readout errors, which only
affect measurements.

{{example noise_mitigation.rs:exact}}

## From shots

The `DeviceBackend` samples measurements, readout errors included. A
tensored readout calibration (one 2×2 confusion matrix per qubit, itself
estimated from shots) corrects the readout errors; zero-noise
extrapolation runs the circuit with the noise amplified by global folding
(scale factors 1, 3 and 5) and extrapolates to zero noise. The table
shows how much of the bias each removes, alone and together.

{{example noise_mitigation.rs:shots}}

## Probabilistic error cancellation and shadows

PEC inverts the noise of every gate as a quasi-probability distribution
over Pauli corrections, so it needs Pauli noise: here the device's CX
errors as depolarising and ZZ channels. The estimate is unbiased when the
noise model is exact; the
overhead (the one-norm of the quasi-probabilities) is the factor by which
its standard error grows. A classical shadow measures random Pauli bases
and estimates any observable from the same snapshots.

{{example noise_mitigation.rs:pec}}

## Beyond density matrices

A density matrix of 20 qubits would need 2⁴⁰ entries; trajectories need
one 20-qubit state vector each.

{{example noise_mitigation.rs:trajectories}}

## Output

{{output noise_mitigation}}
