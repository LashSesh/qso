# Spin dynamics and magnetic resonance

A spin-½ nucleus is a qubit, and the nuclear spins of a molecule are a
small quantum register with its own Hamiltonian: Zeeman terms in the
field, chemical shifts and scalar couplings. Magnetic resonance moves
polarisation through this register with pulses, delays and fields.
Hyperpolarisation methods do this deliberately: parahydrogen
(PASADENA, ALTADENA, SABRE) and dissolution DNP make molecules such as
[1-¹³C]pyruvate visible in MRI. `SpinSystem` simulates these dynamics
exactly, for designing and understanding such experiments. It does not
simulate images or make clinical claims.

## Spin systems

```python
import metatron_qso as q
from metatron_qso import spin

sys = q.SpinSystem(9.4)                    # field in tesla, 298.15 K
h = sys.add_spin("1H", 4.0)                # isotope, shift in ppm
c = sys.add_spin("13C", 60.0, t1=2.0, t2=0.5)
sys.couple(h, c, 140.0)                    # J in Hz
```

- **Isotopes.** ¹H, ²H (spin 1), ¹³C, ¹⁵N (negative γ), ¹⁹F and ³¹P, with
  the IUPAC gyromagnetic ratios.
- **Hamiltonian.** In the laboratory frame (`frame="laboratory"`) it is
  `Σ −γ_i B (1 + δ_i·10⁻⁶) I_zi + Σ 2πJ_ij I_i·I_j` in rad/s. In the rotating
  frame (the default) each isotope rotates at its carrier
  (`with_carrier`, 0 ppm by default). Couplings within an isotope stay
  complete, couplings between isotopes keep only `2πJ I_zi I_zj`. That
  frame is exact for one isotope at any field. With several isotopes it
  needs their Larmor frequencies to differ by much more than J, so
  low-field experiments with several isotopes use the laboratory frame.
- **States.** States are density matrices over the product basis, spin 0
  the most significant factor, and `m = I, …, −I` within a spin (for spin
  ½, α comes first). `thermal_state()` gives Boltzmann populations at the
  field and temperature. `parahydrogen_state(i, j, fraction)` puts two
  spins in the singlet. `time_average(rho)` removes the coherences between
  eigenstates of different energy, which models a state formed at random
  times. Energies are compared as in the laboratory frame, so in the
  rotating frame coherences between different `M` of an isotope go too:
  they precess at its Larmor frequency.
- **Blocks.** `H` conserves `M` of every isotope in the rotating frame and
  the total `M` in the laboratory frame. Spectra, time averages and free
  evolution use the eigendecomposition of each block, shifted by its mean
  energy, so the large Zeeman energies of the laboratory frame cost no
  precision.
- **Limits.** Up to 2048 Hilbert-space dimensions (eleven spins ½).
  `coupling_graph()` returns the coupling network as a QSO graph.

## Sequences

`evolve(rho, events)` applies a list of events:

- `spin.pulse(target, angle, phase)`, an ideal pulse
  `exp(−iθ(cos φ F_x + sin φ F_y))` in degrees. The target is an isotope
  (`"1H"`) or one spin (its index).
  In the laboratory frame the phase refers to the laboratory axes, not to
  a carrier, so pulses after a delay differ in phase between the frames.
- `spin.shaped(target, [(duration, ν₁, phase), …])`, a shaped pulse of
  piecewise constant amplitude (Hz) and phase, in the rotating frame and
  the rotating-wave approximation (the field acts on its isotope only).
- `spin.delay(t)`, free evolution with relaxation.
- `spin.set_field(B)` and `spin.sweep(to, duration, steps)`, field jumps
  and linear sweeps. They change the field for the rest of the sequence;
  the system keeps its own, so a state left at another field is observed
  with `sys.with_field(sys.field_after(events))`.

`spin.inept`, `spin.inept_cycle`, `spin.spin_echo` and
`spin.inversion_recovery` build the standard sequences. `phase_cycle`
sums the FIDs of a cycle with their receiver phases.

## Relaxation

`T₁` and `T₂` per spin are phenomenological. Lindblad operators `√k↓ I₋`
and `√k↑ I₊` relax towards thermal equilibrium at the current field and
temperature, with `k↑ + k↓ = 1/T₁` and
`k↑/k↓ = exp(ħγB(1 + δ·10⁻⁶)/k_BT)`. `√(2γ_φ) I_z` dephases, with
`γ_φ = 1/T₂ − 1/(2T₁)`, so `T₂ ≤ 2T₁` for every spin. For spin ½ this
gives exactly the decay times `T₁` of `⟨I_z⟩` and `T₂` of `⟨I₊⟩`. For ²H
the same operators give them in the high-temperature limit, and the
equilibrium is thermal; quadrupolar relaxation is not modelled.
Cross-relaxation follows from Redfield theory (next section).

The master equation conserves the coherence orders of `H`'s blocks. Each
block of its superoperator, up to 2048 elements (six protons have blocks
of 924), is exponentiated exactly, for the deviation of the state from
`1/d`: thermal polarisations are parts in 10⁵ of `1/d`, and they keep
their relative precision. A free induction decay needs only the
coherences of order −1. Short intervals and shaped pulses, which mix the
orders, use adaptive Dormand–Prince steps with a local error of 10⁻¹¹ of
that deviation when they are cheaper. A propagation estimated at more
than 1.5·10¹² operations, about a minute, is refused with an error. One
second of five relaxing protons at 9.4 T takes 0.2 s, in either frame,
and of six about ten seconds.

## Relaxation from first principles

Relaxation times can also follow from the molecule. Give spins positions
(Å) and shielding tensors (ppm, in the same molecular frame), and the
system a rotational correlation time `τ_c` (s) of isotropic tumbling:

```python
sys = q.SpinSystem(9.4)
c = sys.add_spin("13C", 171.0)
h = sys.add_spin("1H", 8.4)
sys.set_position(c, (0.0, 0.0, 0.0))
sys.set_position(h, (1.10, 0.0, 0.0))
sys.set_csa(c, spin.axial_csa(-140.0, (0.0, 1.0, 0.0)))
sys = sys.with_correlation_time(2e-11)
g = sys.relaxation_rates([sys.spin_operator(c, "z"), sys.spin_operator(h, "z")])
print(1 / g[0, 0], g[0, 1])        # 2.34 s, 0.202 /s: ¹³C T₁ and the ¹³C–¹H cross-relaxation
```

The dipolar couplings `b_ij = (μ₀/4π) γ_i γ_j ħ / r³` of every pair with
positions and the anisotropic part of every shielding tensor relax the
spins by Bloch–Wangsness–Redfield theory, cross-correlations between them
included: `dρ/dt = −i[H, ρ] − R(ρ − ρ_eq)` with spectral densities
`J(ω) = τ_c/(1 + ω²τ_c²)` at the transition frequencies of the
laboratory-frame Hamiltonian. In the rotating frame only the part of
`R` that keeps every isotope's coherence order is kept (secular with
respect to the carriers); in the laboratory frame `R` is used in full.
Redfield relaxation adds to `T₁` and `T₂` given per spin, and runs through
the same block exponentials.

- **Conventions.** Tensors are shielding tensors σ in ppm, in the frame of
  the positions; a chemical-shift tensor is `−σ`, which flips the sign of
  the interference between shielding and dipolar couplings. Only the
  traceless symmetric part relaxes.
- **Equilibrium.** The thermal state is a fixed point at any temperature,
  but the correction `−R(ρ − ρ_eq)` is semiclassical: away from
  equilibrium it is first order in `ħω/k_BT`, so near full polarisation
  (millikelvin) states far from equilibrium can turn unphysical. In the
  laboratory frame the flip-flop terms between isotopes shift the steady
  state by about `J/Δν` (5·10⁻⁷ for ¹³C–¹H at 9.4 T).
- **Validity.** Redfield theory needs motion much faster than the
  interactions, `|H₁|·τ_c ≪ 1`; QSO refuses `|A|·τ_c > 1` (the Frobenius
  norm of an interaction's spatial tensor in rad/s), so a C–H pair at
  τ_c = 1 ms fails while a protein at 50 ns does not.
- **Not modelled.** Dynamic frequency shifts, anisotropic tumbling,
  internal motion, quadrupolar relaxation, scalar relaxation (of the first
  and second kind, which matters for ²H at low field), spin-rotation,
  antisymmetric shielding, intermolecular, solvent, oxygen and
  paramagnetic relaxation, and chemical exchange.
- **Size.** The superoperator is built block by block of coherence order,
  each block at most 2048 elements: up to six relaxing protons of one
  isotope, or six spins in the laboratory frame; more is an error (`T₁`
  and `T₂` alone fall back to Dormand–Prince steps). Its build is counted
  in the work limit; one build takes about 0.04 s for five spins and
  0.5 s for six. `evolve` reuses it while the field stays, so shaped
  pulses cost little more, but every step of a sweep builds it anew.

`relaxation_rates(ops)` returns the matrix `Γ_ij = −Re Tr(O_i† D(O_j)) /
Tr(O_i† O_i)` of the relaxation `D` (without `H`): its diagonal holds the
auto-relaxation rates (`1/T₁` of `I_z`, `1/T₂` of `I₊` for an isolated
spin), the rest the cross-relaxation; with cross-relaxation, recovery has
several rates. `qso spins --rates` prints the auto-relaxation times
`1/Γ_ii` of `I_z` and `I₊` of every spin.

## Chemical exchange

Molecules change their environment: a substrate binds to a catalyst and
leaves again, hydrogen is replaced. `spin.Exchange` holds chemical sites,
each a `SpinSystem` with a concentration, and first-order channels between
them (the Bloch–McConnell equations in Liouville space, McConnell 1958).
A channel converts a fraction `rate` of its source per second; the spins
it carries keep their state, the source's other spins are traced out, and
the target's other spins arrive thermal or, for hydrogen, in the singlet
of parahydrogen. SABRE with a ¹⁵N substrate:

```python
complex_ = q.SpinSystem(1.4e-7, frame="laboratory")  # two hydrides, bound ¹⁵N
for shift, iso in ((-22.0, "1H"), (-22.0, "1H"), (250.0, "15N")):
    complex_.add_spin(iso, shift)
complex_.couple(0, 1, -7.0)
complex_.couple(0, 2, -2.0)
free = q.SpinSystem(1.4e-7, frame="laboratory")
free.add_spin("15N", 310.0, t1=30.0)
x = spin.Exchange()
x.add_site(complex_, 1.0)                  # concentrations
x.add_site(free, 10.0)
x.add_channel(0, 1, 2.0, carried=[(2, 0)])                         # substrate leaves
x.add_channel(1, 0, 0.2, carried=[(0, 2)], parahydrogen=(0, 1))    # and binds with fresh H₂
x.add_channel(0, 0, 0.5, carried=[(2, 2)], parahydrogen=(0, 1))    # hydrogen exchange
steady = x.steady_state()
print(x.polarization(steady, 1, 0))       # 0.74: the free ¹⁵N at the anti-crossing
```

Concentrations and rates must form a chemical steady state (each site
gains as many molecules per second as it loses), so bimolecular steps
enter with pseudo-first-order rates. The equations are linear, keep every
isotope's coherence order, and are propagated exactly over all sites;
`evolve` takes pulses on isotopes, delays, field jumps and sweeps,
`steady_state` solves for the stationary state (an error when it is not
unique, as without relaxation), and `fid` sums the sites weighted by
concentration. Each site relaxes by its own `T₁`, `T₂` and Redfield
relaxation. This is exact for the average over molecules that change site
at independent, exponentially distributed times (Poisson events), as a
Monte Carlo of such events in the tests confirms. Lindale et al. (Nat.
Commun. 10, 395 (2019)) simulate SABRE by sampling a fixed number of
dissociation events uniformly in time and argue that ensemble-averaged
equations fail in their regime; QSO has not been compared with their
simulations.

What the model leaves out:

- **Irreversible chemistry.** The amounts of the sites must be steady, so
  a hydrogenation that consumes its substrate (PHIP, PHIP-SAH) is not a
  channel; simulate the product as a `SpinSystem` from the moment it
  forms, or as a site that is formed and removed at the same rate.
- **Partners.** A channel has one source: in a bimolecular step only the
  spins of that source keep their state, the partner's arrive fresh (a
  mean-field picture of the partner).
- **Fresh spins** arrive thermal as a product of single-spin Boltzmann
  populations, or as the parahydrogen singlet.
- **Events.** Pulses act on an isotope in every site; there are no
  selective or shaped pulses or spin locks during exchange.
- **Size and cost.** One block of coherence order over all sites holds at
  most 2048 elements (about six plus four spins ½ in the laboratory
  frame); every step of a sweep rebuilds and exponentiates all blocks,
  and sweeps beyond the work limit are refused before they start.
- **Precision.** Next to parahydrogen order of 1, thermal polarisation
  (10⁻¹⁰ at a millitesla) is resolved to about 10⁻¹⁵ by `steady_state`
  (checked against a 40-digit solution); long delays in the laboratory
  frame lose about `ε·ω·t`, 10⁻⁹ after 600 s at 1 T.

## Long-lived states and transfer

- **Singlet order.** `spin.m2s(isotope, J, Δν)` and `spin.s2m` (Pileio,
  Carravetta and Levitt, PNAS 107, 17135 (2010)) convert magnetisation of a
  near-equivalent pair (`0 < Δν < J`) into singlet order and back:
  `90°(−y) − [τ − 180° − τ]^{n₁} − 90°x − τ − [τ − 180° − τ]^{n₂}` with
  `τ = 1/(4J)`, `n₁ = round(π/(2 arctan(Δν/J)))`, `n₂ = ⌊n₁/2⌋`, and S2M in
  reverse order. Stored as singlet order, at most 2/3 of the magnetisation
  comes back (at high temperature), inverted: S2M after M2S gives about
  −2/3 of the initial `I_z`.
- **Spin locks.** `spin.spin_lock(isotope, ν₁, duration)` irradiates
  continuously; with `ν₁ ≫ Δν` the singlet stays an eigenstate, and its
  order lives for the Redfield lifetime `T_S` of `I₁·I₂` instead of mixing
  with the triplets. Relaxation during the lock is that of free
  precession (the Redfield superoperator is not transformed into the
  frame of the lock).
- **Field cycling.** Parahydrogen order moves to a heteronucleus when the
  field is raised adiabatically from near zero (`spin.set_field` and
  `spin.sweep`): through the ultra-low fields where the levels come within
  a few hertz slowly, then faster.

## Optimal control

Gradient ascent pulse engineering (GRAPE; Khaneja et al., J. Magn. Reson.
172, 296 (2005)) designs pulses and field profiles numerically. A
`spin.Control` holds piecewise-constant controls over `steps` equal steps
of a fixed duration — the `x` and `y` amplitudes (Hz) of rf fields on
channels in the rotating frame, or the field (T) — an initial operator
`ρ(0)` and a target `C`. The transfer `Re Tr(C† ρ(T))` and its gradient
with respect to every control are exact up to rounding:

```python
s = q.SpinSystem(9.4)
s.add_spin("1H", 0.0)
s.add_spin("13C", 0.0)
s.couple(0, 1, 140.0)
rho, sz = s.thermal_state(), s.spin_operator(1, "z")
c = spin.Control.rf(["1H", "13C"], 1.2 / 140.0, 40, rho, sz, limit=5000.0)
c.ensemble = [(1.0, [k, k], []) for k in (0.9, 1.0, 1.1)]  # rf errors
r = s.optimize_control(c, seed=0, scale=500.0)
print(r["transfer"] / spin.unitary_bound(rho, sz))   # close to 1
pulse = s.control_events(c, r["controls"])           # for evolve
```

- **Ensembles.** Members `(weight, rf scales, offsets in Hz)` per
  channel average the transfer over rf inhomogeneity and offsets (an
  offset `Δ` adds `2πΔF_z`: a line at `+Δ` Hz, `−Δ/ν₀` ppm);
  `optimize_control` maximises that average less `power · Σ u²Δt` and
  `smoothness · Σ (u_{k+1} − u_k)²`. A pulse is robust only against
  what its ensemble contains: one optimised for rf errors alone, with
  about 1 kHz of rf, loses much of its transfer at 50 Hz offset.
- **Limits.** `optimize_control` keeps each channel's amplitude
  `√(x² + y²)` within `limit` and fields within their range (by a sine of
  free variables, which reaches a bound without a plateau beyond it);
  starts in the outer 2 % of a range are moved inside, starts outside
  refused. `control_transfer`, `control_events` and `evolve` take any
  amplitude.
- **Relaxation.** If the system relaxes (`T₁`, `T₂`, Redfield theory),
  the controls are optimised in Liouville space (up to six spins ½), with
  the same relaxation as `evolve` (free precession's, during the pulses as
  well), the state propagated as its deviation from the identity. The
  field derivative of relaxation and of thermal populations, in field
  profiles, is taken by Richardson-extrapolated central differences;
  against the derivative of the Boltzmann rates written out it agrees to
  10⁻¹⁴ from 1 K to room temperature and from millitesla to 20 T.
  Everything else is exact.
- **Replay.** `control_events` gives `spin.rf_pulse(...)` (rf on several
  channels at once) or a field and a delay per step; `evolve` reproduces
  the transfer.
- **Exchange.** `Exchange.optimize_profile` shapes the field of a
  chemical-exchange model (`spin.Profile`: range, steps, states and one
  target operator per site), e.g. for SABRE.
- **Local optima.** L-BFGS finds a local optimum from each start: take
  the best of several seeds for pulses. Field profiles are strongly
  multimodal; start them where the physics starts (parahydrogen at zero
  field, not a random profile).
- **Cost.** One evaluation costs about 1 ms for two spins and 100 steps
  and 26 ms for four spins and 200 steps without relaxation. With
  relaxation it grows with the rf amplitude times the duration (the
  Taylor series needs more terms): two spins and 100 steps of up to 2 kHz
  take 50 ms, three spins 0.7 s, at 10 kHz four times as long. Work and memory
  beyond the limits are refused before anything is allocated; an
  evaluation the optimiser tries beyond them, possible with relaxation
  and no `limit`, ends the run with an error. Ctrl-C reaches Python only
  when an optimisation ends.
- **The bound.** `spin.unitary_bound(ρ, C)` is the largest `Tr(C UρU†)`
  over all unitaries (Glaser et al., Science 280, 421 (1998)), the sum of
  the products of the ordered eigenvalues: no pulse sequence exceeds it.

## Spectra

`lines(rho, isotope)` gives the exact stick spectrum of a state. Each
line has a frequency in the frame (Hz), a chemical shift (ppm) and the
complex amplitude of `e^{2πift}` in `Tr(ρ(t) F₊)`. Lines at one frequency,
to the rounding of the energies, are merged, and lines below 10⁻¹³ of
the largest entry of the state, which is rounding, are dropped. At zero
field the shift of a frequency is undefined (NaN). `fid(rho, isotope,
dwell, points)` gives the free induction decay with relaxation, and
`spin.spectrum(fid, dwell, broadening)` its Fourier transform.

```python
eq = sys.thermal_state()
direct = sys.lines(sys.evolve(eq, [spin.pulse("13C", 90)]), "13C")
inept = sys.lines(sys.evolve(eq, spin.inept("1H", "13C", 140.0)), "13C")
print([abs(a["amplitude"]) / abs(b["amplitude"]) for a, b in zip(inept, direct)])
# [4.97, 2.97]: the antiphase transfer γ_H/γ_C ± carbon's own signal
# (slightly less than 4.98 and 2.98, as carbon relaxes during the delays)
```

## Parahydrogen

Hydrogenation with parahydrogen puts two protons of the product in the
singlet. At high field (PASADENA) the singlet dephases into
`¼ − I_zi I_zj`, and a 45° pulse gives two antiphase doublets. After an
adiabatic transport from low field (ALTADENA) the pair ends in one of
`|αβ⟩` and `|βα⟩`, and each multiplet shows one dominant line, the two of
opposite sign. In SABRE parahydrogen binds to a catalyst together with
the substrate, and polarisation moves from the hydrides to the substrate
where their levels anti-cross:

```python
cat = q.SpinSystem(0.0)
for shift in (-22.0, -22.0, 8.0):          # two hydrides and a substrate proton
    cat.add_spin("1H", shift)
cat.couple(0, 1, -7.0)
cat.couple(0, 2, 1.2)
for b in (2e-3, 5.7e-3, 9e-3):
    at = cat.with_field(b)
    rho = at.time_average(at.parahydrogen_state(0, 1))
    print(b, at.polarization(rho, 2))     # −0.009, −0.495, −0.017: largest near 5.7 mT
```

## NV centres

A nitrogen-vacancy centre in diamond is an electron spin 1 that is
pumped into `m_s = 0` and read out optically, coupled to its nitrogen
nucleus and to nearby ¹³C nuclei. `spin.NvCenter(field, nitrogen,
carbons)` holds it in a field (T) along its axis, with `"14N"`, `"15N"` or
no nitrogen and ¹³C couplings `(A_∥, A_⊥)` in Hz, in the secular
approximation (`D = 2.87 GHz` far above every coupling):

```text
H/2π = (D − f) S_z² + γ_e B S_z + A_N S_z I_z^N + P (I_z^N)² − γ_N B I_z^N
     + Σ_k [S_z (A_∥ I_z + A_⊥ I_x)_k − γ_C B I_z^k]
```

in the frame rotating at the microwave frequency `f`. Only `S_z`
couples to the nuclei, so a carbon's hyperfine tensor reduces to `A_∥`
and `A_⊥` (its azimuth can be rotated away, one nucleus at a time). The
electron starts in `m_s = 0`, the nuclei unpolarised, and every signal is
the `m_s = 0` population:

```python
nv = spin.NvCenter(0.0403, nitrogen="14N", carbons=[(60e3, 30e3)])
strong = [line for line in nv.lines() if line[1] > 0.01]
print(strong[:2])     # (1738.42 MHz, 0.083), (1738.48 MHz, 0.083): weak lines
                      # (10⁻⁴) flip the carbon
p = nv.signal(spin.cpmg(64, 0.541e-6))   # 0.13: the carbon's first dip
```

- **Pulses.** Sequences are lists of `spin.nv_pulse(angle, phase)` and
  `spin.nv_free(t)`; `ramsey`, `hahn`, `cpmg(n, τ)` and `xy8(n, τ)` build
  the usual ones, π pulses spaced `2τ`, the last π/2 about `−x` so that
  the signal is `(1 + coherence)/2`. A pulse drives the transition
  `0 ↔ transition` (`−1`, the default, or `+1`) at `frequency` (default:
  on resonance with the bare line, `D ± γ_e B`). The phase is the
  carrier's: a linearly polarised microwave `√2·2πΩ cos(2πft + φ) S_x`
  becomes `2π(Ω/2)(e^{−iφ}|±1⟩⟨0| + h.c.)` in the rotating-wave
  approximation, the same phase on both transitions, which matters at low
  field where both are driven. On the qubit `{m, 0}` (`|m⟩` first) a pulse
  rotates about `cos φ x + sin φ y`. With `rabi=None` pulses are ideal:
  instantaneous and on that transition alone. With a Rabi frequency `Ω`
  (Hz) the microwave drives both transitions for `θ/(360°·Ω)` seconds
  while everything else evolves, and free periods count between pulse
  centres (from the start or to the end at the ends of a sequence), as in
  experiments; adjacent free periods are merged first, and a negative
  angle is the positive one about the opposite axis.
- **Frequencies are signed.** The transition `0 ↔ m` lies at
  `D + m γ_e B` (plus the nuclear terms), which is negative for `m = −1`
  above `B = D/γ_e ≈ 102 mT`, where that level falls below `m_s = 0`. A
  microwave at `|f|` of phase `φ` is then the frame frequency `−|f|` with
  phase `−φ`; `lines`, `odmr`, `frequency` and the default all use the
  signed value.
- **Pulse errors are physical.** Finite pulses are off resonance for the
  other ¹⁴N lines (±2.14 MHz) and slightly drive the other transition.
  At 20 MHz the first carbon dip above keeps its depth under XY-8 (0.12)
  but not under CPMG (0.64): CPMG corrects pulse errors only for coherence
  along its pulse axis, not for the electron–carbon terms the resonance
  builds up.
- **ODMR.** `lines()` gives the resonance lines with their strengths
  `|⟨b|S_x|a⟩|²` averaged over the nuclei (½ per transition), lines within
  1 mHz merged (degenerate levels, e.g. at zero field), computed one
  nucleus at a time; `odmr(frequencies, rabi, duration)` gives the
  `m_s = 0` population after one pulse at each frequency (pulsed ODMR),
  one full exponential per frequency, refused beyond the work limit.
  Continuous-wave ODMR, which needs the optical cycle, is not modelled.
- **Field noise.** `spin.FieldNoise.telegraph(amplitude, rate)`,
  `.ornstein_uhlenbeck(sigma, correlation_time)` and `.pink(sigma,
  lowest, highest, fluctuators)` describe classical noise `b(t)` (Hz)
  along the axis, coupling as `2π b(t) S_z`; 1/f noise is a sum of
  Ornstein–Uhlenbeck fluctuators with corners spaced evenly in their
  logarithm. `spectrum(ω)` and `correlation(τ)` give its spectrum and
  correlation function.
- **Filter functions.** For instantaneous pulses (π/2, free periods and
  π pulses, π/2; angles modulo 360°) `spin.filter_function(events, ω)`
  is `|Y(ω)|²` (s²), `Y` the Fourier transform of the sign of the
  coherence (Cywiński et al., Phys. Rev. B 77, 174509 (2008)), and
  `spin.decay_exponent(events, noise)` is `χ = ½⟨φ²⟩ = 2π∫₀^∞ S(ω)|Y(ω)|²
  dω`, computed exactly in the time domain in one pass over the periods.
  The coherence of Gaussian noise decays as `e^{−χ}`; telegraph noise is
  not Gaussian, and its `χ` is only the second-order term. For finite
  pulses both are the approximation of free periods between pulse
  centres: a Hahn echo of 2 × 0.4 μs under Ornstein–Uhlenbeck noise of
  1 MHz and 20 ns gives 0.7735 ± 0.0011 with 2 MHz pulses, 0.7774 ±
  0.0012 with ideal ones, where `(1 + e^{−χ})/2` is 0.7788.
- **Exact trajectories.** `signal_with_noise(events, noise, trajectories,
  seed)` averages the signal over sampled noise paths and returns the
  mean and its standard error. Between pulses `S_z` commutes with the
  Hamiltonian, so a path enters as the phase `2π∫b dt` of each period,
  sampled exactly (jointly Gaussian with the end value for
  Ornstein–Uhlenbeck noise, from the flip times for telegraph noise).
  During a finite pulse the noise is held at its exact mean over pieces
  of at most `1/(16θ)` (`θ` its fastest correlation rate), up to 16 per
  pulse and one if it barely changes during the pulse; each piece is a
  full exponential. One trajectory has no standard error (NaN).

```python
noise = spin.FieldNoise.ornstein_uhlenbeck(2e5, 1e-6)   # 200 kHz, 1 μs
bare = spin.NvCenter(0.0403)
for events in (spin.ramsey(1e-6), spin.hahn(1e-6), spin.cpmg(8, 0.125e-6)):
    chi = spin.decay_exponent(events, noise)
    mean, err = bare.signal_with_noise(events, noise, 20000, seed=1)
    print((1 + math.exp(-chi)) / 2, mean, err)
# 0.7797 0.7791 0.0017 / 0.7940 0.7929 0.0016 / 0.99193 0.99195 0.00008
```

- **Cost.** In the secular approximation the Hamiltonian is block
  diagonal in the electron level and, within a block, a sum of one term
  per nucleus, so free evolution applies one 2 × 2 or 3 × 3 factor per
  nucleus to each block; equal periods and pulses are computed once.
  CPMG-64 with ¹⁴N and one ¹³C (dimension 18) takes 0.4 ms with ideal
  pulses and 1.2 ms with finite ones; CPMG-16 with ¹⁴N and six ¹³C
  (dimension 576, the largest with ¹⁴N) 0.4 s and 8.5 s (a finite pulse
  is a full matrix exponential). 10⁴ noisy trajectories of CPMG-16 on a
  bare centre take 0.05 s (0.15–0.2 s with finite pulses), 10³ of
  CPMG-64 with ¹⁴N and two ¹³C and finite pulses 8.5 s. Drawing the noise
  costs about 80 ns per component and period: 1/f noise of 1000
  fluctuators over CPMG-100 takes 1.7 s for 10³ trajectories. Propagations,
  ODMR sweeps and trajectories beyond the work limit (about a minute on
  one core) are refused before they start; `lines` and `decay_exponent`
  cost little at any size (0.03 s for dimension 768).

## Validation

- **Single spins and pairs, against closed forms.** The FID
  `−(iP/2)e^{iΩt − t/T₂}` holds to 10⁻¹⁴. The spin echo refocuses any
  offset and decays as `e^{−2τ/T₂}`. For two coupled protons it is
  modulated as `cos(2πJτ)`. Inversion recovery follows
  `P(1 − 2e^{−t/T₁})`, and every state relaxes to the thermal one.
- **INEPT.** The phase-cycled transfer gives antiphase ¹³C lines exactly
  `P_H/P_C` times those of direct excitation, 3.976 at 9.4 T (≈ γ_H/γ_C;
  the shifts change it by parts in 10⁵).
- **Parahydrogen.** The PASADENA lines are `i·sinθ cosθ·m/2`. The ALTADENA
  lines, after a 20 s sweep to 1 T and the jump into the magnet, are
  `i(∓sinθ/4 + sinθ cosθ·m/2)`, within 0.2 % of `sinθ/2` (the final state
  is reached with probability 99.9 %). The PASADENA lines agree to 0.3 %;
  what is left is strong coupling, `J/2Δν = 6·10⁻⁴`.
- **SABRE.** The polarisation transferred to a substrate proton (rotating
  frame) and to ¹⁵N (laboratory frame, about 0.14 μT) peaks within 2 % of
  the level anti-crossing `ν_S − ν_H = ±(J_AB − J_AS/4)`.
- **Propagators.** For four protons at 9.4 T and 298 K, whose state
  deviates from `1/16` by 10⁻⁶, the block exponentials agree with
  nalgebra's exponential of the full superoperator to 10⁻¹¹ of that
  deviation and Dormand–Prince steps to 10⁻⁷. The laboratory frame gives
  the polarisations and line magnitudes of the rotating frame to 10⁻⁷.
- **QuTiP.** The cross-validation builds five systems independently with
  QuTiP's spin operators (Hamiltonian, thermal state, pulses, Lindblad
  operators): three protons at 0.05 K; at 9.4 T and 298 K ¹H–¹H–¹³C–¹⁵N
  after INEPT, five protons, and two protons in the laboratory frame; and
  SABRE-SHEATH. Relative to the largest expectation, QSO agrees with the
  exact exponential of QuTiP's `liouvillian` to 7·10⁻¹² in the rotating
  frame and 2·10⁻¹⁰ in the laboratory frame at 9.4 T (rounding grows with
  the Larmor phase), and with `mesolve` to 4·10⁻⁶ and 2.5·10⁻⁴, the error
  of its adaptive steps.

- **Redfield relaxation.** Two dipolar spins follow Solomon's equations
  (auto-, cross- and transverse rates, ¹H–¹H and ¹³C–¹H, three
  correlation times, both frames) to 10⁻⁹, and two like spins the total
  rate of Bloembergen, Purcell and Pound. Carbon's Overhauser enhancement
  under proton saturation is `1 + γ_H/(2γ_C)` = 2.988 in extreme
  narrowing, and the transient after inverting the protons follows
  Solomon's equations to 10⁻⁷. Shielding anisotropy gives
  `R₁ = (ω₀²/5)(σ:σ) J(ω)` and `R₂ = (ω₀²/30)(σ:σ)(4J(0) + 3J(ω))` to
  10⁻⁹, four times faster per doubling of the field. The singlet order of
  two equivalent protons does not relax by their dipolar coupling
  (rate below 10⁻¹² of the magnetisation's), and deuteration slows the
  dipolar relaxation of carbon 15.9 times. Against QuTiP's
  `bloch_redfield_tensor`, with bath operators from an exact average over
  orientations rather than QSO's closed form, the rates of a ¹³C–¹H₂, a
  ¹³CH₃ and an amide ¹⁵N–¹H (with shielding cross-correlation) agree to
  5·10⁻¹⁵ and the expectations after exact propagation to 3·10⁻⁹.

- **Exchange.** The free induction decay of one proton exchanging
  between two sites follows the closed-form Bloch–McConnell equations to
  10⁻¹⁰ (equal and unequal populations, with and without `T₂`), and the
  lines coalesce at `k = πΔν/√2` (two maxima at 0.85 times that rate, one
  at 1.15). Exchange between sites of equal shifts keeps thermal states
  thermal; renewal at rate `k` relaxes like `e^{−kt}`; slow renewal with
  parahydrogen tends to the time average of 1.33. A SABRE complex agrees
  with a Monte Carlo of exponentially distributed exchange events
  (written with NumPy alone, 4000 molecules) within its statistical error
  (about 1 % of the values), and the free ¹⁵N substrate is polarised most
  within 3 % of the level anti-crossing.
- **Exchange against a dense reference.** An independent NumPy
  implementation of the homogeneous equations (partial traces by
  `einsum`, spin orders by axis permutations) agrees with delays, sweeps,
  FIDs and pulses on an isotope that not every site has to 10⁻¹⁰ and with
  steady states (rotating frame) to 10⁻⁹, for three sites with different
  isotopes, channels that reorder spins or swap protons within a site,
  parahydrogen on spins that are not neighbours, ²H, and the laboratory
  frame. SABRE's steady state in the laboratory frame from
  1 mT to 9.4 T agrees with a 40-digit solution to 10⁻¹⁴.
- **Transfer and storage.** M2S followed by S2M through stored singlet
  order returns 97–99.9 % of the 2/3 bound (inverted) for three pairs,
  among them the J and Δν of Pileio's ¹⁵N₂O. Parahydrogen order carried
  adiabatically from zero field puts −0.8127 polarisation on ¹³C in an
  AA'X system, against −0.8128 from sudden-adiabatic theory (the
  zero-field eigenstates' populations, computed numerically, go to the
  eigenstates of the same rank and total M); a sudden jump transfers
  none. Under a 1 kHz spin lock the singlet order of a near-equivalent
  pair decays at its Redfield rate (the rates of 1.34, validated against
  QuTiP) to 4·10⁻⁴, a lifetime 17.5 times its magnetisation's; without
  the lock 15 % less singlet order is left after half that lifetime
  (0.515 against 0.607).

- **Optimal control.** The gradients agree with central differences
  (rf and field controls, with and without relaxation, ensembles with
  rf errors and offsets, complex operators) and the transfer with
  `evolve` and with an independent NumPy propagation. GRAPE reaches the
  unitary bound of thermal ¹H → ¹³C transfer (3.976 times carbon's own
  polarisation) and of I₂S (4.476 times) to 10⁻⁶, and follows the
  time-optimal limits of Khaneja, Brockett and Glaser (Phys. Rev. A 63,
  032308 (2001)) for two spins: the in-phase transfer `S⁻ → I⁻` reaches
  their `η*(T)` to 10⁻⁹ at `JT` = 0.8 and 1.2 and one at 1.7, the
  antiphase transfer `2I_zS⁻ → I⁻` 99.4 % of `sin(πJT/2)` at `JT` = 0.5
  (100 steps), 99.7 % at 0.8 and one at 1.2. A pulse optimised for rf
  amplitudes from 80 % to 120 % transfers more than 99.99 % over that
  range (and more than 99.98 % from 75 % to 125 %), where refocused INEPT
  falls to 0.09 and 0.29. A field profile of 2 s, started at zero field,
  carries parahydrogen order to ¹³C as well as the adiabatic passage of
  50 s: −0.8128, the unitary bound within the blocks of total `M` that a
  field conserves, so no field profile does better (a linear sweep of 2 s
  gives −0.25). For SABRE build-up over 5 s, the optimised profile is the
  constant field at the anti-crossing, from its middle or its lower bound.
  The gradients agree with an independent SciPy build (`expm`,
  `expm_frechet`) to 10⁻⁹ with relaxation.

On the command line, `qso spins system.json --observe 13C --sequence
events.json` prints the lines and polarisations after a sequence
(`metatron-qso-rs/examples/spins`). `--initial pasadena:0,1` starts from
parahydrogen averaged at high field.

- **NV centres.** The lines lie at `D ± γ_e B` split by `A_N m_I` to
  10⁻³ Hz, and pulsed ODMR dips to 2/3 at each ¹⁴N line. The CPMG signal
  of a ¹³C (A_∥ = 60 kHz, A_⊥ = 30 kHz, 40 mT, with ¹⁴N) follows the
  closed form of Taminiau et al. (Phys. Rev. Lett. 109, 137602 (2012))
  to 10⁻¹⁰ over 1360 spacings, with six dips within 1.5 % of
  `τ_k = (2k − 1)π/(2ω_L + A_∥)`. An independent NumPy propagation
  (¹⁴N and two ¹³C, both transitions, ideal and finite pulses, a
  negative angle, a random state, whose coherences show the pulse phases)
  agrees to 10⁻¹⁰, and the block propagation agrees with the dense one
  for arbitrary states, Hermitian or not, to 10⁻¹⁰. Against a linearly
  polarised carrier simulated in the laboratory frame (no rotating-wave
  approximation) at zero field, 0.5 mT and 2 mT, Hahn, XY-8 and
  phase-shifted Ramsey sequences agree to 10⁻² (the rotating-wave error
  is about `Ω/f`), and ideal pulses are the limit of selective finite ones
  (2·10⁻³ at 30 MHz and 1 T). Noise during finite pulses agrees with 128
  pieces per pulse within the statistical error. The decay exponents of
  Ornstein–Uhlenbeck noise equal the closed forms of Klauder and
  Anderson for Ramsey and Hahn echoes to 10⁻⁹ and a quadrature of
  `S(ω)|Y(ω)|²` for 1/f noise to 10⁻⁵; trajectories match `e^{−χ}` within
  their statistical error, and telegraph noise the exact Kubo–Anderson
  results of 1.30. With 20 MHz pulses, XY-8 keeps the first three carbon
  dips within 0.03 of the ideal ones; CPMG loses more than 0.3 of their
  depth.
