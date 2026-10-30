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

On the command line, `qso spins system.json --observe 13C --sequence
events.json` prints the lines and polarisations after a sequence
(`metatron-qso-rs/examples/spins`). `--initial pasadena:0,1` starts from
parahydrogen averaged at high field.
