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
  times.
- **Limits.** Up to 2048 Hilbert-space dimensions (eleven spins ½).
  `coupling_graph()` returns the coupling network as a QSO graph.

## Sequences

`evolve(rho, events)` applies a list of events:

- `spin.pulse(target, angle, phase)`, an ideal pulse
  `exp(−iθ(cos φ F_x + sin φ F_y))` in degrees. The target is an isotope
  (`"1H"`) or one spin (its index).
- `spin.shaped(target, [(duration, ν₁, phase), …])`, a shaped pulse of
  piecewise constant amplitude (Hz) and phase, in the rotating frame.
- `spin.delay(t)`, free evolution with relaxation.
- `spin.set_field(B)` and `spin.sweep(to, duration, steps)`, field jumps
  and linear sweeps.

`spin.inept`, `spin.inept_cycle`, `spin.spin_echo` and
`spin.inversion_recovery` build the standard sequences. `phase_cycle`
sums the FIDs of a cycle with their receiver phases.

## Relaxation

`T₁` and `T₂` per spin are phenomenological. Lindblad operators `√k↓ I₋`
and `√k↑ I₊` relax towards thermal equilibrium at the current field and
temperature, with `k↑ + k↓ = 1/T₁` and `k↑/k↓ = exp(ħγB/k_BT)`.
`√(2γ_φ) I_z` dephases, with `γ_φ = 1/T₂ − 1/(2T₁)`. For spin ½ this gives
exactly the times `T₁` and `T₂`. For ²H the same operators are used, and
`T₂` is only approximate. Up to dimension 8 the master equation is
integrated by the exponential of its superoperator, above by adaptive
Dormand–Prince steps with a local error of 10⁻¹². Cross-relaxation and
Redfield theory are not included.

## Spectra

`lines(rho, isotope)` gives the exact stick spectrum of a state. Each
line has a frequency in the frame (Hz), a chemical shift (ppm) and the
complex amplitude of `e^{2πift}` in `Tr(ρ(t) F₊)`. `fid(rho, isotope,
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
- **Propagators.** The superoperator and Dormand–Prince propagators agree
  to 2·10⁻¹¹.
- **QuTiP.** The cross-validation compares four systems with QuTiP's
  `mesolve`, with the same Lindblad operators built from the formulas
  above: three and five protons, ¹H–¹H–¹³C–¹⁵N after INEPT, and SABRE-SHEATH
  in the laboratory frame. Every expectation agrees to 8·10⁻⁹.

On the command line, `qso spins system.json --observe 13C --sequence
events.json` prints the lines and polarisations after a sequence
(`metatron-qso-rs/examples/spins`). `--initial pasadena:0,1` starts from
parahydrogen averaged at high field.
