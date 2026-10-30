"""Nuclear spin dynamics and magnetic resonance.

``SpinSystem`` holds the nuclei (isotope, chemical shift in ppm, optional
T1 and T2 in s), their scalar couplings J (Hz), the field (T), the
temperature (K) and the frame ("rotating" or "laboratory"). States are
density matrices over the product basis, spin 0 the most significant
factor, m = I, ..., -I within a spin (for spin 1/2: alpha first). Pulse
sequences are lists of events::

    import metatron_qso as q
    from metatron_qso import spin

    sys = q.SpinSystem(9.4)
    h = sys.add_spin("1H", 4.0)
    c = sys.add_spin("13C", 60.0)
    sys.couple(h, c, 140.0)
    rho = sys.evolve(sys.thermal_state(), spin.inept("1H", "13C", 140.0))
    print(sys.lines(rho, "13C"))

``NvCenter`` is the electron spin of an NV centre in diamond with its
nitrogen and nearby 13C nuclei (secular approximation); sensing sequences
(``ramsey``, ``hahn``, ``cpmg``, ``xy8``) run with ideal or finite pulses,
and ``FieldNoise`` gives filter-function decays and exact noisy
trajectories::

    nv = spin.NvCenter(0.04, nitrogen="14N", carbons=[(60e3, 30e3)])
    print(nv.signal(spin.cpmg(64, 0.54e-6)))

``OpticalNv`` is the centre in its optical cycle (seven levels, any field
direction): CW-ODMR, pulsed readout and sensitivities::

    optical = spin.OpticalNv((0, 0, 0.005))
    print(optical.cw_line(optical.lines()[0][0], 20e6, 1e6, 5e6)["sensitivity"])

``SpinBath`` is the bath of hundreds of 13C around a centre, its coherence
from the cluster-correlation expansion::

    bath = spin.SpinBath.diamond(0.05, seed=1)
    print(bath.signal([spin.hahn(t) for t in (100e-6, 300e-6)]))
"""

from ._core import (
    DIAMOND_LATTICE_CONSTANT,
    MAX_BATH_SPINS,
    MAX_BATH_TRAJECTORIES,
    MAX_CLUSTER_ORDER,
    MAX_EXACT_BATH,
    MAX_NV_DIMENSION,
    NATURAL_C13_ABUNDANCE,
    NV_EXCITED_SPLITTING,
    NV_GYROMAGNETIC_RATIO,
    NV_ZERO_FIELD_SPLITTING,
    Control,
    Exchange,
    FieldNoise,
    NvCenter,
    NvEvent,
    OpticalNv,
    Profile,
    SpinBath,
    SpinEvent,
    SpinSystem,
    TwoLevelOdmr,
    axial_and_transverse,
    axial_csa,
    bath_ensemble_signal,
    best_sensitivity,
    cpmg,
    decay_exponent,
    diamond_sites,
    echo_sensitivity,
    ensemble_lines,
    field_from_lines,
    filter_function,
    gyromagnetic_ratio,
    hahn,
    inept,
    inept_cycle,
    inversion_recovery,
    m2s,
    nv_free,
    nv_orientations,
    nv_pulse,
    point_dipole_hyperfine,
    ramsey,
    ramsey_sensitivity,
    readout_contrast,
    rf_pulse,
    s2m,
    spin_echo,
    spin_lock,
    unitary_bound,
    xy8,
)
from ._core import fid_spectrum as spectrum
from ._core import spin_delay as delay
from ._core import spin_field as set_field
from ._core import spin_pulse as pulse
from ._core import spin_shaped as shaped
from ._core import spin_sweep as sweep

__all__ = [
    "DIAMOND_LATTICE_CONSTANT",
    "MAX_BATH_SPINS",
    "MAX_BATH_TRAJECTORIES",
    "MAX_CLUSTER_ORDER",
    "MAX_EXACT_BATH",
    "MAX_NV_DIMENSION",
    "NATURAL_C13_ABUNDANCE",
    "NV_EXCITED_SPLITTING",
    "NV_GYROMAGNETIC_RATIO",
    "NV_ZERO_FIELD_SPLITTING",
    "Control",
    "Exchange",
    "FieldNoise",
    "NvCenter",
    "NvEvent",
    "OpticalNv",
    "Profile",
    "SpinBath",
    "SpinEvent",
    "SpinSystem",
    "TwoLevelOdmr",
    "axial_and_transverse",
    "axial_csa",
    "bath_ensemble_signal",
    "best_sensitivity",
    "cpmg",
    "decay_exponent",
    "delay",
    "diamond_sites",
    "echo_sensitivity",
    "ensemble_lines",
    "field_from_lines",
    "filter_function",
    "gyromagnetic_ratio",
    "hahn",
    "inept",
    "inept_cycle",
    "inversion_recovery",
    "m2s",
    "nv_free",
    "nv_orientations",
    "nv_pulse",
    "point_dipole_hyperfine",
    "pulse",
    "ramsey",
    "ramsey_sensitivity",
    "readout_contrast",
    "rf_pulse",
    "s2m",
    "set_field",
    "shaped",
    "spectrum",
    "spin_echo",
    "spin_lock",
    "sweep",
    "unitary_bound",
    "xy8",
]
