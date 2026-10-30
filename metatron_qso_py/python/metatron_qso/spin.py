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
"""

from ._core import (
    Control,
    Exchange,
    Profile,
    SpinEvent,
    SpinSystem,
    axial_csa,
    gyromagnetic_ratio,
    inept,
    inept_cycle,
    inversion_recovery,
    m2s,
    rf_pulse,
    s2m,
    spin_echo,
    spin_lock,
    unitary_bound,
)
from ._core import fid_spectrum as spectrum
from ._core import spin_delay as delay
from ._core import spin_field as set_field
from ._core import spin_pulse as pulse
from ._core import spin_shaped as shaped
from ._core import spin_sweep as sweep

__all__ = [
    "Control",
    "Exchange",
    "Profile",
    "SpinEvent",
    "SpinSystem",
    "axial_csa",
    "delay",
    "gyromagnetic_ratio",
    "inept",
    "inept_cycle",
    "inversion_recovery",
    "m2s",
    "pulse",
    "rf_pulse",
    "s2m",
    "set_field",
    "shaped",
    "spectrum",
    "spin_echo",
    "spin_lock",
    "sweep",
    "unitary_bound",
]
