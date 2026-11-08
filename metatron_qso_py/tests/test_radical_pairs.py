"""Radical pairs: closed forms, sums, field effects, products and sources."""

import math

import numpy as np
import pytest

import metatron_qso as q
from metatron_qso import spin

# |gamma_e| (rad s^-1 T^-1) and the coupling of 1 mT (Hz).
GAMMA_E = 1.760_859_630_23e11
MILLITESLA_HZ = 28_024_951.424_207_15
HBAR = 1.054_571_817e-34
BOLTZMANN = 1.380_649e-23
BIRTHS = ["singlet", "t0", "t+", "t-", "triplet", "uncorrelated"]


def one_nucleus(isotope, k):
    """A nucleus of 1 mT on radical 1 at zero field, exponential model."""
    sys = q.SpinSystem(0.0, frame="laboratory")
    e1 = sys.add_electron()
    sys.add_electron()
    n = sys.add_spin(isotope)
    sys.hyperfine(e1, n, 1.0)
    return spin.RadicalPair(sys, singlet=k, triplet=k)


def two_protons():
    """Laboratory frame at 1 mT: 1H 0.8 mT on radical 1, 1H -0.35 mT on
    radical 2, g differing by 1e-3, exchange -0.05 mT."""
    sys = q.SpinSystem(1e-3, frame="laboratory")
    e1 = sys.add_electron(spin.FREE_ELECTRON_G + 1e-3)
    e2 = sys.add_electron()
    h1 = sys.add_spin("1H")
    h2 = sys.add_spin("1H")
    sys.hyperfine(e1, h1, 0.8)
    sys.hyperfine(e2, h2, -0.35)
    sys.exchange_coupling(e1, e2, -0.05)
    return spin.RadicalPair(sys, 3e7, 8e6, 2e6)


def iz(sys, rho, j):
    return np.trace(rho @ sys.spin_operator(j, "z")).real


def test_units_and_isotopes():
    assert spin.gyromagnetic_ratio("e") == -GAMMA_E
    assert spin.gyromagnetic_ratio("14N") == 1.933_779_2e7
    assert spin.FREE_ELECTRON_G == 2.002_319_304_362_56
    sys = q.SpinSystem(0.3)
    e1 = sys.add_electron(2.0031, t1=2e-7, t2=1e-7)
    e2 = sys.add_electron()
    n = sys.add_spin("14N")
    h = sys.add_spin("1H")
    sys.hyperfine(e1, h, 1.0)
    sys.exchange_coupling(e1, e2, 0.1)
    assert sys.dimension == 2 * 2 * 3 * 2
    kinds = [s[0] for s in sys.spins]
    assert kinds == ["e", "e", "14N", "1H"]
    assert sys.spins[e1][1] == pytest.approx(389.895_675_349_638_5, rel=1e-13)
    assert sys.spins[e1][2:] == (2e-7, 1e-7)
    couplings = {(i, j): c for i, j, c in sys.couplings}
    assert couplings[(e1, h)] == pytest.approx(MILLITESLA_HZ, rel=1e-15)
    assert couplings[(e1, e2)] == pytest.approx(-0.2 * MILLITESLA_HZ, rel=1e-15)
    # A hyperfine coupling joins an electron and a nucleus; exchange two
    # electrons; g is positive and finite.
    for bad in (
        lambda: sys.hyperfine(n, h, 1.0),
        lambda: sys.hyperfine(e1, e2, 1.0),
        lambda: sys.exchange_coupling(e1, h, 1.0),
        lambda: sys.add_electron(0.0),
        lambda: sys.add_electron(math.nan),
    ):
        with pytest.raises(ValueError):
            bad()
    assert sys.dimension == 24


@pytest.mark.parametrize(
    "isotope, want", [("1H", 0.7), ("14N", 0.6), ("2H", 0.6)]
)
def test_one_nucleus_at_zero_field(isotope, want):
    k = 0.5 * GAMMA_E * 1e-3
    pair = one_nucleus(isotope, k)
    y = pair.solve("singlet").yields
    assert abs(y["singlet"] - want) <= 1e-13
    assert abs(y["singlet"] + y["triplet"] + y["escape"] - 1.0) <= 1e-13
    assert y["iterations"] == 0


def test_g_difference_at_high_field():
    sys = q.SpinSystem(0.3)
    sys.add_electron(2.0031)
    sys.add_electron(2.0027)
    h = sys.hamiltonian()
    # alpha-beta minus beta-alpha.
    dw = h[1, 1].real - h[2, 2].real
    assert dw == pytest.approx(1.055_292_006_461_082_6e7, rel=1e-12)
    for k, want in [(abs(dw), 0.75), (3e7, 0.944_943_606_620_163_1)]:
        pair = spin.RadicalPair(sys, k, k)
        y = pair.solve().yields
        assert abs(y["singlet"] - want) <= 1e-13
        assert abs(y["singlet"] - (0.5 + 0.5 * k * k / (k * k + dw * dw))) <= 1e-13
        t0 = pair.solve("t0").yields
        assert abs(t0["singlet"] - (0.5 - 0.5 * k * k / (k * k + dw * dw))) <= 1e-13
        assert y["cidnp"].shape == (0, 3) and y["nuclei"] == []


def test_yields_sum_to_one_for_each_birth():
    pair = two_protons()
    assert pair.electrons == (0, 1)
    assert pair.nuclei == [2, 3]
    assert pair.rates == (3e7, 8e6, 2e6)
    assert pair.recombination == "haberkorn"
    singlet = pair.solve("singlet").yields
    assert singlet["singlet"] == pytest.approx(0.694_062_627_181_530_1, abs=1e-13)
    assert singlet["triplet"] == pytest.approx(0.207_733_224_805_094_85, abs=1e-13)
    assert singlet["escape"] == pytest.approx(0.098_204_148_013_375_72, abs=1e-13)
    want = [
        [-0.008_415_589_688_610_586, 0.013_045_661_244_178_282, 0.002_700_375_998_470_537],
        [0.011_587_716_401_277_204, -0.008_119_976_320_934_804, -0.001_257_479_653_481_890_7],
    ]
    assert np.max(np.abs(singlet["cidnp"] - np.array(want))) <= 1e-14
    assert singlet["nuclei"] == [2, 3] and singlet["field"] == 1e-3
    for model in ("haberkorn", "jones-maeda-hore", "jmh"):
        sys = pair.system
        sys.set_relaxation(0, 2e-7, 1e-7)
        sys.set_relaxation(1, 2e-7, 1e-7)
        for s in (pair.system, sys):
            p = spin.RadicalPair(s, 3e7, 8e6, 2e6, recombination=model)
            for birth in BIRTHS + ["S", "T"]:
                y = p.solve(birth).yields
                assert abs(y["singlet"] + y["triplet"] + y["escape"] - 1.0) <= 1e-12
                assert y["backward_error"] <= 1e-12
    jmh = spin.RadicalPair(pair.system, 3e7, 8e6, 2e6, "jmh")
    assert jmh.recombination == "jones-maeda-hore"
    assert jmh.solve().yields["iterations"] > 0


def test_births_projectors_and_starts():
    pair = two_protons()
    sys = pair.system
    assert np.max(np.abs(pair.birth_state("singlet") - sys.parahydrogen_state(0, 1, 1.0))) <= 1e-15
    assert np.max(np.abs(pair.birth_state("T") - sys.parahydrogen_state(0, 1, 0.0))) <= 1e-15
    four = sum(pair.birth_state(b) for b in ("S", "t0", "t+", "t-")) / 4
    assert np.max(np.abs(four - pair.birth_state("uncorrelated"))) <= 1e-15
    for b in BIRTHS:
        p = pair.projector(b)
        assert np.max(np.abs(p @ p - p)) <= 1e-15
    # A start given as a matrix (any positive trace) is normalised.
    rho0 = 3.0 * pair.birth_state("t0")
    a = pair.solve(rho0=rho0).yields
    b = pair.solve("t0").yields
    for key in ("singlet", "triplet", "escape"):
        assert a[key] == pytest.approx(b[key], abs=1e-15)
    x = pair.solve().integral()
    assert x.shape == (16, 16)
    assert np.trace(x).real * 2e6 == pytest.approx(pair.solve().yields["escape"], rel=1e-14)


def test_field_effect_shapes_and_values():
    pair = two_protons()
    fields = [0.0, 1e-3, 5e-3]
    curve = pair.field_effect(fields)
    for key in ("field", "singlet", "triplet", "escape", "backward_error", "iterations"):
        assert curve[key].shape == (3,)
    assert curve["cidnp"].shape == (3, 2, 3)
    assert curve["nuclei"] == [2, 3]
    assert list(curve["field"]) == fields
    assert np.max(np.abs(curve["singlet"] + curve["triplet"] + curve["escape"] - 1.0)) <= 1e-13
    one = pair.solve().yields
    assert curve["singlet"][1] == one["singlet"]
    assert np.array_equal(curve["cidnp"][1], one["cidnp"])
    moved = pair.with_field(5e-3).solve().yields
    assert curve["singlet"][2] == moved["singlet"]
    t = pair.field_effect([1e-3], birth="triplet")
    assert t["singlet"][0] == pytest.approx(0.259_666_531_006_368_66, abs=1e-13)
    # Fields as a NumPy array (the stub's ArrayLike), and births by the
    # names the command line's JSON writes.
    arr = pair.field_effect(np.asarray(fields))
    assert np.array_equal(arr["singlet"], curve["singlet"])
    for short, long in (("t0", "triplet-zero"), ("t+", "triplet-plus"), ("t-", "triplet-minus")):
        a = pair.field_effect([1e-3], birth=short)
        b = pair.field_effect([1e-3], birth=long)
        assert np.array_equal(a["singlet"], b["singlet"])


def test_evolution_without_rates_is_the_spin_systems():
    pair = two_protons()
    still = spin.RadicalPair(pair.system)
    rho = pair.birth_state("singlet")
    got = still.evolve(rho, 50e-9)
    want = pair.system.evolve(rho, [spin.delay(50e-9)])
    assert np.max(np.abs(got - want)) <= 1e-13
    decayed = pair.evolve(rho, 50e-9)
    assert np.trace(decayed).real < 1.0


def test_products_on_spin_systems():
    pair = two_protons()
    solution = pair.solve("singlet")
    y = solution.yields
    product = solution.product("singlet")
    assert product.shape == (4, 4)
    assert np.trace(product).real == pytest.approx(y["singlet"], abs=1e-14)
    total = sum(solution.product(p) for p in ("singlet", "triplet", "escape"))
    assert np.trace(total).real == pytest.approx(1.0, abs=1e-13)
    copy = q.SpinSystem(1e-3, frame="laboratory")
    copy.add_spin("1H")
    copy.add_spin("1H")
    routes = [(2, 0), (3, 1)]
    for which, column in (("singlet", 0), ("triplet", 1), ("escape", 2)):
        phi = y[which]
        on = solution.product_on(which, copy, routes)
        assert np.max(np.abs(on - solution.product(which) / phi)) <= 1e-15
        for j in range(2):
            p = copy.polarization(on, j) * 0.5
            assert p == pytest.approx(y["cidnp"][j][column] / phi, abs=1e-13)
    # Two fresh protons of the target arrive in parahydrogen.
    hydro = q.SpinSystem(1e-3, frame="laboratory")
    for _ in range(4):
        hydro.add_spin("1H")
    on = solution.product_on("singlet", hydro, routes, parahydrogen=(2, 3, 1.0))
    assert np.trace(on).real == pytest.approx(1.0, abs=1e-14)

    def fresh_pair(m):
        return sum(m[4 * u : 4 * u + 4, 4 * u : 4 * u + 4] for u in range(4))

    para = hydro.parahydrogen_state(2, 3, 1.0)
    assert np.max(np.abs(fresh_pair(on) - fresh_pair(para))) <= 1e-15
    # Refusals: no escape product, 1H onto 13C, an electron carried,
    # an unknown product.
    carbon = q.SpinSystem(1e-3, frame="laboratory")
    carbon.add_spin("13C")
    no_escape = spin.RadicalPair(pair.system, 3e7, 8e6).solve()
    for bad in (
        lambda: no_escape.product_on("escape", copy, routes),
        lambda: solution.product_on("singlet", carbon, [(2, 0)]),
        lambda: solution.product_on("singlet", copy, [(0, 0)]),
        lambda: solution.product("cage"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_a_product_forms_at_its_yield_through_a_source():
    pair = two_protons()
    solution = pair.solve("singlet")
    phi = solution.yields["singlet"]
    site = q.SpinSystem(9.4, temperature=300.0)
    site.add_spin("1H", 1.0, t1=2.0)
    site.add_spin("1H", 3.0, t1=2.0)
    state = solution.product_on("singlet", site, [(2, 0), (3, 1)])
    g, kd, t1 = 1e-3, 0.5, 2.0
    r = spin.Reactions()
    p = r.add_site(site, 0.0)
    r.add_species("D", 0.0)
    assert r.add_source(p, g * phi, state) == 0
    r.add_reaction([p], ["D"], kd)
    (got_site, rate, stored), = r.sources()
    assert got_site == p and rate == g * phi
    assert np.max(np.abs(stored - state)) <= 1e-15
    back = spin.Reactions.from_json(r.to_json())
    assert len(back.sources()) == 1
    x = HBAR * spin.gyromagnetic_ratio("1H") * 9.4 / (BOLTZMANN * 300.0)
    f = g * phi
    for t in (0.5, 4.0):
        (sigma,) = r.evolve([np.zeros((4, 4))], [spin.delay(t)])
        c = f / kd * (1.0 - math.exp(-kd * t))
        assert np.trace(sigma).real / c - 1.0 == pytest.approx(0.0, abs=1e-12)
        assert r.amounts(t)[0] / c - 1.0 == pytest.approx(0.0, abs=1e-12)
        for j, shift in ((0, 1.0), (1, 3.0)):
            meq = 0.5 * math.tanh(x * (1.0 + shift * 1e-6) / 2.0)
            ps = iz(site, state, j)
            m = c * meq + f * (ps - meq) / (kd + 1.0 / t1) * (
                1.0 - math.exp(-(kd + 1.0 / t1) * t)
            )
            assert iz(site, sigma, j) / m - 1.0 == pytest.approx(0.0, abs=1e-11)
    # Sources are checked.
    for bad in (
        lambda: r.add_source(5, 1.0, state),
        lambda: r.add_source(p, -1.0, state),
        lambda: r.add_source(p, 1.0, 2.0 * state),
        lambda: r.add_source(p, 1.0, np.eye(2)),
    ):
        with pytest.raises(ValueError):
            bad()


def test_refusals():
    one = q.SpinSystem(0.0, frame="laboratory")
    one.add_electron()
    one.add_spin("1H")
    with pytest.raises(ValueError, match="exactly two electrons"):
        spin.RadicalPair(one, 1e7, 1e7)
    pair = two_protons()
    with pytest.raises(ValueError):
        spin.RadicalPair(pair.system, -1.0, 1e7)
    with pytest.raises(ValueError):
        spin.RadicalPair(pair.system, 1e7, 1e7, recombination="exponential")
    skew = pair.birth_state("singlet").copy()
    skew[0, 1] += 0.1
    with pytest.raises(ValueError, match="Hermitian"):
        pair.solve(rho0=skew)
    with pytest.raises(ValueError):
        pair.solve("singlet", rho0=pair.birth_state("singlet"))
    with pytest.raises(ValueError, match="unknown birth"):
        pair.solve("quartet")
    with pytest.raises(ValueError):
        pair.projector("t1")
    with pytest.raises(ValueError):
        pair.evolve(np.eye(4), 1e-9)
    # A zero-field pair recombining from the singlet only never empties
    # its F = 3/2 states.
    k_s_only = one_nucleus("1H", 0.0)
    for model in ("haberkorn", "jmh"):
        p = spin.RadicalPair(k_s_only.system, 1e7, recombination=model)
        with pytest.raises(ValueError, match="never recombine"):
            p.solve("singlet")
    escaping = spin.RadicalPair(k_s_only.system, 1e7, 0.0, 1e5)
    y = escaping.solve("singlet").yields
    assert abs(y["singlet"] + y["triplet"] + y["escape"] - 1.0) <= 1e-13
