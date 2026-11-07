"""Transport to a trap: efficiency, mean arrival time, dark states, scans."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pytest

import metatron_qso as q

if TYPE_CHECKING:
    from metatron_qso._core import TransportResult

# The detuned dimer: H = [[3, −1], [−1, 0]], a trap of rate 1 at node 1,
# loss 0.05 on both nodes, dephasing g on both. Exact rationals (SymPy):
# g → (η, τ, lost).
DIMER = {
    0.0: (
        Fraction(224000, 290427),
        Fraction(9200540, 2032989),
        Fraction(66427, 290427),
    ),
    1.0: (Fraction(32000, 38881), Fraction(139260, 38881), Fraction(6881, 38881)),
    3.0: (
        Fraction(288000, 348689),
        Fraction(100414940, 28243809),
        Fraction(60689, 348689),
    ),
    10.0: (
        Fraction(7072000, 9575681),
        Fraction(11335156860, 2116225501),
        Fraction(2503681, 9575681),
    ),
    100.0: (
        Fraction(64672000, 235515281),
        Fraction(7074070324860, 475976382901),
        Fraction(170843281, 235515281),
    ),
}

RESULT_KEYS = {
    "efficiency",
    "arrival_time",
    "trapped",
    "lost",
    "dark",
    "added_dephasing",
    "dimension",
    "slowest_decay",
    "condition",
    "error",
    "solves",
}


def dimer(generator: str = "adjacency", rate: float = 0.0) -> q.DephasingWalk:
    return q.DephasingWalk(
        q.Graph.path(2),
        generator,
        1.0,
        rate,
        site_energies=[3.0, 0.0],
        traps={1: 1.0},
        losses=0.05,
    )


def arrival(r: TransportResult) -> float:
    tau = r["arrival_time"]
    assert tau is not None
    return tau


def complete_symmetries(n: int, fixed: int) -> list[list[int]]:
    """Generators of the permutations of `fixed..n` (nodes below `fixed`
    stay): a transposition and a cycle."""
    swap = list(range(n))
    swap[fixed], swap[fixed + 1] = fixed + 1, fixed
    cycle = list(range(fixed)) + list(range(fixed + 1, n)) + [fixed]
    return [swap, cycle]


@pytest.mark.parametrize("generator", ["adjacency", "laplacian"])
def test_the_detuned_dimer_matches_the_exact_rationals(generator):
    for g, (eta, tau, lost) in DIMER.items():
        r = dimer(generator, g).transport(0)
        assert set(r) == RESULT_KEYS
        assert r["efficiency"] == pytest.approx(float(eta), abs=1e-13)
        assert arrival(r) == pytest.approx(float(tau), rel=1e-11)
        assert r["lost"] == pytest.approx(float(lost), abs=1e-13)
        assert r["dark"] == 0.0
        assert r["dimension"] == 2
        assert r["added_dephasing"] == 0.0
        assert r["error"] <= 1e-13
        assert r["trapped"].shape == (1,)
        assert r["trapped"][0] == pytest.approx(float(eta), abs=1e-13)
        assert r["slowest_decay"] is not None and r["slowest_decay"] > 0
        assert r["condition"] >= 1.0
        names = [s["name"] for s in r["solves"]]
        assert names[0] == "schur" and "X" in names and "Y" in names
        for s in r["solves"]:
            assert set(s) == {"name", "iterations", "backward_error", "condition"}
            assert s["backward_error"] < 1e-12
    # The rounded values of the roadmap.
    rounded = [0.771278, 0.823024, 0.825951, 0.738538, 0.274598]
    got = [dimer(generator, g).transport(0)["efficiency"] for g in DIMER]
    assert np.round(got, 6).tolist() == rounded


def test_traps_and_losses_take_every_form():
    path = q.Graph.path(2)
    reference = dimer(rate=1.0).transport(0)
    for traps in ({1: 1.0}, [(1, 1.0)], ((1, 1),), [[1, 1.0]]):
        for losses in (0.05, [0.05, 0.05], np.full(2, 0.05)):
            w = q.DephasingWalk(
                path,
                "adjacency",
                rate=1.0,
                site_energies=np.array([3, 0]),
                traps=traps,
                losses=losses,
            )
            r = w.transport(0)
            assert r["efficiency"] == reference["efficiency"]
            assert r["arrival_time"] == reference["arrival_time"]
    w = dimer(rate=1.0)
    assert w.traps == [(1, 1.0)]
    assert w.losses.tolist() == [0.05, 0.05]
    assert w.site_energies.tolist() == [3.0, 0.0]
    assert w.node_dephasing.tolist() == [1.0, 1.0]
    plain = q.DephasingWalk(path)
    assert plain.traps == [] and plain.losses.tolist() == [0.0, 0.0]
    assert plain.site_energies.tolist() == [0.0, 0.0]
    # Per-node dephasing equal to the uniform rate is the same walk.
    per_node = q.DephasingWalk(
        path,
        "adjacency",
        site_energies=[3.0, 0.0],
        node_dephasing=[1.0, 1.0],
        traps={1: 1.0},
        losses=0.05,
    )
    # Key by key: pytest 9's approx refuses dicts with array or list values.
    r = per_node.transport(0)
    assert set(r) == set(reference)
    np.testing.assert_allclose(r["trapped"], reference["trapped"])
    for key in (
        "efficiency",
        "arrival_time",
        "lost",
        "dark",
        "added_dephasing",
        "slowest_decay",
        "condition",
        "error",
    ):
        assert r[key] == pytest.approx(reference[key]), key
    assert r["dimension"] == reference["dimension"]
    assert [s["name"] for s in r["solves"]] == [s["name"] for s in reference["solves"]]


def test_one_schur_form_serves_the_dimer_scan():
    rates = list(DIMER)
    s = dimer().dephasing_scan(0, rates)
    assert set(s) == {
        "rates",
        "efficiency",
        "arrival_time",
        "trapped",
        "lost",
        "dark",
        "condition",
        "error",
    }
    assert s["rates"].tolist() == rates
    assert s["trapped"].shape == (5, 1)
    for k, (eta, tau, lost) in enumerate(DIMER.values()):
        assert s["efficiency"][k] == pytest.approx(float(eta), abs=1e-13)
        assert s["arrival_time"][k] == pytest.approx(float(tau), rel=1e-11)
        assert s["lost"][k] == pytest.approx(float(lost), abs=1e-13)
        assert s["trapped"][k, 0] == pytest.approx(float(eta), abs=1e-13)
        assert s["dark"][k] == 0.0
        assert s["error"][k] <= 1e-13
    empty = dimer().dephasing_scan(0, [])
    assert empty["efficiency"].shape == (0,) and empty["trapped"].shape == (0, 1)


def test_the_best_rate_of_the_dimer_is_39_over_20():
    best = dimer().optimal_dephasing(0, 0.01, 100.0)
    assert set(best) == RESULT_KEYS | {"rate", "interior", "evaluations"}
    assert best["rate"] == pytest.approx(1.95, rel=1e-6)
    assert best["added_dephasing"] == best["rate"]
    assert best["interior"] is True
    assert best["evaluations"] > 13
    assert best["efficiency"] == pytest.approx(800 / 963, abs=1e-13)
    # η is flat at its maximum, so the rate is known to about 1e-7 and τ
    # moves with it at dτ/dγ ≈ 0.0142.
    slack = 1e-9 * 3320 / 963 + 0.015 * abs(best["rate"] - 1.95)
    assert abs(arrival(best) - 3320 / 963) <= slack
    # The defaults search [1e-3, 1e3].
    assert dimer().optimal_dephasing(0)["rate"] == pytest.approx(1.95, rel=1e-6)


def test_transport_from_a_density_matrix_is_linear():
    w = dimer(rate=1.0)
    rho = np.diag([1.0, 0.0]).astype(complex)
    assert w.transport_from(rho)["efficiency"] == pytest.approx(32000 / 38881, abs=1e-13)
    a, b = w.transport(0), w.transport(1)
    mixed = w.transport_from(np.eye(2) / 2)
    assert mixed["efficiency"] == pytest.approx(
        (a["efficiency"] + b["efficiency"]) / 2, abs=1e-13
    )
    assert mixed["lost"] == pytest.approx((a["lost"] + b["lost"]) / 2, abs=1e-13)


def test_complete_graphs_dense_and_on_orbitals():
    n = 64
    graph = q.Graph.complete(n)
    one = [0.0] * n
    one[2] = 0.8
    # (node dephasing, nodes fixed by the symmetries, η, τ, dark, dense dimension)
    cases = [
        (None, 2, Fraction(1, 63), Fraction(3971, 126), Fraction(62, 63), 2),
        (one, 3, Fraction(1, 62), Fraction(2320471, 36456), None, 3),
        ([1.0] * n, 2, Fraction(1), Fraction(1056), Fraction(0), n),
    ]
    for rates, fixed, eta, tau, dark, dimension in cases:
        dense = q.DephasingWalk(
            graph, "adjacency", node_dephasing=rates, traps=[(1, 1.0)]
        ).transport(0)
        orbital_walk = q.SymmetricOpenWalk(
            graph,
            "adjacency",
            symmetries=complete_symmetries(n, fixed),
            node_dephasing=rates,
            traps=[(1, 1.0)],
        )
        assert orbital_walk.num_orbitals == (10 if fixed == 2 else 17)
        assert orbital_walk.traps == [(1, 1.0)]
        orbital = orbital_walk.transport(0)
        assert dense["dimension"] == dimension
        assert orbital["dimension"] == orbital_walk.num_orbitals
        assert orbital["slowest_decay"] is None
        for r in (dense, orbital):
            assert r["efficiency"] == pytest.approx(float(eta), abs=1e-12)
            assert arrival(r) == pytest.approx(float(tau), rel=1e-10)
            if dark is not None:
                assert r["dark"] == pytest.approx(float(dark), abs=1e-12)


def test_dark_states_span_the_complement():
    n = 16
    w = q.DephasingWalk(q.Graph.complete(n), "adjacency", traps={1: 1.0})
    v = w.dark_states()
    assert v.shape == (n, n - 2)
    assert np.abs(v.T @ v - np.eye(n - 2)).max() <= 1e-12
    assert np.abs(v[1]).max() <= 1e-12
    dark = w.transport(0)["dark"]
    assert float(np.sum(v[0] ** 2)) == pytest.approx(dark, abs=1e-12)
    assert dark == pytest.approx((n - 2) / (n - 1), abs=1e-12)


def test_hypercube_q10_table():
    coherent = q.HypercubeOpenWalk(10, traps=[(10, 1.0)])
    assert coherent.traps == [(10, 1.0)]
    r = coherent.transport()
    assert r["efficiency"] == pytest.approx(1.0, abs=1e-11)
    assert arrival(r) == pytest.approx(5.8380146133, rel=1e-8)
    assert r["dimension"] == 286
    # Loss 0.05 and site dephasing γ: (γ, η, τ).
    table = [
        (0.0, 0.831038466686, 3.118369259),
        (0.1, 0.690879244942, 2.600291555),
        (1.0, 0.234277029669, 2.905175645),
        (10.0, 0.011425968764, 25.57861533),
    ]
    scan = q.HypercubeOpenWalk(10, traps={10: 1.0}, losses=0.05).dephasing_scan(
        [t[0] for t in table]
    )
    for k, (rate, eta, tau) in enumerate(table):
        r = q.HypercubeOpenWalk(10, site_rate=rate, traps={10: 1.0}, losses=0.05).transport()
        assert r["efficiency"] == pytest.approx(eta, abs=1e-11)
        assert arrival(r) == pytest.approx(tau, rel=1e-8)
        assert r["dark"] == 0.0
        assert scan["efficiency"][k] == pytest.approx(eta, abs=1e-11)
        assert scan["arrival_time"][k] == pytest.approx(tau, rel=1e-8)


def test_hypercube_dephasing_by_weight_equals_the_site_rate():
    # Q4, loss 0.05, dephasing 0.5 on every node (exact Liouville solve).
    a = q.HypercubeOpenWalk(4, site_rate=0.5, traps={4: 1.0}, losses=[0.05] * 5)
    b = q.HypercubeOpenWalk(4, node_dephasing=[0.5] * 5, traps={4: 1.0}, losses=0.05)
    for w in (a, b):
        r = w.transport()
        assert r["efficiency"] == pytest.approx(0.7103501369857, abs=1e-11)
        assert arrival(r) == pytest.approx(4.4093444882, rel=1e-10)
    best = a.optimal_dephasing(1e-3, 10.0)
    assert 1e-3 <= best["rate"] <= 10.0
    assert best["efficiency"] >= a.dephasing_scan([1e-3, 10.0])["efficiency"].max() - 1e-12


def test_traces_conserve_probability():
    times = np.linspace(0.0, 20.0, 41)
    t = dimer(rate=1.0).transport_trace(0, times)
    assert set(t) == {"times", "populations", "trapped", "lost"}
    assert t["times"].tolist() == times.tolist()
    assert t["populations"].shape == (41, 2) and t["trapped"].shape == (41, 1)
    total = t["populations"].sum(axis=1) + t["trapped"].sum(axis=1) + t["lost"]
    assert np.abs(total - 1.0).max() <= 1e-9
    assert np.all(np.diff(t["trapped"][:, 0]) >= 0) and np.all(np.diff(t["lost"]) >= 0)
    assert t["trapped"][-1, 0] <= 32000 / 38881
    # Q3 by weight on the coordinate orbitals and on the hypercube walk.
    d = 3
    eps = [0.3 * k - 0.1 * k * k for k in range(d + 1)]
    loss = [0.02 + 0.01 * k for k in range(d + 1)]
    weight = [bin(u).count("1") for u in range(2**d)]
    hyper = q.HypercubeOpenWalk(d, site_energies=eps, losses=loss, traps={d: 1.0})
    sym = q.SymmetricOpenWalk(
        q.Graph.hypercube(d),
        "adjacency",
        symmetries=q.coordinate_permutations(d),
        site_energies=[eps[k] for k in weight],
        losses=[loss[k] for k in weight],
        traps={2**d - 1: 1.0},
    )
    times = np.linspace(0.0, 5.0, 11)
    a, b = hyper.transport_trace(times), sym.transport_trace(0, times)
    assert a["populations"].shape == (11, d + 1) and b["populations"].shape == (11, 2**d)
    shells = [[u for u in range(2**d) if weight[u] == k] for k in range(d + 1)]
    by_weight = np.stack([b["populations"][:, s].sum(axis=1) for s in shells], axis=1)
    assert np.abs(a["populations"] - by_weight).max() <= 1e-10
    assert np.abs(a["trapped"] - b["trapped"]).max() <= 1e-10
    assert np.abs(a["lost"] - b["lost"]).max() <= 1e-10
    for r in (a, b):
        total = r["populations"].sum(axis=1) + r["trapped"].sum(axis=1) + r["lost"]
        assert np.abs(total - 1.0).max() <= 1e-10
    # The same Q3 walk on orbitals and as a hypercube (QuTiP on the full cube).
    for r in (hyper.transport(), sym.transport(0)):
        assert r["efficiency"] == pytest.approx(0.922920860248438, abs=1e-12)
        assert arrival(r) == pytest.approx(2.262737985267, rel=1e-11)


@pytest.mark.parametrize("generator", ["adjacency", "laplacian"])
def test_site_energies_of_the_continuous_walk_equal_the_driven_walk(generator):
    g = q.Graph.hypercube(4)
    eps = np.random.default_rng(163).uniform(-1.0, 1.0, 16)
    walk = q.ContinuousWalk(g, generator, 0.7, site_energies=eps)
    site_energies = walk.site_energies
    assert site_energies is not None and site_energies.tolist() == eps.tolist()
    driven = q.DrivenWalk(16, [(generator, g), ("potential", eps)])
    psi0 = np.zeros(16, complex)
    psi0[0] = 1.0
    exact = driven.evolve(psi0, 0.0, 1.3, lambda t: [0.7, 1.0], tol=1e-12)["state"]
    assert np.abs(walk.evolve(psi0, 1.3) - exact).max() <= 1e-9
    assert q.ContinuousWalk(g, generator, 0.7).site_energies is None


def test_the_lazy_spectrum_with_site_energies():
    walk = q.ContinuousWalk(q.Graph.path(2), "adjacency", site_energies=[3.0, 0.0])
    spectrum = walk.spectrum
    assert spectrum is not None
    root = np.sqrt(13.0)
    assert np.abs(spectrum - [(3 - root) / 2, (3 + root) / 2]).max() <= 1e-14
    # Rabi: from node 0 the walker reaches node 1 with probability 4/13 at most.
    t = np.pi / root
    assert walk.probabilities(0, t)[1] == pytest.approx(4 / 13, abs=1e-12)


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: q.DephasingWalk(q.Graph.cycle(4), rate=0.5, node_dephasing=[0.1] * 4),
            "give either rate or node_dephasing",
        ),
        (lambda: q.DephasingWalk(q.Graph.cycle(4), traps=[(0,)]), "each trap is a pair"),  # type: ignore[list-item]
        (lambda: q.DephasingWalk(q.Graph.cycle(4), traps=[1, 2]), "each trap is a pair"),  # type: ignore[list-item]
        (lambda: q.DephasingWalk(q.Graph.cycle(4), traps=7), "each trap is a pair"),  # type: ignore[arg-type]
        (lambda: q.DephasingWalk(q.Graph.cycle(4), traps={0: "x"}), "each trap is a pair"),  # type: ignore[dict-item]
        (lambda: q.DephasingWalk(q.Graph.cycle(4), traps=[(-1, 1.0)]), "each trap is a pair"),
        (lambda: q.SymmetricOpenWalk(q.Graph.cycle(4), traps=[(0, 1.0, 2.0)]), "each trap is a pair"),  # type: ignore[list-item]
        (lambda: q.HypercubeOpenWalk(3, traps="ab"), "each trap is a pair"),  # type: ignore[arg-type]
        (lambda: q.DephasingWalk(q.Graph.cycle(4), losses="abc"), "number or one value per site"),
        (lambda: q.DephasingWalk(q.Graph.cycle(4), losses=[0.1] * 3), "dimension"),
        (lambda: q.HypercubeOpenWalk(3, losses=[0.1] * 3), "dimension"),
        # A single loss is expanded only after d is checked (this allocated
        # 8 GB and aborted).
        (lambda: q.HypercubeOpenWalk(10**9, losses=0.1), "d ≤ 230"),
        (lambda: q.HypercubeOpenWalk(2**64 - 1, losses=0.1), "d ≤ 230"),
        (
            lambda: q.HypercubeOpenWalk(62, site_rate=1.0, traps={62: 1.0}).transport(),
            r"needs 138290880 band entries \(limit 134217728\)",
        ),
    ],
)
def test_binding_errors_name_the_problem(call, message):
    with pytest.raises(ValueError, match=message):
        call()
