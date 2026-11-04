"""Interacting walkers by minimal images (1.58)."""

import itertools

import numpy as np
import pytest

import metatron_qso as q


def test_equals_the_union_find_orbits_of_1_43() -> None:
    g = q.Graph.paley(13)
    ours = q.GroupInteractingWalk(g, [0, 0], statistics="bosons", on_site=0.5)
    theirs = q.SymmetricInteractingWalk.from_start(g, [0, 0], statistics="bosons", on_site=0.5)
    assert ours.num_orbits == theirs.num_orbits
    x = ours.evolve(ours.state_of([0, 0]), 0.9)
    y = theirs.evolve(theirs.state_of([0, 0]), 0.9)
    for sites in itertools.combinations_with_replacement(range(13), 2):
        assert ours.probability_of(x, list(sites)) == pytest.approx(
            theirs.probability_of(y, list(sites)), abs=1e-12
        )
    assert ours.densities(x).sum() == pytest.approx(2.0)


def test_whole_group_orbit_states() -> None:
    g = q.Graph.paley(101)
    w = q.GroupInteractingWalk(g, [0, 1, 3], whole_group=True, statistics="fermions")
    assert w.log_group_order == pytest.approx(np.log(101 * 50))
    # About C(101, 3) / |G| orbits.
    assert 10 < w.num_orbits < 60
    x = w.evolve(w.state_of([0, 1, 3]), 1.0)
    assert np.sum(np.abs(x) ** 2) == pytest.approx(1.0)
    sizes = sum(w.orbit_size(o) for o in range(w.num_orbits))
    assert sizes <= np.exp(w.log_num_configurations) + 0.5
    with pytest.raises(ValueError):
        q.GroupInteractingWalk(g, [0, 1, 2], whole_group=True, statistics="fermions")
