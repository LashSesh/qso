import pytest

import metatron_qso as q


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def test_paths_and_cycles_follow_the_theorems():
    for n in range(2, 60):
        m = n + 1
        want = is_prime(m) or (m % 2 == 0 and is_prime(m // 2)) or (m & (m - 1)) == 0
        assert q.PrettyGoodTransfer.path(n, 0, n - 1).pretty_good == want
        assert q.PrettyGoodTransfer.path(n, 0, n - 1, "laplacian").pretty_good == ((n & (n - 1)) == 0)
    for n in range(4, 40, 2):
        assert q.PrettyGoodTransfer.cycle(n, 0, n // 2).pretty_good == ((n & (n - 1)) == 0)


def test_decisions_times_and_witnesses():
    p4 = q.PrettyGoodTransfer.path(4, 0, 3)
    assert p4.pretty_good and p4.perfect_time is None and p4.obstruction is None
    best = p4.best_transfer(2000.0, 200_000)
    assert best["fidelity"] > 0.999
    p8 = q.PrettyGoodTransfer.path(8, 0, 7)
    assert p8.strongly_cospectral and not p8.pretty_good and p8.obstruction is not None
    cube = q.PrettyGoodTransfer.cayley([2, 2, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0], [1, 1, 1])
    assert cube.pretty_good and cube.rank == 1
    assert cube.perfect_time == pytest.approx(3.141592653589793 / 2)
    assert q.PrettyGoodTransfer.from_graph(q.Graph.path(6), 0, 5).pretty_good
    with pytest.raises(ValueError):
        q.PrettyGoodTransfer.from_graph(q.Graph.hypercube(3), 0, 7)
    with pytest.raises(ValueError):
        q.PrettyGoodTransfer.path(4, 1, 1)
    with pytest.raises(ValueError):
        q.PrettyGoodTransfer.cayley([6], [[1]], [0], [3])
    with pytest.raises(ValueError):
        q.PrettyGoodTransfer.path(4, 0, 3, "hamming")
