import numpy as np

from sandbox.simulation.sim import PvPSim


def _run(seed: int, n: int = 120):
    sim = PvPSim(seed=seed)
    rng = np.random.default_rng(999)
    for _ in range(n):
        c0 = {"move_x": 0.0, "move_z": 1.0, "sprint": False, "jump": False,
              "attack": bool(rng.integers(2)), "yaw_delta": float(rng.uniform(-10, 10)),
              "pitch_delta": 0.0}
        c1 = {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
              "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}
        sim.step(c0, c1)
    return sim


def test_same_seed_same_trajectory():
    a, b = _run(42), _run(42)
    np.testing.assert_allclose(a.players[0].pos, b.players[0].pos)
    np.testing.assert_allclose(a.players[1].vel, b.players[1].vel)
    assert a.players[0].health == b.players[0].health
    assert a.tick == b.tick


def test_different_seeds_differ():
    a, b = _run(1), _run(2)
    assert not np.allclose(a.players[0].pos, b.players[0].pos)


def test_reset_reproducible():
    sim = PvPSim(seed=7)
    p0 = sim.players[0].pos.copy()
    sim.reset(seed=7)
    np.testing.assert_allclose(p0, sim.players[0].pos)
