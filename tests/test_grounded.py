"""Phase 2.75: grounded opponents — deterministic, distinct, beatable."""
import numpy as np

from rl.opponents.grounded import GROUNDED
from rl.opponents.scripted import create
from sandbox.simulation.sim import PvPSim

NAMES = sorted(GROUNDED)


def _idle():
    return {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
            "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}


def test_all_grounded_engage_and_differ():
    ttks = {}
    for name in NAMES:
        foe = create(name, seed=41)
        sim = PvPSim(seed=41)
        for _ in range(500):
            sim.step(_idle(), foe.act(sim, 1))
            if sim.done:
                break
        ttks[name] = sim.tick
        assert sim.damage_dealt[1] > 0, name  # every grounded lands hits
    assert len(set(ttks.values())) > 1, ttks  # distinct behavior/pace


def test_grounded_deterministic():
    def run(name):
        foe = create(name, seed=43)
        sim = PvPSim(seed=43)
        for _ in range(80):
            sim.step(_idle(), foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    for name in NAMES:
        p1, h1 = run(name)
        p2, h2 = run(name)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2


def test_grounded_stay_grounded():
    """Exemplars practice what they preach: ~no jumping."""
    for name in NAMES:
        foe = create(name, seed=7)
        sim = PvPSim(seed=7)
        air = 0
        for _ in range(300):
            c = foe.act(sim, 1)
            if c["jump"]:
                air += 1
            sim.step(_idle(), c)
            if sim.done:
                break
        assert air <= 15, (name, air)


def test_antiair_only_commits_to_predictable_air():
    from rl.opponents.grounded import GroundAntiAir
    foe = GroundAntiAir(seed=0)
    sim = PvPSim(seed=0)
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [13, 0, 10]
    # single airborne blip must NOT trigger commitment
    sim.players[0].on_ground = False
    for _ in range(3):
        foe.act(sim, 1)
    assert not foe._committed
