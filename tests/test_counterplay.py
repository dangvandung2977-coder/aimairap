"""Phase 2.5: counterplay opponents — deterministic, distinct, beatable."""
import numpy as np

from rl.opponents.counterplay import COUNTERS
from rl.opponents.scripted import create
from sandbox.simulation.sim import PvPSim

NAMES = sorted(COUNTERS)


def _idle():
    return {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
            "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}


def test_all_counters_engage_and_differ():
    dealt = {}
    for name in NAMES:
        foe = create(name, seed=21)
        sim = PvPSim(seed=21)
        for _ in range(400):
            sim.step(_idle(), foe.act(sim, 1))
            if sim.done:
                break
        dealt[name] = round(float(sim.damage_dealt[1]), 1)
        assert sim.damage_dealt[1] > 0, name  # every counter lands hits
    assert len(set(dealt.values())) > 1, dealt  # distinct behavior


def test_counters_deterministic():
    def run(name):
        foe = create(name, seed=33)
        sim = PvPSim(seed=33)
        for _ in range(80):
            sim.step(_idle(), foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    for name in NAMES:
        p1, h1 = run(name)
        p2, h2 = run(name)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2


def test_landing_punisher_holds_vs_air():
    from rl.opponents.counterplay import LandingPunisher
    foe = LandingPunisher(seed=0)
    sim = PvPSim(seed=0)
    # hold foe airborne above punisher: no blind swings at air
    sim.players[0].pos[:] = [10, 3, 10]
    sim.players[0].on_ground = False
    sim.players[1].pos[:] = [10, 0, 10]
    swings = 0
    for _ in range(12):
        c = foe.act(sim, 1)
        sim.step(_idle(), c)
        # pin target airborne: hold branch must never swing at air
        sim.players[0].pos[1] = 3.0
        sim.players[0].vel[1] = 0.0
        sim.players[0].on_ground = False
        if c["attack"]:
            swings += 1
    assert swings == 0  # holds while target airborne and close
    # release: landing must trigger pressure attacks
    swung = False
    for _ in range(30):
        c = foe.act(sim, 1)
        sim.step(_idle(), c)
        if c["attack"]:
            swung = True
    assert swung


def test_miss_punisher_counters_whiff():
    from rl.opponents.counterplay import MissPunisher
    foe = MissPunisher(seed=0)
    sim = PvPSim(seed=0)
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [12.5, 0, 10]
    sim.players[0].yaw = 180.0  # P0 faces AWAY: its attacks whiff
    whiff = dict(_idle())
    whiff["attack"] = True
    countered = False
    for _ in range(40):
        c = foe.act(sim, 1)
        sim.step(whiff, c)
        if foe._counter_t > 0:
            countered = True
    assert countered  # whiff detected -> counter window opened
