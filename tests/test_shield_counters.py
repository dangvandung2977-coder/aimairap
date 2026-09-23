"""Shield-counter opponents: deterministic, imperfect, beatable."""
import numpy as np

from rl.opponents.shield_counter import SHIELD_COUNTERS
from rl.opponents.scripted import create
from sandbox.simulation.sim import PvPSim, idle_ctrl

NAMES = sorted(SHIELD_COUNTERS)


def test_all_shield_counters_block_and_differ():
    info = {}
    for name in NAMES:
        foe = create(name, seed=71)
        sim = PvPSim(seed=71)
        atk = dict(idle_ctrl())
        atk["attack"] = True
        for _ in range(300):
            sim.step(atk, foe.act(sim, 1))
            if sim.done:
                break
        info[name] = (sim.blocks[1], round(float(sim.damage_dealt[1]), 1),
                      sim.tick)
    for name, (blocks, dealt, ticks) in info.items():
        print(name, "blocks_by_foe=", blocks, "foe_dealt=", dealt)
    # turtles/baiters/strafer/pressure/mixed must actually raise shield
    assert info["shieldturtle"][0] > 0
    assert info["shieldstrafer"][0] > 0
    assert info["shieldpressure"][0] > 0
    assert info["shieldmixed"][0] > 0


def test_shield_counters_deterministic():
    def run(name):
        foe = create(name, seed=73)
        sim = PvPSim(seed=73)
        for _ in range(80):
            sim.step(idle_ctrl(), foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    for name in NAMES:
        p1, h1 = run(name)
        p2, h2 = run(name)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2


def test_baiter_exposes_windows():
    from rl.opponents.shield_counter import ShieldBaiter
    foe = ShieldBaiter(seed=0)
    sim = PvPSim(seed=0)
    swung = 0
    for _ in range(200):
        c = foe.act(sim, 1)
        sim.step(idle_ctrl(), c)
        if c.get("attack"):
            swung += 1
        if sim.done:
            break
    assert swung > 0  # baiter drops shield and swings


def test_counters_beatable_by_flank():
    """A flanking attacker (teleport behind each tick) must deal damage:
    proves counters are not perfect walls."""
    for name in NAMES:
        foe = create(name, seed=9)
        sim = PvPSim(seed=9)
        for _ in range(120):
            p0, p1 = sim.players[0], sim.players[1]
            # stand behind the foe, facing it
            import math
            yr = math.radians(p1.yaw)
            p0.pos[:] = [p1.pos[0] - math.cos(yr) * 2.0, 0, p1.pos[2] - math.sin(yr) * 2.0]
            d = p1.pos - p0.pos
            p0.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
            atk = dict(idle_ctrl())
            atk["attack"] = True
            sim.step(atk, foe.act(sim, 1))
            if sim.done:
                break
        assert sim.damage_dealt[0] > 0, name
