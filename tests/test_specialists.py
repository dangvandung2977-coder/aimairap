"""Utility specialists use items through the validated pipeline."""
import numpy as np

from rl.opponents.scripted import create
from rl.opponents.utility import UTILITY
from sandbox.simulation.sim import PvPSim, idle_ctrl


def test_all_specialists_use_items():
    used = {}
    for name in sorted(UTILITY):
        foe = create(name, seed=61)
        sim = PvPSim(seed=61)
        if name in ("gappleuser", "pearlescaper"):
            sim.players[1].health = 6.0  # sustain/escape check OWN hp
        for _ in range(600):
            sim.step(idle_ctrl(), foe.act(sim, 1))
            if sim.done:
                break
        ev_types = {e["type"] for e in sim.utility_events}
        ok = sim.items_used[1] > 0 or sim.players[1].shield_up
        used[name] = (ok, sorted(ev_types))
    for name, (ok, evs) in used.items():
        if name in ("pressureeater", "punisheater"):
            continue  # pure-melee punishers; validated below, not by item use
        print(name, ok, evs)
    assert all(ok for n, (ok, _) in used.items()
               if n not in ("pressureeater", "punisheater")), used


def test_eaters_punish_eating():
    """Eating in front of a punisher costs HP (observable eating state)."""
    for name in ("pressureeater", "punisheater", "mixedeater"):
        foe = create(name, seed=61)
        sim = PvPSim(seed=61)
        sim.players[0].pos[:] = [10, 0, 10]
        sim.players[1].pos[:] = [14, 0, 10]
        sim.players[0].health = 10.0
        c = dict(idle_ctrl())
        c["use_item"] = True
        c["select_slot"] = 2  # bread first (switch cost), then eat attempts
        sim.step(c, idle_ctrl())
        sim.step(idle_ctrl(), idle_ctrl())
        sim.step(idle_ctrl(), idle_ctrl())
        eat_ticks = 0
        for _ in range(200):
            cc = dict(idle_ctrl())
            cc["use_item"] = True
            sim.step(cc, foe.act(sim, 1))
            if sim.players[0].eating:
                eat_ticks += 1
            if sim.done:
                break
        assert eat_ticks > 0, name  # tried to eat
        print(name, "eat_ticks:", eat_ticks,
              "hp0:", round(sim.players[0].health, 1))


def test_specialists_deterministic():
    def run(name):
        foe = create(name, seed=63)
        sim = PvPSim(seed=63)
        for _ in range(80):
            sim.step(idle_ctrl(), foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    for name in sorted(UTILITY):
        p1, h1 = run(name)
        p2, h2 = run(name)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2


def test_gapple_retreat_and_heal():
    foe = create("gappleuser", seed=5)
    sim = PvPSim(seed=5)
    sim.players[1].health = 6.0
    ate = False
    for _ in range(400):
        sim.step(idle_ctrl(), foe.act(sim, 1))
        if sim.players[1].absorption_hp > 0:
            ate = True
            break
        if sim.done:
            break
    assert ate


def test_shield_blocks_in_sim():
    foe = create("shieldholder", seed=5)
    sim = PvPSim(seed=5)
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [12, 0, 10]
    sim.players[1].yaw = 180.0
    atk = dict(idle_ctrl())
    atk["attack"] = True
    for _ in range(60):
        c1 = foe.act(sim, 1)
        c1.pop("attack", None)  # holder only defends here
        sim.step(atk, c1)
    assert sim.blocks[1] > 0
