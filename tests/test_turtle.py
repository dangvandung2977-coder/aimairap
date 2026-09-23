"""Turtles: observable windows, determinism, beatable; mirror symmetry."""
import numpy as np

from rl.opponents.scripted import create
from sandbox.simulation.sim import PvPSim, idle_ctrl

NAMES = ["turtleA", "turtleB", "turtleC"]


def test_turtles_hold_and_open():
    for name in ["turtleA", "turtleB", "turtleC"]:
        foe = create(name, seed=81)
        sim = PvPSim(seed=81)
        held = opened = 0
        for _ in range(400):
            c = foe.act(sim, 1)
            sim.step(idle_ctrl(), c)
            if sim.done:
                break
        # phase trace via fresh foe: count hold vs open ticks
        foe2 = create(name, seed=81)
        sim2 = PvPSim(seed=81)
        for _ in range(400):
            c = foe2.act(sim2, 1)
            sim2.step(idle_ctrl(), c)
            if foe2._phase == "hold":
                held += 1
            else:
                opened += 1
            if sim2.done:
                break
        assert held > opened > 0, (name, held, opened)  # long holds, real windows


def test_turtles_deterministic():
    def run(name):
        foe = create(name, seed=83)
        sim = PvPSim(seed=83)
        for _ in range(80):
            sim.step(idle_ctrl(), foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    for name in NAMES:
        p1, h1 = run(name)
        p2, h2 = run(name)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2


def test_turtle_windows_beatable_by_patient_attacker():
    """Attack only during open windows: must deal damage (legit openings)."""
    for name in ["turtleA", "turtleB", "turtleC"]:
        foe = create(name, seed=9)
        sim = PvPSim(seed=9)
        sim.players[0].pos[:] = [10, 0, 10]
        sim.players[1].pos[:] = [12, 0, 10]
        sim.players[1].yaw = 180.0
        for _ in range(500):
            c = foe.act(sim, 1)
            atk = dict(idle_ctrl())
            # patient attacker: swing only when foe shield is down
            foe_up = sim.players[1].shield_up
            atk["attack"] = not foe_up
            sim.step(atk, c)
            if sim.done:
                break
        assert sim.damage_dealt[0] > 0, name


def test_circlers_demand_matching_direction():
    """Identical constant local-left strafing tracks the two circlers with
    different quality: foe motion direction determines which strafe side
    holds range and lands accurately (positioning/hit-rate asymmetry)."""
    def run(foe_name: str):
        foe = create(foe_name, seed=9)
        sim = PvPSim(seed=9)
        dists = []
        for _ in range(400):
            p0, p1 = sim.players[0], sim.players[1]
            c = dict(idle_ctrl())
            c["move_x"] = -1.0  # fixed local-left strafe
            c["move_z"] = 0.5
            c["attack"] = True
            d = p1.pos - p0.pos
            want = float(np.degrees(np.arctan2(d[2], d[0])))
            c["yaw_delta"] = float(np.clip((want - p0.yaw + 180) % 360 - 180,
                                           -30.0, 30.0))
            sim.step(c, foe.act(sim, 1))
            dists.append(sim.distance())
            if sim.done:
                break
        att = max(1, sim.attempts[0])
        return (round(sim.hits[0] / att, 3), round(float(np.mean(dists)), 2))
    l, r = run("circlerL"), run("circlerR")
    assert l != r, (l, r)  # same movement, different tracking quality


def test_mirror_swaps_sides():
    s1 = PvPSim(seed=3)
    s1.reset(seed=3, mirror=False)
    s2 = PvPSim(seed=3)
    s2.reset(seed=3, mirror=True)
    np.testing.assert_allclose(s1.players[0].pos, s2.players[1].pos)
    np.testing.assert_allclose(s1.players[1].pos, s2.players[0].pos)
    assert s1.players[0].yaw == 0.0 and s2.players[0].yaw == 180.0
    # mirrored rollout is deterministic too
    for _ in range(50):
        c = dict(idle_ctrl())
        c["move_z"] = 1.0
        s2.step(c, idle_ctrl())
    s3 = PvPSim(seed=3)
    s3.reset(seed=3, mirror=True)
    for _ in range(50):
        c = dict(idle_ctrl())
        c["move_z"] = 1.0
        s3.step(c, idle_ctrl())
    np.testing.assert_array_equal(s2.players[0].pos, s3.players[0].pos)
