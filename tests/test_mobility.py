"""Mobility opponents + terrain layouts: deterministic, imperfect, fair."""
import numpy as np

from rl.opponents.scripted import create
from sandbox.simulation.sim import PvPSim, idle_ctrl

NAMES = ["farfighter", "retreatfighter", "knockbackbully", "chasebait"]


def test_mobility_deterministic():
    def run(name):
        foe = create(name, seed=91)
        sim = PvPSim(seed=91)
        for _ in range(80):
            sim.step(idle_ctrl(), foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    for name in NAMES:
        p1, h1 = run(name)
        p2, h2 = run(name)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2


def test_retreat_creates_distance():
    foe = create("retreatfighter", seed=3)
    sim = PvPSim(seed=3)
    sim.players[1].health = 6.0  # fleeing
    d0 = sim.distance()
    for _ in range(60):
        sim.step(idle_ctrl(), foe.act(sim, 1))
        if sim.done:
            break
    assert sim.distance() > d0  # genuinely disengages


def test_chasebait_flees_after_damage():
    foe = create("chasebait", seed=3)
    sim = PvPSim(seed=3)
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [12, 0, 10]
    sim.players[0].yaw = 0.0
    atk = dict(idle_ctrl())
    atk["attack"] = True
    d0 = sim.distance()
    for _ in range(80):
        sim.step(atk, foe.act(sim, 1))
        if sim.done:
            break
    assert sim.distance() > d0 or sim.done  # fled or died fighting


def test_obstacle_layout_swap():
    from rl.environment.gym_env import PvPEnv
    from sandbox.config import WorldConfig
    env = PvPEnv(opponent="dummy",
                 world_cfg=WorldConfig(size=20, wall_margin=1.0, obstacles=()))
    env.reset(seed=0)
    assert not env.sim.world.obstacles
    wall = [(9.5, 4.0, 10.5, 16.0, 3.0)]
    env.set_obstacles(wall)
    assert len(env.sim.world.obstacles) == 1
    assert env.sim.world.is_solid_at(10.0, 1.0, 10.0)
    env.reset(seed=0)
    assert len(env.sim.world.obstacles) == 1  # persists across resets
    env.set_obstacles([])
    assert not env.sim.world.obstacles
