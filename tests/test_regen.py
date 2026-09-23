"""Regen mechanics: deterministic, gated-off default, counterfactual gate."""
import numpy as np

from sandbox.simulation.sim import PvPSim, idle_ctrl


def test_regen_off_by_default():
    sim = PvPSim(seed=0)
    assert sim.foe_regen_rate == 0.0
    sim.players[1].health = 10.0
    for _ in range(200):
        sim.step(idle_ctrl(), idle_ctrl())
    assert sim.players[1].health == 10.0


def test_regen_heals_when_quiet():
    sim = PvPSim(seed=0)
    sim.foe_regen_rate = 0.05
    sim.foe_regen_delay = 60
    sim.players[1].health = 10.0
    for _ in range(200):
        sim.step(idle_ctrl(), idle_ctrl())
    assert sim.players[1].health > 10.0


def test_regen_pauses_after_damage():
    sim = PvPSim(seed=0)
    sim.foe_regen_rate = 1.0
    sim.foe_regen_delay = 100
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [12, 0, 10]
    sim.players[0].yaw = 0.0
    atk = dict(idle_ctrl())
    atk["attack"] = True
    sim.step(atk, idle_ctrl())  # hit lands, timer resets
    hp_after_hit = sim.players[1].health
    for _ in range(50):
        sim.step(idle_ctrl(), idle_ctrl())
    assert sim.players[1].health == hp_after_hit  # still in delay window


def test_regen_deterministic():
    def run():
        sim = PvPSim(seed=4)
        sim.foe_regen_rate = 0.05
        sim.foe_regen_delay = 60
        sim.players[1].health = 12.0
        for _ in range(150):
            sim.step(idle_ctrl(), idle_ctrl())
        return sim.players[1].health
    assert run() == run()
