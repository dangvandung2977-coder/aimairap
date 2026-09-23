"""Stage 4A bow infrastructure: mechanics + scripted differential + determinism.

Fast, framework-light checks (matches the repo's existing test style). These
guard the *environment affordance* the bow curriculum depends on -- not policy
quality (that is the trained-checkpoint gate in tools/stage4a_gate.py)."""
from __future__ import annotations

import numpy as np

from sandbox.config import CombatConfig, PhysicsConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from sandbox.items.defs import HOTBAR
from rl.opponents.scripted import create as make_opp
from tools import bow_scenario as BS

_BOW_SLOT = HOTBAR.index("bow")


def _sim(seed=0, arena=40, arrows=16):
    sim = PvPSim(WorldConfig(size=arena, wall_margin=1.0, obstacles=()),
                 PhysicsConfig(),
                 CombatConfig(landing_attack_delay_ticks=4),
                 max_ticks=900, seed=seed,
                 loadout={"sword": 1, "bow": 1, "arrow": arrows},
                 enabled_items={"sword", "bow"})
    sim.reset(seed)
    return sim


def test_ranged_opponents_registered():
    for name in ("bowfighter", "rangedrunner"):
        assert make_opp(name, seed=1) is not None


def test_bow_fires_and_consumes_arrow():
    sim = _sim(seed=1, arrows=3)
    inv = sim.inventories[0]
    inv.selected = _BOW_SLOT
    before = inv.counts["arrow"]
    n_proj = len(sim.projectiles)
    ctrl = idle_ctrl()
    ctrl["use_item"] = True
    ctrl["select_slot"] = _BOW_SLOT
    sim.step(ctrl, idle_ctrl())
    assert inv.counts["arrow"] == before - 1, "arrow not consumed"
    assert len(sim.projectiles) == n_proj + 1, "no arrow projectile spawned"


def test_bow_out_of_ammo_is_noop():
    sim = _sim(seed=1, arrows=0)
    sim.inventories[0].selected = _BOW_SLOT
    ctrl = idle_ctrl()
    ctrl["use_item"] = True
    ctrl["select_slot"] = _BOW_SLOT
    sim.step(ctrl, idle_ctrl())
    assert len(sim.projectiles) == 0, "fired with no arrows"


def test_scripted_bow_lands_hits_on_kiter():
    row = BS.run_ep(BS.bow_shooter, BS.BOW_LOADOUT, "bowfighter", seed=7000,
                    arena=40, foe_arrows=32, start_dist=16.0)
    assert row["arrow_hits"] >= 1, f"scripted shooter never hit: {row}"


def test_bow_scenario_shows_advantage_and_is_deterministic():
    def delta(seed0, n=6):
        bow = [BS.run_ep(BS.bow_shooter, BS.BOW_LOADOUT, "bowfighter",
                         seed0 + i, 40, foe_arrows=32, start_dist=16.0)
               for i in range(n)]
        mel = [BS.run_ep(BS.melee_chaser, BS.MELEE_LOADOUT, "bowfighter",
                         seed0 + i, 40, foe_arrows=32, start_dist=16.0)
               for i in range(n)]
        return (BS._summ(bow), BS._summ(mel))
    bow, mel = delta(7000)
    # bow must be a real advantage: more wins OR much less damage taken
    assert (bow["win_rate"] - mel["win_rate"] > 0.25
            or bow["taken"] - mel["taken"] < -6.0), (bow, mel)
    # deterministic: identical seeds reproduce identical summaries
    bow2, mel2 = delta(7000)
    assert bow == bow2 and mel == mel2, "bow scenario not deterministic"
