"""Stage 4A scripted differential: is the bow genuinely advantageous?

Methodology (matches the project's pearl/sustain gates): PROVE the scenario
affords a real outcome differential with SCRIPTED agents BEFORE any PPO. No
reward / obs / action / mechanic changes -- pure measurement on the existing
simulator.

Scenario: big open arena, a pure kiter foe (rangedrunner) that sprints away
and never fights. Two scripted agents, paired seeds:
  - MELEE chaser (sword only): equal top speed -> cannot corner the kiter ->
    little/no damage, timeout draws.
  - BOW shooter (bow + arrows): stays at range, leads the target, and lands
    arrows -> real damage, real kills.

A large positive (bow - melee) damage/win delta proves aiming/leading is
economically meaningful through outcomes, not reward shaping.

  python -m tools.bow_scenario --episodes 20 --out logs/bow_scenario.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from sandbox.config import CombatConfig, PhysicsConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from sandbox.items.defs import HOTBAR
from rl.opponents.scripted import create as make_opp
from rl.opponents.ranged import _aim_at, _yaw_error

_BOW_SLOT = HOTBAR.index("bow")

BOW_LOADOUT = {"sword": 1, "shield": 0, "bread": 0, "golden_apple": 0,
               "ender_pearl": 0, "cobweb": 0, "block": 0,
               "strength_potion": 0, "bow": 1, "arrow": 16}
MELEE_LOADOUT = dict(BOW_LOADOUT, bow=1, arrow=0)  # identical except no ammo
ENABLED = {"sword", "bow"}


def melee_chaser(sim, idx: int) -> dict:
    """Sword-only: sprint at the foe, swing in range."""
    me, foe = sim.players[idx], sim.players[1 - idx]
    c = idle_ctrl()
    dd = foe.pos - me.pos
    want = math.degrees(math.atan2(dd[2], dd[0]))
    c["yaw_delta"] = float(np.clip((want - me.yaw + 180.0) % 360.0 - 180.0, -30, 30))
    d = float(np.linalg.norm(dd[[0, 2]]))
    if d > 2.8:
        c["move_z"] = 1.0
        c["sprint"] = True
    else:
        c["attack"] = me.attack_cooldown <= 0
    return c


def bow_shooter(sim, idx: int) -> dict:
    """Bow: hold mid range, lead the target, release when aimed and ready."""
    me, foe = sim.players[idx], sim.players[1 - idx]
    inv = sim.inventories[idx]
    c = idle_ctrl()
    c.update(_aim_at(sim, idx, lead=True, elev=6.0))
    if inv.selected != _BOW_SLOT:
        c["select_slot"] = _BOW_SLOT
    d = float(np.linalg.norm((foe.pos - me.pos)[[0, 2]]))
    # keep a shooting pocket: chase if the runner opens too far, else hold
    if d > 12.0:
        c["move_z"] = 1.0
        c["sprint"] = True
    elif d < 5.0:
        c["move_z"] = -0.6
    aimed = abs(_yaw_error(sim, idx)) < 10.0
    ready = inv.cooldowns.get("bow", 0) <= 0 and inv.switch_timer <= 0
    if aimed and ready and inv.selected == _BOW_SLOT and 3.0 < d < 20.0:
        c["use_item"] = True
    return c


def run_ep(agent_fn, loadout: dict, opp_name: str, seed: int,
           arena: int = 40, max_ticks: int = 900, foe_arrows: int = 0,
           start_dist: float = 16.0) -> dict:
    wc = WorldConfig(size=arena, wall_margin=1.0, obstacles=())
    cc = CombatConfig(attack_arc_deg=75.0, attack_arc_vertical_deg=75.0,
                      crit_mult=1.5, landing_attack_delay_ticks=4)
    sim = PvPSim(wc, PhysicsConfig(), cc, max_ticks=max_ticks, seed=seed,
                 loadout=loadout, enabled_items=set(ENABLED))
    sim.reset(seed)
    # foe gets its own ammo (independent of the agent's arrow count) so the
    # counterfactual varies ONLY the agent's bow access, never the opponent.
    sim.inventories[1].counts["arrow"] = foe_arrows
    sim.inventories[1].enabled = set(ENABLED)
    # fixed starting separation (agent west, foe east), centered
    cx = arena * 0.5
    sim.players[0].pos[:] = (cx - start_dist / 2, 0.0, cx)
    sim.players[1].pos[:] = (cx + start_dist / 2, 0.0, cx)
    sim.players[0].yaw, sim.players[1].yaw = 0.0, 180.0
    for i in (0, 1):
        sim.prev_pos[i] = [sim.players[i].pos.copy() for _ in range(4)]
    opp = make_opp(opp_name, seed=seed)
    opp.reset(seed)
    shots = hits0 = 0
    while not sim.done:
        c0 = agent_fn(sim, 0)
        c1 = opp.act(sim, 1)
        pearls_before = sim.inventories[0].counts.get("arrow", 0)
        sim.step(c0, c1)
        if sim.inventories[0].counts.get("arrow", 0) < pearls_before:
            shots += 1
    return {
        "seed": seed,
        "win": 1 if sim.winner == 0 else 0,
        "ticks": sim.tick,
        "dealt": round(float(sim.damage_dealt[0]), 1),
        "taken": round(float(sim.damage_dealt[1]), 1),
        "foe_hp": round(float(sim.players[1].health), 1),
        "shots": shots,
        "arrow_hits": int(sim.hits[0]),
    }


def _summ(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "episodes": n,
        "win_rate": round(sum(r["win"] for r in rows) / n, 3),
        "dealt": round(float(np.mean([r["dealt"] for r in rows])), 2),
        "taken": round(float(np.mean([r["taken"] for r in rows])), 2),
        "foe_hp_left": round(float(np.mean([r["foe_hp"] for r in rows])), 2),
        "shots_per_ep": round(float(np.mean([r["shots"] for r in rows])), 2),
        "hits_per_ep": round(float(np.mean([r["arrow_hits"] for r in rows])), 2),
        "mean_ticks": round(float(np.mean([r["ticks"] for r in rows])), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--opp", default="bowfighter")
    ap.add_argument("--arena", type=int, default=40)
    ap.add_argument("--foe-arrows", type=int, default=32)
    ap.add_argument("--start-dist", type=float, default=16.0)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--out", default="logs/bow_scenario.json")
    a = ap.parse_args()
    bow_rows, mel_rows = [], []
    for ep in range(a.episodes):
        seed = a.seed0 + ep
        bow_rows.append(run_ep(bow_shooter, BOW_LOADOUT, a.opp, seed, a.arena,
                               foe_arrows=a.foe_arrows, start_dist=a.start_dist))
        mel_rows.append(run_ep(melee_chaser, MELEE_LOADOUT, a.opp, seed, a.arena,
                               foe_arrows=a.foe_arrows, start_dist=a.start_dist))
    bow, mel = _summ(bow_rows), _summ(mel_rows)
    res = {"scenario": {"opp": a.opp, "arena": a.arena, "enabled": sorted(ENABLED),
                        "foe_arrows": a.foe_arrows, "start_dist": a.start_dist},
           "bow": {"summary": bow, "episodes_detail": bow_rows},
           "melee_only": {"summary": mel, "episodes_detail": mel_rows},
           "delta": {"win": round(bow["win_rate"] - mel["win_rate"], 3),
                     "dealt": round(bow["dealt"] - mel["dealt"], 2),
                     "taken": round(bow["taken"] - mel["taken"], 2)}}
    print(f"BOW   win={bow['win_rate']} dealt={bow['dealt']} taken={bow['taken']} "
          f"shots={bow['shots_per_ep']} hits={bow['hits_per_ep']}")
    print(f"MELEE win={mel['win_rate']} dealt={mel['dealt']} taken={mel['taken']}")
    print(f"DELTA win={res['delta']['win']:+} dealt={res['delta']['dealt']:+} "
          f"taken={res['delta']['taken']:+}")
    # bow is advantageous if it wins more OR takes materially less damage
    verdict = "ADVANTAGE" if (res["delta"]["win"] > 0.25
                              or res["delta"]["taken"] < -6.0) else "NO_DIFFERENTIAL"
    res["verdict"] = verdict
    print("VERDICT:", verdict)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
