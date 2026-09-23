"""Stage 3D scripted gate: terrain/positional pearl utility (no PPO).

Paired deterministic trials: same seed/spawns/HP/foe/terrain; control tries
normal movement (climb/jump/chase), treatment pearls to the new position.
Only scenario parameters vary. Mechanics/reward/obs/action untouched.

Usage:
  python -m tools.stage3d_gate --sweep            # heights x foes
  python -m tools.stage3d_gate --episodes 20 --out logs/stage3d_gate.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from sandbox.config import CombatConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from rl.opponents.scripted import create

CC = CombatConfig(attack_arc_deg=75.0, attack_arc_vertical_deg=75.0,
                  crit_mult=1.5, landing_attack_delay_ticks=4)
LOADOUT = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
           "ender_pearl": 4, "cobweb": 0, "block": 0,
           "strength_potion": 0, "bow": 1, "arrow": 0}
ENABLED = {"sword", "shield", "bread", "golden_apple", "ender_pearl"}


def platform_wc(h: float, size: int = 32, cx: float = 16.0,
                half: float = 2.0) -> WorldConfig:
    return WorldConfig(size=size, wall_margin=1.0, obstacles=(
        (cx - half, cx - half, cx + half, cx + half, h),))


def make_sim(seed: int, pearls: int, wc: WorldConfig, arrows: int = 0,
             agent_hp: float = 20.0, foe_hp: float = 20.0,
             foe_xy=(16.0, 16.0), foe_y: float = 0.0,
             agent_xy=(7.0, 16.0)) -> PvPSim:
    loadout = dict(LOADOUT)
    loadout["ender_pearl"] = pearls
    loadout["arrow"] = arrows
    sim = PvPSim(world_cfg=wc, combat_cfg=CC, max_ticks=900, seed=seed,
                 loadout=loadout, enabled_items=set(ENABLED))
    sim.reset(seed)
    sim.players[0].pos[:] = (agent_xy[0], 0.0, agent_xy[1])
    sim.players[1].pos[:] = (foe_xy[0], foe_y, foe_xy[1])
    sim.players[0].vel[:] = 0.0
    sim.players[1].vel[:] = 0.0
    sim.players[0].yaw = 0.0
    sim.players[1].yaw = 180.0
    for i in (0, 1):
        sim.prev_pos[i] = [sim.players[i].pos.copy() for _ in range(4)]
    sim.players[0].health = agent_hp
    sim.players[1].health = foe_hp
    return sim


def _aim_at(sim: PvPSim, tx: float, tz: float, pitch: float) -> None:
    me = sim.players[0]
    d = np.array([tx - me.pos[0], tz - me.pos[2]])
    me.yaw = float(math.degrees(math.atan2(d[1], d[0])))
    me.pitch = pitch


def run(seed: int, mode: str, foe: str = "dummy", h: float = 3.0,
        pitch: float = 25.0, arrows: int = 4,
        max_delay: int | None = None, half: float = 2.0) -> dict:
    """Modes: climb (normal movement + jump spam + melee) | pearl_up.
    Foe starts on the platform top (y=h). half: platform half-width."""
    wc = platform_wc(h, half=half)
    sim = make_sim(seed, 0 if mode == "climb" else 4, wc, arrows=arrows,
                   foe_xy=(16.0, 16.0), foe_y=h)
    foe_drv = (create("dummy") if foe == "dummy"
               else create(foe, seed=seed))
    melee = create("melee", seed=seed)
    phase = 0
    throw_tick = land_tick = first_hit = None
    hp_before_pearl = None
    land_pos = None
    dist_sum, dist_n = 0.0, 0
    contact = 0
    t = 0
    while not sim.done and t < 900:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        d = sim.distance()
        dist_sum += d
        dist_n += 1
        if d <= 3.0:
            contact += 1
        if mode == "climb":
            # honest normal-movement attempt: push into platform, jump, swing
            c = melee.act(sim, 0)
            c["jump"] = True
        else:
            if phase == 0:
                over = max_delay is not None and t < max_delay
                if not over:
                    phase = 9
                else:
                    c = melee.act(sim, 0)
                    c["jump"] = True
            if phase == 9:
                _aim_at(sim, 16.0, 16.0, pitch)
                hp_before_pearl = float(me.health)
                c["move_z"] = 1.0
                c["sprint"] = True
                inv = sim.inventories[0]
                if inv.selected != 4:
                    c["select_slot"] = 4
                else:
                    c["use_item"] = True
                if inv.counts.get("ender_pearl", 0) < 4:
                    phase = 1
                    throw_tick = t
            elif phase == 1:
                c["move_z"] = 1.0
                c["sprint"] = True
                if len(sim.projectiles) == 0 and t - throw_tick > 3:
                    phase = 2
                    land_tick = t
                    land_pos = [round(float(v), 1) for v in me.pos]
            elif phase == 2:
                c = melee.act(sim, 0)
                # gaze discipline: the pearl aim pitch persists (scripted
                # drivers only touch yaw); level gaze for the melee arc.
                c["pitch_delta"] = -float(me.pitch)
                if first_hit is None and sim.hits[0] > 0:
                    first_hit = t
        sim.step(c, foe_drv.act(sim, 1))
    return {
        "win": sim.winner == 0, "draw": sim.winner is None and not sim.invalid,
        "ticks": sim.tick, "hp_end": round(float(me.health), 1),
        "foe_end": round(float(sim.players[1].health), 1),
        "taken": round(float(sim.damage_dealt[1]), 1),
        "dealt": round(float(sim.damage_dealt[0]), 1),
        "mean_dist": round(dist_sum / max(1, dist_n), 1),
        "contact_rate": round(contact / max(1, sim.tick), 3),
        "throw_tick": throw_tick, "land_tick": land_tick,
        "land_pos": land_pos,
        "first_hit_after": (first_hit - land_tick) if first_hit and land_tick else None,
        "hp_before_pearl": hp_before_pearl,
        "pearls_left": sim.inventories[0].counts.get("ender_pearl", 0),
        "stuck": bool(sim.invalid),
        "agent_y_end": round(float(me.pos[1]), 1),
    }


def paired(foe: str, h: float, seeds, half: float = 2.0, **kw) -> dict:
    ctrl = [run(s, "climb", foe, h, half=half) for s in seeds]
    treat = [run(s, "pearl_up", foe, h, half=half, **kw) for s in seeds]

    def summ(rows):
        return {"wins": sum(r["win"] for r in rows),
                "draws": sum(r["draw"] for r in rows),
                "n": len(rows)}
    return {"climb": summ(ctrl), "pearl": summ(treat),
            "detail": {"climb": ctrl, "pearl": treat}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--out", default="logs/stage3d_gate.json")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    if a.sweep:
        grid = []
        for foe in ("dummy", "rangedfighter"):
            for h in (1.0, 2.0, 3.0, 4.0):
                r = paired(foe, h, seeds)
                grid.append({"foe": foe, "h": h,
                             "climb": r["climb"], "pearl": r["pearl"]})
                print(f"{foe} h={h}: climb {r['climb']['wins']}/{a.episodes} "
                      f"(draws {r['climb']['draws']}) | pearl "
                      f"{r['pearl']['wins']}/{a.episodes} "
                      f"(draws {r['pearl']['draws']})")
        with open(a.out, "w") as f:
            json.dump(grid, f, indent=2)
        print("wrote", a.out)
        return
    r = paired("dummy", 3.0, seeds)
    print(r["climb"], r["pearl"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
