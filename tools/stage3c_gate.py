"""Stage 3C scripted gate: chase vs pearl-cutoff (scenario design, no PPO).

Paired deterministic trials: same seed/spawns/HP/foe driver; control chases
with melee, treatment pearls to cutoff at a distance trigger then melees.
Only scenario parameters vary. Mechanics/reward/obs/action untouched.

Usage:
  python -m tools.stage3c_gate --sweep            # patterns x distances
  python -m tools.stage3c_gate --timing           # throw-threshold sweep
  python -m tools.stage3c_gate --episodes 10 --out logs/stage3c_gate.json
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
OPEN = WorldConfig(size=32, wall_margin=1.0, obstacles=())
DIVIDER = WorldConfig(size=32, wall_margin=1.0,
                      obstacles=((15.5, 4.0, 16.5, 28.0, 3.0),))
LOADOUT = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
           "ender_pearl": 4, "cobweb": 0, "block": 0,
           "strength_potion": 0, "bow": 1, "arrow": 0}
ENABLED = {"sword", "shield", "bread", "golden_apple", "ender_pearl"}


def make_sim(seed: int, pearls: int, start_dist: float,
             hp0: float = 20.0, foe_hp: float = 20.0,
             wc: WorldConfig = OPEN) -> PvPSim:
    loadout = dict(LOADOUT)
    loadout["ender_pearl"] = pearls
    sim = PvPSim(world_cfg=wc, combat_cfg=CC, max_ticks=900, seed=seed,
                 loadout=loadout, enabled_items=set(ENABLED))
    sim.reset(seed)
    cx = wc.size * 0.5
    sim.players[0].pos[:] = (cx - start_dist / 2, 0.0, cx)
    sim.players[1].pos[:] = (cx + start_dist / 2, 0.0, cx)
    sim.players[0].vel[:] = 0.0
    sim.players[1].vel[:] = 0.0
    sim.players[0].yaw, sim.players[1].yaw = 0.0, 180.0
    for i in (0, 1):
        sim.prev_pos[i] = [sim.players[i].pos.copy() for _ in range(4)]
    sim.players[0].health = hp0
    sim.players[1].health = foe_hp
    return sim


def _aim_at(sim: PvPSim, tx: float, tz: float, pitch: float) -> None:
    me = sim.players[0]
    d = np.array([tx - me.pos[0], tz - me.pos[2]])
    me.yaw = float(math.degrees(math.atan2(d[1], d[0])))
    me.pitch = pitch


def _foe_vel(sim: PvPSim):
    return sim.players[1].vel.copy()


def run(seed: int, mode: str, opp: str = "farfighter",
        start_dist: float = 12.0, pitch: float = 15.0,
        trigger: float = 10.0, lead: float = 1.0,
        max_delay: int | None = None,
        wc: WorldConfig = OPEN,
        foe_kwargs: dict | None = None) -> dict:
    """Modes: chase | pearl_cutoff. trigger: pearl when dist > trigger.
    lead: seconds of foe velocity to lead the aim (cutoff point).
    max_delay: only throw after this tick (timing test); None = immediate."""
    sim = make_sim(seed, 0 if mode == "chase" else 4, start_dist, wc=wc)
    foe = create(opp, seed=seed, **(foe_kwargs or {}))
    melee = create("melee", seed=seed)
    phase = 0  # 0 chase, 1 pearl flight, 2 re-engage melee
    throw_tick = land_tick = first_hit = None
    hp_before_pearl = None
    gap_at_land = None
    dist_sum, dist_n = 0.0, 0
    min_dist = sim.distance()
    contact_ticks = 0
    pearls_left = 4
    t = 0
    while not sim.done and t < 900:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        d = sim.distance()
        dist_sum += d
        dist_n += 1
        min_dist = min(min_dist, d)
        if d <= 3.0:
            contact_ticks += 1
        if mode == "chase":
            c = melee.act(sim, 0)
        else:
            if phase == 0:
                c = melee.act(sim, 0)
                over = max_delay is not None and t < max_delay
                if d > trigger and not over:
                    phase = 9  # start throw sequence next tick
            elif phase == 9:
                fv = _foe_vel(sim)
                ax = float(sim.players[1].pos[0] + fv[0] * lead)
                az = float(sim.players[1].pos[2] + fv[2] * lead)
                _aim_at(sim, ax, az, pitch)
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
                    gap_at_land = sim.distance()
                    pearls_left = sim.inventories[0].counts.get("ender_pearl", 0)
            else:
                c = melee.act(sim, 0)
                if first_hit is None and sim.hits[0] > 0:
                    first_hit = t
        sim.step(c, foe.act(sim, 1))
    return {
        "win": sim.winner == 0, "draw": sim.winner is None and not sim.invalid,
        "ticks": sim.tick, "hp_end": round(float(sim.players[0].health), 1),
        "foe_end": round(float(sim.players[1].health), 1),
        "taken": round(float(sim.damage_dealt[1]), 1),
        "dealt": round(float(sim.damage_dealt[0]), 1),
        "mean_dist": round(dist_sum / max(1, dist_n), 1),
        "min_dist": round(min_dist, 1),
        "contact_rate": round(contact_ticks / max(1, sim.tick), 3),
        "throw_tick": throw_tick, "land_tick": land_tick,
        "first_hit_after": (first_hit - land_tick) if first_hit and land_tick else None,
        "gap_at_land": round(gap_at_land, 1) if gap_at_land else None,
        "hp_before_pearl": hp_before_pearl,
        "pearls_left": sim.inventories[0].counts.get("ender_pearl", 0),
        "stuck": bool(sim.invalid),
    }


def paired(opp: str, start_dist: float, seeds,
           wc: WorldConfig = OPEN, foe_kwargs: dict | None = None,
           **kw) -> dict:
    ctrl = [run(s, "chase", opp, start_dist, wc=wc, foe_kwargs=foe_kwargs)
            for s in seeds]
    treat = [run(s, "pearl_cutoff", opp, start_dist, wc=wc,
                 foe_kwargs=foe_kwargs, **kw) for s in seeds]

    def summ(rows):
        return {"wins": sum(r["win"] for r in rows),
                "draws": sum(r["draw"] for r in rows),
                "n": len(rows)}
    return {"chase": summ(ctrl), "pearl": summ(treat),
            "detail": {"chase": ctrl, "pearl": treat}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--out", default="logs/stage3c_gate.json")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    if a.sweep:
        grid = []
        for arena, wc in (("open", OPEN), ("divider", DIVIDER)):
            for opp in ("farfighter", "retreatfighter", "chasebait"):
                for dist in (8.0, 12.0, 16.0):
                    r = paired(opp, dist, seeds, wc=wc)
                    grid.append({"arena": arena, "opp": opp, "dist": dist,
                                 "chase": r["chase"], "pearl": r["pearl"]})
                    print(f"{arena} {opp} d={dist}: chase {r['chase']['wins']}/"
                          f"{a.episodes} (draws {r['chase']['draws']}) | pearl "
                          f"{r['pearl']['wins']}/{a.episodes} "
                          f"(draws {r['pearl']['draws']})")
        with open(a.out, "w") as f:
            json.dump(grid, f, indent=2)
        print("wrote", a.out)
        return
    if a.timing:
        for thr, delay, tag in ((8.0, None, "early"), (12.0, None, "mid"),
                                (16.0, None, "late-trigger"),
                                (12.0, 150, "delayed-150")):
            r = paired("farfighter", 12.0, seeds, wc=DIVIDER, trigger=thr,
                       max_delay=delay)
            print(f"{tag}: chase {r['chase']['wins']}/{a.episodes} "
                  f"(draws {r['chase']['draws']}) | "
                  f"pearl {r['pearl']['wins']}/{a.episodes}")
        return
    r = paired("farfighter", 12.0, seeds)
    print(r["chase"], r["pearl"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
