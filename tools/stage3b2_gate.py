"""Stage 3B2 scripted pressure differential (scenario design only, no PPO).

Paired deterministic trials: same seed + spawns + HP, control (pearls=0)
vs treatment (pearls=4, scripted pearl-away -> eat -> re-engage).
Only scenario parameters vary (hp0, start distance, foe, pearl pitch).
Mechanics/reward/obs/action/item semantics untouched.

Usage:
  python -m tools.stage3b2_gate --episodes 10 --out logs/stage3b2_gate.json
  python -m tools.stage3b2_gate --sweep  # coarse A-D variant grid
"""
from __future__ import annotations

import argparse
import json
import math
import os

from sandbox.config import CombatConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from rl.opponents.scripted import create

CC = CombatConfig(attack_arc_deg=75.0, attack_arc_vertical_deg=75.0,
                  crit_mult=1.5, landing_attack_delay_ticks=4)
WC = WorldConfig(size=32, wall_margin=1.0, obstacles=())
BASE_LOADOUT = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
                "ender_pearl": 4, "cobweb": 0, "block": 0,
                "strength_potion": 0, "bow": 1, "arrow": 0}
ENABLED = {"sword", "shield", "bread", "golden_apple", "ender_pearl"}


def make_sim(seed: int, hp0: float, pearls: int, start_dist: float | None,
             foe_hp: float = 20.0) -> PvPSim:
    loadout = dict(BASE_LOADOUT)
    loadout["ender_pearl"] = pearls
    sim = PvPSim(world_cfg=WC, combat_cfg=CC, max_ticks=900, seed=seed,
                 loadout=loadout, enabled_items=set(ENABLED))
    sim.reset(seed)
    if start_dist is not None:
        cx = WC.size * 0.5
        sim.players[0].pos[:] = (cx - start_dist / 2, 0.0, cx)
        sim.players[1].pos[:] = (cx + start_dist / 2, 0.0, cx)
        sim.players[0].vel[:] = 0.0
        sim.players[1].vel[:] = 0.0
        # agent faces foe (+x), foe faces agent (-x). yaw: cos(yaw)=dx?
        sim.players[0].yaw = 0.0
        sim.players[1].yaw = 180.0
        for i in (0, 1):
            sim.prev_pos[i] = [sim.players[i].pos.copy() for _ in range(4)]
    sim.players[0].health = hp0
    sim.players[1].health = foe_hp
    return sim


def _face_away(sim: PvPSim) -> None:
    me, foe = sim.players[0], sim.players[1]
    away = me.pos - foe.pos
    me.yaw = float(math.degrees(math.atan2(away[2], away[0])))


def run(seed: int, hp0: float, mode: str, opp: str = "pressureeater",
        start_dist: float | None = None, pitch: float = 35.0,
        eat_move: float = 0.0, foe_hp: float = 20.0,
        lateral: bool = False) -> dict:
    """Modes: fight | eat_place | pearl_eat."""
    sim = make_sim(seed, hp0, 0 if mode == "fight_nopearl" else 4,
                   start_dist, foe_hp=foe_hp)
    if mode == "fight_nopearl":
        sim.inventories[0].counts["ender_pearl"] = 0
        mode = "fight"
    foe = create(opp, seed=seed)
    melee = create("melee", seed=seed)
    if lateral:
        def _fa(sim, _b=_face_away):
            _b(sim)
            sim.players[0].yaw += 90.0
        face = _fa
    else:
        face = _face_away
    # diagonal-away handled via lateral='diag' string
    if lateral == "diag":
        def _fd(sim, _b=_face_away):
            _b(sim)
            sim.players[0].yaw += 45.0
        face = _fd
    elif lateral == "corner":
        def _fc(sim):
            me, foe = sim.players[0], sim.players[1]
            # farthest in-bounds corner on the away half-plane (max legal gap)
            cx = 1.0 if me.pos[0] < foe.pos[0] else 31.0
            cz = 1.0 if me.pos[2] <= 16.0 else 31.0
            d = (cx - me.pos[0], cz - me.pos[2])
            me.yaw = float(math.degrees(math.atan2(d[1], d[0])))
        face = _fc
    phase = 0
    throw_tick = land_tick = eat_start = eat_end = None
    eat_ok = False
    maxsep = sim.distance()
    hp_before_pearl = None
    t = 0
    while not sim.done and t < 900:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        d = sim.distance()
        maxsep = max(maxsep, d)
        if mode == "fight":
            c = melee.act(sim, 0)
        elif mode == "eat_place":
            if phase == 0:
                inv = sim.inventories[0]
                if inv.selected != 3:
                    c["select_slot"] = 3
                else:
                    c["use_item"] = True
                if me.eating:
                    phase = 1
                    eat_start = t
            elif phase == 1:
                if not me.eating:
                    phase = 2
                    eat_ok = True
                    eat_end = t
            else:
                c = melee.act(sim, 0)
        elif mode == "pearl_eat":
            if phase == 0:
                face(sim)
                me.pitch = pitch
                hp_before_pearl = float(me.health)
                # run while aiming: hold distance through the windup
                c["move_z"] = 1.0
                c["sprint"] = True
                if sim.inventories[0].selected != 4:
                    c["select_slot"] = 4
                else:
                    c["use_item"] = True
                if sim.inventories[0].counts.get("ender_pearl", 0) < 4:
                    phase = 1
                    throw_tick = t
            elif phase == 1:
                face(sim)
                c["move_z"] = 1.0
                c["sprint"] = True
                if len(sim.projectiles) == 0 and t - throw_tick > 3:
                    phase = 2
                    land_tick = t
            elif phase == 2:
                inv = sim.inventories[0]
                if not me.eating and inv.counts.get("golden_apple", 0) > 0:
                    if inv.selected != 3:
                        c["select_slot"] = 3
                    else:
                        c["use_item"] = True
                if eat_move:
                    face(sim)
                    c["move_z"] = 1.0
                    c["sprint"] = True
                if me.eating:
                    phase = 3
                    eat_start = t
            elif phase == 3:
                if eat_move:
                    face(sim)
                    c["move_z"] = 1.0
                    c["sprint"] = True
                if not me.eating:
                    phase = 4
                    eat_ok = True
                    eat_end = t
            else:
                c = melee.act(sim, 0)
        sim.step(c, foe.act(sim, 1))
    hp_end = float(sim.players[0].health)
    abs_end = float(sim.players[0].absorption_hp)
    net = None
    if hp_before_pearl is not None and eat_ok:
        # recovered via gapple absorption+regen tick remainder; report components
        net = round((hp_end + abs_end) - hp_before_pearl, 2)
    return {
        "win": sim.winner == 0, "ticks": sim.tick,
        "hp_end": round(hp_end, 1), "abs_end": round(abs_end, 1),
        "eat_ok": eat_ok, "maxsep": round(maxsep, 1),
        "taken": round(float(sim.damage_dealt[1]), 1),
        "dealt": round(float(sim.damage_dealt[0]), 1),
        "throw_tick": throw_tick, "land_tick": land_tick,
        "eat_ticks": ((eat_end - eat_start) if eat_start and eat_end else None),
        "gap_at_land": None, "hp_before_pearl": hp_before_pearl, "net": net,
    }


def paired(hp0: float, opp: str, start_dist: float | None, seeds,
           foe_hp: float = 20.0, **kw) -> dict:
    ctrl = [run(s, hp0, "eat_place", opp, start_dist, foe_hp=foe_hp)
            for s in seeds]
    treat = [run(s, hp0, "pearl_eat", opp, start_dist, foe_hp=foe_hp, **kw)
             for s in seeds]
    fight = [run(s, hp0, "fight_nopearl", opp, start_dist, foe_hp=foe_hp)
             for s in seeds]
    def summ(rows):
        return {"wins": sum(r["win"] for r in rows),
                "eats": sum(r["eat_ok"] for r in rows),
                "n": len(rows)}
    return {"fight": summ(fight), "eat_place": summ(ctrl),
            "pearl_eat": summ(treat),
            "detail": {"fight": fight, "eat_place": ctrl, "pearl_eat": treat}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--out", default="logs/stage3b2_gate.json")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    if a.sweep:
        grid = []
        for opp in ("pressureeater", "aggressive", "punisheater", "mixedeater"):
            for hp0 in (7.0, 5.0, 4.0, 3.0, 2.0):
                for dist in (None, 8.0, 4.0):
                    r = paired(hp0, opp, dist, seeds)
                    grid.append({
                        "opp": opp, "hp0": hp0,
                        "dist": dist if dist else "spawn",
                        "fight": r["fight"], "eat_place": r["eat_place"],
                        "pearl_eat": r["pearl_eat"]})
                    print(f"{opp} hp={hp0} d={dist if dist else 'spawn'}: "
                          f"fight {r['fight']['wins']}/{a.episodes} | "
                          f"eat {r['eat_place']['wins']}/{a.episodes} "
                          f"(eats {r['eat_place']['eats']}) | "
                          f"pearl {r['pearl_eat']['wins']}/{a.episodes} "
                          f"(eats {r['pearl_eat']['eats']})")
        with open(a.out, "w") as f:
            json.dump(grid, f, indent=2)
        print("wrote", a.out)
        return
    r = paired(4.0, "pressureeater", 4.0, seeds)
    print(r["fight"], r["eat_place"], r["pearl_eat"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
