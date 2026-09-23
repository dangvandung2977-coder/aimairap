"""Stage 3E scripted gate: Pearl micro-skills with horizon classes (no PPO).

Candidates (paired, deterministic, mechanics/reward/obs/action untouched):
  E1 lateral-escape   (4.1): low HP + pressure -> lateral pearl -> survive/sep. H1
  E2 gap-cross        (4.2): stub wall + short window -> pearl over -> damage. H2
  E3 behind-foe       (4.3): close foe -> pearl past -> angle flip. H1 positional
  E4 short-elev       (4.4): h=1 platform holder -> jump vs pearl-up. H1-H2
  E5 quick-melee      (4.5): kiter + short window -> pearl-onto -> kill. H2-H3

Usage: python -m tools.stage3e_gate --sweep --episodes 10
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from sandbox.config import CombatConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from rl.opponents.scripted import create

CC = CombatConfig(attack_arc_deg=75.0, attack_arc_vertical_deg=75.0,
                  crit_mult=1.5, landing_attack_delay_ticks=4)
OPEN = WorldConfig(size=32, wall_margin=1.0, obstacles=())
LOADOUT = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
           "ender_pearl": 4, "cobweb": 0, "block": 0,
           "strength_potion": 0, "bow": 1, "arrow": 0}
ENABLED = {"sword", "shield", "bread", "golden_apple", "ender_pearl"}


def base_sim(seed: int, pearls: int, wc: WorldConfig, max_ticks: int,
             a_hp: float = 20.0, f_hp: float = 20.0,
             a_xy=(7.0, 16.0), f_xy=(19.0, 16.0)) -> PvPSim:
    loadout = dict(LOADOUT)
    loadout["ender_pearl"] = pearls
    sim = PvPSim(world_cfg=wc, combat_cfg=CC, max_ticks=max_ticks, seed=seed,
                 loadout=loadout, enabled_items=set(ENABLED))
    sim.reset(seed)
    sim.players[0].pos[:] = (a_xy[0], 0.0, a_xy[1])
    sim.players[1].pos[:] = (f_xy[0], 0.0, f_xy[1])
    sim.players[0].vel[:] = 0.0
    sim.players[1].vel[:] = 0.0
    sim.players[0].yaw, sim.players[1].yaw = 0.0, 180.0
    for i in (0, 1):
        sim.prev_pos[i] = [sim.players[i].pos.copy() for _ in range(4)]
    sim.players[0].health = a_hp
    sim.players[1].health = f_hp
    return sim


def _aim_at(sim: PvPSim, tx: float, tz: float, pitch: float) -> None:
    me = sim.players[0]
    d = np.array([tx - me.pos[0], tz - me.pos[2]])
    me.yaw = float(math.degrees(math.atan2(d[1], d[0])))
    me.pitch = pitch


def _away_yaw(sim: PvPSim, lateral_deg: float = 0.0) -> None:
    me, foe = sim.players[0], sim.players[1]
    away = me.pos - foe.pos
    me.yaw = float(math.degrees(math.atan2(away[2], away[0]))) + lateral_deg


def _throw_pearl(sim: PvPSim, c: dict, slot: int = 4) -> bool:
    """Run one aim/throw tick. Returns True once the pearl left."""
    inv = sim.inventories[0]
    if inv.selected != slot:
        c["select_slot"] = slot
    else:
        c["use_item"] = True
    return inv.counts.get("ender_pearl", 0) < 4


def e1(seed: int, pearl: bool):
    """Lateral escape: hp6, aggressive at d4. Control runs; treatment pearls
    laterally at once, then runs. Success: alive at tick 150 + sep."""
    sim = base_sim(seed, 4 if pearl else 0, OPEN, 900, a_hp=6.0,
                   a_xy=(13.0, 16.0), f_xy=(17.0, 16.0))
    foe = create("aggressive", seed=seed)
    melee = create("melee", seed=seed)
    phase = 0
    throw_tick = land_tick = None
    maxsep = 4.0
    t = 0
    while not sim.done and t < 150:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        maxsep = max(maxsep, sim.distance())
        if not pearl:
            away = me.pos - sim.players[1].pos
            me.yaw = float(math.degrees(math.atan2(away[2], away[0])))
            c["move_z"] = 1.0
            c["sprint"] = True
        elif phase == 0:
            _away_yaw(sim, 90.0)
            me.pitch = 20.0
            c["move_z"] = 1.0
            c["sprint"] = True
            if _throw_pearl(sim, c):
                phase = 1
                throw_tick = t
        elif phase == 1:
            _away_yaw(sim)
            c["move_z"] = 1.0
            c["sprint"] = True
            if len(sim.projectiles) == 0 and t - throw_tick > 3:
                phase = 2
                land_tick = t
        else:
            away = me.pos - sim.players[1].pos
            me.yaw = float(math.degrees(math.atan2(away[2], away[0])))
            c["move_z"] = 1.0
            c["sprint"] = True
        sim.step(c, foe.act(sim, 1))
    alive = sim.players[0].alive
    return {"win": alive and sim.distance() > 8.0, "alive": alive,
            "ticks": t, "maxsep": round(maxsep, 1),
            "end_dist": round(sim.distance(), 1),
            "hp_end": round(float(me.health), 1),
            "throw_tick": throw_tick, "land_tick": land_tick,
            "horizon": "H1"}


def e2(seed: int, pearl: bool):
    """Gap cross: 12-long stub wall between agent and dummy, window 150.
    Control detours; treatment pearls over. Success: damage/kill."""
    wc = WorldConfig(size=32, wall_margin=1.0,
                     obstacles=((16.0, 6.0, 17.0, 22.0, 3.0),))
    sim = base_sim(seed, 4 if pearl else 0, wc, 150,
                   a_xy=(10.0, 16.0), f_xy=(24.0, 16.0))
    foe = create("dummy")
    melee = create("melee", seed=seed)
    phase = 0
    throw_tick = land_tick = first_hit = None
    t = 0
    while not sim.done and t < 150:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        if not pearl:
            c = melee.act(sim, 0)
        elif phase == 0:
            _aim_at(sim, 24.0, 16.0, 20.0)
            c["move_z"] = 1.0
            c["sprint"] = True
            if _throw_pearl(sim, c):
                phase = 1
                throw_tick = t
        elif phase == 1:
            c["move_z"] = 1.0
            c["sprint"] = True
            if len(sim.projectiles) == 0 and t - throw_tick > 3:
                phase = 2
                land_tick = t
        else:
            c = melee.act(sim, 0)
            c["pitch_delta"] = -float(me.pitch)
            if first_hit is None and sim.hits[0] > 0:
                first_hit = t
        sim.step(c, foe.act(sim, 1))
    return {"win": sim.winner == 0,
            "dealt": round(float(sim.damage_dealt[0]), 1),
            "ticks": t, "throw_tick": throw_tick, "land_tick": land_tick,
            "first_hit_after": (first_hit - land_tick)
            if first_hit and land_tick else None,
            "horizon": "H2"}


def e3(seed: int, pearl: bool):
    """Behind-foe: melee foe at d4 facing agent. Treatment pearls just past
    the foe. Success: relative angle flip (agent behind foe). Positional."""
    sim = base_sim(seed, 4 if pearl else 0, OPEN, 900,
                   a_xy=(12.0, 16.0), f_xy=(16.0, 16.0))
    foe = create("melee", seed=seed)
    melee = create("melee", seed=seed)

    def behind() -> bool:
        me, fo = sim.players[0], sim.players[1]
        to_me = me.pos - fo.pos
        import math as _m
        from sandbox.physics.engine import look_dir
        lk = look_dir(fo.yaw, fo.pitch)
        n = float(np.linalg.norm(to_me)) + 1e-9
        return float(np.dot(lk, to_me / n)) < -0.5 and n < 6.0

    phase = 0
    throw_tick = land_tick = None
    flipped = False
    t = 0
    while not sim.done and t < 120:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        if not pearl:
            c = melee.act(sim, 0)
        elif phase == 0:
            fo = sim.players[1]
            _aim_at(sim, float(2 * fo.pos[0] - me.pos[0]),
                    float(2 * fo.pos[2] - me.pos[2]), 10.0)
            if _throw_pearl(sim, c):
                phase = 1
                throw_tick = t
        elif phase == 1:
            if len(sim.projectiles) == 0 and t - throw_tick > 3:
                phase = 2
                land_tick = t
        else:
            if behind():
                flipped = True
                break
        sim.step(c, foe.act(sim, 1))
    return {"win": flipped, "ticks": t, "throw_tick": throw_tick,
            "land_tick": land_tick, "horizon": "H1"}


def e4(seed: int, pearl: bool):
    """Short elevation: h=1 platform holder. Control jumps up; treatment
    pearls up. Success: kill."""
    wc = WorldConfig(size=32, wall_margin=1.0,
                     obstacles=((14.0, 14.0, 18.0, 18.0, 1.0),))
    sim = base_sim(seed, 4 if pearl else 0, wc, 900,
                   a_xy=(7.0, 16.0), f_xy=(16.0, 16.0))
    sim.players[1].pos[:] = (16.0, 1.0, 16.0)
    foe = create("dummy")
    melee = create("melee", seed=seed)
    phase = 0
    throw_tick = land_tick = None
    t = 0
    while not sim.done and t < 900:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        if not pearl:
            c = melee.act(sim, 0)
            c["jump"] = True
        elif phase == 0:
            _aim_at(sim, 16.0, 16.0, 45.0)
            c["move_z"] = 1.0
            if _throw_pearl(sim, c):
                phase = 1
                throw_tick = t
        elif phase == 1:
            if len(sim.projectiles) == 0 and t - throw_tick > 3:
                phase = 2
                land_tick = t
        else:
            c = melee.act(sim, 0)
            c["pitch_delta"] = -float(me.pitch)
        sim.step(c, foe.act(sim, 1))
    return {"win": sim.winner == 0, "ticks": t,
            "throw_tick": throw_tick, "land_tick": land_tick,
            "horizon": "H2"}


def e5(seed: int, pearl: bool):
    """Quick melee: farfighter kiter d10, window 150. Control chases;
    treatment pearls onto the kiter then melees. Success: kill."""
    sim = base_sim(seed, 4 if pearl else 0, OPEN, 150,
                   a_xy=(11.0, 16.0), f_xy=(21.0, 16.0))
    foe = create("farfighter", seed=seed)
    melee = create("melee", seed=seed)
    phase = 0
    throw_tick = land_tick = first_hit = None
    t = 0
    while not sim.done and t < 150:
        t += 1
        c = dict(idle_ctrl())
        me = sim.players[0]
        if not pearl:
            c = melee.act(sim, 0)
        elif phase == 0:
            fo = sim.players[1]
            _aim_at(sim, float(fo.pos[0]), float(fo.pos[2]), 12.0)
            c["move_z"] = 1.0
            c["sprint"] = True
            if _throw_pearl(sim, c):
                phase = 1
                throw_tick = t
        elif phase == 1:
            c["move_z"] = 1.0
            c["sprint"] = True
            if len(sim.projectiles) == 0 and t - throw_tick > 3:
                phase = 2
                land_tick = t
        else:
            c = melee.act(sim, 0)
            c["pitch_delta"] = -float(me.pitch)
            if first_hit is None and sim.hits[0] > 0:
                first_hit = t
        sim.step(c, foe.act(sim, 1))
    return {"win": sim.winner == 0,
            "dealt": round(float(sim.damage_dealt[0]), 1),
            "ticks": t, "throw_tick": throw_tick, "land_tick": land_tick,
            "first_hit_after": (first_hit - land_tick)
            if first_hit and land_tick else None,
            "horizon": "H2-H3"}


CANDIDATES = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--out", default="logs/stage3e_gate.json")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    names = [a.only] if a.only else sorted(CANDIDATES)
    res = {}
    for name in names:
        fn = CANDIDATES[name]
        ctrl = [fn(s, False) for s in seeds]
        treat = [fn(s, True) for s in seeds]
        cw = sum(1 for r in ctrl if r["win"])
        tw = sum(1 for r in treat if r["win"])
        res[name] = {"control": ctrl, "treatment": treat,
                     "horizon": ctrl[0]["horizon"]}
        print(f"{name} [{ctrl[0]['horizon']}]: no-pearl {cw}/{a.episodes} | "
              f"pearl {tw}/{a.episodes}")
        if name in ("e1",):
            print(f"   ctrl alive/ticks: "
                  f"{[(r['alive'], r['ticks']) for r in ctrl][:4]}")
            print(f"   treat maxsep/end: "
                  f"{[(r['maxsep'], r['end_dist']) for r in treat][:4]}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
