"""Stage 3B gate: Pearl-available vs Pearl-unavailable under identical low-HP pressure.

Open flat arena (32, no obstacles), aggressive pressuring foes, agent starts low.
Paired seeds: same reset seed + foe seed for both conditions; only pearl count
differs (4 vs 0, same enabled set so obs/action dims are identical).
No reward/obs/action/mechanic changes here -- pure measurement.

Usage:
  python -m tools.stage3b_gate --checkpoint <ckpt> --episodes 20 --out logs/stage3b_gate.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

OPPS = ("aggressive", "pressureeater", "punisheater", "mixedeater")
LOADOUT = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
           "ender_pearl": 4, "cobweb": 0, "block": 0,
           "strength_potion": 0, "bow": 1, "arrow": 0}
ENABLED = {"sword", "shield", "bread", "golden_apple", "ender_pearl"}


def _run_ep(model, obs_v: int, opp: str, seed: int, hp0: float, pearls: int,
             foe_hp: float = 20.0, start_dist: float | None = None) -> dict:
    from rl.environment.gym_env import PvPEnv
    from sandbox.config import CombatConfig, WorldConfig
    loadout = dict(LOADOUT)
    loadout["ender_pearl"] = pearls
    wc = WorldConfig(size=32, wall_margin=1.0, obstacles=())
    cc = CombatConfig(attack_arc_deg=75.0, attack_arc_vertical_deg=75.0,
                      crit_mult=1.5, landing_attack_delay_ticks=4)
    env = PvPEnv(opponent=opp, opponent_kwargs={"seed": seed}, obs_version=obs_v,
                 loadout=loadout, enabled_items=set(ENABLED), world_cfg=wc,
                 combat_cfg=cc, agent_start_hp=[hp0, hp0])
    obs, _ = env.reset(seed=seed)
    if foe_hp != 20.0:
        env.sim.players[1].health = min(env.sim.players[1].max_health, foe_hp)
    if start_dist is not None:
        cx = 32 * 0.5
        env.sim.players[0].pos[:] = (cx - start_dist / 2, 0.0, cx)
        env.sim.players[1].pos[:] = (cx + start_dist / 2, 0.0, cx)
        env.sim.players[0].vel[:] = 0.0
        env.sim.players[1].vel[:] = 0.0
        env.sim.players[0].yaw, env.sim.players[1].yaw = 0.0, 180.0
        for i in (0, 1):
            env.sim.prev_pos[i] = [env.sim.players[i].pos.copy()
                                   for _ in range(4)]
    me = env.sim.players[0]
    hp_start = float(me.health)
    throws = 0
    prev_pearls = pearls
    eat_starts = eat_ok = eat_break = 0
    dmg_before_eat = dmg_during_eat = 0.0
    prev_taken = 0.0
    was_eating = False
    eat_start_tick = eat_start_hp = eat_start_dist = None
    food_at_start = 0
    first_throw_tick = first_ok_tick = None
    throw_hp = []
    throw_dist = []
    throw_tick = []
    maxsep = float(env.sim.distance())
    sep_at_eat = []
    complete_tick = None
    reengage_tick = None
    eating_now = False
    while True:
        # damage delta attribution (decision granularity)
        taken = float(env.sim.damage_dealt[1])
        eating_now = bool(env.sim.players[0].eating)
        d = taken - prev_taken
        if eating_now:
            dmg_during_eat += d
        elif eat_starts == 0:
            dmg_before_eat += d
        prev_taken = taken
        # eating edges
        if eating_now and not was_eating:
            eat_starts += 1
            eat_start_tick = env.sim.tick
            eat_start_hp = float(env.sim.players[0].health)
            eat_start_dist = float(env.sim.distance())
            food_at_start = (env.sim.inventories[0].counts.get("bread", 0)
                             + env.sim.inventories[0].counts.get("golden_apple", 0))
            if first_throw_tick is not None:
                sep_at_eat.append(eat_start_dist)
        if was_eating and not eating_now:
            food_now = (env.sim.inventories[0].counts.get("bread", 0)
                        + env.sim.inventories[0].counts.get("golden_apple", 0))
            if food_now < food_at_start:
                eat_ok += 1
                if first_ok_tick is None:
                    first_ok_tick = env.sim.tick
                    complete_tick = env.sim.tick
            else:
                eat_break += 1
        was_eating = eating_now
        # re-engage: first dist<3 after a completed eat
        if complete_tick is not None and reengage_tick is None \
                and float(env.sim.distance()) < 3.0:
            reengage_tick = env.sim.tick
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        maxsep = max(maxsep, float(env.sim.distance()))
        cur = env.sim.inventories[0].counts.get("ender_pearl", 0)
        if cur < prev_pearls:
            throws += prev_pearls - cur
            if first_throw_tick is None:
                first_throw_tick = env.sim.tick
            throw_hp.append(float(env.sim.players[0].health))
            throw_dist.append(float(env.sim.distance()))
            throw_tick.append(env.sim.tick)
        prev_pearls = cur
        if term or trunc:
            break
    hp_end = float(env.sim.players[0].health)
    pearl_to_eat = int(first_throw_tick is not None and first_ok_tick is not None
                       and first_ok_tick > first_throw_tick)
    return {
        "seed": seed, "win": info.get("win", 0), "ticks": info["tick"],
        "hp_start": hp_start, "hp_end": hp_end,
        "hp_delta": round(hp_end - hp_start, 2),
        "abs_end": round(float(env.sim.players[0].absorption_hp), 1),
        "dealt": round(float(info["damage_dealt"]), 1),
        "taken": round(float(info["damage_taken"]), 1),
        "attacks": info["attacks"], "hits": info["hits"],
        "hit_rate": round(info["hits"] / info["attacks"], 3) if info["attacks"] else 0.0,
        "throws": throws, "throw_hp": [round(x, 1) for x in throw_hp],
        "throw_dist": [round(x, 1) for x in throw_dist],
        "throw_tick": list(throw_tick),
        "maxsep": round(maxsep, 1),
        "eat_starts": eat_starts, "eat_ok": eat_ok, "eat_break": eat_break,
        "dmg_before_eat": round(dmg_before_eat, 1),
        "dmg_during_eat": round(dmg_during_eat, 1),
        "pearl_to_eat": pearl_to_eat,
        "sep_at_eat": [round(x, 1) for x in sep_at_eat],
        "time_to_eat": (eat_start_tick if eat_starts else None),
        "time_to_reengage": ((reengage_tick - complete_tick)
                             if reengage_tick is not None else None),
        "final_dist": round(float(info["distance"]), 2),
    }


def _summ(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "episodes": n,
        "win_rate": round(sum(r["win"] for r in rows) / n, 3),
        "survival_ticks": round(float(np.mean([r["ticks"] for r in rows])), 1),
        "hp_delta": round(float(np.mean([r["hp_delta"] for r in rows])), 2),
        "eat_ok_rate": round(sum(1 for r in rows if r["eat_ok"] > 0) / n, 3),
        "eats_per_ep": round(sum(r["eat_ok"] for r in rows) / n, 3),
        "eat_break_per_ep": round(sum(r["eat_break"] for r in rows) / n, 3),
        "dmg_before_eat": round(float(np.mean([r["dmg_before_eat"] for r in rows])), 1),
        "dmg_during_eat": round(float(np.mean([r["dmg_during_eat"] for r in rows])), 1),
        "throws_per_ep": round(sum(r["throws"] for r in rows) / n, 3),
        "mean_throw_hp": round(float(np.mean(
            [h for r in rows for h in r["throw_hp"]])), 1) if any(
            r["throw_hp"] for r in rows) else None,
        "mean_throw_dist": round(float(np.mean(
            [v for r in rows for v in r["throw_dist"]])), 1) if any(
            r["throw_dist"] for r in rows) else None,
        "mean_throw_tick": round(float(np.mean(
            [v for r in rows for v in r["throw_tick"]])), 1) if any(
            r["throw_tick"] for r in rows) else None,
        "mean_maxsep": round(float(np.mean([r["maxsep"] for r in rows])), 1),
        "mean_sep_at_eat": round(float(np.mean(
            [v for r in rows for v in r["sep_at_eat"]])), 1) if any(
            r["sep_at_eat"] for r in rows) else None,
        "pearl_to_eat_rate": round(sum(r["pearl_to_eat"] for r in rows) / n, 3),
        "unnecessary_pearl_eps": sum(1 for r in rows if r["throws"] > 0 and r["eat_ok"] == 0),
        "hit_rate": round(float(np.mean([r["hit_rate"] for r in rows])), 3),
        "dealt": round(float(np.mean([r["dealt"] for r in rows])), 1),
        "taken": round(float(np.mean([r["taken"] for r in rows])), 1),
        "mean_len": round(float(np.mean([r["ticks"] for r in rows])), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hp", type=float, default=10.0)
    ap.add_argument("--foe-hp", type=float, default=20.0)
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--opps", default=None,
                    help="comma-separated subset, default all four")
    ap.add_argument("--seed0", type=int, default=8100)
    ap.add_argument("--out", default="logs/stage3b_gate.json")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.checkpoint)
    obs_v = {39: 1, 44: 2, 47: 3, 73: 4}[int(model.observation_space.shape[0])]
    res: dict = {"checkpoint": a.checkpoint, "hp0": a.hp, "obs_version": obs_v,
                 "arena": {"size": 32, "obstacles": []},
                 "foe_hp": a.foe_hp, "start_dist": a.dist,
                 "conditions": {}}
    opps = a.opps.split(",") if a.opps else OPPS
    for pearls, tag in ((4, "pearl_available"), (0, "pearl_unavailable")):
        per_opp: dict = {}
        all_rows: list = []
        for oi, opp in enumerate(opps):
            rows = [_run_ep(model, obs_v, opp, a.seed0 + oi * 1000 + ep,
                            a.hp, pearls, a.foe_hp, a.dist)
                    for ep in range(a.episodes)]
            per_opp[opp] = {"summary": _summ(rows), "episodes_detail": rows}
            all_rows += rows
            s = per_opp[opp]["summary"]
            print(f"{tag} {opp}: win={s['win_rate']} eat_ok={s['eat_ok_rate']} "
                  f"p2e={s['pearl_to_eat_rate']} throws={s['throws_per_ep']} "
                  f"thr_hp={s['mean_throw_hp']} thr_d={s['mean_throw_dist']} "
                  f"maxsep={s['mean_maxsep']} sepEat={s['mean_sep_at_eat']} "
                  f"taken={s['taken']} hit={s['hit_rate']}")
        res["conditions"][tag] = {"summary": _summ(all_rows), "per_opponent": per_opp}
    av = res["conditions"]["pearl_available"]["summary"]
    no = res["conditions"]["pearl_unavailable"]["summary"]
    print(f"OVERALL avail: win={av['win_rate']} eat={av['eat_ok_rate']} "
          f"p2e={av['pearl_to_eat_rate']} throws={av['throws_per_ep']}")
    print(f"OVERALL no-pearl: win={no['win_rate']} eat={no['eat_ok_rate']}")
    print(f"DELTA win={av['win_rate']-no['win_rate']:+.3f} "
          f"eat={av['eat_ok_rate']-no['eat_ok_rate']:+.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
