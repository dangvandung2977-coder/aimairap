"""Stage 4A bow gate: did the policy LEARN conditional ranged combat?

Paired counterfactual (bow arrows 16 vs 0, identical enabled set so obs/action
dims match) against ranged foes, plus a melee-retention check. Drives the sim
directly (per-tick) so we can attribute every arrow shot and hit and measure
point-blank spam. No reward/obs/action/mechanic changes -- measurement only.

Evidence required to PASS (decision rule from the brief):
  - conditional use: bow-available wins more OR takes materially less than
    bow-unavailable vs the same ranged foes;
  - useful projectiles: arrows actually land (hits > 0, sane hit fraction);
  - not spam: few point-blank (<3 block) shots, shots not maxed every episode;
  - retention: melee win rate stays >= tolerance vs a pure melee foe.

  python -m tools.stage4a_gate --checkpoint <ckpt> --episodes 20 --out logs/stage4a_gate.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sandbox.config import CombatConfig, PhysicsConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from rl.environment import action_map, obs_schema
from rl.opponents.scripted import create as make_opp

ENABLED = ["sword", "shield", "bread", "golden_apple", "ender_pearl", "bow"]
BASE_LOADOUT = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
                "ender_pearl": 4, "cobweb": 0, "block": 0,
                "strength_potion": 0, "bow": 1, "arrow": 16}
RANGED_FOES = ("bowfighter", "rangedrunner")
FRAME_SKIP = 4


def _run_ep(model, opp_name, seed, arrows, arena=40, foe_arrows=32,
            start_dist=16.0, max_ticks=900):
    wc = WorldConfig(size=arena, wall_margin=1.0, obstacles=())
    cc = CombatConfig(attack_arc_deg=75.0, attack_arc_vertical_deg=75.0,
                      crit_mult=1.5, landing_attack_delay_ticks=4)
    loadout = dict(BASE_LOADOUT, arrow=arrows)
    sim = PvPSim(wc, PhysicsConfig(), cc, max_ticks=max_ticks, seed=seed,
                 loadout=loadout, enabled_items=set(ENABLED))
    sim.reset(seed)
    sim.inventories[1].counts["arrow"] = foe_arrows
    sim.inventories[1].enabled = set(ENABLED)
    cx = arena * 0.5
    sim.players[0].pos[:] = (cx - start_dist / 2, 0.0, cx)
    sim.players[1].pos[:] = (cx + start_dist / 2, 0.0, cx)
    sim.players[0].yaw, sim.players[1].yaw = 0.0, 180.0
    for i in (0, 1):
        sim.prev_pos[i] = [sim.players[i].pos.copy() for _ in range(4)]
    opp = make_opp(opp_name, seed=seed)
    opp.reset(seed)
    shots = arrow_hits = point_blank = 0
    ctrl0 = idle_ctrl()
    t = 0
    while not sim.done:
        if t % FRAME_SKIP == 0:
            obs = obs_schema.build_obs(sim, agent=0, version=4).astype(np.float32)
            act, _ = model.predict(obs, deterministic=True)
            ctrl0 = action_map.to_controller(act)
        c1 = opp.act(sim, 1)
        sim.step(ctrl0, c1)
        for ev in sim.utility_events:
            if ev.get("type") == "bow_shot" and ev.get("player") == 0:
                shots += 1
                if sim.distance() < 3.0:
                    point_blank += 1
            if ev.get("type") == "arrow_hit" and ev.get("owner") == 0:
                arrow_hits += 1
        t += 1
    return {"seed": seed, "win": 1 if sim.winner == 0 else 0, "ticks": sim.tick,
            "dealt": round(float(sim.damage_dealt[0]), 1),
            "taken": round(float(sim.damage_dealt[1]), 1),
            "shots": shots, "arrow_hits": arrow_hits, "point_blank": point_blank}


def _summ(rows):
    n = len(rows)
    sh = sum(r["shots"] for r in rows)
    return {"episodes": n,
            "win_rate": round(sum(r["win"] for r in rows) / n, 3),
            "dealt": round(float(np.mean([r["dealt"] for r in rows])), 2),
            "taken": round(float(np.mean([r["taken"] for r in rows])), 2),
            "shots_per_ep": round(sh / n, 2),
            "hits_per_ep": round(sum(r["arrow_hits"] for r in rows) / n, 2),
            "hit_frac": round(sum(r["arrow_hits"] for r in rows) / sh, 3) if sh else 0.0,
            "point_blank_per_ep": round(sum(r["point_blank"] for r in rows) / n, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=41000)
    ap.add_argument("--melee-tol", type=float, default=0.6,
                    help="min melee win rate to count as retained")
    ap.add_argument("--out", default="logs/stage4a_gate.json")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.checkpoint)
    assert int(model.observation_space.shape[0]) == 73, "expects obs v4 (73)"

    res = {"checkpoint": a.checkpoint, "conditions": {}}
    for arrows, tag in ((16, "bow_available"), (0, "bow_unavailable")):
        rows = []
        for oi, opp in enumerate(RANGED_FOES):
            for ep in range(a.episodes):
                rows.append(_run_ep(model, opp, a.seed0 + oi * 1000 + ep, arrows))
        res["conditions"][tag] = _summ(rows)
        s = res["conditions"][tag]
        print(f"{tag}: win={s['win_rate']} taken={s['taken']} shots={s['shots_per_ep']} "
              f"hits={s['hits_per_ep']} hitfrac={s['hit_frac']} pblank={s['point_blank_per_ep']}")
    # melee retention (bow present but foe is pure melee)
    mel = _summ([_run_ep(model, "melee", a.seed0 + 5000 + ep, 16)
                 for ep in range(a.episodes)])
    res["melee_retention"] = mel
    print(f"melee_retention: win={mel['win_rate']} pblank={mel['point_blank_per_ep']}")

    av = res["conditions"]["bow_available"]
    no = res["conditions"]["bow_unavailable"]
    checks = {
        "conditional_advantage": (av["win_rate"] - no["win_rate"] > 0.2)
        or (av["taken"] - no["taken"] < -4.0),
        "projectiles_land": av["hits_per_ep"] >= 1.0 and av["hit_frac"] >= 0.15,
        "no_pointblank_spam": av["point_blank_per_ep"] <= 1.5,
        "melee_retained": mel["win_rate"] >= a.melee_tol,
    }
    res["checks"] = checks
    res["verdict"] = "PASS" if all(checks.values()) else "NOT_PROVEN"
    res["deltas"] = {"win": round(av["win_rate"] - no["win_rate"], 3),
                     "taken": round(av["taken"] - no["taken"], 2)}
    print("checks:", checks)
    print("VERDICT:", res["verdict"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
