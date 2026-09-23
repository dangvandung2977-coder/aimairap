"""Replay fight analysis: streaks, spacing, strafing, sprint-timing, pressure.

  python -m tools.analyze --replay <f>.npz [--out logs/analysis.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _shield_stats(f: dict, n: int) -> dict:
    """Anti-shield conditionality. Missing keys (old replays) -> {}."""
    need = ("a0_attack", "sh1", "p1_yaw", "p0_pos", "p1_pos", "d0", "a0_mx")
    if any(k not in f for k in need):
        return {"shield_counter": "n/a (no foe-shield fields)"}
    import math
    atk = f["a0_attack"].astype(bool)
    sh = f["sh1"].astype(bool)
    mx = f["a0_mx"].astype(float)
    p0, p1 = f["p0_pos"].astype(float), f["p1_pos"].astype(float)
    yaw = f["p1_yaw"].astype(float)
    dealt = np.diff(f["d0"].astype(float), prepend=f["d0"].astype(float)[0])
    out: dict = {}

    def _p(mask):
        m = np.asarray(mask).astype(bool)
        return round(float(atk[m].mean()), 3) if m.sum() else None
    out["P_attack|foe_shield"] = _p(sh)
    out["P_attack|foe_open"] = _p(~sh)
    # inferred blocked: attack + foe shielding + foe facing + no damage
    to_foe = p1 - p0
    L = np.linalg.norm(to_foe, axis=1, keepdims=True) + 1e-9
    yr = np.radians(yaw)
    vl = np.stack([np.cos(yr), np.zeros(n), np.sin(yr)], axis=1)
    facing = (vl * (p0 - p1) / (L + 1e-9)).sum(axis=1) > 0.2
    blocked_tick = atk & sh & facing & (dealt <= 0)
    out["blocked_ticks"] = int(blocked_tick.sum())
    # reposition: movement-vector change within 8 ticks after a blocked tick
    mv = np.stack([f["a0_mx"].astype(float),
                   f["a0_mz"].astype(float) if "a0_mz" in f else np.zeros(n)],
                  axis=1)
    changed = (np.abs(np.diff(mv, axis=0)).sum(axis=1) > 0.5)
    bi = np.where(blocked_tick)[0]
    repo = 0
    for i in bi:
        if np.any(changed[i:min(n - 1, i + 8)]):
            repo += 1
    out["P_reposition|blocked"] = round(repo / max(1, len(bi)), 3)
    out["strafe|foe_shield"] = round(float((mx[sh] != 0).mean()), 3) if sh.sum() else None
    out["dmg|foe_shield"] = round(float(dealt[sh].sum()), 1)
    out["dmg|foe_open"] = round(float(dealt[~sh].sum()), 1)
    # damage within 20 ticks after a shield drop
    drops = np.where(np.diff(sh.astype(int)) < 0)[0]
    after = 0.0
    for i in drops:
        after += float(dealt[i:min(n, i + 20)].sum())
    out["dmg_20t_after_drop"] = round(after, 1)
    out["drops_seen"] = int(len(drops))
    # attack angle: 0 = agent directly in front of foe, pi = behind
    ang = np.arccos(np.clip((vl * (p0 - p1) / (L + 1e-9)).sum(axis=1), -1, 1))
    out["mean_attack_angle"] = round(float(ang[atk].mean()), 3) if atk.sum() else None
    # wasted attacks: attack ticks vs shielding-facing foe, and max streak
    wasted = blocked_tick
    out["wasted_attacks"] = int(wasted.sum())
    cur = best = 0
    for v in wasted.astype(int):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    out["wasted_max_streak"] = best
    l = int(((mx[atk] < 0).sum())) if atk.sum() else 0
    r = int(((mx[atk] > 0).sum())) if atk.sum() else 0
    out["attack_strafe_L"] = l
    out["attack_strafe_R"] = r
    return {"shield_counter": out}


def _pearl_stats(f: dict, n: int) -> dict:
    """Pearl usage, timing, conditionality. Missing keys -> {}."""
    need = ("cnt", "p0_pos", "p0_hp", "d0", "d1")
    if any(k not in f for k in need):
        return {"pearl": "n/a (no inventory fields)"}
    cnt = f["cnt"].astype(int)
    pidx = 4  # ender_pearl hotbar slot
    throws = []
    if cnt.shape[1] > pidx and n > 1:
        use = np.where(np.diff(cnt[:, pidx], prepend=cnt[0, pidx]) < 0)[0]
        throws = use.tolist()
    pos = f["p0_pos"].astype(float)
    jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    teleports = (np.where(jumps > 4.0)[0] + 1).tolist()
    hp = f["p0_hp"].astype(float)
    foe_hp = f["p1_hp"].astype(float) if "p1_hp" in f else np.zeros(n)
    dealt = np.diff(f["d0"].astype(float), prepend=f["d0"].astype(float)[0])
    taken = np.diff(f["d1"].astype(float), prepend=f["d1"].astype(float)[0])
    dist = np.linalg.norm(f["p1_pos"].astype(float) - pos, axis=1)
    sh = f["sh1"].astype(bool) if "sh1" in f else np.zeros(n, bool)
    out = {
        "throws": len(throws),
        "teleports": len(teleports),
        "throw_ticks": throws[:10],
        "teleport_dist": [round(float(jumps[i - 1]), 1) for i in teleports[:10]
                          if 0 < i <= len(jumps)],
        "hp_at_throw": [round(float(hp[i]), 1) for i in throws[:10]],
        "dist_at_throw": [round(float(dist[i]), 2) for i in throws[:10]],
        "foehp_at_throw": [round(float(foe_hp[i]), 1) for i in throws[:10]],
        "shield_up_at_throw": [bool(sh[i]) for i in throws[:10]],
        "dmg_20t_after_teleport": round(float(sum(
            dealt[i:min(n, i + 20)].sum() for i in teleports)), 1),
        "dmg_taken_20t_after_teleport": round(float(sum(
            taken[i:min(n, i + 20)].sum() for i in teleports)), 1),
        "pearls_remaining": int(cnt[-1, pidx]) if n else 0,
    }
    return {"pearl": out}


def _sustain_stats(f: dict, n: int) -> dict:
    """Eating/gapple decision quality. Missing keys (old replays) -> {}."""
    need = ("eat0", "p0_hp", "cnt", "d0", "d1")
    if any(k not in f for k in need):
        return {"sustain": "n/a (no eating fields)"}
    eat = f["eat0"].astype(int)  # -1 idle else ticks-left
    hp = f["p0_hp"].astype(float)
    eating = eat >= 0
    # eat attempts: transitions idle -> active
    started = int(np.sum((~eating[:-1]) & eating[1:])) if n > 1 else 0
    completed = int(np.sum(eating[:-1] & (~eating[1:]) & (eat[:-1] <= 1))) if n > 1 else 0
    starts_idx = [0] if (n and eating[0]) else []
    starts_idx += (np.where(~eating[:-1] & eating[1:])[0] + 1).tolist()
    hp_at_start = [round(float(hp[i]), 1) for i in starts_idx[:8]]
    dealt = np.diff(f["d0"].astype(float), prepend=f["d0"].astype(float)[0])
    taken = np.diff(f["d1"].astype(float), prepend=f["d1"].astype(float)[0])
    out = {
        "eat_starts": started,
        "eat_completions": completed,
        "interrupt_rate": round(1 - completed / max(1, started), 3),
        "hp_at_eat_start": hp_at_start,
        "dmg_taken_while_eating": round(float(taken[eating].sum()), 1),
        "dmg_dealt_while_eating": round(float(dealt[eating].sum()), 1),
        "eat_frac": round(float(eating.mean()), 3),
    }
    # gapple context: bread vs gapple via cnt deltas
    cnt = f["cnt"].astype(int)
    out["bread_used"] = int(max(0, cnt[0, 2] - cnt[-1, 2])) if n else 0
    out["gapple_used"] = int(max(0, cnt[0, 3] - cnt[-1, 3])) if n else 0
    # HP/dist/foe-cd at eat starts
    dist = np.linalg.norm(f["p1_pos"].astype(float) - f["p0_pos"].astype(float), axis=1)
    foe_cd = f["p1_cd"].astype(float) if "p1_cd" in f else np.zeros(n)
    out["dist_at_eat_start"] = [round(float(dist[i]), 2) for i in starts_idx[:8]]
    out["foecd_at_eat_start"] = [round(float(foe_cd[i]), 1) for i in starts_idx[:8]]
    return {"sustain": out}


def _item_stats(f: dict, n: int) -> dict:
    """Item usage, timing, conditionality. Missing keys (old replays) -> {}."""
    need = ("cnt", "sh0", "abs0", "eat0", "nblocks", "nwebs", "nfluids",
            "p0_hp", "p1_cd")
    if any(k not in f for k in need):
        return {"items": "n/a (pre-utility replay)"}
    out: dict = {}
    cnt = f["cnt"].astype(int)  # per-tick slot counts
    names = ["sword", "shield", "bread", "gapple", "pearl", "cobweb",
             "block", "potion", "bow"]
    for i, name in enumerate(names):
        used = int(cnt[0, i] - cnt[-1, i]) if n else 0
        out[f"use_{name}"] = max(0, used)
    sh = f["sh0"].astype(bool)
    out["shield_frac"] = round(float(sh.mean()), 3)
    foe_cd = f["p1_cd"].astype(float) > 0
    out["P_shield|foe_attacking"] = round(float(sh[foe_cd].mean()), 3) \
        if foe_cd.sum() else None
    hp = f["p0_hp"].astype(float)
    out["P_eat|low_hp"] = round(float((f["eat0"].astype(int) >= 0)[hp < 10].mean()), 3) \
        if (hp < 10).sum() else None
    out["hp_recovered"] = round(float(max(0.0, hp[-1] - hp.min())), 1)
    out["blocks_placed"] = int(f["nblocks"].astype(int).max())
    out["webs_placed"] = int(f["nwebs"].astype(int).max())
    out["fluids_placed"] = int(f["nfluids"].astype(int).max())
    out["abs_end"] = round(float(f["abs0"].astype(float)[-1]), 1)
    return {"items": out}


def _symmetry_stats(f: dict, n: int) -> dict:
    """Signed lateral diagnostics. side>0 = agent on foe's left, <0 = right
    (arbitrary but consistent convention). Missing keys -> {}."""
    need = ("p0_pos", "p1_pos", "p1_yaw", "a0_mx", "d0", "sh1")
    if any(k not in f for k in need):
        return {"symmetry": "n/a"}
    p0, p1 = f["p0_pos"].astype(float), f["p1_pos"].astype(float)
    fyaw = np.radians(f["p1_yaw"].astype(float))
    d = p0 - p1
    side = np.sin(np.arctan2(d[:, 2], d[:, 0]) - fyaw)
    mx = f["a0_mx"].astype(float)
    dealt = np.diff(f["d0"].astype(float), prepend=f["d0"].astype(float)[0])
    hit_tick = dealt > 0
    sh = f["sh1"].astype(bool)
    out: dict = {
        "mean_side": round(float(side.mean()), 3),
        "strafe_L_frac": round(float((mx < 0).mean()), 3),
        "strafe_R_frac": round(float((mx > 0).mean()), 3),
        "flank_L_ticks": int(((np.abs(np.arcsin(np.clip(side, -1, 1))) > 0.6) & (side > 0)).sum()),
        "flank_R_ticks": int(((np.abs(np.arcsin(np.clip(side, -1, 1))) > 0.6) & (side < 0)).sum()),
        "dmg_from_L": round(float(dealt[side > 0.2].sum()), 1),
        "dmg_from_R": round(float(dealt[side < -0.2].sum()), 1),
        "dmg_vs_shield": round(float(dealt[sh].sum()), 1),
        "shield_up_frac": round(float(sh.mean()), 3),
    }
    ang = np.arcsin(np.clip(side, -1, 1))  # -pi/2..pi/2 signed flank angle
    hist, _ = np.histogram(ang, bins=8, range=(-np.pi / 2, np.pi / 2))
    out["side_hist"] = [int(x) for x in hist]
    return {"symmetry": out}


def analyze(f: dict) -> dict:
    n = len(f["tick"])
    g = lambda k, d=0.0: f[k] if k in f else np.full(n, d)
    d0 = g("d0").astype(float)
    hits = np.diff(d0, prepend=d0[0]) > 0  # ticks where agent landed damage
    # streaks of hits separated by <=60 ticks (combo window)
    hit_idx = np.where(hits)[0]
    streaks, cur, best = [], 1, 0
    for a, b in zip(hit_idx, hit_idx[1:]):
        if b - a <= 60:
            cur += 1
        else:
            streaks.append(cur)
            best = max(best, cur)
            cur = 1
    if len(hit_idx):
        streaks.append(cur)
        best = max(best, cur)
    p0, p1 = f["p0_pos"], f["p1_pos"]
    dist = np.linalg.norm(p1 - p0, axis=1)
    mx = g("a0_mx").astype(float)
    dir_changes = int(np.sum(np.diff(np.sign(mx)) != 0))
    # sprint transitions within +-3 ticks of a hit
    sp = g("a0_sprint").astype(bool)
    sp_trans = int(np.sum(np.diff(sp.astype(int)) != 0))
    near_hit_trans = 0
    for i in hit_idx:
        lo, hi = max(0, i - 3), min(n - 1, i + 3)
        if np.any(np.diff(sp[max(0, lo - 1):hi + 1].astype(int)) != 0):
            near_hit_trans += 1
    # movement block
    jump = g("a0_jump").astype(bool)
    air = ~g("p0_ground", 1.0).astype(bool) if "p0_ground" in f else None
    if air is None:
        air = (f["p0_pos"][:, 1].astype(float) > 0.05)
    # positioning: separations (dist > 6) and re-engagement ticks to < 3.5
    seps, reeng = 0, []
    i = 0
    while i < n:
        if dist[i] > 6.0:
            seps += 1
            j = i
            while j < n and dist[j] >= 3.5:
                j += 1
            if j < n:
                reeng.append(j - i)
            i = j if j < n else n
        else:
            i += 1
    # counterplay base quantities
    d1 = g("d1").astype(float)
    air_ticks = air.astype(int)
    dmg_air_dealt = float(np.sum(np.diff(d0, prepend=d0[0])[air_ticks.astype(bool)]))
    dmg_air_taken = float(np.sum(np.diff(d1, prepend=d1[0])[air_ticks.astype(bool)]))
    # own miss = attack requested, cd tripped, no damage that tick
    atk_req = g("a0_attack").astype(bool)
    dealt_delta = np.diff(d0, prepend=d0[0])
    miss = atk_req & (dealt_delta <= 0)
    miss_idx = np.where(miss)[0]
    punished = 0
    for i in miss_idx:
        hi = min(n, i + 16)
        if np.any(np.diff(d1[i:hi], prepend=d1[i]) > 0):
            punished += 1
    # situational jump rates (§13: conditional movement, not aggregate rate)
    jmp = g("a0_jump").astype(bool)
    far = dist > 4.0
    near = dist <= 3.0
    hurt_any = (g("p0_hurt").astype(int) > 0) if "p0_hurt" in f else np.zeros(n, bool)
    after_miss = np.zeros(n, bool)
    for i in miss_idx:
        after_miss[i:min(n, i + 12)] = True

    def _rate(mask):
        m = np.asarray(mask).astype(bool)
        return round(float(jmp[m].mean()), 3) if m.sum() else None

    def _strafe_rate(mask):
        m = np.asarray(mask).astype(bool)
        return round(float((mx[m] != 0).mean()), 3) if m.sum() else None

    # foe-state conditionals (p1_ground / p1_cd present in phase-2.5+ replays)
    foe_ground = g("p1_ground", 1.0).astype(bool)
    foe_cd = g("p1_cd", 0.0).astype(float) > 0
    # crit fraction: damage ticks > 6.0 are crits (5.0 vs 7.5), else grounded hits
    crit_frac = float(((dealt_delta > 6.0).sum() / max(1, int(hits.sum()))))
    # jump streaks: consecutive airborne ticks
    air_i = air.astype(int)
    cur, best_js, tot_js, n_js = 0, 0, 0, 0
    for v in air_i:
        if v:
            cur += 1
        else:
            if cur:
                best_js = max(best_js, cur)
                tot_js += cur
                n_js += 1
            cur = 0
    if cur:
        best_js = max(best_js, cur)
        tot_js += cur
        n_js += 1
    jump_streak_max = best_js
    jump_streak_mean = round(tot_js / max(1, n_js), 1)
    # grounded exchanges: both grounded and in range
    own_ground = ~air
    gex_frac = float(np.mean(own_ground & foe_ground & (dist <= 3.5)))
    return {
        "ticks": n,
        "hits": int(hits.sum()),
        "max_streak": best,
        "mean_streak": round(float(np.mean(streaks)), 2) if streaks else 0.0,
        "mean_dist": round(float(dist.mean()), 2),
        "dist_p10": round(float(np.percentile(dist, 10)), 2),
        "dist_p50": round(float(np.percentile(dist, 50)), 2),
        "dist_p90": round(float(np.percentile(dist, 90)), 2),
        "strafe_dir_changes": dir_changes,
        "strafe_fraction": round(float(np.mean(mx != 0)), 3),
        "sprint_transitions": sp_trans,
        "hits_with_sprint_transition_nearby": near_hit_trans,
        "jump_ticks": int(g("a0_jump").astype(bool).sum()),
        "jump_rate": round(float(jump.mean()), 3),
        "strafe_rate": round(float(np.mean(mx != 0)), 3),
        "sprint_rate": round(float(sp.mean()), 3),
        "airborne_frac": round(float(air.mean()), 3),
        "grounded_frac": round(float((~air).mean()), 3),
        "separations": seps,
        "reengage_mean_ticks": round(float(np.mean(reeng)), 1) if reeng else None,
        "dmg_airborne_dealt": round(dmg_air_dealt, 1),
        "dmg_airborne_taken": round(dmg_air_taken, 1),
        "own_misses": int(miss.sum()),
        "misses_punished": punished,
        "miss_punish_rate": round(punished / max(1, int(miss.sum())), 3),
        "jump_when_far": _rate(far),
        "jump_when_near": _rate(near),
        "jump_when_hurt": _rate(hurt_any),
        "jump_after_miss": _rate(after_miss),
        "jump_when_airborne": _rate(air),
        "jump_when_foe_airborne": _rate(~foe_ground),
        "jump_when_foe_grounded": _rate(foe_ground),
        "strafe_when_foe_cd": _strafe_rate(foe_cd),
        "crit_frac": round(crit_frac, 3),
        "jump_streak_max": jump_streak_max,
        "jump_streak_mean": jump_streak_mean,
        "grounded_exchange_frac": round(gex_frac, 3),
        **_pearl_stats(f, n),
        **_sustain_stats(f, n),
        **_symmetry_stats(f, n),
        **_shield_stats(f, n),
        **_item_stats(f, n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from replay.recorder import load_replay
    f, meta = load_replay(a.replay)
    res = analyze(f)
    res["meta"] = {k: meta.get(k) for k in ("winner", "damage", "total_reward")}
    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f2:
            json.dump(res, f2, indent=2)


if __name__ == "__main__":
    main()
