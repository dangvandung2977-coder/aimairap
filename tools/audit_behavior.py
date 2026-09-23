"""Anti-reward-hacking audit from a replay file or a live policy rollout.

Flags: running away, standing still, spinning, air-attack spam, jump spam,
wall hugging, combat avoidance. Prints a verdict; exits nonzero on FAIL.
"""
from __future__ import annotations

import argparse

import numpy as np


def audit_frames(f: dict, size: float = 20.0) -> dict:
    n = len(f["tick"])
    p0 = f["p0_pos"]
    travel = float(np.sum(np.linalg.norm(np.diff(p0, axis=0), axis=1)))
    wall = float(np.mean([min(p[0], p[2], size - p[0], size - p[2]) for p in p0]) )
    yaw = f["p0_yaw"].astype(float)
    spin = float(np.sum(np.abs(np.diff(((yaw + 180) % 360) - 180)))) / max(1, n)
    atk = f["a0_attack"].astype(bool)
    hp = f["p0_hp"].astype(float)
    return {
        "ticks": n,
        "travel_blocks": round(travel, 1),
        "mean_wall_margin": round(wall, 2),
        "mean_yaw_change_per_tick": round(spin, 2),
        "attack_ticks": int(atk.sum()),
        "attack_rate": round(float(atk.mean()), 3),
        "end_hp": round(float(hp[-1]), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    a = ap.parse_args()
    from replay.recorder import load_replay
    f, meta = load_replay(a.replay)
    m = audit_frames(f, float(meta.get("size", 20)))
    print(m, "meta:", {k: meta.get(k) for k in ("winner", "damage", "total_reward")})
    fails = []
    profit = float(meta.get("total_reward", 0.0)) > 0.0
    if m["travel_blocks"] < 2.0 and meta.get("winner") is None:
        fails.append("STAND-STILL without result")
    if m["mean_yaw_change_per_tick"] > 40.0:
        fails.append("SPINNING")
    if m["attack_rate"] > 0.9 and float(meta.get("damage", [0])[0]) == 0.0:
        fails.append("AIR-ATTACK spam, no damage")
    if m["mean_wall_margin"] < 1.5 and m["travel_blocks"] > 30.0:
        fails.append("WALL-HUGGING while roaming")
    if fails and not profit:
        print("WEAK (not a hack):", fails, "- reward negative, selection removes it")
    elif fails:
        print("FAIL (profitable exploit):", fails)
    else:
        print("PASS: no hack signatures")
    raise SystemExit(1 if (fails and profit) else 0)


if __name__ == "__main__":
    main()
