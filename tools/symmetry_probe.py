"""Symmetry probe: per-episode strafe-side fractions vs circlerL/R (+mirror).

Usage: python -m tools.symmetry_probe --checkpoint <ckpt> --episodes 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="logs/symmetry_probe.json")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from rl.environment import action_map
    from rl.environment.gym_env import PvPEnv
    model = PPO.load(a.checkpoint)
    dim = int(model.observation_space.shape[0])
    obs_v = {39: 1, 44: 2, 47: 3, 73: 4}[dim]
    res = {}
    for prof in ("circlerL", "circlerR"):
        for mirror in (False, True):
            rows = []
            for ep in range(a.episodes):
                seed = 5000 + ep
                env = PvPEnv(opponent=prof, opponent_kwargs={"seed": seed},
                             obs_version=obs_v, mirror=mirror)
                obs, _ = env.reset(seed=seed)
                L = R = 0
                n = 0
                done = False
                while not done:
                    act, _ = model.predict(obs, deterministic=True)
                    # MultiDiscrete raw: move is branch 0
                    move = int(np.asarray(act).flatten()[0])
                    if move == 3:
                        L += 1
                    elif move == 4:
                        R += 1
                    n += 1
                    obs, _, term, trunc, info = env.step(act)
                    done = term or trunc
                rows.append({"seed": seed, "L": L / max(1, n), "R": R / max(1, n),
                             "win": info.get("win", 0)})
            key = f"{prof}{'_mirror' if mirror else ''}"
            res[key] = {"episodes": rows,
                        "mean_L": float(np.mean([r["L"] for r in rows])),
                        "mean_R": float(np.mean([r["R"] for r in rows])),
                        "win_rate": sum(r["win"] for r in rows) / len(rows)}
            print(key, "L=%.2f R=%.2f win=%.2f" % (res[key]["mean_L"],
                  res[key]["mean_R"], res[key]["win_rate"]))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
