"""Skill regression: does a later checkpoint retain earlier-stage skills?

Runs melee/shield/sustain scenario probes with the appropriate unlock sets.
Usage: python -m tools.regression --checkpoint <ckpt> --out logs/regression.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SCENARIOS = {
    # name: (opponent, enabled items)
    "melee": ("melee", ["sword"]),
    "shield_defense": ("aggressive", ["sword", "shield"]),
    "sustain": ("melee", ["sword", "shield", "bread", "golden_apple"]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="logs/regression.json")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from rl.environment.gym_env import PvPEnv
    model = PPO.load(a.checkpoint)
    dim = int(model.observation_space.shape[0])
    obs_v = {39: 1, 44: 2, 47: 3, 73: 4}[dim]
    res = {}
    for name, (opp, enabled) in SCENARIOS.items():
        wins, dealt, taken, blocks, used = 0, [], [], [], []
        for ep in range(a.episodes):
            env = PvPEnv(opponent=opp, obs_version=obs_v,
                         enabled_items=set(enabled))
            obs, _ = env.reset(seed=9000 + ep)
            done, total = False, 0.0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(act)
                total += rew
                done = term or trunc
            wins += info.get("win", 0)
            dealt.append(info["damage_dealt"])
            taken.append(info["damage_taken"])
            blocks.append(info.get("blocks", 0))
            used.append(info.get("items_used", 0))
        res[name] = {"win_rate": wins / a.episodes,
                     "damage_dealt": float(np.mean(dealt)),
                     "damage_taken": float(np.mean(taken)),
                     "blocks": float(np.mean(blocks)),
                     "items_used": float(np.mean(used))}
        print(name, res[name])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
