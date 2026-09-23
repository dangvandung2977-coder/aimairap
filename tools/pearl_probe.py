"""Pearl counterfactuals (§12): available vs unavailable, matched states.

Usage: python -m tools.pearl_probe --checkpoint <ckpt> --episodes 10
Compares the same policy with pearls present vs removed from inventory.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _run(model, obs_v: int, opponent: str, pearls: int, episodes: int,
         seed0: int, start_hp=None, arena: int = 20,
         obstacles: list | None = None) -> dict:
    from rl.environment.gym_env import PvPEnv
    from sandbox.config import WorldConfig
    loadout = {"sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
               "ender_pearl": pearls, "cobweb": 0, "block": 0,
               "strength_potion": 0, "bow": 1, "arrow": 0}
    wc = WorldConfig(size=arena, wall_margin=1.0, obstacles=tuple(obstacles or ()))
    wins, dealt, taken, throws = 0, [], [], 0
    for ep in range(episodes):
        env = PvPEnv(opponent=opponent, obs_version=obs_v, loadout=dict(loadout),
                     enabled_items={"sword", "shield", "bread", "golden_apple",
                                    "ender_pearl"},
                     agent_start_hp=start_hp, world_cfg=wc)
        obs, _ = env.reset(seed=seed0 + ep)
        prev_n = pearls
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            cur_n = env.sim.inventories[0].counts.get("ender_pearl", 0)
            throws += max(0, prev_n - cur_n)
            prev_n = cur_n
            done = term or trunc
        wins += info.get("win", 0)
        dealt.append(info["damage_dealt"])
        taken.append(info["damage_taken"])
    n = episodes
    return {"episodes": n, "win_rate": wins / n,
            "damage_dealt": round(float(np.mean(dealt)), 1),
            "damage_taken": round(float(np.mean(taken)), 1),
            "throws": throws, "throws_per_ep": round(throws / n, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="logs/pearl_probe.json")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.checkpoint)
    obs_v = {39: 1, 44: 2, 47: 3, 73: 4}[int(model.observation_space.shape[0])]
    res = {}
    kite = {"arena": 32, "obstacles": [(15.5, 4.0, 16.5, 28.0, 3.0)]}
    for opp, hp, extra in (("retreatfighter", None, {}),
                           ("farfighter", None, kite),
                           ("knockbackbully", [6.0, 10.0], {})):
        avail = _run(model, obs_v, opp, 4, a.episodes, 8000, hp, **extra)
        no = _run(model, obs_v, opp, 0, a.episodes, 8000, hp, **extra)
        res[opp] = {"pearl_available": avail, "pearl_removed": no}
        print(opp, "avail:", avail, "removed:", no)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
