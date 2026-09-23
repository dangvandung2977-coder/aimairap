"""Anti-spam sustain probes (§15): full-HP waste, threat eating, safe windows.

Drives a checkpoint through constructed scenarios; reports use counts.
Usage: python -m tools.sustain_probe --checkpoint <ckpt> --episodes 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _run(model, obs_v: int, opponent: str, start_hp: float | None,
         episodes: int, seed0: int, opp_kwargs=None) -> dict:
    from rl.environment.gym_env import PvPEnv
    uses, wins, hp_start, hp_end = 0, 0, [], []
    for ep in range(episodes):
        env = PvPEnv(opponent=opponent, opponent_kwargs=opp_kwargs or {},
                     obs_version=obs_v, record=False,
                     agent_start_hp=[start_hp, start_hp] if start_hp else None)
        obs, _ = env.reset(seed=seed0 + ep)
        hp_start.append(float(env.sim.players[0].health))
        done = False
        prev_eating = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            eating = bool(env.sim.players[0].eating)
            if eating and not prev_eating:
                uses += 1
            prev_eating = eating
            done = term or trunc
        hp_end.append(float(env.sim.players[0].health))
        wins += info.get("win", 0)
    return {"episodes": episodes, "eats": uses,
            "eats_per_ep": round(uses / episodes, 2),
            "win_rate": wins / episodes,
            "mean_hp_start": round(float(np.mean(hp_start)), 1),
            "mean_hp_end": round(float(np.mean(hp_end)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default="logs/sustain_probe.json")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.checkpoint)
    obs_v = {39: 1, 44: 2, 47: 3, 73: 4}[int(model.observation_space.shape[0])]
    res = {
        "full_hp_melee": _run(model, obs_v, "melee", 20.0, a.episodes, 7000),
        "low_hp_threat": _run(model, obs_v, "punisheater", 7.0, a.episodes, 7100),
        "low_hp_safe": _run(model, obs_v, "dummy", 7.0, a.episodes, 7200),
    }
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
