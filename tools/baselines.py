"""Experiments A/B/C vs scripted melee. Machine-readable JSON results.

  A: random agent vs melee (weak baseline)
  B: scripted melee vs melee (combat sanity: both sides same policy class)
  C: PPO checkpoint vs melee (learning measurement; --checkpoint for trained)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rl.environment import action_map
from rl.environment.gym_env import PvPEnv
from rl.opponents.scripted import create as create_opponent


def run_episodes(driver: str, opponent: str, episodes: int, seed: int,
                 checkpoint: str | None = None) -> dict:
    model = None
    if driver == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(checkpoint)
    foe = create_opponent(opponent)
    scripted_driver = create_opponent("melee") if driver == "scripted" else None
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        env = PvPEnv(opponent="dummy")  # sim/cfg identical; foe driven manually
        env.opponent = foe
        if scripted_driver is not None:
            scripted_driver.reset(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        # NOTE: env.opponent swap after construction keeps sim/cfg identical.
        done, total = False, 0.0
        while not done:
            if driver == "random":
                act = action_map.random_controller(rng)
                obs, rew, term, trunc, info = env.step(act)
                total += rew
            elif driver == "scripted":
                # hold one scripted controller decision for frame_skip ticks
                from rl.environment import obs_schema
                ctrl = scripted_driver.act(env.sim, 0)
                for _ in range(env.frame_skip):
                    if env.sim.done:
                        break
                    c1 = env.opponent.act(env.sim, 1)
                    env.sim.step(ctrl, c1)
                    r, _ = env.book.step(env.sim, 0, env.sim.done, env._facing(0))
                    total += r
                obs = obs_schema.build_obs(env.sim, agent=0).astype(np.float32)
                info = env._info(done=env.sim.done)
            else:  # ppo
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(act)
                total += rew
            done = bool(info["done"])
        rows.append({"reward": total, "win": info.get("win", 0),
                     "damage_dealt": info["damage_dealt"],
                     "damage_taken": info["damage_taken"],
                     "attacks": info["attacks"], "hits": info["hits"],
                     "hit_rate": info["hit_rate"], "ticks": info["tick"],
                     "hp": info["hp"], "distance": info["distance"]})
    atk = sum(r["attacks"] for r in rows)
    hit = sum(r["hits"] for r in rows)
    return {
        "driver": driver, "opponent": opponent, "episodes": episodes, "seed": seed,
        "win_rate": sum(r["win"] for r in rows) / episodes,
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_damage_dealt": float(np.mean([r["damage_dealt"] for r in rows])),
        "mean_damage_taken": float(np.mean([r["damage_taken"] for r in rows])),
        "hit_rate": (hit / atk) if atk else 0.0,
        "mean_ticks": float(np.mean([r["ticks"] for r in rows])),
        "mean_hp": float(np.mean([r["hp"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="logs/baselines.json")
    a = ap.parse_args()
    out = {"A_random_vs_melee": run_episodes("random", "melee", a.episodes, a.seed)}
    print(out["A_random_vs_melee"])
    out["B_scripted_vs_melee"] = run_episodes("scripted", "melee", a.episodes, a.seed)
    print(out["B_scripted_vs_melee"])
    if a.checkpoint:
        out["C_ppo_vs_melee"] = run_episodes("ppo", "melee", a.episodes, a.seed,
                                             checkpoint=a.checkpoint)
        print(out["C_ppo_vs_melee"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
