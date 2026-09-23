"""Evaluate a checkpoint vs an opponent; record replay + action histogram."""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from rl.environment.gym_env import PvPEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="melee")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", default="replays/episode_latest.npz")
    a = ap.parse_args()
    model = PPO.load(a.checkpoint)
    wins = 0
    lens, rews, dealt, taken, hp = [], [], [], [], []
    acts = []
    for ep in range(a.episodes):
        env = PvPEnv(opponent=a.opponent, record=(ep == 0))
        obs, _ = env.reset(seed=a.seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            acts.append(np.asarray(act).copy())
            obs, rew, term, trunc, info = env.step(act)
            total += rew
            steps += 1
            done = term or trunc
        if ep == 0:
            env.save_replay(a.replay)
            print(f"saved replay {a.replay}")
        wins += info.get("kills", 0)
        lens.append(steps)
        rews.append(total)
        dealt.append(info["damage_dealt"])
        taken.append(info["damage_taken"])
        hp.append(info["hp"])
    acts = np.array(acts)
    print(f"[{a.opponent}] win_rate={wins/a.episodes:.2f} avg_rew={np.mean(rews):+.2f} "
          f"avg_len={np.mean(lens):.0f} dealt={np.mean(dealt):.1f} "
          f"taken={np.mean(taken):.1f} hp_left={np.mean(hp):.1f}")
    for branch, n in enumerate([5, 2, 2, 2, 3, 3]):
        hist = np.bincount(acts[:, branch].astype(int), minlength=n)
        print(f"  branch{branch}: " + " ".join(f"{h/len(acts):.2f}" for h in hist))


if __name__ == "__main__":
    main()
