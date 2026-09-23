"""Record a scripted demo replay for the viewer."""
import argparse

import numpy as np

from rl.environment.gym_env import PvPEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="melee")
    ap.add_argument("--out", default="replays/episode_latest.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = PvPEnv(opponent=a.opponent, record=True, max_ticks=600)
    obs, _ = env.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)
    done = False
    while not done:
        act = np.array([rng.integers(5), 1, rng.integers(2), 1, rng.integers(3), 1])
        obs, rew, term, trunc, info = env.step(act)
        done = term or trunc
    path = env.save_replay(a.out)
    print(f"saved {path} winner={info['winner']} dealt={info['damage_dealt']:.1f}")


if __name__ == "__main__":
    main()
