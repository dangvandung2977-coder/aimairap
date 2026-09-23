"""Random-policy episodes across opponents; prints metrics."""
import numpy as np

from rl.environment.gym_env import PvPEnv


def main(episodes: int = 5, opponent: str = "melee", seed: int = 0):
    for opp in (["dummy", "melee", "strafe"] if opponent == "all" else [opponent]):
        env = PvPEnv(opponent=opp, max_ticks=900)
        rng = np.random.default_rng(seed)
        wins = kills = 0
        rews, lens, dealt, hp = [], [], [], []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done, total, steps = False, 0.0, 0
            while not done:
                act = action_map.random_controller(rng)
                obs, rew, term, trunc, info = env.step(act)
                total += rew
                steps += 1
                done = term or trunc
            rews.append(total)
            lens.append(steps)
            dealt.append(info["damage_dealt"])
            hp.append(info["hp"])
            wins += info.get("kills", 0)
            kills += info.get("kills", 0)
        print(f"[{opp}] win_rate={wins/episodes:.2f} avg_rew={np.mean(rews):+.2f} "
              f"avg_len={np.mean(lens):.0f} avg_dealt={np.mean(dealt):.1f} "
              f"avg_hp={np.mean(hp):.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--opponent", default="all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.episodes, a.opponent, a.seed)
