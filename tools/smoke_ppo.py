"""Short PPO smoke: train ~2k steps vs dummy, save/load/eval."""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.environment.gym_env import PvPEnv


def main():
    os.makedirs("checkpoints", exist_ok=True)
    vec = DummyVecEnv([lambda: Monitor(PvPEnv(opponent="dummy", seed=i, max_ticks=200))
                       for i in range(2)])
    model = PPO("MlpPolicy", vec, verbose=1, seed=0, n_steps=128, batch_size=64)
    model.learn(total_timesteps=2048)
    model.save("checkpoints/ppo_smoke.zip")
    print("saved checkpoints/ppo_smoke.zip")
    loaded = PPO.load("checkpoints/ppo_smoke.zip")
    env = PvPEnv(opponent="dummy", max_ticks=200)
    obs, _ = env.reset(seed=123)
    done, total = False, 0.0
    while not done:
        act, _ = loaded.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(act)
        total += rew
        done = term or trunc
    print(f"eval vs dummy: reward={total:+.2f} winner={info['winner']} "
          f"dealt={info['damage_dealt']:.1f} hp={info['hp']:.1f}")


if __name__ == "__main__":
    main()
