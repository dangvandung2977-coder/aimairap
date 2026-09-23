import numpy as np

from rl.environment import action_map
from rl.environment.gym_env import PvPEnv


def _random_episodes(opponent: str, episodes: int = 3, seed: int = 0):
    env = PvPEnv(opponent=opponent, max_ticks=200)
    rng = np.random.default_rng(seed)
    totals = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        assert obs.shape == (73,) and np.all(np.isfinite(obs))
        done = False
        total = 0.0
        steps = 0
        while not done:
            act = action_map.random_action(rng)
            obs, rew, term, trunc, info = env.step(act)
            assert obs.shape == (73,)
            assert np.all(np.isfinite(obs)) and np.isfinite(rew)
            assert not np.any(np.isnan(obs)) and rew not in (np.inf, -np.inf)
            total += rew
            steps += 1
            done = term or trunc
            assert steps < 500
        totals.append(total)
    return totals


def test_random_episodes_dummy():
    _random_episodes("dummy", episodes=2)


def test_random_episodes_melee():
    _random_episodes("melee", episodes=2)


def test_random_episodes_strafe():
    _random_episodes("strafe", episodes=2)


def test_parallel_envs_headless():
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.environment.gym_env import make_env
    vec = DummyVecEnv([make_env("dummy", seed=i, max_ticks=100) for i in range(4)])
    obs = vec.reset()
    assert obs.shape == (4, 73)
    for _ in range(5):
        acts = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 1]] * 4)
        obs, rews, dones, infos = vec.step(acts)
        assert obs.shape == (4, 73) and np.all(np.isfinite(obs))
    vec.close()
