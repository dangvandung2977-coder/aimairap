"""Short PPO run: starts, updates, checkpoints, loads, evaluates."""
import os

import numpy as np
import torch


def test_ppo_smoke(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.environment.gym_env import PvPEnv

    def _mk(seed):
        def _t():
            return Monitor(PvPEnv(opponent="dummy", seed=seed, max_ticks=120))
        return _t

    vec = DummyVecEnv([_mk(0), _mk(1)])
    model = PPO("MlpPolicy", vec, verbose=0, seed=0, n_steps=64,
                batch_size=32, n_epochs=2)
    params_before = [p.clone() for p in model.policy.parameters()]
    model.learn(total_timesteps=256)
    params_after = [p.clone() for p in model.policy.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after))
    ckpt = os.path.join(str(tmp_path), "ppo_smoke.zip")
    model.save(ckpt)
    assert os.path.exists(ckpt)
    del model
    loaded = PPO.load(ckpt, env=vec)
    env = PvPEnv(opponent="dummy", max_ticks=120)
    obs, _ = env.reset(seed=0)
    done, total = False, 0.0
    while not done:
        act, _ = loaded.predict(obs, deterministic=True)
        obs, rew, term, trunc, _ = env.step(act)
        total += rew
        done = term or trunc
    assert np.isfinite(total)
    vec.close()
