"""Resume: train -> save -> terminate -> resume continues step count."""
import os


def test_resume_continues_timesteps(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.environment.gym_env import PvPEnv

    vec = DummyVecEnv([lambda: Monitor(PvPEnv(opponent="dummy", max_ticks=100))])
    m1 = PPO("MlpPolicy", vec, verbose=0, seed=0, n_steps=32, batch_size=16, n_epochs=1)
    m1.learn(total_timesteps=64)
    ckpt = os.path.join(str(tmp_path), "latest.zip")
    m1.save(ckpt)
    n1 = m1.num_timesteps
    assert n1 >= 64
    del m1
    # simulate restart: fresh process state, load checkpoint
    m2 = PPO.load(ckpt, env=vec)
    assert m2.num_timesteps == n1
    m2.learn(total_timesteps=64, reset_num_timesteps=False)
    assert m2.num_timesteps == n1 + 64
    vec.close()
