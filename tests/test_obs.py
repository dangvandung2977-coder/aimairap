"""Observation + Gymnasium API validation."""
import numpy as np

from rl.environment import action_map
from rl.environment import obs_schema
from rl.environment.gym_env import PvPEnv


def test_obs_shape_stable_and_finite():
    env = PvPEnv(opponent="melee", max_ticks=200)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (obs_schema.OBS_DIMS[4],) == (73,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["obs_version"] == 4
    for _ in range(60):
        act = action_map.random_action(rng)
        obs, rew, term, trunc, info = env.step(act)
        assert isinstance(rew, float) and np.isfinite(rew)
        assert isinstance(term, bool) and isinstance(trunc, bool)
        assert isinstance(info, dict)
        assert obs.shape == (73,) and np.all(np.isfinite(obs))
        assert -5.0 <= obs.min() and obs.max() <= 5.0
        # required stat keys
        for k in ("damage_dealt", "damage_taken", "attacks", "hits", "hit_rate",
                  "reward_components", "win", "loss", "kb_dealt", "kb_taken",
                  "attacks_in_range"):
            assert k in info, k
        if term or trunc:
            break


def test_obs_v1_compat_for_baseline():
    env = PvPEnv(opponent="dummy", obs_version=1)
    assert env.observation_space.shape == (39,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (39,) and np.all(np.isfinite(obs))
    assert info["obs_version"] == 1


def test_reset_valid_after_done():
    env = PvPEnv(opponent="dummy", max_ticks=40)
    env.reset(seed=1)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([1, 1, 0, 1, 0, 0, 0, 1, 1]))
        done = term or trunc
    obs, _ = env.reset(seed=2)
    assert obs.shape == (73,) and np.all(np.isfinite(obs))


def test_gym_spaces_match():
    env = PvPEnv()
    assert env.observation_space.shape == (73,)
    assert list(env.action_space.nvec) == [5, 2, 2, 2, 2, 2, 9, 3, 3]
    assert env.action_space.contains(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert not env.action_space.contains(np.array([5, 0, 0, 0, 0, 0, 0, 0, 0]))
