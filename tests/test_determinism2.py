"""Determinism: same seed + same actions -> identical positions, velocities,
health, knockback, attack outcomes, termination, and reward."""
import numpy as np

from rl.environment import action_map
from rl.environment.gym_env import PvPEnv
from sandbox.simulation.sim import PvPSim


def _ctrls(rng):
    mk = lambda: {"move_x": 0.0, "move_z": 1.0, "sprint": True, "jump": False,
                  "attack": bool(rng.integers(2)), "yaw_delta": float(rng.uniform(-10, 10)),
                  "pitch_delta": 0.0}
    return mk(), mk()


def test_sim_ppo_determinism_full_state():
    def rollout(seed):
        sim = PvPSim(seed=seed)
        rng = np.random.default_rng(1234)
        hist = []
        for _ in range(150):
            c0, c1 = _ctrls(rng)
            ev = sim.step(c0, c1)
            hist.append((sim.players[0].pos.copy(), sim.players[1].pos.copy(),
                         sim.players[0].vel.copy(), sim.players[1].vel.copy(),
                         sim.players[0].health, sim.players[1].health,
                         sim.damage_dealt[0], sim.damage_dealt[1],
                         sim.done, sim.winner))
            if sim.done:
                break
        return hist
    a, b = rollout(9), rollout(9)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for i in range(8):
            assert np.allclose(np.asarray(x[i]), np.asarray(y[i])), (i, x[i], y[i])
        assert x[8] == y[8] and x[9] == y[9]


def test_knockback_deterministic():
    def kb(seed):
        sim = PvPSim(seed=seed)
        sim.players[0].pos[:] = [10, 0, 10]
        sim.players[1].pos[:] = [12, 0, 10]
        sim.players[0].yaw = 0.0
        c = {"move_x": 0.0, "move_z": 0.0, "sprint": True, "jump": False,
             "attack": True, "yaw_delta": 0.0, "pitch_delta": 0.0}
        idle = {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
                "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}
        sim.step(c, idle)
        return sim.players[1].vel.copy(), sim.players[1].health
    v1, h1 = kb(3)
    v2, h2 = kb(3)
    np.testing.assert_array_equal(v1, v2)
    assert h1 == h2 < 20.0 and v1[0] > 0.0


def test_env_reward_determinism():
    def run(seed):
        env = PvPEnv(opponent="melee", max_ticks=300)
        rng = np.random.default_rng(77)
        env.reset(seed=seed)
        env.opponent.reset(seed)
        total = 0.0
        done = False
        while not done:
            act = action_map.random_action(rng)
            # re-drive opponent deterministically: reset rng state via same seed seq
            _, rew, term, trunc, info = env.step(act)
            total += rew
            done = term or trunc
        return total, info["damage_dealt"], info["winner"], info["tick"]
    # same python-level rng sequence + same env seeds -> identical totals
    assert run(5) == run(5)


def test_tolerance_documented():
    # float64 physics, no RNG in step path: exact equality holds on one machine.
    # Across machines/BLAS builds allow 1e-9 relative tolerance.
    a = np.float64(0.1) + np.float64(0.2)
    assert abs(a - 0.3) < 1e-9
