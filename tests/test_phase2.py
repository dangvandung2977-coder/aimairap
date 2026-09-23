"""Phase 2: profiles, kb/in-range tracking, curriculum, obs v2 determinism."""
import numpy as np

from rl.environment import action_map
from rl.environment.gym_env import PvPEnv
from rl.opponents.scripted import PROFILES, create, create_profile
from sandbox.simulation.sim import PvPSim


def test_all_profiles_fight_and_differ():
    seen = set()
    for name in list(PROFILES) + ["random"]:
        foe = create(name, seed=7)
        sim = PvPSim(seed=7)
        for _ in range(100):
            idle = {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
                    "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}
            sim.step(idle, foe.act(sim, 1))
            if sim.done:
                break
        seen.add(round(float(sim.players[0].health), 1))
        assert sim.damage_dealt[1] > 0, name  # every profile lands hits on idle foe
    assert len(seen) > 1  # profiles behave differently


def test_profiles_deterministic():
    def run():
        foe = create("aggressive", seed=3)
        sim = PvPSim(seed=3)
        for _ in range(60):
            idle = {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
                    "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}
            sim.step(idle, foe.act(sim, 1))
        return sim.players[0].pos.copy(), sim.players[0].health
    p1, h1 = run()
    p2, h2 = run()
    np.testing.assert_array_equal(p1, p2)
    assert h1 == h2


def test_kb_and_in_range_tracked():
    sim = PvPSim(seed=0)
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [12, 0, 10]
    sim.players[0].yaw = 0.0
    atk = {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
           "attack": True, "yaw_delta": 0.0, "pitch_delta": 0.0}
    idle = dict(atk)
    idle["attack"] = False
    sim.step(atk, idle)
    assert sim.hits[0] == 1 and sim.attempts_in_range[0] == 1
    assert sim.kb_dealt[0] > 0.0


def test_obs_v2_deterministic_and_versioned():
    from rl.environment import obs_schema
    assert obs_schema.OBS_DIMS[2] == 44
    assert obs_schema.OBS_DIMS[3] == 47
    env = PvPEnv(opponent="strafer", max_ticks=200)
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(10):
        act = action_map.random_action(rng)
        o1, _, _, _, _ = env.step(act)
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(10):
        act = action_map.random_action(rng)
        o2, _, _, _, _ = env.step(act)
    np.testing.assert_array_equal(o1, o2)


def test_set_opponent_switches_foe():
    env = PvPEnv(opponent="melee")
    env.reset(seed=0)
    assert env.opponent_name == "melee"
    env.set_opponent("defensive", {"seed": 5}, seed=5)
    assert env.opponent_name == "defensive"
    c = env.opponent.act(env.sim, 1)
    assert set(c) == {"move_x", "move_z", "sprint", "jump", "attack",
                      "yaw_delta", "pitch_delta"}
