"""Every action branch maps to a sane controller; movement works end-to-end."""
import numpy as np

from rl.environment import action_map
from sandbox.simulation.sim import PvPSim


def test_all_moves_produce_motion_or_stop():
    for move in range(5):
        sim = PvPSim(seed=0)
        p0 = sim.players[0].pos.copy()
        c0 = action_map.to_controller([move, 0, 0, 0, 0, 0, 0, 1, 1])
        idle = action_map.to_controller([0, 0, 0, 0, 0, 0, 0, 1, 1])
        for _ in range(30):
            sim.step(c0, idle)
        moved = float(np.linalg.norm(sim.players[0].pos - p0))
        if move == 0:
            assert moved < 3.0  # stop: only knockback-free drift (foe idle)
        else:
            assert moved > 1.0, move


def test_sprint_jump_attack_flags():
    c = action_map.to_controller([1, 1, 1, 1, 0, 0, 0, 1, 1])
    assert c == {"move_x": 0.0, "move_z": 1.0, "sprint": True, "jump": True,
                 "attack": True, "block": False, "use_item": False,
                 "select_slot": 0, "yaw_delta": 0.0, "pitch_delta": 0.0}
    left = action_map.to_controller([3, 0, 0, 0, 1, 1, 4, 0, 2])
    assert left["move_x"] == -1.0 and left["yaw_delta"] == -15.0
    assert left["pitch_delta"] == 10.0
    assert left["block"] and left["use_item"] and left["select_slot"] == 4
    right = action_map.to_controller([4, 0, 0, 0, 0, 0, 0, 2, 0])
    assert right["move_x"] == 1.0 and right["yaw_delta"] == 15.0
    assert right["pitch_delta"] == -10.0


def test_jump_lifts_agent_in_env():
    from rl.environment.gym_env import PvPEnv
    env = PvPEnv(opponent="dummy")
    env.reset(seed=0)
    y0 = float(env.sim.players[0].pos[1])
    env.step(np.array([0, 0, 1, 0, 0, 0, 0, 1, 1]))  # jump held: airborne after 1 decision
    assert float(env.sim.players[0].pos[1]) > y0 and \
        not env.sim.players[0].on_ground


def test_attack_action_consumes_cooldown():
    sim = PvPSim(seed=0)
    sim.players[0].pos[:] = [10, 0, 10]
    sim.players[1].pos[:] = [12, 0, 10]
    sim.players[0].yaw = 0.0
    atk = action_map.to_controller([0, 0, 0, 1, 0, 0, 0, 1, 1])
    idle = action_map.to_controller([0, 0, 0, 0, 0, 0, 0, 1, 1])
    sim.step(atk, idle)
    assert sim.attempts[0] == 1 and sim.players[0].attack_cooldown > 0
