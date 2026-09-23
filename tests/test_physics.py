import numpy as np

from sandbox.config import PhysicsConfig, WorldConfig
from sandbox.entities.player import PlayerState
from sandbox.physics.engine import step_player
from sandbox.simulation.sim import idle_ctrl
from sandbox.world.arena import VoxelWorld


def _player(x=10.0, z=10.0, y=0.0) -> PlayerState:
    p = PlayerState()
    p.reset([x, y, z], yaw=0.0)
    return p


def test_gravity_pulls_down_and_lands():
    world = VoxelWorld()
    cfg = PhysicsConfig()
    p = _player(y=5.0)
    p.on_ground = False
    for _ in range(200):
        step_player(p, idle_ctrl(), world, cfg)
    assert p.pos[1] == 0.0 and p.on_ground


def test_jump_rises_then_lands():
    world = VoxelWorld()
    cfg = PhysicsConfig()
    p = _player()
    c = idle_ctrl()
    c["jump"] = True
    step_player(p, c, world, cfg)
    assert not p.on_ground and p.vel[1] > 0.0
    peak = p.pos[1]
    c["jump"] = False
    for _ in range(200):
        step_player(p, c, world, cfg)
        peak = max(peak, float(p.pos[1]))
    assert peak > 0.5 and p.on_ground


def test_walls_clamp_position():
    world = VoxelWorld(size=20)
    cfg = PhysicsConfig()
    p = _player(x=18.5, z=10.0)
    c = idle_ctrl()
    c["move_z"] = 1.0  # yaw 0 -> +X, into east wall
    for _ in range(120):
        step_player(p, c, world, cfg)
    assert p.pos[0] <= world.hi - cfg.player_radius + 1e-9


def test_sprint_faster_than_walk():
    world = VoxelWorld()
    cfg = PhysicsConfig()

    def run(sprint: bool) -> float:
        p = _player(x=5.0)
        c = idle_ctrl()
        c["move_z"] = 1.0
        c["sprint"] = sprint
        for _ in range(60):
            step_player(p, c, world, cfg)
        return float(p.pos[0])

    assert run(True) > run(False) + 0.5


def test_no_input_stops_on_ground():
    world = VoxelWorld()
    cfg = PhysicsConfig()
    p = _player()
    c = idle_ctrl()
    c["move_z"] = 1.0
    for _ in range(30):
        step_player(p, c, world, cfg)
    assert abs(p.vel[0]) > 1.0
    c["move_z"] = 0.0
    for _ in range(120):
        step_player(p, idle_ctrl(), world, cfg)
    assert abs(p.vel[0]) < 0.2 and abs(p.vel[2]) < 0.2


def test_obstacle_blocks_movement():
    world = VoxelWorld(size=20, obstacles=[(9.0, 9.5, 11.0, 10.5, 2.0)])
    cfg = PhysicsConfig()
    p = _player(x=7.0, z=10.0)
    c = idle_ctrl()
    c["move_z"] = 1.0
    for _ in range(120):
        step_player(p, c, world, cfg)
    assert p.pos[0] < 9.0  # stopped at pillar face
