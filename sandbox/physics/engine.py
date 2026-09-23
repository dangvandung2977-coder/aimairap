"""Minecraft-like movement physics. Pure + deterministic (no RNG)."""
from __future__ import annotations

import math

import numpy as np

from sandbox.config import PhysicsConfig
from sandbox.world.arena import VoxelWorld
from sandbox.entities.player import PlayerState


def wrap_yaw(yaw: float) -> float:
    while yaw > 180.0:
        yaw -= 360.0
    while yaw <= -180.0:
        yaw += 360.0
    return yaw


def look_dir(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    cp = math.cos(p)
    return np.array([math.cos(y) * cp, math.sin(p), math.sin(y) * cp], dtype=np.float64)


def step_player(p: PlayerState, ctrl: dict, world: VoxelWorld, cfg: PhysicsConfig) -> None:
    """Advance one physics tick. ctrl keys: move_x(-1..1, strafe right+),
    move_z(-1..1, forward+), sprint, jump, yaw_delta, pitch_delta."""
    if not p.alive:
        p.vel[:] = 0.0
        p.sprinting = False
        return
    dt = cfg.dt
    # rotation
    p.yaw = wrap_yaw(p.yaw + float(ctrl.get("yaw_delta", 0.0)))
    p.pitch = max(-90.0, min(90.0, p.pitch + float(ctrl.get("pitch_delta", 0.0))))

    mx = float(np.clip(ctrl.get("move_x", 0.0), -1.0, 1.0))
    mz = float(np.clip(ctrl.get("move_z", 0.0), -1.0, 1.0))
    want_sprint = bool(ctrl.get("sprint", False)) and mz > 0.0
    p.sprinting = want_sprint

    # wish direction (yaw space)
    yr = math.radians(p.yaw)
    fwd = np.array([math.cos(yr), 0.0, math.sin(yr)], dtype=np.float64)
    right = np.array([-math.sin(yr), 0.0, math.cos(yr)], dtype=np.float64)
    wish = fwd * mz + right * mx
    n = float(np.linalg.norm(wish))
    if n > 1.0:
        wish = wish / n
    speed = cfg.sprint_speed if want_sprint else cfg.walk_speed
    try:
        from sandbox.items import effects as _FX
        fl = world.field.fluid_at(float(p.pos[0]), float(p.pos[2])) \
            if world.field is not None else None
        web = world.field.web_at(float(p.pos[0]), float(p.pos[2])) \
            if world.field is not None else False
        speed *= _FX.speed_mult(p, 0.6 if fl == "water" else 0.5 if fl == "lava" else 1.0, web)
    except ImportError:
        pass
    target = wish * speed

    accel = cfg.ground_accel if p.on_ground else cfg.air_accel
    k = min(1.0, accel * dt)
    p.vel[0] += (target[0] - p.vel[0]) * k
    p.vel[2] += (target[2] - p.vel[2]) * k

    # jump
    if bool(ctrl.get("jump", False)) and p.on_ground:
        p.vel[1] = cfg.jump_velocity
        p.on_ground = False

    # gravity
    if not p.on_ground:
        p.vel[1] = max(cfg.terminal_velocity, p.vel[1] - cfg.gravity * dt)
        if p.vel[1] < 0.0:
            p.fall_distance += -p.vel[1] * dt
    else:
        if p.vel[1] < 0.0:
            p.vel[1] = 0.0

    prev_y = float(p.pos[1])
    new_pos, on_ground, _ = world.collide_move(p.pos, p.vel, dt, cfg.player_radius, cfg.player_height)
    # zero vertical vel on landing (floor/pillar), keep horizontal (knockback slides)
    if on_ground and p.vel[1] <= 0.0:
        p.vel[1] = 0.0
    p.pos = new_pos
    p.on_ground = on_ground
    if on_ground:
        p.fall_distance = 0.0
    elif float(p.pos[1]) <= prev_y and p.vel[1] <= 0.0:
        pass  # falling: fall_distance already accumulated

    # timers
    if p.attack_cooldown > 0:
        p.attack_cooldown -= 1
    if p.hurt_time > 0:
        p.hurt_time -= 1
