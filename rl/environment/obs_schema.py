"""Versioned observation schema. v1: Box[39]; v2: Box[44] (adds aim bearing,
foe speed, foe recent displacement). v1 builder kept for baseline comparison."""
from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

OBS_VERSION = 4
OBS_DIMS = {1: 39, 2: 44, 3: 47, 4: 73}
OBS_DIM = OBS_DIMS[OBS_VERSION]


def obs_space(version: int = OBS_VERSION) -> spaces.Box:
    return spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIMS[version],), dtype=np.float32)


def _clip(x: float) -> float:
    return float(max(-5.0, min(5.0, x)))


def build_obs(sim, agent: int = 0, version: int = OBS_VERSION) -> np.ndarray:
    if version == 1:
        return build_obs_v1(sim, agent)
    if version == 2:
        return build_obs_v2(sim, agent)
    if version == 3:
        return build_obs_v3(sim, agent)
    assert version == 4, version
    base = build_obs_v3(sim, agent).tolist()
    # v4 (+26): inventory / effects / utility. Only self-known info plus
    # directly observable foe state (shield, absorption, eating) — no hidden
    # foe inventory.
    from sandbox.items import effects as FX
    from sandbox.items.defs import HOTBAR
    me, foe = sim.players[agent], sim.players[1 - agent]
    inv, finv = sim.inventories[agent], sim.inventories[1 - agent]
    base += [_clip(inv.selected / 8.0)]
    base += [_clip(inv.counts.get(it, 0) / 16.0) for it in HOTBAR]
    base += [_clip(inv.cooldowns.get("ender_pearl", 0) / 20.0),
             _clip(inv.cooldowns.get("golden_apple", 0) / 100.0),
             _clip(inv.cooldowns.get("bow", 0) / 15.0)]
    base += [_clip(me.absorption_hp / 8.0), 1.0 if me.shield_up else 0.0]
    base += [_clip(me.eating[1] / 32.0) if me.eating else 0.0]
    base += [_clip(me.hunger / 20.0)]
    base += [_clip(foe.absorption_hp / 8.0), 1.0 if foe.shield_up else 0.0,
             1.0 if foe.eating else 0.0]
    nproj = len(sim.projectiles)
    base += [_clip(nproj / 4.0)]
    if nproj:
        d = sim.projectiles[0].pos - me.pos
        base += [_clip(d[0] / 20.0), _clip(d[1] / 10.0), _clip(d[2] / 20.0)]
    else:
        base += [0.0, 0.0, 0.0]
    base += [_clip(FX.effect_time(me, "speed") / 1200.0),
             _clip(FX.effect_time(me, "strength") / 1200.0)]
    assert len(base) == OBS_DIMS[4], len(base)
    arr = np.array(base, dtype=np.float32)
    assert np.all(np.isfinite(arr))
    return arr


def build_obs_v3(sim, agent: int = 0) -> np.ndarray:
    base = build_obs_v2(sim, agent).tolist()
    # v3 (+3): ticks since own / foe last attack attempt, own last attempt hit?
    # Lets the policy distinguish recent hit vs recent miss and punish windows.
    own_since = min(sim.tick - sim.last_attempt_tick[agent], 100) / 20.0
    foe_since = min(sim.tick - sim.last_attempt_tick[1 - agent], 100) / 20.0
    base += [_clip(own_since), _clip(foe_since),
             float(sim.last_attempt_hit[agent])]
    assert len(base) == OBS_DIMS[3], len(base)
    arr = np.array(base, dtype=np.float32)
    assert np.all(np.isfinite(arr))
    return arr


def build_obs_v2(sim, agent: int = 0) -> np.ndarray:
    base = build_obs_v1(sim, agent).tolist()
    me, foe = sim.players[agent], sim.players[1 - agent]
    size = float(sim.world.size)
    d = foe.pos - me.pos
    # aim bearing: horizontal angle to foe relative to own yaw (explicit aim error)
    bearing = math.atan2(d[2], d[0]) - math.radians(me.yaw)
    base += [math.sin(bearing), math.cos(bearing)]
    # foe horizontal speed
    base += [_clip(float(np.linalg.norm(foe.vel[[0, 2]])) / 10.0)]
    # foe recent displacement (oldest stored -> now, ~4 ticks), normalized
    old = sim.prev_pos[1 - agent][0]
    dd = foe.pos - old
    base += [_clip(dd[0] / size), _clip(dd[2] / size)]
    assert len(base) == OBS_DIMS[2], len(base)
    arr = np.array(base, dtype=np.float32)
    assert np.all(np.isfinite(arr))
    return arr


def build_obs_v1(sim, agent: int = 0) -> np.ndarray:
    me, foe = sim.players[agent], sim.players[1 - agent]
    size = float(sim.world.size)
    out: list[float] = []
    # SELF (13)
    out += [_clip(me.pos[0] / size), _clip(me.pos[1] / 10.0), _clip(me.pos[2] / size)]
    out += [_clip(me.vel[0] / 10.0), _clip(me.vel[1] / 10.0), _clip(me.vel[2] / 10.0)]
    out += [math.sin(math.radians(me.yaw)), math.cos(math.radians(me.yaw))]
    out += [_clip(me.pitch / 90.0)]
    out += [_clip(me.health / me.max_health)]
    out += [1.0 if me.on_ground else 0.0]
    out += [_clip(me.attack_cooldown / 12.0), _clip(me.hurt_time / 10.0)]
    # ENEMY (13): relative state from agent frame
    d = foe.pos - me.pos
    out += [_clip(d[0] / size), _clip(d[1] / 10.0), _clip(d[2] / size)]
    dv = foe.vel - me.vel
    out += [_clip(dv[0] / 10.0), _clip(dv[1] / 10.0), _clip(dv[2] / 10.0)]
    dyaw = math.radians(foe.yaw - me.yaw)
    out += [math.sin(dyaw), math.cos(dyaw)]
    out += [_clip(foe.pitch / 90.0)]
    out += [_clip(foe.health / foe.max_health)]
    out += [1.0 if foe.on_ground else 0.0]
    out += [_clip(foe.attack_cooldown / 12.0), _clip(foe.hurt_time / 10.0)]
    # COMBAT (5)
    dist = float(np.linalg.norm(d))
    out += [_clip(dist / size)]
    out += [_clip(sim.damage_dealt[agent] / 20.0), _clip(sim.damage_dealt[1 - agent] / 20.0)]
    out += [_clip(sim.combo[agent] / 5.0)]
    tsh = sim.time_since_hit[agent]
    out += [_clip(min(tsh, 100) / 20.0)]
    # WORLD (8)
    wd = sim.world.wall_distances(me.pos) / size
    out += [_clip(float(v)) for v in wd]
    out += [float(v) for v in sim.world.obstacle_sensor(me.pos)]
    assert len(out) == OBS_DIMS[1], len(out)
    arr = np.array(out, dtype=np.float32)
    assert np.all(np.isfinite(arr))
    return arr


def obs_summary(obs: np.ndarray) -> str:
    o = np.asarray(obs, dtype=float).flatten()
    return (f"hp={o[9]:.2f} foeHp={o[22]:.2f} dist={o[26]:.2f} "
            f"combo={o[29]:.2f} walls={o[31]:.2f},{o[32]:.2f}")
