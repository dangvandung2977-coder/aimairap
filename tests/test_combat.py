import numpy as np

from sandbox.config import CombatConfig, PhysicsConfig, WorldConfig
from sandbox.combat.melee import try_attack
from sandbox.entities.player import PlayerState
from sandbox.simulation.sim import PvPSim, idle_ctrl


def _face(a: PlayerState, b: PlayerState):
    d = b.pos - a.pos
    a.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
    a.pitch = 0.0


def test_attack_outside_range_misses():
    cfg = CombatConfig()
    a, b = PlayerState(), PlayerState()
    a.reset([5, 0, 10], yaw=0)
    b.reset([15, 0, 10], yaw=180)
    _face(a, b)
    ev = try_attack(a, b, cfg)
    assert ev["attempted"] and not ev["hit"]
    assert b.health == cfg.max_health


def test_attack_inside_range_hits_and_damages():
    cfg = CombatConfig()
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0)
    b.reset([12, 0, 10], yaw=180)
    _face(a, b)
    ev = try_attack(a, b, cfg)
    assert ev["hit"] and ev["damage"] == cfg.base_damage
    assert b.health == cfg.max_health - cfg.base_damage
    assert a.attack_cooldown == cfg.cooldown_ticks
    assert b.hurt_time == cfg.hurt_ticks


def test_attack_facing_away_misses():
    cfg = CombatConfig()
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0)
    b.reset([12, 0, 10], yaw=180)
    _face(a, b)
    a.yaw += 180.0
    if a.yaw > 180.0:
        a.yaw -= 360.0
    ev = try_attack(a, b, cfg)
    assert not ev["hit"]


def test_cooldown_blocks_second_attack():
    cfg = CombatConfig()
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0)
    b.reset([12, 0, 10], yaw=180)
    _face(a, b)
    ev1 = try_attack(a, b, cfg)
    assert ev1["hit"]
    ev2 = try_attack(a, b, cfg)
    assert not ev2["attempted"] and not ev2["hit"]


def test_knockback_pushes_victim_away():
    cfg = CombatConfig()
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0)
    b.reset([12, 0, 10], yaw=180)
    _face(a, b)
    v0 = b.vel.copy()
    try_attack(a, b, cfg)
    moved = b.vel - v0
    # victim at +X of attacker -> pushed +X, popped up
    assert moved[0] > 0.5 and b.vel[1] > 0.0


def test_crit_when_falling():
    cfg = CombatConfig()
    a, b = PlayerState(), PlayerState()
    a.reset([10, 2, 10], yaw=0)
    b.reset([12, 0, 10], yaw=180)
    _face(a, b)
    a.on_ground = False
    a.vel[1] = -5.0
    ev = try_attack(a, b, cfg)
    assert ev["hit"] and ev["crit"]
    assert ev["damage"] == cfg.base_damage * cfg.crit_mult


def test_death_and_reset():
    sim = PvPSim(seed=1)
    a, b = sim.players[0], sim.players[1]
    # teleport into range, face each other
    a.pos[:] = [10, 0, 10]
    b.pos[:] = [12, 0, 10]
    d = b.pos - a.pos
    a.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
    b.health = 3.0
    c = idle_ctrl()
    c["attack"] = True
    sim.step(c, idle_ctrl())
    assert not b.alive and sim.done and sim.winner == 0
    sim.reset(seed=1)
    assert sim.players[1].alive and sim.players[1].health == sim.combat.max_health
    assert not sim.done and sim.winner is None
