"""Phase 3: mechanics variants — versioned, deterministic, baseline-identical."""
import numpy as np

from sandbox.combat.melee import try_attack
from sandbox.config import CombatConfig, PhysicsConfig
from sandbox.entities.player import PlayerState
from sandbox.simulation.sim import PvPSim, idle_ctrl


def _pair(dist: float = 2.0, dy: float = 0.0):
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0.0)
    b.reset([10 + dist, dy, 10], yaw=180.0)
    d = b.pos - a.pos
    a.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
    a.pitch = 0.0
    return a, b


def test_split_arc_identical_at_default():
    cfg = CombatConfig()  # arcV == arcH -> legacy single-cone path
    assert cfg.attack_arc_vertical_deg == cfg.attack_arc_deg
    for dist, dy in [(2.0, 0.0), (2.5, 1.4), (1.5, 2.0), (2.9, 0.5)]:
        a, b = _pair(dist, dy)
        ev = try_attack(a, b, cfg)
        # legacy cone: all these are within 75 deg -> hit
        assert ev["hit"], (dist, dy)


def test_narrow_vertical_arc_blocks_high_angle():
    cfg = CombatConfig(attack_arc_vertical_deg=30.0)
    a, b = _pair(2.0, 1.4)  # elev ~35 deg, attacker pitch 0 -> miss
    ev = try_attack(a, b, cfg)
    assert ev["in_range"] and not ev["hit"]
    # looking up fixes it: vertical aim matters now
    a2, b2 = _pair(2.0, 1.4)
    a2.pitch = 35.0
    ev2 = try_attack(a2, b2, cfg)
    assert ev2["hit"]


def test_crit_mult_scales_damage():
    for mult, expect in ((1.5, 7.5), (1.0, 5.0), (1.25, 6.25)):
        cfg = CombatConfig(crit_mult=mult)
        a, b = _pair()
        a.on_ground = False
        a.vel[1] = -5.0
        ev = try_attack(a, b, cfg)
        assert ev["crit"] and ev["damage"] == expect


def test_landing_delay_blocks_attack():
    cfg = CombatConfig(landing_attack_delay_ticks=4)
    sim = PvPSim(combat_cfg=cfg, seed=0)
    sim.players[0].pos[:] = [10, 2, 10]
    sim.players[0].on_ground = False
    sim.players[0].vel[1] = -8.0
    sim.players[1].pos[:] = [12, 0, 10]
    atk = dict(idle_ctrl())
    atk["attack"] = True
    # fall untouched until touchdown
    while not sim.players[0].on_ground:
        sim.step(idle_ctrl(), idle_ctrl())
    assert sim.attempts[0] == 0
    # first attack right after landing is swallowed by recovery
    sim.step(atk, idle_ctrl())
    assert sim.players[0].attack_cooldown >= 3
    assert sim.attempts[0] == 0  # lockout consumed no attempts


def test_baseline_reproducible_with_new_fields():
    s1 = PvPSim(seed=5)
    s2 = PvPSim(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(100):
        c0 = dict(idle_ctrl())
        c0["attack"] = bool(rng.integers(2))
        s1.step(c0, idle_ctrl())
        c1 = dict(idle_ctrl())
        c1["attack"] = c0["attack"]
        s2.step(c1, idle_ctrl())
    np.testing.assert_array_equal(s1.players[0].pos, s2.players[0].pos)


def test_air_accel_changes_air_authority():
    def redirect(accel: float) -> float:
        sim = PvPSim(phys_cfg=PhysicsConfig(air_accel=accel), seed=0)
        p = sim.players[0]
        p.pos[:] = [10, 3, 10]
        p.on_ground = False
        p.vel[:] = [0, 0, 0]
        c = dict(idle_ctrl())
        c["move_x"] = 1.0
        for _ in range(20):
            sim.step(c, idle_ctrl())
            p.pos[1] = 3.0  # pin airborne
            p.vel[1] = 0.0
            p.on_ground = False
        return float(p.vel[2])  # yaw 0: strafe (move_x) runs along +Z
    assert redirect(2.5) > redirect(0.6) + 0.5
