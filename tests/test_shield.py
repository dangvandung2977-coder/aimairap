"""Shield mechanics regression: frontal block, flank bypass, timing."""
import numpy as np

from sandbox.combat.melee import try_attack
from sandbox.config import CombatConfig
from sandbox.entities.player import PlayerState


def _swing(axy, bxy, byaw):
    a, b = PlayerState(), PlayerState()
    a.reset([axy[0], 0, axy[1]], yaw=0.0)
    b.reset([bxy[0], 0, bxy[1]], yaw=byaw)
    d = b.pos - a.pos
    a.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
    b.shield_up = True
    return try_attack(a, b, CombatConfig())


def test_frontal_blocked():
    ev = _swing((10, 10), (12, 10), 180)
    assert ev.get("blocked") and ev["damage"] == 0.0


def test_behind_bypasses():
    ev = _swing((10, 10), (12, 10), 0)
    assert not ev.get("blocked") and ev["damage"] == 5.0


def test_side_bypasses():
    ev = _swing((10, 10), (12, 10), 90)
    assert not ev.get("blocked") and ev["damage"] == 5.0


def test_no_shield_no_block():
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0.0)
    b.reset([12, 0, 10], yaw=180.0)
    ev = try_attack(a, b, CombatConfig())
    assert not ev.get("blocked") and ev["damage"] == 5.0
