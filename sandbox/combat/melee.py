"""Deterministic sword melee. Pure function over two PlayerStates."""
from __future__ import annotations

import math

import numpy as np

from sandbox.config import CombatConfig
from sandbox.entities.player import PlayerState
from sandbox.physics.engine import look_dir

CENTER_H = 0.9


def _centers(a: PlayerState, b: PlayerState):
    ac = a.pos + np.array([0.0, CENTER_H, 0.0])
    bc = b.pos + np.array([0.0, CENTER_H, 0.0])
    return ac, bc


def try_attack(att: PlayerState, vic: PlayerState, cfg: CombatConfig) -> dict:
    """Attempt one attack. Always consumes cooldown. Returns event dict."""
    out = {"attempted": True, "hit": False, "damage": 0.0, "crit": False,
           "killed": False, "in_range": False, "kb": 0.0}
    if not att.alive or not vic.alive:
        return out
    if att.attack_cooldown > 0:
        out["attempted"] = False
        return out
    att.attack_cooldown = cfg.cooldown_ticks
    ac, bc = _centers(att, vic)
    delta = bc - ac
    dist = float(np.linalg.norm(delta))
    if dist > cfg.attack_range:
        return out
    out["in_range"] = True
    if dist > 1e-9:
        look = look_dir(att.yaw, att.pitch)
        if cfg.attack_arc_vertical_deg == cfg.attack_arc_deg:
            # baseline single-cone check (bit-identical to phase <= 2.75)
            cos_arc = math.cos(math.radians(cfg.attack_arc_deg))
            if float(np.dot(look, delta / dist)) < cos_arc:
                return out
        else:
            # split check: horizontal cone + independent vertical tolerance
            dh = delta.copy()
            dh[1] = 0.0
            hn = float(np.linalg.norm(dh))
            lh = look.copy()
            lh[1] = 0.0
            ln = float(np.linalg.norm(lh))
            if hn < 1e-9 or ln < 1e-9:
                return out
            cos_h = math.cos(math.radians(cfg.attack_arc_deg))
            if float(np.dot(lh / ln, dh / hn)) < cos_h:
                return out
            elev = math.degrees(math.asin(max(-1.0, min(1.0, delta[1] / dist))))
            look_elev = att.pitch
            if abs(elev - look_elev) > cfg.attack_arc_vertical_deg:
                return out
    # hit
    crit = (not att.on_ground) and float(att.vel[1]) < -1.0
    dmg = cfg.base_damage * (cfg.crit_mult if crit else 1.0)
    try:
        from sandbox.items import effects as _FX
        dmg *= _FX.damage_mult(att)
    except ImportError:
        pass
    # shield block: victim must face the attacker
    blocked = False
    if vic.shield_up:
        back = -delta / (dist or 1.0)
        vlook = look_dir(vic.yaw, vic.pitch)
        if float(np.dot(vlook, back)) > 0.2:
            blocked = True
    if blocked:
        out.update({"hit": True, "damage": 0.0, "crit": False, "blocked": True})
        vic.hurt_time = 4
        hdir = delta.copy()
        hdir[1] = 0.0
        hn = float(np.linalg.norm(hdir)) or 1.0
        vic.vel[0] += hdir[0] / hn * 1.2
        vic.vel[2] += hdir[2] / hn * 1.2
        return out
    absorbed = min(vic.absorption_hp, dmg)
    vic.absorption_hp -= absorbed
    vic.health = max(0.0, vic.health - (dmg - absorbed))
    vic.hurt_time = 10 if cfg.hurt_ticks == 10 else cfg.hurt_ticks
    # knockback away from attacker, horizontal + pop-up
    hdir = delta.copy()
    hdir[1] = 0.0
    hn = float(np.linalg.norm(hdir))
    if hn < 1e-9:
        hdir = look_dir(att.yaw, 0.0)
        hdir[1] = 0.0
        hn = float(np.linalg.norm(hdir)) or 1.0
    hdir = hdir / hn
    kb = cfg.knockback + (cfg.knockback_sprint_bonus if att.sprinting else 0.0)
    kb_speed = kb * 8.0
    vic.vel[0] += hdir[0] * kb_speed
    vic.vel[2] += hdir[2] * kb_speed
    vic.vel[1] += cfg.knockback_up * 4.5
    out["kb"] = float(kb_speed)
    vic.on_ground = False
    out.update({"hit": True, "damage": float(dmg), "crit": bool(crit)})
    if vic.health <= 0.0:
        vic.alive = False
        vic.vel[:] = 0.0
        out["killed"] = True
    return out
