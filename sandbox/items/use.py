"""Deterministic item-use pipeline.

Policy -> validate availability -> validate cooldown -> validate conditions
-> start/use -> apply mechanics -> consume -> record event.
Invalid use is a silent no-op and yields no reward.
"""
from __future__ import annotations

import math

import numpy as np

from sandbox.items import effects as FX
from sandbox.items.defs import REGISTRY
from sandbox.combat.projectile import Projectile
from sandbox.physics.engine import look_dir


def _aim_point(sim, idx: int, reach: float = 2.5):
    me = sim.players[idx]
    look = look_dir(me.yaw, me.pitch)
    return me.pos + np.array([look[0] * reach, 0.4, look[2] * reach])


def use_selected(sim, idx: int) -> dict | None:
    """Attempt to use the selected hotbar item. Returns event or None."""
    me = sim.players[idx]
    inv = sim.inventories[idx]
    item_id = inv.selected_item()
    if item_id not in inv.enabled:
        return None
    if not inv.available(item_id) or inv.on_cooldown(item_id):
        return None
    if inv.switch_timer > 0 or me.eating:
        return None
    d = REGISTRY[item_id]
    size = sim.world.size

    if d.use_time_ticks > 0 and item_id != "bow":
        # hold-to-use: eating/drinking starts (vulnerable while active)
        me.eating = [item_id, d.use_time_ticks]
        inv.set_cooldown(item_id)
        return {"type": "use_start", "player": idx, "item": item_id}

    if item_id == "bow":
        if inv.counts.get("arrow", 0) <= 0:
            return None
        inv.consume("arrow")
        inv.set_cooldown(item_id)
        look = look_dir(me.yaw, me.pitch)
        eye = me.pos + np.array([0.0, 1.4, 0.0])
        sim.projectiles.append(Projectile("arrow", idx, eye, look * 30.0,
                                          gravity=9.0, damage=d.damage))
        return {"type": "bow_shot", "player": idx}

    if item_id == "ender_pearl":
        inv.consume(item_id)
        inv.set_cooldown(item_id)
        look = look_dir(me.yaw, me.pitch)
        eye = me.pos + np.array([0.0, 1.4, 0.0])
        sim.projectiles.append(Projectile("pearl", idx, eye, look * 18.0,
                                          gravity=12.0, damage=d.damage))
        return {"type": "pearl_throw", "player": idx}

    if item_id == "cobweb":
        pt = _aim_point(sim, idx)
        if sim.field.add_web(pt[0], pt[2], size):
            inv.consume(item_id)
            inv.set_cooldown(item_id)
            return {"type": "web_place", "player": idx,
                    "pos": [float(pt[0]), float(pt[2])]}
        return None

    if item_id == "block":
        pt = _aim_point(sim, idx)
        gx, gz = round(float(pt[0])), round(float(pt[2]))
        if sim.field.add_block(gx, 0.0, gz, size):
            inv.consume(item_id)
            inv.set_cooldown(item_id)
            return {"type": "block_place", "player": idx, "pos": [gx, 0.0, gz]}
        return None

    if item_id in ("water_bucket", "lava_bucket"):
        pt = _aim_point(sim, idx)
        kind = "water" if item_id == "water_bucket" else "lava"
        if sim.field.add_fluid(kind, float(pt[0]), float(pt[2]), size):
            inv.set_cooldown(item_id)
            return {"type": "fluid_place", "player": idx, "fluid": kind}
        return None

    return None


def tick_eating(sim, idx: int) -> dict | None:
    """Advance eating/drinking. Damage interrupts (progress lost, item kept)."""
    me = sim.players[idx]
    if not me.eating:
        return None
    item_id, left = me.eating
    if me.hurt_time >= 9:  # just took damage -> interrupted
        me.eating = []
        return {"type": "use_interrupt", "player": idx, "item": item_id}
    left -= 1
    me.eating[1] = left
    if left > 0:
        return None
    me.eating = []
    inv = sim.inventories[idx]
    if not inv.consume(item_id):
        return None
    d = REGISTRY[item_id]
    if d.heal:
        me.health = min(me.max_health, me.health + d.heal)
    if d.hunger:
        me.hunger = min(20.0, me.hunger + d.hunger)
    if d.absorption:
        me.absorption_hp = max(me.absorption_hp, d.absorption)
    if d.effect:
        FX.apply_effect(me, d.effect, d.effect_duration, d.effect_amp)
    return {"type": "use_complete", "player": idx, "item": item_id,
            "heal": d.heal, "absorption": d.absorption}
