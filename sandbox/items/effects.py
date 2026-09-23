"""Deterministic status effects. apply_effect(...) only — no per-item branches."""
from __future__ import annotations

# effect_id -> tick handler name; durations in ticks, amp int
EFFECTS = ("regeneration", "absorption", "speed", "strength",
           "fire_resistance", "slow")


def apply_effect(p, effect_id: str, duration: int, amp: int = 0) -> bool:
    if not effect_id or effect_id not in EFFECTS:
        return False
    if effect_id == "absorption":
        p.absorption_hp = max(p.absorption_hp, 4.0 * (amp + 2))
    cur = p.effects.get(effect_id)
    if cur is None or duration > cur[0]:
        p.effects[effect_id] = [duration, amp]
    return True


def has_effect(p, effect_id: str) -> bool:
    e = p.effects.get(effect_id)
    return e is not None and e[0] > 0


def effect_time(p, effect_id: str) -> int:
    e = p.effects.get(effect_id)
    return int(e[0]) if e else 0


def tick_effects(p) -> float:
    """Advance one tick. Returns healing applied (for logging)."""
    healed = 0.0
    dead = []
    for eid, (t, amp) in p.effects.items():
        if eid == "regeneration" and p.alive and t % 20 == 0:
            p.health = min(p.max_health, p.health + 1.0 + amp)
            healed += 1.0 + amp
        t -= 1
        if t <= 0:
            dead.append(eid)
        else:
            p.effects[eid][0] = t
    for eid in dead:
        del p.effects[eid]
    return healed


def damage_mult(p) -> float:
    return 1.3 if has_effect(p, "strength") else 1.0


def speed_mult(p, in_fluid: float = 1.0, in_web: bool = False) -> float:
    m = 1.0
    if has_effect(p, "speed"):
        m *= 1.2
    if in_web:
        m *= 0.35
    return m * in_fluid
