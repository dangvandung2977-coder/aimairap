"""Mobility-scenario opponents: far / retreat / knockback / chase-bait.

Each creates a situation where Pearl MAY pay (distance, escape, separation)
but normal movement stays viable. Deterministic under reset(seed).
"""
from __future__ import annotations

import numpy as np

from rl.opponents.scripted import MeleeOpponent, Opponent, _dist, _idle, _yaw_to


class _MBase(Opponent):
    name = "mobility"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._melee = MeleeOpponent()
        self._melee.reset(self.seed if seed is None else seed)


class FarFighter(_MBase):
    """Kites with wall-aware steering: away from the agent and walls plus a
    tangential drift, always sprinting. Equal speed + head start + steering
    means it runs out the clock (draw) unless the agent pearls to cut it off.
    Pearl is the catch tool."""
    name = "farfighter"

    def __init__(self, keep: float = 9.0, seed: int = 0,
                 switch_ticks: int = 50):
        self.keep = keep
        self.switch_ticks = switch_ticks
        super().__init__(seed=seed)

    def act(self, sim, idx: int) -> dict:
        import math
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        d = _dist(sim, idx)
        away = me.pos - foe.pos
        n = float(np.linalg.norm(away[[0, 2]])) + 1e-9
        wx = away / n
        size = sim.world.size
        m = 1.0
        wall_push = np.zeros(2)
        if me.pos[0] < m + 3.0:
            wall_push[0] += 1.0
        if me.pos[0] > size - m - 3.0:
            wall_push[0] -= 1.0
        if me.pos[2] < m + 3.0:
            wall_push[1] += 1.0
        if me.pos[2] > size - m - 3.0:
            wall_push[1] -= 1.0
        tangent = np.array([-wx[2], wx[0]]) * self._dir
        wish = wx[[0, 2]] * (1.0 if d < self.keep + 3.0 else 0.2) \
            + wall_push * 1.5 + tangent * 0.6
        wn = float(np.linalg.norm(wish)) + 1e-9
        wish = wish / wn
        want = float(math.degrees(math.atan2(wish[1], wish[0])))
        diff = (want - me.yaw + 180.0) % 360.0 - 180.0
        c["yaw_delta"] = float(np.clip(diff, -30.0, 30.0))
        c["move_z"] = 1.0
        c["sprint"] = True
        if self._t % self.switch_ticks == 0:
            self._dir *= -1.0
        return c


class RetreatFighter(_MBase):
    """Backs off while hurt; re-engages when recovered. Chasing pays."""
    name = "retreatfighter"

    def __init__(self, flee_hp: float = 12.0, seed: int = 0):
        self.flee_hp = flee_hp
        super().__init__(seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        d = _dist(sim, idx)
        if me.health < self.flee_hp:
            import math
            away = me.pos - foe.pos
            want = math.degrees(math.atan2(away[2], away[0]))
            diff = (want - me.yaw + 180.0) % 360.0 - 180.0
            c["yaw_delta"] = float(np.clip(diff, -30.0, 30.0))
            c["move_z"] = 1.0
            c["sprint"] = True
            c["move_x"] = 0.3 * self._dir
        else:
            c["yaw_delta"] = _yaw_to(sim, idx)
            if d > 2.8:
                c["move_z"] = 1.0
                c["sprint"] = d > 4.5
            else:
                c["attack"] = me.attack_cooldown <= 0
        if self._t % 45 == 0:
            self._dir *= -1.0
        return c


class KnockbackBully(_MBase):
    """Sprints at the agent swinging; knockback separates the fight."""
    name = "knockbackbully"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        c["move_z"] = 1.0
        c["sprint"] = True
        if d <= 3.0 and me.attack_cooldown <= 0:
            c["attack"] = True
        return c


class ChaseBait(_MBase):
    """Retreats for a while after TAKING damage (chase window), then turns.
    Tests pearl-chase timing, not blind pursuit."""
    name = "chasebait"

    def __init__(self, retreat_ticks: int = 60, seed: int = 0):
        self.retreat_ticks = retreat_ticks
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._flee_t = 0
        self._last_hp = 20.0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        if me.health < self._last_hp - 0.01:
            self._flee_t = self.retreat_ticks
        self._last_hp = me.health
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if self._flee_t > 0:
            self._flee_t -= 1
            c["move_z"] = -1.0
            c["sprint"] = True
            c["move_x"] = 0.4 * self._dir
        elif d > 2.8:
            c["move_z"] = 1.0
            c["sprint"] = d > 4.5
        else:
            c["attack"] = me.attack_cooldown <= 0
        return c


MOBILITY = {"farfighter": FarFighter, "retreatfighter": RetreatFighter,
            "knockbackbully": KnockbackBully, "chasebait": ChaseBait}
