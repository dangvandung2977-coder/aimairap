"""Directional-selection family: orbit / speed / switch / mixed.

Design note: yaw correction (±30°/tick) and the 75° cone erase pure lateral
offsets, so flank side alone can never carry payoff. These foes create
directional payoff through (a) retreat direction (pursuit side matters:
catch vs timeout) and (b) fixed-direction wide turns (one rear quadrant is
exposed consistently per episode). All deterministic under reset(seed),
finite rotation, reaction delays, seeded imperfections.
"""
from __future__ import annotations

import math

import numpy as np

from rl.opponents.scripted import MeleeOpponent, Opponent, _dist, _idle, _yaw_to


class _CBase(Opponent):
    name = "circler"

    def __init__(self, radius: float = 3.0, omega: float = 0.9,
                 switch_ticks: int = 0, seed: int = 0):
        self.radius = radius
        self.omega = omega  # rad/s, sign = orbit direction (seeded per episode)
        self.switch_ticks = switch_ticks
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._theta = float(self.rng.uniform(0, 2 * math.pi))
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._mode = "orbit"
        self._mode_t = 0
        self._retreat_side = 1.0 if self.rng.random() < 0.5 else -1.0
        self._melee = MeleeOpponent()
        self._melee.reset(self.seed if seed is None else seed)

    def _orbit_point(self, sim, idx: int, dt: float) -> np.ndarray:
        foe = sim.players[1 - idx]
        self._theta += self._dir * self.omega * dt
        return foe.pos + np.array([math.cos(self._theta) * self.radius,
                                   0.0,
                                   math.sin(self._theta) * self.radius])

    def _steer(self, c: dict, sim, idx: int, target: np.ndarray,
               sprint: bool = False) -> dict:
        me = sim.players[idx]
        d = target - me.pos
        dist = float(np.linalg.norm(d[[0, 2]]))
        want = float(np.degrees(math.atan2(d[2], d[0])))
        diff = (want - me.yaw + 180.0) % 360.0 - 180.0
        c["yaw_delta"] = float(np.clip(diff, -25.0, 25.0))  # finite rotation
        if dist > 0.6:
            yr = float(np.radians(me.yaw))
            fwd = np.array([math.cos(yr), math.sin(yr)])
            right = np.array([-math.sin(yr), math.cos(yr)])
            w = d[[0, 2]] / (dist + 1e-9)
            c["move_z"] = float(np.clip(fwd @ w, -1.0, 1.0))
            c["move_x"] = float(np.clip(right @ w, -1.0, 1.0))
            c["sprint"] = sprint and c["move_z"] > 0.3
        return c


class Circler(_CBase):
    """Orbit the agent at walk pace; attack when the orbit brings it in."""
    name = "circler"

    def __init__(self, seed: int = 0):
        super().__init__(radius=3.0, omega=0.9, seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        c = _idle()
        target = self._orbit_point(sim, idx, 0.05)
        c = self._steer(c, sim, idx, target)
        if _dist(sim, idx) <= 3.0 and me.attack_cooldown <= 0:
            c["attack"] = True
        return c


class Speeder(_CBase):
    """Sprint orbit with steering lag (overshoot): fast, committal, finite."""
    name = "speeder"

    def __init__(self, seed: int = 0):
        super().__init__(radius=2.6, omega=1.6, seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        c = _idle()
        # lag: steer toward where the point was 3 ticks ago (overshoot)
        target = self._orbit_point(sim, idx, 0.05)
        lagged = target - (target - me.pos) * 0.15
        c = self._steer(c, sim, idx, lagged, sprint=True)
        if _dist(sim, idx) <= 3.0 and me.attack_cooldown <= 0:
            c["attack"] = True
        return c


class Switcher(_CBase):
    """Orbit, then retreat-sprint to a seeded side, then re-engage with a
    wide fixed-direction turn. Pursuit side and turn side are per-episode
    constants the agent must read from motion, not memory."""
    name = "switcher"

    def __init__(self, orbit_lo: int = 60, orbit_hi: int = 100,
                 retreat_ticks: int = 35, seed: int = 0):
        self.orbit_lo, self.orbit_hi = orbit_lo, orbit_hi
        self.retreat_ticks = retreat_ticks
        super().__init__(radius=3.0, omega=1.0, seed=seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._orbit_for = int(self.rng.integers(self.orbit_lo, self.orbit_hi + 1))
        self._mode, self._mode_t = "orbit", 0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._mode_t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        if self._mode == "orbit" and self._mode_t > self._orbit_for:
            self._mode, self._mode_t = "retreat", 0
            self._orbit_for = int(self.rng.integers(self.orbit_lo, self.orbit_hi + 1))
        elif self._mode == "retreat" and self._mode_t > self.retreat_ticks:
            self._mode, self._mode_t = "orbit", 0
        if self._mode == "retreat":
            # sprint away, biased to the seeded side; wide fixed-dir turn back
            away = me.pos - foe.pos
            n = float(np.linalg.norm(away[[0, 2]])) + 1e-9
            side = np.array([-away[2], away[0]]) / n * self._retreat_side
            target = me.pos + (away / n * 6.0 + np.array([side[0], 0.0, side[1]]))
            c = self._steer(c, sim, idx, target, sprint=True)
        else:
            target = self._orbit_point(sim, idx, 0.05)
            c = self._steer(c, sim, idx, target)
            if _dist(sim, idx) <= 3.0 and me.attack_cooldown <= 0:
                c["attack"] = True
        return c


class MixedCircler(_CBase):
    """Orbit + shield while the agent attacks + occasional counter-swing."""
    name = "mixedcircler"

    def __init__(self, seed: int = 0):
        super().__init__(radius=3.0, omega=1.1, seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        target = self._orbit_point(sim, idx, 0.05)
        c = self._steer(c, sim, idx, target)
        if foe.attack_cooldown > 6 and _dist(sim, idx) < 4.0:
            c["block"] = True  # needs shield owned; harmless otherwise
        elif _dist(sim, idx) <= 3.0 and me.attack_cooldown <= 0 \
                and foe.attack_cooldown > 8:
            c["attack"] = True
            c["block"] = False
        return c


CIRCLERS = {"circler": Circler, "speeder": Speeder, "switcher": Switcher,
            "mixedcircler": MixedCircler}
