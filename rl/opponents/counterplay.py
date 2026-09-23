"""Counterplay opponents: punish jump-spam / poor re-engage / mistimed attacks.

Only structured state a real controller could have is used:
positions, velocities, on_ground, cooldowns, health, distance.
All deterministic under reset(seed). Train seeds < 1000, eval seeds >= 5000.
"""
from __future__ import annotations

import numpy as np

from rl.opponents.scripted import Opponent, _dist, _idle, _yaw_to


class _Base(Opponent):
    name = "counter"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._mode = "engage"
        self._mode_t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._last_foe_cd = 0
        self._last_foe_hp = 20.0
        self._foe_was_air = False
        self._counter_t = 0
        self._pressure_t = 0


class BackoffFighter(_Base):
    """Engage, then retreat to punish weak re-engagement, then re-engage."""

    name = "backoff"

    def __init__(self, engage_ticks: int = 60, retreat_ticks: int = 30,
                 retreat_dist: float = 6.0, seed: int = 0):
        self.engage_ticks = engage_ticks
        self.retreat_ticks = retreat_ticks
        self.retreat_dist = retreat_dist
        super().__init__(seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._mode_t += 1
        me = sim.players[idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if self._mode == "engage" and self._mode_t > self.engage_ticks:
            self._mode, self._mode_t = "retreat", 0
            if self.rng.random() < 0.5:
                self._dir *= -1.0
        elif self._mode == "retreat" and (self._mode_t > self.retreat_ticks
                                          or d > self.retreat_dist):
            self._mode, self._mode_t = "engage", 0
        if self._mode == "retreat":
            c["move_z"] = -1.0
            c["move_x"] = 0.5 * self._dir
            if d <= 3.0 and me.attack_cooldown <= 0:
                c["attack"] = True  # punish the gap-closer
        else:
            if d > 2.8:
                c["move_z"] = 1.0
                c["move_x"] = 0.4 * self._dir
                c["sprint"] = d > 4.0
            else:
                c["move_x"] = 0.6 * self._dir
                c["attack"] = me.attack_cooldown <= 0
        return c


class LandingPunisher(_Base):
    """Yield space while foe is airborne; punish the landing, not the jump."""

    name = "landpunish"

    def __init__(self, hold_dist: float = 3.0, pressure_ticks: int = 16,
                 punish_prob: float = 0.65, seed: int = 0):
        self.hold_dist = hold_dist
        self.pressure_ticks = pressure_ticks
        self.punish_prob = punish_prob
        super().__init__(seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        foe_air = not foe.on_ground
        just_landed = self._foe_was_air and not foe_air
        self._foe_was_air = foe_air
        if self._pressure_t > 0:
            self._pressure_t -= 1
        if just_landed and d < 4.0 and self.rng.random() < self.punish_prob:
            self._pressure_t = self.pressure_ticks
        if foe_air:
            # hold spacing, don't swing at air; drift laterally
            if d < self.hold_dist:
                c["move_z"] = -0.8
            elif d > self.hold_dist + 1.0:
                c["move_z"] = 0.8
            c["move_x"] = 0.6 * self._dir
            if self._t % 30 == 0:
                self._dir *= -1.0
        elif self._pressure_t > 0:
            if d > 2.8:
                c["move_z"] = 1.0
                c["sprint"] = True
            else:
                c["attack"] = me.attack_cooldown <= 0
        else:
            if d > 2.8:
                c["move_z"] = 1.0
                c["move_x"] = 0.2 * self._dir
                c["sprint"] = d > 4.5
            else:
                c["attack"] = me.attack_cooldown <= 0  # grounded foe = pressure
        return c


class MissPunisher(_Base):
    """Counter-attack right after the foe whiffs (cooldown tripped, no damage)."""

    name = "misspunish"

    def __init__(self, counter_ticks: int = 20, counter_prob: float = 0.85,
                 seed: int = 0):
        self.counter_ticks = counter_ticks
        self.counter_prob = counter_prob
        super().__init__(seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        # foe attempt detected via cooldown trip; miss = our hp unchanged
        foe_attempt = self._last_foe_cd <= 0 < foe.attack_cooldown
        if foe_attempt and me.health >= self._last_foe_hp \
                and self.rng.random() < self.counter_prob:
            self._counter_t = self.counter_ticks
        self._last_foe_cd = foe.attack_cooldown
        self._last_foe_hp = me.health
        if self._counter_t > 0:
            self._counter_t -= 1
        if self._counter_t > 0:
            c["move_x"] = 1.0 * self._dir
            if d > 3.0:
                c["move_z"] = 1.0
            else:
                c["attack"] = me.attack_cooldown <= 0
            if self.rng.random() < 0.05:
                self._dir *= -1.0
        elif d > 2.8:
            c["move_z"] = 1.0
            c["move_x"] = 0.5 * self._dir
            c["sprint"] = d > 4.0
            if self._t % 40 == 0:
                self._dir *= -1.0
        else:
            c["attack"] = me.attack_cooldown <= 0
        return c


class StrafePressure(_Base):
    """Heavy lateral movement with variable duration + forward/back weave."""

    name = "strafepress"

    def __init__(self, switch_min: int = 12, switch_max: int = 30,
                 weave_ticks: int = 25, seed: int = 0):
        self.switch_min = switch_min
        self.switch_max = switch_max
        self.weave_ticks = weave_ticks
        super().__init__(seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._switch = self.rng.integers(self.switch_min, self.switch_max + 1)
        self._fwd = 1.0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        if self._t % int(self._switch) == 0:
            self._dir *= -1.0
            self._switch = self.rng.integers(self.switch_min, self.switch_max + 1)
        if self._t % self.weave_ticks == 0:
            self._fwd *= -1.0
        me = sim.players[idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        c["move_x"] = 1.0 * self._dir
        if d > 3.2:
            c["move_z"] = 1.0
            c["sprint"] = d > 5.0
        elif d < 1.8:
            c["move_z"] = -0.7
        else:
            c["move_z"] = 0.3 * self._fwd
            c["attack"] = me.attack_cooldown <= 0
        return c


class MixedPunisher(_Base):
    """Rotates backoff / landing / miss / strafe / aggro modes with cooldowns."""

    name = "mixed"

    MODES = ("backoff", "landing", "miss", "strafe", "aggro")

    def __init__(self, mode_ticks: int = 70, seed: int = 0):
        self.mode_ticks = mode_ticks
        self._subs = {"backoff": BackoffFighter(seed=seed),
                      "landing": LandingPunisher(seed=seed),
                      "miss": MissPunisher(seed=seed),
                      "strafe": StrafePressure(seed=seed),
                      "aggro": None}
        super().__init__(seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        s = self.seed if seed is None else seed
        for k, sub in self._subs.items():
            if sub is not None:
                sub.reset(s)
        self._mode = "aggro"
        self._mode_t = 0
        self._cool: dict[str, int] = {}

    def act(self, sim, idx: int) -> dict:
        from rl.opponents.scripted import MeleeOpponent
        self._t += 1
        self._mode_t += 1
        for k in list(self._cool):
            self._cool[k] -= 1
            if self._cool[k] <= 0:
                del self._cool[k]
        if self._mode_t > self.mode_ticks:
            choices = [m for m in self.MODES if m not in self._cool]
            self._mode = choices[int(self.rng.integers(len(choices)))]
            self._cool[self._mode] = 2  # no immediate repeat
            self._mode_t = 0
        if self._mode == "aggro":
            if not hasattr(self, "_aggro"):
                self._aggro = MeleeOpponent()
                self._aggro.reset(self.seed)
            return self._aggro.act(sim, idx)
        return self._subs[self._mode].act(sim, idx)


COUNTERS = {"backoff": BackoffFighter, "landpunish": LandingPunisher,
            "misspunish": MissPunisher, "strafepress": StrafePressure,
            "mixed": MixedPunisher}
