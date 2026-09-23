"""Grounded counterplay: make staying grounded tactically valuable.

All perception passes through a reaction-delay buffer (decisions use the foe
state from `delay` ticks ago) plus prediction error on intercepts. No future
info, no hidden state, no damage bonuses. Deterministic under reset(seed).
Train seeds < 1000, eval seeds >= 5000.
"""
from __future__ import annotations

import numpy as np

from rl.opponents.scripted import Opponent, _dist, _idle, _yaw_to


class _GroundedBase(Opponent):
    name = "grounded"

    def __init__(self, delay: int = 4, pred_err: float = 0.6,
                 strafe_mag: float = 0.6, strafe_ticks: int = 30,
                 seed: int = 0):
        self.delay = delay
        self.pred_err = pred_err
        self.strafe_mag = strafe_mag
        self.strafe_ticks = strafe_ticks
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._switch = self.strafe_ticks + int(self.rng.integers(-6, 7))
        self._hist: list = []  # foe (pos, vel, on_ground) snapshots
        self._mode = "engage"
        self._mode_t = 0
        self._air_t = 0  # consecutive ticks foe seen airborne (delayed view)

    def _observe(self, sim, idx: int):
        foe = sim.players[1 - idx]
        self._hist.append((foe.pos.copy(), foe.vel.copy(), foe.on_ground))
        self._hist = self._hist[-(self.delay + 2):]
        old = self._hist[0]
        if not old[2]:
            self._air_t += 1
        else:
            self._air_t = 0
        return old  # (pos, vel, on_ground) as seen `delay` ticks ago

    def _intercept(self, own_pos: np.ndarray, fpos: np.ndarray,
                   fvel: np.ndarray, ahead: float = 0.35) -> np.ndarray:
        pred = fpos + fvel * ahead
        pred = pred + self.rng.normal(0.0, self.pred_err, size=3)
        pred[1] = 0.0
        return pred

    def _step_to(self, c: dict, sim, idx: int, target: np.ndarray,
                 sprint_far: float = 4.5) -> dict:
        me = sim.players[idx]
        d = target - me.pos
        dist = float(np.linalg.norm(d[[0, 2]]))
        want = float(np.degrees(np.arctan2(d[2], d[0])))
        diff = (want - me.yaw + 180.0) % 360.0 - 180.0
        c["yaw_delta"] = float(np.clip(diff, -30.0, 30.0))
        if dist > 2.9:
            # run toward intercept in yaw space: decompose into local move
            yr = float(np.radians(me.yaw))
            fwd = np.array([np.cos(yr), np.sin(yr)])
            right = np.array([-np.sin(yr), np.cos(yr)])
            w = d[[0, 2]] / (dist + 1e-9)
            c["move_z"] = float(np.clip(fwd @ w, -1.0, 1.0))
            c["move_x"] = float(np.clip(right @ w, -1.0, 1.0))
            c["sprint"] = dist > sprint_far and c["move_z"] > 0.3
        else:
            c["move_x"] = self.strafe_mag * self._dir
            c["attack"] = me.attack_cooldown <= 0
        return c


class GroundSpacing(_GroundedBase):
    """A. Grounded spacing exemplar: band control + lateral drift, never jumps."""

    name = "gspace"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        if self._t % max(8, self._switch) == 0:
            self._dir *= -1.0
        self._observe(sim, idx)
        me = sim.players[idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if d > 3.0:
            c["move_z"] = 1.0
            c["move_x"] = self.strafe_mag * self._dir
            c["sprint"] = d > 4.5
        elif d < 2.0:
            c["move_z"] = -0.8
            c["move_x"] = self.strafe_mag * self._dir
        else:
            c["move_x"] = self.strafe_mag * self._dir
            c["attack"] = me.attack_cooldown <= 0
        return c


class GroundAntiAir(_GroundedBase):
    """B. Punishes *predictable* airtime: reacts late, predicts with error,
    only commits when foe seen airborne several ticks in a row."""

    name = "gantiair"

    def __init__(self, air_thresh: int = 6, punish_prob: float = 0.7,
                 delay: int = 4, pred_err: float = 0.6, seed: int = 0):
        self.air_thresh = air_thresh
        self.punish_prob = punish_prob
        super().__init__(delay=delay, pred_err=pred_err, seed=seed)
        self._committed = False

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        fpos, fvel, f_air = self._observe(sim, idx)
        me = sim.players[idx]
        c = _idle()
        predictable = self._air_t >= self.air_thresh
        if predictable and not self._committed:
            self._committed = self.rng.random() < self.punish_prob
        if not f_air:
            self._committed = False
        d = _dist(sim, idx)
        if f_air and self._committed:
            # reposition to noisy intercept; swing only once foe is down
            target = self._intercept(me.pos, fpos, fvel)
            c = self._step_to(c, sim, idx, target)
            c["attack"] = False
        elif not f_air and d <= 3.0 and me.attack_cooldown <= 0:
            c["yaw_delta"] = _yaw_to(sim, idx)
            # punish the landing/recovery window, then release
            c["attack"] = True
            if self._t % max(8, self._switch) == 0:
                self._dir *= -1.0
        else:
            c["yaw_delta"] = _yaw_to(sim, idx)
            if d > 3.0:
                c["move_z"] = 1.0
                c["move_x"] = 0.3 * self._dir
                c["sprint"] = d > 4.5
            else:
                c["move_x"] = 0.4 * self._dir
                c["attack"] = me.attack_cooldown <= 0
        return c


class GroundReengage(_GroundedBase):
    """C. Grounded backoff: retreats laterally, sidesteps overextension."""

    name = "greengage"

    def __init__(self, engage_ticks: int = 55, retreat_ticks: int = 28,
                 retreat_dist: float = 6.0, seed: int = 0):
        self.engage_ticks = engage_ticks
        self.retreat_ticks = retreat_ticks
        self.retreat_dist = retreat_dist
        super().__init__(seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._mode_t += 1
        fpos, fvel, _ = self._observe(sim, idx)
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if self._mode == "engage" and self._mode_t > self.engage_ticks:
            self._mode, self._mode_t = "retreat", 0
        elif self._mode == "retreat" and (self._mode_t > self.retreat_ticks
                                          or d > self.retreat_dist):
            self._mode, self._mode_t = "engage", 0
        foe_speed = float(np.linalg.norm(foe.vel[[0, 2]]))
        closing = bool(np.dot(foe.vel[[0, 2]], (me.pos - foe.pos)[[0, 2]]) < 0)
        if self._mode == "retreat":
            if foe_speed > 4.5 and closing and d < 4.5:
                # overextension: hard sidestep + counter as it arrives
                c["move_x"] = 1.0 * self._dir
                c["move_z"] = 0.2
                if d <= 3.0:
                    c["attack"] = me.attack_cooldown <= 0
            else:
                c["move_z"] = -0.9
                c["move_x"] = 0.7 * self._dir
        elif d > 2.9:
            c["move_z"] = 1.0
            c["move_x"] = 0.3 * self._dir
            c["sprint"] = d > 4.5
        else:
            c["move_x"] = 0.5 * self._dir
            c["attack"] = me.attack_cooldown <= 0
        if self._t % 34 == 0:
            self._dir *= -1.0
        return c


class GroundMixed(_GroundedBase):
    """D. Rotates spacing / anti-air / re-engage modes; stays grounded."""

    name = "gmixed"

    MODES = ("spacing", "antiair", "reengage")

    def __init__(self, mode_ticks: int = 80, seed: int = 0):
        self.mode_ticks = mode_ticks
        self._subs = {"spacing": GroundSpacing(seed=seed),
                      "antiair": GroundAntiAir(seed=seed),
                      "reengage": GroundReengage(seed=seed)}
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        s = self.seed if seed is None else seed
        for sub in self._subs.values():
            sub.reset(s)
        self._mode = "spacing"
        self._mode_t = 0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._mode_t += 1
        if self._mode_t > self.mode_ticks:
            choices = [m for m in self.MODES if m != self._mode]
            self._mode = choices[int(self.rng.integers(len(choices)))]
            self._mode_t = 0
        return self._subs[self._mode].act(sim, idx)


GROUNDED = {"gspace": GroundSpacing, "gantiair": GroundAntiAir,
            "greengage": GroundReengage, "gmixed": GroundMixed}
