"""Shield-counter opponents: turtle/baiter/strafer/pressure/mixed.

Deliberately imperfect: reaction-delay buffers, timing jitter, facing noise,
occasional errors. Deterministic under reset(seed). The goal is teachable
counterplay, not an unbeatable wall.
"""
from __future__ import annotations

import numpy as np

from rl.opponents.scripted import MeleeOpponent, Opponent, _dist, _idle, _yaw_to


class _SBase(Opponent):
    name = "shieldcounter"

    def __init__(self, delay: int = 2, seed: int = 0):
        self.delay = delay
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._hist: list = []  # delayed foe snapshots (pos, cd)
        self._mode = "block"
        self._mode_t = 0
        self._melee = MeleeOpponent()
        self._melee.reset(self.seed if seed is None else seed)

    def _see_foe(self, sim, idx: int):
        foe = sim.players[1 - idx]
        self._hist.append((foe.pos.copy(), foe.attack_cooldown))
        self._hist = self._hist[-(self.delay + 2):]
        return self._hist[0]

    def _block_ctrl(self, sim, idx: int, move_z: float = 0.0,
                    move_x: float = 0.0) -> dict:
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        c["block"] = True
        c["move_z"], c["move_x"] = move_z, move_x
        return c


class ShieldTurtle(_SBase):
    """Rarely exposes itself; swings only in safe windows (+jitter)."""
    name = "shieldturtle"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        _, foe_cd = self._see_foe(sim, idx)
        d = _dist(sim, idx)
        if d > 3.0:
            c = self._block_ctrl(sim, idx, move_z=0.8)
            return c
        # safe window: own swing ready, foe long cooldown, plus jitter delay
        if me.attack_cooldown <= 0 and foe_cd > 10 and self.rng.random() < 0.7:
            c = self._melee.act(sim, idx)
            c["block"] = False
            return c
        return self._block_ctrl(sim, idx)


class ShieldBaiter(_SBase):
    """block -> drop -> attack -> block cycle with seeded durations."""
    name = "shieldbaiter"

    def __init__(self, block_lo: int = 15, block_hi: int = 30, seed: int = 0):
        self.block_lo, self.block_hi = block_lo, block_hi
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._block_for = int(self.rng.integers(self.block_lo, self.block_hi + 1))
        self._mode, self._mode_t = "block", 0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._mode_t += 1
        self._see_foe(sim, idx)
        if self._mode == "block" and self._mode_t > self._block_for:
            self._mode, self._mode_t = "swing", 0
        elif self._mode == "swing" and self._mode_t > 8:
            self._mode, self._mode_t = "block", 0
            self._block_for = int(self.rng.integers(self.block_lo, self.block_hi + 1))
        if self._mode == "swing":
            return self._melee.act(sim, idx)
        return self._block_ctrl(sim, idx)


class ShieldStrafer(_SBase):
    """Blocks while strafing with facing noise; swings at long foe cooldowns."""
    name = "shieldstrafer"

    def __init__(self, switch_ticks: int = 30, seed: int = 0):
        self.switch_ticks = switch_ticks
        super().__init__(seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        if self._t % self.switch_ticks == 0:
            self._dir *= -1.0
        me = sim.players[idx]
        _, foe_cd = self._see_foe(sim, idx)
        d = _dist(sim, idx)
        if d < 4.5 and not (me.attack_cooldown <= 0 and foe_cd > 8):
            c = self._block_ctrl(sim, idx, move_x=0.8 * self._dir)
            c["yaw_delta"] += float(self.rng.uniform(-3.0, 3.0))
            return c
        c = self._melee.act(sim, idx)
        if d > 3.0:
            c["move_x"] = 0.5 * self._dir
        return c


class ShieldPressure(_SBase):
    """Advances behind the shield; drops it to punish at close range."""
    name = "shieldpressure"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        _, foe_cd = self._see_foe(sim, idx)
        d = _dist(sim, idx)
        if d < 2.2 and me.attack_cooldown <= 0 and foe_cd > 6:
            c = self._melee.act(sim, idx)
            c["block"] = False
            return c
        return self._block_ctrl(sim, idx, move_z=1.0 if d > 2.0 else 0.0)


class ShieldMixed(_SBase):
    """Rotates turtle / baiter / strafer modes; realistic final exam."""
    name = "shieldmixed"

    MODES = ("turtle", "baiter", "strafer")

    def __init__(self, mode_ticks: int = 80, seed: int = 0):
        self.mode_ticks = mode_ticks
        self._subs = {"turtle": ShieldTurtle(seed=seed),
                      "baiter": ShieldBaiter(seed=seed),
                      "strafer": ShieldStrafer(seed=seed)}
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        s = self.seed if seed is None else seed
        for sub in self._subs.values():
            sub.reset(s)
        self._mode = "turtle"
        self._mode_t = 0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._mode_t += 1
        self._see_foe(sim, idx)
        if self._mode_t > self.mode_ticks:
            choices = [m for m in self.MODES if m != self._mode]
            self._mode = choices[int(self.rng.integers(len(choices)))]
            self._mode_t = 0
        return self._subs[self._mode].act(sim, idx)


SHIELD_COUNTERS = {"shieldturtle": ShieldTurtle, "shieldbaiter": ShieldBaiter,
                   "shieldstrafer": ShieldStrafer,
                   "shieldpressure": ShieldPressure,
                   "shieldmixed": ShieldMixed}


class _TBase(_SBase):
    """True turtle base: long holds, observable release windows, reaction
    delay on re-raise, slight rotation, occasional attack attempts."""

    def __init__(self, hold_lo: int, hold_hi: int, open_lo: int, open_hi: int,
                 reraise_delay: int = 3, seed: int = 0):
        self.hold_lo, self.hold_hi = hold_lo, hold_hi
        self.open_lo, self.open_hi = open_lo, open_hi
        self.reraise_delay = reraise_delay
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._hold_for = int(self.rng.integers(self.hold_lo, self.hold_hi + 1))
        self._open_for = int(self.rng.integers(self.open_lo, self.open_hi + 1))
        self._phase = "hold"
        self._phase_t = 0
        self._reraise_wait = 0
        self._rot = self.rng.uniform(-8.0, 8.0)

    def _advance(self, foe_close: bool) -> str:
        self._phase_t += 1
        if self._phase == "hold" and self._phase_t > self._hold_for:
            self._phase, self._phase_t = "open", 0
            self._open_for = int(self.rng.integers(self.open_lo, self.open_hi + 1))
        elif self._phase == "open" and self._phase_t > self._open_for:
            self._phase, self._phase_t = "hold", 0
            self._hold_for = int(self.rng.integers(self.hold_lo, self.hold_hi + 1))
            self._reraise_wait = self.reraise_delay  # late re-raise = punishable
        if self._reraise_wait > 0:
            self._reraise_wait -= 1
            return "open"  # still down: legitimate punish window
        return self._phase

    def _hold_ctrl(self, sim, idx: int) -> dict:
        c = self._block_ctrl(sim, idx)
        c["yaw_delta"] += self._rot  # slow drift, not perfect tracking
        return c


class TurtleA(_TBase):
    """Long predictable-ish holds (60-90t) with real release windows (12-20t)."""
    name = "turtleA"

    def __init__(self, seed: int = 0):
        super().__init__(60, 90, 12, 20, reraise_delay=2, seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._see_foe(sim, idx)
        d = _dist(sim, idx)
        phase = self._advance(d < 4.0)
        if phase == "open":
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            c["move_x"] = 0.3 * self._dir
            if d <= 3.0 and sim.players[idx].attack_cooldown <= 0 \
                    and self.rng.random() < 0.5:
                c["attack"] = True  # occasional attack attempt, not optimal
            return c
        return self._hold_ctrl(sim, idx)


class TurtleB(_TBase):
    """Variable timing (30-70t holds, 8-25t windows) with slower re-raise."""
    name = "turtleB"

    def __init__(self, seed: int = 0):
        super().__init__(30, 70, 8, 25, reraise_delay=4, seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._see_foe(sim, idx)
        d = _dist(sim, idx)
        if self._t % 47 == 0:
            self._dir *= -1.0
        phase = self._advance(d < 4.0)
        if phase == "open":
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            c["move_x"] = 0.5 * self._dir
            return c
        c = self._hold_ctrl(sim, idx)
        c["move_x"] = 0.2 * self._dir
        return c


class TurtleC(_TBase):
    """Hold + occasional melee pressure after windows (2-swing bursts)."""
    name = "turtleC"

    def __init__(self, seed: int = 0):
        super().__init__(45, 75, 10, 18, reraise_delay=3, seed=seed)

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        self._see_foe(sim, idx)
        d = _dist(sim, idx)
        phase = self._advance(d < 4.0)
        if phase == "open":
            if d <= 3.0:
                c = self._melee.act(sim, idx)
                c["block"] = False
                return c
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            return c
        return self._hold_ctrl(sim, idx)


SHIELD_COUNTERS.update({"turtleA": TurtleA, "turtleB": TurtleB,
                        "turtleC": TurtleC})
