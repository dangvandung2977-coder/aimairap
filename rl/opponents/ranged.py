"""Ranged-scenario opponents for Stage 4A/5.

Two roles:
  - bow-advantage foes (the AGENT should learn to shoot): pure kiters that
    stay out of melee range so a sword-only agent can only draw; bow reaches.
    ``FarFighter`` (mobility.py) already kites; ``rangedrunner`` is a stricter
    no-attack version so the counterfactual (arrows vs no arrows) is clean.
  - ranged attackers (the AGENT must close / shield): ``bowfighter`` selects
    the bow slot and shoots arrows, backpedalling to keep distance.

Deterministic under reset(seed). No reward / obs / action / mechanic changes.
"""
from __future__ import annotations

import math

import numpy as np

from rl.opponents.scripted import Opponent, _dist, _idle, _yaw_to
from sandbox.items import effects as FX
from sandbox.items.defs import HOTBAR

_BOW_SLOT = HOTBAR.index("bow")


class RangedRunner(Opponent):
    """Pure kiter that keeps itself hasted (speed effect) so it is genuinely
    FASTER than the agent -> a sword-only chaser can never corner it (bounded
    arenas otherwise let equal-speed melee catch a runner). It never attacks,
    so the only way to damage it is at range with the bow. Clean, honest bow
    counterfactual: arrows vs no arrows."""
    name = "rangedrunner"

    def __init__(self, keep: float = 8.0, seed: int = 0, switch_ticks: int = 60):
        self.keep = keep
        self.switch_ticks = switch_ticks
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        FX.apply_effect(me, "speed", 40, 0)  # stay hasted: uncatchable in melee
        c = _idle()
        d = _dist(sim, idx)
        away = me.pos - foe.pos
        n = float(np.linalg.norm(away[[0, 2]])) + 1e-9
        wx = away / n
        size = sim.world.size
        wall = np.zeros(2)
        if me.pos[0] < 4.0:
            wall[0] += 1.0
        if me.pos[0] > size - 4.0:
            wall[0] -= 1.0
        if me.pos[2] < 4.0:
            wall[1] += 1.0
        if me.pos[2] > size - 4.0:
            wall[1] -= 1.0
        tangent = np.array([-wx[2], wx[0]]) * self._dir
        wish = wx[[0, 2]] * (1.0 if d < self.keep + 4.0 else 0.3) \
            + wall * 1.5 + tangent * 0.5
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


class BowFighter(Opponent):
    """Backpedals to a preferred range and shoots arrows at the agent.
    Falls back to melee if cornered at point-blank. The AGENT must close the
    gap, shield the arrows, or out-range it. Needs bow+arrow in its loadout
    and bow enabled."""
    name = "bowfighter"

    def __init__(self, keep: float = 7.0, seed: int = 0, aggression: float = 0.0):
        self.keep = keep
        self.aggression = aggression  # 0 pure kiter-shooter, 1 pushes in
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        inv = sim.inventories[idx]
        c = _idle()
        d = _dist(sim, idx)
        # aim (with lead + small elevation to fight arrow drop)
        c.update(_aim_at(sim, idx, lead=True, elev=6.0))
        if inv.selected != _BOW_SLOT:
            c["select_slot"] = _BOW_SLOT
        # movement: keep preferred range
        if d < self.keep - 1.0:
            c["move_z"] = -1.0  # back off
            c["move_x"] = 0.6 * self._dir
            c["sprint"] = True
        elif d > self.keep + 2.0:
            c["move_z"] = 1.0 * (1.0 if self.aggression else 0.4)
            c["move_x"] = 0.4 * self._dir
        else:
            c["move_x"] = 0.8 * self._dir  # strafe at range
        # fire when aimed, in slot, off cooldown, and not point-blank
        aimed = abs(_yaw_error(sim, idx)) < 12.0
        ready = inv.cooldowns.get("bow", 0) <= 0 and inv.switch_timer <= 0
        if aimed and ready and inv.selected == _BOW_SLOT and 2.5 < d < 20.0:
            c["use_item"] = True
        elif d <= 2.2 and me.attack_cooldown <= 0:
            c["attack"] = True  # melee fallback if rushed
        if self._t % 55 == 0:
            self._dir *= -1.0
        return c


def _yaw_error(sim, idx: int) -> float:
    me, foe = sim.players[idx], sim.players[1 - idx]
    dd = foe.pos - me.pos
    want = math.degrees(math.atan2(dd[2], dd[0]))
    return (want - me.yaw + 180.0) % 360.0 - 180.0


def _aim_at(sim, idx: int, lead: bool = True, elev: float = 6.0) -> dict:
    """Yaw toward the foe (optionally lead its velocity), pitch up a touch to
    counter arrow gravity. Returns partial controller dict (yaw/pitch deltas)."""
    me, foe = sim.players[idx], sim.players[1 - idx]
    target = foe.pos.copy()
    if lead:
        d = float(np.linalg.norm((foe.pos - me.pos)[[0, 2]]))
        tflight = d / 30.0  # arrow horizontal speed ~30
        target = foe.pos + foe.vel * tflight
    dd = target - me.pos
    want_yaw = math.degrees(math.atan2(dd[2], dd[0]))
    dyaw = (want_yaw - me.yaw + 180.0) % 360.0 - 180.0
    # look_dir uses +pitch -> +y, so aiming up (to counter arrow drop) is a
    # small POSITIVE pitch. Flat aim already hits inside ~10 blocks.
    want_pitch = elev
    dpitch = want_pitch - me.pitch
    return {"yaw_delta": float(np.clip(dyaw, -30.0, 30.0)),
            "pitch_delta": float(np.clip(dpitch, -10.0, 10.0))}


RANGED = {"rangedrunner": RangedRunner, "bowfighter": BowFighter}
