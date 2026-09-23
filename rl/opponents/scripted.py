"""Opponent interface + scripted baselines + policy wrapper (self-play ready)."""
from __future__ import annotations

import math

import numpy as np


class Opponent:
    name = "base"

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, sim, idx: int) -> dict:
        raise NotImplementedError


def _idle() -> dict:
    return {"move_x": 0.0, "move_z": 0.0, "sprint": False, "jump": False,
            "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0}


def _yaw_to(sim, idx: int) -> float:
    me, foe = sim.players[idx], sim.players[1 - idx]
    d = foe.pos - me.pos
    want = math.degrees(math.atan2(d[2], d[0]))
    diff = (want - me.yaw + 180.0) % 360.0 - 180.0
    return float(np.clip(diff, -30.0, 30.0))


def _dist(sim, idx: int) -> float:
    return float(np.linalg.norm(sim.players[1 - idx].pos - sim.players[idx].pos))


class DummyOpponent(Opponent):
    """Stationary target. Never moves, never attacks."""
    name = "dummy"

    def act(self, sim, idx: int) -> dict:
        return _idle()


class MeleeOpponent(Opponent):
    """Approach, strafe, and attack. Deterministic under reset(seed);
    strafe phase + rare jumps give controlled variation so policies
    can't overfit to one exact trajectory."""

    name = "melee"

    def __init__(self, attack_range: float = 2.8, strafe_ticks: int = 35,
                 jump_prob: float = 0.02, seed: int = 0):
        self.attack_range = attack_range
        self.strafe_ticks = strafe_ticks
        self.jump_prob = jump_prob
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._switch = self.strafe_ticks + int(self.rng.integers(-8, 9))

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        if self._t % max(8, self._switch) == 0:
            self._dir *= -1.0
        me = sim.players[idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if d > self.attack_range:
            c["move_z"] = 1.0
            c["move_x"] = 0.5 * self._dir
            c["sprint"] = d > 4.0
            if me.on_ground and self.rng.random() < self.jump_prob:
                c["jump"] = True
        else:
            c["move_x"] = 1.0 * self._dir
            c["attack"] = me.attack_cooldown <= 0
            if me.on_ground and self.rng.random() < self.jump_prob * 0.5:
                c["jump"] = True
        return c


class StrafeOpponent(Opponent):
    """Orbit the enemy while closing to range; harder to hit."""
    name = "strafe"

    def __init__(self, attack_range: float = 2.8, switch_ticks: int = 40):
        self.attack_range = attack_range
        self.switch_ticks = switch_ticks
        self._t = 0
        self._dir = 1.0

    def reset(self, seed=None) -> None:
        self._t = 0
        self._dir = 1.0

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        if self._t % self.switch_ticks == 0:
            self._dir *= -1.0
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if d > self.attack_range + 0.5:
            c["move_z"] = 1.0
            c["move_x"] = 0.5 * self._dir
            c["sprint"] = d > 5.0
        else:
            c["move_x"] = 1.0 * self._dir
            c["attack"] = sim.players[idx].attack_cooldown <= 0
        return c


class PolicyOpponent(Opponent):
    """Wraps an SB3 model (or any obs->raw-action fn) as the enemy.

    Future self-play: point at current policy / historical checkpoints.
    Not used in phase-1 training; interface proven here.
    """
    name = "policy"

    def __init__(self, predict_fn, agent_index: int = 1):
        self.predict_fn = predict_fn
        self.agent_index = agent_index

    def act(self, sim, idx: int) -> dict:
        from rl.environment import action_map, obs_schema
        obs = obs_schema.build_obs(sim, agent=idx)
        raw, _ = self.predict_fn(obs)
        return action_map.to_controller(raw)


_OPPONENTS = {"dummy": DummyOpponent, "melee": MeleeOpponent, "strafe": StrafeOpponent}


class FighterOpponent(Opponent):
    """Parameterized melee profile. One class, five presets (chaser, strafer,
    aggressive, defensive, random) differing in approach/strafe/attack/spacing.
    All deterministic under reset(seed)."""

    name = "fighter"

    def __init__(self, strafe_mag: float = 0.0, strafe_ticks: int = 35,
                 alternate: bool = True, fixed_dir: float = 0.0,
                 sprint_far: bool = True, far_dist: float = 4.0,
                 attack_range: float = 2.8, attack_patience: int = 0,
                 prefer_dist: float = 0.0, back_speed: float = 0.0,
                 jump_prob: float = 0.0, yaw_jitter: float = 0.0, seed: int = 0):
        self.p = dict(strafe_mag=strafe_mag, strafe_ticks=strafe_ticks,
                      alternate=alternate, fixed_dir=fixed_dir,
                      sprint_far=sprint_far, far_dist=far_dist,
                      attack_range=attack_range, attack_patience=attack_patience,
                      prefer_dist=prefer_dist, back_speed=back_speed,
                      jump_prob=jump_prob, yaw_jitter=yaw_jitter)
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._wait = 0
        if self.p["alternate"]:
            self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        else:
            self._dir = float(self.p["fixed_dir"]) or 1.0
        self._switch = self.p["strafe_ticks"] + int(self.rng.integers(-8, 9))

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        if self._t % max(8, self._switch) == 0 and self.p["alternate"]:
            self._dir *= -1.0
        me = sim.players[idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        if self.p["yaw_jitter"]:
            c["yaw_delta"] += float(self.rng.uniform(-self.p["yaw_jitter"],
                                                     self.p["yaw_jitter"]))
        d = _dist(sim, idx)
        if d > self.p["attack_range"]:
            c["move_z"] = 1.0
            c["move_x"] = self.p["strafe_mag"] * self._dir
            c["sprint"] = self.p["sprint_far"] and d > self.p["far_dist"]
        elif self.p["prefer_dist"] and d < self.p["prefer_dist"]:
            # spacing: back off to preferred distance while facing foe
            c["move_z"] = -self.p["back_speed"]
            c["move_x"] = self.p["strafe_mag"] * self._dir
        else:
            c["move_x"] = self.p["strafe_mag"] * self._dir or 0.5 * self._dir
            if self._wait > 0:
                self._wait -= 1
            elif me.attack_cooldown <= 0:
                # patience: occasionally hold the swing one decision (timing variety)
                if self.p["attack_patience"] and self.rng.random() < 0.25:
                    self._wait = self.p["attack_patience"]
                else:
                    c["attack"] = True
        if me.on_ground and self.rng.random() < self.p["jump_prob"]:
            c["jump"] = True
        return c


PROFILES = {
    # 1. Basic Chaser: runs down the player, no strafe, no jump.
    "chaser": dict(strafe_mag=0.0, sprint_far=True, attack_range=2.8,
                   jump_prob=0.0, yaw_jitter=0.0),
    # 2. Strafing Fighter: orbits, alternating, mid switch rate.
    "strafer": dict(strafe_mag=1.0, strafe_ticks=30, sprint_far=True,
                    attack_range=2.8, jump_prob=0.01, yaw_jitter=0.0),
    # 3. Aggressive: sprints everywhere, strafes while closing, jumps often.
    "aggressive": dict(strafe_mag=0.3, strafe_ticks=25, sprint_far=True,
                       far_dist=0.0, attack_range=3.0,
                       jump_prob=0.05, yaw_jitter=2.0),
    # 4. Defensive/Spacing: keeps ~2.5 blocks, backs off inside, patient swings.
    "defensive": dict(strafe_mag=0.7, strafe_ticks=45, sprint_far=False,
                      attack_range=2.6, attack_patience=1, prefer_dist=2.5,
                      back_speed=0.8, jump_prob=0.01, yaw_jitter=0.0),
    # 6/7. Constant-direction circlers: never switch strafe side, so tracking
    # them requires matching their rotational direction (direction probe).
    "circlerL": dict(strafe_mag=1.0, alternate=False, fixed_dir=-1.0,
                     sprint_far=False, attack_range=2.8, jump_prob=0.0,
                     yaw_jitter=0.0),
    "circlerR": dict(strafe_mag=1.0, alternate=False, fixed_dir=1.0,
                     sprint_far=False, attack_range=2.8, jump_prob=0.0,
                     yaw_jitter=0.0),
    # 8/9. Fast directional speeders: sprint + fixed strafe side. Wrong-side
    # tracking cannot keep up (strong directional pressure, no reward change).
    "speederL": dict(strafe_mag=1.0, alternate=False, fixed_dir=-1.0,
                     sprint_far=True, far_dist=0.0, attack_range=2.8,
                     jump_prob=0.0, yaw_jitter=0.0),
    "speederR": dict(strafe_mag=1.0, alternate=False, fixed_dir=1.0,
                     sprint_far=True, far_dist=0.0, attack_range=2.8,
                     jump_prob=0.0, yaw_jitter=0.0),
}


class RandomFighter(FighterOpponent):
    """5. Randomized: params drawn from controlled ranges at reset(seed)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.p = {}
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        r = self.rng
        self.p = dict(
            strafe_mag=float(r.choice([0.0, 0.5, 1.0])),
            strafe_ticks=int(r.integers(20, 51)),
            alternate=bool(r.random() < 0.8),
            fixed_dir=float(r.choice([-1.0, 1.0])),
            sprint_far=bool(r.random() < 0.7),
            far_dist=float(r.uniform(2.0, 6.0)),
            attack_range=float(r.uniform(2.4, 3.0)),
            attack_patience=int(r.integers(0, 2)),
            prefer_dist=float(r.choice([0.0, 0.0, 2.2])),
            back_speed=0.8,
            jump_prob=float(r.uniform(0.0, 0.04)),
            yaw_jitter=float(r.uniform(0.0, 3.0)),
        )
        self._t = 0
        self._wait = 0
        self._dir = 1.0 if r.random() < 0.5 else -1.0
        self._switch = self.p["strafe_ticks"]


_OPPONENTS.update({"fighter": FighterOpponent, "random": RandomFighter})


def create_profile(name: str, seed: int = 0, **over) -> Opponent:
    if name == "random":
        return RandomFighter(seed=seed)
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}, choose from {sorted(PROFILES)}")
    kw = dict(PROFILES[name])
    kw.update(over)
    return FighterOpponent(seed=seed, **kw)


def create(name: str, **kw) -> Opponent:
    if name in PROFILES:
        seed = kw.pop("seed", 0)
        return create_profile(name, seed=seed, **kw)
    try:
        from rl.opponents.counterplay import COUNTERS
        if name in COUNTERS:
            return COUNTERS[name](**kw)
        from rl.opponents.grounded import GROUNDED
        if name in GROUNDED:
            return GROUNDED[name](**kw)
        from rl.opponents.utility import UTILITY
        if name in UTILITY:
            return UTILITY[name](**kw)
        from rl.opponents.shield_counter import SHIELD_COUNTERS
        if name in SHIELD_COUNTERS:
            return SHIELD_COUNTERS[name](**kw)
        from rl.opponents.circler import CIRCLERS
        if name in CIRCLERS:
            return CIRCLERS[name](**kw)
        from rl.opponents.mobility import MOBILITY
        if name in MOBILITY:
            return MOBILITY[name](**kw)
        from rl.opponents.sustain import REGEN
        if name in REGEN:
            return REGEN[name](**kw)
        from rl.opponents.ranged import RANGED
        if name in RANGED:
            return RANGED[name](**kw)
    except ImportError:
        pass
    if name not in _OPPONENTS:
        raise ValueError(f"unknown opponent {name!r}, choose from "
                         f"{sorted(_OPPONENTS)} + profiles {sorted(PROFILES)}")
    return _OPPONENTS[name](**kw)
