"""Mechanic specialists for the utility curriculum + regression tests.

Each drives the shared controller dict (block/use_item/select_slot) through
the same validated pipeline as the RL policy. Deterministic under reset(seed).
"""
from __future__ import annotations

import numpy as np

from rl.opponents.scripted import MeleeOpponent, Opponent, _dist, _idle, _yaw_to

SWORD, SHIELD, BREAD, GAPPLE, PEARL, WEB, BLOCK, POT, BOW = range(9)


class _UBase(Opponent):
    name = "utility"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self._t = 0
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._melee = MeleeOpponent()
        self._melee.reset(self.seed if seed is None else seed)

    def _item(self, sim, idx: int, slot: int, move_z: float = 0.0,
              move_x: float = 0.0) -> dict:
        """Face foe, hold movement, select slot then use when ready."""
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        c["move_z"], c["move_x"] = move_z, move_x
        inv = sim.inventories[idx]
        if inv.selected != slot:
            c["select_slot"] = slot
        else:
            c["use_item"] = True
        return c

    def _melee_act(self, sim, idx: int) -> dict:
        return self._melee.act(sim, idx)


class ShieldHolder(_UBase):
    """Hold block while engaged; drop it to counter when the foe can't punish
    (foe on long cooldown and own swing ready)."""
    name = "shieldholder"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        d = _dist(sim, idx)
        if d < 4.5 and not (me.attack_cooldown <= 0 and foe.attack_cooldown > 8):
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            c["block"] = True
            return c
        return self._melee_act(sim, idx)


class GappleUser(_UBase):
    """Retreat and eat at low HP; else melee."""
    name = "gappleuser"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        inv = sim.inventories[idx]
        if (me.health < 9.0 or me.eating) and inv.available("golden_apple"):
            return self._item(sim, idx, GAPPLE, move_z=-1.0)
        return self._melee_act(sim, idx)


class PearlEscaper(_UBase):
    """Pearl away from the foe at low HP; else melee."""
    name = "pearlescaper"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        inv = sim.inventories[idx]
        if me.health < 8.0 and inv.available("ender_pearl") and not me.eating:
            c = _idle()
            away = me.pos - foe.pos
            import math
            # away-yaw: look_dir(yaw) = [cos yaw, sin yaw] must equal
            # away/|away|, i.e. atan2(away_z, away_x) (same form as _yaw_to).
            want = math.degrees(math.atan2(away[2], away[0]))
            diff = (want - me.yaw + 180.0) % 360.0 - 180.0
            c["yaw_delta"] = float(np.clip(diff, -30.0, 30.0))
            if inv.selected != PEARL:
                c["select_slot"] = PEARL
            else:
                c["use_item"] = True
            return c
        return self._melee_act(sim, idx)


class PearlChaser(_UBase):
    """Pearl toward a distant foe; else melee."""
    name = "pearlchaser"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        inv = sim.inventories[idx]
        if _dist(sim, idx) > 8.0 and inv.available("ender_pearl"):
            return self._item(sim, idx, PEARL, move_z=1.0)
        return self._melee_act(sim, idx)


class CobwebController(_UBase):
    """Web a nearby foe, then pressure; else melee."""
    name = "cobwebcontroller"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        inv = sim.inventories[idx]
        d = _dist(sim, idx)
        if d < 6.0 and inv.available("cobweb"):
            return self._item(sim, idx, WEB)
        if sim.field.web_at(float(foe.pos[0]), float(foe.pos[2])) and d < 4.0:
            return self._melee_act(sim, idx)
        return self._melee_act(sim, idx)


class BlockDefender(_UBase):
    """Drop a block at mid range, strafe behind it; else melee."""
    name = "blockdefender"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        inv = sim.inventories[idx]
        d = _dist(sim, idx)
        if d < 8.0 and inv.available("block") and len(sim.field.blocks) < 3:
            return self._item(sim, idx, BLOCK, move_x=0.5 * self._dir)
        if self._t % 40 == 0:
            self._dir *= -1.0
        return self._melee_act(sim, idx)


class PotionFighter(_UBase):
    """Drink strength early, then melee with the buff."""
    name = "potionfighter"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        inv = sim.inventories[idx]
        from sandbox.items import effects as FX
        if (self._t < 120 and not FX.has_effect(me, "strength")
                and inv.available("strength_potion") and not me.eating):
            return self._item(sim, idx, POT, move_z=1.0)
        return self._melee_act(sim, idx)


class RangedFighter(_UBase):
    """Keep 8-12 range and shoot; melee only when rushed."""
    name = "rangedfighter"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me = sim.players[idx]
        inv = sim.inventories[idx]
        d = _dist(sim, idx)
        if d < 3.0:
            return self._melee_act(sim, idx)
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        if d > 12.0:
            c["move_z"] = 1.0
        elif d < 8.0:
            c["move_z"] = -0.8
        else:
            c["move_x"] = 0.6 * self._dir
        if inv.selected != BOW:
            c["select_slot"] = BOW
        elif not inv.on_cooldown("bow") and inv.counts.get("arrow", 0) > 0:
            c["use_item"] = True
        if self._t % 45 == 0:
            self._dir *= -1.0
        return c


class MixedUtility(_UBase):
    """Gapple when hurt, shield vs attacks, pearl-chase far, else melee."""
    name = "mixedutility"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        inv = sim.inventories[idx]
        d = _dist(sim, idx)
        if (me.health < 9.0 or me.eating) and inv.available("golden_apple"):
            return self._item(sim, idx, GAPPLE, move_z=-1.0)
        if d < 4.5 and inv.available("shield"):
            if me.attack_cooldown <= 0 and foe.attack_cooldown > 8:
                return self._melee_act(sim, idx)
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            c["block"] = True
            return c
        if d > 8.0 and inv.available("ender_pearl"):
            return self._item(sim, idx, PEARL, move_z=1.0)
        return self._melee_act(sim, idx)


class PressureEater(_UBase):
    """Closes distance relentlessly; sprints at an eating agent."""
    name = "pressureeater"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        c["move_z"] = 1.0
        c["sprint"] = True
        if foe.eating:
            c["move_x"] = 0.3 * self._dir  # slight angle while charging
        if d <= 3.0 and me.attack_cooldown <= 0:
            c["attack"] = True
        if self._t % 40 == 0:
            self._dir *= -1.0
        return c


class PunishEater(_UBase):
    """Holds mid-range; when the agent starts eating (observable), sprints
    in to punish. Otherwise patient melee."""
    name = "punisheater"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        c = _idle()
        c["yaw_delta"] = _yaw_to(sim, idx)
        d = _dist(sim, idx)
        if foe.eating:
            c["move_z"] = 1.0
            c["sprint"] = True
            if d <= 3.0 and me.attack_cooldown <= 0:
                c["attack"] = True
        elif d > 5.0:
            c["move_z"] = 0.7
            c["move_x"] = 0.4 * self._dir
        elif d < 3.0:
            c["attack"] = me.attack_cooldown <= 0
        else:
            c["move_x"] = 0.6 * self._dir
        if self._t % 35 == 0:
            self._dir *= -1.0
        return c


class MixedEater(_UBase):
    """Melee + shield + reposition; always punishes eating agent."""
    name = "mixedeater"

    def act(self, sim, idx: int) -> dict:
        self._t += 1
        me, foe = sim.players[idx], sim.players[1 - idx]
        inv = sim.inventories[idx]
        d = _dist(sim, idx)
        if foe.eating:
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            c["move_z"] = 1.0
            c["sprint"] = True
            if d <= 3.0 and me.attack_cooldown <= 0:
                c["attack"] = True
            return c
        if d < 4.5 and inv.available("shield"):
            if me.attack_cooldown <= 0 and foe.attack_cooldown > 8:
                return self._melee_act(sim, idx)
            c = _idle()
            c["yaw_delta"] = _yaw_to(sim, idx)
            c["block"] = True
            return c
        c = self._melee_act(sim, idx)
        if d > 3.0:
            c["move_x"] = 0.5 * self._dir
        if self._t % 40 == 0:
            self._dir *= -1.0
        return c


UTILITY = {"shieldholder": ShieldHolder, "gappleuser": GappleUser,
           "pearlescaper": PearlEscaper, "pearlchaser": PearlChaser,
           "cobwebcontroller": CobwebController,
           "blockdefender": BlockDefender, "potionfighter": PotionFighter,
           "rangedfighter": RangedFighter, "mixedutility": MixedUtility,
           "pressureeater": PressureEater, "punisheater": PunishEater,
           "mixedeater": MixedEater}
