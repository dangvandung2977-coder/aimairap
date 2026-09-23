"""Attrition opponents: foe-side regeneration carried as sim params.

Regen lives in the SIM (observable via foe HP rising), not in the brain.
Brains are ordinary melee/defensive drivers. Deterministic; seeds as usual.
"""
from __future__ import annotations

from rl.opponents.scripted import FighterOpponent, MeleeOpponent, Opponent


class _RegenBase(Opponent):
    SIM_PARAMS: dict = {"rate": 0.0, "delay": 0}
    name = "regen"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self._driver = MeleeOpponent()
        self._driver.reset(self.seed if seed is None else seed)

    def act(self, sim, idx: int) -> dict:
        return self._driver.act(sim, idx)


class RegenA(_RegenBase):
    """Low regen: 0.5 HP/s after 5s quiet."""
    name = "regenA"
    SIM_PARAMS = {"rate": 0.025, "delay": 100}


class RegenB(_RegenBase):
    """Moderate regen: 1 HP/s after 3s quiet."""
    name = "regenB"
    SIM_PARAMS = {"rate": 0.05, "delay": 60}


class RegenC(_RegenBase):
    """Moderate regen + defensive spacing brain."""
    name = "regenC"
    SIM_PARAMS = {"rate": 0.05, "delay": 60}

    def reset(self, seed: int | None = None) -> None:
        from rl.opponents.scripted import create_profile
        self._driver = create_profile("defensive",
                                      seed=self.seed if seed is None else seed)


REGEN = {"regenA": RegenA, "regenB": RegenB, "regenC": RegenC}
