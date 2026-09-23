"""Item registry. Adding an item = one entry here + (optionally) an applier
in use.py. No RL architecture changes needed. Numbers are sandbox
approximations, isolated here for later 1.21.1 calibration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ItemDefinition:
    item_id: str
    category: str  # MELEE DEFENSE FOOD HEALING MOBILITY PROJECTILE BLOCK UTILITY POTION RANGED
    stack_size: int = 1
    consumable: bool = False
    cooldown_ticks: int = 0
    use_time_ticks: int = 0  # hold-to-use duration (food/potions/bow-draw)
    effect: str = ""         # status effect id applied on consume
    effect_duration: int = 0
    effect_amp: int = 0
    damage: float = 0.0
    heal: float = 0.0
    hunger: int = 0
    absorption: float = 0.0
    projectile: str = ""     # projectile type spawned on use
    switch_cost_ticks: int = 2  # hotbar switch lockout


REGISTRY: dict[str, ItemDefinition] = {}


def _reg(d: ItemDefinition) -> ItemDefinition:
    REGISTRY[d.item_id] = d
    return d


SWORD = _reg(ItemDefinition("sword", "MELEE", stack_size=1))
SHIELD = _reg(ItemDefinition("shield", "DEFENSE", stack_size=1))
BREAD = _reg(ItemDefinition("bread", "FOOD", stack_size=16, consumable=True,
                            use_time_ticks=32, heal=2.0, hunger=6))
GAPPLE = _reg(ItemDefinition("golden_apple", "HEALING", stack_size=8,
                             consumable=True, use_time_ticks=32,
                             cooldown_ticks=100, effect="regeneration",
                             effect_duration=200, effect_amp=0,
                             absorption=8.0, hunger=4))
PEARL = _reg(ItemDefinition("ender_pearl", "MOBILITY", stack_size=16,
                            consumable=True, cooldown_ticks=20,
                            projectile="pearl", damage=2.0))
COBWEB = _reg(ItemDefinition("cobweb", "BLOCK", stack_size=16, consumable=True,
                             cooldown_ticks=10))
BLOCK = _reg(ItemDefinition("block", "BLOCK", stack_size=64, consumable=True,
                            cooldown_ticks=4))
WATER = _reg(ItemDefinition("water_bucket", "UTILITY", stack_size=1,
                            consumable=False, cooldown_ticks=20))
LAVA = _reg(ItemDefinition("lava_bucket", "UTILITY", stack_size=1,
                           consumable=False, cooldown_ticks=20))
SPEED_POT = _reg(ItemDefinition("speed_potion", "POTION", stack_size=1,
                                consumable=True, use_time_ticks=32,
                                effect="speed", effect_duration=1200,
                                effect_amp=0))
STR_POT = _reg(ItemDefinition("strength_potion", "POTION", stack_size=1,
                              consumable=True, use_time_ticks=32,
                              effect="strength", effect_duration=1200,
                              effect_amp=0))
FIRERES_POT = _reg(ItemDefinition("fire_resistance_potion", "POTION",
                                  stack_size=1, consumable=True,
                                  use_time_ticks=32, effect="fire_resistance",
                                  effect_duration=1200, effect_amp=0))
HEAL_POT = _reg(ItemDefinition("healing_potion", "POTION", stack_size=1,
                               consumable=True, use_time_ticks=20,
                               effect="", heal=8.0))
BOW = _reg(ItemDefinition("bow", "RANGED", stack_size=1, cooldown_ticks=15,
                          projectile="arrow", damage=6.0))

# Fixed hotbar layout (slot index -> item). Locked slots are enforced by the
# curriculum's enabled_items set, not by changing this layout.
HOTBAR = ["sword", "shield", "bread", "golden_apple", "ender_pearl",
          "cobweb", "block", "strength_potion", "bow"]

CATEGORIES = ("MELEE", "DEFENSE", "FOOD", "HEALING", "MOBILITY", "PROJECTILE",
              "BLOCK", "UTILITY", "POTION", "RANGED")
