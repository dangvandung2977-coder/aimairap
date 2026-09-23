"""Hotbar inventory: fixed slots, counts, selection, cooldowns, consumption."""
from __future__ import annotations

from sandbox.items.defs import HOTBAR, REGISTRY


class Inventory:
    def __init__(self, loadout: dict[str, int] | None = None):
        # counts per item id; sword/shield/buckets/bow are tools (count>=1 = owned)
        self.counts: dict[str, int] = dict(loadout) if loadout else {}
        self.selected: int = 0
        self.cooldowns: dict[str, int] = {}
        self.switch_timer: int = 0
        self.enabled: set[str] = set(HOTBAR)  # curriculum lock

    def reset(self, loadout: dict[str, int] | None = None,
              enabled: set[str] | None = None) -> None:
        self.counts = dict(loadout) if loadout else {}
        self.selected = 0
        self.cooldowns = {}
        self.switch_timer = 0
        if enabled is not None:
            self.enabled = set(enabled)

    def slot_item(self, slot: int) -> str:
        return HOTBAR[slot % len(HOTBAR)]

    def selected_item(self) -> str:
        return self.slot_item(self.selected)

    def available(self, item_id: str) -> bool:
        if item_id not in self.enabled:
            return False
        d = REGISTRY[item_id]
        if d.consumable:
            return self.counts.get(item_id, 0) > 0
        if item_id == "bow":
            return True  # bow itself infinite; arrows counted separately
        return self.counts.get(item_id, 0) > 0

    def on_cooldown(self, item_id: str) -> bool:
        return self.cooldowns.get(item_id, 0) > 0

    def set_cooldown(self, item_id: str) -> None:
        cd = REGISTRY[item_id].cooldown_ticks
        if cd:
            self.cooldowns[item_id] = cd

    def consume(self, item_id: str) -> bool:
        if self.counts.get(item_id, 0) <= 0:
            return False
        self.counts[item_id] -= 1
        return True

    def select(self, slot: int) -> bool:
        slot = int(slot) % len(HOTBAR)
        if slot == self.selected:
            return False
        self.selected = slot
        self.switch_timer = REGISTRY[self.slot_item(slot)].switch_cost_ticks
        return True

    def tick(self) -> None:
        for k in list(self.cooldowns):
            self.cooldowns[k] -= 1
            if self.cooldowns[k] <= 0:
                del self.cooldowns[k]
        if self.switch_timer > 0:
            self.switch_timer -= 1

    def snapshot(self) -> dict:
        return {"counts": dict(self.counts), "selected": self.selected,
                "cooldowns": dict(self.cooldowns),
                "switch_timer": self.switch_timer}


DEFAULT_LOADOUT = {"sword": 1, "shield": 1, "bread": 5, "golden_apple": 2,
                   "ender_pearl": 4, "cobweb": 4, "block": 16,
                   "strength_potion": 1, "bow": 1, "arrow": 16}
