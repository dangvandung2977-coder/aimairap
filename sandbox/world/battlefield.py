"""Placed blocks, cobwebs, fluids. Cheap deterministic lists; arena-bounded."""
from __future__ import annotations

import numpy as np


class PlacedBlock:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Cobweb:
    def __init__(self, x: float, z: float, duration: int = 10**9):
        self.x, self.z = float(x), float(z)
        self.duration = duration


class Fluid:
    def __init__(self, kind: str, x: float, z: float, duration: int = 600):
        assert kind in ("water", "lava")
        self.kind, self.x, self.z = kind, float(x), float(z)
        self.duration = duration


class BattleField:
    """Dynamic battlefield state, reset per episode."""

    def __init__(self, max_blocks: int = 24, max_webs: int = 6):
        self.blocks: list[PlacedBlock] = []
        self.webs: list[Cobweb] = []
        self.fluids: list[Fluid] = []
        self.max_blocks = max_blocks
        self.max_webs = max_webs

    def reset(self) -> None:
        self.blocks.clear()
        self.webs.clear()
        self.fluids.clear()

    def add_block(self, x: float, y: float, z: float, size: int) -> bool:
        if len(self.blocks) >= self.max_blocks:
            return False
        if not (0 <= x <= size and 0 <= z <= size and 0 <= y <= 6):
            return False
        for b in self.blocks:
            if abs(b.x - x) < 0.9 and abs(b.z - z) < 0.9 and abs(b.y - y) < 0.9:
                return False
        self.blocks.append(PlacedBlock(x, y, z))
        return True

    def remove_near(self, x: float, y: float, z: float, r: float = 2.0) -> bool:
        for i, b in enumerate(self.blocks):
            if (b.x - x) ** 2 + (b.y - y) ** 2 + (b.z - z) ** 2 < r * r:
                del self.blocks[i]
                return True
        return False

    def add_web(self, x: float, z: float, size: int) -> bool:
        if len(self.webs) >= self.max_webs:
            return False
        if not (0 <= x <= size and 0 <= z <= size):
            return False
        self.webs.append(Cobweb(x, z))
        return True

    def add_fluid(self, kind: str, x: float, z: float, size: int) -> bool:
        if not (0 <= x <= size and 0 <= z <= size):
            return False
        self.fluids.append(Fluid(kind, x, z))
        return True

    def web_at(self, x: float, z: float) -> bool:
        return any(abs(w.x - x) < 0.8 and abs(w.z - z) < 0.8 for w in self.webs)

    def fluid_at(self, x: float, z: float) -> str | None:
        for fl in self.fluids:
            if abs(fl.x - x) < 1.2 and abs(fl.z - z) < 1.2:
                return fl.kind
        return None

    def solid_at(self, x: float, y: float, z: float) -> bool:
        for b in self.blocks:
            if (abs(b.x - x) < 0.5 and abs(b.z - z) < 0.5
                    and b.y <= y < b.y + 1.0):
                return True
        return False

    def tick(self) -> None:
        for w in self.webs:
            w.duration -= 1
        self.webs = [w for w in self.webs if w.duration > 0]
        for fl in self.fluids:
            fl.duration -= 1
        self.fluids = [fl for fl in self.fluids if fl.duration > 0]
