"""Voxel arena. World model is independent from rendering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Obstacle:
    x0: float
    z0: float
    x1: float
    z1: float
    height: float


class VoxelWorld:
    """Flat floor + boundary walls + optional AABB pillar obstacles.

    Block API (clean, minimal):
      get_block(x, y, z) -> 0 air / 1 solid
      is_solid_at(x, y, z) -> bool
    Coordinates: x,z in [0, size], y >= 0. Feet position; floor top at y=0.
    """

    AIR = 0
    SOLID = 1

    def __init__(self, size: int = 20, wall_margin: float = 1.0, obstacles=()):
        self.size = int(size)
        self.wall_margin = float(wall_margin)
        self.set_obstacles(obstacles or ())
        lo = self.wall_margin
        hi = self.size - self.wall_margin
        self.lo = lo
        self.hi = hi
        self.field = None  # BattleField, attached by PvPSim (placed blocks)

    def set_obstacles(self, obstacles) -> None:
        self.obstacles = [Obstacle(*o) for o in (obstacles or ())]

    # -- block API ---------------------------------------------------------
    def is_solid_at(self, x: float, y: float, z: float) -> bool:
        if y < 0:
            return True
        if self.field is not None and self.field.solid_at(x, y, z):
            return True
        if x < self.lo or x > self.hi or z < self.lo or z > self.hi:
            # walls extend infinitely high around the playable area
            return y < 4.0
        for o in self.obstacles:
            if o.x0 <= x <= o.x1 and o.z0 <= z <= o.z1 and y < o.height:
                return True
        return False

    def get_block(self, x: float, y: float, z: float) -> int:
        return self.SOLID if self.is_solid_at(x, y, z) else self.AIR

    # -- collision ---------------------------------------------------------
    def collide_move(self, pos: np.ndarray, vel: np.ndarray, dt: float,
                     radius: float, height: float = 1.8):
        """Move a capsule approximated as a point with radius; resolve floor,
        walls and obstacle pillars. Returns (new_pos, on_ground, hit_head)."""
        p = pos.astype(np.float64).copy()
        # integrate
        p = p + vel.astype(np.float64) * dt
        on_ground = False
        # floor
        if p[1] <= 0.0:
            p[1] = 0.0
            on_ground = True
        else:
            # land on top of obstacle pillars
            for o in self.obstacles:
                if o.x0 - radius <= p[0] <= o.x1 + radius and o.z0 - radius <= p[2] <= o.z1 + radius:
                    if vel[1] <= 0.0 and pos[1] >= o.height - 1e-9 and p[1] <= o.height:
                        p[1] = o.height
                        on_ground = True
        # walls (clamp center so body stays inside)
        r = radius
        if p[0] < self.lo + r:
            p[0] = self.lo + r
        elif p[0] > self.hi - r:
            p[0] = self.hi - r
        if p[2] < self.lo + r:
            p[2] = self.lo + r
        elif p[2] > self.hi - r:
            p[2] = self.hi - r
        # pillar side push-out (only when body intersects vertically)
        feet, head = p[1], p[1] + height
        for o in self.obstacles:
            if head <= 0.0 or feet >= o.height:
                continue
            cx = min(max(p[0], o.x0), o.x1)
            cz = min(max(p[2], o.z0), o.z1)
            dx, dz = p[0] - cx, p[2] - cz
            d2 = dx * dx + dz * dz
            if d2 < r * r:
                if d2 < 1e-12:
                    # inside: push out along smallest penetration axis
                    pl = min(p[0] - o.x0, o.x1 - p[0])
                    pw = min(p[2] - o.z0, o.z1 - p[2])
                    if pl < pw:
                        p[0] = o.x0 - r if (p[0] - o.x0) < (o.x1 - p[0]) else o.x1 + r
                    else:
                        p[2] = o.z0 - r if (p[2] - o.z0) < (o.z1 - p[2]) else o.z1 + r
                else:
                    d = float(np.sqrt(d2))
                    p[0] = cx + dx / d * r
                    p[2] = cz + dz / d * r
        # standing check: on pillar top counts as ground
        if not on_ground and p[1] > 0.0:
            for o in self.obstacles:
                if abs(p[1] - o.height) < 1e-9 and o.x0 - r <= p[0] <= o.x1 + r and o.z0 - r <= p[2] <= o.z1 + r:
                    on_ground = True
                    break
        return p, on_ground, False

    # -- sensors for WORLD obs ---------------------------------------------
    def wall_distances(self, pos: np.ndarray) -> np.ndarray:
        x, z = float(pos[0]), float(pos[2])
        return np.array([x - self.lo, self.hi - x, z - self.lo, self.hi - z], dtype=np.float64)

    def obstacle_sensor(self, pos: np.ndarray) -> np.ndarray:
        """4 binary cells: obstacle solid within 2.0m in +X/-X/+Z/-Z of feet."""
        out = np.zeros(4, dtype=np.float64)
        for i, (dx, dz) in enumerate(((2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0))):
            for t in (0.75, 1.5, 2.0):
                x = float(pos[0]) + dx * (t / 2.0)
                z = float(pos[2]) + dz * (t / 2.0)
                if self.is_solid_at(x, 0.5, z):
                    # ignore outer walls beyond margin: sensor is for obstacles;
                    # walls are covered by wall_distances. Only count pillars.
                    inside = self.lo <= x <= self.hi and self.lo <= z <= self.hi
                    pillar = any(o.x0 <= x <= o.x1 and o.z0 <= z <= o.z1 for o in self.obstacles)
                    if inside and pillar:
                        out[i] = 1.0
                        break
        return out

    def to_dict(self) -> dict:
        return {"size": self.size, "wall_margin": self.wall_margin,
                "obstacles": [(o.x0, o.z0, o.x1, o.z1, o.height) for o in self.obstacles]}
