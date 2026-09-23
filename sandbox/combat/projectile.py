"""Deterministic projectiles: pearls (teleport) and arrows (damage).
Shared infrastructure for future throwable items."""
from __future__ import annotations

import numpy as np


class Projectile:
    def __init__(self, kind: str, owner: int, pos, vel, gravity: float,
                 damage: float = 0.0, life: int = 200):
        self.kind = kind
        self.owner = owner
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.gravity = gravity
        self.damage = damage
        self.life = life
        self.dead = False

    def step(self, sim, dt: float) -> dict | None:
        """Advance; returns landing event or None. Deterministic."""
        from sandbox.items import effects as FX
        self.vel[1] -= self.gravity * dt
        self.pos += self.vel * dt
        self.life -= 1
        x, y, z = (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        world = sim.world
        hit_world = y <= 0.02 or world.is_solid_at(x, y, z)
        # hit players (not owner; radius 0.6; arrow must be within body height)
        hit_p = None
        for i, p in enumerate(sim.players):
            if i == self.owner or not p.alive:
                continue
            d = p.pos - self.pos
            if d[0] * d[0] + d[2] * d[2] < 0.36 and -1.9 < d[1] < 0.2:
                hit_p = i
                break
        if self.kind == "pearl" and (hit_world or self.life <= 0):
            self.dead = True
            owner_p = sim.players[self.owner]
            land = self.pos.copy()
            land[1] = max(0.0, land[1])
            # nudge out of solid
            for _ in range(4):
                if not world.is_solid_at(land[0], land[1] + 0.1, land[2]):
                    break
                land[1] += 1.0
            owner_p.pos = land
            owner_p.vel[:] = 0.0
            owner_p.on_ground = True
            owner_p.health = max(0.0, owner_p.health - 2.0)  # pearl self-damage
            if owner_p.health <= 0.0:
                owner_p.alive = False
            return {"type": "pearl_land", "owner": self.owner,
                    "pos": land.copy()}
        if self.kind == "arrow" and (hit_world or self.life <= 0):
            self.dead = True
            return {"type": "arrow_stuck", "owner": self.owner}
        if self.kind == "arrow" and hit_p is not None:
            self.dead = True
            vic = sim.players[hit_p]
            dmg = self.damage
            absorbed = min(vic.absorption_hp, dmg)
            vic.absorption_hp -= absorbed
            vic.health = max(0.0, vic.health - (dmg - absorbed))
            vic.hurt_time = 10
            if vic.health <= 0.0:
                vic.alive = False
            return {"type": "arrow_hit", "owner": self.owner,
                    "victim": hit_p, "damage": float(dmg)}
        return None
