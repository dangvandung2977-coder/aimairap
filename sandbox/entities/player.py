"""Player entity state."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlayerState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    yaw: float = 0.0    # degrees, [-180, 180]; 0 -> +X, 90 -> +Z
    pitch: float = 0.0  # degrees, [-90, 90]; +up
    health: float = 20.0
    max_health: float = 20.0
    on_ground: bool = True
    sprinting: bool = False
    attack_cooldown: int = 0
    hurt_time: int = 0
    fall_distance: float = 0.0
    alive: bool = True
    # --- phase-4 utility state ---
    effects: dict = field(default_factory=dict)
    absorption_hp: float = 0.0
    hunger: float = 20.0
    shield_up: bool = False
    shield_timer: int = 0
    eating: list = field(default_factory=list)  # [item_id, ticks_left] or []

    def reset(self, pos, yaw: float, health: float | None = None):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = float(yaw)
        self.pitch = 0.0
        self.health = float(health if health is not None else self.max_health)
        self.on_ground = True
        self.sprinting = False
        self.attack_cooldown = 0
        self.hurt_time = 0
        self.fall_distance = 0.0
        self.alive = True
        self.effects = {}
        self.absorption_hp = 0.0
        self.hunger = 20.0
        self.shield_up = False
        self.shield_timer = 0
        self.eating = []

    def snapshot(self) -> dict:
        return {
            "pos": self.pos.copy(), "vel": self.vel.copy(),
            "yaw": self.yaw, "pitch": self.pitch, "health": self.health,
            "on_ground": self.on_ground, "sprinting": self.sprinting,
            "attack_cooldown": self.attack_cooldown, "hurt_time": self.hurt_time,
            "fall_distance": self.fall_distance, "alive": self.alive,
        }
