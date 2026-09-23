"""Efficient per-tick replay recorder (npz)."""
from __future__ import annotations

import os

import numpy as np


class ReplayRecorder:
    def __init__(self):
        self.frames: list[dict] = []
        self.meta: dict = {}
        self._reward_sum = 0.0

    def begin(self, sim) -> None:
        self.frames = []
        self._reward_sum = 0.0
        self.meta = {"size": sim.world.size,
                     "max_ticks": sim.max_ticks,
                     "world": sim.world.to_dict()}

    def capture(self, sim, ctrl0: dict, ctrl1: dict, reward: float, comp: dict) -> None:
        p0, p1 = sim.players[0], sim.players[1]
        self._reward_sum += float(reward)
        from sandbox.items import effects as FX
        from sandbox.items.defs import HOTBAR
        fx_bits = lambda p: (1 if FX.has_effect(p, "regeneration") else 0) | \
            (2 if FX.has_effect(p, "speed") else 0) | \
            (4 if FX.has_effect(p, "strength") else 0) | \
            (8 if FX.has_effect(p, "fire_resistance") else 0)
        nproj = len(sim.projectiles)
        proj = sim.projectiles[0].pos - p0.pos if nproj else np.zeros(3)
        self.frames.append({
            "tick": sim.tick,
            "p0_pos": p0.pos.copy(), "p1_pos": p1.pos.copy(),
            "p0_vel": p0.vel.copy(), "p1_vel": p1.vel.copy(),
            "p0_yaw": p0.yaw, "p1_yaw": p1.yaw,
            "p0_pitch": p0.pitch, "p1_pitch": p1.pitch,
            "p0_hp": p0.health, "p1_hp": p1.health,
            "p0_cd": p0.attack_cooldown, "p1_cd": p1.attack_cooldown,
            "p0_hurt": p0.hurt_time, "p1_hurt": p1.hurt_time,
            "p0_ground": p0.on_ground, "p1_ground": p1.on_ground,
            "p0_sprint": p0.sprinting, "p1_sprint": p1.sprinting,
            "d0": sim.damage_dealt[0], "d1": sim.damage_dealt[1],
            "a0_attack": bool(ctrl0.get("attack", False)),
            "a1_attack": bool(ctrl1.get("attack", False)),
            "a0_mx": float(ctrl0.get("move_x", 0.0)),
            "a0_mz": float(ctrl0.get("move_z", 0.0)),
            "a0_sprint": bool(ctrl0.get("sprint", False)),
            "a0_jump": bool(ctrl0.get("jump", False)),
            "a0_yawd": float(ctrl0.get("yaw_delta", 0.0)),
            "reward": float(reward),
            # --- utility (phase 4) ---
            "sel": sim.inventories[0].selected,
            "cnt": np.array([sim.inventories[0].counts.get(it, 0) for it in HOTBAR]),
            "cd_pearl": sim.inventories[0].cooldowns.get("ender_pearl", 0),
            "cd_gapple": sim.inventories[0].cooldowns.get("golden_apple", 0),
            "cd_bow": sim.inventories[0].cooldowns.get("bow", 0),
            "sh0": p0.shield_up, "sh1": p1.shield_up,
            "abs0": p0.absorption_hp, "abs1": p1.absorption_hp,
            "eat0": p0.eating[1] if p0.eating else -1,
            "eat1": p1.eating[1] if p1.eating else -1,
            "fx0": fx_bits(p0), "fx1": fx_bits(p1),
            "hunger0": p0.hunger,
            "nproj": nproj,
            "proj0": np.array(proj, dtype=float),
            "nblocks": len(sim.field.blocks),
            "nwebs": len(sim.field.webs),
            "nfluids": len(sim.field.fluids),
            "nutil": len(sim.utility_events),
        })

    def finish(self, sim) -> None:
        self.meta.update({"winner": sim.winner, "ticks": sim.tick,
                          "damage": list(sim.damage_dealt),
                          "total_reward": self._reward_sum})

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        keys = self.frames[0].keys() if self.frames else []
        arr = {k: np.array([f[k] for f in self.frames]) for k in keys}
        np.savez_compressed(path, **arr, meta=np.array([self.meta], dtype=object))
        return path


def load_replay(path: str) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=True)
    raw = z["meta"][0]
    meta = dict(raw.item() if hasattr(raw, "item") else raw)
    frames = {k: z[k] for k in z.files if k != "meta"}
    return frames, meta
