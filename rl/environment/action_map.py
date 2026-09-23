"""Compact high-level action space. Policy -> controller dict translation.

v2 MultiDiscrete [move(5), sprint(2), jump(2), attack(2), block(2), use(2),
                  slot(9), yaw(3), pitch(3)]:
  move: 0 stop, 1 forward, 2 back, 3 left, 4 right
  block: hold shield (needs owned shield; sword attack takes precedence)
  use: activate selected hotbar slot through the validated item pipeline
  slot: select hotbar slot (2-tick switch cost; see HOTBAR layout in defs.py)
  yaw:  0 -YAW_STEP, 1 keep, 2 +YAW_STEP  (per decision; repeated via frame-skip)
  pitch: 0 -PITCH_STEP, 1 keep, 2 +PITCH_STEP
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from sandbox.items.defs import HOTBAR

YAW_STEP = 15.0
PITCH_STEP = 10.0

NVEC = [5, 2, 2, 2, 2, 2, 9, 3, 3]


def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(NVEC)


def to_controller(a) -> dict:
    a = [int(x) for x in np.asarray(a).flatten().tolist()]
    move, sprint, jump, attack, block, use, slot, yaw, pitch = a
    mx, mz = 0.0, 0.0
    if move == 1:
        mz = 1.0
    elif move == 2:
        mz = -1.0
    elif move == 3:
        mx = -1.0
    elif move == 4:
        mx = 1.0
    return {
        "move_x": mx, "move_z": mz,
        "sprint": bool(sprint), "jump": bool(jump), "attack": bool(attack),
        "block": bool(block), "use_item": bool(use),
        "select_slot": int(slot),
        "yaw_delta": float([-YAW_STEP, 0.0, YAW_STEP][yaw]),
        "pitch_delta": float([-PITCH_STEP, 0.0, PITCH_STEP][pitch]),
    }


def random_controller(rng: np.random.Generator) -> dict:
    return to_controller(random_action(rng))


def random_action(rng: np.random.Generator) -> list:
    return [int(rng.integers(n)) for n in NVEC]


def describe(a) -> str:
    names = ["stop", "fwd", "back", "left", "right"]
    a = [int(x) for x in np.asarray(a).flatten().tolist()]
    yaw = ["L", "-", "R"][a[7]]
    extra = ""
    if a[4]:
        extra += "B"
    if a[5]:
        extra += f"U{HOTBAR[a[6]]}"
    return f"{names[a[0]]}{'S' if a[1] else ''}{'J' if a[2] else ''}{'X' if a[3] else ''}{extra}{yaw}"
