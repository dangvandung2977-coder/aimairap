"""Action validation for the inference bridge.

Authoritative contract: rl/environment/action_map.py
  NVEC = [5, 2, 2, 2, 2, 2, 9, 3, 3], YAW_STEP = 15.0, PITCH_STEP = 10.0.

Action index -> Minecraft effect (traceable to to_controller(), same file):
  action[0] move   0 stop, 1 forward, 2 back, 3 left, 4 right
                   (move_z=+1 fwd / -1 back, move_x=-1 left / +1 right)
  action[1] sprint 0/1 (only applies while moving forward, engine.py)
  action[2] jump   0/1 (only when on_ground, engine.py)
  action[3] attack 0/1 (sword swing; takes precedence over block, sim.py)
  action[4] block  0/1 (raise shield; needs owned shield; 2-tick raise, sim.py)
  action[5] use    0/1 (activate selected hotbar slot via item pipeline, use.py)
  action[6] slot   0-8 -> HOTBAR (sandbox/items/defs.py:76):
                   0 sword, 1 shield, 2 bread, 3 golden_apple, 4 ender_pearl,
                   5 cobweb, 6 block, 7 strength_potion, 8 bow
                   (2-tick switch lockout, inventory.py)
  action[7] yaw    0 -15 deg, 1 keep, 2 +15 deg (per decision, engine.wrap_yaw)
  action[8] pitch  0 -10 deg, 1 keep, 2 +10 deg (clamped to [-90, 90])
"""
from __future__ import annotations

from rl.environment.action_map import NVEC  # authoritative, not a copy

NAMES = ["move", "sprint", "jump", "attack", "block", "use", "slot", "yaw", "pitch"]


def validate_action(act) -> tuple[bool, dict]:
    """Returns (ok, info). Never raises on bad input."""
    info = {"expected_len": len(NVEC), "nvec": list(NVEC)}
    try:
        vals = [int(x) for x in list(act)]
    except (TypeError, ValueError):
        return False, {**info, "raw_shape": None, "error": "not_an_int_list"}
    info["raw_shape"] = len(vals)
    if len(vals) != len(NVEC):
        return False, {**info, "error": f"need {len(NVEC)} dims"}
    for i, (v, n) in enumerate(zip(vals, NVEC)):
        if not 0 <= v < n:
            return False, {**info, "error": f"{NAMES[i]}[{i}]={v} outside [0,{n})"}
    return True, {**info, "named": dict(zip(NAMES, vals))}
