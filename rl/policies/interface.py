"""Policy interface shared by SB3 now and the future NeoForge 1.21.1 adapter.

Contract: obs (Box[39] v1) in -> MultiDiscrete[5,2,2,2,3,3] out.
The adapter will translate game state to the same obs and controller actions
to clicks/keys; policies stay portable.
"""
from __future__ import annotations

import numpy as np

from rl.environment import action_map, obs_schema

OBS_VERSION = obs_schema.OBS_VERSION
OBS_DIM = obs_schema.OBS_DIM


class PolicyInterface:
    def predict(self, obs: np.ndarray):
        raise NotImplementedError

    def action_space(self):
        return action_map.action_space()

    def observation_space(self):
        return obs_schema.obs_space()
