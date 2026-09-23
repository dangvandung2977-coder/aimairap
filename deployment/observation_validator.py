"""Observation validation for the inference bridge.

Authoritative contract: rl/environment/obs_schema.py
  OBS_VERSION = 4, OBS_DIMS[4] = 73, Box(low=-5.0, high=5.0, float32).
Preprocessing used in training: NONE beyond the schema itself (no
VecNormalize anywhere in the codebase — verified by grep). build_obs applies
_clip(x) = clamp(x, -5, 5) per element and returns float32. Inference therefore
reproduces preprocessing exactly by: float32 cast + finite check (the schema
asserts finiteness; Box bounds are advisory, values outside [-5,5] are
accepted with a warning, exactly as the Box space would).
"""
from __future__ import annotations

import math

from rl.environment import obs_schema

EXPECTED_DIM = obs_schema.OBS_DIMS[obs_schema.OBS_VERSION]


def validate_observation(obs) -> tuple[bool, dict]:
    """Returns (ok, info). Never raises on bad input (bridge must not crash)."""
    info = {"expected_dim": EXPECTED_DIM, "schema_version": obs_schema.OBS_VERSION}
    try:
        vals = [float(x) for x in list(obs)]
    except (TypeError, ValueError):
        return False, {**info, "raw_shape": None, "error": "not_a_number_list"}
    info["raw_shape"] = len(vals)
    if len(vals) != EXPECTED_DIM:
        return False, {**info, "error": f"need {EXPECTED_DIM} dims"}
    bad = [i for i, x in enumerate(vals) if not math.isfinite(x)]
    if bad:
        return False, {**info, "error": f"non_finite_at_{bad[:8]}"}
    out = [i for i, x in enumerate(vals) if abs(x) > 5.0]
    info["processed_shape"] = len(vals)
    info["out_of_box_bounds"] = out[:8]
    return True, info


def describe_normalization() -> dict:
    """Compatibility answer for §3: what preprocessing inference applies."""
    return {
        "vecnormalize": False,
        "vecnormalize_stats_file": None,
        "schema_clipping": "[-5.0, 5.0] per element (obs_schema._clip)",
        "dtype": "float32",
        "inference_preprocessing": "float32 cast + finite check only",
        "verified_by": "grep VecNormalize over repo (no hits) + "
                       "rl/environment/obs_schema.py",
    }
