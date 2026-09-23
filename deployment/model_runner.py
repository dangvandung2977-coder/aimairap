"""Read-only PPO model runner for deployment. Loads once, infers deterministically.

Never trains, never writes checkpoints. Verifies the canonical SHA256 before
loading; any mismatch raises and the server refuses to start.
"""
from __future__ import annotations

import hashlib
import os
import time

CANONICAL_SHA256 = "660d524dab2010fc390d2a82be52a7735f2f307ad3291c835f1ac23e7e8d5e3e"
CANONICAL_PATH = os.path.join("checkpoints", "phase4_stage3", "3a2", "final.zip")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRunner:
    def __init__(self, checkpoint: str = CANONICAL_PATH,
                 expected_sha256: str = CANONICAL_SHA256):
        from stable_baselines3 import PPO
        self.checkpoint = checkpoint
        self.actual_sha256 = sha256_file(checkpoint)
        if self.actual_sha256 != expected_sha256:
            raise ValueError(
                f"checkpoint hash mismatch: got {self.actual_sha256}, "
                f"expected {expected_sha256}. REFUSING to serve.")
        self.model = PPO.load(checkpoint)
        self.policy_name = type(self.model.policy).__name__
        self.obs_dim = int(self.model.observation_space.shape[0])
        from rl.environment.action_map import NVEC
        self.nvec = list(NVEC)
        assert list(map(int, self.model.action_space.nvec)) == self.nvec
        # warmup (allocates, proves predict path) with a real-schema vector
        import numpy as np
        self.predict(np.zeros(self.obs_dim, dtype=np.float32))

    def predict(self, obs) -> tuple[list[int], float]:
        """Deterministic inference. Returns (action, inference_ms)."""
        import numpy as np
        from deployment.action_validator import validate_action
        from deployment.observation_validator import validate_observation
        ok, _ = validate_observation(obs)
        if not ok:
            raise ValueError("invalid observation for inference")
        arr = np.asarray(list(obs), dtype=np.float32)
        t0 = time.perf_counter()
        act, _ = self.model.predict(arr, deterministic=True)
        dt = (time.perf_counter() - t0) * 1000.0
        vals = [int(x) for x in np.asarray(act).flatten().tolist()]
        ok, _ = validate_action(vals)
        if not ok:
            raise RuntimeError(f"model emitted illegal action {vals}")
        return vals, dt

    def compat_report(self) -> dict:
        """Smoke evidence through the AUTHORITATIVE obs pipeline (sim-built)."""
        import numpy as np
        from rl.environment import obs_schema
        from sandbox.simulation.sim import PvPSim
        from deployment.observation_validator import describe_normalization
        sim = PvPSim(seed=0)
        sim.reset(seed=0)
        raw = obs_schema.build_obs(sim, agent=0, version=4)
        act, ms = self.predict(raw)
        act2, _ = self.predict(raw)
        return {
            "checkpoint": self.checkpoint,
            "sha256": self.actual_sha256,
            "sha_match": self.actual_sha256 == CANONICAL_SHA256,
            "model_type": "PPO/ActorCriticPolicy(MlpPolicy)",
            "policy_class": self.policy_name,
            "raw_observation_shape": int(raw.shape[0]),
            "processed_observation_shape": int(raw.shape[0]),
            "expected_shape": 73,
            "normalization": describe_normalization(),
            "action": act,
            "action_shape": len(act),
            "deterministic_repeat_identical": bool(
                (np.asarray(act) == np.asarray(act2)).all()),
            "finite": bool(np.all(np.isfinite(raw))),
            "inference_ms": round(ms, 3),
        }
