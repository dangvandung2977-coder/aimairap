"""Golden test: replay Minecraft-recorded observations through the checkpoint.

  python -m deployment.replay_golden debug/recordings/obs_000123.json [...]
  python -m deployment.replay_golden debug/recordings/

Each file: {"tick": int, "observation": [73]} (as dumped by /pvpbot debug).
Verifies: exactly 73 dims -> valid PPO input -> legal action, and identical
action on deterministic repeat. Exit 0 iff every file passes.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from deployment.action_validator import validate_action
from deployment.model_runner import ModelRunner
from deployment.observation_validator import validate_observation


def check_file(runner: ModelRunner, path: str) -> tuple[bool, str]:
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        return False, f"{path}: unreadable ({e})"
    obs = rec.get("observation")
    ok, info = validate_observation(obs or [])
    if not ok:
        return False, f"{path}: invalid PPO input ({info.get('error')})"
    try:
        a1, _ = runner.predict(obs)
        a2, _ = runner.predict(obs)
    except (ValueError, RuntimeError) as e:
        return False, f"{path}: inference failed ({e})"
    if list(a1) != list(a2):
        return False, f"{path}: NON-DETERMINISTIC {a1} vs {a2}"
    ok, _ = validate_action(a1)
    if not ok:
        return False, f"{path}: illegal action {a1}"
    return True, f"{path}: tick={rec.get('tick')} action={a1} OK"


def main(paths: list[str]) -> int:
    files: list[str] = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) \
            else [p]
    if not files:
        print("no recording files")
        return 2
    runner = ModelRunner()
    bad = 0
    for f in files:
        ok, msg = check_file(runner, f)
        print(("PASS " if ok else "FAIL ") + msg)
        bad += not ok
    print(f"{len(files) - bad}/{len(files)} golden files pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
