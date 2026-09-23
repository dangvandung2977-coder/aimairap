"""Versioned JSON-newline bridge protocol (Python <-> Minecraft mod).

Wire format (one JSON object per line, UTF-8):
  request:  {"version": 1, "tick": int, "observation": [73 floats], "metadata": {...}}
  response: {"version": 1, "tick": int, "action": [9 ints] | null,
             "inference_ms": float | null, "error": null | {"code": str, "message": str}}

Validation mirrors the training contract exactly:
  rl/environment/obs_schema.py  -> OBS_DIMS[4] == 73, finite float32
  rl/environment/action_map.py  -> NVEC == [5,2,2,2,2,2,9,3,3]
"""
from __future__ import annotations

import json
import socket

PROTOCOL_VERSION = 1

# Traceable to rl/environment/action_map.py: NVEC (imported, not copied,
# in action_validator.py; repeated here only for the error-message table).
NVEC = [5, 2, 2, 2, 2, 2, 9, 3, 3]
OBS_DIM = 73


def encode_request(tick: int, observation: list, metadata: dict | None = None) -> bytes:
    return (json.dumps({"version": PROTOCOL_VERSION, "tick": int(tick),
                        "observation": [float(x) for x in observation],
                        "metadata": dict(metadata or {})}) + "\n").encode("utf-8")


def decode_request(raw: bytes | str) -> tuple[bool, dict | str]:
    try:
        msg = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (ValueError, UnicodeDecodeError) as e:
        return False, f"bad_json: {e}"
    if not isinstance(msg, dict):
        return False, "bad_json: top-level must be an object"
    if msg.get("version") != PROTOCOL_VERSION:
        return False, f"bad_version: {msg.get('version')!r}"
    if not isinstance(msg.get("tick"), int):
        return False, "bad_tick: must be int"
    obs = msg.get("observation")
    if not isinstance(obs, list) or len(obs) != OBS_DIM:
        n = len(obs) if isinstance(obs, list) else type(obs).__name__
        return False, f"bad_observation: need list of {OBS_DIM}, got {n}"
    import math
    for x in obs:
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return False, "bad_observation: all values must be finite numbers"
    return True, {"tick": msg["tick"], "observation": [float(x) for x in obs],
                  "metadata": msg.get("metadata") or {}}


def encode_response(tick: int, action: list | None,
                    inference_ms: float | None = None,
                    error: dict | None = None) -> bytes:
    return (json.dumps({"version": PROTOCOL_VERSION, "tick": int(tick),
                        "action": ([int(x) for x in action] if action is not None
                                   else None),
                        "inference_ms": inference_ms, "error": error}) + "\n").encode("utf-8")


def encode_error(tick: int, code: str, message: str) -> bytes:
    return encode_response(tick, None, None, {"code": code, "message": str(message)})


def decode_response(raw: bytes | str) -> tuple[bool, dict | str]:
    try:
        msg = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (ValueError, UnicodeDecodeError) as e:
        return False, f"bad_json: {e}"
    if not isinstance(msg, dict) or msg.get("version") != PROTOCOL_VERSION:
        return False, "bad_version_or_shape"
    return True, msg


def request_action(host: str, port: int, tick: int, observation: list,
                   metadata: dict | None = None, timeout: float = 0.5) -> dict | None:
    """One-shot client used by tests and tooling. Returns the response dict,
    or None on any transport failure (mirrors the mod's safe-fallback rule:
    network trouble must never raise into game logic)."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(encode_request(tick, observation, metadata))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    return None
                buf += chunk
        ok, msg = decode_response(buf)
        return msg if ok else None
    except (OSError, ValueError):
        return None
