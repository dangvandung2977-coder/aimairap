"""Localhost PPO inference server. One observation in, one action out.

  python deployment/inference_server.py [--config deployment/config.yaml]
                                        [--port 25575] [--checkpoint ...]

Binds 127.0.0.1 by default (never expose publicly). Malformed packets get a
safe error response; the server never crashes on bad input.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow both `python -m deployment.inference_server` and
# `python deployment/inference_server.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socketserver
import time

from deployment import protocol
from deployment.action_validator import validate_action
from deployment.model_runner import CANONICAL_PATH, CANONICAL_SHA256, ModelRunner
from deployment.observation_validator import validate_observation

RUNNER: ModelRunner | None = None


class Handler(socketserver.StreamRequestHandler):
    timeout = 5.0

    def handle(self) -> None:
        assert RUNNER is not None
        t_req = time.perf_counter()
        try:
            raw = self.rfile.readline(1 << 20)
        except (OSError, ValueError):
            return
        if not raw:
            return
        ok, msg = protocol.decode_request(raw)
        if not ok:
            self._send(protocol.encode_error(-1, "invalid_request", str(msg)))
            return
        tick, obs = msg["tick"], msg["observation"]
        vok, _ = validate_observation(obs)
        if not vok:
            self._send(protocol.encode_error(tick, "invalid_observation",
                                             "need 73 finite numbers"))
            return
        try:
            act, ms = RUNNER.predict(obs)
        except (ValueError, RuntimeError) as e:
            self._send(protocol.encode_error(tick, "inference_failed", str(e)))
            return
        aok, _ = validate_action(act)
        if not aok:  # belt and braces; predict() already validated
            self._send(protocol.encode_error(tick, "invalid_action", str(act)))
            return
        total_ms = (time.perf_counter() - t_req) * 1000.0
        import json as _json
        body = {"version": protocol.PROTOCOL_VERSION, "tick": tick,
                "action": [int(x) for x in act],
                "inference_ms": round(ms, 3),
                "bridge_ms": round(total_ms - ms, 3), "error": None}
        self._send((_json.dumps(body) + "\n").encode())

    def _send(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except OSError:
            pass


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    global RUNNER
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="deployment/config.yaml")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    host = a.host or cfg.get("host", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"refusing non-localhost bind: {host}")
    port = a.port or int(cfg.get("port", 25575))
    ckpt = a.checkpoint or cfg.get("checkpoint", CANONICAL_PATH)
    RUNNER = ModelRunner(ckpt, cfg.get("expected_sha256", CANONICAL_SHA256))
    print(f"[inference] serving {ckpt} on {host}:{port} "
          f"(sha {RUNNER.actual_sha256[:12]}…, obs {RUNNER.obs_dim})", flush=True)
    with socketserver.ThreadingTCPServer((host, port), Handler,
                                         bind_and_activate=False) as srv:
        srv.allow_reuse_address = True
        srv.daemon_threads = True
        srv.server_bind()
        srv.server_activate()
        srv.serve_forever()


if __name__ == "__main__":
    main()
