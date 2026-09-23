"""Deployment bridge tests: hash, load, shapes, protocol, server, determinism.

Does not touch training code paths (read-only PPO.load of the canonical
checkpoint). No Minecraft needed: server smoke runs on an ephemeral localhost
port; the Java encoder is cross-checked via the authoritative Python schema.
"""
from __future__ import annotations

import json
import socket
import threading

import numpy as np

from deployment import protocol
from deployment.action_validator import validate_action
from deployment.model_runner import CANONICAL_SHA256, ModelRunner, sha256_file
from deployment.observation_validator import (
    EXPECTED_DIM,
    describe_normalization,
    validate_observation,
)
from rl.environment import action_map, obs_schema
from sandbox.simulation.sim import PvPSim

CANON = "checkpoints/phase4_stage3/3a2/final.zip"


def _sim_obs(seed: int = 7) -> np.ndarray:
    sim = PvPSim(seed=seed)
    sim.reset(seed=seed)
    return obs_schema.build_obs(sim, agent=0, version=4)


def test_canonical_hash():
    assert sha256_file(CANON) == CANONICAL_SHA256


def test_model_load_spaces():
    m = ModelRunner()
    assert m.obs_dim == 73 == EXPECTED_DIM == obs_schema.OBS_DIMS[4]
    assert m.nvec == [5, 2, 2, 2, 2, 2, 9, 3, 3] == list(action_map.NVEC)


def test_no_vecnormalize():
    d = describe_normalization()
    assert d["vecnormalize"] is False and d["vecnormalize_stats_file"] is None


def test_synthetic_obs_inference_legal():
    m = ModelRunner()
    act, ms = m.predict(_sim_obs())
    assert len(act) == 9
    ok, _ = validate_action(act)
    assert ok
    assert ms < 50.0  # generous CI bound; real target <5ms (see compat report)


def test_deterministic_repeat():
    m = ModelRunner()
    obs = _sim_obs()
    assert m.predict(obs)[0] == m.predict(obs)[0]


def test_observation_validation():
    assert validate_observation(_sim_obs())[0] is True
    assert validate_observation([0.0] * 72)[0] is False
    bad = [0.0] * 73
    bad[3] = float("nan")
    assert validate_observation(bad)[0] is False
    bad[3] = float("inf")
    assert validate_observation(bad)[0] is False


def test_action_validation_ranges():
    assert validate_action([1, 1, 0, 1, 0, 0, 8, 1, 1])[0] is True
    assert validate_action([5, 0, 0, 0, 0, 0, 0, 1, 1])[0] is False
    assert validate_action([0] * 8)[0] is False
    assert validate_action([0, 0, 0, 0, 0, 0, 9, 1, 1])[0] is False


def test_protocol_roundtrip():
    obs = [float(x) for x in _sim_obs()]
    raw = protocol.encode_request(123, obs, {"arena": "test"})
    ok, msg = protocol.decode_request(raw)
    assert ok and msg["tick"] == 123 and len(msg["observation"]) == 73
    raw = protocol.encode_response(123, [1, 0, 0, 1, 0, 0, 0, 1, 1], 0.8)
    ok, msg = protocol.decode_response(raw)
    assert ok and msg["action"] == [1, 0, 0, 1, 0, 0, 0, 1, 1]


def test_malformed_packets_never_raise():
    assert protocol.decode_request(b"not json")[0] is False
    assert protocol.decode_request(
        json.dumps({"version": 999, "tick": 1,
                    "observation": [0.0] * 73}).encode())[0] is False
    assert protocol.decode_request(
        json.dumps({"version": 1, "tick": 1,
                    "observation": [0.0] * 10}).encode())[0] is False
    err = protocol.encode_error(5, "invalid_observation", "need 73")
    ok, msg = protocol.decode_response(err)
    assert ok and msg["action"] is None and msg["error"]["code"] == "invalid_observation"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_smoke_and_timeout_fallback():
    import socketserver
    from deployment import inference_server
    inference_server.RUNNER = ModelRunner()
    port = _free_port()
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port),
                                          inference_server.Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        obs = [float(x) for x in _sim_obs()]
        resp = protocol.request_action("127.0.0.1", port, 42, obs, timeout=5.0)
        assert resp is not None and resp["tick"] == 42
        assert validate_action(resp["action"])[0] is True
        # malformed -> safe error response, server stays up
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b"garbage\n")
            got = s.recv(4096)
        ok, msg = protocol.decode_response(got)
        assert ok and msg["action"] is None
        # server still serving after the bad packet
        resp = protocol.request_action("127.0.0.1", port, 43, obs, timeout=5.0)
        assert resp is not None and resp["tick"] == 43
    finally:
        srv.shutdown()
        srv.server_close()
    # dead port -> None (safe fallback), never raises
    assert protocol.request_action("127.0.0.1", port, 1, [0.0] * 73,
                                   timeout=0.2) is None
