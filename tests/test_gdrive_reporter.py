"""Google Drive reporter tests. No real Drive API calls; all network/auth mocked."""
from __future__ import annotations

import json
import os
import sys
import time
import types


def _fresh(monkeypatch, tmp_path):
    import tools.google_drive_reporter as gdr
    monkeypatch.setattr(gdr, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(gdr, "CONFIG_PATH", str(tmp_path / "gd.json"))
    monkeypatch.setattr(gdr, "STATE_PATH", str(tmp_path / "st.json"))
    monkeypatch.setattr(gdr, "REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(gdr, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(gdr, "LOCK_PATH", str(tmp_path / "lock"))
    monkeypatch.setattr(gdr, "LOG_PATH", str(tmp_path / "r.log"))
    monkeypatch.setattr(gdr, "_log", None)
    # stub the resumable-upload helper so no google deps are needed
    fake_http = types.ModuleType("googleapiclient.http")

    class _Media:
        def __init__(self, *a, **k):
            pass

    fake_http.MediaFileUpload = _Media
    monkeypatch.setitem(sys.modules, "googleapiclient.http", fake_http)
    return gdr


def test_report_with_missing_manifest_is_nulls_not_crash(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    rep = gdr.collect_report()
    assert rep["runtime"]["elapsed_hours"] is None
    assert rep["latest_regression"] is None
    assert rep["checkpoints"]["best_overall"] is None
    assert rep["system"]["controller"] in ("UNKNOWN", "STOPPED")
    assert set(rep["skills"]) >= {"melee", "bow", "crossbow"}


def test_local_reports_write_json_html_and_latest(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    jp, hp = gdr.write_local_reports(gdr.collect_report())
    assert os.path.exists(jp) and os.path.exists(hp)
    assert os.path.exists(os.path.join(str(tmp_path / "reports"), "latest.json"))
    assert os.path.exists(os.path.join(str(tmp_path / "reports"), "latest.html"))
    assert json.load(open(jp))["generated_utc"]


def test_config_defaults_and_roundtrip(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    assert gdr.load_config()["interval_hours"] == 12
    gdr.save_config({**gdr.load_config(), "reports_folder_id": "abc"})
    assert gdr.load_config()["reports_folder_id"] == "abc"
    assert "client_secret" not in json.dumps(gdr.load_config())


def test_missing_token_gives_none_not_crash(monkeypatch, tmp_path):
    import tools.google_drive_auth as gda
    monkeypatch.setattr(gda, "TOKEN_PATH", str(tmp_path / "nope.json"))
    assert gda.load_credentials() is None
    assert gda.auth_status()["configured"] is False


def test_token_refresh_path(monkeypatch, tmp_path):
    import tools.google_drive_auth as gda
    fake_google = types.ModuleType("google")
    fake_oauth = types.ModuleType("google.oauth2.credentials")
    seen = {}

    class FakeCreds:
        valid = True
        expired = True
        refresh_token = "r"

        @classmethod
        def from_authorized_user_info(cls, data, scopes):
            return cls()

        def refresh(self, req):
            seen["refreshed"] = True

        def to_json(self):
            return "{}"

    fake_oauth.Credentials = FakeCreds
    fake_req = types.ModuleType("google.auth.transport.requests")

    class _Req:
        pass

    fake_req.Request = _Req
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", fake_oauth)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", fake_req)
    tp = tmp_path / "tok.json"
    tp.write_text("{}")
    monkeypatch.setattr(gda, "TOKEN_PATH", str(tp))
    assert gda.load_credentials() is not None
    assert seen.get("refreshed") is True


class _FailTwice:
    def __init__(self):
        self.n = 0

    def files(self):
        return self

    def create(self, **kw):
        return self

    def execute(self):
        self.n += 1
        if self.n < 3:
            raise TimeoutError("boom")
        return {"id": "fid"}


def test_upload_retries_then_succeeds(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    f = tmp_path / "r.json"
    f.write_text("{}")
    ok, fid = gdr.upload_file(_FailTwice(), "folder", str(f))
    assert ok is True and fid == "fid"


def test_drive_unavailable_never_raises(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    f = tmp_path / "r.json"
    f.write_text("{}")

    class _Dead:
        def files(self):
            raise ConnectionError("down")

    assert gdr.upload_file(_Dead(), "folder", str(f), retries=2) == (False, "ConnectionError")
    assert gdr.ensure_folders(_Dead(), dict(gdr.DEFAULT_CONFIG)) is False


def test_scheduler_due_and_catchup_single(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    cfg = {"enabled": True, "interval_hours": 12}
    now = time.time()
    st = gdr.load_state()
    assert gdr.is_due(st, cfg, now=now) is True  # first run
    # overdue by 5 intervals still yields a single due=True (one catch-up)
    old = datetime_iso(now - 5 * 12 * 3600)
    st["last_successful_report_utc"] = old
    st["last_attempt_utc"] = datetime_iso(now - 5 * 12 * 3600)
    assert gdr.is_due(st, cfg, now=now) is True
    st["last_successful_report_utc"] = datetime_iso(now - 3600)
    st["last_attempt_utc"] = datetime_iso(now - 3600)
    assert gdr.is_due(st, cfg, now=now) is False
    assert gdr.is_due({**st, "last_attempt_utc": datetime_iso(now)},
                      cfg, now=now) is False  # race guard
    assert gdr.is_due(st, {**cfg, "enabled": False}, now=now) is False


def datetime_iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def test_duplicate_event_prevention(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    st = gdr.load_state()
    st["sent_event_ids"] = ["skill_proven:bow:abc"]
    gdr.save_state(st)
    assert gdr.send_event("skill_proven:bow:abc", "x") is True
    # reload proves restart persistence
    assert "skill_proven:bow:abc" in gdr.load_state()["sent_event_ids"]


def test_report_once_without_auth_still_writes_local(monkeypatch, tmp_path):
    gdr = _fresh(monkeypatch, tmp_path)
    import tools.google_drive_auth as gda
    monkeypatch.setattr(gda, "get_drive_service", lambda: None)
    assert gdr.report_once(upload=True) is False  # upload skipped, no crash
    assert gdr.load_state()["last_report_file"] is not None
    assert gdr.load_state()["consecutive_failures"] == 1
