"""12-hour Google Drive training reporter (monitoring ONLY, never PPO-critical).

Reads the EXISTING training state (manifest.json, logs/, checkpoints/) and
uploads a JSON + HTML report to Drive. Training continues whether Drive
succeeds or fails; this module never raises into the training loop and never
exits nonzero in periodic mode.

CLI:
  python -m tools.google_drive_reporter --local-only    # dry run, no upload
  python -m tools.google_drive_reporter --report-once   # one report + upload
  python -m tools.google_drive_reporter --check         # upload only if 12h due
  python -m tools.google_drive_reporter --event <id> --summary <text>
"""
from __future__ import annotations

import glob
import hashlib
import html
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "google_drive.json")
STATE_PATH = os.path.join(PROJECT_ROOT, "state", "google_drive_report_state.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_7day",
                             "manifest.json")
LOCK_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_7day",
                         "controller.lock")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "google_drive_reporter.log")
ROOT_FOLDER_NAME = "PVP-RL Training"

DEFAULT_CONFIG = {"enabled": True, "interval_hours": 12,
                  "root_folder_id": None, "reports_folder_id": None,
                  "events_folder_id": None,
                  "root_folder_name": ROOT_FOLDER_NAME}

_log = None


def logger() -> logging.Logger:
    global _log
    if _log is None:
        _log = logging.getLogger("gdrive_reporter")
        _log.setLevel(logging.INFO)
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        h = logging.FileHandler(LOG_PATH)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(message)s"))
        _log.addHandler(h)
    return _log


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- config / state (atomic writes, no secrets inside) -----------------------
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def load_state() -> dict:
    st: dict = {"last_successful_report_utc": None, "last_attempt_utc": None,
                "last_report_file": None, "consecutive_failures": 0,
                "sent_event_ids": []}
    try:
        with open(STATE_PATH) as f:
            st.update(json.load(f))
    except (OSError, ValueError):
        pass
    return st


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_PATH)


def _parse_utc(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def is_due(state: dict, config: dict, now: float | None = None) -> bool:
    """True if a 12h report is overdue. One catch-up max: callers generate a
    single report per invocation, then resume the normal schedule."""
    if not config.get("enabled", True):
        return False
    # avoid tight duplicate loops (timer + controller hook racing)
    last_attempt = _parse_utc(state.get("last_attempt_utc"))
    now = time.time() if now is None else now
    if last_attempt and now - last_attempt < 600:
        return False
    last_ok = _parse_utc(state.get("last_successful_report_utc"))
    if last_ok is None:
        return True  # first run ever
    return (now - last_ok) >= float(config.get("interval_hours", 12)) * 3600.0


# --- report collection (authoritative sources only; null when missing) -------
def _parse_ctime(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return time.mktime(time.strptime(s, "%a %b %d %H:%M:%S %Y"))
    except (ValueError, TypeError):
        return None


def _file_info(path: str | None) -> dict | None:
    if not path:
        return None
    ap = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(ap):
        return None
    try:
        h = hashlib.sha256()
        with open(ap, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        st = os.stat(ap)
        return {"path": path, "bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(
                    st.st_mtime, timezone.utc).isoformat(),
                "sha256": h.hexdigest()}
    except OSError:
        return None


def _latest_json(pattern: str) -> dict | None:
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)),
                   key=os.path.getmtime)
    if not files:
        return None
    try:
        with open(files[-1]) as f:
            return {"file": os.path.relpath(files[-1], PROJECT_ROOT),
                    "data": json.load(f)}
    except (OSError, ValueError):
        return None


def _mem_info() -> dict:
    try:
        with open("/proc/meminfo") as f:
            kv = {}
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[0].endswith(":"):
                    kv[p[0][:-1]] = int(p[1])
        return {"total_mb": round(kv.get("MemTotal", 0) / 1024.0, 1),
                "available_mb": round(kv.get("MemAvailable", 0) / 1024.0, 1)}
    except OSError:
        return {"total_mb": None, "available_mb": None}


def _controller_status(manifest: dict) -> str:
    if manifest.get("ended") and not any(
            v.get("status") == "running" for v in manifest.get("legs", {}).values()):
        return "COMPLETE"
    try:
        pid = int(open(LOCK_PATH).read().strip())
        os.kill(pid, 0)
        return "RUNNING"
    except (OSError, ValueError):
        pass
    if manifest.get("legs"):
        return "STOPPED"
    return "UNKNOWN"


def _skill_labels(manifest: dict) -> dict:
    """Labels come from the project's own gate/manifest status. Base skills are
    PROVEN by the validated 3A2 baseline; bow reflects its gate verdict."""
    skills: dict = {}
    base = manifest.get("baseline_retention", {})
    for s in ("melee", "crit", "strafe", "symmetry", "shield", "anti_shield",
              "food", "golden_apple", "sustain", "pearl_kite_cutoff"):
        skills[s] = "PROVEN" if base else "UNKNOWN"
    bow_verdict, bow_running = None, False
    for leg in manifest.get("legs", {}).values():
        if leg.get("skill") == "bow":
            if leg.get("status") == "running":
                bow_running = True
            v = (leg.get("gate") or {}).get("verdict")
            if v:
                bow_verdict = v
    if bow_running and bow_verdict != "PASS":
        skills["bow"] = "TRAINING"
    elif bow_verdict == "PASS":
        skills["bow"] = "PROVEN"
    elif bow_verdict:
        skills["bow"] = "NOT_PROVEN"
    else:
        skills["bow"] = "UNKNOWN"
    for extra in ("crossbow", "potions", "water_lava"):
        skills[extra] = "BLOCKED"  # blocked by hotbar/obs invariant, see audit
    return skills


def collect_report() -> dict:
    manifest: dict = {}
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        pass
    legs = manifest.get("legs", {})
    running = [k for k, v in legs.items() if v.get("status") == "running"]
    done = [k for k, v in legs.items() if v.get("status") == "done"]
    now = time.time()
    started = _parse_ctime(manifest.get("started"))
    deadline = manifest.get("deadline")
    reg = _latest_json("logs/reg_*.json")
    gate = _latest_json("logs/gate_*.json")
    best = manifest.get("best_overall")
    events = []
    for name in list(legs)[-10:]:
        v = legs[name]
        if v.get("status") == "done" and v.get("promoted"):
            events.append({"type": "checkpoint_promoted", "leg": name,
                           "checkpoint": v.get("promoted"),
                           "score": v.get("score")})
        if v.get("retention_ok") is False:
            events.append({"type": "skill_regressed", "leg": name,
                           "retention": v.get("retention")})
        if v.get("status") in ("error", "interrupted"):
            events.append({"type": "leg_issue", "leg": name,
                           "status": v.get("status")})
    du = shutil.disk_usage(PROJECT_ROOT)
    return {
        "generated_utc": utcnow(),
        "runtime": {
            "elapsed_hours": round((now - started) / 3600.0, 2)
            if started else None,
            "remaining_hours": round((deadline - now) / 3600.0, 2)
            if isinstance(deadline, (int, float)) else None,
            "current_leg": running[-1] if running else (done[-1] if done else None),
            "legs_done": len(done), "legs_running": len(running),
            "planned_steps_current_leg": (legs[running[-1]].get("timesteps")
                                          if running else None),
            "planned_steps_completed": sum(v.get("timesteps", 0)
                                           for v in legs.values()
                                           if v.get("status") == "done"),
            "throughput_steps_per_s": None,  # parsed opportunistically below
            "deadline_epoch": deadline,
        },
        "skills": _skill_labels(manifest),
        "latest_regression": reg["data"] if reg else None,
        "latest_gate": gate["data"] if gate else None,
        "checkpoints": {
            "canonical": _file_info(manifest.get("canonical")
                                    or "checkpoints/phase4_stage3/3a2/final.zip"),
            "best_overall": _file_info(best),
        },
        "system": {
            "ram": _mem_info(),
            "disk_free_mb": round(du.free / (1024.0 * 1024.0), 1),
            "controller": _controller_status(manifest),
        },
        "recent_events": events,
        "manifest_best": {"best_overall": best,
                          "best_score": manifest.get("best_score"),
                          "best_leg": manifest.get("best_leg")},
    }


# --- rendering ---------------------------------------------------------------
def _esc(v) -> str:
    return html.escape("unavailable" if v is None else str(v))


def render_html(rep: dict) -> str:
    rt, sys_ = rep["runtime"], rep["system"]
    rows = lambda d: "".join(f"<tr><td>{html.escape(k)}</td><td>{_esc(v)}</td></tr>"
                             for k, v in d.items())
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>PVP-RL Training Report</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
h2{{margin-top:1.5em;border-bottom:1px solid #ddd}}</style></head><body>
<h1>PVP-RL Training Report</h1><p>Generated {_esc(rep['generated_utc'])}</p>
<h2>Runtime</h2><table>{rows(rt)}</table>
<h2>Skill Status</h2><table>{rows(rep['skills'])}</table>
<h2>Checkpoints</h2><table>{rows({k: (v or {}).get('path') for k, v in rep['checkpoints'].items()})}</table>
<h2>System</h2><table>{rows({'ram_total_mb': sys_['ram'].get('total_mb'), 'ram_avail_mb': sys_['ram'].get('available_mb'), 'disk_free_mb': sys_.get('disk_free_mb'), 'controller': sys_.get('controller')})}</table>
<h2>Recent Events</h2><pre>{_esc(json.dumps(rep['recent_events'], indent=1)[:4000])}</pre>
<h2>Latest Regression</h2><pre>{_esc(json.dumps(rep['latest_regression'], indent=1)[:4000])}</pre>
<h2>Latest Gate</h2><pre>{_esc(json.dumps(rep['latest_gate'], indent=1)[:3000])}</pre>
</body></html>"""


def write_local_reports(rep: dict) -> tuple[str, str]:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    jp = os.path.join(REPORTS_DIR, f"{stamp}.json")
    hp = os.path.join(REPORTS_DIR, f"{stamp}.html")
    with open(jp, "w") as f:
        json.dump(rep, f, indent=2)
    with open(hp, "w") as f:
        f.write(render_html(rep))
    for name, src in (("latest.json", jp), ("latest.html", hp)):
        dst = os.path.join(REPORTS_DIR, name)
        try:
            shutil.copyfile(src, dst)
        except OSError:
            pass
    return jp, hp


# --- Drive IO (all failures -> (False, reason); never raises) ----------------
def _find_or_create_folder(service, name: str, parent_id: str | None) -> tuple[bool, str]:
    try:
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        res = service.files().list(q=q, fields="files(id,name)",
                                   pageSize=10).execute()
        files = res.get("files", [])
        if files:
            return True, files[0]["id"]
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            meta["parents"] = [parent_id]
        created = service.files().create(body=meta, fields="id").execute()
        return True, created["id"]
    except Exception as e:
        logger().warning("Drive folder %r unavailable; training continues (%s)",
                         name, type(e).__name__)
        return False, ""


def ensure_folders(service, cfg: dict) -> bool:
    ok1, root = _find_or_create_folder(service, cfg.get("root_folder_name",
                                                        ROOT_FOLDER_NAME), None)
    if not ok1:
        return False
    ok2, rep = _find_or_create_folder(service, "reports", root)
    ok3, ev = _find_or_create_folder(service, "events", root)
    if not (ok2 and ok3):
        return False
    cfg.update({"root_folder_id": root, "reports_folder_id": rep,
                "events_folder_id": ev})
    save_config(cfg)
    return True


def upload_file(service, folder_id: str, local_path: str,
                retries: int = 3) -> tuple[bool, str]:
    """Upload with exponential backoff. Returns (ok, file_id_or_reason)."""
    import time as _t
    last = "unknown"
    for attempt in range(retries):
        try:
            from googleapiclient.http import MediaFileUpload
            name = os.path.basename(local_path)
            media = MediaFileUpload(local_path, resumable=False)
            created = service.files().create(
                body={"name": name, "parents": [folder_id]},
                media_body=media, fields="id").execute()
            return True, created.get("id", "")
        except Exception as e:
            last = type(e).__name__
            logger().warning("Drive upload attempt %d/%d failed (%s); "
                             "training continues", attempt + 1, retries, last)
            _t.sleep(2 ** attempt)
    return False, last


def setup_folders_and_test_upload(service) -> bool:
    cfg = load_config()
    if not ensure_folders(service, cfg):
        return False
    rep = collect_report()
    jp, hp = write_local_reports(rep)
    ok, _ = upload_file(service, cfg["reports_folder_id"], jp)
    if ok:
        upload_file(service, cfg["reports_folder_id"], hp)
    st = load_state()
    st.update({"last_attempt_utc": utcnow(),
               "last_report_file": jp,
               "last_successful_report_utc": utcnow() if ok else
               st.get("last_successful_report_utc"),
               "consecutive_failures": 0 if ok else
               st.get("consecutive_failures", 0) + 1})
    save_state(st)
    return ok


def send_event(event_id: str, summary: str, payload: dict | None = None) -> bool:
    """Deduplicated event upload into the events/ folder. Never raises."""
    st = load_state()
    if event_id in st.get("sent_event_ids", []):
        return True
    try:
        from tools import google_drive_auth as gda
        service = gda.get_drive_service()
        if service is None:
            logger().warning("Event %s skipped (Drive not authorized); "
                             "training continues", event_id)
            return False
        cfg = load_config()
        if not cfg.get("events_folder_id") and not ensure_folders(service, cfg):
            return False
        tmp = os.path.join(REPORTS_DIR, f"event_{int(time.time())}.json")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({"event_id": event_id, "summary": summary,
                       "utc": utcnow(), "payload": payload or {}}, f, indent=2)
        ok, _ = upload_file(service, cfg["events_folder_id"], tmp)
        if ok:
            st["sent_event_ids"] = (st.get("sent_event_ids", []) + [event_id])[-200:]
            save_state(st)
        return ok
    except Exception as e:
        logger().warning("Event %s failed (%s); training continues",
                         event_id, type(e).__name__)
        return False


def report_once(upload: bool = True) -> bool:
    """Generate local report; upload unless upload=False or no auth. Never raises."""
    st = load_state()
    st["last_attempt_utc"] = utcnow()
    save_state(st)
    try:
        rep = collect_report()
        jp, hp = write_local_reports(rep)
        logger().info("Generated report %s", jp)
        if not upload:
            st.update({"last_report_file": jp})
            save_state(st)
            return True
        from tools import google_drive_auth as gda
        service = gda.get_drive_service()
        if service is None:
            from tools.google_drive_auth import SETUP_GUIDE
            logger().warning("Drive not authorized; training continues. "
                             "One-time setup needed.")
            print(SETUP_GUIDE)
            print("REPORT_STATUS: local_only_not_authorized")
            st.update({"last_report_file": jp,
                       "consecutive_failures": st.get("consecutive_failures", 0) + 1})
            save_state(st)
            return False
        cfg = load_config()
        if not cfg.get("reports_folder_id") and not ensure_folders(service, cfg):
            st["consecutive_failures"] = st.get("consecutive_failures", 0) + 1
            save_state(st)
            return False
        ok1, _ = upload_file(service, cfg["reports_folder_id"], jp)
        ok2, _ = upload_file(service, cfg["reports_folder_id"], hp)
        ok = ok1 and ok2
        st.update({"last_report_file": jp,
                   "last_successful_report_utc": utcnow() if ok else
                   st.get("last_successful_report_utc"),
                   "consecutive_failures": 0 if ok else
                   st.get("consecutive_failures", 0) + 1})
        save_state(st)
        logger().info("Upload successful" if ok else
                      "Drive unavailable; training continues")
        print("REPORT_STATUS:", "uploaded" if ok else "upload_failed_training_continues")
        return ok
    except Exception as e:
        logger().warning("Reporter failed (%s); training continues",
                         type(e).__name__)
        try:
            st["consecutive_failures"] = st.get("consecutive_failures", 0) + 1
            save_state(st)
        except Exception:
            pass
        return False


def maybe_report() -> bool | None:
    """Periodic entrypoint for the controller hook and the timer.

    Returns True (reported), False (attempt failed, training continues) or
    None (not due / disabled). Never raises, never exits."""
    try:
        cfg = load_config()
        st = load_state()
        if not is_due(st, cfg):
            return None
        return report_once(upload=True)
    except Exception:
        return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--report-once", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--event", default=None)
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    if a.event:
        print("EVENT_STATUS:", "sent" if send_event(a.event, a.summary) else "failed_or_skipped")
        return
    if a.local_only:
        print("REPORT_STATUS: local_only", write_local_reports(collect_report())[0])
        return
    if a.check:
        r = maybe_report()
        print("REPORT_STATUS:", "not_due" if r is None else
              ("uploaded" if r else "upload_failed_training_continues"))
        return
    report_once(upload=not a.local_only)


if __name__ == "__main__":
    main()
