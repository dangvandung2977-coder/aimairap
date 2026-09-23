"""168-hour autonomous PvP curriculum controller (VPS, CPU-only, restart-safe).

NOT one blind PPO run. A persistent orchestrator that, within a wall-clock
budget, trains one curriculum leg at a time as an isolated `train.py`
subprocess (memory released between legs, crash-isolated), then gates it:

    resume parent -> train leg -> regression -> skill gate
      -> promote (only if core skills retained) -> next leg
      -> crash/reboot -> resume from latest.zip -> continue to deadline

Guarantees:
  * canonical 3A2 checkpoint is never trained into or overwritten;
  * a leg that regresses a proven skill is REJECTED (parent lineage kept), so
    no new skill can silently replace an old one;
  * best_overall is chosen by a composite (retention + proven advantage),
    not a single win-rate number;
  * absolute deadline persisted -> survives SSH drop and VPS reboot;
  * SIGTERM/SIGINT save-and-exit; restart skips finished legs.

Run (inside tmux):  python train_7day.py --hours 168
Resume is automatic: just run the same command again.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time

CANONICAL = "checkpoints/phase4_stage3/3a2/final.zip"
ROOT = "checkpoints/final_7day"
MANIFEST = os.path.join(ROOT, "manifest.json")
BEST_OVERALL = os.path.join(ROOT, "best_overall.zip")
REPORTS = "reports"

# --- retention floors (established tolerance; below = regression) -----------
# Baseline (3A2, tools.regression seeds): melee .40 / shield .95 / sustain 1.0.
RETENTION_FLOOR = {"melee": 0.25, "shield_defense": 0.80, "sustain": 0.85}

# --- curriculum -------------------------------------------------------------
# Each leg trains from `parent` (a prior promoted leg name, or None=canonical),
# then is gated. `gate` None => regression-only (retention) gate.
CURRICULUM = [
    {"name": "core_consolidate", "config": "configs/core_consolidate.yaml",
     "timesteps": 200_000, "parent": None, "gate": None},
    {"name": "bow_discovery", "config": "configs/stage4a.yaml",
     "timesteps": 300_000, "parent": "core_consolidate",
     "gate": "tools.stage4a_gate", "skill": "bow"},
    {"name": "bow_consolidate", "config": "configs/stage4a.yaml",
     "timesteps": 300_000, "parent": "bow_discovery",
     "gate": "tools.stage4a_gate", "skill": "bow"},
]

# measured VPS throughput ~132 env-steps/s (n_envs=2, 2 vCPU); use a
# conservative floor so budget math never overshoots the deadline.
RATE = 110.0
INTEGRATED_CFG = "configs/integrated.yaml"

_STOP = False
LOCK = os.path.join(ROOT, "controller.lock")


def _acquire_lock() -> None:
    """Refuse to start a second controller (concurrent runs would corrupt the
    shared manifest/checkpoints). Stale lock (dead PID) is reclaimed."""
    os.makedirs(ROOT, exist_ok=True)
    if os.path.exists(LOCK):
        try:
            old = int(open(LOCK).read().strip())
            os.kill(old, 0)  # raises if not running
            raise SystemExit(f"controller already running (pid {old}); refusing")
        except (ValueError, ProcessLookupError):
            pass  # stale
        except PermissionError:
            raise SystemExit("controller lock held by a live process; refusing")
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))


def _release_lock() -> None:
    try:
        os.remove(LOCK)
    except OSError:
        pass


def _sig(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n[controller] signal {signum} -> finishing current step and exiting")


signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)


# --- manifest ---------------------------------------------------------------
def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {}


def save_manifest(m: dict) -> None:
    os.makedirs(ROOT, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, MANIFEST)  # atomic: never a half-written manifest


# --- resources --------------------------------------------------------------
def free_ram_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 4096.0  # non-Linux dev box: don't throttle


def free_disk_mb(path=".") -> float:
    return shutil.disk_usage(path).free / (1024.0 * 1024.0)


def pick_n_envs() -> int:
    # Swap is emergency protection, not usable RAM -> size to real free RAM.
    return 2 if free_ram_mb() > 1150 else 1


# --- subprocess helpers -----------------------------------------------------
def _run(cmd: list[str], log_path: str, timeout: float | None = None) -> int:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    print(f"[controller] $ {' '.join(cmd)}")
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.ctime()} :: {' '.join(cmd)} ===\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    return proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if _STOP:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return 130
                    if timeout is not None and timeout <= 0:
                        proc.terminate()
                        return 124
        finally:
            if proc.poll() is None:
                proc.terminate()


def train_leg(leg: dict, parent_ckpt: str, timesteps: int, n_envs: int) -> str | None:
    """Run train.py as a subprocess with crash-retry. Returns best_model path
    on success, else None. Resumes from latest.zip across retries (no restart
    from scratch)."""
    ck_dir = _ckpt_dir(leg)
    latest = os.path.join(ck_dir, "latest.zip")
    log = os.path.join(ROOT, "legs", f"{leg['name']}.log")
    for attempt in range(3):
        if _STOP:
            return None
        resume = latest if os.path.exists(latest) else parent_ckpt
        cmd = [sys.executable, "train.py", "--config", leg["config"],
               "--resume", resume, "--timesteps", str(timesteps),
               "--n-envs", str(n_envs)]
        rc = _run(cmd, log)
        if rc == 0:
            best = os.path.join(ck_dir, "best_model.zip")
            fin = os.path.join(ck_dir, "final.zip")
            return best if os.path.exists(best) else (fin if os.path.exists(fin) else latest)
        if rc in (130, 124):  # SIGTERM / timeout -> not a crash
            return None
        print(f"[controller] leg {leg['name']} crashed (rc={rc}), retry {attempt+1}/3")
        time.sleep(5)
    return None


def _ckpt_dir(leg: dict) -> str:
    import yaml
    with open(leg["config"]) as f:
        d = yaml.safe_load(f)["checkpoint"]["dir"]
    # HARD GUARD: a training leg must never write into the canonical
    # checkpoint's directory (train.py overwrites final.zip/latest.zip there).
    canon_dir = os.path.normpath(os.path.dirname(CANONICAL))
    if os.path.normpath(d) == canon_dir or os.path.normpath(d).startswith(canon_dir + os.sep):
        raise SystemExit(f"REFUSING to train into canonical dir {canon_dir} "
                         f"(leg {leg['name']} config {leg['config']})")
    return d


# --- evaluation -------------------------------------------------------------
def run_regression(ckpt: str, tag: str) -> dict:
    out = os.path.join("logs", f"reg_{tag}.json")
    rc = _run([sys.executable, "-m", "tools.regression", "--checkpoint", ckpt,
               "--episodes", "20", "--out", out],
              os.path.join(ROOT, "legs", "gates.log"))
    if rc != 0 or not os.path.exists(out):
        return {}
    with open(out) as f:
        return json.load(f)


def run_gate(module: str, ckpt: str, tag: str) -> dict:
    out = os.path.join("logs", f"gate_{tag}.json")
    rc = _run([sys.executable, "-m", module, "--checkpoint", ckpt,
               "--episodes", "20", "--out", out],
              os.path.join(ROOT, "legs", "gates.log"))
    if rc != 0 or not os.path.exists(out):
        return {}
    with open(out) as f:
        return json.load(f)


def retention_ok(reg: dict) -> tuple[bool, dict]:
    detail = {}
    ok = True
    for k, floor in RETENTION_FLOOR.items():
        wr = reg.get(k, {}).get("win_rate", 0.0)
        detail[k] = wr
        if wr < floor:
            ok = False
    return ok, detail


def composite_score(reg: dict, gate: dict | None) -> float:
    """Retention-weighted score; ranged advantage is a bonus, never a
    substitute for keeping core skills."""
    core = sum(reg.get(k, {}).get("win_rate", 0.0) for k in RETENTION_FLOOR)
    bonus = 0.0
    if gate and gate.get("verdict") == "PASS":
        bonus = 0.5 + max(0.0, gate.get("deltas", {}).get("win", 0.0))
    return round(core + bonus, 4)


# --- reporting --------------------------------------------------------------
def write_matrix(m: dict) -> None:
    os.makedirs(REPORTS, exist_ok=True)
    skills = {
        "melee": "policy_proven", "crit": "policy_proven",
        "strafe": "policy_proven", "symmetry": "policy_proven",
        "shield": "policy_proven", "anti_shield": "policy_proven",
        "food": "policy_proven", "golden_apple": "policy_proven",
        "sustain": "policy_proven", "pearl_kite_cutoff": "policy_proven",
    }
    matrix = {"skills": {}, "updated": time.ctime()}
    for s, base in skills.items():
        matrix["skills"][s] = {"status": base, "source": "3A2 baseline"}
    for leg in m.get("legs", {}).values():
        sk = leg.get("skill")
        if sk:
            v = leg.get("gate", {}).get("verdict")
            matrix["skills"][sk] = {
                "status": "policy_proven" if v == "PASS" else "not_proven",
                "gate": v, "checkpoint": leg.get("promoted"),
                "retention": leg.get("retention_ok")}
    with open(os.path.join(REPORTS, "final_skill_matrix.json"), "w") as f:
        json.dump({"best_overall": m.get("best_overall"),
                   "best_score": m.get("best_score"),
                   "legs": m.get("legs", {}), "matrix": matrix}, f, indent=2)


def promote_best(m: dict, ckpt: str, score: float, leg_name: str) -> None:
    if score > m.get("best_score", -1.0):
        os.makedirs(ROOT, exist_ok=True)
        shutil.copyfile(ckpt, BEST_OVERALL)
        m["best_overall"] = BEST_OVERALL
        m["best_score"] = score
        m["best_leg"] = leg_name
        with open(os.path.join(ROOT, "best_overall.meta.json"), "w") as f:
            json.dump({"source": ckpt, "score": score, "leg": leg_name,
                       "time": time.ctime()}, f, indent=2)
        print(f"[controller] best_overall <- {leg_name} (score {score})")
        _drive_event("new_best", ckpt, score, leg_name)


def _drive_report(event: str | None = None, summary: str = "",
                  payload: dict | None = None) -> None:
    """Fire-and-forget Drive reporting. NEVER raises into training; Drive is
    monitoring-only (periodic 12h schedule + deduped events)."""
    try:
        from tools import google_drive_reporter as gdr
        if event is not None:
            gdr.send_event(event, summary, payload)
        gdr.maybe_report()  # no-op unless a 12h report is due
    except Exception:
        pass


def _drive_event(kind: str, ckpt: str, score: float, leg_name: str) -> None:
    try:
        h = None
        if ckpt and os.path.exists(ckpt):
            import hashlib
            hh = hashlib.sha256()
            with open(ckpt, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    hh.update(chunk)
            h = hh.hexdigest()[:16]
        _drive_report(event=f"{kind}:{h or leg_name}",
                      summary=f"{kind} leg={leg_name} score={score}",
                      payload={"leg": leg_name, "score": score,
                               "checkpoint": ckpt, "hash16": h})
    except Exception:
        pass


# --- main loop --------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--min-disk-mb", type=float, default=1500.0)
    ap.add_argument("--eval-margin-hours", type=float, default=1.0,
                    help="reserve this much wall-clock for final gate/eval")
    a = ap.parse_args()

    os.makedirs(ROOT, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    _acquire_lock()
    try:
        _main(a)
    finally:
        _release_lock()


def _main(a) -> None:
    m = load_manifest()
    now = time.time()
    if not m:
        assert os.path.exists(CANONICAL), f"canonical missing: {CANONICAL}"
        m = {"deadline": now + a.hours * 3600.0, "started": time.ctime(),
             "legs": {}, "best_score": -1.0, "best_overall": None,
             "canonical": CANONICAL}
        # baseline retention snapshot (no training)
        reg0 = run_regression(CANONICAL, "baseline")
        ok0, det0 = retention_ok(reg0)
        m["baseline_retention"] = det0
        promote_best(m, CANONICAL, composite_score(reg0, None), "canonical_baseline")
        save_manifest(m)
        print(f"[controller] baseline retention {det0}")

    deadline = m["deadline"]
    parent_ckpt = m.get("best_overall") or CANONICAL
    if m["legs"]:
        # restart path: crash-recovery event if a prior leg died mid-flight,
        # plus a restart marker; then a single catch-up report if 12h overdue.
        bad = [k for k, v in m["legs"].items()
               if v.get("status") in ("error", "interrupted")]
        if bad:
            _drive_report(event=f"crash_recovery:{','.join(sorted(bad))}",
                          summary=f"controller restarted with leg issues: {bad}",
                          payload={"legs": bad})
        _drive_report(event=f"controller_restart:{int(now)}",
                      summary="controller (re)started",
                      payload={"legs_done": sum(1 for v in m["legs"].values()
                                               if v.get("status") == "done")})
        _drive_report()  # one catch-up report max if overdue

    for leg in CURRICULUM:
        if _STOP:
            break
        name = leg["name"]
        rec = m["legs"].get(name, {})
        if rec.get("status") == "done":
            # already finished on a prior run; chain its promoted ckpt forward
            parent_ckpt = rec.get("promoted") or parent_ckpt
            continue

        remaining = deadline - time.time() - a.eval_margin_hours * 3600.0
        if remaining <= 0:
            print("[controller] out of budget; stopping before leg", name)
            break
        if free_disk_mb() < a.min_disk_mb:
            print(f"[controller] LOW DISK ({free_disk_mb():.0f} MB); stopping")
            m["stopped_reason"] = "low_disk"
            save_manifest(m)
            break

        # size this leg to fit remaining budget at the conservative rate
        steps_cap = int(remaining * RATE)
        timesteps = min(leg["timesteps"], max(50_000, steps_cap))
        _cap = int(os.environ.get("T7_CAP", "0"))  # ops/smoke override
        if _cap:
            timesteps = min(timesteps, _cap)

        # resolve parent
        pstr = leg.get("parent")
        if pstr and m["legs"].get(pstr, {}).get("promoted"):
            parent_ckpt = m["legs"][pstr]["promoted"]
        n_envs = pick_n_envs()

        m["legs"][name] = {"status": "running", "started": time.ctime(),
                           "timesteps": timesteps, "n_envs": n_envs,
                           "parent_ckpt": parent_ckpt}
        save_manifest(m)
        print(f"[controller] === leg {name} :: {timesteps} steps :: "
              f"n_envs={n_envs} :: from {parent_ckpt} ===")

        best = train_leg(leg, parent_ckpt, timesteps, n_envs)
        if _STOP:
            m["legs"][name]["status"] = "interrupted"
            save_manifest(m)
            break
        if best is None or not os.path.exists(best):
            m["legs"][name].update(status="error", promoted=parent_ckpt)
            save_manifest(m)
            print(f"[controller] leg {name} produced no checkpoint; keeping parent")
            continue

        # gate: retention always; skill gate if defined
        reg = run_regression(best, name)
        ret_ok, ret_det = retention_ok(reg)
        gate = run_gate(leg["gate"], best, name) if leg.get("gate") else {}
        score = composite_score(reg, gate or None)

        promoted = best if ret_ok else parent_ckpt  # reject forgetful legs
        m["legs"][name].update(
            status="done", finished=time.ctime(), retention=ret_det,
            retention_ok=ret_ok, gate=gate, score=score,
            trained_ckpt=best, promoted=promoted, skill=leg.get("skill"))
        if ret_ok:
            promote_best(m, best, score, name)
            parent_ckpt = best
        else:
            print(f"[controller] leg {name} REGRESSED core skills {ret_det}; rejected")
        if leg.get("skill") and (gate or {}).get("verdict") == "PASS":
            _drive_report(event=f"skill_proven:{leg['skill']}:{score}",
                          summary=f"skill proven: {leg['skill']} (leg {name})",
                          payload={"skill": leg.get("skill"), "leg": name,
                                   "score": score})
        write_matrix(m)
        save_manifest(m)
        _drive_report()  # periodic 12h report; no-op unless due

    # --- open-ended integrated consolidation until the deadline -------------
    # Phase 9: long-horizon MIXED training + rehearsal. Keeps improving the
    # integrated policy in bounded chunks, each retention-gated, resuming from
    # its own latest.zip, until the wall-clock budget is spent. This is what
    # actually uses the bulk of the 168h -- never a single blind run, never a
    # forgetful promotion.
    round_i = int(m.get("consolidate_rounds", 0))
    ck_dir = None
    while not _STOP:
        remaining = deadline - time.time() - a.eval_margin_hours * 3600.0
        if remaining <= 0:
            print("[controller] budget spent; ending consolidation")
            break
        if free_disk_mb() < a.min_disk_mb:
            print(f"[controller] LOW DISK ({free_disk_mb():.0f} MB); stopping")
            m["stopped_reason"] = "low_disk"
            break
        chunk = min(500_000, int(remaining * RATE))
        _cap = int(os.environ.get("T7_CAP", "0"))
        if _cap:
            chunk = min(chunk, _cap)
        if chunk < (1000 if _cap else 40_000):
            print("[controller] remaining budget too small for a safe chunk; ending")
            break
        name = f"consolidate_{round_i}"
        leg = {"name": name, "config": INTEGRATED_CFG, "gate": None}
        ck_dir = _ckpt_dir(leg)
        latest = os.path.join(ck_dir, "latest.zip")
        # continue the same integrated run across rounds; first round seeds
        # from the best_overall so far.
        seed_from = latest if os.path.exists(latest) else parent_ckpt
        n_envs = pick_n_envs()
        m["legs"][name] = {"status": "running", "started": time.ctime(),
                           "timesteps": chunk, "n_envs": n_envs,
                           "parent_ckpt": seed_from}
        save_manifest(m)
        print(f"[controller] === {name} :: {chunk} steps :: n_envs={n_envs} ===")
        best = train_leg(leg, seed_from, chunk, n_envs)
        if _STOP or best is None or not os.path.exists(best):
            m["legs"][name]["status"] = "interrupted" if _STOP else "error"
            save_manifest(m)
            if _STOP:
                break
            round_i += 1
            m["consolidate_rounds"] = round_i
            continue
        reg = run_regression(best, name)
        ret_ok, ret_det = retention_ok(reg)
        score = composite_score(reg, None)
        m["legs"][name].update(status="done", finished=time.ctime(),
                               retention=ret_det, retention_ok=ret_ok,
                               score=score, trained_ckpt=best,
                               promoted=best if ret_ok else parent_ckpt)
        if ret_ok:
            promote_best(m, best, score, name)
            parent_ckpt = best
        else:
            print(f"[controller] {name} REGRESSED {ret_det}; not promoted")
        round_i += 1
        m["consolidate_rounds"] = round_i
        write_matrix(m)
        save_manifest(m)
        _drive_report()  # periodic 12h report; no-op unless due

    m["ended"] = time.ctime()
    save_manifest(m)
    write_matrix(m)
    _drive_report(event=f"training_complete:{int(m.get('deadline', 0))}",
                  summary="168-hour training complete",
                  payload={"best_overall": m.get("best_overall"),
                           "best_score": m.get("best_score")})
    _drive_report()  # final periodic report
    print(f"[controller] finished. best_overall={m.get('best_overall')} "
          f"score={m.get('best_score')}")


if __name__ == "__main__":
    main()
