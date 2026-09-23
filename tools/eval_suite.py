"""Phase-2 melee evaluation suite: 5 profiles x N seeded episodes.

Train/eval split: default eval seeds (5000+) never used in training configs.
Per-opponent breakdown (never one collapsed score) + JSON output.

  python -m tools.eval_suite --checkpoint <ckpt> --episodes 30 --out logs/eval_suite.json
  python -m tools.eval_suite --baseline  # phase-1.5 ckpt (obs v1) vs suite
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PROFILES = ["chaser", "strafer", "aggressive", "defensive", "random",
            "backoff", "landpunish", "misspunish", "strafepress", "mixed",
            "gspace", "gantiair", "greengage", "gmixed"]
# held-out: training configs use foe seeds < 1000; suite uses 5000+.
EVAL_SEEDS = list(range(5000, 5060))


def _obs_version_for(checkpoint: str | None, model=None) -> int:
    if model is not None:
        try:
            dim = int(model.observation_space.shape[0])
            return {39: 1, 44: 2, 47: 3}[dim]
        except (KeyError, AttributeError, TypeError):
            pass
    if checkpoint and "phase25" in checkpoint:
        return 3
    if checkpoint and "baseline" in checkpoint:
        return 1
    if checkpoint and "phase2" in checkpoint:
        return 2
    return 4


def _mechanics_from(path: str | None) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    import yaml
    with open(path) as f:
        m = yaml.safe_load(f)
    return m.get("combat", {}), m.get("physics", {})


def evaluate(checkpoint: str | None, episodes: int, profiles=None,
             record_dir: str | None = None, deterministic: bool = True,
             mechanics: str | None = None, mirror: bool = False,
             enabled: str | list | None = None,
             loadout: str | dict | None = None,
             arena_size: int = 20, obstacles: list | None = None) -> dict:
    from rl.environment.gym_env import PvPEnv
    from sandbox.config import CombatConfig, PhysicsConfig, WorldConfig
    try:
        from rl.opponents.sustain import REGEN
    except ImportError:
        REGEN = {}
    wc = WorldConfig(size=arena_size, wall_margin=1.0,
                     obstacles=tuple(obstacles) if obstacles is not None
                     else WorldConfig().obstacles)
    model = None
    if checkpoint:
        from stable_baselines3 import PPO
        model = PPO.load(checkpoint)
    profiles = profiles or PROFILES
    obs_v = _obs_version_for(checkpoint, model)
    ckwargs, pkwargs = _mechanics_from(mechanics)
    ccfg, pcfg = CombatConfig(**ckwargs), PhysicsConfig(**pkwargs)
    out: dict = {"checkpoint": checkpoint, "obs_version": obs_v,
                 "mechanics": {"combat": ckwargs, "physics": pkwargs},
                 "profiles": {}}
    if isinstance(enabled, str):
        enabled = set(enabled.split(","))
    elif enabled is not None:
        enabled = set(enabled)
    if isinstance(loadout, str):
        loadout = json.loads(loadout)
    for prof in profiles:
        rows = []
        for e in range(episodes):
            seed = EVAL_SEEDS[e % len(EVAL_SEEDS)]
            env = PvPEnv(opponent=prof, opponent_kwargs={"seed": seed},
                         record=(record_dir is not None and e == 0),
                         obs_version=obs_v, combat_cfg=ccfg, phys_cfg=pcfg,
                         mirror=mirror, loadout=loadout,
                         enabled_items=enabled, world_cfg=wc)
            if prof in REGEN:
                env.set_regen(**REGEN[prof].SIM_PARAMS)
            obs, _ = env.reset(seed=seed)
            dists = []
            done, total = False, 0.0
            while not done:
                if model is None:
                    rng = np.random.default_rng(seed + 999)
                    act = np.array([rng.integers(5), rng.integers(2), rng.integers(2),
                                    rng.integers(2), rng.integers(3), rng.integers(3)])
                else:
                    act, _ = model.predict(obs, deterministic=deterministic)
                obs, rew, term, trunc, info = env.step(act)
                dists.append(info["distance"])
                total += rew
                done = term or trunc
            if record_dir is not None and e == 0:
                os.makedirs(record_dir, exist_ok=True)
                env.save_replay(os.path.join(record_dir, f"suite_{prof}.npz"))
            rows.append({"seed": seed, "win": info.get("win", 0), "reward": total,
                         "dealt": info["damage_dealt"], "taken": info["damage_taken"],
                         "attacks": info["attacks"], "hits": info["hits"],
                         "in_range": info["attacks_in_range"],
                         "blocks": info.get("blocks", 0),
                         "items_used": info.get("items_used", 0),
                         "kb_dealt": info["kb_dealt"], "kb_taken": info["kb_taken"],
                         "ticks": info["tick"], "hp": info["hp"],
                         "mean_dist": float(np.mean(dists))})
        atk = sum(r["attacks"] for r in rows)
        out["profiles"][prof] = {
            "episodes": episodes,
            "win_rate": sum(r["win"] for r in rows) / episodes,
            "mean_reward": float(np.mean([r["reward"] for r in rows])),
            "damage_dealt": float(np.mean([r["dealt"] for r in rows])),
            "damage_taken": float(np.mean([r["taken"] for r in rows])),
            "hit_rate": (sum(r["hits"] for r in rows) / atk) if atk else 0.0,
            "attacks": float(np.mean([r["attacks"] for r in rows])),
            "hits": float(np.mean([r["hits"] for r in rows])),
            "in_range_rate": (sum(r["in_range"] for r in rows) / atk) if atk else 0.0,
            "kb_dealt": float(np.mean([r["kb_dealt"] for r in rows])),
            "kb_taken": float(np.mean([r["kb_taken"] for r in rows])),
            "mean_len": float(np.mean([r["ticks"] for r in rows])),
            "hp_left": float(np.mean([r["hp"] for r in rows])),
            "mean_dist": float(np.mean([r["mean_dist"] for r in rows])),
            "blocks": float(np.mean([r["blocks"] for r in rows])),
            "items_used": float(np.mean([r["items_used"] for r in rows])),
            "episodes_detail": rows,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="logs/eval_suite.json")
    ap.add_argument("--record-dir", default="replays/suite")
    ap.add_argument("--profiles", default=None,
                    help="comma-separated subset, e.g. gspace,gantiair,strafer")
    ap.add_argument("--mechanics", default=None,
                    help="yaml with combat:/physics: sections matching training")
    ap.add_argument("--mirror", action="store_true",
                    help="horizontally mirrored spawns (agent starts east)")
    ap.add_argument("--enabled", default=None,
                    help="comma-separated unlock set matching training locks")
    ap.add_argument("--loadout", default=None,
                    help="JSON item counts, e.g. '{\"bread\": 2}' (default full)")
    ap.add_argument("--arena-size", type=int, default=20)
    ap.add_argument("--obstacles", default=None,
                    help="JSON obstacle list, e.g. '[[15.5,4,16.5,28,3]]'")
    a = ap.parse_args()
    profs = a.profiles.split(",") if a.profiles else None
    import json as _json
    res = evaluate(a.checkpoint, a.episodes, profiles=profs,
                   record_dir=a.record_dir, mechanics=a.mechanics,
                   mirror=a.mirror, enabled=a.enabled, loadout=a.loadout,
                   arena_size=a.arena_size,
                   obstacles=_json.loads(a.obstacles) if a.obstacles else None)
    for prof, m in res["profiles"].items():
        print(f"{prof:10s} win={m['win_rate']:.2f} dealt={m['damage_dealt']:.1f} "
              f"taken={m['damage_taken']:.1f} hit={m['hit_rate']:.2f} "
              f"dist={m['mean_dist']:.2f} hp={m['hp_left']:.1f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
