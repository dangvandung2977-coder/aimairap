"""VPS sizing benchmark: steps/sec for n_envs in {1,2,4,8}. Writes JSON.

  python -m tools.bench --opponent melee --decisions 400
Headless; random actions; measures env throughput (no learning).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.environment.gym_env import PvPEnv


def bench(n_envs: int, opponent: str, decisions: int, seed: int = 0) -> dict:
    def _mk(i: int):
        def _t():
            return PvPEnv(opponent=opponent, seed=seed + i)
        return _t
    vec = (DummyVecEnv if n_envs == 1 else SubprocVecEnv)([_mk(i) for i in range(n_envs)])
    from rl.environment import action_map
    rng = np.random.default_rng(seed)
    vec.reset()
    t0 = time.perf_counter()
    ticks = 0
    for _ in range(decisions):
        acts = np.stack([[int(rng.integers(n)) for n in action_map.NVEC]
                         for _ in range(n_envs)])
        _, _, _, infos = vec.step(acts)
        ticks += 4 * n_envs  # frame_skip=4 physics ticks per decision per env
    dt = time.perf_counter() - t0
    vec.close()
    return {"n_envs": n_envs, "ticks": ticks, "seconds": round(dt, 2),
            "ticks_per_sec": round(ticks / dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="melee")
    ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--out", default="logs/bench.json")
    a = ap.parse_args()
    import multiprocessing
    print("cpus:", os.cpu_count())
    results = [bench(n, a.opponent, a.decisions) for n in (1, 2, 4, 8)]
    for r in results:
        print(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"cpu_count": os.cpu_count(),
                   "agent": multiprocessing.current_process().name,
                   "results": results}, f, indent=2)
    best = max(results, key=lambda r: r["ticks_per_sec"])
    print("best throughput:", best)


if __name__ == "__main__":
    main()
