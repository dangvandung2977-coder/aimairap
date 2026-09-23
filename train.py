"""Headless PPO training entrypoint. No GUI imports; VPS-safe.

  python train.py --config configs/baseline_melee.yaml
  python train.py --config ... --timesteps 20000 --seed 1
  python train.py --resume checkpoints/baseline/latest.zip [--config ...]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.environment.gym_env import PvPEnv
from rl.rewards.shaping import RewardConfig
from rl.training.callbacks import CombatEvalCallback
from sandbox.config import CombatConfig, PhysicsConfig, WorldConfig


class StepCheckpointCallback(BaseCallback):
    """Every save_freq steps: step_N.zip + latest.zip, prune old steps (keep K)."""

    def __init__(self, freq: int, directory: str, keep: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self.directory = directory
        self.keep = keep
        self._last = 0
        os.makedirs(directory, exist_ok=True)

    def _on_step(self) -> bool:
        # NOTE: n_calls counts vector steps (timesteps/n_envs); threshold on
        # num_timesteps so freq is in real env steps regardless of n_envs.
        if self.model.num_timesteps - self._last < self.freq:
            return True
        self._last = self.model.num_timesteps
        step = self.model.num_timesteps
        self.model.save(os.path.join(self.directory, f"step_{step}"))
        self.model.save(os.path.join(self.directory, "latest"))
        paths = sorted(glob.glob(os.path.join(self.directory, "step_*.zip")),
                       key=os.path.getmtime)
        for old in paths[:-self.keep]:
            os.remove(old)
        return True


def deep_update(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def stage_pool(cfg: dict, stage_idx: int) -> list[dict]:
    """Stage = one entry of env.opponent_pool, or the fixed env.opponent.
    Sample-pool stages (no fixed 'pool') return a placeholder; the sampler
    takes over in StageCallback._apply."""
    pool = cfg["env"].get("opponent_pool")
    if not pool:
        e = cfg["env"]
        return [{"opponent": e.get("opponent", "melee"),
                 "kwargs": e.get("opponent_kwargs", {})}]
    entry = pool[stage_idx]
    if "pool" in entry:
        return entry["pool"]
    sample = entry.get("sample_pool") or []
    if sample:
        first = sample[0]
        return [{"opponent": first.get("opponent", "melee"),
                 "kwargs": first.get("kwargs", {}),
                 "enabled": entry.get("enabled")}]
    return [{"opponent": cfg["env"].get("opponent", "melee"),
             "kwargs": cfg["env"].get("opponent_kwargs", {})}]


class StageCallback(BaseCallback):
    """Swap foe profiles at step thresholds (simple curriculum, no self-play)."""

    def __init__(self, cfg: dict, verbose: int = 0):
        super().__init__(verbose)
        pool = cfg["env"].get("opponent_pool") or []
        self.untils = [s.get("until") for s in pool]
        self.cfg = cfg
        self.stage = 0

    def _apply(self, stage: int) -> None:
        pool = stage_pool(self.cfg, stage)
        cfg_stage = (self.cfg["env"].get("opponent_pool") or [{}])[stage] \
            if self.cfg["env"].get("opponent_pool") else {}
        n = self.model.get_env().num_envs
        base_seed = self.cfg["run"]["seed"]
        for i in range(n):
            entry = pool[i % len(pool)]
            self.model.get_env().env_method(
                "set_opponent", entry.get("opponent", "melee"),
                entry.get("kwargs", {}), base_seed + stage * 1000 + i,
                indices=i)
        enabled = cfg_stage.get("enabled", pool[0].get("enabled", None))
        if enabled is not None:
            self.model.get_env().env_method("set_enabled", enabled)
        loadout = cfg_stage.get("loadout", None)
        if loadout is not None:
            self.model.get_env().env_method("set_loadout", loadout)
        regen = cfg_stage.get("regen", None)
        if regen is not None:
            self.model.get_env().env_method(
                "set_regen", float(regen.get("rate", 0.0)),
                int(regen.get("delay", 0)))
        obstacles = cfg_stage.get("obstacles", None)
        if obstacles is not None:
            self.model.get_env().env_method("set_obstacles", obstacles)
        sample = cfg_stage.get("sample_pool", None)
        # sample_pool replaces fixed per-env assignment with per-episode draws
        self.model.get_env().env_method("set_foe_pool", sample,
                                        base_seed + stage)
        self.logger.record("curriculum/stage", stage)
        if self.verbose:
            print(f"curriculum -> stage {stage}: "
                  f"{[e.get('opponent') for e in pool]} "
                  f"enabled={enabled} reason=scheduled")

    def _on_training_start(self) -> None:
        self._apply(0)

    def _on_step(self) -> bool:
        if self.stage < len(self.untils) and self.untils[self.stage] is not None:
            if self.model.num_timesteps >= self.untils[self.stage]:
                self.stage += 1
                self._apply(self.stage)
        return True


def load_config(path: str, cli_over: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return deep_update(cfg, cli_over or {})


def build_envs(cfg: dict):
    e, ecf = cfg["env"], cfg["env"]
    world = WorldConfig(size=e.get("arena_size", 20),
                        wall_margin=e.get("wall_margin", 1.0),
                        obstacles=tuple(map(tuple, e.get("obstacles", []))))
    reward = RewardConfig(**cfg.get("reward", {}))
    combat = CombatConfig(**cfg.get("combat", {}))
    phys = PhysicsConfig(**cfg.get("physics", {}))

    def _mk(i: int):
        pool0 = stage_pool(cfg, 0)
        entry = pool0[i % len(pool0)]

        def _thunk():
            env = PvPEnv(opponent=entry.get("opponent", "melee"),
                         opponent_kwargs=entry.get("kwargs", {}),
                         reward_cfg=copy.deepcopy(reward),
                         world_cfg=world, phys_cfg=phys, combat_cfg=combat,
                         max_ticks=e.get("max_ticks", 900),
                         frame_skip=e.get("frame_skip", 4),
                         obs_version=e.get("obs_version", 4),
                         loadout=e.get("loadout", None),
                         enabled_items=(entry.get("enabled", None)
                                        or e.get("enabled_items", None)),
                         agent_start_hp=e.get("agent_start_hp", None),
                         foe_start_hp=e.get("foe_start_hp", None),
                         start_dist=e.get("start_dist", None),
                         seed=cfg["run"]["seed"] + i)
            return Monitor(env)
        return _thunk

    n = cfg["perf"].get("n_envs", 4)
    vec_cls = DummyVecEnv if n == 1 else SubprocVecEnv
    return vec_cls([_mk(i) for i in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_melee.yaml")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-freq", type=int, default=None)
    ap.add_argument("--ckpt-freq", type=int, default=None)
    args = ap.parse_args()

    cli = {"run": {}}
    if args.seed is not None:
        cli["run"]["seed"] = args.seed
    if args.timesteps is not None:
        cli.setdefault("training", {})["timesteps"] = args.timesteps
    if args.n_envs is not None:
        cli.setdefault("perf", {})["n_envs"] = args.n_envs
    if args.eval_freq is not None:
        cli.setdefault("eval", {})["freq"] = args.eval_freq
    if args.ckpt_freq is not None:
        cli.setdefault("checkpoint", {})["freq"] = args.ckpt_freq
    cfg = load_config(args.config, cli)
    run, tr, ev, ck = cfg["run"], cfg["training"], cfg["eval"], cfg["checkpoint"]
    os.makedirs(ck["dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["tensorboard"], exist_ok=True)
    with open(os.path.join(ck["dir"], "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    vec = build_envs(cfg)
    from sandbox.config import CombatConfig as _CC, PhysicsConfig as _PC
    _cc = _CC(**cfg.get("combat", {}))
    _pc = _PC(**cfg.get("physics", {}))
    eval_cb = CombatEvalCallback(opponent=cfg["env"].get("opponent", "melee"),
                                 eval_freq=ev.get("freq", 50000),
                                 n_episodes=ev.get("episodes", 10),
                                 seeds=ev.get("seeds"),
                                 log_path=cfg["logging"]["tensorboard"],
                                 replay_dir=os.path.join(ck["dir"], "eval_replays"),
                                 best_dir=ck["dir"],
                                 obs_version=cfg["env"].get("obs_version", 4),
                                 combat_cfg=_cc, phys_cfg=_pc,
                                 loadout=cfg["env"].get("loadout", None),
                                 enabled_items=cfg["env"].get("enabled_items", None))
    ckpt_cb = StepCheckpointCallback(freq=ck.get("freq", 100000),
                                     directory=ck["dir"], keep=ck.get("keep", 5))
    stage_cb = StageCallback(cfg, verbose=1)

    if args.resume:
        print(f"resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec,
                         tensorboard_log=cfg["logging"]["tensorboard"])
        reset = False
    else:
        model = PPO("MlpPolicy", vec, verbose=1, seed=run.get("seed", 0),
                    tensorboard_log=cfg["logging"]["tensorboard"],
                    learning_rate=tr.get("lr", 3e-4),
                    n_steps=tr.get("n_steps", 512),
                    batch_size=tr.get("batch_size", 256),
                    n_epochs=tr.get("n_epochs", 4),
                    gamma=tr.get("gamma", 0.99),
                    gae_lambda=tr.get("gae_lambda", 0.95),
                    clip_range=tr.get("clip_range", 0.2),
                    ent_coef=tr.get("ent_coef", 0.0),
                    vf_coef=tr.get("vf_coef", 0.5),
                    max_grad_norm=tr.get("max_grad_norm", 0.5),
                    policy_kwargs=dict(net_arch=tr.get("net_arch", [128, 128])))
        reset = True
    total = tr.get("timesteps", 1_000_000)
    model.learn(total_timesteps=total, callback=[ckpt_cb, eval_cb, stage_cb],
                reset_num_timesteps=reset)
    model.save(os.path.join(ck["dir"], "final"))
    print(f"done. checkpoints in {ck['dir']}")


if __name__ == "__main__":
    main()
