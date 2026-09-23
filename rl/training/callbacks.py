"""Periodic deterministic evaluation with combat metrics.

Separate envs from training; multi-seed; TensorBoard + JSONL + replay.
Also saves best_model.zip by (win_rate, mean_reward)."""
from __future__ import annotations

import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl.environment.gym_env import PvPEnv


class CombatEvalCallback(BaseCallback):
    def __init__(self, opponent: str = "melee", eval_freq: int = 20000,
                 n_episodes: int = 10, seeds: list[int] | None = None,
                 log_path: str = "logs", replay_dir: str = "replays/eval",
                 best_dir: str = "checkpoints", obs_version: int = 4,
                 combat_cfg=None, phys_cfg=None, loadout=None,
                 enabled_items=None, verbose: int = 0):
        super().__init__(verbose)
        self.opponent = opponent
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.seeds = seeds or [1000 + i for i in range(n_episodes)]
        self.log_path = log_path
        self.replay_dir = replay_dir
        self.best_dir = best_dir
        self.obs_version = obs_version
        self.combat_cfg = combat_cfg
        self.phys_cfg = phys_cfg
        self.loadout = loadout
        self.enabled_items = enabled_items
        self._best_score = (-1.0, float("-inf"))
        self._last = 0
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # threshold on num_timesteps (n_calls counts vector steps, not env steps)
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        res = self._evaluate()
        for k, v in res.items():
            if isinstance(v, float):
                self.logger.record(f"eval_combat/{k}", v)
        # reward components (episode-mean) also to tensorboard
        for k, v in res["reward_components_mean"].items():
            self.logger.record(f"eval_reward_comp/{k}", v)
        line = {"timesteps": self.num_timesteps, **{k: v for k, v in res.items()
                if k != "reward_components_mean"},
                "reward_components": res["reward_components_mean"]}
        with open(os.path.join(self.log_path, "eval_combat.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
        score = (res["win_rate"], res["mean_reward"])
        if score > self._best_score:
            self._best_score = score
            self.model.save(os.path.join(self.best_dir, "best_model"))
            if self.verbose:
                print(f"new best {score} at step {self.num_timesteps}")
        return True

    def _evaluate(self) -> dict:
        wins = 0
        rews, lens, dealt, taken, hits, atks, hps, dists = [], [], [], [], [], [], [], []
        comp_sum: dict = {}
        for i in range(self.n_episodes):
            env = PvPEnv(opponent=self.opponent, record=(i == 0),
                         obs_version=self.obs_version,
                         combat_cfg=self.combat_cfg, phys_cfg=self.phys_cfg,
                         loadout=self.loadout, enabled_items=self.enabled_items)
            obs, _ = env.reset(seed=self.seeds[i % len(self.seeds)])
            done, total = False, 0.0
            while not done:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(act)
                total += rew
                done = term or trunc
            if i == 0:
                env.save_replay(os.path.join(
                    self.replay_dir, f"eval_{self.num_timesteps}_ep0.npz"))
            wins += info.get("win", 0)
            rews.append(total)
            lens.append(info["tick"] // max(1, env.frame_skip))
            dealt.append(info["damage_dealt"])
            taken.append(info["damage_taken"])
            hits.append(info["hits"])
            atks.append(info["attacks"])
            hps.append(info["hp"])
            dists.append(info["distance"])
            for k, v in info["reward_components"].items():
                comp_sum[k] = comp_sum.get(k, 0.0) + v
        n = self.n_episodes
        atk_tot = sum(atks)
        return {
            "win_rate": wins / n,
            "mean_reward": float(np.mean(rews)),
            "mean_len": float(np.mean(lens)),
            "damage_dealt": float(np.mean(dealt)),
            "damage_taken": float(np.mean(taken)),
            "hit_rate": (sum(hits) / atk_tot) if atk_tot else 0.0,
            "attacks": float(np.mean(atks)),
            "hp_left": float(np.mean(hps)),
            "final_dist": float(np.mean(dists)),
            "reward_components_mean": {k: v / n for k, v in comp_sum.items()},
        }
