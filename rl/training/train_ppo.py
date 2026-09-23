"""PPO training entrypoint: random / vs dummy / vs scripted melee."""
from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.environment.gym_env import PvPEnv


def build_envs(opponent: str, n_envs: int, seed: int, frame_skip: int, max_ticks: int):
    def _mk(i: int):
        def _thunk():
            env = PvPEnv(opponent=opponent, seed=seed + i,
                         frame_skip=frame_skip, max_ticks=max_ticks)
            env = Monitor(env)
            return env
        return _thunk
    if n_envs == 1:
        return DummyVecEnv([_mk(0)])
    return SubprocVecEnv([_mk(i) for i in range(n_envs)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="dummy", choices=["dummy", "melee", "strafe"])
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=900)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-freq", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    vec = build_envs(args.opponent, args.n_envs, args.seed, args.frame_skip, args.max_ticks)
    eval_env = Monitor(PvPEnv(opponent=args.opponent, seed=args.seed + 10_000,
                              frame_skip=args.frame_skip, max_ticks=args.max_ticks))
    eval_cb = EvalCallback(eval_env, eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
                           log_path=args.logdir, best_model_save_path=args.checkpoints,
                           deterministic=True)
    ckpt_cb = CheckpointCallback(save_freq=max(1, args.eval_freq // max(1, args.n_envs)),
                                 save_path=args.checkpoints, name_prefix=f"ppo_{args.opponent}")

    if args.resume:
        print(f"resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec, tensorboard_log=args.logdir)
    else:
        model = PPO("MlpPolicy", vec, verbose=1, seed=args.seed,
                    learning_rate=args.lr, n_steps=args.n_steps,
                    batch_size=args.batch_size, tensorboard_log=args.logdir)
    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, eval_cb])
    final = os.path.join(args.checkpoints, f"ppo_{args.opponent}_final.zip")
    model.save(final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
