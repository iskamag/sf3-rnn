#!/usr/bin/env python3
"""PPO fine-tuning with CleanRL-style trajectory replay.

Usage:
    python scripts/train_rl.py --bc-ckpt checkpoints/bc_epoch099.pt

Audio is captured from the system and fed into the agent alongside
visual observations.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import Config
from env.sf3_env import SF3Env
from train.ppo import PPOTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-ckpt", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--steps-per-batch", type=int, default=2048,
                   help="Environment frames collected per PPO update")
    args = p.parse_args()

    cfg = Config()
    env = SF3Env(cfg)
    trainer = PPOTrainer(env, cfg, bc_checkpoint=args.bc_ckpt)

    t0 = time.time()

    while trainer.total_steps < args.total_steps:
        t_col = time.time()
        trajectories = trainer.collect_rollouts(args.steps_per_batch)
        col_time = time.time() - t_col
        collected = sum(len(t) for t in trajectories)

        if not trajectories:
            continue

        t_upd = time.time()
        metrics = trainer.update(trajectories)
        upd_time = time.time() - t_upd

        print(
            f"[PPO] step {trainer.total_steps:7d}  "
            f"ep {trainer.episode:4d}  "
            f"f {collected:4d}  col {col_time:.1f}s  upd {upd_time:.2f}s  "
            f"loss {metrics['loss']:.4f}  "
            f"v {metrics['value_loss']:.4f}  "
            f"ent {metrics['entropy']:.4f}"
        )

        trainer.writer.add_scalar("train/loss", metrics["loss"], trainer.total_steps)
        trainer.writer.add_scalar("train/policy_loss", metrics["policy_loss"], trainer.total_steps)
        trainer.writer.add_scalar("train/value_loss", metrics["value_loss"], trainer.total_steps)
        trainer.writer.add_scalar("train/entropy", metrics["entropy"], trainer.total_steps)
        sps = collected / max(0.01, col_time + upd_time)
        trainer.writer.add_scalar("train/sps", sps, trainer.total_steps)

        if trainer.episode % args.save_every == 0:
            trainer.save()

    trainer.save(tag="_final")
    trainer.close()
    env.close()
    print(f"Done — {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
