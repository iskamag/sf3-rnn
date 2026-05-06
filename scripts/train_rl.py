#!/usr/bin/env python3
"""PPO fine-tuning with dual-latency pipeline and gymnasium interface.

    python scripts/train_rl.py --bc-ckpt checkpoints/bc_epoch099.pt

The environment handles sensory lag (ReactionBuffer) internally.
Motor lag (OutputDelayBuffer) lives in the training loop so that stored
(action, log_prob, reward) triples remain properly aligned — the action
in the buffer is the one that was *actually executed* (motor-delayed),
paired with the log_prob from when the model produced it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import Config
from env.sf3_env import SF3Env
from env.audio_capture import AudioCapture, AudioCaptureStub
from model.audio_buffer import AudioBuffer
from model.output_buffer import OutputDelayBuffer
from train.ppo import SF3PPO, Transition


def _make_audio_pipeline(cfg, device_str):
    audio_buf: AudioBuffer | None = None
    audio_cap: AudioCapture | AudioCaptureStub = AudioCaptureStub()
    if cfg.use_audio:
        audio_buf = AudioBuffer(
            audio_lag=cfg.audio_lag_frames,
            sample_rate=cfg.audio_sample_rate,
            feat_dim=cfg.audio_feat_dim,
            fps=cfg.fps,
            device=device_str,
        )
        audio_cap = AudioCapture(
            callback=audio_buf.add_samples,
            sample_rate=cfg.audio_sample_rate,
        )
    return audio_buf, audio_cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-ckpt", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()

    cfg = Config()
    device_str = cfg.device if torch.cuda.is_available() else "cpu"

    # ── environment (gymnasium, sensory lag in ReactionBuffer) ─────
    env = SF3Env(cfg)

    # ── PPO trainer (loads BC checkpoint) ──────────────────────────
    trainer = SF3PPO(cfg, bc_checkpoint=args.bc_ckpt)

    # ── audio ──────────────────────────────────────────────────────
    audio_buf, audio_cap = _make_audio_pipeline(cfg, device_str)
    audio_cap.start()

    # ── main loop ──────────────────────────────────────────────────
    t0 = time.time()

    while trainer.total_steps < args.total_steps:
        obs, info = env.reset()
        trainer.reset()

        # Fresh motor delay buffer for this episode.
        motor_buf = OutputDelayBuffer(motor_lag=cfg.motor_lag_frames)

        ep_reward = 0.0
        ep_len = 0

        while True:
            # Audio features (delayed auditory pathway).
            audio_feats = None
            if audio_buf is not None:
                af = audio_buf.get_features()
                audio_feats = af.squeeze(0) if af is not None else None

            # Policy produces a raw action (stride-N accumulation inside).
            flow_t = torch.from_numpy(obs).unsqueeze(0)
            mov_int, btn_mask, logp, val = trainer.act(
                flow_t, audio=audio_feats,
            )

            # Motor delay: push raw action, pop delayed action.
            delayed = motor_buf.push(mov_int, btn_mask, logp, val)

            if delayed is not None:
                # Execute the motor-delayed action.
                action = np.concatenate(
                    [[delayed.movement], delayed.buttons.numpy().astype(np.int64)])
            else:
                # Motor buffer still priming — send neutral.
                action = np.zeros(7, dtype=np.int64)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            # Only store transitions when the motor buffer is fully primed,
            # so the stored action matches what was actually executed.
            if delayed is not None:
                trainer.buffer.add(Transition(
                    flow=flow_t.squeeze(0).cpu(),
                    audio=audio_feats.cpu() if audio_feats is not None else None,
                    movement=delayed.movement,
                    buttons=delayed.buttons.cpu(),
                    log_prob=delayed.log_prob.cpu(),
                    value=delayed.value.cpu(),
                    reward=reward,
                    done=terminated or truncated,
                ))

            if terminated or truncated:
                break
            if ep_len > 3600:
                break

        trainer.total_steps += ep_len
        trainer.episode += 1

        if ep_len < 10:
            continue

        # ── PPO update (sequence-aware, full trajectories) ─────────
        metrics = trainer.update()

        print(f"[PPO] ep {trainer.episode:4d}  steps {trainer.total_steps:7d}  "
              f"(len={ep_len})  reward {ep_reward:+.2f}  "
              f"loss {metrics['loss']:.4f}  entropy {metrics['entropy']:.4f}")

        trainer.writer.add_scalar("episode/reward", ep_reward, trainer.total_steps)
        trainer.writer.add_scalar("episode/length", ep_len, trainer.total_steps)
        for k, v in metrics.items():
            trainer.writer.add_scalar(f"train/{k}", v, trainer.total_steps)

        if trainer.episode % args.save_every == 0:
            trainer.save()

    trainer.save(tag="_final")
    audio_cap.stop()
    env.close()
    print(f"Done — {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
