#!/usr/bin/env python3
"""PPO fine-tuning with dual-latency pipeline.

    python scripts/train_rl.py --bc-ckpt checkpoints/bc_epoch099.pt

The dual-buffer design models both sensory and motor latency separately.
Logged transitions use the *total-delay* flow so that PPO updates see
correctly aligned (state, action, reward) tuples.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from config import Config
from model.rnn import SF3RNN
from model.action_space import Direction, Button
from model.reaction_buffer import ReactionBuffer
from model.audio_buffer import AudioBuffer
from model.output_buffer import OutputDelayBuffer
from train.ppo import PPOTrainer
from train.reward import compute_step_reward, RewardConfig
from vision.game_state import extract_game_state, GameState
from env.sf3_env import SF3Env, MssScreenEnv
from env.audio_capture import AudioCapture, AudioCaptureStub


def _reward_config(cfg: Config) -> RewardConfig:
    return RewardConfig(
        health_delta=cfg.reward_health_delta,
        round_win=cfg.reward_round_win,
        match_win=cfg.reward_match_win,
        idle=cfg.reward_idle_penalty,
    )


# ---------------------------------------------------------------------------
def collect_episode(env: SF3Env,
                    model: SF3RNN,
                    trainer: PPOTrainer,
                    sensory_buf: ReactionBuffer,
                    audio_buf: AudioBuffer | None,
                    motor_buf: OutputDelayBuffer,
                    cfg: Config,
                    max_steps: int = 3600,
                    ) -> float:
    """Run one episode through the dual-latency pipeline, fill the PPO buffer."""

    model.eval()
    model.reset_hidden()

    frame = env.reset()
    sensory_buf.add_frame(frame)
    total_reward = 0.0
    prev_state: GameState | None = None

    roi = {
        "p1_health": cfg.p1_health_roi,
        "p2_health": cfg.p2_health_roi,
        "p1_stun":   cfg.p1_stun_roi,
        "p2_stun":   cfg.p2_stun_roi,
    }

    device = next(model.parameters()).device
    reward_cfg = _reward_config(cfg)

    for _ in range(max_steps):
        flow = sensory_buf.get_flow()
        if flow is None:
            frame, _, _, _ = env.step(Direction.NEUTRAL, [])
            sensory_buf.add_frame(frame)
            continue

        # ---- LSTM forward ----
        audio_feats = audio_buf.get_features() if audio_buf is not None else None
        with torch.no_grad():
            actions, hidden = model.step(flow, model.hidden,
                                         audio_feats=audio_feats)
            model.hidden = hidden

        mov_dist = torch.distributions.Categorical(logits=actions["movement"])
        movement = mov_dist.sample().item()

        btn_probs = torch.sigmoid(actions["buttons"])
        btn_dist = torch.distributions.Bernoulli(btn_probs)
        button_mask = btn_dist.sample()

        log_prob = (mov_dist.log_prob(torch.tensor(movement, device=device)) +
                    btn_dist.log_prob(button_mask).sum())

        # ---- Push into motor-lag pipeline ----
        delayed = motor_buf.push(
            movement, button_mask,
            log_prob.detach(),
            actions["value"].detach(),
        )

        if delayed is None:
            # Pipeline still priming — idle
            frame, _, _, _ = env.step(Direction.NEUTRAL, [])
            sensory_buf.add_frame(frame)
            continue

        # ---- Execute the motor-delayed action ----
        direction = Direction(delayed.movement)
        buttons = [Button(i) for i, v in enumerate(delayed.buttons) if v > 0.5]
        frame, _, done, info = env.step(direction, buttons)

        # ---- Reward from health delta ----
        curr_state = extract_game_state(frame, roi)
        reward = 0.0
        if prev_state is not None:
            reward = compute_step_reward(
                prev_state, curr_state, env.is_p1(),
                reward_cfg, len(buttons) > 0,
            )
        prev_state = curr_state

        sensory_buf.add_frame(frame)

        # ---- Log transition ----
        decision_flow = sensory_buf.get_flow_at(cfg.total_delay_frames)
        if decision_flow is None:
            decision_flow = flow

        kwargs = {}
        if cfg.use_audio:
            kwargs["audio_feats"] = audio_feats.squeeze(0) if audio_feats is not None \
                else torch.zeros(cfg.audio_feat_dim)

        trainer.buffer.add(
            decision_flow, delayed.movement, delayed.buttons,
            delayed.log_prob.to(device), delayed.value.to(device),
            reward, done, **kwargs,
        )

        total_reward += reward
        if done:
            break

    return total_reward


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-ckpt", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--save-every", type=int, default=100,
                   help="Episodes between checkpoints")
    args = p.parse_args()

    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    trainer = PPOTrainer(cfg, bc_checkpoint=args.bc_ckpt)

    env = MssScreenEnv(cfg)

    sensory_buf = ReactionBuffer(
        sensory_lag=cfg.sensory_lag_frames,
        frame_size=cfg.frame_size, flow_norm=cfg.flow_norm,
        device=str(device))
    motor_buf = OutputDelayBuffer(motor_lag=cfg.motor_lag_frames)

    audio_buf: AudioBuffer | None = None
    audio_cap: AudioCapture | AudioCaptureStub = AudioCaptureStub()
    if cfg.use_audio:
        audio_buf = AudioBuffer(
            audio_lag=cfg.audio_lag_frames,
            sample_rate=cfg.audio_sample_rate,
            feat_dim=cfg.audio_feat_dim,
            fps=cfg.fps,
            device=str(device))
        audio_cap = AudioCapture(callback=audio_buf.add_samples,
                                 sample_rate=cfg.audio_sample_rate)
    audio_cap.start()

    total_steps = 0
    episode = 0
    t0 = time.time()

    while total_steps < args.total_steps:
        ep_reward = collect_episode(
            env, trainer.model, trainer,
            sensory_buf, audio_buf, motor_buf, cfg,
        )

        ep_len = len(trainer.buffer)
        total_steps += ep_len
        episode += 1

        if ep_len < 10:
            trainer.buffer.reset()
            continue

        metrics = trainer.update()
        trainer.step = total_steps

        print(f"[PPO] ep {episode:4d}  steps {total_steps:7d}  "
              f"(len={ep_len})  reward {ep_reward:+.2f}  "
              f"loss {metrics['loss']:.4f}  entropy {metrics['entropy']:.4f}")

        trainer.writer.add_scalar("episode/reward", ep_reward, total_steps)
        for k, v in metrics.items():
            trainer.writer.add_scalar(f"train/{k}", v, total_steps)

        if episode % args.save_every == 0:
            trainer.save()

    trainer.save(tag="_final")
    audio_cap.stop()
    env.close()
    print(f"Done — {(time.time() - t0)/3600:.1f} h")


if __name__ == "__main__":
    main()
