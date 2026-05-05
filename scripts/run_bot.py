#!/usr/bin/env python3
"""Run a trained checkpoint in real time with dual-latency pipeline.

    python scripts/run_bot.py --ckpt checkpoints/ppo_step001000.pt

The model runs once per *temporal_stride* (N) frames and produces N consecutive
actions in one recurrence step.  Vision still streams at 60 fps.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from config import Config
from model.rnn import SF3RNN
from model.action_space import decode_action, Direction, Button
from model.reaction_buffer import ReactionBuffer
from model.audio_buffer import AudioBuffer
from model.output_buffer import OutputDelayBuffer
from vision.game_state import has_health_bar
from env.sf3_env import MssScreenEnv
from env.audio_capture import AudioCapture, AudioCaptureStub


# ---------------------------------------------------------------------------
def _parse_direction(name: str) -> Direction:
    return getattr(Direction, name.upper())


def _parse_buttons(names: tuple[str, ...]) -> list[Button]:
    return [getattr(Button, n.upper()) for n in names]


# ---------------------------------------------------------------------------
def navigate_menu(env: MssScreenEnv,
                  sensory_buf: ReactionBuffer,
                  audio_buf: AudioBuffer | None,
                  motor_buf: OutputDelayBuffer,
                  cfg: Config,
                  roi: dict,
                  ) -> int:
    seq = cfg.char_select_seq
    post_stall = cfg.post_match_stall
    menu_detect = cfg.menu_detect_frames

    for _ in range(post_stall):
        env.step(Direction.NEUTRAL, [])

    health_seen_since = 0
    frames_spent = post_stall

    for direction_name, button_names, hold in seq:
        d = _parse_direction(direction_name)
        b = _parse_buttons(button_names)

        for _ in range(hold):
            frame, _, _, _ = env.step(d, b)
            sensory_buf.add_frame(frame)
            frames_spent += 1

            if has_health_bar(frame, roi):
                health_seen_since += 1
            else:
                health_seen_since = 0

    safety = menu_detect
    while safety > 0:
        frame, _, _, _ = env.step(Direction.NEUTRAL, [])
        sensory_buf.add_frame(frame)
        frames_spent += 1

        if has_health_bar(frame, roi):
            health_seen_since += 1
            if health_seen_since >= 6:
                break
        else:
            health_seen_since = 0
        safety -= 1

    motor_buf._queue.clear()
    return frames_spent


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--sensory-lag", type=int, default=None)
    p.add_argument("--motor-lag", type=int, default=None)
    p.add_argument("--no-menu-nav", action="store_true",
                   help="Disable auto character-select (manual control)")
    args = p.parse_args()

    cfg = Config()
    if args.sensory_lag is not None:
        cfg.sensory_lag_frames = args.sensory_lag
    if args.motor_lag is not None:
        cfg.motor_lag_frames = args.motor_lag

    N = cfg.temporal_stride
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    roi = {
        "p1_health": cfg.p1_health_roi,
        "p2_health": cfg.p2_health_roi,
    }

    model = SF3RNN(
        feature_dim=cfg.feature_dim,
        use_audio=cfg.use_audio,
        audio_dim=cfg.audio_dim,
        context_dim=cfg.context_dim,
        context_layers=cfg.context_layers,
        action_embed_dim=cfg.action_embed_dim,
        temporal_stride=cfg.temporal_stride,
        mingru_hidden=cfg.mingru_hidden,
        mingru_layers=cfg.mingru_layers,
        mingru_dropout=cfg.mingru_dropout,
        num_directions=cfg.num_directions,
        num_buttons=cfg.num_buttons,
        cnn_channels=cfg.cnn_channels,
    ).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded checkpoint (step {ckpt.get('step', '?')})")

    env = MssScreenEnv(cfg)
    frame = env.reset()

    # Need N extra buffer slots to collect multi-frame flows
    sensory_buf = ReactionBuffer(
        sensory_lag=cfg.sensory_lag_frames,
        extra=cfg.temporal_stride,
        frame_width=cfg.frame_width, frame_height=cfg.frame_height,
        flow_norm=cfg.flow_norm,
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

    sensory_buf.add_frame(frame)
    frame_ms = 1000.0 / args.fps
    step = 0
    actions_sent = 0

    health_absent = 0
    navigations = 0

    # Stride-N pipeline state
    pending_flows: list[torch.Tensor] = []
    pending_audio: list[torch.Tensor] = []
    pending_actions: list[tuple] = []                # (dir_int, btn_mask, logp, val)

    print(f"Running  (sensory={cfg.sensory_lag_frames}f  motor={cfg.motor_lag_frames}f  "
          f"stride={N}  total={cfg.total_delay_frames}f).  Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.perf_counter()

            # ---- menu detection ----
            health_ok = has_health_bar(frame, roi)
            if health_ok:
                health_absent = 0
            else:
                health_absent += 1

            if health_absent > cfg.menu_detect_frames and not args.no_menu_nav:
                print(f"  [menu] navigating character select …")
                spent = navigate_menu(env, sensory_buf, audio_buf, motor_buf, cfg, roi)
                model.reset_hidden()
                pending_flows.clear()
                pending_audio.clear()
                pending_actions.clear()
                health_absent = 0
                navigations += 1
                print(f"  [menu] done — {spent} f spent, resuming")
                frame, _, _, _ = env.step(Direction.NEUTRAL, [])
                sensory_buf.add_frame(frame)
                continue

            # ---- accumulate sensory data ----
            flow = sensory_buf.get_flow()
            if flow is not None:
                pending_flows.append(flow.squeeze(0).squeeze(0))              # (2, H, W)
                if audio_buf is not None:
                    af = audio_buf.get_features()
                    pending_audio.append(af.squeeze(0) if af is not None else torch.zeros(cfg.audio_feat_dim))

            # ---- model forward every N frames ----
            if len(pending_flows) >= N and len(pending_actions) == 0:
                flows_in = torch.stack(pending_flows[-N:], dim=0).unsqueeze(0)  # (1, N, 2, H, W)
                audio_in = None
                if cfg.use_audio and len(pending_audio) >= N:
                    audio_in = torch.stack(pending_audio[-N:], dim=0).unsqueeze(0)  # (1, N, 15)
                elif cfg.use_audio:
                    audio_in = torch.zeros(1, N, cfg.audio_feat_dim, device=device)

                with torch.inference_mode():
                    actions, hidden = model.step(flows_in.to(device), model.hidden,
                                                  audio_feats=audio_in.to(device) if audio_in is not None else None)
                    model.hidden = hidden

                # Decode all N actions
                for i in range(N):
                    mov_i = actions["movement"][i]                                # (9,)
                    btn_i = actions["buttons"][i]                                 # (6,)
                    val_i = actions["value"][i]                                   # scalar

                    if args.deterministic:
                        direction, buttons = decode_action(mov_i, btn_i, threshold=0.5)
                        direction_int = int(direction)
                        btn_probs = torch.sigmoid(btn_i)
                        btn_mask = (btn_probs > 0.5).float()
                        mov_dist = torch.distributions.Categorical(logits=mov_i)
                        log_prob = (mov_dist.log_prob(torch.tensor(direction_int, device=device)) +
                                    torch.distributions.Bernoulli(btn_probs).log_prob(btn_mask).sum())
                    else:
                        mov_dist = torch.distributions.Categorical(logits=mov_i)
                        direction_int = mov_dist.sample().item()
                        btn_probs = torch.sigmoid(btn_i)
                        btn_dist = torch.distributions.Bernoulli(btn_probs)
                        btn_mask = btn_dist.sample()
                        log_prob = (mov_dist.log_prob(torch.tensor(direction_int, device=device)) +
                                    btn_dist.log_prob(btn_mask).sum())

                    pending_actions.append((direction_int, btn_mask, log_prob.detach(), val_i.detach()))

                pending_flows = pending_flows[-1:]                                 # keep last for continuity
                pending_audio = pending_audio[-1:]

            # ---- consume next action from pending batch ----
            if pending_actions:
                dir_int, btn_mask, logp, val = pending_actions.pop(0)
                delayed = motor_buf.push(dir_int, btn_mask, logp, val)
                if delayed is not None:
                    direction = Direction(delayed.movement)
                    buttons = [Button(i) for i, v in enumerate(delayed.buttons) if v > 0.5]
                    actions_sent += 1
                else:
                    direction, buttons = Direction.NEUTRAL, []
            else:
                direction, buttons = Direction.NEUTRAL, []

            # ---- step environment ----
            frame, _, _, _ = env.step(direction, buttons)
            sensory_buf.add_frame(frame)

            step += 1
            elapsed_ms = (time.perf_counter() - loop_start) * 1000
            sleep_ms = max(0.0, frame_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)

            if step % 60 == 0:
                btn_str = "+".join(b.name for b in buttons) if buttons else "--"
                print(f"  step {step:5d}  {direction.name:14s}  [{btn_str}]  "
                      f"loop={elapsed_ms:.1f}ms  sent={actions_sent}"
                      f"  nav={navigations}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        audio_cap.stop()
        env.close()


if __name__ == "__main__":
    main()
