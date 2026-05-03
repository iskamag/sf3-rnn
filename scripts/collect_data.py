#!/usr/bin/env python3
"""Data-collection pipeline — converts captured replays into training .npz files.

Two modes:

  1.  Fightcade .fc2 replays + optional frame/audio captures:
      python scripts/collect_data.py --replay-dir data/replays \\
          --frames-dir data/frames --audio-dir data/audio \\
          --output-dir data/processed

  2.  Direct capture output  (from emulator/capture_replay.sh):
      python scripts/collect_data.py --capture-dir data/captures \\
          --output-dir data/processed

In capture mode each directory contains:
  capture.mkv     — H.264 video at native or scaled resolution
  audio.wav       — 8-bit mono 16 kHz PCM
  input_log.txt   — "frame bits p1hp p2hp timer p1wins p2wins" per line
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from model.action_space import bitfield_to_action
from replay.parser import parse_replay
from vision.optical_flow import compute_flow, preprocess_flow
from vision.audio_features import extract_features
from vision.game_state import (
    health_signal_from_inputs, find_match_boundaries, has_health_bar,
)

ROI_KEYS = ["p1_health", "p2_health", "p1_stun", "p2_stun"]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio_features(wav_path: Path,
                        num_frames: int,
                        sample_rate: int = 16000,
                        ) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
        if sw == 1:
            # 8-bit unsigned
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        if wf.getnchannels() == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

    if sr != sample_rate:
        ratio = sample_rate / sr
        n_out = int(len(samples) * ratio)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n_out),
            np.arange(len(samples)), samples,
        ).astype(np.float32)

    spf = 256
    feats = np.zeros((num_frames, 15), dtype=np.float32)
    for i in range(num_frames):
        start = i * spf
        chunk = samples[start : start + spf]
        if len(chunk) >= 16:
            feats[i] = extract_features(chunk, sample_rate)
    return feats


# ---------------------------------------------------------------------------
# Video frame loading
# ---------------------------------------------------------------------------
def load_frames_from_video(video_path: Path,
                           target_h: int, target_w: int,
                           ) -> np.ndarray | None:
    """Read all frames from a video file, resize to (target_h, target_w)
    greyscale.  Returns (T, H, W) uint8 or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_w, target_h))
        frames.append(gray)
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Input log parsing
# ---------------------------------------------------------------------------
def parse_input_log(path: Path) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
    """Parse the Lua input log.

    Returns (movements, buttons, hp1, hp2, timer, wins) each of length T.
    """
    lines = path.read_text().strip().splitlines()
    T = len(lines)
    movements = np.zeros(T, dtype=np.int64)
    buttons = np.zeros((T, 6), dtype=np.float32)
    hp1 = np.zeros(T, dtype=np.float32)
    hp2 = np.zeros(T, dtype=np.float32)
    timer = np.zeros(T, dtype=np.int32)
    p1_wins = np.zeros(T, dtype=np.int32)
    p2_wins = np.zeros(T, dtype=np.int32)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 3:
            continue
        bits = int(parts[1])
        d, bs = bitfield_to_action(bits)
        movements[i] = int(d)
        for b in bs:
            buttons[i, int(b)] = 1.0
        if len(parts) >= 7:
            hp1[i] = float(parts[2])
            hp2[i] = float(parts[3])
            timer[i] = int(parts[4])
            p1_wins[i] = int(parts[5])
            p2_wins[i] = int(parts[6])

    return movements, buttons, hp1, hp2, timer, p1_wins, p2_wins


# ---------------------------------------------------------------------------
# Match splitting
# ---------------------------------------------------------------------------
def build_health_signal(hp1: np.ndarray, hp2: np.ndarray) -> np.ndarray:
    return (hp1 > 0.05) & (hp2 > 0.05)


def build_health_signal_from_buttons(buttons: np.ndarray) -> np.ndarray:
    return health_signal_from_inputs(buttons)


# ---------------------------------------------------------------------------
def save_match(output_dir: Path, stem: str, idx: int,
               flows: np.ndarray, movements: np.ndarray,
               buttons: np.ndarray, hp1: np.ndarray,
               hp2: np.ndarray, audio_feats: np.ndarray,
               start: int, end: int):
    f_start = max(0, start - 1)
    f_end = min(flows.shape[0], end)
    out = output_dir / f"{stem}_match{idx:02d}.npz"
    np.savez_compressed(
        out,
        flows=flows[f_start:f_end],
        movement=movements[start:end],
        buttons=buttons[start:end],
        health_p1=hp1[start:end],
        health_p2=hp2[start:end],
        audio=audio_feats[start:end],
    )
    return out


# ---------------------------------------------------------------------------
# Capture-dir mode: process video + input_log + audio
# ---------------------------------------------------------------------------
def process_capture_dir(capture_dir: Path, output_dir: Path, cfg: Config):
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in capture_dir.iterdir() if d.is_dir())
    if not dirs:
        print(f"No capture subdirs in {capture_dir}")
        return

    total_matches = 0
    for d in tqdm(dirs, desc="Captures"):
        video = d / "capture.mkv"
        audio = d / "audio.wav"
        log = d / "input_log.txt"

        if not log.exists():
            tqdm.write(f"  [skip] {d.name}: no input_log.txt")
            continue

        # ---- parse inputs ----
        movements, buttons_arr, hp1_raw, hp2_raw, timer, p1_w, p2_w = \
            parse_input_log(log)
        n_frames = len(movements)
        if n_frames < 120:
            tqdm.write(f"  [skip] {d.name}: too short ({n_frames} frames)")
            continue

        # ---- load frames from video ----
        if video.exists():
            frames = load_frames_from_video(video, cfg.frame_size, cfg.frame_size)
            if frames is not None and len(frames) > 1:
                n_flow = min(len(frames) - 1, n_frames - 1)
                flows_list = []
                for i in range(n_flow):
                    f = compute_flow(frames[i], frames[i + 1])
                    flows_list.append(preprocess_flow(f, cfg.frame_size, cfg.flow_norm))
                flows = torch.stack(flows_list, dim=0).numpy()
            else:
                flows = np.zeros((max(0, n_frames - 1), 2,
                                  cfg.frame_size, cfg.frame_size), dtype=np.float32)
        else:
            tqdm.write(f"  [warn] {d.name}: no capture.mkv, flows will be zeros")
            flows = np.zeros((max(0, n_frames - 1), 2,
                              cfg.frame_size, cfg.frame_size), dtype=np.float32)

        # ---- audio features ----
        if audio.exists() and cfg.use_audio:
            audio_feats = load_audio_features(audio, n_frames, cfg.audio_sample_rate)
        else:
            audio_feats = np.zeros((n_frames, cfg.audio_feat_dim), dtype=np.float32)

        # ---- match boundaries ----
        signal = build_health_signal(hp1_raw, hp2_raw)
        if signal.sum() < 120:
            signal = build_health_signal_from_buttons(buttons_arr)

        boundaries = find_match_boundaries(signal)
        if not boundaries:
            out = output_dir / f"{d.name}.npz"
            np.savez_compressed(out, flows=flows, movement=movements,
                                buttons=buttons_arr,
                                health_p1=hp1_raw, health_p2=hp2_raw,
                                audio=audio_feats)
            tqdm.write(f"  {out.name}  ({n_frames} f)")
            total_matches += 1
            continue

        for mi, (start, end) in enumerate(boundaries):
            save_match(output_dir, d.name, mi,
                       flows, movements, buttons_arr,
                       hp1_raw, hp2_raw, audio_feats, start, end)
        total_matches += len(boundaries)
        tqdm.write(f"  {d.name} → {len(boundaries)} matches  "
                   f"({n_frames} f)")

    print(f"\nDone — {total_matches} matches saved to {output_dir}")


# ---------------------------------------------------------------------------
# Replay-dir mode: process .fc2 + optional frames/audio
# ---------------------------------------------------------------------------
def process_replay_dir(replay_dir: Path, frames_dir: Path | None,
                       audio_dir: Path | None, output_dir: Path, cfg: Config):
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_files = sorted(replay_dir.glob("*.fc2"))
    if not replay_files:
        print("No .fc2 files found in", replay_dir)
        return

    total_matches = 0
    for rp in tqdm(replay_files, desc="Replays"):
        try:
            meta, inputs = parse_replay(rp)
        except Exception as e:
            tqdm.write(f"  [skip] {rp.name}: {e}")
            continue

        p1_stream = inputs[0]
        n_frames = meta.num_frames
        if n_frames < 120:
            continue

        # ---- frames ----
        frames = None
        hp1_raw = np.zeros(n_frames, dtype=np.float32)
        hp2_raw = np.zeros(n_frames, dtype=np.float32)
        if frames_dir:
            root = frames_dir / rp.stem
            frame_files = sorted(root.glob("frame_*.png"))
            if frame_files:
                frames_bgr = [cv2.imread(str(f)) for f in frame_files if cv2.imread(str(f)) is not None]
                if frames_bgr:
                    frames = np.array([
                        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                   (cfg.frame_size, cfg.frame_size))
                        for f in frames_bgr
                    ])
                    native_frames = np.array([
                        cv2.resize(f, (cfg.native_width, cfg.native_height))
                        for f in frames_bgr
                    ])
                    roi = {k: getattr(cfg, k + "_roi") for k in ROI_KEYS}
                    from vision.game_state import read_health
                    for i, nf in enumerate(native_frames):
                        hp1_raw[i] = read_health(
                            nf[roi["p1_health"][1]:
                               roi["p1_health"][1] + roi["p1_health"][3],
                               roi["p1_health"][0]:
                               roi["p1_health"][0] + roi["p1_health"][2]])
                        hp2_raw[i] = read_health(
                            nf[roi["p2_health"][1]:
                               roi["p2_health"][1] + roi["p2_health"][3],
                               roi["p2_health"][0]:
                               roi["p2_health"][0] + roi["p2_health"][2]])

        # ---- optical flow ----
        if frames is not None and len(frames) > 1:
            flows_list = []
            for i in range(len(frames) - 1):
                f = compute_flow(frames[i], frames[i + 1])
                flows_list.append(preprocess_flow(f, cfg.frame_size, cfg.flow_norm))
            flows = torch.stack(flows_list, dim=0).numpy()
        else:
            flows = np.zeros((max(0, n_frames - 1), 2,
                              cfg.frame_size, cfg.frame_size), dtype=np.float32)

        # ---- audio ----
        audio_feats = np.zeros((n_frames, cfg.audio_feat_dim), dtype=np.float32)
        ad = audio_dir or frames_dir
        if cfg.use_audio and ad:
            wav = ad / rp.stem / "audio.wav"
            if wav.exists():
                audio_feats = load_audio_features(wav, n_frames, cfg.audio_sample_rate)

        # ---- actions ----
        movements = np.zeros(n_frames, dtype=np.int64)
        buttons_arr = np.zeros((n_frames, 6), dtype=np.float32)
        for i in range(min(n_frames, len(p1_stream))):
            d, bs = bitfield_to_action(p1_stream[i])
            movements[i] = int(d)
            for b in bs:
                buttons_arr[i, int(b)] = 1.0

        # ---- match boundaries ----
        if frames is not None:
            roi = {k: getattr(cfg, k + "_roi") for k in ROI_KEYS}
            signal = np.array([has_health_bar(f, roi) for f in native_frames], dtype=bool)
        else:
            signal = build_health_signal_from_buttons(buttons_arr)

        if signal is None or signal.sum() < 120:
            out = output_dir / f"{rp.stem}.npz"
            np.savez_compressed(out, flows=flows, movement=movements,
                                buttons=buttons_arr,
                                health_p1=hp1_raw, health_p2=hp2_raw,
                                audio=audio_feats)
            tqdm.write(f"  {out.name}  ({n_frames} f)")
            total_matches += 1
            continue

        boundaries = find_match_boundaries(signal)
        if not boundaries:
            out = output_dir / f"{rp.stem}.npz"
            np.savez_compressed(out, flows=flows, movement=movements,
                                buttons=buttons_arr,
                                health_p1=hp1_raw, health_p2=hp2_raw,
                                audio=audio_feats)
            tqdm.write(f"  {out.name}  ({n_frames} f)")
            total_matches += 1
            continue

        for mi, (start, end) in enumerate(boundaries):
            save_match(output_dir, rp.stem, mi,
                       flows, movements, buttons_arr,
                       hp1_raw, hp2_raw, audio_feats, start, end)
        total_matches += len(boundaries)
        tqdm.write(f"  {rp.stem} → {len(boundaries)} matches  "
                   f"({n_frames} f)")

    print(f"\nDone — {total_matches} matches saved to {output_dir}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Process replays into per-match training data")
    p.add_argument("--replay-dir", type=Path, default=None,
                   help="Directory of .fc2 replay files")
    p.add_argument("--capture-dir", type=Path, default=None,
                   help="Directory of capture outputs (from capture_replay.sh)")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Pre-captured frames (replay mode)")
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="Recorded audio WAVs (replay mode, defaults to --frames-dir)")

    args = p.parse_args()
    cfg = Config()

    if args.capture_dir:
        process_capture_dir(args.capture_dir, args.output_dir, cfg)
    elif args.replay_dir:
        process_replay_dir(args.replay_dir, args.frames_dir,
                           args.audio_dir or args.frames_dir,
                           args.output_dir, cfg)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
