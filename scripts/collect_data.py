#!/usr/bin/env python3
"""Convert preprocessed captures into training .npz files.

Each capture directory contains:
  video.mkv      — 128×128 grayscale H.264, 60 fps
  audio.wav      — 8-bit mono 16 kHz PCM
  input_log.txt  — "frame bits p1hp p2hp timer p1wins p2wins" per line

Usage:
  python scripts/collect_data.py --capture-dir data/captures \\
      --output-dir data/processed
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
from vision.optical_flow import compute_flow, preprocess_flow
from vision.audio_features import extract_features
from vision.game_state import find_match_boundaries


# ---------------------------------------------------------------------------
def load_audio_features(wav_path: Path, num_frames: int,
                        sample_rate: int = 16000) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
        if sw == 1:
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
            np.arange(len(samples)), samples).astype(np.float32)

    spf = 256
    feats = np.zeros((num_frames, 15), dtype=np.float32)
    for i in range(num_frames):
        chunk = samples[i * spf : i * spf + spf]
        if len(chunk) >= 16:
            feats[i] = extract_features(chunk, sample_rate)
    return feats


# ---------------------------------------------------------------------------
def load_video_frames(path: Path) -> np.ndarray | None:
    """Read all frames as 128×128 uint8 greyscale."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


# ---------------------------------------------------------------------------
def parse_input_log(path: Path) -> tuple:
    """Returns (movements, buttons, hp1, hp2, timer, p1w, p2w)."""
    lines = path.read_text().strip().splitlines()
    T = len(lines)
    movements = np.zeros(T, dtype=np.int64)
    buttons = np.zeros((T, 6), dtype=np.float32)
    hp1 = np.zeros(T, dtype=np.float32)
    hp2 = np.zeros(T, dtype=np.float32)
    timer = np.zeros(T, dtype=np.int32)
    p1w = np.zeros(T, dtype=np.int32)
    p2w = np.zeros(T, dtype=np.int32)

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
            p1w[i] = int(parts[5])
            p2w[i] = int(parts[6])
    return movements, buttons, hp1, hp2, timer, p1w, p2w


# ---------------------------------------------------------------------------
def save_match(output_dir: Path, stem: str, idx: int,
               flows: np.ndarray, movements: np.ndarray,
               buttons: np.ndarray, hp1: np.ndarray, hp2: np.ndarray,
               audio_feats: np.ndarray, start: int, end: int):
    f_start = max(0, start - 1)
    f_end = min(flows.shape[0], end)
    out = output_dir / f"{stem}_match{idx:02d}.npz"
    np.savez_compressed(out,
        flows=flows[f_start:f_end],
        movement=movements[start:end],
        buttons=buttons[start:end],
        health_p1=hp1[start:end],
        health_p2=hp2[start:end],
        audio=audio_feats[start:end])
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Convert captures to .npz training data")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    cfg = Config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in args.capture_dir.iterdir() if d.is_dir())
    if not dirs:
        print(f"No capture subdirectories in {args.capture_dir}")
        return

    total_matches = 0

    for d in tqdm(dirs, desc="Captures"):
        video = d / "video.mkv"
        audio = d / "audio.wav"
        log = d / "input_log.txt"

        if not log.exists():
            tqdm.write(f"  [skip] {d.name}: no input_log.txt")
            continue
        if not video.exists():
            tqdm.write(f"  [skip] {d.name}: no video.mkv")
            continue

        # ---- inputs ----
        movements, buttons_arr, hp1_raw, hp2_raw, timer, p1w, p2w = \
            parse_input_log(log)
        n_frames = len(movements)
        if n_frames < 120:
            continue

        # ---- video → optical flow ----
        frames = load_video_frames(video)
        if frames is None or len(frames) < 2:
            tqdm.write(f"  [skip] {d.name}: cannot read video")
            continue

        nf = min(len(frames) - 1, n_frames - 1)
        flows_list = [preprocess_flow(compute_flow(frames[i], frames[i + 1]),
                                      cfg.frame_size, cfg.flow_norm)
                      for i in range(nf)]
        flows = torch.stack(flows_list, dim=0).numpy()

        # ---- audio ----
        if audio.exists() and cfg.use_audio:
            audio_feats = load_audio_features(audio, n_frames, cfg.audio_sample_rate)
        else:
            audio_feats = np.zeros((n_frames, cfg.audio_feat_dim), dtype=np.float32)

        # ---- match boundaries ----
        signal = (hp1_raw > 0.05) & (hp2_raw > 0.05)
        boundaries = find_match_boundaries(signal)
        if not boundaries:
            out = args.output_dir / f"{d.name}.npz"
            np.savez_compressed(out, flows=flows, movement=movements,
                                buttons=buttons_arr,
                                health_p1=hp1_raw, health_p2=hp2_raw,
                                audio=audio_feats)
            tqdm.write(f"  {out.name}  ({n_frames} f)")
            total_matches += 1
            continue

        for mi, (start, end) in enumerate(boundaries):
            save_match(args.output_dir, d.name, mi,
                       flows, movements, buttons_arr,
                       hp1_raw, hp2_raw, audio_feats, start, end)
        total_matches += len(boundaries)
        tqdm.write(f"  {d.name} → {len(boundaries)} matches  ({n_frames} f)")

    print(f"\nDone — {total_matches} matches saved to {args.output_dir}")


if __name__ == "__main__":
    main()
