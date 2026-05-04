#!/usr/bin/env python3
"""Convert replay captures into mmap-friendly training shards.

Each capture directory contains:
  video.mkv      — native-resolution cache video, 60 fps
  audio.wav      — 8-bit mono 16 kHz PCM
  input_log.txt  — "frame p1bits p2bits p1hp p2hp p1stun p2stun timer p1wins p2wins" per line

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
from tqdm import tqdm

from config import Config
from data.shard import FLOW_MAX, FLOW_MIN, quantize_flow, write_json
from model.action_space import bitfield_to_action
from vision.optical_flow import compute_flow, preprocess_flow
from vision.audio_features import extract_features
from vision.game_state import find_match_boundaries


# ---------------------------------------------------------------------------
def load_audio_features(wav_path: Path, num_frames: int,
                        sample_rate: int = 16000,
                        fps: int = 60) -> np.ndarray:
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

    fft_window = 256
    half_window = fft_window // 2
    samples_per_frame = sample_rate / fps
    feats = np.zeros((num_frames, 15), dtype=np.float32)
    for i in range(num_frames):
        center = int(round((i + 0.5) * samples_per_frame))
        start = center - half_window
        end = start + fft_window
        chunk = np.zeros(fft_window, dtype=np.float32)
        src_start = max(start, 0)
        src_end = min(end, len(samples))
        if src_end > src_start:
            dst_start = src_start - start
            chunk[dst_start:dst_start + (src_end - src_start)] = samples[src_start:src_end]
            feats[i] = extract_features(chunk, sample_rate)
    return feats


# ---------------------------------------------------------------------------
def load_video_frames(path: Path, frame_size: int) -> np.ndarray | None:
    """Read video frames as model-sized uint8 greyscale."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


# ---------------------------------------------------------------------------
def parse_input_log(path: Path) -> tuple:
    """Returns (p1_mov, p1_btn, p2_mov, p2_btn, hp1, hp2, stun1, stun2, timer, p1w, p2w)."""
    lines = path.read_text().strip().splitlines()
    T = len(lines)
    p1_mov = np.zeros(T, dtype=np.int64)
    p1_btn = np.zeros((T, 6), dtype=np.float32)
    p2_mov = np.zeros(T, dtype=np.int64)
    p2_btn = np.zeros((T, 6), dtype=np.float32)
    hp1 = np.zeros(T, dtype=np.float32)
    hp2 = np.zeros(T, dtype=np.float32)
    stun1 = np.zeros(T, dtype=np.int32)
    stun2 = np.zeros(T, dtype=np.int32)
    timer = np.zeros(T, dtype=np.int32)
    p1w = np.zeros(T, dtype=np.int32)
    p2w = np.zeros(T, dtype=np.int32)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue
        # P1 actions
        p1_bits = int(parts[1])
        d, bs = bitfield_to_action(p1_bits)
        p1_mov[i] = int(d)
        for b in bs:
            p1_btn[i, int(b)] = 1.0
        # P2 actions — raw joystick-relative, no mirroring (the model gets a
        # player-indicator bit to learn the P1/P2 mapping from visual context).
        p2_bits = int(parts[2])
        d2, bs2 = bitfield_to_action(p2_bits)
        p2_mov[i] = int(d2)
        for b in bs2:
            p2_btn[i, int(b)] = 1.0

        if len(parts) >= 10:
            hp1[i] = float(parts[3])
            hp2[i] = float(parts[4])
            stun1[i] = int(parts[5])
            stun2[i] = int(parts[6])
            timer[i] = int(parts[7])
            p1w[i] = int(parts[8])
            p2w[i] = int(parts[9])
        elif len(parts) >= 7:
            # back-compat: old 7-field format (no P2 bits or stun)
            hp1[i] = float(parts[2])
            hp2[i] = float(parts[3])
            timer[i] = int(parts[4])
            p1w[i] = int(parts[5])
            p2w[i] = int(parts[6])
    return p1_mov, p1_btn, p2_mov, p2_btn, hp1, hp2, stun1, stun2, timer, p1w, p2w


# ---------------------------------------------------------------------------
def save_match_npz(output_dir: Path, stem: str, idx: int,
                   flows: np.ndarray, p1_mov: np.ndarray, p1_btn: np.ndarray,
                   p2_mov: np.ndarray, p2_btn: np.ndarray,
                   hp1: np.ndarray, hp2: np.ndarray,
                   stun1: np.ndarray, stun2: np.ndarray,
                   audio_feats: np.ndarray, start: int, end: int):
    f_start = start
    f_end = min(flows.shape[0], end - 1)
    out = output_dir / f"{stem}_match{idx:02d}.npz"
    np.savez_compressed(out,
        flows=flows[f_start:f_end],
        movement=p1_mov[start:end],
        buttons=p1_btn[start:end],
        p2_movement=p2_mov[start:end],
        p2_buttons=p2_btn[start:end],
        health_p1=hp1[start:end],
        health_p2=hp2[start:end],
        stun_p1=stun1[start:end],
        stun_p2=stun2[start:end],
        audio=audio_feats[start:end])
    return out


def save_match_shard(output_dir: Path, stem: str, idx: int,
                     flows_u8: np.ndarray, p1_mov: np.ndarray, p1_btn: np.ndarray,
                     p2_mov: np.ndarray, p2_btn: np.ndarray,
                     hp1: np.ndarray, hp2: np.ndarray,
                     stun1: np.ndarray, stun2: np.ndarray,
                     audio_feats: np.ndarray, start: int, end: int,
                     cfg: Config):
    f_start = start
    f_end = min(flows_u8.shape[0], end - 1)
    out = output_dir / f"{stem}_match{idx:02d}"
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "flow_u8.npy", flows_u8[f_start:f_end])
    np.save(out / "audio_f16.npy", audio_feats[start:end].astype(np.float16))
    np.save(out / "movement_u8.npy", p1_mov[start:end].astype(np.uint8))
    np.save(out / "buttons_u8.npy", p1_btn[start:end].astype(np.uint8))
    np.save(out / "p2_movement_u8.npy", p2_mov[start:end].astype(np.uint8))
    np.save(out / "p2_buttons_u8.npy", p2_btn[start:end].astype(np.uint8))
    np.save(out / "health_p1_u8.npy", np.rint(np.clip(hp1[start:end], 0.0, 1.0) * 255).astype(np.uint8))
    np.save(out / "health_p2_u8.npy", np.rint(np.clip(hp2[start:end], 0.0, 1.0) * 255).astype(np.uint8))
    np.save(out / "stun_p1_u8.npy", stun1[start:end].astype(np.uint8))
    np.save(out / "stun_p2_u8.npy", stun2[start:end].astype(np.uint8))
    write_json(out / "meta.json", {
        "format": "st-rnn-shard-v1",
        "source_capture": stem,
        "match_index": idx,
        "start_frame": start,
        "end_frame": end,
        "num_frames": end - start,
        "num_flows": f_end - f_start,
        "fps": cfg.fps,
        "frame_size": cfg.frame_size,
        "flow_dtype": "uint8",
        "flow_min": FLOW_MIN,
        "flow_max": FLOW_MAX,
        "audio_dtype": "float16",
    })
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Convert captures to training shards")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--format", choices=("shard", "npz"), default="shard")
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
        p1_mov, p1_btn, p2_mov, p2_btn, hp1_raw, hp2_raw, stun1, stun2, timer, p1w, p2w = \
            parse_input_log(log)
        n_frames = len(p1_mov)
        if n_frames < 120:
            continue

        # ---- video → optical flow ----
        frames = load_video_frames(video, cfg.frame_size)
        if frames is None or len(frames) < 2:
            tqdm.write(f"  [skip] {d.name}: cannot read video")
            continue

        nf = min(len(frames) - 1, n_frames - 1)
        if args.format == "shard":
            flows = np.empty((nf, 2, cfg.frame_size, cfg.frame_size), dtype=np.uint8)
            for i in range(nf):
                flow = preprocess_flow(compute_flow(frames[i], frames[i + 1]),
                                       cfg.frame_size, cfg.flow_norm).numpy()
                flows[i] = quantize_flow(flow)
        else:
            flows = np.empty((nf, 2, cfg.frame_size, cfg.frame_size), dtype=np.float32)
            for i in range(nf):
                flows[i] = preprocess_flow(compute_flow(frames[i], frames[i + 1]),
                                           cfg.frame_size, cfg.flow_norm).numpy()

        # ---- audio ----
        if audio.exists() and cfg.use_audio:
            audio_feats = load_audio_features(audio, n_frames,
                                              cfg.audio_sample_rate, cfg.fps)
        else:
            audio_feats = np.zeros((n_frames, cfg.audio_feat_dim), dtype=np.float32)

        # ---- match boundaries ----
        signal = (hp1_raw > 0.05) & (hp2_raw > 0.05)
        boundaries = find_match_boundaries(signal)
        if not boundaries:
            saver = save_match_shard if args.format == "shard" else save_match_npz
            if args.format == "shard":
                out = saver(args.output_dir, d.name, 0, flows, p1_mov, p1_btn,
                            p2_mov, p2_btn, hp1_raw, hp2_raw, stun1, stun2,
                            audio_feats, 0, n_frames, cfg)
            else:
                out = saver(args.output_dir, d.name, 0, flows, p1_mov, p1_btn,
                            p2_mov, p2_btn, hp1_raw, hp2_raw, stun1, stun2,
                            audio_feats, 0, n_frames)
            tqdm.write(f"  {out.name}  ({n_frames} f)")
            total_matches += 1
            continue

        for mi, (start, end) in enumerate(boundaries):
            if args.format == "shard":
                save_match_shard(args.output_dir, d.name, mi,
                                 flows, p1_mov, p1_btn, p2_mov, p2_btn,
                                 hp1_raw, hp2_raw, stun1, stun2, audio_feats,
                                 start, end, cfg)
            else:
                save_match_npz(args.output_dir, d.name, mi,
                               flows, p1_mov, p1_btn, p2_mov, p2_btn,
                               hp1_raw, hp2_raw, stun1, stun2, audio_feats,
                               start, end)
        total_matches += len(boundaries)
        tqdm.write(f"  {d.name} → {len(boundaries)} matches  ({n_frames} f)")

    print(f"\nDone — {total_matches} matches saved to {args.output_dir}")


if __name__ == "__main__":
    main()
