#!/usr/bin/env python3
"""Convert replay captures into mmap-friendly training shards.

Each capture directory contains:
  video.raw      — 160×120 grayscale raw bytes, 60 fps
  audio.wav      — 8-bit mono 16 kHz PCM
  input_log.txt  — "frame p1bits p2bits p1hp p2hp p1stun p2stun timer p1wins p2wins reset" per line
"""

from __future__ import annotations

import argparse
import os
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import Config
from data.shard import FLOW_MAX, FLOW_MIN, quantize_flow, write_json
from model.action_space import bitfield_to_action
from vision.optical_flow import compute_flow
from vision.audio_features import dct_batch, get_mel_filterbank
from vision.game_state import audio_match_signal


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

    if num_frames == 0:
        return np.zeros((0, 15), dtype=np.float32)

    fft_window = 256
    half_window = fft_window // 2
    samples_per_frame = sample_rate / fps

    # Build the interior windows with sliding_window_view and fill the
    # boundary windows with a small explicit loop.
    chunks_raw = np.zeros((num_frames, fft_window), dtype=np.float32)
    window = np.hanning(fft_window).astype(np.float32)
    centers = np.round((np.arange(num_frames) + 0.5) * samples_per_frame).astype(int)
    starts = centers - half_window

    valid = (starts >= 0) & (starts + fft_window <= len(samples))
    if valid.any():
        windows = np.lib.stride_tricks.sliding_window_view(samples, fft_window)
        chunks_raw[valid] = windows[starts[valid]]

    for i in np.where(~valid)[0]:
        src_start = max(starts[i], 0)
        src_end = min(starts[i] + fft_window, len(samples))
        if src_end > src_start:
            dst_start = src_start - starts[i]
            chunk_len = src_end - src_start
            chunks_raw[i, dst_start:dst_start + chunk_len] = samples[src_start:src_end]

    # RMS energy (dB-scaled) from raw (unwindowed) chunks
    rms = np.sqrt(np.mean(chunks_raw.astype(np.float64) ** 2, axis=1))
    rms_db = 20.0 * np.log10(rms + 1e-10)
    rms_db = np.clip(rms_db, -60.0, 0.0) / 60.0 + 1.0

    # Apply Hann window (zero-padded regions stay 0)
    chunks = chunks_raw * window

    # Batched FFT
    fft = np.fft.rfft(chunks)                               # (N, 129)
    mag = np.abs(fft[:, 1:])                                 # (N, 128) skip DC
    power = mag * mag

    # Mel filterbank (matrix multiply)
    fbank = get_mel_filterbank(sample_rate)                 # (40, 128)
    mel_energies = power @ fbank.T                           # (N, 40)
    log_mel = np.log(mel_energies + 1e-10)

    # DCT-II
    mfccs = dct_batch(log_mel, 13)                          # (N, 13)

    # Spectral centroid
    freqs = np.fft.rfftfreq(fft_window, 1.0 / sample_rate)[1:]
    centroid = np.sum(freqs * mag, axis=1) / (np.sum(mag, axis=1) + 1e-10)
    centroid_norm = np.clip(centroid / (sample_rate / 2), 0.0, 1.0)

    feats = np.column_stack([mfccs, rms_db, centroid_norm])
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
def load_video_frames(path: Path, frame_width: int, frame_height: int) -> np.ndarray | None:
    """Read raw video frames as uint8 greyscale (already 160×120 GRAY8)."""
    raw = path.read_bytes()
    frame_bytes = frame_width * frame_height
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, frame_height, frame_width)


# ---------------------------------------------------------------------------
def parse_input_log(path: Path) -> tuple:
    """Returns (p1_mov, p1_btn, p2_mov, p2_btn, hp1, hp2, stun1, stun2, timer, p1w, p2w, reset)."""
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
    reset = np.zeros(T, dtype=np.int32)

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
        # P2 actions stay joystick-relative, and the player indicator bit
        # supplies perspective information to the model.
        p2_bits = int(parts[2])
        d2, bs2 = bitfield_to_action(p2_bits)
        p2_mov[i] = int(d2)
        for b in bs2:
            p2_btn[i, int(b)] = 1.0

        if len(parts) >= 11:
            hp1[i] = float(parts[3])
            hp2[i] = float(parts[4])
            stun1[i] = int(parts[5])
            stun2[i] = int(parts[6])
            timer[i] = int(parts[7])
            p1w[i] = int(parts[8])
            p2w[i] = int(parts[9])
            reset[i] = int(parts[10])
    return p1_mov, p1_btn, p2_mov, p2_btn, hp1, hp2, stun1, stun2, timer, p1w, p2w, reset


# ---------------------------------------------------------------------------
def save_match_shard(output_dir: Path, stem: str, idx: int,
                     flows_u8: np.ndarray, p1_mov: np.ndarray, p1_btn: np.ndarray,
                     p2_mov: np.ndarray, p2_btn: np.ndarray,
                     hp1: np.ndarray, hp2: np.ndarray,
                     stun1: np.ndarray, stun2: np.ndarray,
                     audio_feats: np.ndarray, start: int, end: int,
                     cfg: Config, input_offset: int = 0):
    # start/end are in input-log frame space.
    # flows and audio_feats may be shorter than the input log (offset frames).
    v_start = start - input_offset
    v_end = end - input_offset
    f_start = max(0, v_start)
    f_end = min(flows_u8.shape[0], v_end - 1)
    if f_end <= f_start:
        return None  # skip segments outside the available flow range
    out = output_dir / f"{stem}_match{idx:03d}"
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "flow_u8.npy", flows_u8[f_start:f_end])
    np.save(out / "audio_f16.npy", audio_feats[v_start:v_end].astype(np.float16))
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
        "start_frame": v_start,
        "end_frame": v_end,
        "num_frames": v_end - v_start,
        "num_flows": f_end - f_start,
        "fps": cfg.fps,
        "frame_width": cfg.frame_width,
        "frame_height": cfg.frame_height,
        "flow_width": cfg.flow_width,
        "flow_height": cfg.flow_height,
        "flow_dtype": "uint8",
        "flow_min": FLOW_MIN,
        "flow_max": FLOW_MAX,
        "audio_dtype": "float16",
    })
    return out


# ---------------------------------------------------------------------------
def _compute_flow_pair(args):
    i, prev, curr, frame_width, frame_height, flow_norm = args
    flow = compute_flow(prev, curr)                            # (2, H, W) f32
    h, w = flow.shape[1], flow.shape[2]
    if h != frame_height or w != frame_width:
        flow = cv2.resize(flow.transpose(1, 2, 0),
                          (frame_width, frame_height),
                          interpolation=cv2.INTER_LINEAR)
        flow = np.ascontiguousarray(flow.transpose(2, 0, 1), dtype=np.float32)
    flow /= flow_norm
    clipped = np.clip(flow, FLOW_MIN, FLOW_MAX)
    return i, quantize_flow(clipped)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Convert captures to training shards")
    p.add_argument("--capture-dir", type=Path, default=Path("data/captures"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--match-template", type=Path, default=Path("hmm_16k_08s.wav"),
                   help="Audio cue template used to detect match starts")
    p.add_argument("--corr-threshold", type=float, default=0.5,
                   help="Normalised cross-correlation threshold for template matching")
    args = p.parse_args()

    cfg = Config()
    project_root = Path(__file__).resolve().parent.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_captures = {d.name for d in args.capture_dir.iterdir() if d.is_dir()}
    if not all_captures:
        print(f"No capture subdirectories in {args.capture_dir}")
        return

    suffix_len = len("_match000")
    processed = set()
    for p in args.output_dir.iterdir():
        if len(p.name) > suffix_len:
            processed.add(p.name[:-suffix_len])

    dirs = sorted(args.capture_dir / name for name in all_captures - processed)
    if not dirs:
        print("All captures already processed.")
        return

    total_matches = 0

    for d in tqdm(dirs, desc="Captures"):
        video = d / "video.raw"
        audio = d / "audio.wav"
        log = d / "input_log.txt"

        if not log.exists():
            tqdm.write(f"  [skip] {d.name}: no input_log.txt")
            continue
        if not video.exists():
            tqdm.write(f"  [skip] {d.name}: no video.raw")
            continue

        # ---- inputs ----
        p1_mov, p1_btn, p2_mov, p2_btn, hp1_raw, hp2_raw, stun1, stun2, timer, p1w, p2w, reset = \
            parse_input_log(log)
        n_frames = len(p1_mov)
        if n_frames < 120:
            continue

        # ---- video → optical flow ----
        frames = load_video_frames(video, cfg.frame_width, cfg.frame_height)
        if frames is None or len(frames) < 2:
            tqdm.write(f"  [skip] {d.name}: cannot read video")
            continue

        nf = min(len(frames) - 1, n_frames - 1)
        flows = np.empty((nf, 2, cfg.flow_height, cfg.flow_width), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            fut_map = {
                pool.submit(_compute_flow_pair,
                            (i, frames[i], frames[i + 1],
                             cfg.flow_width, cfg.flow_height, cfg.flow_norm)): i
                for i in range(nf)
            }
            for fut in as_completed(fut_map):
                try:
                    i, flow = fut.result()
                except Exception as e:
                    tqdm.write(f"  [warn] flow frame {fut_map[fut]} failed: {e}")
                    continue
                flows[i] = flow

        # ---- audio ----
        if audio.exists() and cfg.use_audio:
            audio_feats = load_audio_features(audio, n_frames,
                                              cfg.audio_sample_rate, cfg.fps)
        else:
            audio_feats = np.zeros((n_frames, cfg.audio_feat_dim), dtype=np.float32)

        # ---- match boundaries (audio start template + logged match ends) ----
        template = args.match_template
        if not template.is_absolute():
            template = project_root / template
        if not audio.exists():
            tqdm.write(f"  [skip] {d.name}: no audio.wav for template-based match starts")
            continue
        if not template.exists():
            tqdm.write(f"  [skip] {d.name}: missing template {template}")
            continue

        # The input logger starts before video/audio capture — compensate.
        input_offset = max(0, n_frames - len(frames))
        boundaries = audio_match_signal(audio, template,
                                        p1w, p2w,
                                        cfg.audio_sample_rate, cfg.fps,
                                        start_offset_frames=cfg.match_start_offset_frames,
                                        input_offset_frames=input_offset,
                                        min_match_len=cfg.match_min_len,
                                        corr_threshold=args.corr_threshold)
        if not boundaries:
            tqdm.write(f"  [skip] {d.name}: no audio/log-derived match boundaries")
            continue

        for mi, (start, end) in enumerate(boundaries):
            save_match_shard(args.output_dir, d.name, mi,
                             flows, p1_mov, p1_btn, p2_mov, p2_btn,
                             hp1_raw, hp2_raw, stun1, stun2, audio_feats,
                             start, end, cfg, input_offset=input_offset)
        total_matches += len(boundaries)
        tqdm.write(f"  {d.name} → {len(boundaries)} matches  ({n_frames} f)")

    print(f"\nDone — {total_matches} matches saved to {args.output_dir}")


if __name__ == "__main__":
    main()
