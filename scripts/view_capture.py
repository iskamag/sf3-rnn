#!/usr/bin/env python3
"""Visualise captured replays and match-boundary splits.

Reads a capture directory (video.raw + input_log.txt), finds match
boundaries using the same logic as collect_data.py, and plays back the
video with a coloured overlay showing which frames belong to which match.

Usage:
    python scripts/view_capture.py data/captures/1777895668498-5828

Controls:
    a / d        — step one frame
    w / s        — ±60 frames
    [ / ]        — previous / next match
    Space        — play / pause
    m            — toggle optical-flow overlay
    Scrub bar    — drag to seek
    q / Esc      — quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from vision.game_state import audio_match_signal
from vision.optical_flow import compute_flow


def load_video(path: Path, w: int, h: int) -> np.ndarray | None:
    raw = path.read_bytes()
    n = len(raw) // (w * h)
    if n == 0:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w)


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse round wins from input_log.txt."""
    lines = path.read_text().strip().splitlines()
    T = len(lines)
    p1w = np.zeros(T, dtype=np.int32)
    p2w = np.zeros(T, dtype=np.int32)
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 10:
            p1w[i] = int(parts[8])
            p2w[i] = int(parts[9])
        elif len(parts) >= 7:
            p1w[i] = int(parts[5])
            p2w[i] = int(parts[6])
    return p1w, p2w


def boundaries_to_mask(bounds: list[tuple[int, int]], n: int) -> np.ndarray:
    """Return integer array of length *n* — -1 = not in match, >=0 = match idx."""
    mask = np.full(n, -1, dtype=np.int32)
    for mi, (a, b) in enumerate(bounds):
        mask[a:b] = mi
    return mask


def draw_overlay(frame: np.ndarray, idx: int, match_idx: int,
                 total_frames: int, total_matches: int,
                 fps: int, playing: bool,
                 show_flow: bool, flow_frame: np.ndarray | None) -> np.ndarray:
    H, W = 120, 160
    out = np.zeros((H + 28, W, 3), dtype=np.uint8)

    if show_flow and flow_frame is not None:
        # Normalise flow to [0,255] for HSV colour-wheel visualisation.
        mag, ang = cv2.cartToPolar(flow_frame[0], flow_frame[1])
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out[:H, :W] = flow_bgr
    else:
        out[:H, :W] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Match-boundary border.
    if match_idx >= 0:
        cols = [(0, 200, 0), (0, 160, 200), (200, 120, 0), (160, 0, 200),
                (0, 200, 200), (200, 0, 160)]
        colour = cols[match_idx % len(cols)]
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), colour, 2)
    else:
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), (80, 80, 80), 2)

    # Info bar.
    sec = idx / fps
    if match_idx >= 0:
        status = f"match {match_idx}/{total_matches - 1}"
    else:
        status = "menu / select"
    text = f"frame {idx}/{total_frames}  {sec:.1f}s  [{status}]"
    if playing:
        text += "  ▶"
    cv2.putText(out, text, (4, H + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser(description="View captured replay with match boundaries")
    p.add_argument("capture_dir", type=Path)
    p.add_argument("--match-template", type=Path, default=Path("hmm_16k_1s.wav"),
                   help="Audio cue template used to detect match starts")
    p.add_argument("--corr-threshold", type=float, default=0.6,
                   help="Normalised cross-correlation threshold for template matching")
    args = p.parse_args()

    cap = args.capture_dir
    if not cap.is_dir():
        print(f"Not a directory: {cap}")
        return

    cfg = Config()
    video = cap / "video.raw"
    audio = cap / "audio.wav"
    log = cap / "input_log.txt"

    if not video.exists():
        print(f"No video.raw in {cap}")
        return
    if not log.exists():
        print(f"No input_log.txt in {cap}")
        return

    frames = load_video(video, cfg.frame_width, cfg.frame_height)
    if frames is None or len(frames) < 2:
        print("Cannot read video frames")
        return

    p1w, p2w = parse_log(log)
    n = len(frames)
    template = args.match_template
    if not template.is_absolute():
        template = Path(__file__).resolve().parent.parent / template
    if not audio.exists():
        print(f"No audio.wav in {cap} (required for template-based match starts)")
        return
    if not template.exists():
        print(f"Missing template: {template}")
        return

    # The input logger starts before video/audio capture — compensate.
    input_offset = max(0, len(p1w) - n)
    bounds = audio_match_signal(audio, template,
                                p1w, p2w,
                                cfg.audio_sample_rate, cfg.fps,
                                start_offset_frames=cfg.match_start_offset_frames,
                                input_offset_frames=input_offset,
                                min_match_len=cfg.match_min_len,
                                corr_threshold=args.corr_threshold)
    # Convert from input-log frame space to video frame space.
    bounds = [(max(0, s - input_offset), max(0, e - input_offset)) for (s, e) in bounds]
    match_mask = boundaries_to_mask(bounds, n)

    print(f"Frames: {n}  ({n / cfg.fps:.1f}s)")
    print(f"Matches: {len(bounds)}")
    for mi, (a, b) in enumerate(bounds):
        print(f"  match {mi}: frames {a}–{b}  ({b - a} frames, {(b - a) / cfg.fps:.1f}s)")

    cv2.namedWindow("capture", cv2.WINDOW_AUTOSIZE)

    def _noop(_val):
        pass

    idx = bounds[0][0] if bounds else 0
    cv2.createTrackbar("frame", "capture", idx, n - 1, _noop)
    playing = False
    show_flow = False
    flow_cache: tuple[int, np.ndarray | None] = (-1, None)

    while True:
        frame = frames[idx]
        match_idx = match_mask[idx]

        flow = None
        if show_flow and idx < n - 1:
            if flow_cache[0] != idx:
                f = compute_flow(frames[idx], frames[idx + 1])
                flow_cache = (idx, f)
            flow = flow_cache[1]

        out = draw_overlay(frame, idx, match_idx, n, len(bounds), cfg.fps, playing, show_flow, flow)
        cv2.imshow("capture", out)

        # Use a short poll so trackbar drags are picked up immediately.
        delay = 16 if playing else 5
        key_raw = cv2.waitKey(delay)

        tb_pos = cv2.getTrackbarPos("frame", "capture")
        if tb_pos != idx:
            idx = tb_pos
            playing = False

        if key_raw == -1:
            if playing:
                idx = min(n - 1, idx + 1)
                if idx >= n - 1:
                    playing = False
        else:
            key = key_raw & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                playing = not playing
            elif key == ord("a"):
                idx = max(0, idx - 1)
                playing = False
            elif key == ord("d"):
                idx = min(n - 1, idx + 1)
                playing = False
            elif key == ord("w"):
                idx = max(0, idx - 60)
                playing = False
            elif key == ord("s"):
                idx = min(n - 1, idx + 60)
                playing = False
            elif key == ord("m"):
                show_flow = not show_flow
                flow_cache = (-1, None)
            elif key == ord("[") and bounds:
                cur_match = match_mask[idx]
                target = cur_match - 1 if cur_match >= 0 else len(bounds) - 1
                if 0 <= target < len(bounds):
                    idx = bounds[target][0]
                    playing = False
            elif key == ord("]") and bounds:
                cur_match = match_mask[idx]
                target = cur_match + 1 if cur_match < len(bounds) - 1 else 0
                if 0 <= target < len(bounds):
                    idx = bounds[target][0]
                    playing = False

        if tb_pos != idx:
            cv2.setTrackbarPos("frame", "capture", idx)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
