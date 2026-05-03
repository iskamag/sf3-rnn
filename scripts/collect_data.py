#!/usr/bin/env python3
"""End-to-end data-collection pipeline.

  1.  Parse Fightcade .fc2 replays  →  per-frame inputs
  2.  (opt) Screen-capture replay playback  →  per-frame images
  3.  Compute optical flow
  4.  Detect match boundaries — trim character select, intros, victory screens
  5.  Save one .npz per match

Usage:
    python scripts/collect_data.py \
        --replay-dir data/replays \
        --output-dir data/processed

If frames are available the script uses health-bar HSV detection to find
match starts/ends.  Without frames it falls back to a button-activity
heuristic (menus have sparse directional inputs; gameplay has frequent
button presses).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from model.action_space import bitfield_to_action
from replay.parser import parse_replay
from vision.optical_flow import compute_flow, preprocess_flow
from vision.game_state import (
    health_signal_from_inputs, find_match_boundaries, has_health_bar,
)

ROI_KEYS = ["p1_health", "p2_health", "p1_stun", "p2_stun"]


# ---------------------------------------------------------------------------
def build_health_signal(frames: np.ndarray | None,
                        cfg: Config) -> np.ndarray | None:
    """Return a (T,) bool array: True where health bars are visible.

    Returns None if neither frames nor a health array is available.
    """
    if frames is not None:
        roi = {k: getattr(cfg, k + "_roi") for k in ROI_KEYS}
        return np.array([has_health_bar(f, roi) for f in frames],
                        dtype=bool)
    return None


def build_health_signal_from_buttons(buttons: np.ndarray) -> np.ndarray:
    """Fallback: approximate gameplay frames from button activity."""
    return health_signal_from_inputs(buttons)


# ---------------------------------------------------------------------------
def save_match(output_dir: Path, stem: str, idx: int,
               flows: np.ndarray, movements: np.ndarray,
               buttons: np.ndarray, hp1: np.ndarray,
               hp2: np.ndarray, start: int, end: int):
    """Save one match window to a .npz file."""
    # flows has one fewer frame than actions
    f_start = max(0, start - 1)      # flow[i] covers frame i → i+1
    f_end   = min(flows.shape[0], end)

    out = output_dir / f"{stem}_match{idx:02d}.npz"
    np.savez_compressed(
        out,
        flows=flows[f_start:f_end],
        movement=movements[start:end],
        buttons=buttons[start:end],
        health_p1=hp1[start:end],
        health_p2=hp2[start:end],
    )
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Process Fightcade replays into per-match training data")
    p.add_argument("--replay-dir", type=Path, default=Path("data/replays"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Directory tree of pre-captured frames  "
                        "(frames-dir/<replay_stem>/frame_*.png).  "
                        "If absent, matches are split via input-activity heuristic.")
    args = p.parse_args()

    cfg = Config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replay_files = sorted(args.replay_dir.glob("*.fc2"))
    if not replay_files:
        print("No .fc2 files found in", args.replay_dir)
        return

    total_matches = 0

    for rp in tqdm(replay_files, desc="Replays"):
        # ---- parse ----
        try:
            meta, inputs = parse_replay(rp)
        except Exception as e:
            tqdm.write(f"  [skip] {rp.name}: {e}")
            continue

        p1_stream, p2_stream = inputs[0], inputs[1]
        n_frames = meta.num_frames
        if n_frames < 120:
            continue

        # ---- load frames (optional) ----
        frames = None
        hp1_raw = np.zeros(n_frames, dtype=np.float32)
        hp2_raw = np.zeros(n_frames, dtype=np.float32)

        if args.frames_dir:
            root = args.frames_dir / rp.stem
            frame_files = sorted(root.glob("frame_*.png"))
            if frame_files:
                frames_bgr = []
                for f in frame_files:
                    img = cv2.imread(str(f))
                    if img is not None:
                        frames_bgr.append(img)
                if frames_bgr:
                    # Resize to native for HUD extraction
                    native_frames = np.array([
                        cv2.resize(f, (cfg.native_width, cfg.native_height))
                        for f in frames_bgr
                    ])
                    # Greyscale version for optical flow
                    frames = np.array([
                        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                   (cfg.frame_size, cfg.frame_size))
                        for f in frames_bgr
                    ])
                    # Extract health from native frames
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

        # ---- actions ----
        movements = np.zeros(n_frames, dtype=np.int64)
        buttons_arr = np.zeros((n_frames, 6), dtype=np.float32)
        for i in range(min(n_frames, len(p1_stream))):
            d, bs = bitfield_to_action(p1_stream[i])
            movements[i] = int(d)
            for b in bs:
                buttons_arr[i, int(b)] = 1.0

        # ---- match-boundary detection ----
        if frames is not None:
            signal = build_health_signal(frames, cfg)
        else:
            signal = build_health_signal_from_buttons(buttons_arr)

        if signal is None or signal.sum() < 120:
            # Save whole file as one match if we can't split
            out = args.output_dir / f"{rp.stem}.npz"
            np.savez_compressed(out, flows=flows, movement=movements,
                                buttons=buttons_arr,
                                health_p1=hp1_raw, health_p2=hp2_raw)
            tqdm.write(f"  {out.name}  ({n_frames} f, unsplit)")
            total_matches += 1
            continue

        boundaries = find_match_boundaries(signal)
        if not boundaries:
            out = args.output_dir / f"{rp.stem}.npz"
            np.savez_compressed(out, flows=flows, movement=movements,
                                buttons=buttons_arr,
                                health_p1=hp1_raw, health_p2=hp2_raw)
            tqdm.write(f"  {out.name}  ({n_frames} f, no matches found)")
            total_matches += 1
            continue

        for mi, (start, end) in enumerate(boundaries):
            out = save_match(args.output_dir, rp.stem, mi,
                             flows, movements, buttons_arr,
                             hp1_raw, hp2_raw, start, end)
        total_matches += len(boundaries)
        tqdm.write(f"  {rp.stem} → {len(boundaries)} matches  "
                   f"(trimmed {n_frames}→{sum(e-s for s,e in boundaries)} frames)")

    print(f"\nDone — {total_matches} matches saved to {args.output_dir}")


if __name__ == "__main__":
    main()
