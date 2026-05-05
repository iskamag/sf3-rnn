"""Extract health, stun, super-meter and timer from SF3 HUD regions."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, replace


@dataclass(slots=True)
class GameState:
    p1_health: float = 1.0
    p2_health: float = 1.0
    p1_stun: float = 0.0
    p2_stun: float = 0.0
    p1_super: float = 0.0
    p2_super: float = 0.0
    round_timer: int = 99
    p1_wins: int = 0
    p2_wins: int = 0

    def to_tensor(self) -> "torch.Tensor":  # deferred import
        import torch
        return torch.tensor([
            self.p1_health, self.p2_health,
            self.p1_stun, self.p2_stun,
            self.p1_super, self.p2_super,
            self.round_timer / 99.0,
            self.p1_wins / 2.0, self.p2_wins / 2.0,
        ], dtype=torch.float32)


@dataclass(slots=True)
class WinTracker:
    """Track round wins from observable state transitions.

    Live PPO does not have the replay logger's memory-backed metadata, so we
    derive round wins from KO / round-reset transitions and carry them forward
    in GameState.
    """

    p1_wins: int = 0
    p2_wins: int = 0
    round_over: bool = False
    ko_threshold: float = 0.01
    full_health_threshold: float = 0.98
    match_wins: int = 2

    def reset(self):
        self.p1_wins = 0
        self.p2_wins = 0
        self.round_over = False

    def update(self, state: "GameState") -> "GameState":
        if (state.p1_health >= self.full_health_threshold
                and state.p2_health >= self.full_health_threshold):
            if self.p1_wins >= self.match_wins or self.p2_wins >= self.match_wins:
                self.p1_wins = 0
                self.p2_wins = 0
            self.round_over = False

        if not self.round_over:
            if state.p1_health <= self.ko_threshold and state.p2_health > self.ko_threshold:
                self.p2_wins = min(self.p2_wins + 1, self.match_wins)
                self.round_over = True
            elif state.p2_health <= self.ko_threshold and state.p1_health > self.ko_threshold:
                self.p1_wins = min(self.p1_wins + 1, self.match_wins)
                self.round_over = True

        return replace(state, p1_wins=self.p1_wins, p2_wins=self.p2_wins)


# ---------------------------------------------------------------------------
# Health bar reading
# ---------------------------------------------------------------------------
def _colour_mask_ratio(bar: np.ndarray,                      # (h, w, 3) BGR
                       lower: tuple,
                       upper: tuple) -> float:
    """Fraction of pixels inside *bar* that fall within the [lower, upper]
    HSV colour range."""
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask.sum() / bar.size * 3  # bar.size = h * w * 3


# Health-bar colours in SF3 are warm (green → yellow → red).
# We catch all of them with a wide saturation/value window and a hue envelope.
_HEALTH_LOWER = np.array([10, 80, 80])    # H, S, V
_HEALTH_UPPER = np.array([25, 255, 255])

# Stun bar is white → grey gradient
_STUN_LOWER = np.array([0, 0, 180])
_STUN_UPPER = np.array([180, 30, 255])

# Super bar is blue
_SUPER_LOWER = np.array([100, 80, 80])
_SUPER_UPPER = np.array([130, 255, 255])


def read_health(bar: np.ndarray) -> float:
    """Return health fraction (0 – 1) from a cropped health-bar image."""
    return _colour_mask_ratio(bar, _HEALTH_LOWER, _HEALTH_UPPER)


def read_stun(bar: np.ndarray) -> float:
    return _colour_mask_ratio(bar, _STUN_LOWER, _STUN_UPPER)


def read_super(bar: np.ndarray) -> float:
    return _colour_mask_ratio(bar, _SUPER_LOWER, _SUPER_UPPER)


# ---------------------------------------------------------------------------
# High-level extraction
# ---------------------------------------------------------------------------

def has_health_bar(frame: np.ndarray, roi: dict,
                   threshold: float = 0.08) -> bool:
    """Quick check — are health bars visible on this frame?  True during
    gameplay, False during character select, intros, and victory screens."""
    def crop(r):
        x, y, w, h = r
        return frame[y:y + h, x:x + w]
    p1 = read_health(crop(roi["p1_health"]))
    p2 = read_health(crop(roi["p2_health"]))
    return (p1 > threshold) and (p2 > threshold)


# ---------------------------------------------------------------------------
# Audio-based match detection — cross-correlates a template .wav against
# captured audio to find match boundaries without any video or health logic.
# ---------------------------------------------------------------------------
_SIGNATURE_LEN = 256  # first N samples used as quick-reject prefix


def _find_template_positions(audio: "np.ndarray",
                             template: "np.ndarray",
                             ) -> "np.ndarray":
    """Return sorted sample offsets where *template* appears verbatim in *audio*.

    Uses a short signature prefix for O(n) early rejection; only runs full
    comparison at candidate positions.
    """
    sig_len = min(_SIGNATURE_LEN, len(template))
    sig = template[:sig_len]
    tlen = len(template)
    alen = len(audio)
    hits = []

    i = 0
    limit = alen - tlen + 1
    while i < limit:
        # Fast scan: skip until signature prefix matches.
        if not np.array_equal(audio[i:i + sig_len], sig):
            i += 1
            continue
        # Full comparison at candidate position.
        if np.array_equal(audio[i:i + tlen], template):
            hits.append(i)
            i += tlen  # skip past the match to avoid overlapping hits
        else:
            i += 1
    return np.array(hits, dtype=np.int64)


def _find_match_end_frames(p1w: "np.ndarray",
                           p2w: "np.ndarray",
                           match_wins: int = 2,
                           ) -> "np.ndarray":
    """Return exclusive end frames for matches once a player reaches match point."""
    ends = []
    for i in range(1, len(p1w)):
        prev_best = max(int(p1w[i - 1]), int(p2w[i - 1]))
        curr_best = max(int(p1w[i]), int(p2w[i]))
        if curr_best >= match_wins and curr_best > prev_best:
            ends.append(i + 1)
    return np.array(ends, dtype=np.int64)


def audio_match_signal(audio_path: "Path",
                       template_path: "Path",
                       p1w: "np.ndarray | None" = None,
                       p2w: "np.ndarray | None" = None,
                       target_sr: int = 16000,
                       fps: int = 60,
                       start_offset_frames: int = 0,
                       min_match_len: int = 120,
                       match_wins: int = 2,
                       ) -> "list[tuple[int, int]]":
    """Return (start, end) frame pairs.

    Match starts come from audio template peaks plus a fixed frame offset;
    match ends come from the first frame where either player reaches the
    final round-win count.
    Unmatched starts are dropped.
    """
    import wave

    def _load_u8(path: "Path") -> "np.ndarray":
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            ch = wf.getnchannels()
        samples = np.frombuffer(raw, dtype=np.uint8)
        if ch == 2:
            samples = samples.reshape(-1, 2).mean(axis=1).astype(np.uint8)
        if sr != target_sr:
            ratio = target_sr / sr
            n_out = int(len(samples) * ratio)
            samples = np.interp(np.linspace(0, len(samples) - 1, n_out),
                                np.arange(len(samples)), samples)
            samples = np.clip(np.rint(samples), 0, 255).astype(np.uint8)
        return samples

    template = _load_u8(template_path)
    audio = _load_u8(audio_path)

    spf = target_sr / fps
    offsets = _find_template_positions(audio, template)
    starts = np.rint(offsets / spf).astype(int)
    if start_offset_frames:
        starts = starts + int(start_offset_frames)
    frame_limit = None
    if p1w is not None:
        frame_limit = len(p1w)
    elif p2w is not None:
        frame_limit = len(p2w)
    if frame_limit is not None:
        starts = starts[(starts >= 0) & (starts < frame_limit)]

    if p1w is None or p2w is None:
        return []

    ends = _find_match_end_frames(p1w, p2w, match_wins=match_wins)

    # Pair each start with the next match-completing win transition.
    boundaries: list[tuple[int, int]] = []
    ei = 0
    for s in starts:
        while ei < len(ends) and ends[ei] <= s:
            ei += 1
        if ei >= len(ends):
            break
        e = int(ends[ei])
        ei += 1
        if e - s >= min_match_len:
            boundaries.append((int(s), e))
    return boundaries


def extract_game_state(frame: np.ndarray,                     # (H, W, 3) BGR
                       roi: dict,
                       ) -> GameState:
    """Extract full GameState from a native-resolution frame."""
    def crop(r):
        x, y, w, h = r
        return frame[y:y + h, x:x + w]

    return GameState(
        p1_health=read_health(crop(roi["p1_health"])),
        p2_health=read_health(crop(roi["p2_health"])),
        p1_stun=read_stun(crop(roi["p1_stun"])),
        p2_stun=read_stun(crop(roi["p2_stun"])),
    )
