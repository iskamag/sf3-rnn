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


def _find_template_positions(audio: "np.ndarray",
                             template: "np.ndarray",
                             corr_threshold: float = 0.5,
                             ) -> "np.ndarray":
    """Return sorted sample offsets where *template* cross-correlates above threshold.

    Uses FFT-based normalised cross-correlation (O(N log N)), then suppresses
    overlapping detections (keeps only the best within a template-length window).
    """
    tlen = len(template)
    alen = len(audio)
    if tlen == 0 or alen < tlen:
        return np.array([], dtype=np.int64)

    # Centre both signals (DC removal).
    ta = template.astype(np.float64) - 128.0
    aa = audio.astype(np.float64) - 128.0

    # FFT-based cross-correlation.
    n_fft = int(2 ** np.ceil(np.log2(alen + tlen - 1)))
    FA = np.fft.rfft(aa, n=n_fft)
    FT = np.fft.rfft(ta[::-1], n=n_fft)
    corr_raw = np.fft.irfft(FA * FT, n=n_fft)

    # Trim to valid range: corr_raw[k] corresponds to shift = k - (tlen - 1).
    valid = corr_raw[tlen - 1:alen]

    # Normalise by local audio energy (window = template length).
    a2 = aa * aa
    cum_a2 = np.concatenate([[0.0], np.cumsum(a2)])
    window_sums = np.empty(alen - tlen + 1, dtype=np.float64)
    for i in range(len(window_sums)):
        window_sums[i] = cum_a2[i + tlen] - cum_a2[i]
    local_norm = np.sqrt(np.maximum(window_sums, 1e-10))
    template_norm = np.linalg.norm(ta)
    denom = template_norm * local_norm
    corr = valid / np.maximum(denom, 1e-10)

    # Find peaks above threshold.
    above = corr > corr_threshold
    if not above.any():
        return np.array([], dtype=np.int64)

    # Non-maximum suppression: slide a template-length window, keep best peak.
    hits = []
    i = 0
    while i < len(corr):
        if not above[i]:
            i += 1
            continue
        end = min(i + tlen, len(corr))
        window_corr = corr[i:end]
        peak = i + int(np.argmax(window_corr))
        if corr[peak] > corr_threshold:
            hits.append(peak)
        i = peak + tlen
    return np.array(hits, dtype=np.int64)




def audio_match_signal(audio_path: "Path",
                       template_path: "Path",
                       p1w: "np.ndarray | None" = None,
                       p2w: "np.ndarray | None" = None,
                       target_sr: int = 16000,
                       fps: int = 60,
                       start_offset_frames: int = 0,
                       input_offset_frames: int = 0,
                       min_match_len: int = 120,
                       corr_threshold: float = 0.5,
                       ) -> "list[tuple[int, int]]":
    """Return (start, end) frame pairs in *input-log* frame space.

    Each audio template hit starts a new match; the previous match ends
    just before the next hit (or at end-of-data for the last one).

    *input_offset_frames* accounts for the video/audio recording starting
    later than the input log — it is added to audio-derived starts so they
    align with input-log frame indices.
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

    offsets = _find_template_positions(audio, template, corr_threshold=corr_threshold)
    spf = target_sr / fps  # samples per frame

    # Convert audio sample offsets → input-log frame starts.
    starts = np.rint(offsets / spf).astype(int)
    if start_offset_frames:
        starts = starts + int(start_offset_frames)
    starts = starts + int(input_offset_frames)

    frame_limit = None
    if p1w is not None:
        frame_limit = len(p1w)
    elif p2w is not None:
        frame_limit = len(p2w)
    if frame_limit is not None:
        starts = starts[(starts >= 0) & (starts < frame_limit)]

    if p1w is None or p2w is None:
        return []

    # Collect every win-counter change as a potential match-end.
    transitions: list[int] = []
    if p1w is not None and p2w is not None:
        for j in range(1, len(p1w)):
            if int(p1w[j]) != int(p1w[j - 1]) or int(p2w[j]) != int(p2w[j - 1]):
                transitions.append(j)

    # Each template hit starts a new match.  The match ends at either the
    # next template hit or the first win-counter change that falls inside
    # the match — whichever comes first.
    boundaries: list[tuple[int, int]] = []
    ti = 0  # index into transitions
    for i, s_frame in enumerate(starts):
        s_frame = int(s_frame)
        if i + 1 < len(starts):
            next_start = int(starts[i + 1])
        else:
            next_start = len(p1w)  # end of data

        # Advance transition pointer past this start.
        while ti < len(transitions) and transitions[ti] <= s_frame:
            ti += 1

        # Pick the earlier of next template start and first win transition.
        if ti < len(transitions) and transitions[ti] < next_start:
            e_frame = transitions[ti]
            ti += 1
        else:
            e_frame = next_start

        if e_frame - s_frame >= min_match_len:
            boundaries.append((s_frame, e_frame))

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
