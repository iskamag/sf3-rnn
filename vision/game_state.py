"""Audio-based match detection for BC data collection.

Cross-correlates a template .wav against captured audio to find match
start boundaries.  Uses FFT-based normalised cross-correlation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Audio-based match detection — cross-correlates a template .wav against
# captured audio to find match boundaries (used for BC data collection).
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

    ta = template.astype(np.float64) - 128.0
    aa = audio.astype(np.float64) - 128.0

    n_fft = int(2 ** np.ceil(np.log2(alen + tlen - 1)))
    FA = np.fft.rfft(aa, n=n_fft)
    FT = np.fft.rfft(ta[::-1], n=n_fft)
    corr_raw = np.fft.irfft(FA * FT, n=n_fft)

    valid = corr_raw[tlen - 1:alen]

    a2 = aa * aa
    cum_a2 = np.concatenate([[0.0], np.cumsum(a2)])
    window_sums = np.empty(alen - tlen + 1, dtype=np.float64)
    for i in range(len(window_sums)):
        window_sums[i] = cum_a2[i + tlen] - cum_a2[i]
    local_norm = np.sqrt(np.maximum(window_sums, 1e-10))
    template_norm = np.linalg.norm(ta)
    denom = template_norm * local_norm
    corr = valid / np.maximum(denom, 1e-10)

    above = corr > corr_threshold
    if not above.any():
        return np.array([], dtype=np.int64)

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
    spf = target_sr / fps

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

    transitions: list[int] = []
    if p1w is not None and p2w is not None:
        for j in range(1, len(p1w)):
            if int(p1w[j]) != int(p1w[j - 1]) or int(p2w[j]) != int(p2w[j - 1]):
                transitions.append(j)

    boundaries: list[tuple[int, int]] = []
    ti = 0
    for i, s_frame in enumerate(starts):
        s_frame = int(s_frame)
        if i + 1 < len(starts):
            next_start = int(starts[i + 1])
        else:
            next_start = len(p1w)

        while ti < len(transitions) and transitions[ti] <= s_frame:
            ti += 1

        if ti < len(transitions) and transitions[ti] < next_start:
            e_frame = transitions[ti]
            ti += 1
        else:
            e_frame = next_start

        if e_frame - s_frame >= min_match_len:
            boundaries.append((s_frame, e_frame))

    return boundaries
