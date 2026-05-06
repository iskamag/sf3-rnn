"""Per-frame audio feature extraction for SF3.

Each ~16 ms window (256 samples @ 16 kHz, close enough to 1/60 s) produces
a 15-d vector:  13 MFCCs + 1 RMS energy + 1 spectral centroid.

Using 256 (= 2⁸) keeps the FFT radix-2 and avoids padding. Each window spans
about 16 ms at 16 kHz.
"""

from __future__ import annotations

import numpy as np

FFT_SIZE: int = 256


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_features(samples: np.ndarray,           # (N,) float32 mono
                     sample_rate: int = 16000,
                     n_mfcc: int = 13,
                     ) -> np.ndarray:               # (15,) float32
    """Compute a 15-d audio feature vector from ~256 raw samples.

    Internally pads or truncates to *FFT_SIZE* (256) so FFT is radix-2.
    """
    n = len(samples)
    if n < 16:
        return np.zeros(n_mfcc + 2, dtype=np.float32)

    # Pad / truncate to FFT_SIZE
    frame = np.zeros(FFT_SIZE, dtype=np.float32)
    window = np.hanning(FFT_SIZE)
    keep = min(n, FFT_SIZE)
    frame[:keep] = samples[:keep].astype(np.float32) * window[:keep]

    # FFT
    fft = np.fft.rfft(frame, n=FFT_SIZE)
    mag = np.abs(fft)
    power = mag ** 2

    # Frequency bins (skip DC)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)[1:]
    mag = mag[1:]
    power = power[1:]

    # ---- MFCC (simplified: mel filterbank + DCT) ----
    n_mels = 40
    mel_energies = _mel_filterbank(power, freqs, n_mels, sample_rate)
    log_mel = np.log(mel_energies + 1e-10)

    # DCT-II (scipy is overkill — manual)
    mfccs = _dct(log_mel, n_mfcc)                  # (13,)

    # ---- RMS energy (dB-scaled) ----
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    rms_db = 20.0 * np.log10(rms + 1e-10)
    rms_db = np.clip(rms_db, -60.0, 0.0) / 60.0 + 1.0   # map [-60,0] → [0, 1]

    # ---- Spectral centroid ----
    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)
    centroid_norm = np.clip(centroid / (sample_rate / 2), 0.0, 1.0)

    feats = np.concatenate([mfccs, [rms_db, centroid_norm]])
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mel_filterbank(power: np.ndarray,
                    freqs: np.ndarray,
                    n_mels: int,
                    sr: int,
                    ) -> np.ndarray:
    """Apply a triangular mel filterbank to a power spectrum."""
    mel_low = _hz_to_mel(0)
    mel_high = _hz_to_mel(sr / 2)
    mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fbank = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        for k, f in enumerate(freqs):
            if left <= f < centre:
                fbank[m, k] = (f - left) / (centre - left + 1e-10)
            elif centre <= f <= right:
                fbank[m, k] = (right - f) / (right - centre + 1e-10)
    return fbank @ power


def _dct(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    """DCT-II of a 1-D array, returning the first *n_coeffs*."""
    N = len(x)
    y = np.zeros(n_coeffs, dtype=np.float32)
    for k in range(n_coeffs):
        y[k] = 2.0 * np.sum(x * np.cos(np.pi * k * (np.arange(N) + 0.5) / N))
    return y


def dct_batch(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    """DCT-II for batched input.  x: (N, M) → (N, n_coeffs)."""
    N, M = x.shape
    k = np.arange(n_coeffs, dtype=np.float32)
    n = np.arange(M, dtype=np.float32)
    cos_mat = np.cos(np.pi * k[:, None] * (n[None, :] + 0.5) / M)
    return 2.0 * (x @ cos_mat.T.astype(np.float32))


_MEL_FB_CACHE: dict[int, np.ndarray] = {}


def get_mel_filterbank(sample_rate: int = 16000) -> np.ndarray:
    """Return the (40, FFT_SIZE//2) mel filterbank matrix, lazily cached."""
    if sample_rate not in _MEL_FB_CACHE:
        n_mels = 40
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)[1:]
        mel_low = _hz_to_mel(0)
        mel_high = _hz_to_mel(sample_rate / 2)
        mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_pts = _mel_to_hz(mel_pts)
        fbank = np.zeros((n_mels, len(freqs)), dtype=np.float32)
        for m in range(n_mels):
            left, centre, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            for k, f in enumerate(freqs):
                if left <= f < centre:
                    fbank[m, k] = (f - left) / (centre - left + 1e-10)
                elif centre <= f <= right:
                    fbank[m, k] = (right - f) / (right - centre + 1e-10)
        _MEL_FB_CACHE[sample_rate] = fbank
    return _MEL_FB_CACHE[sample_rate]


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
