"""Audio capture backend — grabs system audio in a background thread.

Uses `sounddevice` (portaudio).  Provide a *callback* that receives raw
float32 mono samples at 16 kHz.  The callback usually feeds an AudioBuffer.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np


class AudioCapture:
    """Continuously capture system audio in a background thread.

    Parameters
    ----------
    callback :  called with `(samples: np.ndarray)` where samples is a
                float32 mono buffer (length varies).
    sample_rate : int
    device : str | int | None — audio input device index or name
    """

    def __init__(self,
                 callback: Callable[[np.ndarray], None],
                 sample_rate: int = 16000,
                 device: Optional[int] = None,
                 ):
        self._callback = callback
        self._sr = sample_rate
        self._device = device
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        import sounddevice as sd
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self._sr, channels=1,
            dtype="float32", device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata: np.ndarray,
                        frames: int, time_info, status):
        if self._running:
            self._callback(indata[:, 0].copy())  # mono


# ---------------------------------------------------------------------------
# Audio capture stub
# ---------------------------------------------------------------------------
class AudioCaptureStub:
    def start(self): pass
    def stop(self): pass
