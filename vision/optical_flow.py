"""Optical-flow computation — ROI-based Farneback for sparse-content game frames."""

from __future__ import annotations

import cv2
import numpy as np
import torch


_FB_KWARGS = dict(
    pyr_scale=0.5, levels=1, winsize=7,
    iterations=1, poly_n=3, poly_sigma=1.2,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
)


def compute_flow(prev_gray: np.ndarray,
                 curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two greyscale frames.

    Only runs Farneback on motion regions (absdiff > threshold); static
    areas stay at zero.  For SF3 with mostly static backgrounds this
    processes only 15–30 % of pixels.

    Parameters
    ----------
    prev_gray, curr_gray : np.ndarray  (H, W) uint8

    Returns
    -------
    flow : np.ndarray  (2, H, W) float32  —  [0] = dx, [1] = dy
    """
    H, W = prev_gray.shape

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    frac = mask.sum() / (255.0 * H * W)
    if frac > 0.7:                                          # screen-filling motion
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **_FB_KWARGS)
        return np.ascontiguousarray(flow.transpose(2, 0, 1), dtype=np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flow = np.zeros((H, W, 2), dtype=np.float32)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 64:                                     # skip tiny regions
            continue
        # Expand ROI to avoid edge artefacts
        x0 = max(0, x - 4)
        y0 = max(0, y - 4)
        x1 = min(W, x + w + 4)
        y1 = min(H, y + h + 4)

        roi = cv2.calcOpticalFlowFarneback(
            prev_gray[y0:y1, x0:x1], curr_gray[y0:y1, x0:x1],
            None, **_FB_KWARGS)
        flow[y0:y1, x0:x1] = roi

    return np.ascontiguousarray(flow.transpose(2, 0, 1), dtype=np.float32)


def preprocess_flow(flow: np.ndarray,
                    target_width: int = 160,
                    target_height: int = 120,
                    norm_factor: float = 20.0) -> torch.Tensor:
    """Resize optical-flow field and normalise.

    Parameters
    ----------
    flow : np.ndarray  (2, H, W) float32
    target_width, target_height : int
    norm_factor : float   divide pixel displacement by this value

    Returns
    -------
    torch.Tensor  (2, target_height, target_width)
    """
    # cv2.resize expects (H, W, C)
    resized = cv2.resize(flow.transpose(1, 2, 0),
                         (target_width, target_height),
                         interpolation=cv2.INTER_LINEAR)
    out = resized.transpose(2, 0, 1) / norm_factor
    out = np.clip(out, -1.0, 1.0)
    return torch.from_numpy(out.astype(np.float32))


