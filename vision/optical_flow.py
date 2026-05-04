"""Dense optical-flow computation via Farneback (CPU or CUDA)."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def compute_flow(prev_gray: np.ndarray,
                 curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two greyscale frames.

    Parameters
    ----------
    prev_gray, curr_gray : np.ndarray  (H, W) uint8

    Returns
    -------
    flow : np.ndarray  (2, H, W) float32  —  [0] = dx, [1] = dy
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1,
        flags=0,
    )
    # cv2 returns (H, W, 2) → (2, H, W)
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
    return torch.from_numpy(out.astype(np.float32))


def frame_to_flow_stack(frames: np.ndarray,                 # (T, H, W) uint8
                        target_width: int = 160,
                        target_height: int = 120,
                        norm_factor: float = 20.0,
                        ) -> torch.Tensor:                   # (T-1, 2, H, W)
    """Convert a sequence of greyscale frames into a normalised flow stack."""
    T = len(frames)
    flows = []
    for i in range(T - 1):
        flow = compute_flow(frames[i], frames[i + 1])
        flows.append(preprocess_flow(flow, target_width, target_height, norm_factor))
    return torch.stack(flows, dim=0)  # (T-1, 2, H, W)
