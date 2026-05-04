from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FLOW_MIN = -1.0
FLOW_MAX = 1.0


def quantize_flow(flow: np.ndarray) -> np.ndarray:
    clipped = np.clip(flow, FLOW_MIN, FLOW_MAX)
    return np.rint((clipped - FLOW_MIN) * 255.0 / (FLOW_MAX - FLOW_MIN)).astype(np.uint8)


def dequantize_flow(flow_u8: np.ndarray) -> np.ndarray:
    return flow_u8.astype(np.float32) * ((FLOW_MAX - FLOW_MIN) / 255.0) + FLOW_MIN


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
