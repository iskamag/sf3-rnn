"""Fightcade 2 replay (.fc2) parser.

Fightcade 2 replays are zlib-compressed JSON containers.  The format
(version-dependent) stores:
  - game / rom info
  - player names & characters
  - frame-by-frame input bitfields for both players

This module extracts the input streams and character selections so they can
be paired with screen captures to build training data.

NOTE: The .fc2 format has evolved across Fightcade versions.  If parsing
fails try `fc2json` (https://github.com/eezstreet/fc2json) first to
convert to plain JSON, then pass the .json path here.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from model.action_space import bitfield_to_action, Direction, Button


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ReplayMeta:
    game: str                   # e.g. "sfiii3"
    players: List[str]          # player names
    characters: List[str]       # "RYU", "KEN", etc.
    num_frames: int


def parse_replay(path: Path) -> Tuple[ReplayMeta, List[List[int]]]:
    """Parse a Fightcade .fc2 replay.

    Returns
    -------
    meta : ReplayMeta
    inputs : List[List[int]]
        inputs[i][j] = input bitfield for player j at frame i.
    """
    raw = path.read_bytes()
    return _parse_fc2_bytes(raw)


def extract_full_inputs(path: Path) -> Tuple[ReplayMeta,
                                             List[List[Tuple[Direction,
                                                             List[Button]]]]]:
    """Convenience: parse replay and decode all bitfields to actions."""
    meta, raw_inputs = parse_replay(path)
    actions = []
    for frame_inputs in raw_inputs:
        frame_actions = [bitfield_to_action(bits) for bits in frame_inputs]
        actions.append(frame_actions)
    return meta, actions


# ---------------------------------------------------------------------------
# Internal parsing
# ---------------------------------------------------------------------------
def _parse_cps3_input_stream(data: bytes) -> List[int]:
    """Decode a CPS3 input stream from the replay's raw bytes.

    Fightcade stores input frames as a run-length encoded stream where each
    entry is either a single frame's bitfield or a repeated run [count, bits].
    The exact encoding depends on the replay version; this is the common variant.

    Returns a list of per-frame bitfield integers.
    """
    inputs = []
    i = 0
    n = len(data)

    while i < n:
        b = data[i]
        i += 1
        if b == 0xFF:                                    # run-length marker
            if i + 1 >= n:
                break
            count = data[i]
            bits = data[i + 1]
            i += 2
            inputs.extend([bits] * count)
        elif b == 0xFE:                                  # single frame marker
            if i >= n:
                break
            bits = data[i]
            i += 1
            inputs.append(bits)
        else:
            # Plain byte — assumes count=1, bits=b for simple streams
            inputs.append(b)

    return inputs


def _parse_fc2_bytes(raw: bytes) -> Tuple[ReplayMeta, List[List[int]]]:
    # Fightcade 2 wraps a JSON header + zlib-compressed input data
    # into a single binary blob.  Try to decompress and find JSON.

    try:
        # The first 4 bytes of a .fc2 are often the uncompressed JSON length
        json_len = struct.unpack("<I", raw[:4])[0]
        json_bytes = raw[4:4 + json_len]
        compressed = raw[4 + json_len:]
    except Exception:
        # If the header format differs, try treating the whole file as
        # JSON with an inline compressed block.
        json_bytes = raw
        compressed = b""

    info = json.loads(json_bytes)

    # Extract metadata
    game = info.get("game", info.get("rom", "sfiii3"))
    players = info.get("players", info.get("player_names", ["P1", "P2"]))
    characters = info.get("characters", info.get("chars", ["UNKNOWN", "UNKNOWN"]))
    num_frames = info.get("num_frames", info.get("total_frames", 0))

    meta = ReplayMeta(
        game=game,
        players=players,
        characters=characters,
        num_frames=num_frames,
    )

    if compressed:
        try:
            decompressed = zlib.decompress(compressed, -15)
        except zlib.error:
            decompressed = zlib.decompress(compressed)

        # Split decompressed data into two player streams.
        # Convention: each frame is a 2-byte pair (P1 bits, P2 bits).
        p1_inputs = []
        p2_inputs = []
        for i in range(0, len(decompressed) - 1, 2):
            p1_inputs.append(decompressed[i])
            p2_inputs.append(decompressed[i + 1])
    else:
        # Plain JSON has pre-parsed input_streams
        input_streams = info.get("input_streams", [])
        if input_streams:
            p1_inputs = _parse_cps3_input_stream(bytes(input_streams[0])) if input_streams else []
            p2_inputs = _parse_cps3_input_stream(bytes(input_streams[1])) if len(input_streams) > 1 else []
        else:
            p1_inputs, p2_inputs = [], []

    return meta, [p1_inputs, p2_inputs]
