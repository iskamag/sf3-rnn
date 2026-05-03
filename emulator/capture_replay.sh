#!/usr/bin/env bash
# capture_replay.sh — stream a Fightcade replay, capture video + audio + inputs
#
# Usage:
#   ./emulator/capture_replay.sh <challenge_id> [game]
#
# Output (data/captures/<challenge_id>/):
#   video.mkv      — 128×128 grayscale H.264, 60 fps
#   audio.wav      — 8-bit mono 16 kHz PCM
#   input_log.txt  — per-frame inputs + health from Lua

set -euo pipefail

CHALLENGE_ID="${1:?Usage: $0 <challenge_id> [game]}"
GAME="${2:-sfiii3nr1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EMU_DIR="$HOME/fightcade-emu"
WINEPREFIX="$HOME/.fightcade-wine"
OUT_DIR="$PROJECT_DIR/data/captures/$CHALLENGE_ID"

export WINEPREFIX WINEDEBUG=-all
mkdir -p "$OUT_DIR"

echo "=== Capture $CHALLENGE_ID ($GAME) ==="

# ---- Lua input logger ----
cp "$SCRIPT_DIR/input_log.lua" "$EMU_DIR/"
rm -f "$EMU_DIR/input_log.txt"

# ---- Launch emulator ----
echo "[1/4] Launching emulator (gamescope) …"
gamescope \
    -W 768 -H 448 -r 60 \
    -- wine "$EMU_DIR/fcadefbneo.exe" \
    "quark:stream,$GAME,$CHALLENGE_ID.2,7100" \
    -q -lua input_log.lua &
EMU_PID=$!
sleep 4

# ---- PipeWire screen capture ----
echo "[2/4] Starting screen capture (accept the dialog) …"
PWNODE=$(python3 "$SCRIPT_DIR/start_screencast.py")
echo "  pipewire node: $PWNODE"

# ---- Video + Audio via pw-v4l2 + ffmpeg ----
echo "[3/4] Recording video+audio (128×128 gray VAAPI, 16kHz u8) …"
HWENC="libx264"
HWOPT="-preset ultrafast -crf 23"
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_vaapi; then
    HWENC="h264_vaapi"
    HWOPT="-b:v 200k"
fi

# pw-v4l2 wraps ffmpeg so V4L2 calls go through PipeWire to the portal node
pw-v4l2 \
    ffmpeg -y \
        -f v4l2 -thread_queue_size 128 -i /dev/video0 \
        -f pulse -i default \
        -map 0:v -map 1:a \
        -vf "scale=128:128:flags=bilinear,format=gray,fps=60" \
        -c:v "$HWENC" $HWOPT \
        -an \
        "$OUT_DIR/video.mkv" \
        \
        -map 1:a \
        -vn \
        -c:a pcm_u8 -ar 16000 -ac 1 \
        "$OUT_DIR/audio.wav" \
    &>/dev/null &
VID_PID=$!

# ---- Wait for replay ----
echo "Recording — waiting for replay to end …"
wait $EMU_PID 2>/dev/null || true
kill $VID_PID 2>/dev/null || true
sleep 1

# ---- Collect input log ----
if [ -f "$EMU_DIR/input_log.txt" ]; then
    cp "$EMU_DIR/input_log.txt" "$OUT_DIR/input_log.txt"
    echo ""
    echo "=== Done ==="
    echo "  Inputs:  $(wc -l < "$OUT_DIR/input_log.txt") frames"
    [ -f "$OUT_DIR/audio.wav" ] && echo "  Audio:   $(du -h "$OUT_DIR/audio.wav" | cut -f1)"
    [ -f "$OUT_DIR/video.mkv" ] && echo "  Video:   $(du -h "$OUT_DIR/video.mkv" | cut -f1)"
    echo ""
    echo "Next:  python scripts/collect_data.py --capture-dir data/captures \\"
    echo "           --output-dir data/processed"
else
    echo "ERROR: input_log.txt not found"
    exit 1
fi
