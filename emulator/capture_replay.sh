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

set -uo pipefail

CHALLENGE_ID="${1:?Usage: $0 <challenge_id> [game]}"
GAME="${2:-sfiii3nr1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EMU_DIR="$HOME/fightcade-emu"
WINEPREFIX="$HOME/.fightcade-wine"
OUT_DIR="$PROJECT_DIR/data/captures/$CHALLENGE_ID"
TOKEN="st_rnn_$$"

export WINEPREFIX WINEDEBUG=-all
mkdir -p "$OUT_DIR"

EMU_PID=""
VID_PID=""
_cleanup() {
    echo ""
    echo "Cleaning up …"
    [ -n "$VID_PID" ] && kill $VID_PID 2>/dev/null || true
    [ -n "$EMU_PID" ] && kill $EMU_PID 2>/dev/null || true
    wait 2>/dev/null || true
    # Collect input log if it exists
    if [ -f "$EMU_DIR/input_log.txt" ]; then
        cp "$EMU_DIR/input_log.txt" "$OUT_DIR/input_log.txt"
    fi
}
trap _cleanup EXIT

echo "=== Capture $CHALLENGE_ID ($GAME) ==="

# ---- Lua input logger ----
cp "$SCRIPT_DIR/input_log.lua" "$EMU_DIR/"
rm -f "$EMU_DIR/input_log.txt"

# ---- Launch emulator ----
echo "[1/3] Launching emulator (gamescope) …"
gamescope \
    -W 768 -H 448 -r 60 \
    -- wine "$EMU_DIR/fcadefbneo.exe" \
    "quark:stream,$GAME,$CHALLENGE_ID.2,7100" \
    -q -lua input_log.lua &
EMU_PID=$!
sleep 5

# ---- Screen capture via portal ----
echo "[2/3] Starting screen capture …"
PWNODE=""
if python3 "$SCRIPT_DIR/start_screencast.py" "$TOKEN" > /tmp/st_rnn_node 2>/tmp/st_rnn_err; then
    PWNODE=$(cat /tmp/st_rnn_node)
    echo "  pipewire node: $PWNODE"
else
    echo "  WARNING: screencast failed — no video will be captured"
    cat /tmp/st_rnn_err >&2
    PWNODE=""
fi

# ---- Video + Audio via pw-v4l2 + ffmpeg ----
HWENC="libx264"
HWOPT="-preset ultrafast -crf 23"
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_vaapi; then
    HWENC="h264_vaapi"
    HWOPT="-b:v 200k"
fi

if [ -n "$PWNODE" ]; then
    echo "[3/3] Recording video+audio …"
    pw-v4l2 \
        ffmpeg -y \
            -f v4l2 -thread_queue_size 128 -i /dev/video0 \
            -f pulse -i default \
            -map 0:v \
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
else
    echo "[3/3] Recording audio only …"
    pw-record --rate 16000 --channels 1 --format u8 \
        "$OUT_DIR/audio.wav" &
    VID_PID=$!
fi

# ---- Wait ----
echo "Recording — press Ctrl+C when replay ends, or wait for emulator to exit …"
wait $EMU_PID 2>/dev/null || true
