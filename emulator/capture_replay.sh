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
echo "[2/3] Starting screen capture (accept the dialog) …"
PWNODE=""
python3 "$SCRIPT_DIR/start_screencast.py" > /tmp/st_rnn_node 2>/tmp/st_rnn_err || true
PWNODE=$(cat /tmp/st_rnn_node 2>/dev/null)
if [ -n "$PWNODE" ]; then
    echo "  pipewire node: $PWNODE"
else
    echo "  WARNING: screencast failed"
    echo "  stderr: $(head -c 200 /tmp/st_rnn_err 2>/dev/null)"
    PWNODE=""
fi

# ---- Video (gstreamer pipewiresrc → ffmpeg VAAPI) + Audio ----
echo "[3/3] Recording …"

HWENC="libx264"
HWOPT="-preset ultrafast -crf 23"
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_vaapi; then
    HWENC="h264_vaapi"
    HWOPT="-b:v 200k"
fi

# Audio: simple pw-record
pw-record --rate 16000 --channels 1 --format u8 \
    "$OUT_DIR/audio.wav" &
AUDIO_PID=$!

if [ -n "$PWNODE" ]; then
    echo "  Capturing video from node $PWNODE"
    # Gstreamer: pipewire → resize → gray → pipe to ffmpeg VAAPI
    gst-launch-1.0 -e \
        pipewiresrc path="$PWNODE" \
            ! video/x-raw,framerate=60/1 \
        ! videoscale \
        ! video/x-raw,width=128,height=128 \
        ! videoconvert \
        ! video/x-raw,format=GRAY8 \
        ! fdsink fd=1 \
        2>/tmp/st_rnn_gst_err \
        | ffmpeg -y -v warning \
            -f rawvideo -pixel_format gray -video_size 128x128 -framerate 60 -i - \
            -c:v "$HWENC" $HWOPT \
            -an \
            "$OUT_DIR/video.mkv" \
        2>/tmp/st_rnn_ffmpeg_err &
    VID_PID=$!
    sleep 1
    if ! kill -0 $VID_PID 2>/dev/null; then
        echo "  ERROR: capture process died immediately"
        echo "  gst: $(head -c 300 /tmp/st_rnn_gst_err 2>/dev/null)"
        echo "  ffmpeg: $(head -c 300 /tmp/st_rnn_ffmpeg_err 2>/dev/null)"
        VID_PID=""
    fi
fi

# ---- Wait ----
echo "Recording — press Ctrl+C when replay ends, or wait …"
wait $EMU_PID 2>/dev/null || true
[ -n "${AUDIO_PID:-}" ] && kill $AUDIO_PID 2>/dev/null || true
