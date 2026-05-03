#!/usr/bin/env bash
# capture_replay.sh — stream a Fightcade replay, capture video + audio + inputs
#
# Usage:
#   ./emulator/capture_replay.sh <challenge_id> [game]
#
# Output (in data/captures/<challenge_id>/):
#   capture.mkv     — H.264 video, 128x128 grayscale, 60 fps (VAAPI)
#   audio.wav       — 8-bit mono 16 kHz PCM
#   input_log.txt   — per-frame CPS3 bitfields + health from Lua
#
# If Wayland screen capture isn't available, set CAPTURE_VIDEO=0 and
# record the gamescope window manually with OBS/wf-recorder.

set -euo pipefail

CHALLENGE_ID="${1:?Usage: $0 <challenge_id> [game]}"
GAME="${2:-sfiii3nr1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EMU_DIR="$HOME/fightcade-emu"
WINEPREFIX="$HOME/.fightcade-wine"
OUT_DIR="$PROJECT_DIR/data/captures/$CHALLENGE_ID"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-1}"

export WINEPREFIX WINEDEBUG=-all
mkdir -p "$OUT_DIR"

echo "=== Capture $CHALLENGE_ID ($GAME) ==="
echo "  Output: $OUT_DIR"

# ---- Lua input logger ----
cp "$SCRIPT_DIR/input_log.lua" "$EMU_DIR/"
rm -f "$EMU_DIR/input_log.txt"

# ---- Launch emulator in gamescope ----
# Native CPS3 resolution is 384x224.  Scale 2x for legible capture window.
echo "[1/3] Launching emulator …"
gamescope \
    -W 768 -H 448 -r 60 \
    -- wine "$EMU_DIR/fcadefbneo.exe" \
    "quark:stream,$GAME,$CHALLENGE_ID.2,7100" \
    -q -lua input_log.lua &
EMU_PID=$!
sleep 4

# ---- Audio capture (8-bit 16 kHz mono WAV) ----
echo "[2/3] Starting audio capture …"
if command -v pw-record &>/dev/null; then
    pw-record --rate 16000 --channels 1 --format u8 \
        "$OUT_DIR/audio.wav" &
    AUDIO_PID=$!
elif command -v parec &>/dev/null; then
    parec --rate=16000 --channels=1 --format=s16le \
        | sox -t raw -r 16000 -e signed -b 16 -c 1 - \
              -b 8 -e unsigned "$OUT_DIR/audio.wav" &
    AUDIO_PID=$!
else
    echo "  WARNING: no audio capture tool (install pw-record or parec)"
    AUDIO_PID=""
fi

# ---- Video capture via pipewire + gstreamer VAAPI ----
if [ "$CAPTURE_VIDEO" = "1" ] && command -v gst-launch-1.0 &>/dev/null; then
    echo "[3/3] Starting video capture (128x128 gray, VAAPI) …"

    # Find the gamescope pipewire node (created when the window appears)
    # The node path changes each run; probe for it.
    SCRNODE=""
    for i in $(seq 1 15); do
        SCRNODE=$(pw-cli list-objects 2>/dev/null | \
            grep -B20 "node.name" | \
            grep -B1 "media.class.*Video/Source" | \
            grep "id " | tail -1 | awk '{print $2}' | tr -d ',')
        [ -n "$SCRNODE" ] && break
        sleep 1
    done

    if [ -n "$SCRNODE" ]; then
        # Gstreamer pipewire → resize+gray → VAAPI encode → mkv
        gst-launch-1.0 -e \
            pipewiresrc path="$SCRNODE" \
                ! video/x-raw,framerate=60/1 \
            ! videoscale \
            ! video/x-raw,width=128,height=128 \
            ! videoconvert \
            ! video/x-raw,format=GRAY8 \
            ! vaapipostproc format=nv12 \
            ! vaapih264enc rate-control=cbr bitrate=200 \
            ! matroskamux \
            ! filesink location="$OUT_DIR/capture.mkv" \
            &>/dev/null &
        VID_PID=$!
    else
        echo "  WARNING: no pipewire video source node found"
        echo "  Record the gamescope window manually and save as:"
        echo "    $OUT_DIR/capture.mkv"
        VID_PID=""
    fi
else
    echo "[3/3] Video capture disabled (set CAPTURE_VIDEO=1 and install gstreamer)"
    echo "  Record the gamescope window manually and save as:"
    echo "    $OUT_DIR/capture.mkv"
    VID_PID=""
fi

# ---- Wait ----
echo "Waiting for replay to end …"
wait $EMU_PID 2>/dev/null || true
[ -n "$AUDIO_PID" ] && kill $AUDIO_PID 2>/dev/null || true
[ -n "$VID_PID" ] && kill $VID_PID 2>/dev/null || true
sleep 1

# ---- Collect input log ----
if [ -f "$EMU_DIR/input_log.txt" ]; then
    cp "$EMU_DIR/input_log.txt" "$OUT_DIR/input_log.txt"
    echo ""
    echo "=== Done ==="
    echo "  Inputs:  $(wc -l < "$OUT_DIR/input_log.txt") frames"
    [ -f "$OUT_DIR/audio.wav" ] && echo "  Audio:   $(du -h "$OUT_DIR/audio.wav" | cut -f1)"
    [ -f "$OUT_DIR/capture.mkv" ] && echo "  Video:   $(du -h "$OUT_DIR/capture.mkv" | cut -f1)"
    echo ""
    echo "Next:  python scripts/collect_data.py --capture-dir data/captures \\"
    echo "           --output-dir data/processed"
else
    echo "ERROR: input_log.txt not found — Lua script may not have run"
fi
