#!/usr/bin/env bash
# capture_replay.sh — stream a Fightcade replay, capture video + audio + inputs
#
# Usage:
#   ./emulator/capture_replay.sh <challenge_id> [game]
#
# Output (data/captures/<challenge_id>/):
#   video.mkv      — 128×128 grayscale MJPEG, 60 fps
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
GST_PID=""
AUDIO_PID=""
_cleanup() {
    echo ""
    echo "Cleaning up …"
    [ -n "${GST_PID:-}" ] && kill $GST_PID 2>/dev/null || true
    [ -n "${AUDIO_PID:-}" ] && kill $AUDIO_PID 2>/dev/null || true
    [ -n "$EMU_PID" ] && kill $EMU_PID 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f /tmp/st_rnn_y4m
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

# ---- Find gamescope's own pipewire video source ----
echo "[2/3] Finding gamescope pipewire node …"
PWNODE=""
for i in $(seq 1 20); do
    PWNODE=$(pw-cli list-objects 2>/dev/null | awk '
        /type PipeWire:Interface:Node/ { nid=$2; gsub(/,/,"",nid) }
        /node.name/   { name=$0; gsub(/.*node.name = "/,"",name); gsub(/".*/,"",name) }
        /media.class = "Video\/Source"/ && name == "gamescope" { print nid; exit }
    ' 2>/tmp/st_rnn_pw_debug)
    [ -n "$PWNODE" ] && break
    sleep 1
done

if [ -z "$PWNODE" ]; then
    echo "  Listing all video nodes:"
    head -20 /tmp/st_rnn_pw_debug 2>/dev/null
    pw-cli list-objects 2>/dev/null | grep -A2 "Video/Source" | head -20
    echo "  ERROR: no Video/Source node found"
    exit 1
fi
echo "  node: $PWNODE"

# Wait for the emulator to actually start rendering (input_log.txt appears
# when the Lua script runs, which happens after the replay connects)
echo "  Waiting for replay to start …"
for i in $(seq 1 60); do
    if [ -f "$EMU_DIR/input_log.txt" ] && [ -s "$EMU_DIR/input_log.txt" ]; then
        echo "  Replay started (frame $(wc -l < "$EMU_DIR/input_log.txt"))"
        break
    fi
    sleep 1
done

# ---- Video (gstreamer pipewiresrc) + Audio (ffmpeg pulse) ----
echo "[3/3] Recording …"

ffmpeg -y -v warning \
    -f pulse -i default \
    -vn -c:a pcm_u8 -ar 16000 -ac 1 \
    "$OUT_DIR/audio.wav" \
    2>/tmp/st_rnn_audio_err &
AUDIO_PID=$!

gst-launch-1.0 -e \
    pipewiresrc path="$PWNODE" do-timestamp=true \
    ! videoconvert \
    ! videorate \
    ! video/x-raw,framerate=60/1 \
    ! videoscale \
    ! video/x-raw,width=128,height=128,format=GRAY8 \
    ! jpegenc quality=92 \
    ! jpegparse \
    ! matroskamux \
    ! filesink location="$OUT_DIR/video.mkv" \
    2>/tmp/st_rnn_gst_err &
GST_PID=$!
sleep 1
if ! kill -0 $GST_PID 2>/dev/null; then
    echo "  ERROR: gstreamer died"
    head -c 300 /tmp/st_rnn_gst_err
fi

# ---- Wait ----
echo "Recording — press Ctrl+C when replay ends, or wait …"
wait $EMU_PID 2>/dev/null || true
