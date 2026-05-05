#!/usr/bin/env bash
# capture_replay.sh — stream Fightcade replays, capture video + audio + inputs
#
# Safe to run multiple instances in parallel — uses numbered worker slots with
# pre-created wine prefixes and flock-based mutual exclusion.
#
# Usage:
#   ./emulator/capture_replay.sh <challenge_id> [game]
#   ./emulator/capture_replay.sh replay_ids.txt [game]
#   ST_RNN_WORKERS=8 ./emulator/capture_replay.sh ...   # worker pool size (default 4)
#
# When given a file, captures every non-empty, non-comment line as a separate
# replay. When given a plain ID, captures that single replay.
#
# Output (data/captures/<challenge_id>/):
#   video.raw      — 160×120 grayscale raw bytes, 60 fps (19200 B/frame)
#   audio.wav      — 8-bit mono 16 kHz PCM
#   input_log.txt  — per-frame inputs + health from Lua

set -uo pipefail

ARG="${1:?Usage: $0 <challenge_id | replay_ids.txt> [game]}"
GAME="${2:-sfiii3nr1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EMU_DIR="$HOME/fightcade-emu"

# ---- Resolve IDs ----
IDS=()
if [ -f "$ARG" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"               # strip comments
        line="${line//[$'\t\r ']}"       # strip whitespace
        [ -n "$line" ] && IDS+=("$line")
    done < "$ARG"
    echo "Loaded ${#IDS[@]} replay IDs from $ARG"
else
    IDS+=("$ARG")
fi

[ "${#IDS[@]}" -gt 0 ] || { echo "No replay IDs found"; exit 1; }

# ---- Worker-slot isolation ----
WORKER_COUNT="${ST_RNN_WORKERS:-4}"
LOCK_DIR="/tmp/st_rnn_worker_locks"
mkdir -p "$LOCK_DIR"

WORKER_SLOT=""
for slot in $(seq 0 $((WORKER_COUNT - 1))); do
    exec {_lock_fd}>"$LOCK_DIR/slot_$slot"
    if flock -n $_lock_fd 2>/dev/null; then
        WORKER_SLOT=$slot
        break
    fi
    exec {_lock_fd}>&-
done
if [ -z "$WORKER_SLOT" ]; then
    echo "All $WORKER_COUNT worker slots busy; waiting on slot 0 …"
    exec {_lock_fd}>"$LOCK_DIR/slot_0"
    flock $_lock_fd
    WORKER_SLOT=0
fi

INSTANCE_ID="${WORKER_SLOT}_$$"
TMPDIR="/tmp/st_rnn_capture_${INSTANCE_ID}"
WINE_PFX="$HOME/.fightcade-wine-${WORKER_SLOT}"

mkdir -p "$TMPDIR"

if [ ! -d "$WINE_PFX/drive_c" ]; then
    echo "  Creating wine prefix: $WINE_PFX"
    cp -a "$HOME/.fightcade-wine" "$WINE_PFX" 2>/dev/null || true
fi
export WINEPREFIX="$WINE_PFX"
export WINEDEBUG=-all

# Per-instance Lua workdir — input_log.txt lands here.
WORKDIR="$TMPDIR/work"
mkdir -p "$WORKDIR"
cp "$SCRIPT_DIR/input_log.lua" "$WORKDIR/"

# ---- Capture one replay ----
capture_one() {
    local challenge_id="$1"
    local out_dir="$PROJECT_DIR/data/captures/$challenge_id"
    local gst_err="$TMPDIR/gst_err_${challenge_id}"
    local audio_err="$TMPDIR/audio_err_${challenge_id}"

    mkdir -p "$out_dir"
    rm -f "$WORKDIR/input_log.txt"

    # Isolated null audio sink.
    local audio_sink="st_rnn_cap_${INSTANCE_ID}"
    local null_sink_id=""
    local audio_src=""
    null_sink_id=$(pactl load-module module-null-sink \
        "sink_name=${audio_sink}" \
        "sink_properties=device.description=${audio_sink}" 2>/dev/null)
    if [ -z "$null_sink_id" ]; then
        echo "  [${challenge_id}] WARNING: could not create null sink"
        audio_src="$(pactl get-default-sink 2>/dev/null).monitor"
    else
        audio_src="${audio_sink}.monitor"
        pactl set-default-sink "$audio_sink" 2>/dev/null || true
    fi

    echo "--- ${challenge_id} (${GAME}) [worker ${WORKER_SLOT}] ---"

    # Launch emulator.
    local emu_pid=""
    (
        cd "$WORKDIR"
        gamescope -W 640 -H 480 -r 60 -- \
            wine "$EMU_DIR/fcadefbneo.exe" \
            "quark:stream,${GAME},${challenge_id}.2,7100" \
            -lua input_log.lua &
    )
    emu_pid=$!
    sleep 5

    # Find this instance's pipewire video node.
    local pwnode=""
    for i in $(seq 1 30); do
        if command -v pw-dump &>/dev/null; then
            pwnode=$(pw-dump 2>/dev/null | python3 -c "
import json, sys
for o in json.load(sys.stdin):
    p = o.get('info',{}).get('props',{})
    if (p.get('media.class') == 'Video/Source'
        and p.get('node.name') == 'gamescope'
        and str(p.get('application.process.id','')) == '$emu_pid'):
        print(o['id']); break
" 2>/dev/null)
        fi
        # Fallback: first gamescope Video/Source node.
        if [ -z "$pwnode" ]; then
            pwnode=$(pw-cli list-objects 2>/dev/null | awk '
                /type PipeWire:Interface:Node/ { nid=$2; gsub(/,/,"",nid) }
                /node.name/   { name=$0; gsub(/.*node.name = "/,"",name); gsub(/".*/,"",name) }
                /media.class = "Video\/Source"/ && name == "gamescope" { print nid; exit }
            ')
        fi
        [ -n "$pwnode" ] && break
        sleep 1
    done

    if [ -z "$pwnode" ]; then
        echo "  [${challenge_id}] ERROR: no pipewire video node found"
        kill $emu_pid 2>/dev/null || true
        [ -n "$null_sink_id" ] && pactl unload-module "$null_sink_id" 2>/dev/null || true
        return 1
    fi
    echo "  pipewire node: $pwnode"

    # Wait for replay to start.
    for i in $(seq 1 90); do
        if [ -f "$WORKDIR/input_log.txt" ] && [ -s "$WORKDIR/input_log.txt" ]; then
            echo "  replay started (frame $(wc -l < "$WORKDIR/input_log.txt"))"
            break
        fi
        sleep 1
    done

    # Record.
    echo "  recording …"

    local audio_pid=""
    ffmpeg -y -v warning \
        -f pulse -i "$audio_src" \
        -vn -c:a pcm_u8 -ar 16000 -ac 1 \
        "$out_dir/audio.wav" \
        2>"$audio_err" &
    audio_pid=$!

    local gst_pid=""
    gst-launch-1.0 -e \
        pipewiresrc path="$pwnode" do-timestamp=true \
        ! videocrop top=0 \
        ! videoconvert \
        ! videorate \
        ! video/x-raw,framerate=60/1 \
        ! videoscale \
        ! video/x-raw,width=160,height=120 \
        ! videoconvert \
        ! video/x-raw,format=GRAY8 \
        ! filesink location="$out_dir/video.raw" \
        2>"$gst_err" &
    gst_pid=$!
    sleep 1
    if ! kill -0 $gst_pid 2>/dev/null; then
        echo "  [${challenge_id}] ERROR: gstreamer died"
        head -c 300 "$gst_err"
    fi

    # Wait for emulator to exit.
    wait $emu_pid 2>/dev/null || true

    kill $audio_pid 2>/dev/null || true
    kill $gst_pid  2>/dev/null || true
    wait 2>/dev/null || true

    [ -n "$null_sink_id" ] && pactl unload-module "$null_sink_id" 2>/dev/null || true

    # Save input log.
    if [ -f "$WORKDIR/input_log.txt" ]; then
        cp "$WORKDIR/input_log.txt" "$out_dir/input_log.txt"
        echo "  input_log.txt: $(wc -l < "$out_dir/input_log.txt") frames"
    fi
    echo "  done — ${challenge_id}"
}

# ---- Cleanup on exit ----
_cleanup() {
    echo ""
    echo "Cleaning up …"
    exec {_lock_fd}>&- 2>/dev/null || true
    rm -rf "$TMPDIR"
}
trap _cleanup EXIT

# ---- Main loop ----
FAILED=0
for id in "${IDS[@]}"; do
    capture_one "$id" || ((FAILED++))
done

echo "=== Done — ${#IDS[@]} replays, ${FAILED} failed ==="
