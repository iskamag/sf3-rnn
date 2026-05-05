#!/usr/bin/env bash
# capture_replay.sh — stream Fightcade replays, capture video + audio + inputs
#
# Uses numbered worker slots with flock-based mutual exclusion. By default it
# reuses the main Fightcade Wine prefix so FBNeo keeps the same fullscreen
# behavior as manual launches; cloned per-worker prefixes are opt-in. Audio is
# captured via native PipeWire nodes. Multi-worker capture can still race on
# `input_log.txt` if FBNeo ignores the per-worker log filename override.
#
# Usage:
#   ./emulator/capture_replay.sh <challenge_id> [game]
#   ./emulator/capture_replay.sh replay_ids.txt [game]
#   ST_RNN_WORKERS=8 ./emulator/capture_replay.sh ...   # worker pool size (default 4)
#   ST_RNN_CLONED_PREFIXES=1 ./emulator/capture_replay.sh ...
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
        line="${line%%#*}"
        line="${line//[$'\t\r ']}"
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
BASE_WINE_PFX="$HOME/.fightcade-wine"
USE_CLONED_PREFIXES="${ST_RNN_CLONED_PREFIXES:-0}"
if [ "$USE_CLONED_PREFIXES" = "1" ]; then
    WINE_PFX="$HOME/.fightcade-wine-${WORKER_SLOT}"
else
    WINE_PFX="$BASE_WINE_PFX"
fi

mkdir -p "$TMPDIR"

if [ "$USE_CLONED_PREFIXES" = "1" ] && [ ! -d "$WINE_PFX/drive_c" ]; then
    echo "  Creating wine prefix: $WINE_PFX"
    cp -a "$BASE_WINE_PFX" "$WINE_PFX" 2>/dev/null || true
fi
export WINEPREFIX="$WINE_PFX"
export WINEDEBUG=-all

# Use the standard Lua script name/path so FBNeo launches the same way as a
# manual run; only the output log filename varies per worker.
EMU_LUA="input_log.lua"
EMU_LOG="st_rnn_input_log_${INSTANCE_ID}.txt"
EMU_LUA_PATH="$EMU_DIR/$EMU_LUA"
EMU_LOG_PATH="$EMU_DIR/$EMU_LOG"
DEFAULT_EMU_LOG_PATH="$EMU_DIR/input_log.txt"
CURRENT_OUT_DIR=""
CURRENT_EMU_PID=""
CURRENT_AUDIO_PID=""
CURRENT_GST_PID=""
cp "$SCRIPT_DIR/input_log.lua" "$EMU_LUA_PATH"

# ---- Snapshot existing PipeWire audio streams before launch ----
_audio_snapshot() {
    pw-dump 2>/dev/null | python3 -c "
import json, sys
for o in json.load(sys.stdin):
    p = o.get('info',{}).get('props',{})
    if p.get('media.class') in ('Stream/Output/Audio',):
        print(o['id'])
" 2>/dev/null
}

_descendant_pids() {
    local root="$1"
    local queue="$root"
    local seen="$root"

    while [ -n "$queue" ]; do
        local next=""
        for pid in $queue; do
            local children
            children=$(pgrep -P "$pid" 2>/dev/null || true)
            for child in $children; do
                case " $seen " in
                    *" $child "*) ;;
                    *)
                        seen="$seen $child"
                        next="$next $child"
                        ;;
                esac
            done
        done
        queue="$next"
    done

    printf '%s\n' $seen
}

_descendant_env_var() {
    local root_pid="$1"
    local var_name="$2"
    local pid
    for pid in $(_descendant_pids "$root_pid"); do
        [ "$pid" = "$root_pid" ] && continue
        local env_file="/proc/$pid/environ"
        [ -r "$env_file" ] || continue
        local value
        value=$(tr '\0' '\n' < "$env_file" | grep "^${var_name}=" | head -n 1 | cut -d= -f2-)
        if [ -n "$value" ]; then
            if [ "$var_name" = "DISPLAY" ] && [ -n "${DISPLAY:-}" ] && [ "$value" = "$DISPLAY" ]; then
                continue
            fi
            printf '%s\n' "$value"
            return 0
        fi
    done
    return 1
}

_current_input_log_path() {
    if [ -f "$EMU_LOG_PATH" ] && [ -s "$EMU_LOG_PATH" ]; then
        printf '%s\n' "$EMU_LOG_PATH"
        return 0
    fi
    if [ -f "$DEFAULT_EMU_LOG_PATH" ] && [ -s "$DEFAULT_EMU_LOG_PATH" ]; then
        printf '%s\n' "$DEFAULT_EMU_LOG_PATH"
        return 0
    fi
    return 1
}

_copy_input_log() {
    [ -n "$CURRENT_OUT_DIR" ] || return 0
    mkdir -p "$CURRENT_OUT_DIR"

    local log_path
    log_path=$(_current_input_log_path || true)
    if [ -n "$log_path" ]; then
        cp "$log_path" "$CURRENT_OUT_DIR/input_log.txt"
        echo "  input_log.txt: $(wc -l < "$CURRENT_OUT_DIR/input_log.txt") frames"
    fi
}

_stop_recorders() {
    if [ -n "$CURRENT_AUDIO_PID" ]; then
        kill "$CURRENT_AUDIO_PID" 2>/dev/null || true
    fi
    if [ -n "$CURRENT_GST_PID" ]; then
        kill "$CURRENT_GST_PID" 2>/dev/null || true
    fi

    if [ -n "$CURRENT_AUDIO_PID" ]; then
        wait "$CURRENT_AUDIO_PID" 2>/dev/null || true
    fi
    if [ -n "$CURRENT_GST_PID" ]; then
        wait "$CURRENT_GST_PID" 2>/dev/null || true
    fi

    CURRENT_AUDIO_PID=""
    CURRENT_GST_PID=""
}

_terminate_emulator_processes() {
    if [ -n "$CURRENT_EMU_PID" ]; then
        kill -TERM -- "-$CURRENT_EMU_PID" 2>/dev/null || kill "$CURRENT_EMU_PID" 2>/dev/null || true
        sleep 1

        local pid
        for pid in $(_descendant_pids "$CURRENT_EMU_PID"); do
            [ "$pid" = "$CURRENT_EMU_PID" ] && continue
            kill "$pid" 2>/dev/null || true
        done

        kill -KILL -- "-$CURRENT_EMU_PID" 2>/dev/null || kill -KILL "$CURRENT_EMU_PID" 2>/dev/null || true
        wait "$CURRENT_EMU_PID" 2>/dev/null || true
    fi

    CURRENT_EMU_PID=""
}

# ---- Find a PipeWire node (video source or new audio stream) ----
_find_video_node() {
    local pid="$1"
    local pwnode=""
    for i in $(seq 1 30); do
        if command -v pw-dump &>/dev/null; then
            pwnode=$(pw-dump 2>/dev/null | python3 -c "
import json, sys
for o in json.load(sys.stdin):
    p = o.get('info',{}).get('props',{})
    if (p.get('media.class') == 'Video/Source'
        and p.get('node.name') == 'gamescope'
        and str(p.get('application.process.id','')) == '$pid'):
        print(o['id']); break
" 2>/dev/null)
        fi
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
    echo "$pwnode"
}

_find_new_audio_node() {
    local prev="$1"
    local root_pid="$2"
    local prev_set
    prev_set=$(printf '%s\n' "$prev" | sort -u)
    local anode=""
    for i in $(seq 1 30); do
        local pid_set
        pid_set=$(_descendant_pids "$root_pid" | sort -u)
        anode=$(pw-dump 2>/dev/null | python3 -c "
import json, sys
prev = set('''$prev_set'''.split())
pids = set('''$pid_set'''.split())
candidates = []
for o in json.load(sys.stdin):
    p = o.get('info',{}).get('props',{})
    if p.get('media.class') in ('Stream/Output/Audio',):
        oid = str(o['id'])
        if oid in prev:
            continue
        pid = str(p.get('application.process.id', ''))
        if pid and pid in pids:
            print(oid)
            raise SystemExit
        candidates.append(oid)

if len(candidates) == 1:
    print(candidates[0])
" 2>/dev/null)
        [ -n "$anode" ] && break
        sleep 1
    done
    echo "$anode"
}

_toggle_fbneo_fullscreen() {
    local root_pid="$1"
    command -v xdotool >/dev/null 2>&1 || return 0

    local nested_display
    nested_display=$(_descendant_env_var "$root_pid" DISPLAY)
    [ -n "$nested_display" ] || return 0

    local nested_xauth
    nested_xauth=$(_descendant_env_var "$root_pid" XAUTHORITY || true)

    local xdo=(env DISPLAY="$nested_display")
    if [ -n "$nested_xauth" ]; then
        xdo+=(XAUTHORITY="$nested_xauth")
    fi
    xdo+=(xdotool)

    local geometry
    geometry=$(${xdo[@]} getdisplaygeometry 2>/dev/null || true)
    [ -n "$geometry" ] || return 0

    local width height
    width=${geometry%% *}
    height=${geometry##* }
    [ -n "$width" ] || return 0
    [ -n "$height" ] || return 0

    local cx=$((width / 2))
    local cy=$((height / 2))
    ${xdo[@]} mousemove "$cx" "$cy" >/dev/null 2>&1 || true
    sleep 0.1
    ${xdo[@]} click --repeat 2 --delay 120 1 >/dev/null 2>&1 || true
}

# ---- Capture one replay ----
capture_one() {
    local challenge_id="$1"
    local out_dir="$PROJECT_DIR/data/captures/$challenge_id"
    local gst_err="$TMPDIR/gst_err_${challenge_id}"
    local audio_err="$TMPDIR/audio_err_${challenge_id}"

    mkdir -p "$out_dir"
    CURRENT_OUT_DIR="$out_dir"
    rm -f "$EMU_LOG_PATH"
    rm -f "$DEFAULT_EMU_LOG_PATH"

    echo "--- ${challenge_id} (${GAME}) [worker ${WORKER_SLOT}] ---"

    # Snapshot current audio streams before launch.
    local prev_streams
    prev_streams=$(_audio_snapshot)

    # Launch emulator.
    local emu_pid=""
    (
        cd "$EMU_DIR" || exit 1
        ST_RNN_INPUT_LOG="$EMU_LOG" \
        exec setsid gamescope -W 640 -H 480 -r 60 -- \
            wine fcadefbneo.exe \
            "quark:stream,${GAME},${challenge_id}.2,7100" \
            -lua "$EMU_LUA"
    ) &
    emu_pid=$!
    CURRENT_EMU_PID="$emu_pid"
    sleep 5

    # Find video node.
    echo "  Finding video node …"
    local pwnode
    pwnode=$(_find_video_node "$emu_pid")
    if [ -z "$pwnode" ]; then
        echo "  [${challenge_id}] ERROR: no pipewire video node found"
        kill $emu_pid 2>/dev/null || true
        return 1
    fi
    echo "  video node: $pwnode"

    # Find new audio stream — wine creates this once audio plays.
    echo "  Finding audio stream …"
    local audio_node
    audio_node=$(_find_new_audio_node "$prev_streams" "$emu_pid")
    if [ -n "$audio_node" ]; then
        echo "  audio node: $audio_node"
    else
        echo "  audio node: not found (recording may be silent)"
    fi

    # Wait for replay to start.
    echo "  Waiting for replay to start …"
    for i in $(seq 1 90); do
        local log_path
        log_path=$(_current_input_log_path || true)
        if [ -n "$log_path" ]; then
            echo "  replay started (frame $(wc -l < "$log_path"))"
            break
        fi
        sleep 1
    done

    echo "  toggling FBNeo fullscreen …"
    _toggle_fbneo_fullscreen "$emu_pid"
    sleep 1

    # Record.
    echo "  recording …"

    local audio_pid=""
    if [ -n "$audio_node" ]; then
        gst-launch-1.0 -e \
            pipewiresrc path="$audio_node" do-timestamp=true \
            ! audioconvert \
            ! audioresample \
            ! audio/x-raw,rate=16000,channels=1,format=U8 \
            ! wavenc \
            ! filesink location="$out_dir/audio.wav" \
            2>"$audio_err" &
        audio_pid=$!
        CURRENT_AUDIO_PID="$audio_pid"
    fi

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
    CURRENT_GST_PID="$gst_pid"
    sleep 1
    if ! kill -0 $gst_pid 2>/dev/null; then
        echo "  [${challenge_id}] ERROR: gstreamer video died"
        head -c 300 "$gst_err"
    fi

    # Wait for emulator to exit, or kill it when Lua writes REPLAY_END.
    _wait_emu_done() {
        local pid="$1"
        local log_path="$2"

        while kill -0 "$pid" 2>/dev/null; do
            sleep 2

            kill -0 "$pid" 2>/dev/null || break

            if [ -f "$log_path" ] && grep -q "REPLAY_END" "$log_path" 2>/dev/null; then
                echo "  replay-end marker detected — stopping emulator"
                kill "$pid" 2>/dev/null || true
                sleep 1
                break
            fi
        done
    }

    _wait_emu_done "$emu_pid" "$(_current_input_log_path)"

    _stop_recorders
    _terminate_emulator_processes

    # Save input log.
    _copy_input_log
    echo "  done — ${challenge_id}"
    CURRENT_OUT_DIR=""
}

# ---- Cleanup on exit ----
_cleanup() {
    echo ""
    echo "Cleaning up …"
    _copy_input_log
    _terminate_emulator_processes
    rm -f "$EMU_LOG_PATH"
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
