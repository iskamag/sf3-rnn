#!/usr/bin/env bash
# capture_all.sh — parallel replay capture from a list file
#
# Reads replay IDs from a file (one per line, # comments allowed) and fans
# them out across N parallel capture workers.  Each worker gets its own
# wine prefix, audio sink, and temp directory.
#
# Usage:
#   ./emulator/capture_all.sh replay_ids.txt [workers] [game]
#   ./emulator/capture_all.sh replay_ids.txt 4 sfiii3nr1
#
# Defaults: 4 workers, sfiii3nr1.

set -uo pipefail

IDS_FILE="${1:?Usage: $0 <replay_ids.txt> [workers] [game]}"
WORKERS="${2:-4}"
GAME="${3:-sfiii3nr1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAPTURE="$SCRIPT_DIR/capture_replay.sh"

[ -x "$CAPTURE" ] || { echo "ERROR: $CAPTURE not found"; exit 1; }
[ -f "$IDS_FILE" ] || { echo "ERROR: $IDS_FILE not found"; exit 1; }

# Read and clean IDs.
IDS=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="${line//[$'\t\r ']}"
    [ -n "$line" ] && IDS+=("$line")
done < "$IDS_FILE"

N="${#IDS[@]}"
[ "$N" -gt 0 ] || { echo "No replay IDs in $IDS_FILE"; exit 1; }

# Distribute IDs across workers (round-robin).
WORKER_IDS=()
for ((i = 0; i < WORKERS; i++)); do WORKER_IDS[i]=""; done
for ((i = 0; i < N; i++)); do
    w=$((i % WORKERS))
    WORKER_IDS[w]="${WORKER_IDS[w]}${IDS[i]}"$'\n'
done

echo "=== ${N} replays → ${WORKERS} workers ==="

# Launch workers in parallel.
PIDS=()
for ((w = 0; w < WORKERS; w++)); do
    ids="${WORKER_IDS[w]}"
    [ -z "$ids" ] && continue
    tmp="$(mktemp)"
    printf '%s\n' "$ids" > "$tmp"                     # one ID per line
    echo "  worker $w: $(wc -l < "$tmp") replays"

    ST_RNN_WORKERS="$WORKERS" "$CAPTURE" "$tmp" "$GAME" &
    PIDS+=($!)
done

# Wait and report.
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAILED++))
done

[ "$FAILED" -gt 0 ] && echo "=== ${FAILED} worker(s) failed ===" || echo "=== All workers finished ==="
