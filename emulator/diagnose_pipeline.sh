#!/usr/bin/env bash
# diagnose_pipeline.sh — test each stage of the pipeline incrementally
# Run while a portal screencast session is active (after start_screencast.py)
#
# Usage: ./emulator/diagnose_pipeline.sh <pipewire_node_id>

set -euo pipefail
NODE="${1:?Usage: $0 <pipewire_node_id>}"

echo "=== Test 1: pipewiresrc → fakesink ==="
timeout 3 gst-launch-1.0 pipewiresrc path="$NODE" ! fakesink sync=false 2>&1 | tail -3

echo "=== Test 2: pipewiresrc → videoconvert → fakesink ==="
timeout 3 gst-launch-1.0 pipewiresrc path="$NODE" ! videoconvert ! fakesink sync=false 2>&1 | tail -3

echo "=== Test 3: pipewiresrc → videoconvert → videoscale → fakesink ==="
timeout 3 gst-launch-1.0 pipewiresrc path="$NODE" ! videoconvert ! videoscale ! fakesink sync=false 2>&1 | tail -3

echo "=== Test 4: pipewiresrc → videoconvert → videoscale → caps filter 128x128 GRAY8 → fakesink ==="
timeout 3 gst-launch-1.0 pipewiresrc path="$NODE" ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! fakesink sync=false 2>&1 | tail -3

echo "=== Test 5: pipewiresrc → videoconvert → videoscale → caps → jpegenc → fakesink ==="
timeout 3 gst-launch-1.0 pipewiresrc path="$NODE" ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! jpegenc ! fakesink sync=false 2>&1 | tail -3

echo "=== Test 6: full pipeline with filesink ==="
timeout 3 gst-launch-1.0 pipewiresrc path="$NODE" ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! jpegenc quality=92 ! jpegparse ! matroskamux ! filesink location=/tmp/test_diag.mkv 2>&1 | tail -3

echo "Done. File: $(ls -l /tmp/test_diag.mkv 2>/dev/null || echo 'not created')"