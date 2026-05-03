#!/usr/bin/env bash
# diagnose_pipeline.sh — test the encoding pipeline step by step
#
# With a portal node:  ./emulator/diagnose_pipeline.sh <pipewire_node_id>
# With synthetic source:  ./emulator/diagnose_pipeline.sh
#
# Tests both pipewiresrc and videotestsrc paths so we can distinguish
# pipewire-specific issues from encoding pipeline issues.

set -euo pipefail
NODE="${1:-}"

run_test() {
    local label="$1" pipeline="$2"
    echo "=== $label ==="
    if timeout 3 bash -c "$pipeline" 2>&1 | tail -3; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
    fi
    echo ""
}

if [ -n "$NODE" ]; then
    SRC="pipewiresrc path=$NODE"
    echo "Testing with pipewire node $NODE"
else
    SRC="videotestsrc num-buffers=5"
    echo "Testing with synthetic source (videotestsrc, 5 frames)"
fi
echo ""

# ---- Synthetic source tests (always work as baseline) ----
if [ -z "$NODE" ]; then
    run_test "S1: src → fakesink" \
        "gst-launch-1.0 $SRC ! fakesink sync=false"

    run_test "S2: src → videoconvert → videoscale → fakesink" \
        "gst-launch-1.0 $SRC ! video/x-raw,width=1920,height=1080 ! videoconvert ! videoscale ! fakesink sync=false"

    run_test "S3: src → videoconvert → videoscale → caps → fakesink" \
        "gst-launch-1.0 $SRC ! video/x-raw,width=1920,height=1080 ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! fakesink sync=false"

    run_test "S4: src → ... → jpegenc → fakesink" \
        "gst-launch-1.0 $SRC ! video/x-raw,width=1920,height=1080 ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! jpegenc ! fakesink sync=false"

    run_test "S5: full pipeline → filesink" \
        "gst-launch-1.0 -e $SRC ! video/x-raw,width=1920,height=1080 ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! jpegenc quality=92 ! jpegparse ! matroskamux ! filesink location=/tmp/test_synth.mkv"
    echo "File: $(ls -l /tmp/test_synth.mkv 2>/dev/null || echo 'not created')"
else
    # ---- pipewire source tests ----
    run_test "P1: pipewiresrc → fakesink" \
        "gst-launch-1.0 $SRC ! fakesink sync=false"

    run_test "P2: +videoconvert" \
        "gst-launch-1.0 $SRC ! videoconvert ! fakesink sync=false"

    run_test "P3: +videoscale" \
        "gst-launch-1.0 $SRC ! videoconvert ! videoscale ! fakesink sync=false"

    run_test "P4: +caps 128×128 GRAY8" \
        "gst-launch-1.0 $SRC ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! fakesink sync=false"

    run_test "P5: +jpegenc" \
        "gst-launch-1.0 $SRC ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! jpegenc ! fakesink sync=false"

    run_test "P6: full pipeline → filesink" \
        "gst-launch-1.0 -e $SRC ! videoconvert ! videoscale ! video/x-raw,width=128,height=128,format=GRAY8 ! jpegenc quality=92 ! jpegparse ! matroskamux ! filesink location=/tmp/test_pw.mkv"
    echo "File: $(ls -l /tmp/test_pw.mkv 2>/dev/null || echo 'not created')"
fi