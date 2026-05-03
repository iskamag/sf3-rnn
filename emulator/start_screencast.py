#!/usr/bin/env python3
"""Start a PipeWire screen-capture source via the desktop portal.

Usage:  python3 start_screencast.py [token]
Prints the PipeWire node ID to stdout for ffmpeg/pw-v4l2.
"""

import sys
import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib

DBusGMainLoop(set_as_default=True)
BUS = dbus.SessionBus()
PORTAL = BUS.get_object("org.freedesktop.portal.Desktop",
                        "/org/freedesktop/portal/desktop")
SC = dbus.Interface(PORTAL, "org.freedesktop.portal.ScreenCast")

TOKEN = sys.argv[1] if len(sys.argv) > 1 else "st_rnn"
SESSION_HANDLE = None
PIPEWIRE_NODE = None
PHASE = 0


def _on_response(response, results, path=None):
    global SESSION_HANDLE, PIPEWIRE_NODE, PHASE

    if response != 0:
        print(f"Portal rejected (response={response})", file=sys.stderr)
        LOOP.quit()
        return

    if PHASE == 0:                      # CreateSession response
        SESSION_HANDLE = results["session_handle"]
        print(f"Session created: {SESSION_HANDLE}", file=sys.stderr)
        PHASE = 1
        SC.SelectSources(SESSION_HANDLE, {
            "handle_token": TOKEN,
            "types": dbus.UInt32(2),    # 2 = window capture
            "multiple": dbus.Boolean(False),
            "cursor_mode": dbus.UInt32(2),
        })
        print("SelectSources called — pick the gamescope window", file=sys.stderr)

    elif PHASE == 1:                    # SelectSources response (dialog accepted)
        print("Source selected, calling Start", file=sys.stderr)
        PHASE = 2
        SC.Start(SESSION_HANDLE, "", {"handle_token": TOKEN})

    elif PHASE == 2:                    # Start response — got streams
        streams = results.get("streams", [])
        print(f"Start response: {len(streams)} streams", file=sys.stderr)
        # Stream struct varies by portal backend.  Try pipewire_node_id
        # from properties, then fall back to scanning for new nodes.
        if streams:
            s = streams[0]
            # Try struct as (uint32, dict)
            if len(s) >= 2 and isinstance(s[1], dict):
                props = s[1]
                node = props.get("pipewire_node_id") or props.get("node_id")
                if node:
                    PIPEWIRE_NODE = str(node)

            if not PIPEWIRE_NODE:
                # Try the first element as a node id directly
                for elem in s:
                    if isinstance(elem, (int, dbus.UInt32)):
                        PIPEWIRE_NODE = str(int(elem))
                        break

        if not PIPEWIRE_NODE:
            # Fallback: scan PipeWire for new video nodes
            print("Scanning for pipewire video source …", file=sys.stderr)
            import subprocess
            out = subprocess.check_output(
                ["pw-cli", "list-objects"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "node.name" in line and "kwin" in line.lower():
                    print(f"  found: {line.strip()}", file=sys.stderr)
            # Just try the first Video/Source we can find
            import re
            for match in re.finditer(
                r'id (\d+),.*\n(?:.*\n)*?.*Video/(Source|Output)', out):
                candidate = match.group(1)
                print(f"  candidate node: {candidate}", file=sys.stderr)

        LOOP.quit()


BUS.add_signal_receiver(
    _on_response,
    signal_name="Response",
    dbus_interface="org.freedesktop.portal.Request",
    bus_name="org.freedesktop.portal.Desktop",
    path_keyword="path",
)

LOOP = GLib.MainLoop()

# Phase 0 — CreateSession
SC.CreateSession({"session_handle_token": TOKEN, "handle_token": TOKEN})

# Timeout after 30s (user must accept dialog)
GLib.timeout_add_seconds(30, LOOP.quit)

LOOP.run()

if PIPEWIRE_NODE:
    print(PIPEWIRE_NODE)
else:
    print("Failed to get pipewire node", file=sys.stderr)
    sys.exit(1)
