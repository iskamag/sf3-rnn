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
        PHASE = 1
        SC.SelectSources(SESSION_HANDLE, {
            "handle_token": TOKEN,
            "types": dbus.UInt32(1),    # monitor
            "multiple": dbus.Boolean(False),
            "cursor_mode": dbus.UInt32(2),
        })

    elif PHASE == 1:                    # SelectSources response (dialog accepted)
        PHASE = 2
        SC.Start(SESSION_HANDLE, "", {"handle_token": TOKEN})

    elif PHASE == 2:                    # Start response — got streams
        streams = results.get("streams", [])
        if streams:
            node = streams[0].get("pipewire_node_id")
            if node:
                PIPEWIRE_NODE = str(node)
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
