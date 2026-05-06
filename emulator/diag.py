#!/usr/bin/env python3
"""Diagnostic: launch one gamescope, dump PipeWire state, print process tree."""

import json, os, shutil, signal, subprocess, sys, time

EMU_DIR = os.path.expanduser("~/fightcade-emu")
WINE_PFX = os.path.expanduser("~/.fightcade-wine-replay-0")
LUA_SCRIPT = os.path.join(os.path.dirname(__file__), "input_log.lua")
CHALLENGE_ID = sys.argv[1] if len(sys.argv) > 1 else "1777967023291-1409"

shutil.copy(LUA_SCRIPT, os.path.join(EMU_DIR, "input_log.lua"))
print("Copied input_log.lua to EMU_DIR")

env = os.environ.copy()
env["WINEPREFIX"] = WINE_PFX
env["WINEDEBUG"] = "-all"

url = f"quark:stream,sfiii3nr1,{CHALLENGE_ID}.2,7100"

print(f"Launching gamescope (start_new_session=True)...")
emu = subprocess.Popen(
    ["gamescope", "-W", "640", "-H", "480", "-r", "60", "--",
     "wine", "fcadefbneo.exe", url, "-lua", "input_log.lua"],
    cwd=EMU_DIR, env=env, stdin=subprocess.DEVNULL,
    start_new_session=True,
)
emu_pid = emu.pid
print(f"emu.pid = {emu_pid}")
print(f"Waiting 10s for gamescope to initialise...")
time.sleep(10)

# Process tree
print(f"\n=== Process tree under {emu_pid} ===")
try:
    subprocess.run(["pstree", "-p", str(emu_pid)], timeout=5)
except Exception:
    try:
        raw = subprocess.check_output(["ps", "--ppid", str(emu_pid), "-o", "pid,comm"], text=True)
        print(raw)
    except Exception as e:
        print(f"  ps error: {e}")

# All children recursively
def all_pids(root):
    seen = {root}
    q = [root]
    while q:
        p = q.pop(0)
        try:
            children = subprocess.check_output(["pgrep", "-P", str(p)], text=True).strip().split()
            for c in children:
                c = int(c)
                if c not in seen:
                    seen.add(c)
                    q.append(c)
        except subprocess.CalledProcessError:
            pass
    return seen

print(f"\n=== All descendant PIDs of {emu_pid} ===")
for pid in sorted(all_pids(emu_pid)):
    if pid == emu_pid:
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        # Check for DISPLAY
        with open(f"/proc/{pid}/environ", "rb") as f:
            env_raw = f.read()
            for e in env_raw.split(b"\0"):
                if e.startswith(b"DISPLAY="):
                    print(f"  PID {pid} DISPLAY={e.decode().split('=',1)[1]}  cmd={cmd[:80]}")
                    break
            else:
                print(f"  PID {pid}  cmd={cmd[:80]}")
    except OSError:
        print(f"  PID {pid}  (gone)")

# PipeWire nodes
print(f"\n=== All Video/Source PipeWire nodes ===")
try:
    r = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=10)
    data = json.loads(r.stdout)
    for obj in data:
        props = obj.get("info", {}).get("props", {})
        if props.get("media.class") == "Video/Source":
            print(f"  id={obj['id']:>4}  pid={props.get('application.process.id', 'N/A'):>6}  "
                  f"name={props.get('node.name', 'N/A'):>20}  "
                  f"desc={props.get('node.description', 'N/A')[:40]}")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\nemu.pid = {emu_pid}")
print(f"Verify: does any Video/Source node's pid match {emu_pid}?")
print("\nKilling gamescope...")
os.killpg(emu_pid, signal.SIGKILL)
emu.wait(timeout=3)
print("Done.")
