#!/usr/bin/env python3
"""Record Fightcade FBNeo replays: video, audio, inputs.

Usage: ./emulator/recorder.py replay_ids.txt [--workers N] [--game sfiii3nr1]

replay_ids.txt format: <challenge_id> <length_seconds> per line (comments after #).
"""

import json, os, re, shutil, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_EMU_DIR = Path.home() / "fightcade-emu"
CAPTURES_DIR = PROJECT_DIR / "data" / "captures"
BASE_WINE_PFX = Path.home() / ".fightcade-wine"
LUA_SCRIPT = Path(__file__).resolve().parent / "input_log.lua"

_shutdown = threading.Event()


# ── gamescope stderr ─────────────────────────────────────────────────


def _parse_gamescope_stderr(stderr) -> tuple[str | None, str | None]:
    """Return (video_pipewire_node, xwayland_display) from stderr lines."""
    video, display = None, None
    for raw in stderr:
        line = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        sys.stdout.write(line); sys.stdout.flush()
        if not video and (m := re.search(r"stream available on node ID:\s*(\d+)", line)):
            video = m.group(1)
        if not display and (m := re.search(r"Starting Xwayland on\s+(:\d+)", line)):
            display = m.group(1)
        if video and display:
            break
    return video, display


# ── fullscreen ───────────────────────────────────────────────────────


def _toggle_fullscreen(display: str | None) -> None:
    if not display or not shutil.which("xdotool"):
        return
    env = os.environ | {"DISPLAY": display}
    try:
        r = subprocess.run(["xdotool", "getdisplaygeometry"], capture_output=True,
                           text=True, env=env, timeout=5)
        w, h = map(int, r.stdout.strip().split())
        subprocess.run(["xdotool", "mousemove", str(w // 2), str(h // 2)], env=env, timeout=5)
        time.sleep(0.1)
        subprocess.run(["xdotool", "click", "--repeat", "2", "--delay", "120", "1"],
                       env=env, timeout=5)
    except Exception:
        pass


# ── GStreamer ────────────────────────────────────────────────────────


def _video_pipe(node: str, out: Path) -> Gst.Pipeline:
    p = Gst.parse_launch(
        f"pipewiresrc path={node} do-timestamp=true "
        "! videocrop top=0 ! videoconvert ! videorate ! video/x-raw,framerate=60/1 "
        "! videoscale ! video/x-raw,width=160,height=120 "
        "! videoconvert ! video/x-raw,format=GRAY8 "
        f"! filesink location={out}/video.raw")
    assert isinstance(p, Gst.Pipeline); return p


def _audio_pipe(node: str, out: Path) -> Gst.Pipeline:
    p = Gst.parse_launch(
        f"pipewiresrc path={node} do-timestamp=true "
        "! audioconvert ! audioresample ! audio/x-raw,rate=16000,channels=1,format=U8 "
        "! wavenc "
        f"! filesink location={out}/audio.wav")
    assert isinstance(p, Gst.Pipeline); return p


# ── audio node via pw-dump ───────────────────────────────────────────


def _find_audio(emu_pid: int, prev: set[str]) -> str | None:
    pids = _descendant_pids(emu_pid)
    for _ in range(60):
        if _shutdown.is_set():
            return None
        try:
            candidates = []
            for obj in json.loads(subprocess.run(
                    ["pw-dump"], capture_output=True, text=True, timeout=10).stdout):
                p = obj.get("info", {}).get("props", {})
                if p.get("media.class") not in ("Stream/Output/Audio",):
                    continue
                oid = str(obj["id"])
                if oid in prev:
                    continue
                pid = str(p.get("application.process.id", "") or "")
                if pid and pid in pids:
                    return oid
                candidates.append(oid)
            if len(candidates) == 1:
                return candidates[0]
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        time.sleep(1)
    return None


def _descendant_pids(root: int) -> set[str]:
    seen, queue = {root}, [root]
    while queue:
        try:
            children = subprocess.check_output(
                ["pgrep", "-P", str(queue.pop(0))], text=True,
                stderr=subprocess.DEVNULL).strip().split()
            for c in children:
                c = int(c)
                if c not in seen:
                    seen.add(c); queue.append(c)
        except subprocess.CalledProcessError:
            pass
    return {str(p) for p in seen}


def _snapshot_audio() -> set[str]:
    try:
        return {str(o["id"]) for o in json.loads(subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=10).stdout)
                if o.get("info", {}).get("props", {}).get("media.class")
                in ("Stream/Output/Audio",)}
    except Exception:
        return set()


# ── capture ──────────────────────────────────────────────────────────


class CaptureError(Exception):
    pass


def capture_one(challenge_id: str, game: str, wine_pfx: Path, emu_dir: Path,
                length: int) -> None:
    out = CAPTURES_DIR / challenge_id
    if (out / "input_log.txt").exists() or (out / "video.raw").exists():
        print(f"SKIP {challenge_id} — already captured"); return
    out.mkdir(parents=True, exist_ok=True)

    log = emu_dir / "input_log.txt"
    shutil.copy(LUA_SCRIPT, emu_dir / "input_log.lua")
    log.unlink(missing_ok=True)

    env = os.environ | {
        "WINEPREFIX": str(wine_pfx), "WINEDEBUG": "-all",
        "ST_RNN_INPUT_LOG": "input_log.txt",
    }

    emu = None
    pipes: list[Gst.Pipeline] = []

    def cleanup(save_log=True):
        for p in pipes:
            p.set_state(Gst.State.NULL); p.get_state(3 * Gst.SECOND)
        if emu:
            try:
                os.killpg(emu.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            time.sleep(0.3)
            try:
                emu.kill()
            except Exception:
                pass
            try:
                emu.wait(timeout=3)
            except Exception:
                pass
        subprocess.run(["wineserver", "-k"], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if save_log and log.exists() and log.stat().st_size > 0:
            dest = out / "input_log.txt"
            shutil.copy(log, dest)
            print(f"  log: {dest.read_text().count(chr(10))} frames")
        (emu_dir / "input_log.lua").unlink(missing_ok=True)

    try:
        print(f"--- {challenge_id} ({game}) ---")
        prev_audio = _snapshot_audio()

        url = f"quark:stream,{game},{challenge_id}.2,7100"
        emu = subprocess.Popen(
            ["gamescope", "-W", "640", "-H", "480", "-r", "60", "--",
             "wine", "fcadefbneo.exe", url, "-lua", "input_log.lua"],
            cwd=emu_dir, env=env, stdin=subprocess.DEVNULL,
            start_new_session=True, stderr=subprocess.PIPE)
        print(f"  pid: {emu.pid}")

        video_node, x11 = _parse_gamescope_stderr(emu.stderr)
        if not video_node:
            raise CaptureError("no video node")
        print(f"  video: {video_node}  display: {x11}")

        for _ in range(90):
            if _shutdown.is_set():
                raise CaptureError("interrupted")
            if log.exists() and log.stat().st_size > 0:
                print(f"  replay started ({log.read_text().count(chr(10))} frames)")
                break
            time.sleep(1)

        _toggle_fullscreen(x11)
        time.sleep(1)

        vp = _video_pipe(video_node, out)
        vp.set_state(Gst.State.PLAYING); pipes.append(vp)
        time.sleep(1)
        if vp.get_state(Gst.SECOND)[1] != Gst.State.PLAYING:
            raise CaptureError(f"video state={vp.get_state(Gst.SECOND)[1]}")
        print("  video: PLAYING")

        audio_node = _find_audio(emu.pid, prev_audio)
        if audio_node:
            ap = _audio_pipe(audio_node, out)
            ap.set_state(Gst.State.PLAYING); pipes.append(ap)
            print(f"  audio: {audio_node}")
        else:
            print("  audio: silent")

        started = time.time()
        while emu.poll() is None and not _shutdown.is_set():
            elapsed = time.time() - started
            if elapsed >= length:
                print(f"  time ({elapsed:.0f}s)"); break
            time.sleep(1)

        print(f"  done — {challenge_id}")

    except CaptureError as e:
        print(f"  [{challenge_id}] ERROR: {e}"); raise
    finally:
        cleanup(save_log=True)


# ── worker ───────────────────────────────────────────────────────────


def capture_chunk(worker_id: int, ids: list[tuple[str, int]], game: str) -> None:
    pfx = Path.home() / f".fightcade-wine-replay-{worker_id}"
    if BASE_WINE_PFX.is_dir() and not (pfx / "drive_c").is_dir():
        print(f"[worker {worker_id}] cloning {pfx}")
        subprocess.run(["cp", "-a", str(BASE_WINE_PFX), str(pfx)], check=True)
    emu_dir = BASE_EMU_DIR.parent / f"fightcade-emu-replay-{worker_id}"
    if BASE_EMU_DIR.is_dir() and not emu_dir.is_dir():
        print(f"[worker {worker_id}] cloning {emu_dir}")
        subprocess.run(["cp", "-a", str(BASE_EMU_DIR), str(emu_dir)], check=True)
    for cid, length in ids:
        if _shutdown.is_set():
            break
        capture_one(cid, game, pfx, emu_dir, length)


# ── main ─────────────────────────────────────────────────────────────


def main():
    import argparse
    p = argparse.ArgumentParser(description="Record Fightcade FBNeo replays")
    p.add_argument("ids_file"); p.add_argument("--workers", "-w", type=int, default=4)
    p.add_argument("--game", "-g", default="sfiii3nr1")
    args = p.parse_args()

    ids = []
    with open(args.ids_file) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"SKIP {line!r} — missing length"); continue
            cid = parts[0]
            try:
                length = int(parts[1])
            except ValueError:
                print(f"SKIP {line!r} — invalid length"); continue
            if length <= 0:
                print(f"SKIP {line!r} — non-positive length"); continue
            if (CAPTURES_DIR / cid / "input_log.txt").exists():
                print(f"SKIP {cid} (already captured)"); continue
            ids.append((cid, length))
    if not ids:
        print("No replays to capture."); return

    nw = min(args.workers, len(ids))
    chunks = [[] for _ in range(nw)]
    for i, (cid, length) in enumerate(ids):
        chunks[i % nw].append((cid, length))
    print(f"=== {len(ids)} replays  ->  {nw} workers ===")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: _shutdown.set())

    with ThreadPoolExecutor(nw) as pool:
        futures = {pool.submit(capture_chunk, w, c, args.game): w
                   for w, c in enumerate(chunks)}
        failed = 0
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  worker {futures[fut]} FAILED: {e}"); failed += 1
    print(f"=== {'All finished' if not failed else f'{failed} failed'} ===")


if __name__ == "__main__":
    main()
