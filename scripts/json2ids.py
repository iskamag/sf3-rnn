#!/usr/bin/env python3
"""Convert the bulk replay JSON into the ids_file format recorder.py expects.

Outputs lines: <quarkid> <duration_seconds>
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_file", type=Path)
    args = p.parse_args()

    data = json.loads(args.json_file.read_text())

    lines = []
    for entry in data:
        if entry.get("num_matches", 0) == 0:
            continue
        qid = entry["quarkid"]
        dur = int(entry["duration"])
        lines.append(f"{qid} {dur}")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stderr.write(f"Wrote {len(lines)} entries\n")


if __name__ == "__main__":
    main()
