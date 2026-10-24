"""Compare a benchmark_report result with the recorded history.

    cargo run --release --example benchmark_report -- --json new.json
    python benchmarks/track.py --history results/ --new new.json [--store]

Every metric of ``new.json`` is compared with the median of its last
``--window`` recorded values. A metric slower by more than ``--threshold``
(default 25 %) is a regression: it is reported as a GitHub Actions error and
the exit status is 1. With ``--store`` the new result is added to the
history as ``<UTC date>-<commit>.json`` (also when it contains regressions,
so that the next run compares against it).

Timings on shared CI machines vary by up to ~1.5× between runs; the median
over several weeks and the threshold keep single outliers from being
flagged, a lasting slowdown is flagged at the second or third run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path


def load_history(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def compare(history: list[dict], new: dict, window: int, threshold: float) -> tuple[list[str], list[str]]:
    rows, regressions = [], []
    for name, seconds in sorted(new["metrics"].items()):
        past = [h["metrics"][name] for h in history if name in h.get("metrics", {})][-window:]
        if not past:
            rows.append(f"| {name} | {seconds:.3g} s | — | new |")
            continue
        reference = statistics.median(past)
        change = seconds / reference - 1.0
        flag = "**regression**" if change > threshold else ""
        rows.append(f"| {name} | {seconds:.3g} s | {reference:.3g} s | {change:+.0%} {flag} |")
        if change > threshold:
            regressions.append(f"{name}: {seconds:.3g} s vs median {reference:.3g} s of the last {len(past)} runs ({change:+.0%})")
    return rows, regressions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--store", action="store_true")
    args = ap.parse_args()

    new = json.loads(args.new.read_text())
    history = load_history(args.history)
    rows, regressions = compare(history, new, args.window, args.threshold)

    summary = [
        f"## Benchmark tracking ({len(history)} earlier runs, threshold {args.threshold:.0%})",
        "",
        "| metric | this run | median of last runs | change |",
        "|---|---:|---:|---:|",
        *rows,
        "",
    ]
    text = "\n".join(summary)
    print(text)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    for r in regressions:
        print(f"::error title=Benchmark regression::{r}")

    if args.store:
        args.history.mkdir(parents=True, exist_ok=True)
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        commit = (new.get("commit") or "unknown")[:12]
        new["date"] = date
        (args.history / f"{date}-{commit}.json").write_text(json.dumps(new, indent=2) + "\n")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
