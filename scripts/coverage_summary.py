"""Markdown summary of a cargo-llvm-cov JSON report (`--json --summary-only`):
line, region, function and branch coverage in total and per component (the
library, the command-line tool and the Python bindings), and the ten files
with the most uncovered lines.

    python scripts/coverage_summary.py coverage.json
"""

import json
import sys


def main(path: str) -> None:
    report = json.load(open(path))
    data = report["data"][0]
    totals = data["totals"]
    print("## Test coverage of the Rust code\n")
    print("| Measure | Covered | Total | Percent |")
    print("|---|---:|---:|---:|")
    for key in ("lines", "branches", "regions", "functions"):
        t = totals.get(key)
        if t and t.get("count"):
            print(f"| {key} | {t['covered']} | {t['count']} | {t['percent']:.1f} % |")
    components = {
        "library (`metatron-qso-rs/src`)": lambda n: "metatron-qso-rs/src/" in n and "/src/bin/" not in n,
        "command line (`src/bin/qso.rs`)": lambda n: "/src/bin/" in n,
        "Python bindings (`metatron_qso_py/src`)": lambda n: "metatron_qso_py/src/" in n,
    }
    rows = []
    for label, member in components.items():
        chosen = [f for f in data.get("files", []) if member(f["filename"])]
        if chosen:
            count = sum(f["summary"]["lines"]["count"] for f in chosen)
            covered = sum(f["summary"]["lines"]["covered"] for f in chosen)
            rows.append((label, covered, count))
    if rows:
        print("\n| Component | Lines covered | Lines | Percent |")
        print("|---|---:|---:|---:|")
        for label, covered, count in rows:
            print(f"| {label} | {covered} | {count} | {100 * covered / max(count, 1):.1f} % |")
    files = sorted(
        data.get("files", []),
        key=lambda f: f["summary"]["lines"]["count"] - f["summary"]["lines"]["covered"],
        reverse=True,
    )
    if files:
        print("\n| File | Uncovered lines | Line coverage |")
        print("|---|---:|---:|")
        for f in files[:10]:
            lines = f["summary"]["lines"]
            name = f["filename"].split("/qso/")[-1]
            print(f"| `{name}` | {lines['count'] - lines['covered']} | {lines['percent']:.1f} % |")


if __name__ == "__main__":
    main(sys.argv[1])
