"""Markdown summary of a cargo-llvm-cov JSON report (`--json --summary-only`):
line, region, function and branch coverage of the crate, and the ten files
with the most uncovered lines.

    python scripts/coverage_summary.py coverage.json
"""

import json
import sys


def main(path: str) -> None:
    report = json.load(open(path))
    data = report["data"][0]
    totals = data["totals"]
    print("## Test coverage of `metatron-qso-rs`\n")
    print("| Measure | Covered | Total | Percent |")
    print("|---|---:|---:|---:|")
    for key in ("lines", "branches", "regions", "functions"):
        t = totals.get(key)
        if t and t.get("count"):
            print(f"| {key} | {t['covered']} | {t['count']} | {t['percent']:.1f} % |")
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
            name = f["filename"].split("metatron-qso-rs/")[-1]
            print(f"| `{name}` | {lines['count'] - lines['covered']} | {lines['percent']:.1f} % |")


if __name__ == "__main__":
    main(sys.argv[1])
