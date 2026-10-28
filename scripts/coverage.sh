#!/usr/bin/env bash
# Line and branch coverage of the Rust code, merged from
#   1. the crate's tests (unit, validation, property and differential tests,
#      and tests/cli.rs, which runs the command-line tool as a program), and
#   2. the Python package's tests, run against an instrumented build of the
#      extension module,
# into one report. Needs nightly Rust with llvm-tools-preview, cargo-llvm-cov,
# and a Python with maturin, pytest and numpy.
#
#   scripts/coverage.sh OUT_DIR [--html]
#
# writes OUT_DIR/summary.json (cargo llvm-cov --json --summary-only),
# OUT_DIR/coverage.txt and, with --html, the HTML report in OUT_DIR.
set -euo pipefail

out=${1:?usage: scripts/coverage.sh OUT_DIR [--html]}
html=${2:-}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p "$out"
out=$(cd "$out" && pwd)

# Instrument every build below and collect the profiles in one place.
source <(cargo +nightly llvm-cov show-env --export-prefix --branch)
export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
cargo +nightly llvm-cov clean --workspace

cargo +nightly test --release -p metatron-qso-rs --all-features

venv=$CARGO_TARGET_DIR/coverage-venv
python3 -m venv "$venv"
"$venv/bin/python" -m pip install -q maturin pytest numpy
(
  # maturin builds with the instrumented RUSTFLAGS from show-env.
  source "$venv/bin/activate"
  cd metatron_qso_py
  maturin develop --release -q
  python -m pytest -q tests
)

# Only the crate and the Python bindings; test code is left out.
ignore='(/tests/|/benches/|/examples/|metatron_qso_wasm/)'
cargo +nightly llvm-cov report --release --branch --ignore-filename-regex "$ignore" \
  --json --summary-only --output-path "$out/summary.json"
cargo +nightly llvm-cov report --release --branch --ignore-filename-regex "$ignore" \
  --summary-only | tee "$out/coverage.txt"
if [ "$html" = "--html" ]; then
  cargo +nightly llvm-cov report --release --branch --ignore-filename-regex "$ignore" \
    --html --output-dir "$out"
fi
