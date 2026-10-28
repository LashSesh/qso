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

# Nightly for every cargo below, maturin's included (branch coverage needs
# -Z coverage-options, which stable rustc rejects).
export RUSTUP_TOOLCHAIN=nightly

# Instrument every build below (cargo-llvm-cov 0.9 does it through a rustc
# wrapper) in a target directory of its own, so that cleaning it and the
# instrumented binaries leave the normal build alone.
export CARGO_TARGET_DIR=$root/target/llvm-cov-target
source <(cargo llvm-cov show-env --export-prefix --branch)
cargo llvm-cov clean --workspace

# A failing test does not discard the report; the script fails at the end.
status=0
cargo test --release -p metatron-qso-rs --all-features --no-fail-fast || status=$?

venv=$CARGO_TARGET_DIR/coverage-venv
python3 -m venv "$venv"
"$venv/bin/python" -m pip install -q maturin pytest numpy
(
  # maturin builds through cargo, so with the instrumentation of show-env.
  # A wheel installed into this venv only: `maturin develop` would put the
  # instrumented module into the source tree, in place of the normal one.
  # (set -e does not apply inside a subshell tested by ||: chain with &&.)
  source "$venv/bin/activate" &&
    rm -rf "$CARGO_TARGET_DIR/coverage-wheels" &&
    maturin build --release -q -m metatron_qso_py/Cargo.toml -o "$CARGO_TARGET_DIR/coverage-wheels" &&
    python -m pip install -q --force-reinstall "$CARGO_TARGET_DIR"/coverage-wheels/*.whl &&
    python -m pytest -q metatron_qso_py/tests
) || status=$?

# Only the crate and the Python bindings, without test files (test modules
# inside source files still count).
ignore='(/tests/|/tests\.rs$|/benches/|/examples/|metatron_qso_wasm/)'
cargo llvm-cov report --release --branch --ignore-filename-regex "$ignore" \
  --json --summary-only --output-path "$out/summary.json"
cargo llvm-cov report --release --branch --ignore-filename-regex "$ignore" \
  --summary-only | tee "$out/coverage.txt"
if [ "$html" = "--html" ]; then
  cargo llvm-cov report --release --branch --ignore-filename-regex "$ignore" \
    --html --output-dir "$out"
fi
exit $status
