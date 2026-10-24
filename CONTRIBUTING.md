# Contributing

## The rule

QSO makes no claim it cannot test. Every new algorithm, model or number in
the documentation needs a test against an **independent** reference: a
closed form, a published value, a dense-matrix computation, or a second
method (e.g. adjoint vs. finite-difference gradients). Tests that only check
`is_finite()` or echo inputs do not count.

Numbers quoted in `README.md` are reproduced by
[`metatron-qso-rs/tests/validation.rs`](metatron-qso-rs/tests/validation.rs);
performance numbers come from
`cargo run --release --example benchmark_report` and must state the machine.

## Workflow

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p metatron-qso-rs --all-features
cargo test -p metatron-qso-rs --no-default-features --lib   # serial build
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p metatron-qso-rs

# Python bindings (inside a virtualenv)
pip install maturin numpy pytest mypy
maturin develop --release -m metatron_qso_py/Cargo.toml
pytest metatron_qso_py/tests
python -m mypy.stubtest metatron_qso._core --ignore-disjoint-bases \
  --allowlist metatron_qso_py/stubtest-allowlist.txt    # stubs match the module
```

CI additionally runs these checks; run them locally when you touch the
corresponding code:

```bash
# unsafe kernels under Miri (nightly; registers are small under Miri)
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" \
  cargo +nightly miri test -p metatron-qso-rs --lib -- quantum::kernels:: parallel::

# parsers and deserialisers (nightly, cargo install cargo-fuzz)
cd metatron-qso-rs
cargo +nightly fuzz run qasm fuzz/corpus/qasm fuzz/seeds/qasm -- -max_total_time=60
#   targets: qasm, pauli, graph_spec, json

# public API against the last release (cargo install cargo-semver-checks)
cargo semver-checks -p metatron-qso-rs --release-type minor
```

An input that made a fuzz target fail is kept as `fuzz/seeds/<target>/regression-*`
together with a unit test, so that CI replays it on every run.

## Conventions

* Little-endian qubit order, `exp(−iHt)`, `R_P(θ) = exp(−iθP/2)` (see the
  README).
* All randomness goes through `metatron_qso::Rng` with an explicit seed.
* Parallel reductions must stay deterministic: use the helpers in
  `parallel.rs`.
* User input errors return `QsoError`; panics are reserved for violated
  internal invariants (and documented builder misuse).
* Keep `unsafe` confined to the kernels and document why each block is
  sound.

## Releasing

Releases are published by [`.github/workflows/release.yml`](.github/workflows/release.yml).

1. Set the new version in `Cargo.toml` (`[workspace.package]` and the
   `metatron-qso-rs` entry in `[workspace.dependencies]`) and in
   `metatron_qso_py/pyproject.toml`, and add a `## [X.Y.Z]` section to
   `CHANGELOG.md`. Merge this into `main`.
2. Optional dry run: start the *Release* workflow manually (Actions →
   Release → Run workflow). It runs every check and build and publishes
   nothing.
3. Tag the merge commit and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "QSO X.Y.Z" && git push origin vX.Y.Z
   ```

   The workflow checks that tag, versions and changelog agree, builds and
   tests the wheels and the source distribution, then publishes
   `metatron-qso-rs` to crates.io and `metatron-qso` to PyPI and creates the
   GitHub release with the changelog section as notes.

Publishing uses Trusted Publishing (OpenID Connect); no registry tokens are
stored. The publishing jobs run in the GitHub environment `release`, which
must be approved by the maintainer, and PyPI and crates.io each trust
`release.yml` in that environment (setup in the header of the workflow).
Versions already on a registry are skipped, so a failed run can be re-run.
