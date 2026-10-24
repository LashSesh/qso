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
pip install maturin numpy pytest
maturin develop --release -m metatron_qso_py/Cargo.toml
pytest metatron_qso_py/tests
```

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

The workflow needs two repository secrets: `CRATES_IO` (crates.io token with
the `publish-update` scope for `metatron-qso-rs`) and `PYPI` (PyPI token
scoped to the project `metatron-qso`). Versions
already on a registry are skipped, so a failed run can be re-run.
