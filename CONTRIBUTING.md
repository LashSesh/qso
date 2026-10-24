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
