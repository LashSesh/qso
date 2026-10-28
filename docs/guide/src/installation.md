# Installation

## Python

```bash
pip install metatron-qso
```

One `abi3` wheel covers CPython 3.9 and newer. Prebuilt wheels exist for
these platforms:

- Linux x86_64 and aarch64, glibc and musl;
- macOS on Intel and Apple silicon;
- Windows x86_64 and arm64.

The only runtime dependency is NumPy. On other platforms pip builds the
package from source, which needs Rust 1.89 or newer.

```python
import metatron_qso as q

print(q.__version__, q.MAX_QUBITS)
```

## Rust

```bash
cargo add metatron-qso-rs
```

The crate is imported as `metatron_qso`. It has two features:

| Feature | Default | Effect |
|---|---|---|
| `parallel` | yes | Multi-threaded kernels and reductions (rayon) |
| `cli` | no | Builds the `qso` command-line tool |

Without `parallel`, everything runs on one thread. The results are
bit-for-bit identical either way: parallel sums use a fixed chunking that
does not depend on the number of threads.

The state-vector kernels run on a worker pool of their own whose threads
spin for 200 µs after each job before they sleep, so a simulation that
follows a short pause (an optimiser step, Python code) finds them awake;
everything else (batches of parameter vectors, trajectories, restarts)
uses rayon. Two environment variables, read once per process:

| Variable | Effect |
|---|---|
| `QSO_NUM_THREADS` | threads of the pool (the calling thread included); unset, 0 or not a number: `RAYON_NUM_THREADS`, else all cores. It does not limit rayon, which reads `RAYON_NUM_THREADS` |
| `QSO_POOL_SPIN_US` | how long the pool's threads spin after a job, in microseconds (default 200, at most 10⁶; not a number: 200). 0 parks them at once |

Spinning trades CPU time for latency: with a short simulation every
millisecond, the process used about 40 % more CPU time than with rayon in
one measurement. Waiting threads yield the core regularly, so several
QSO processes on one machine share it; to save the CPU time anyway, set
`QSO_POOL_SPIN_US=0`.

## Command-line tool

```bash
cargo install metatron-qso-rs --features cli
qso graph metatron
```

## From source

No C or C++ toolchain is needed, only Rust, plus maturin for the Python
package:

```bash
git clone https://github.com/LashSesh/qso && cd qso
cargo test -p metatron-qso-rs
pip install maturin
maturin develop --release -m metatron_qso_py/Cargo.toml
```
