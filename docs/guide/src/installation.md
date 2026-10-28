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
everything else uses rayon. The pool runs on `QSO_NUM_THREADS` threads,
else on `RAYON_NUM_THREADS` (which rayon reads too), else on all cores;
`QSO_POOL_SPIN_US` sets the spin time in microseconds (0 parks at once).

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
