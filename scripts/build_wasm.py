"""Build the WebAssembly module into the npm package.

    python scripts/build_wasm.py

Compiles `metatron_qso_wasm` for wasm32-unknown-unknown (release) and
copies the module and the licence into `metatron_qso_wasm/js`, which is
then the complete npm package `metatron-qso` (test it with `npm test`
there). Needs the Rust target: `rustup target add wasm32-unknown-unknown`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "metatron_qso_wasm" / "js"


def build() -> Path:
    """Build the module and return the package directory."""
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--target", "wasm32-unknown-unknown",
         "-p", "metatron_qso_wasm"],
        cwd=ROOT, check=True,
    )
    module = ROOT / "target" / "wasm32-unknown-unknown" / "release" / "metatron_qso_wasm.wasm"
    shutil.copyfile(module, PACKAGE / "qso.wasm")
    shutil.copyfile(ROOT / "LICENSE", PACKAGE / "LICENSE")
    print(f"{PACKAGE / 'qso.wasm'}: {module.stat().st_size} bytes")
    return PACKAGE


if __name__ == "__main__":
    build()
