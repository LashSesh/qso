"""Build the documentation site: user guide, tutorials and Python reference.

    python docs/build_site.py --out site            # everything
    python docs/build_site.py --out site --no-rust  # skip compiling Rust snippets

Steps:

1. execute the notebooks in docs/tutorials and convert them (with their
   outputs) to Markdown pages of the guide (docs/guide/src/tutorials);
2. run every ```python block of the guide, each in a fresh interpreter, and
   compile and run every ```rust block as an example program against the
   crate in this repository, so that no snippet can go stale;
3. build the guide with mdBook and the Python API reference with pdoc.

Needs the `metatron_qso` package installed in the running interpreter,
nbconvert, ipykernel, matplotlib and pdoc, and `mdbook` on the PATH.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
GUIDE = DOCS / "guide"
TUTORIALS = DOCS / "tutorials"

FENCE = re.compile(r"^```(\w+)\n(.*?)^```", re.M | re.S)


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def notebooks() -> None:
    out = GUIDE / "src" / "tutorials"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for nb in sorted(TUTORIALS.glob("*.ipynb")):
        run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "markdown", "--execute",
             "--ExecutePreprocessor.timeout=900", "--output-dir", str(out), str(nb)])


def snippets(rust: bool) -> None:
    pages = sorted((GUIDE / "src").glob("*.md"))
    blocks = [(page, lang, code) for page in pages
              for lang, code in FENCE.findall(page.read_text())]
    python = [(p, c) for p, lang, c in blocks if lang == "python"]
    for k, (page, code) in enumerate(python):
        print(f"python snippet {k + 1}/{len(python)} from {page.name}", flush=True)
        subprocess.run([sys.executable, "-c", code], check=True, cwd=tempfile.gettempdir(),
                       stdout=subprocess.DEVNULL)
    rust_blocks = [(p, c) for p, lang, c in blocks if lang == "rust"]
    if not rust or not rust_blocks:
        return
    with tempfile.TemporaryDirectory() as tmp:
        crate = Path(tmp)
        (crate / "examples").mkdir()
        (crate / "src").mkdir()
        (crate / "src" / "lib.rs").write_text("")
        (crate / "Cargo.toml").write_text(
            '[package]\nname = "guide-snippets"\nversion = "0.0.0"\nedition = "2024"\n'
            "publish = false\n\n[dependencies]\n"
            f'metatron-qso-rs = {{ path = "{(ROOT / "metatron-qso-rs").as_posix()}" }}\n\n[workspace]\n'
        )
        for k, (page, code) in enumerate(rust_blocks):
            (crate / "examples" / f"{page.stem}_{k}.rs").write_text(code)
        target = ROOT / "target" / "guide-snippets"
        for k, (page, _) in enumerate(rust_blocks):
            print(f"rust snippet {k + 1}/{len(rust_blocks)} from {page.name}", flush=True)
            run(["cargo", "run", "-q", "--release", "--example", f"{page.stem}_{k}",
                 "--target-dir", str(target)], cwd=crate, stdout=subprocess.DEVNULL)


def site(out: Path) -> None:
    run(["mdbook", "build", str(GUIDE)])
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(GUIDE / "book", out)
    run([sys.executable, "-m", "pdoc", "metatron_qso", "-o", str(out / "python"),
         "--docformat", "markdown"])
    (out / ".nojekyll").write_text("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    ap.add_argument("--no-rust", action="store_true", help="do not compile the Rust snippets")
    args = ap.parse_args()
    notebooks()
    snippets(rust=not args.no_rust)
    site(args.out.resolve())
    print(f"site written to {args.out}")


if __name__ == "__main__":
    main()
