"""Build the documentation site: user guide, tutorials and Python reference.

    python docs/build_site.py --out site            # everything
    python docs/build_site.py --out site --no-rust  # skip compiling Rust snippets
                                                    # and running the examples
    python docs/build_site.py --out site --versions cache  # with every release

Steps:

1. execute the notebooks in docs/tutorials and convert them (with their
   outputs) to Markdown pages of the guide (docs/guide/src/tutorials), and
   render the Rust tutorials (docs/guide/rust) into docs/guide/src/rust:
   their code comes from the examples of the crate, between
   `// ANCHOR: name` and `// ANCHOR_END: name` lines, and their output
   from running those examples;
2. run every ```python block of the guide, each in a fresh interpreter, and
   compile and run every ```rust block as an example program against the
   crate in this repository, so that no snippet can go stale;
3. build the guide with mdBook and the Python API reference with pdoc;
4. build the WebAssembly module (scripts/build_wasm.py) and add the
   playground (docs/playground) with the npm package at /playground/;
5. with --versions, add the guide and the Python reference of every release
   since 1.2.0 (the first with a guide) at /<version>/, built from the
   release's tag with that release's package from PyPI (its notebooks are
   executed against it), and a version selector on every guide page.
   Releases are built once and kept in the given cache directory.

Needs the `metatron_qso` package installed in the running interpreter,
nbconvert, ipykernel, matplotlib and pdoc, `mdbook` on the PATH, and the
Rust target wasm32-unknown-unknown.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
GUIDE = DOCS / "guide"
TUTORIALS = DOCS / "tutorials"
RUST_TUTORIALS = GUIDE / "rust"
EXAMPLES = ROOT / "metatron-qso-rs" / "examples"

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


# `{{example file.rs}}`, `{{example file.rs:anchor}}` and `{{output name}}`
# on a line of their own.
DIRECTIVE = re.compile(r"^\{\{(example|output) ([\w.]+)(?::(\w+))?\}\}$", re.M)
ANCHOR = re.compile(r"^\s*// ANCHOR(?:_END)?: \w+$")


def example_code(file: str, anchor: str | None) -> str:
    lines = (EXAMPLES / file).read_text().splitlines()
    if anchor:
        marks = [k for k, line in enumerate(lines)
                 if line.strip() in (f"// ANCHOR: {anchor}", f"// ANCHOR_END: {anchor}")]
        if len(marks) != 2:
            raise SystemExit(f"{file}: anchor {anchor!r} not found exactly once")
        lines = lines[marks[0] + 1:marks[1]]
    code = textwrap.dedent("\n".join(line for line in lines if not ANCHOR.match(line)))
    return "```rust,noplayground\n" + code.strip("\n") + "\n```"


def rust_tutorials(run_examples: bool) -> None:
    out = GUIDE / "src" / "rust"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    outputs: dict[str, str] = {}

    def output(name: str) -> str:
        if not run_examples:
            return "```text\n(not run: the site was built with --no-rust)\n```"
        if name not in outputs:
            print(f"running example {name}", flush=True)
            p = subprocess.run(["cargo", "run", "-q", "--release", "--example", name],
                               cwd=ROOT / "metatron-qso-rs", check=True,
                               capture_output=True, text=True)
            outputs[name] = p.stdout.rstrip("\n")
        return "```text\n" + outputs[name] + "\n```"

    def expand(m: re.Match) -> str:
        kind, name, anchor = m.groups()
        return example_code(name, anchor) if kind == "example" else output(name)

    for page in sorted(RUST_TUTORIALS.glob("*.md")):
        (out / page.name).write_text(DIRECTIVE.sub(expand, page.read_text()))


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
    playground(out / "playground")
    (out / ".nojekyll").write_text("")


def playground(out: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_wasm import build

    package = build()
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(DOCS / "playground", out)
    (out / "qso").mkdir()
    for name in ["index.js", "qso.wasm", "LICENSE"]:
        shutil.copyfile(package / name, out / "qso" / name)


# The first release with a guide.
FIRST_VERSIONED = (1, 2, 0)


def releases() -> list[str]:
    """Release versions with a guide, newest first."""
    tags = subprocess.run(["git", "tag", "--list", "v*"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.split()
    found = []
    for tag in tags:
        m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if m and tuple(map(int, m.groups())) >= FIRST_VERSIONED:
            found.append(tuple(map(int, m.groups())))
    return [".".join(map(str, v)) for v in sorted(found, reverse=True)]


def build_release(version: str, dest: Path) -> None:
    """The guide and Python reference of a release, from its tag."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        tree.mkdir()
        # The whole tree: notebooks read files elsewhere in the repository.
        archive = subprocess.run(["git", "archive", f"v{version}"], cwd=ROOT,
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
        venv = Path(tmp) / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        python = venv / "bin" / "python"
        run([str(python), "-m", "pip", "install", "-q", f"metatron-qso=={version}",
             "-r", str(tree / "docs" / "requirements.txt")])
        guide = tree / "docs" / "guide"
        pages = guide / "src" / "tutorials"
        pages.mkdir(parents=True, exist_ok=True)
        for nb in sorted((tree / "docs" / "tutorials").glob("*.ipynb")):
            run([str(python), "-m", "jupyter", "nbconvert", "--to", "markdown", "--execute",
                 "--ExecutePreprocessor.timeout=900", "--output-dir", str(pages), str(nb)])
        run(["mdbook", "build", str(guide)])
        run([str(python), "-m", "pdoc", "metatron_qso", "-o", str(guide / "book" / "python"),
             "--docformat", "markdown"])
        # Only a complete build enters the cache.
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(guide / "book", dest)


def versions(out: Path, cache: Path) -> None:
    """Every release at out/<version>/, versions.json and the selector."""
    entries = [{"version": "development", "path": ""}]
    for version in releases():
        cached = cache / version
        if not (cached / "index.html").exists():
            print(f"building the documentation of {version}", flush=True)
            build_release(version, cached)
        shutil.copytree(cached, out / version)
        entries.append({"version": version, "path": f"{version}/"})
    (out / "versions.json").write_text(json.dumps(entries, indent=1) + "\n")
    shutil.copyfile(DOCS / "versions" / "version-selector.js", out / "version-selector.js")
    # Guide pages only: not the Python reference, the playground or the
    # coverage report.
    skip = {"python", "playground", "coverage"}
    for page in out.rglob("*.html"):
        parts = page.relative_to(out).parts
        if any(p in skip for p in parts[:-1]):
            continue
        prefix = "../" * (len(parts) - 1)
        html = page.read_text()
        if "version-selector.js" not in html and "</body>" in html:
            tag = f'<script src="{prefix}version-selector.js"></script>\n</body>'
            page.write_text(html.replace("</body>", tag, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    ap.add_argument("--no-rust", action="store_true",
                    help="do not compile the Rust snippets or run the examples")
    ap.add_argument("--versions", type=Path, metavar="CACHE",
                    help="add every release, kept in (and reused from) CACHE")
    args = ap.parse_args()
    notebooks()
    rust_tutorials(run_examples=not args.no_rust)
    snippets(rust=not args.no_rust)
    site(args.out.resolve())
    if args.versions:
        versions(args.out.resolve(), args.versions.resolve())
    print(f"site written to {args.out}")


if __name__ == "__main__":
    main()
