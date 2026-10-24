# Technical note

*Validated Quantum Algorithm Benchmarks of the Metatron QSO Simulation
Framework* — version 2 of the technical note first published on Zenodo on
26 November 2025. Version 2 withdraws all results of version 1 (see its
Section 1 and [AUDIT_v0.1.md](../AUDIT_v0.1.md)) and replaces them with
validated ones. DOI of version 2:
[10.5281/zenodo.23623226](https://doi.org/10.5281/zenodo.23623226).

| File | Content |
|---|---|
| [QSO_Technical_Note_v2.pdf](QSO_Technical_Note_v2.pdf) | The note |
| [note.md](note.md) | Text with placeholders for every number |
| [data/](data/) | The results the PDF was built from |
| [build.py](build.py), [style.css](style.css), [print.cjs](print.cjs) | Build: fills the placeholders, converts to HTML (pandoc) and prints to PDF (Chromium) |

## Reproducing the data

```bash
cargo run --release --example technical_note -- --out docs/technical-note/data/technical_note.json
python benchmarks/cross_framework/run.py --out docs/technical-note/data/cross_framework.json
```

The second command needs the reference frameworks; see
[benchmarks/cross_framework/](../../benchmarks/cross_framework/). Validation
results are deterministic; timings depend on the machine.

## Building the PDF

Requires Python 3, pandoc ≥ 3 and Node with Playwright's Chromium.

```bash
python docs/technical-note/build.py                                 # without DOI
python docs/technical-note/build.py --doi 10.5281/zenodo.23623226   # as published
```
