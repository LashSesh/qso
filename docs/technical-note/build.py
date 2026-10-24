"""Build the QSO technical note from its template and the benchmark data.

    python docs/technical-note/build.py                 # HTML + PDF
    python docs/technical-note/build.py --doi 10.5281/zenodo.NNNN

Every number in the note comes from ``data/technical_note.json``
(``cargo run --release --example technical_note``) or
``data/cross_framework.json`` (``benchmarks/cross_framework/run.py``);
``data/tests.json`` holds the test counts. Placeholders in ``note.md``:

* ``{{path:spec}}`` — a value from the data, formatted with the Python
  format ``spec`` (``tn.`` = technical note, ``xf.`` = cross-framework,
  ``tests.`` = test counts; list indices are numbers);
* ``{{@name}}`` — a table produced by ``table_<name>`` below.

The PDF is printed by headless Chromium through Playwright (Node), with
pandoc converting Markdown to HTML (formulas as MathML).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load() -> dict:
    data = {
        "tn": json.loads((DATA / "technical_note.json").read_text()),
        "xf": json.loads((DATA / "cross_framework.json").read_text()),
        "tests": json.loads((DATA / "tests.json").read_text()),
    }
    squares = data["tn"]["graph_classifier"]["squares"]["distinct_graphs"]
    data["derived"] = {
        "max_state_diff": max(data["xf"]["correctness"]["max_amplitude_difference_vs_qso"].values()),
        "girth5_distinct_min": min(d["train_negative"] for d in squares),
        "girth5_distinct_max": max(d["train_negative"] for d in squares),
    }
    return data


def lookup(data: dict, path: str):
    value = data
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


# ---------------------------------------------------------------------------
# Formatting helpers


def sci(x: float, digits: int = 1) -> str:
    """1.2 × 10⁻¹⁵ style."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    sup = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10{sup}"


def ms(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.3g} ms" if seconds * 1e3 < 100 else f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.2f} s"


def pm(d: dict, digits: int = 3) -> str:
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tables


def table_grover_single(d):
    rows = []
    for r in d["tn"]["grover"]["single"]:
        if r["n_qubits"] in (4, 6, 8, 10, 12, 14, 16):
            rows.append([r["n_qubits"], f"{r['items']:,}".replace(",", " "), r["iterations"],
                         f"{r['success']:.6f}", f"{r['theory']:.6f}", sci(r["abs_error"]),
                         f"{r['query_ratio']:.1f}"])
    return table(["n", "N = 2ⁿ", "k", "P (simulated)", "sin²((2k+1)θ)", "|Δ|",
                  "(N+1)/2 ÷ k"], rows)


def table_grover_multi(d):
    rows = [[r["marked"], r["iterations"], f"{r['success']:.6f}", f"{r['theory']:.6f}", sci(r["abs_error"])]
            for r in d["tn"]["grover"]["multi"]]
    return table(["M", "k", "P (simulated)", "sin²((2k+1)θ)", "|Δ|"], rows)


def table_mixing(d):
    rows = [[f"{r['epsilon']:g}", f"{r['mixing_time']:.3f}", f"{r['closed_form']:.3f}"]
            for r in d["tn"]["walks"]["metatron_mixing_times"]]
    return table(["ε", "M_ε (QSO)", "M_ε (closed form)"], rows)


def table_search_scaling(d):
    rows = [[r["nodes"], f"{r['quantum_time']:.2f}", f"{r['success_at_quantum_time']:.12f}",
             f"{r['classical_mean_hitting_time']:.2f}", f"{r['ratio']:.2f}"]
            for r in d["tn"]["walks"]["complete_graph_scaling"]]
    return table(["N", "T = π√N/2", "P(T)", "classical hitting time", "ratio"], rows)


def table_fruit_search(d):
    names = {0: "centre", 1: "inner ring", 7: "outer ring"}
    rows = [[names.get(r["marked"], r["marked"]), f"{r['gamma']:.4f}", f"{r['time']:.2f}",
             f"{r['success']:.4f}", f"{r['expected_time']:.2f}", f"{r['classical_mean_hitting_time']:.2f}"]
            for r in d["tn"]["walks"]["fruit_of_life_search"]]
    return table(["marked vertex", "γ", "T", "P(T)", "T / P(T)", "classical hitting time"], rows)


def table_automorphisms(d):
    rows = [[r["graph"], f"{r['order']:,}".replace(",", " "), r["reference"]]
            for r in d["tn"]["walks"]["automorphism_groups"]]
    return table(["graph", "|Aut| (QSO)", "reference"], rows, "lrr")


def table_qaoa_reference(d):
    q = d["tn"]["qaoa"]
    rows = [[f"ring C₁₀, p = {r['p']}", f"{r['ratio']:.10f}", f"{r['reference']:.10f}", "[6]"]
            for r in q["ring_c10"]]
    rows.append(["cube Q₃, p = 1", f"{q['cube_q3_p1']['ratio']:.6f}", "0.6925", "[5]"])
    rows.append(["Heawood graph, p = 2", f"{q['heawood_p2']['ratio']:.6f}", "0.7559", "[8]"])
    return table(["instance", "ratio (QSO)", "reference", "source"], rows, "lrrl")


def table_qaoa_depth(d):
    q = d["tn"]["qaoa"]
    rows = []
    for k in range(4):
        m = q["metatron_cube"][k] if k < len(q["metatron_cube"]) else None
        p = q["petersen"][k] if k < len(q["petersen"]) else None
        r = q["random_regular"]["layers"][k] if k < len(q["random_regular"]["layers"]) else None
        rows.append([
            k + 1,
            f"{m['ratio']:.6f} / {m['optimal_probability']:.4f}" if m else "—",
            f"{p['ratio']:.4f} / {p['optimal_probability']:.4f}" if p else "—",
            f"{r['ratio_mean']:.4f} ± {r['ratio_std']:.4f}" if r else "—",
        ])
    n = q["random_regular"]["nodes"]
    return table(["p", "Metatron's Cube: ratio / P(opt)", "Petersen: ratio / P(opt)",
                  f"random cubic, n = {n}: ratio"], rows)


def table_vqe(d):
    rows = []
    for r in d["tn"]["vqe"]["spin_models"]:
        rows.append([r["model"], r["ansatz"], r["layers"], r["parameters"], f"{r['energy']:.6f}",
                     f"{r['exact']:.6f}", sci(r["relative_error"])])
    return table(["model", "ansatz", "layers", "parameters", "E (VQE)", "E₀ (Lanczos)",
                  "relative error"], rows, "llrrrrr")


def table_vqc(d):
    rows = [[r["dataset"], pm(r["vqc_train"]), pm(r["vqc_test"]), pm(r["logistic_linear_test"]),
             pm(r["logistic_quadratic_test"])] for r in d["tn"]["vqc"]["datasets"]]
    return table(["dataset", "VQC train", "VQC test", "logistic (linear) test",
                  "logistic (quadratic) test"], rows)


def table_graph_classifier(d):
    rows = []
    for key in ("triangles", "squares"):
        t = d["tn"]["graph_classifier"][key]
        for m in t["models"]:
            rows.append([t["task"], t["nodes"], m["layers"], pm(m["train"]), pm(m["test"]),
                         ms(m["seconds"]["mean"])])
    return table(["task", "nodes", "layers", "train accuracy", "test accuracy", "training time"],
                 rows, "lrrrrr")


def table_bunching(d):
    rows = [[r["photons"], f"{r['ratio']:.10g}", f"{r['reference']:g}"]
            for r in d["tn"]["boson_sampling"]["full_bunching"]]
    return table(["photons n", "P_boson / P_dist (QSO)", "n!"], rows)


def table_boson_distance(d):
    rows = [[r["photons"], r["modes"], f"{int(r['patterns']):,}".replace(",", " "),
             f"{r['tvd_mean']:.3f} ± {r['tvd_std']:.3f}"]
            for r in d["tn"]["boson_sampling"]["boson_vs_distinguishable"]]
    return table(["photons", "modes", "output patterns", "TVD(boson, distinguishable)"], rows)


def _timing(d, kind, n, lib):
    for r in d["xf"]["timing"]:
        if r["kind"] == kind and r["n"] == n and r["library"] == lib:
            return r
    return None


def table_boson_timing(d):
    b = d["tn"]["boson_sampling"]
    perm = {r["n"]: r["seconds"] for r in b["permanent_seconds"]}
    rows = []
    for n, t in perm.items():
        walrus = _timing(d, "permanent", n, "thewalrus")
        rows.append([n, ms(t), ms(walrus["median"]) if walrus else "—"])
    return table(["n", "permanent, QSO", "permanent, The Walrus"], rows)


def table_sampling_timing(d):
    rows = [[r["photons"], r["modes"], ms(r["seconds_per_sample"])]
            for r in d["tn"]["boson_sampling"]["sampling_seconds"]]
    return table(["photons n", "modes m = n²", "time per exact sample"], rows)


LIB_NAMES = {
    "qso": "QSO", "qulacs": "Qulacs", "aer": "Qiskit Aer", "pennylane": "PennyLane Lightning",
    "cirq": "Cirq", "thewalrus": "The Walrus",
}


def table_correctness(d):
    c = d["xf"]["correctness"]
    rows = [[name, sci(v)] for name, v in c["max_amplitude_difference_vs_qso"].items()]
    return table(["framework", "max |ψ_QSO − ψ|"], rows)


def table_layers(d):
    sizes = sorted({r["n"] for r in d["xf"]["timing"] if r["kind"] == "1"})
    libs = ["qso", "qulacs", "aer", "pennylane", "cirq"]
    rows = []
    for lib in libs:
        row = [LIB_NAMES[lib]]
        for n in sizes:
            r = _timing(d, "1", n, lib)
            row.append(f"{ms(r['median'])} ({ms(r['minimum'])})" if r and "median" in r else "—")
        rows.append(row)
    return table(["framework"] + [f"n = {n}" for n in sizes], rows)


def table_deep(d):
    sizes = sorted({r["n"] for r in d["xf"]["timing"] if r["kind"] == "10"})
    rows = []
    for lib in ["qso", "qulacs", "aer", "pennylane"]:
        row = [LIB_NAMES[lib]]
        for n in sizes:
            r = _timing(d, "10", n, lib)
            row.append(f"{ms(r['median'])} ({ms(r['minimum'])})" if r and "median" in r else "—")
        rows.append(row)
    return table(["framework"] + [f"n = {n}, 10 layers" for n in sizes], rows)


def table_gradients(d):
    rows = []
    names = {"qso": "QSO, adjoint", "qulacs": "Qulacs, backprop", "pennylane": "PennyLane Lightning, adjoint"}
    diffs = d["xf"]["correctness"]["gradients_vs_qso"]
    diff_of = {"pennylane": diffs["pennylane-lightning (adjoint)"], "qulacs": diffs["qulacs (backprop)"]}
    for lib in ["qso", "qulacs", "pennylane"]:
        r = _timing(d, "gradient", 10, lib)
        diff = diff_of.get(lib)
        rows.append([names[lib], f"{ms(r['median'])} ({ms(r['minimum'])})",
                     sci(diff["max_gradient_diff"]) if diff else "reference"])
    return table(["method", "value + gradient", "max |∇ − ∇_QSO|"], rows)


def table_performance(d):
    rows = [[r["qubits"], r["gates"], ms(r["seconds"]), f"{r['ns_per_gate_amplitude']:.3f}"]
            for r in d["tn"]["performance"]["statevector_layers"]]
    return table(["qubits", "gates", "time per layer", "ns per gate·amplitude"], rows)


def table_v1_claims(d):
    rows = [
        ["Grover: success 0.0754, “speed-up 7.21”", "withdrawn",
         "13-level system without amplification (0.0754 ≈ 1/13); 13/1.803 compares unrelated quantities"],
        ["Multi-target Grover: success 0.459", "withdrawn", "same engine; no reference value"],
        ["Boson sampling: visibility 0.75, correlation 0.65", "withdrawn",
         "multi-photon path returned an error; both values were literals in the code"],
        ["QML on graphs: test accuracy 0.40", "withdrawn", "not reproducible; below chance for two classes"],
        ["Cross-system scores 0.937 vs 0.735–0.777", "withdrawn",
         "scores of Qiskit, Cirq, PennyLane and ProjectQ were hard-coded constants"],
        ["Execution times 0.189 ms vs 1100–1250 ms", "withdrawn",
         "baseline times were constants; QSO times refer to a 13 × 13 system"],
        ["Integration: fidelity 0.95, ratio 1.0, speed-up 14.42", "withdrawn", "derived from the items above"],
        ["QAOA: ratio 1.0 on 3–5-node graphs", "withdrawn",
         "cost operator was −L/2 on a 13-level system, evolution used the non-unitary I − iHt"],
        ["Walks: stationary peak 0.774, mixing ≈ 19", "withdrawn",
         "degenerate eigenspaces ignored; the correct limiting value is 145/169 ≈ 0.858"],
        ["Walks: quantum 4.15 vs classical 12 steps, speed-up 2.89", "withdrawn",
         "probabilities at successive times treated as independent measurements; the classical value 12 is correct"],
        ["VQC: train and test accuracy 1.0", "withdrawn", "no dataset or split recorded; not reproducible"],
        ["VQE: errors 0.013 / 0.000 / 0.001", "withdrawn",
         "13 × 13 Hamiltonian with trivial spectrum {0, −13}; the “Metatron ansatz” no longer exists"],
        ["“CI-validated”", "withdrawn",
         "the CI only checked that JSON files existed; the comparison script accepted unknown metrics"],
    ]
    return table(["claim in version 1 (26 November 2025)", "status", "reason"], rows, "lll")


TABLES = {name[len("table_"):]: fn for name, fn in globals().items() if name.startswith("table_")}


# ---------------------------------------------------------------------------


def render(template: str, data: dict) -> str:
    def replace(m: re.Match) -> str:
        key = m.group(1).strip()
        if key.startswith("@"):
            return TABLES[key[1:]](data)
        path, _, spec = key.partition(":")
        value = lookup(data, path.strip())
        if spec == "sci":
            return sci(value)
        if spec == "ms":
            return ms(value)
        return format(value, spec) if spec else str(value)

    out = re.sub(r"\{\{(.+?)\}\}", replace, template)
    leftover = re.findall(r"\{\{|\}\}", out)
    if leftover:
        raise SystemExit("unresolved placeholders remain")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", help="DOI of this version (shown on the title page)")
    ap.add_argument("--out", default=str(HERE / "QSO_Technical_Note_v2.pdf"))
    args = ap.parse_args()
    data = load()
    data["doi"] = args.doi or ""
    text = render((HERE / "note.md").read_text(), data)
    if args.doi:
        text = text.replace("<!-- DOI -->", f"DOI: [{args.doi}](https://doi.org/{args.doi})")
    filled = HERE / "build" / "note.md"
    filled.parent.mkdir(exist_ok=True)
    filled.write_text(text)
    html = HERE / "build" / "note.html"
    subprocess.run(
        ["pandoc", str(filled), "--standalone", "--mathml", "--css", str(HERE / "style.css"),
         "--embed-resources", "--metadata", "lang=en", "-o", str(html)],
        check=True,
    )
    subprocess.run(["node", str(HERE / "print.cjs"), str(html), args.out], check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
