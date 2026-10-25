// The WebAssembly module against the test vectors of the native build:
// the OpenQASM example programs (metatron-qso-rs/tests/openqasm.rs) and
// known graph and walk results.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { QsoError, load } from "../index.js";

const qso = await load();
const examples = new URL("../../../metatron-qso-rs/tests/data/openqasm/", import.meta.url);
const read = (version, name) => readFileSync(new URL(`${version}/${name}.qasm`, examples), "utf8");
const close = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);
// Keys are sorted here: JavaScript lists integer-like keys ("111") first.
const entries = (r) => Object.entries(r.probabilities).sort(([a], [b]) => (a < b ? -1 : 1));

test("version matches the package", () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.equal(qso.version(), pkg.version);
});

test("OpenQASM 2 examples give their known results", () => {
  // Ripple-carry adder: 1 + 15 = 16.
  const adder = qso.qasm(read("v2", "adder"));
  assert.deepEqual(entries(adder).map(([k, p]) => [parseInt(k, 2), p]), [[16, 1]]);
  // Phase estimation of e^{i·3π/8}: the estimate 3/16 is exact in four bits.
  for (const name of ["pea_3_pi_8", "ipea_3_pi_8"]) {
    const d = entries(qso.qasm(read("v2", name)));
    assert.equal(d.length, 1);
    const value = parseInt(d[0][0], 2);
    let digits = 0;
    for (let k = 0; k < 4; k++) digits += ((value >> k) & 1) / 2 ** (4 - k);
    close(digits * 2 * Math.PI, (3 * Math.PI) / 8);
  }
  // Teleportation: the third bit reads 1 with probability sin²(0.15).
  for (const name of ["teleport", "teleportv2"]) {
    const r = qso.qasm(read("v2", name));
    const p1 = entries(r)
      .filter(([k]) => k[k.length - 3] === "1")
      .reduce((s, [, p]) => s + p, 0);
    close(p1, Math.sin(0.15) ** 2);
  }
  // W state and QFT of |0101⟩.
  const w = qso.qasm(read("v2", "W-state"));
  assert.deepEqual(entries(w).map(([k]) => parseInt(k, 2)), [1, 2, 4]);
  for (const [, p] of entries(w)) close(p, 1 / 3, 1e-5);
  const qft = entries(qso.qasm(read("v2", "qft")));
  assert.equal(qft.length, 16);
  for (const [, p] of qft) close(p, 1 / 16);
  for (const name of ["invalid_gate_no_found", "invalid_missing_semicolon"]) {
    assert.throws(() => qso.qasm(read("v2", name)), QsoError);
  }
});

test("OpenQASM 3 examples agree with their OpenQASM 2 versions", () => {
  for (const name of ["teleport", "inverseqft1", "inverseqft2", "qft", "rb", "qpt"]) {
    const v3 = qso.qasm(read("v3", name)).probabilities;
    const v2 = qso.qasm(read("v2", name)).probabilities;
    assert.deepEqual(Object.keys(v3).sort(), Object.keys(v2).sort(), name);
    for (const k of Object.keys(v3)) close(v3[k], v2[k], 1e-10);
  }
  assert.throws(() => qso.qasm(read("v3", "vqe")), /not supported|unsupported|expected/);
});

test("round trips and sampling", () => {
  const r = qso.qasm(read("v2", "inverseqft1"), { shots: 4000, seed: 3 });
  const back = qso.qasm(r.qasm3);
  for (const k of Object.keys(r.probabilities)) close(back.probabilities[k], r.probabilities[k], 1e-10);
  const total = Object.values(r.counts).reduce((s, c) => s + c, 0);
  assert.equal(total, 4000);
  // The same seed gives the same counts.
  assert.deepEqual(qso.qasm(read("v2", "inverseqft1"), { shots: 4000, seed: 3 }).counts, r.counts);
  const ghz = qso.qasm('OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; h q[0]; cx q[0], q[1]; cx q[1], q[2];');
  assert.deepEqual(Object.keys(ghz.state.probabilities).sort(), ["000", "111"]);
  close(ghz.state.amplitudes[7][0], Math.SQRT1_2);
});

test("graphs, walks and search", () => {
  const cube = qso.graph("metatron");
  assert.equal(cube.nodes, 13);
  assert.equal(cube.edges.length, 78);
  assert.equal(cube.automorphism_order, "6227020800"); // 13!
  assert.equal(qso.graph("petersen").automorphism_order, "120");
  assert.deepEqual(qso.graph("paley:13").strongly_regular, [13, 6, 2, 3]);
  const triangle = qso.graph({ edgeList: "0 1\n1 2\n2 0\n" });
  assert.equal(triangle.summary.triangles, 1);
  assert.equal(qso.isomorphism("petersen", "kneser:5,2").isomorphic, true);
  assert.equal(qso.isomorphism("petersen", "cycle:10").isomorphic, false);
  // Perfect state transfer on the path of two nodes at t = π/2 (adjacency, γ = 1).
  const w = qso.walk("path:2", { start: 0, times: [Math.PI / 2], generator: "adjacency" });
  close(w.probabilities[0][1], 1);
  // Search on Metatron's Cube (K13) finds any node with probability 1.
  const s = qso.search("metatron", [5]);
  close(s.success_probability, 1, 1e-9);
  close(s.gamma, 1 / 13, 1e-8);
});

test("invalid input is a QsoError, and the module stays usable", () => {
  assert.throws(() => qso.call("nope"), /unknown method/);
  assert.throws(() => qso.walk("cycle:5", { times: [1], start: 9 }), QsoError);
  assert.throws(() => qso.graph("cycle:3000"), /at most 2048/);
  assert.throws(() => qso.graph(42), QsoError);
  assert.equal(qso.graph("cycle:5").nodes, 5);
});
