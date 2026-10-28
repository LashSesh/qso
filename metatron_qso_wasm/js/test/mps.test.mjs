// Matrix-product states in WebAssembly give the native results: the
// requests in mps_native.json were answered by the native build (the test
// mps_fixture_matches_the_native_build of the Rust crate keeps them
// current) and are replayed here.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { load } from "../index.js";

const qso = await load();
const fixture = JSON.parse(readFileSync(new URL("./mps_native.json", import.meta.url), "utf8"));

function close(a, b, path) {
  if (typeof b === "number") {
    assert.equal(typeof a, "number", path);
    assert.ok(Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(b)), `${path}: ${a} vs ${b}`);
  } else if (Array.isArray(b)) {
    assert.ok(Array.isArray(a), path);
    assert.equal(a.length, b.length, path);
    b.forEach((v, k) => close(a[k], v, `${path}[${k}]`));
  } else if (b !== null && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), path);
    for (const k of Object.keys(b)) close(a[k], b[k], `${path}.${k}`);
  } else {
    assert.equal(a, b, path);
  }
}

test("matrix-product states give the native results", () => {
  assert.ok(fixture.cases.length >= 4);
  for (const [k, { request, response }] of fixture.cases.entries()) {
    const { method, params } = request;
    close(qso.call(method, params), response.ok, `case ${k}`);
  }
});

test("mps() on 1000 qubits, and its input checks", () => {
  const n = 1000;
  let src = `OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[${n}];\nh q[0];\n`;
  for (let k = 1; k < n; k++) src += `cx q[${k - 1}],q[${k}];\n`;
  const r = qso.mps(src, { shots: 20, seed: 3, observables: ["Z0 Z999", "X0"] });
  assert.equal(r.method, "mps");
  assert.deepEqual([...new Set(Object.keys(r.counts))].every((s) => /^(0+|1+)$/.test(s)), true);
  assert.ok(Math.abs(r.expectations[0] - 1) < 1e-9 && Math.abs(r.expectations[1]) < 1e-12);
  assert.equal(r.fidelity_lower_bound, 1);
  assert.throws(() => qso.mps(src, { maxBond: 0 }), /bond dimension/);
  const measured = 'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; bit[1] c; h q[0]; c[0] = measure q[0];';
  assert.throws(() => qso.mps(measured), /shots/);
  assert.equal(qso.mps(measured, { shots: 100 }).counts["0"] + qso.mps(measured, { shots: 100 }).counts["1"], 100);
});
