// Runs a wasm32-wasip1 test binary under Node's WASI implementation, so that
// `cargo test --target wasm32-wasip1` needs no extra runtime:
//
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node scripts/wasi-runner.mjs" \
//     cargo test --target wasm32-wasip1 -p metatron-qso-rs --no-default-features
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import process from "node:process";

const [, , wasmPath, ...args] = process.argv;
const wasi = new WASI({
  version: "preview1",
  args: [wasmPath, ...args],
  env: process.env,
  preopens: { "/": "/" },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exit(wasi.start(instance));
