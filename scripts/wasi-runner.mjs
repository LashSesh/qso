// Runs a wasm32-wasip1 test binary under Node's WASI implementation, so that
// `cargo test --target wasm32-wasip1` needs no extra runtime:
//
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node scripts/wasi-runner.mjs" \
//     cargo test --target wasm32-wasip1 -p metatron-qso-rs --no-default-features
import { readFile } from "node:fs/promises";
import v8 from "node:v8";
import { WASI } from "node:wasi";
import process from "node:process";

// Node's WASI writes (uvwasi_fd_write) run as V8 fast API calls and can
// trigger a garbage collection there when external memory grows, which
// crashes Node 20–22 (segmentation fault in MarkCompactCollector) for
// some heap sizes. Without fast API calls the writes take the regular
// path.
v8.setFlagsFromString("--no-turbo-fast-api-calls");

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
