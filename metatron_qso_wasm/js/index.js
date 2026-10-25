// QSO in WebAssembly: a thin wrapper over the module's JSON interface.
//
// The module (qso.wasm) needs no imports. A request is written into its
// memory as UTF-8 JSON, qso_call answers with {"ok": ...} or
// {"error": ...}, and both buffers are freed again.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Error reported by QSO: invalid input, or a failure inside the module. */
export class QsoError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = "QsoError";
  }
}

async function compile(source) {
  if (source instanceof WebAssembly.Module) return source;
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    return WebAssembly.compile(source);
  }
  const url = new URL(source ?? "./qso.wasm", import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return WebAssembly.compile(await readFile(url));
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new QsoError(`cannot load ${url}: HTTP ${response.status}`);
  }
  return WebAssembly.compile(await response.arrayBuffer());
}

function graphParams(graph, key = "graph") {
  if (typeof graph === "string") return { [key]: graph };
  if (key === "graph" && graph && typeof graph.edgeList === "string") {
    return { edge_list: graph.edgeList };
  }
  throw new QsoError(
    key === "graph"
      ? "a graph is a specification string or {edgeList: string}"
      : "a graph is a specification string",
  );
}

/** A QSO module instance. Create one with {@link load}. */
export class Qso {
  #module;
  #exports;

  constructor(module) {
    this.#module = module;
    this.#exports = new WebAssembly.Instance(module, {}).exports;
  }

  /** Call a method of the JSON interface and return its result. */
  call(method, params = {}) {
    const request = encoder.encode(JSON.stringify({ method, params }));
    let text;
    try {
      const e = this.#exports;
      const ptr = e.qso_alloc(request.length);
      new Uint8Array(e.memory.buffer, ptr, request.length).set(request);
      const out = e.qso_call(ptr, request.length);
      const len = e.qso_response_len();
      text = decoder.decode(new Uint8Array(e.memory.buffer, out, len));
      e.qso_free(out, len);
    } catch (cause) {
      // A trap (for example memory exhausted) leaves the instance in an
      // unknown state: continue with a fresh one.
      this.#exports = new WebAssembly.Instance(this.#module, {}).exports;
      throw new QsoError(`QSO failed: ${cause.message}`, { cause });
    }
    const response = JSON.parse(text);
    if ("error" in response) throw new QsoError(response.error);
    return response.ok;
  }

  /** Version of the QSO library inside the module. */
  version() {
    return this.call("version").version;
  }

  /** Run an OpenQASM 2 or 3 program. */
  qasm(source, { shots, seed } = {}) {
    return this.call("qasm", { source, shots, seed });
  }

  /** Structure, spectra and symmetry group of a graph. */
  graph(graph) {
    return this.call("graph", graphParams(graph));
  }

  /** Continuous-time quantum walk from a node, at the given times. */
  walk(graph, { start, times, gamma, generator } = {}) {
    return this.call("walk", { ...graphParams(graph), start, times, gamma, generator });
  }

  /** Spatial search for marked nodes. */
  search(graph, marked, { gamma, generator, tMax, points } = {}) {
    return this.call("search", {
      ...graphParams(graph),
      marked,
      gamma,
      generator,
      t_max: tMax,
      points,
    });
  }

  /** Whether two graphs are isomorphic, with a mapping when they are. */
  isomorphism(first, second) {
    return this.call("isomorphism", {
      ...graphParams(first, "first"),
      ...graphParams(second, "second"),
    });
  }
}

/**
 * Load the module. `source` is a URL or path of qso.wasm (default: the
 * file next to this module), its bytes, or a compiled WebAssembly.Module.
 */
export async function load(source) {
  return new Qso(await compile(source));
}
