/** Error reported by QSO: invalid input, or a failure inside the module. */
export class QsoError extends Error {}

/** A graph: a specification such as "metatron", "cycle:8" or "johnson:7,3", or an edge list. */
export type GraphInput = string | { edgeList: string };

export type Generator = "laplacian" | "adjacency";

export interface Register {
  name: string;
  offset: number;
  size: number;
}

export interface QasmResult {
  qubits: number;
  bits: number;
  operations: number;
  /** True when the program has no measurement, reset or condition. */
  unitary: boolean;
  qubit_registers: Register[];
  bit_registers: Register[];
  /**
   * Exact probability of each classical record (bit 0 on the right), or
   * null when the program has more than 4096 measurement branches and
   * `shots` were requested (then only `counts` are computed).
   */
  probabilities: Record<string, number> | null;
  /** Sampled records, when `shots` was given. */
  counts: Record<string, number> | null;
  /** Final state of a unitary program (qubit 0 on the right). */
  state: {
    probabilities: Record<string, number>;
    /** True when only the 1024 most probable states are listed. */
    truncated: boolean;
    /** Amplitudes [re, im] for at most 10 qubits. */
    amplitudes: [number, number][] | null;
  } | null;
  qasm3: string;
  /** OpenQASM 2 export, or null when the program needs OpenQASM 3. */
  qasm2: string | null;
}

export interface GraphSummary {
  name: string;
  nodes: number;
  edges: number;
  min_degree: number;
  max_degree: number;
  regular: boolean;
  connected: boolean;
  bipartite: boolean;
  diameter: number | null;
  triangles: number;
  algebraic_connectivity: number;
}

export interface GraphResult {
  name: string;
  nodes: number;
  /** Edges [u, v, weight]. */
  edges: [number, number, number][];
  /** Node coordinates [x, y, z], when the graph has an embedding. */
  positions: [number, number, number][] | null;
  summary: GraphSummary;
  /** Order of the automorphism group as a decimal string, or null above 2^128. */
  automorphism_order: string | null;
  log10_automorphism_order: number;
  orbits: number[][];
  adjacency_spectrum: number[];
  laplacian_spectrum: number[];
  /** (n, k, λ, μ) when the graph is strongly regular. */
  strongly_regular: [number, number, number, number] | null;
}

export interface WalkResult {
  times: number[];
  /** Node probabilities, one array per time. */
  probabilities: number[][];
  limiting_distribution: number[];
}

export interface SearchResult {
  gamma: number;
  time: number;
  success_probability: number;
  expected_time: number;
  curve: { times: number[]; success_probability: number[] };
}

export interface IsomorphismResult {
  isomorphic: boolean;
  /** Node u of the first graph maps to mapping[u] of the second. */
  mapping: number[] | null;
}

export class Qso {
  /** Call a method of the JSON interface and return its result. */
  call(method: string, params?: Record<string, unknown>): any;
  version(): string;
  qasm(source: string, options?: { shots?: number; seed?: number }): QasmResult;
  graph(graph: GraphInput): GraphResult;
  walk(
    graph: GraphInput,
    options: { start?: number; times: number[]; gamma?: number; generator?: Generator },
  ): WalkResult;
  search(
    graph: GraphInput,
    marked: number[],
    options?: { gamma?: number; generator?: Generator; tMax?: number; points?: number },
  ): SearchResult;
  isomorphism(first: string, second: string): IsomorphismResult;
}

/**
 * Load the module. `source` is a URL or path of qso.wasm (default: the
 * file next to this module), its bytes, or a compiled WebAssembly.Module.
 */
export function load(source?: string | URL | BufferSource | WebAssembly.Module): Promise<Qso>;
