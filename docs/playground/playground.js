// QSO playground: the npm package's WebAssembly module, run in a worker.

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Worker calls.
// ---------------------------------------------------------------------------

// A call that runs longer than this stops the worker; a fresh one takes
// over, so one heavy request cannot block the page's later requests.
const TIMEOUT_MS = 30000;
const pending = new Map();
let nextId = 0;
let worker;

function startWorker() {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    const entry = pending.get(data.id);
    if (!entry) return;
    pending.delete(data.id);
    clearTimeout(entry.timer);
    if ("error" in data) entry.reject(new Error(data.error));
    else entry.resolve(data.ok);
  };
}
startWorker();

function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => {
      worker.terminate();
      for (const entry of pending.values()) {
        clearTimeout(entry.timer);
        entry.reject(new Error(`Stopped after ${TIMEOUT_MS / 1000} s; try a smaller input.`));
      }
      pending.clear();
      startWorker();
    }, TIMEOUT_MS);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, method, params });
  });
}

/** Run `task` with the button disabled and the outcome in the status line. */
async function busy(button, status, task) {
  button.disabled = true;
  status.textContent = "Running…";
  status.classList.remove("error");
  const start = performance.now();
  try {
    const message = await task();
    status.textContent = `${message ?? "Done"} (${Math.round(performance.now() - start)} ms)`;
  } catch (e) {
    status.textContent = e.message;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Drawing helpers.
// ---------------------------------------------------------------------------

const SVG = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, parent = null) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, content, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = content;
  return t;
}

const tooltip = $("tooltip");
function showTip(event, content) {
  tooltip.textContent = content;
  tooltip.hidden = false;
  const pad = 12;
  const { innerWidth: w } = window;
  const box = tooltip.getBoundingClientRect();
  const x = event.clientX + pad + box.width > w ? event.clientX - pad - box.width : event.clientX + pad;
  tooltip.style.left = `${Math.max(4, x)}px`;
  tooltip.style.top = `${event.clientY + pad}px`;
}
function hideTip() {
  tooltip.hidden = true;
}

const fmt = (p) => (p === 0 ? "0" : p < 1e-3 ? p.toExponential(2) : p.toFixed(4));

/** Gridline values for [0, max]. */
function ticks(max, count = 4) {
  const raw = max / count;
  const step = [1, 2, 2.5, 5, 10].map((m) => m * 10 ** Math.floor(Math.log10(raw))).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

/**
 * Vertical bars: `items` = [{label, value, tip}], one series on a 0-based
 * axis, a hover tooltip per bar.
 */
function barChart(container, items, { max } = {}) {
  container.replaceChildren();
  const W = 520, H = 240, left = 44, right = 8, top = 8;
  const rotate = items.length > 16;
  const bottom = rotate ? 64 : 24;
  const top_ = max ?? Math.max(...items.map((d) => d.value), 1e-12);
  const scaleMax = ticks(top_).at(-1) >= top_ ? ticks(top_).at(-1) : top_;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const ph = H - top - bottom, pw = W - left - right;
  const y = (v) => top + ph - (v / scaleMax) * ph;
  for (const t of ticks(scaleMax)) {
    el("line", { class: "gridline", x1: left, x2: W - right, y1: y(t), y2: y(t) }, svg);
    text(svg, left - 6, y(t) + 4, t, { "text-anchor": "end" });
  }
  const slot = pw / items.length;
  const gap = Math.min(2, slot * 0.2);
  const barW = Math.max(1, slot - gap);
  items.forEach((d, i) => {
    const x = left + i * slot + gap / 2;
    const h = Math.max(0, y(0) - y(d.value));
    const r = Math.min(4, barW / 2, h);
    // Rounded at the data end, square at the baseline.
    const path = `M${x},${y(0)} V${y(0) - h + r} q0,${-r} ${r},${-r} H${x + barW - r} q${r},0 ${r},${r} V${y(0)} Z`;
    const hit = el("rect", { class: "hit", x: left + i * slot, y: top, width: slot, height: ph }, svg);
    const bar = el("path", { class: "bar", d: h > 0 ? path : "" }, svg);
    hit.addEventListener("mousemove", (e) => {
      bar.classList.add("active");
      showTip(e, d.tip);
    });
    hit.addEventListener("mouseleave", () => {
      bar.classList.remove("active");
      hideTip();
    });
    if (items.length <= 64) {
      const lx = x + barW / 2, ly = H - bottom + 14;
      // Long bit strings (thousands of qubits) show their last digits.
      const label = d.label.length > 16 ? `…${d.label.slice(-12)}` : d.label;
      text(svg, lx, ly, label, rotate
        ? { transform: `rotate(-60 ${lx} ${ly})`, "text-anchor": "end" }
        : { "text-anchor": "middle" });
    }
  });
  el("line", { class: "axis", x1: left, x2: W - right, y1: y(0), y2: y(0) }, svg);
  svg.setAttribute("aria-label", items.map((d) => `${d.label}: ${fmt(d.value)}`).join(", "));
}

/** One line over [0, tMax] with a crosshair tooltip and a marked point. */
function lineChart(container, xs, ys, { mark, xLabel, yLabel }) {
  container.replaceChildren();
  const W = 520, H = 260, left = 44, right = 12, top = 10, bottom = 36;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const xMax = xs.at(-1), yMax = 1;
  const x = (v) => left + (v / xMax) * (W - left - right);
  const y = (v) => top + (1 - v / yMax) * (H - top - bottom);
  for (const t of ticks(yMax)) {
    el("line", { class: "gridline", x1: left, x2: W - right, y1: y(t), y2: y(t) }, svg);
    text(svg, left - 6, y(t) + 4, t, { "text-anchor": "end" });
  }
  for (const t of ticks(xMax, 6)) text(svg, x(t), H - bottom + 16, t, { "text-anchor": "middle" });
  text(svg, W - right, H - 4, xLabel, { "text-anchor": "end" });
  el("line", { class: "axis", x1: left, x2: W - right, y1: y(0), y2: y(0) }, svg);
  el("path", { class: "line", d: xs.map((v, i) => `${i ? "L" : "M"}${x(v)},${y(ys[i])}`).join("") }, svg);
  if (mark) el("circle", { class: "marker", cx: x(mark.x), cy: y(mark.y), r: 5 }, svg);
  const cross = el("line", { class: "crosshair", y1: top, y2: y(0), visibility: "hidden" }, svg);
  const dot = el("circle", { class: "marker", r: 4, visibility: "hidden" }, svg);
  const hit = el("rect", { class: "hit", x: left, y: top, width: W - left - right, height: H - top - bottom }, svg);
  hit.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const vx = ((e.clientX - box.left) / box.width) * W;
    const t = Math.min(xMax, Math.max(0, ((vx - left) / (W - left - right)) * xMax));
    const i = Math.round((t / xMax) * (xs.length - 1));
    cross.setAttribute("x1", x(xs[i]));
    cross.setAttribute("x2", x(xs[i]));
    dot.setAttribute("cx", x(xs[i]));
    dot.setAttribute("cy", y(ys[i]));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("visibility", "visible");
    showTip(e, `t = ${xs[i].toFixed(3)}\n${yLabel} = ${fmt(ys[i])}`);
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
    hideTip();
  });
  svg.setAttribute("aria-label", `${yLabel} against ${xLabel}`);
}

/** Interpolate the sequential ramp (CSS --seq-0 → --seq-1) at u ∈ [0, 1]. */
function ramp(u) {
  const style = getComputedStyle(document.documentElement);
  const parse = (h) => h.trim().match(/[0-9a-f]{2}/gi).map((c) => parseInt(c, 16));
  const a = parse(style.getPropertyValue("--seq-0"));
  const b = parse(style.getPropertyValue("--seq-1"));
  const t = Math.min(1, Math.max(0, u));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

/** Node coordinates in [0, 1]²: the graph's embedding, or a circle. */
function layout(graph) {
  let pts;
  if (graph.positions) {
    // Rotate 3D embeddings slightly so that no nodes coincide.
    const [a, b] = [0.5, 0.35];
    pts = graph.positions.map(([px, py, pz]) => {
      const x1 = px * Math.cos(a) + pz * Math.sin(a);
      const z1 = -px * Math.sin(a) + pz * Math.cos(a);
      return [x1, py * Math.cos(b) - z1 * Math.sin(b)];
    });
  } else {
    const n = graph.nodes;
    pts = Array.from({ length: n }, (_, k) => [
      Math.cos((2 * Math.PI * k) / n - Math.PI / 2),
      Math.sin((2 * Math.PI * k) / n - Math.PI / 2),
    ]);
  }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const span = Math.max(x1 - x0, y1 - y0) || 1;
  return pts.map(([x, y]) => [
    0.5 + (x - (x0 + x1) / 2) / span,
    0.5 - (y - (y0 + y1) / 2) / span,
  ]);
}

/** Draw a graph; returns a function that recolours its nodes. */
function drawGraph(container, graph, { onClick, marked = [] }) {
  container.replaceChildren();
  const S = 400, pad = 24;
  const svg = el("svg", { viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": graph.name }, container);
  const pos = layout(graph).map(([x, y]) => [pad + x * (S - 2 * pad), pad + y * (S - 2 * pad)]);
  for (const [u, v] of graph.edges) {
    el("line", { class: "edge", x1: pos[u][0], y1: pos[u][1], x2: pos[v][0], y2: pos[v][1] }, svg);
  }
  const r = Math.max(4, Math.min(14, 160 / Math.sqrt(graph.nodes)));
  let values = new Array(graph.nodes).fill(0);
  const labels = [];
  const nodes = pos.map(([x, y], k) => {
    const c = el("circle", { class: "node", cx: x, cy: y, r }, svg);
    if (marked.includes(k)) c.classList.add("marked");
    c.addEventListener("mousemove", (e) => showTip(e, `node ${k}\nP = ${fmt(values[k])}`));
    c.addEventListener("mouseleave", hideTip);
    c.addEventListener("click", () => onClick?.(k));
    if (graph.nodes <= 40) labels[k] = text(svg, x, y, k, { class: "label" });
    return c;
  });
  return (probabilities, scaleMax) => {
    values = probabilities;
    nodes.forEach((c, k) => {
      const fill = ramp(probabilities[k] / scaleMax);
      c.setAttribute("fill", fill);
      // Label ink by the fill's luminance, so it stays readable on every step.
      const [r, g, b] = fill.match(/\d+/g).map(Number);
      labels[k]?.setAttribute("style", `fill: ${0.299 * r + 0.587 * g + 0.114 * b < 140 ? "#fff" : "#0b0b0b"}`);
    });
  };
}

function graphInfo(g) {
  const s = g.summary;
  const parts = [
    `${g.name}: ${s.nodes} nodes, ${s.edges} edges`,
    s.regular ? `${s.max_degree}-regular` : `degree ${s.min_degree}–${s.max_degree}`,
    `|Aut| = ${g.automorphism_order ?? `10^${g.log10_automorphism_order.toFixed(1)}`}`,
    `${g.orbits.length} orbit${g.orbits.length === 1 ? "" : "s"}`,
  ];
  if (g.strongly_regular) parts.push(`strongly regular (${g.strongly_regular.join(", ")})`);
  if (!s.connected) parts.push("not connected");
  return parts.join(" · ");
}

// ---------------------------------------------------------------------------
// Tabs.
// ---------------------------------------------------------------------------

const tabs = [...document.querySelectorAll('[role="tab"]')];
function select(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", on);
    $(t.getAttribute("aria-controls")).hidden = !on;
  }
  history.replaceState(null, "", `#${tab.getAttribute("aria-controls")}`);
}
for (const t of tabs) t.addEventListener("click", () => select(t));
const initial = tabs.find((t) => `#${t.getAttribute("aria-controls")}` === location.hash);
if (initial) select(initial);

// ---------------------------------------------------------------------------
// Circuits.
// ---------------------------------------------------------------------------

const EXAMPLES = {
  "Bell state": `OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
`,
  "GHZ state (no measurement: final state)": `OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
`,
  "Teleportation with feed-forward": `OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
// The state to teleport: U(0.3, 0.2, 0.1)|0⟩. Bit 2 reads 1
// with probability sin²(0.15) = 0.0223, whatever bits 0 and 1 say.
U(0.3, 0.2, 0.1) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1] == 1) x q[2];
if (c[0] == 1) z q[2];
c[2] = measure q[2];
`,
  "Grover search for |11⟩": `OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q;
// Oracle: phase flip of |11⟩.
cz q[0], q[1];
// Diffusion: reflection about the uniform state.
h q;
x q;
cz q[0], q[1];
x q;
h q;
c = measure q;
`,
  "Gate definitions and modifiers": `OPENQASM 3.0;
include "stdgates.inc";
// Quantum Fourier transform on three qubits, then its inverse.
gate qft3 a, b, c {
  h c;
  ctrl @ p(pi/2) b, c;
  ctrl @ p(pi/4) a, c;
  h b;
  ctrl @ p(pi/2) a, b;
  h a;
  swap a, c;
}
qubit[3] q;
bit[3] c;
x q[0];
x q[2];
qft3 q[0], q[1], q[2];
inv @ qft3 q[0], q[1], q[2];
c = measure q;
`,
  "OpenQASM 2: ripple-carry adder fragment": `OPENQASM 2.0;
include "qelib1.inc";
// Majority and unmajority-add gates of Cuccaro et al.: 1 + 3 = 4.
gate majority a, b, c { cx c, b; cx c, a; ccx a, b, c; }
gate unmaj a, b, c { ccx a, b, c; cx c, a; cx a, b; }
qreg cin[1];
qreg a[2];
qreg b[2];
qreg cout[1];
creg ans[3];
x a[0];
x b[0];
x b[1];
majority cin[0], b[0], a[0];
majority a[0], b[1], a[1];
cx a[1], cout[0];
unmaj a[0], b[1], a[1];
unmaj cin[0], b[0], a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure cout[0] -> ans[2];
`,
};

// A GHZ state on 1000 qubits: Clifford, so it runs on the stabilizer
// simulator (a state vector would need 2¹⁰⁰⁰ amplitudes).
EXAMPLES["GHZ on 1000 qubits (stabilizer simulator)"] = (() => {
  const lines = [
    "OPENQASM 3.0;",
    'include "stdgates.inc";',
    "// Clifford programs too large for a state vector run on the",
    "// stabilizer simulator; the two ends always agree.",
    "qubit[1000] q;",
    "bit[2] c;",
    "h q[0];",
  ];
  for (let k = 1; k < 1000; k++) lines.push(`cx q[${k - 1}], q[${k}];`);
  lines.push("c[0] = measure q[0];", "c[1] = measure q[999];");
  return lines.join("\n") + "\n";
})();

const exampleSelect = $("example");
for (const name of Object.keys(EXAMPLES)) exampleSelect.add(new Option(name, name));
exampleSelect.addEventListener("change", () => {
  $("source").value = EXAMPLES[exampleSelect.value];
  runQasm();
});
$("source").value = EXAMPLES["Bell state"];

function table(container, header, rows) {
  const t = document.createElement("table");
  t.innerHTML = `<thead><tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
  const body = t.createTBody();
  for (const r of rows) body.insertRow().innerHTML = r.map((c) => `<td>${c}</td>`).join("");
  container.replaceChildren(t);
}

function runQasm() {
  const shots = Number($("shots").value) || 0;
  const seed = Number($("seed").value) || 0;
  return busy($("run-qasm"), $("qasm-status"), async () => {
    const r = await call("qasm", { source: $("source").value, shots, seed });
    $("qasm-result").hidden = false;
    let entries, title, header;
    if (r.state) {
      entries = Object.entries(r.state.probabilities);
      title = `Final state: basis-state probabilities (${r.qubits} qubit${r.qubits === 1 ? "" : "s"})`;
      header = ["state", "probability", "amplitude"];
    } else if (r.probabilities) {
      entries = Object.entries(r.probabilities);
      title = `Outcome probabilities of ${r.bits} classical bit${r.bits === 1 ? "" : "s"}`;
      header = r.counts ? ["outcome", "probability", "counts"] : ["outcome", "probability"];
    } else {
      // Too many measurement branches for the exact distribution: shots only.
      entries = Object.entries(r.counts).map(([k, c]) => [k, c / shots]);
      title = `Sampled frequencies of ${r.bits} classical bit${r.bits === 1 ? "" : "s"} (${shots} shots)`;
      header = ["outcome", "frequency", "counts"];
    }
    entries.sort(([a], [b]) => (a < b ? -1 : 1));
    let shown = entries;
    if (entries.length > 64) {
      shown = [...entries].sort((a, b) => b[1] - a[1]).slice(0, 64).sort(([a], [b]) => (a < b ? -1 : 1));
      title += ` — the 64 most probable of ${entries.length}`;
    }
    $("qasm-chart-title").textContent = title;
    const amp = (k) => {
      const a = r.state?.amplitudes?.[parseInt(k, 2)];
      return a ? `${a[0].toFixed(4)}${a[1] < 0 ? "−" : "+"}${Math.abs(a[1]).toFixed(4)}i` : "";
    };
    const measure = r.probabilities || r.state ? "P" : "frequency";
    barChart($("qasm-chart"), shown.map(([k, p]) => ({
      label: k,
      value: p,
      tip: `${k}\n${measure} = ${fmt(p)}${r.counts ? `\ncounts ${r.counts[k] ?? 0} of ${shots}` : ""}`,
    })));
    table($("qasm-table"), header, entries.map(([k, p]) => [
      k, fmt(p), ...(r.state ? [amp(k)] : r.counts ? [r.counts[k] ?? 0] : []),
    ]));
    $("qasm3").textContent = r.qasm3;
    $("qasm2-box").hidden = r.qasm2 === null;
    $("qasm2").textContent = r.qasm2 ?? "";
    const method = r.method === "stabilizer" ? ", stabilizer simulator" : "";
    return `${r.qubits} qubits, ${r.bits} bits, ${r.operations} operations${method}`;
  });
}
$("run-qasm").addEventListener("click", runQasm);

// ---------------------------------------------------------------------------
// Quantum walks.
// ---------------------------------------------------------------------------

const FRAMES = 200;
let walkState = null;
let walkStart = 0;
let playing = null;

function showFrame(i) {
  const { result, recolour, scaleMax } = walkState;
  const p = result.probabilities[i];
  $("walk-t").textContent = result.times[i].toFixed(2);
  recolour(p, scaleMax);
  barChart($("walk-chart"), p.map((v, k) => ({ label: k, value: v, tip: `node ${k}\nP = ${fmt(v)}` })), { max: 1 });
}

function runWalk() {
  stop();
  return busy($("run-walk"), $("walk-status"), async () => {
    const graph = $("walk-graph").value.trim();
    const tMax = Number($("walk-tmax").value);
    const times = Array.from({ length: FRAMES + 1 }, (_, k) => (tMax * k) / FRAMES);
    const g = await call("graph", { graph });
    if (walkStart >= g.nodes) walkStart = 0;
    const result = await call("walk", {
      graph,
      start: walkStart,
      times,
      gamma: Number($("walk-gamma").value),
      generator: $("walk-generator").value,
    });
    $("walk-result").hidden = false;
    $("walk-info").textContent = `${graphInfo(g)} · start at node ${walkStart}`;
    // One colour scale for all frames after t = 0, so colours compare across time.
    const later = result.probabilities.slice(1).flat();
    const scaleMax = Math.max(...later, 1e-12);
    const recolour = drawGraph($("walk-graph-view"), g, {
      onClick: (k) => {
        walkStart = k;
        runWalk();
      },
      marked: [walkStart],
    });
    walkState = { result, recolour, scaleMax };
    $("walk-time").max = FRAMES;
    showFrame(Number($("walk-time").value));
    return `${times.length} times`;
  });
}

function stop() {
  if (playing) clearInterval(playing);
  playing = null;
  $("walk-play").textContent = "▶";
  $("walk-play").setAttribute("aria-label", "Play");
}
$("walk-play").addEventListener("click", () => {
  if (playing) return stop();
  if (!walkState) return;
  $("walk-play").textContent = "❚❚";
  $("walk-play").setAttribute("aria-label", "Pause");
  playing = setInterval(() => {
    const slider = $("walk-time");
    const i = (Number(slider.value) + 1) % (FRAMES + 1);
    slider.value = i;
    showFrame(i);
  }, 50);
});
$("walk-time").addEventListener("input", (e) => walkState && showFrame(Number(e.target.value)));
$("run-walk").addEventListener("click", runWalk);

// ---------------------------------------------------------------------------
// Spatial search.
// ---------------------------------------------------------------------------

function parseMarked() {
  return $("search-marked").value.split(/[\s,]+/).filter(Boolean).map(Number);
}

function runSearch() {
  return busy($("run-search"), $("search-status"), async () => {
    const graph = $("search-graph").value.trim();
    const marked = parseMarked();
    const g = await call("graph", { graph });
    const r = await call("search", { graph, marked, points: 300 });
    $("search-result").hidden = false;
    $("search-info").textContent =
      `${graphInfo(g)} · best γ = ${r.gamma.toFixed(5)}, t* = ${r.time.toFixed(3)}, ` +
      `success ${fmt(r.success_probability)}, expected time with repetition ${r.expected_time.toFixed(3)}`;
    const recolour = drawGraph($("search-graph-view"), g, {
      marked,
      onClick: (k) => {
        const m = new Set(parseMarked());
        if (m.has(k)) m.delete(k);
        else m.add(k);
        if (m.size === 0) return;
        $("search-marked").value = [...m].sort((a, b) => a - b).join(", ");
        runSearch();
      },
    });
    // Uniform start: colour the marked nodes by the success probability at t*.
    recolour(Array.from({ length: g.nodes }, (_, k) =>
      (marked.includes(k) ? r.success_probability / marked.length : 0)), 1);
    lineChart($("search-chart"), r.curve.times, r.curve.success_probability, {
      mark: { x: r.time, y: r.success_probability },
      xLabel: "time t",
      yLabel: "P(marked)",
    });
  });
}
$("run-search").addEventListener("click", runSearch);

// ---------------------------------------------------------------------------
// Start.
// ---------------------------------------------------------------------------

call("version").then(
  (v) => {
    $("version").textContent = `QSO ${v.version}.`;
    runQasm();
    runWalk();
    runSearch();
  },
  (e) => {
    $("qasm-status").textContent = `Could not load QSO: ${e.message}`;
    $("qasm-status").classList.add("error");
  },
);
