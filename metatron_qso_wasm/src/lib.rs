//! WebAssembly module of QSO with a JSON interface.
//!
//! The module exports one function, [`call`]: it takes a request
//! `{"method": NAME, "params": {...}}` and returns `{"ok": RESULT}` or
//! `{"error": MESSAGE}`. The JavaScript package (`js/`) wraps it; the
//! methods and their parameters are documented in `js/README.md`.
//!
//! The exports `qso_alloc`, `qso_free`, `qso_call` and `qso_response_len`
//! move the request and the response through the module's memory as UTF-8
//! bytes. The module needs no imports, so it runs in any WebAssembly host.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use metatron_qso::graph::{Graph, generators, io};
use metatron_qso::quantum::{Program, StateVector, bitstring};
use metatron_qso::walks::{ContinuousWalk, Generator, SpatialSearch};
use serde_json::{Map, Value, json};

/// Largest graph the graph and walk methods accept: their dense
/// eigendecompositions take O(n³) time.
pub const MAX_NODES: usize = 2048;
/// Most times a walk evaluates in one call.
pub const MAX_TIMES: usize = 10_000;
/// Most shots one call samples.
pub const MAX_SHOTS: usize = 1_000_000;
/// Most basis states a unitary program lists, the most probable first.
pub const MAX_LISTED_STATES: usize = 1024;
/// Largest register whose amplitudes are returned.
pub const MAX_AMPLITUDE_QUBITS: usize = 10;
/// Points on a search's success-probability curve.
pub const MAX_CURVE_POINTS: usize = 10_000;

type Outcome = std::result::Result<Value, String>;

/// Handle a JSON request and return the JSON response.
pub fn call(request: &str) -> String {
    let response = match serde_json::from_str::<Value>(request) {
        Ok(v) => dispatch(v),
        Err(e) => Err(format!("request is not JSON: {e}")),
    };
    match response {
        Ok(v) => json!({ "ok": v }),
        Err(e) => json!({ "error": e }),
    }
    .to_string()
}

fn dispatch(request: Value) -> Outcome {
    let Value::Object(mut request) = request else {
        return Err("request must be an object {\"method\", \"params\"}".into());
    };
    let method = match request.remove("method") {
        Some(Value::String(m)) => m,
        _ => return Err("missing method".into()),
    };
    let params = match request.remove("params") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(p)) => p,
        Some(_) => return Err("params must be an object".into()),
    };
    if let Some(key) = request.keys().next() {
        return Err(format!("unknown request field '{key}'"));
    }
    let mut p = Params(params);
    let result = match method.as_str() {
        "version" => Ok(json!({ "version": env!("CARGO_PKG_VERSION") })),
        "qasm" => qasm(&mut p),
        "graph" => graph_info(&mut p),
        "walk" => walk(&mut p),
        "search" => search(&mut p),
        "isomorphism" => isomorphism(&mut p),
        other => return Err(format!("unknown method '{other}'")),
    }?;
    p.finish()?;
    Ok(result)
}

/// Request parameters; every key must be used.
struct Params(Map<String, Value>);

impl Params {
    fn take(&mut self, key: &str) -> Option<Value> {
        self.0.remove(key).filter(|v| !v.is_null())
    }

    fn string(&mut self, key: &str) -> std::result::Result<Option<String>, String> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(format!("{key} must be a string")),
        }
    }

    fn required_string(&mut self, key: &str) -> std::result::Result<String, String> {
        self.string(key)?.ok_or_else(|| format!("missing {key}"))
    }

    fn uint(&mut self, key: &str) -> std::result::Result<Option<u64>, String> {
        match self.take(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| format!("{key} must be a non-negative integer")),
        }
    }

    fn index(&mut self, key: &str) -> std::result::Result<Option<usize>, String> {
        self.uint(key)?
            .map(|x| usize::try_from(x).map_err(|_| format!("{key} is too large")))
            .transpose()
    }

    fn number(&mut self, key: &str) -> std::result::Result<Option<f64>, String> {
        match self.take(key) {
            None => Ok(None),
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() => Ok(Some(x)),
                _ => Err(format!("{key} must be a finite number")),
            },
        }
    }

    fn numbers(&mut self, key: &str) -> std::result::Result<Option<Vec<f64>>, String> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| v.as_f64().filter(|x| x.is_finite()))
                .collect::<Option<Vec<f64>>>()
                .map(Some)
                .ok_or_else(|| format!("{key} must be an array of finite numbers")),
            Some(_) => Err(format!("{key} must be an array")),
        }
    }

    fn indices(&mut self, key: &str) -> std::result::Result<Option<Vec<usize>>, String> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| v.as_u64().and_then(|x| usize::try_from(x).ok()))
                .collect::<Option<Vec<usize>>>()
                .map(Some)
                .ok_or_else(|| format!("{key} must be an array of node indices")),
            Some(_) => Err(format!("{key} must be an array")),
        }
    }

    fn generator(&mut self) -> std::result::Result<Generator, String> {
        match self.string("generator")?.as_deref() {
            None | Some("laplacian") => Ok(Generator::Laplacian),
            Some("adjacency") => Ok(Generator::Adjacency),
            Some(other) => Err(format!(
                "unknown generator '{other}' (use laplacian or adjacency)"
            )),
        }
    }

    /// The graph given by `graph` (a specification such as `metatron` or
    /// `cycle:8`) or by `edge_list` (one edge `u v [w]` per line).
    fn graph(&mut self, spec_key: &str) -> std::result::Result<Graph, String> {
        let spec = self.string(spec_key)?;
        let list = if spec_key == "graph" {
            self.string("edge_list")?
        } else {
            None
        };
        let g = match (spec, list) {
            (Some(s), None) => generators::from_spec(&s),
            (None, Some(text)) => io::from_edge_list(&text).map(|g| g.with_name("edge list")),
            (Some(_), Some(_)) => return Err("give either graph or edge_list, not both".into()),
            (None, None) => return Err(format!("missing {spec_key}")),
        }
        .map_err(|e| e.to_string())?;
        if g.num_nodes() > MAX_NODES {
            return Err(format!(
                "{} has {} nodes; at most {MAX_NODES} are supported here",
                g.name(),
                g.num_nodes()
            ));
        }
        Ok(g)
    }

    fn finish(self) -> std::result::Result<(), String> {
        match self.0.keys().next() {
            Some(key) => Err(format!("unknown parameter '{key}'")),
            None => Ok(()),
        }
    }
}

/// Run an OpenQASM 2 or 3 program: exact outcome probabilities of its
/// classical bits, optional samples, the final state of a unitary program
/// and the program exported to both versions.
fn qasm(p: &mut Params) -> Outcome {
    let source = p.required_string("source")?;
    let shots = p.index("shots")?.unwrap_or(0);
    let seed = p.uint("seed")?.unwrap_or(0);
    if shots > MAX_SHOTS {
        return Err(format!("shots must be at most {MAX_SHOTS}"));
    }
    let program = Program::from_qasm(&source).map_err(|e| e.to_string())?;
    let (nq, nb) = (program.n_qubits(), program.n_bits());
    let label = |record: u64| bitstring(record as usize, nb);
    let probabilities: BTreeMap<String, f64> = program
        .probabilities()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(r, q)| (label(r), q))
        .collect();
    let counts = if shots > 0 {
        let c: BTreeMap<String, usize> = program
            .sample(shots, seed)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|(r, c)| (label(r), c))
            .collect();
        Some(c)
    } else {
        None
    };
    let state = if program.is_unitary() {
        let circuit = program.to_circuit().map_err(|e| e.to_string())?;
        let psi = StateVector::from_circuit(&circuit, &[]).map_err(|e| e.to_string())?;
        let probs = psi.probabilities();
        let mut order: Vec<usize> = (0..probs.len()).filter(|&b| probs[b] > 1e-12).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
        let truncated = order.len() > MAX_LISTED_STATES;
        order.truncate(MAX_LISTED_STATES);
        order.sort_unstable();
        let listed: Map<String, Value> = order
            .iter()
            .map(|&b| (bitstring(b, nq), json!(probs[b])))
            .collect();
        let amplitudes = (nq <= MAX_AMPLITUDE_QUBITS).then(|| {
            psi.amplitudes()
                .iter()
                .map(|a| [a.re, a.im])
                .collect::<Vec<_>>()
        });
        Some(json!({"probabilities": listed, "truncated": truncated, "amplitudes": amplitudes}))
    } else {
        None
    };
    Ok(json!({
        "qubits": nq,
        "bits": nb,
        "operations": program.len(),
        "unitary": program.is_unitary(),
        "qubit_registers": program.qubit_registers(),
        "bit_registers": program.bit_registers(),
        "probabilities": probabilities,
        "counts": counts,
        "state": state,
        "qasm3": program.to_qasm3().map_err(|e| e.to_string())?,
        "qasm2": program.to_qasm2().ok(),
    }))
}

/// Structure, spectra and symmetry group of a graph, with its edges and
/// (when it has them) node positions for drawing.
fn graph_info(p: &mut Params) -> Outcome {
    let g = p.graph("graph")?;
    let group = g.automorphism_group();
    Ok(json!({
        "name": g.name(),
        "nodes": g.num_nodes(),
        "edges": g.edges(),
        "positions": g.positions(),
        "summary": g.summary(),
        "automorphism_order": group.order().map(|o| o.to_string()),
        "log10_automorphism_order": group.log10_order(),
        "orbits": group.orbits(),
        "adjacency_spectrum": g.adjacency_spectrum(),
        "laplacian_spectrum": g.laplacian_spectrum(),
        "strongly_regular": g.strongly_regular_parameters(),
    }))
}

/// Continuous-time walk `exp(−iHt)` from one node: node probabilities at
/// each requested time and the limiting (time-averaged) distribution.
fn walk(p: &mut Params) -> Outcome {
    let g = p.graph("graph")?;
    let start = p.index("start")?.unwrap_or(0);
    let gamma = p.number("gamma")?.unwrap_or(1.0);
    let generator = p.generator()?;
    let times = p.numbers("times")?.ok_or("missing times")?;
    if times.len() > MAX_TIMES {
        return Err(format!("at most {MAX_TIMES} times per call"));
    }
    let w = ContinuousWalk::new(&g, generator, gamma).map_err(|e| e.to_string())?;
    let probabilities = times
        .iter()
        .map(|&t| w.probabilities(start, t))
        .collect::<metatron_qso::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    let limit = w.limiting_distribution(start).map_err(|e| e.to_string())?;
    Ok(json!({
        "times": times,
        "probabilities": probabilities,
        "limiting_distribution": limit,
    }))
}

/// Spatial search for marked nodes: the best hopping rate (unless given)
/// and time, and the success probability over `[0, t_max]`.
fn search(p: &mut Params) -> Outcome {
    let g = p.graph("graph")?;
    let marked = p.indices("marked")?.ok_or("missing marked")?;
    let gamma = p.number("gamma")?;
    let generator = p.generator()?;
    let t_max = p
        .number("t_max")?
        .unwrap_or(PI * (g.num_nodes() as f64).sqrt());
    let points = p.index("points")?.unwrap_or(200);
    if !(2..=MAX_CURVE_POINTS).contains(&points) {
        return Err(format!("points must be between 2 and {MAX_CURVE_POINTS}"));
    }
    if t_max <= 0.0 {
        return Err("t_max must be positive".into());
    }
    let result = match gamma {
        Some(x) => {
            SpatialSearch::new(&g, &marked, generator, x).map(|s| s.optimize_time(t_max, 2000))
        }
        None => SpatialSearch::optimize(&g, &marked, generator, t_max, 1000),
    }
    .map_err(|e| e.to_string())?;
    let s = SpatialSearch::new(&g, &marked, generator, result.gamma).map_err(|e| e.to_string())?;
    let times: Vec<f64> = (0..points)
        .map(|k| t_max * k as f64 / (points - 1) as f64)
        .collect();
    let success: Vec<f64> = times.iter().map(|&t| s.success_probability(t)).collect();
    let mut out = serde_json::to_value(&result).map_err(|e| e.to_string())?;
    out["curve"] = json!({ "times": times, "success_probability": success });
    Ok(out)
}

/// Whether two graphs are isomorphic, with a mapping when they are.
fn isomorphism(p: &mut Params) -> Outcome {
    let g = p.graph("first")?;
    let h = p.graph("second")?;
    let mapping = g.isomorphism(&h);
    Ok(json!({ "isomorphic": mapping.is_some(), "mapping": mapping }))
}

// ---------------------------------------------------------------------------
// WebAssembly exports.
// ---------------------------------------------------------------------------

static RESPONSE_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Allocate `len` bytes for a request; the caller writes UTF-8 into them.
#[unsafe(no_mangle)]
pub extern "C" fn qso_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Free a buffer returned by [`qso_alloc`] or [`qso_call`].
///
/// # Safety
///
/// `ptr` and `len` must come from one call of [`qso_alloc`] (with that
/// length) or from [`qso_call`] (with [`qso_response_len`]'s value), and
/// the buffer must not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qso_free(ptr: *mut u8, len: usize) {
    // SAFETY: the caller passes a pointer and capacity from a leaked Vec.
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

/// Handle the request in `ptr[..len]` (freeing it) and return the
/// response; its length is [`qso_response_len`].
///
/// # Safety
///
/// `ptr` must come from [`qso_alloc`]`(len)` and hold `len` written bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qso_call(ptr: *mut u8, len: usize) -> *mut u8 {
    // SAFETY: the caller allocated `len` bytes with qso_alloc and wrote them.
    let request = unsafe { Vec::from_raw_parts(ptr, len, len.max(1)) };
    let response = match std::str::from_utf8(&request) {
        Ok(text) => call(text),
        Err(_) => json!({ "error": "request is not UTF-8" }).to_string(),
    };
    // A boxed slice has capacity equal to its length, as qso_free expects.
    let bytes = response.into_bytes().into_boxed_slice();
    RESPONSE_LEN.store(bytes.len(), std::sync::atomic::Ordering::Relaxed);
    Box::into_raw(bytes).cast::<u8>()
}

/// Length in bytes of the last response of [`qso_call`].
#[unsafe(no_mangle)]
pub extern "C" fn qso_response_len() -> usize {
    RESPONSE_LEN.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(method: &str, params: Value) -> Value {
        let r: Value = serde_json::from_str(&call(
            &json!({"method": method, "params": params}).to_string(),
        ))
        .unwrap();
        assert!(r.get("error").is_none(), "{r}");
        r["ok"].clone()
    }

    fn err(method: &str, params: Value) -> String {
        let r: Value = serde_json::from_str(&call(
            &json!({"method": method, "params": params}).to_string(),
        ))
        .unwrap();
        r["error"].as_str().expect("an error").to_string()
    }

    #[test]
    fn bell_state() {
        let src = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;";
        let r = ok("qasm", json!({"source": src, "shots": 1000, "seed": 7}));
        assert_eq!(r["qubits"], 2);
        let p = &r["probabilities"];
        assert!((p["00"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert!((p["11"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        let c = r["counts"].as_object().unwrap();
        assert_eq!(c.values().map(|v| v.as_u64().unwrap()).sum::<u64>(), 1000);
        assert!(r["state"].is_null());
        // The OpenQASM 2 export reads back to the same distribution.
        let back = ok("qasm", json!({"source": r["qasm2"]}));
        assert_eq!(back["probabilities"], r["probabilities"]);
    }

    #[test]
    fn unitary_program_returns_its_state() {
        let src =
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; h q[0]; cx q[0],q[1]; cx q[1],q[2];";
        let r = ok("qasm", json!({"source": src}));
        let s = &r["state"];
        assert_eq!(s["probabilities"].as_object().unwrap().len(), 2);
        assert!((s["probabilities"]["111"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        let a = s["amplitudes"].as_array().unwrap();
        assert_eq!(a.len(), 8);
        assert!((a[7][0].as_f64().unwrap() - 0.5f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn metatron_cube_graph_and_walks() {
        let g = ok("graph", json!({"graph": "metatron"}));
        assert_eq!(g["nodes"], 13);
        assert_eq!(g["edges"].as_array().unwrap().len(), 78);
        assert_eq!(g["positions"].as_array().unwrap().len(), 13);
        let w = ok(
            "walk",
            json!({"graph": "cycle:6", "start": 0, "times": [0.0, 1.0]}),
        );
        let p = w["probabilities"].as_array().unwrap();
        assert!((p[0][0].as_f64().unwrap() - 1.0).abs() < 1e-12);
        let total: f64 = p[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        // Search on K_16 for one node succeeds with probability ≈ 1.
        let s = ok("search", json!({"graph": "complete:16", "marked": [3]}));
        assert!(s["success_probability"].as_f64().unwrap() > 0.99);
        assert_eq!(s["curve"]["times"].as_array().unwrap().len(), 200);
        let iso = ok(
            "isomorphism",
            json!({"first": "petersen", "second": "kneser:5,2"}),
        );
        assert_eq!(iso["isomorphic"], true);
        let e = ok("graph", json!({"edge_list": "0 1\n1 2\n2 0\n"}));
        assert_eq!(e["summary"]["triangles"], 1);
    }

    #[test]
    fn invalid_requests_are_errors() {
        assert!(call("not json").contains("error"));
        assert!(call("[]").contains("error"));
        assert!(call(r#"{"method": "nope"}"#).contains("unknown method"));
        assert!(call(r#"{"method": "version", "extra": 1}"#).contains("unknown request field"));
        assert!(err("version", json!({"x": 1})).contains("unknown parameter"));
        assert!(err("qasm", json!({})).contains("missing source"));
        assert!(
            err(
                "qasm",
                json!({"source": "OPENQASM 2.0; qreg q[1]; foo q[0];"})
            )
            .contains("foo")
        );
        assert!(
            err(
                "qasm",
                json!({"source": "OPENQASM 3.0;", "shots": 2_000_000})
            )
            .contains("shots")
        );
        assert!(
            err(
                "walk",
                json!({"graph": "cycle:5", "times": [1.0], "start": 9})
            )
            .contains('9')
        );
        assert!(err("walk", json!({"graph": "cycle:5", "times": "1"})).contains("array"));
        assert!(
            err(
                "walk",
                json!({"graph": "cycle:5", "times": [1.0], "generator": "x"})
            )
            .contains("generator")
        );
        assert!(err("graph", json!({"graph": "cycle:3000"})).contains("3000 nodes"));
        assert!(err("graph", json!({"graph": "cycle:5", "edge_list": "0 1"})).contains("either"));
        assert!(
            err(
                "search",
                json!({"graph": "cycle:5", "marked": [1], "t_max": -1.0})
            )
            .contains("t_max")
        );
        assert!(
            err(
                "search",
                json!({"graph": "cycle:5", "marked": [1], "points": 1})
            )
            .contains("points")
        );
    }

    #[test]
    fn buffers_round_trip_through_the_exports() {
        let request = br#"{"method": "version"}"#;
        let ptr = qso_alloc(request.len());
        unsafe {
            std::ptr::copy_nonoverlapping(request.as_ptr(), ptr, request.len());
            let out = qso_call(ptr, request.len());
            let len = qso_response_len();
            let text = std::str::from_utf8(std::slice::from_raw_parts(out, len)).unwrap();
            assert!(text.contains(env!("CARGO_PKG_VERSION")));
            qso_free(out, len);
        }
    }
}
