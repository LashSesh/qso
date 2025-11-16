/*!
 * Deterministic Ledger⇔Spiral coupling utilities.
 *
 * This module implements the 5D Spiral coupling specification:
 * - Inject ledger events as spiral resonance seeds and HDAG nodes
 * - Synchronise HDAG edges by evaluating resonance against threshold
 * - Navigate the spiral by selecting best candidate step
 * - Condense history windows into Temporal Information Crystals (TICs)
 * - Query the condensed TIC catalogue using resonance functional
 *
 * All calculations are deterministic – same input leads to identical artifacts.
 */

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const ISO_EPOCH: &str = "1970-01-01T00:00:00Z";

/// Parameters controlling the spiral projection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiralParameters {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub theta_step: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Default for SpiralParameters {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 0.5,
            c: 0.1,
            theta_step: 0.017,
            alpha: 0.3,
            beta: 0.3,
        }
    }
}

impl SpiralParameters {
    /// Compute the 5D spiral coordinates for a given angle
    pub fn coordinates(&self, theta: f64) -> Vec<f64> {
        vec![
            self.a * theta.cos(),
            self.a * theta.sin(),
            self.b * (2.0 * theta).cos(),
            self.b * (2.0 * theta).sin(),
            self.c * theta,
        ]
    }
}

/// Configuration of the resonance functional
#[derive(Debug, Clone)]
pub struct ResonanceMetric {
    pub metric: String,
}

impl Default for ResonanceMetric {
    fn default() -> Self {
        Self {
            metric: "cosine".to_string(),
        }
    }
}

impl ResonanceMetric {
    /// Score the resonance between two vectors
    pub fn score(&self, x: &[f64], y: &[f64]) -> f64 {
        if self.metric == "l2sq" {
            let sum: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(xi, yi)| (xi - yi).powi(2))
                .sum();
            return -sum;
        }

        // Default to cosine similarity
        let dot: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let norm_x = x.iter().map(|xi| xi.powi(2)).sum::<f64>().sqrt();
        let norm_y = y.iter().map(|yi| yi.powi(2)).sum::<f64>().sqrt();

        if norm_x == 0.0 || norm_y == 0.0 {
            return 0.0;
        }

        dot / (norm_x * norm_y)
    }
}

/// Stable JSON encoding for determinism
fn stable_json(data: &Value) -> Vec<u8> {
    serde_json::to_string(data).unwrap().as_bytes().to_vec()
}

/// SHA256 hash of bytes
fn sha256_bytes(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

/// UUID v5 generation
fn uuid5(namespace: &Uuid, name: &str) -> String {
    Uuid::new_v5(namespace, name.as_bytes()).to_string()
}

/// Coupling state stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CouplingState {
    event_counter: usize,
    seeds: HashMap<String, Value>,
    hdag: HDAGState,
    pending_steps: Vec<String>,
    steps: Vec<StepRecord>,
    tics: HashMap<String, Value>,
    tic_order: Vec<String>,
    #[serde(default)]
    pipeline_base: PipelineBase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HDAGState {
    nodes: HashMap<String, Value>,
    edges: Vec<EdgePayload>,
    head: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgePayload {
    from: String,
    to: String,
    weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepRecord {
    step: String,
    payload: Value,
    hash: String,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PipelineBase {
    #[serde(rename = "H_embed")]
    h_embed: String,
    #[serde(rename = "H_solve")]
    h_solve: String,
    #[serde(rename = "H_gate")]
    h_gate: String,
    #[serde(rename = "H_index")]
    h_index: String,
}

impl Default for CouplingState {
    fn default() -> Self {
        Self {
            event_counter: 0,
            seeds: HashMap::new(),
            hdag: HDAGState {
                nodes: HashMap::new(),
                edges: Vec::new(),
                head: None,
            },
            pending_steps: Vec::new(),
            steps: Vec::new(),
            tics: HashMap::new(),
            tic_order: Vec::new(),
            pipeline_base: PipelineBase {
                h_embed: "0".repeat(64),
                h_solve: "0".repeat(64),
                h_gate: "0".repeat(64),
                h_index: "0".repeat(64),
            },
        }
    }
}

/// Stateful engine that materialises coupling, navigation and TIC logic
pub struct SpiralCouplingEngine {
    #[allow(dead_code)]
    base_path: PathBuf,
    state_path: PathBuf,
    params: SpiralParameters,
    resonance: ResonanceMetric,
    #[allow(dead_code)]
    eps_pi: f64,
    #[allow(dead_code)]
    zk_mu: f64,
    state: CouplingState,
}

impl SpiralCouplingEngine {
    /// Initialize the coupling engine
    pub fn new(
        base_path: Option<PathBuf>,
        params: Option<SpiralParameters>,
        resonance: Option<ResonanceMetric>,
        eps_pi: f64,
        zk_mu: f64,
    ) -> Result<Self> {
        let base_path = base_path.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("mef")
                .join("coupling")
        });

        fs::create_dir_all(&base_path)?;
        let state_path = base_path.join("coupling_state.json");

        let state = Self::load_state(&state_path)?;
        let params = params.unwrap_or_default();
        let resonance = resonance.unwrap_or_default();

        Ok(Self {
            base_path,
            state_path,
            params,
            resonance,
            eps_pi,
            zk_mu,
            state,
        })
    }

    /// Create a resonance seed for a ledger event and persist it
    pub fn inject_seed(&mut self, event: &Value) -> Result<Value> {
        let canonical_event = stable_json(event);
        let event_hash = sha256_bytes(&canonical_event);

        let counter = self.state.event_counter;
        let theta = counter as f64 * self.params.theta_step;
        let coords = self.params.coordinates(theta);
        self.state.event_counter += 1;

        let seed_id = uuid5(
            &Uuid::NAMESPACE_URL,
            &format!("seed:{}:{:.12}", event_hash, theta),
        );
        let hdag_node_id = uuid5(
            &Uuid::NAMESPACE_URL,
            &format!("hdag:{}:{:.12}", event_hash, theta),
        );

        let epoch = ISO_EPOCH.parse::<DateTime<Utc>>().unwrap();
        let timestamp = (epoch + Duration::milliseconds(counter as i64)).to_rfc3339();

        let seed_data = serde_json::json!({
            "event": event,
            "event_hash": event_hash,
            "theta": theta,
            "coords": coords,
            "hdag_node_id": hdag_node_id,
            "timestamp": timestamp
        });

        self.state.seeds.insert(seed_id.clone(), seed_data);

        let node_data = serde_json::json!({
            "id": hdag_node_id,
            "seed_id": seed_id,
            "tensor": coords,
            "theta": theta,
            "event_hash": event_hash,
            "index": counter,
            "timestamp": timestamp
        });

        self.state
            .hdag
            .nodes
            .insert(hdag_node_id.clone(), node_data);

        let step_hash = self.register_step(
            "SPIRAL_WRITE",
            &serde_json::json!({
                "seed_id": seed_id,
                "hdag_node_id": hdag_node_id,
                "theta": theta,
                "x5": coords,
                "event_hash": event_hash
            }),
        )?;

        self.state.pending_steps.push(step_hash);
        self.persist_state()?;

        Ok(serde_json::json!({
            "seed_id": seed_id,
            "hdag_node_id": hdag_node_id,
            "resonance_seed": {
                "theta": theta,
                "x5": coords
            }
        }))
    }

    /// Create HDAG edges when resonance exceeds the provided threshold
    pub fn sync_hdag(&mut self, threshold: f64) -> Result<Value> {
        let nodes = &self.state.hdag.nodes;
        let existing: HashSet<(String, String)> = self
            .state
            .hdag
            .edges
            .iter()
            .map(|e| (e.from.clone(), e.to.clone()))
            .collect();

        let mut ordered_nodes: Vec<(&String, &Value)> = nodes.iter().collect();
        ordered_nodes.sort_by_key(|(_, v)| v.get("index").and_then(|i| i.as_u64()).unwrap_or(0));

        let mut added = 0;

        for i in 0..ordered_nodes.len() {
            let (source_id, source_val) = ordered_nodes[i];
            let source_tensor = source_val
                .get("tensor")
                .and_then(|t| t.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())
                .unwrap_or_default();

            for (target_id, target_val) in ordered_nodes.iter().skip(i + 1) {
                let key = (source_id.clone(), (*target_id).clone());

                if existing.contains(&key) {
                    continue;
                }

                let target_tensor = target_val
                    .get("tensor")
                    .and_then(|t| t.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())
                    .unwrap_or_default();

                let score = self.resonance.score(&source_tensor, &target_tensor);

                if score <= threshold {
                    continue;
                }

                self.state.hdag.edges.push(EdgePayload {
                    from: source_id.clone(),
                    to: (*target_id).clone(),
                    weight: score,
                });

                added += 1;
            }
        }

        // Sort edges
        self.state
            .hdag
            .edges
            .sort_by(|a, b| (&a.from, &a.to).cmp(&(&b.from, &b.to)));

        // Compute HDAG head
        self.state.hdag.head = Some(self.compute_hdag_head());
        self.persist_state()?;

        Ok(serde_json::json!({
            "edges_added": added,
            "hdag_head": self.state.hdag.head
        }))
    }

    /// Select the next theta that maximises resonance
    pub fn navigate_spiral(
        &mut self,
        theta_current: f64,
        candidates: &[f64],
        params: Option<&HashMap<String, f64>>,
    ) -> Result<Value> {
        if candidates.is_empty() {
            return Err(anyhow!("candidates must not be empty"));
        }

        let local_params = self.override_params(params);
        let current_coords = local_params.coordinates(theta_current);

        let mut best_theta: Option<f64> = None;
        let mut best_score = f64::NEG_INFINITY;

        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for candidate in sorted_candidates {
            let coords = local_params.coordinates(candidate);
            let score = self.resonance.score(&current_coords, &coords);

            if score > best_score
                || (score == best_score
                    && (best_theta.is_none() || candidate < best_theta.unwrap()))
            {
                best_score = score;
                best_theta = Some(candidate);
            }
        }

        let best_theta = best_theta.ok_or_else(|| anyhow!("no candidate selected"))?;

        let step_hash = self.register_step(
            "SPIRAL_NAV",
            &serde_json::json!({
                "theta_current": theta_current,
                "theta_next": best_theta,
                "candidates": candidates,
                "score": best_score
            }),
        )?;

        self.state.pending_steps.push(step_hash);
        self.persist_state()?;

        Ok(serde_json::json!({
            "theta_next": best_theta,
            "score": best_score
        }))
    }

    /// Condense a history window into a TIC artefact
    pub fn condense_histories(&mut self, histories: &[Vec<f64>], mode: &str) -> Result<Value> {
        if mode != "argmax_sumF" {
            return Err(anyhow!("unsupported condensation mode"));
        }
        if histories.is_empty() {
            return Err(anyhow!("histories must not be empty"));
        }

        // Find best vector by argmax sumF
        let mut best_vector: Option<Vec<f64>> = None;
        let mut best_sum = f64::NEG_INFINITY;

        for candidate in histories {
            let score: f64 = histories
                .iter()
                .map(|other| self.resonance.score(candidate, other))
                .sum();

            if score > best_sum {
                best_sum = score;
                best_vector = Some(candidate.clone());
            }
        }

        let best_vector = best_vector.ok_or_else(|| anyhow!("no best vector found"))?;

        // Compute invariants
        let scores: Vec<f64> = histories
            .iter()
            .map(|other| self.resonance.score(&best_vector, other))
            .collect();

        let (delta_pi, _variance, stability) = if !scores.is_empty() {
            let baseline = scores[0];
            let delta_pi = scores
                .iter()
                .map(|s| (s - baseline).abs())
                .fold(0.0f64, f64::max);

            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance =
                scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;

            let stability = (1.0 - variance.min(1.0)).max(0.0);
            (delta_pi, variance, stability)
        } else {
            (0.0, 0.0, 1.0)
        };

        let tic_id = uuid5(
            &Uuid::NAMESPACE_URL,
            &sha256_bytes(&stable_json(&serde_json::json!(best_vector))),
        );

        // Flush pending steps and get seed steps
        let pending_steps = self.flush_pending_steps();
        let seed_steps = self.seed_step_hashes(histories);
        let mut combined_steps: Vec<String> = seed_steps
            .into_iter()
            .chain(pending_steps)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        combined_steps.sort();

        let pipeline_proof = self.assemble_pipeline_proof(&combined_steps);

        let invariants = serde_json::json!({
            "delta_pi": delta_pi,
            "stability": stability
        });

        let mut tic_payload = serde_json::json!({
            "tic_id": tic_id,
            "vector": best_vector,
            "argmax_sumF": best_sum,
            "invariants": invariants,
            "proof": {
                "pipeline_proof": pipeline_proof,
                "steps": combined_steps
            },
            "meta": {
                "created": Self::deterministic_timestamp(&tic_id),
                "mode": mode,
                "hdag_head": self.state.hdag.head
            }
        });

        let step_hash = self.register_step(
            "SPIRAL_CONDENSE",
            &serde_json::json!({
                "tic_id": tic_id,
                "argmax_sumF": best_sum,
                "invariants": invariants
            }),
        )?;

        // Include the condensation step in the proof
        combined_steps.push(step_hash);
        let pipeline_proof = self.assemble_pipeline_proof(&combined_steps);

        tic_payload["proof"]["pipeline_proof"] = serde_json::json!(pipeline_proof);
        tic_payload["proof"]["steps"] = serde_json::json!(combined_steps);

        self.state.tics.insert(tic_id.clone(), tic_payload.clone());
        if !self.state.tic_order.contains(&tic_id) {
            self.state.tic_order.push(tic_id);
        }
        self.persist_state()?;

        Ok(tic_payload)
    }

    /// Import an externally produced TIC into the coupling catalogue
    pub fn register_tic(&mut self, tic: &Value, persist: bool) -> Result<()> {
        let vector_source = tic
            .get("vector")
            .or_else(|| tic.get("fixpoint"))
            .ok_or_else(|| anyhow!("tic must include vector or fixpoint"))?;

        let vector: Vec<f64> = vector_source
            .as_array()
            .ok_or_else(|| anyhow!("vector must be an array"))?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        let tic_id = tic
            .get("tic_id")
            .or_else(|| tic.get("id"))
            .and_then(|id| id.as_str())
            .ok_or_else(|| anyhow!("tic must include tic_id"))?
            .to_string();

        let invariants = tic
            .get("invariants")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let meta = tic.get("meta").cloned().unwrap_or_else(|| {
            serde_json::json!({
                "source_snapshot": tic.get("source_snapshot"),
                "seed": tic.get("seed")
            })
        });

        let proof = tic.get("proof").cloned().unwrap_or(serde_json::json!({}));

        let payload = serde_json::json!({
            "tic_id": tic_id,
            "vector": vector,
            "invariants": invariants,
            "meta": meta,
            "proof": {
                "pipeline_proof": proof.get("pipeline_proof")
            }
        });

        self.state.tics.insert(tic_id.clone(), payload);
        if !self.state.tic_order.contains(&tic_id) {
            self.state.tic_order.push(tic_id);
        }

        if persist {
            self.persist_state()?;
        }

        Ok(())
    }

    /// Query TICs by resonance score
    pub fn query_tics(&self, vector: &[f64], k: usize) -> Result<Vec<Value>> {
        if k == 0 {
            return Err(anyhow!("k must be positive"));
        }

        let items: Vec<&Value> = self
            .state
            .tic_order
            .iter()
            .filter_map(|tic_id| self.state.tics.get(tic_id))
            .collect();

        let mut scored: Vec<Value> = items
            .iter()
            .filter_map(|item| {
                let item_vector = item
                    .get("vector")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())?;

                let score = self.resonance.score(vector, &item_vector);

                Some(serde_json::json!({
                    "tic_id": item.get("tic_id"),
                    "score": score,
                    "meta": item.get("meta"),
                    "pipeline_proof": item.get("proof")
                        .and_then(|p| p.get("pipeline_proof"))
                }))
            })
            .collect();

        scored.sort_by(|a, b| {
            let score_a = a.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
            let score_b = b.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
            let tic_id_a = a.get("tic_id").and_then(|id| id.as_str()).unwrap_or("");
            let tic_id_b = b.get("tic_id").and_then(|id| id.as_str()).unwrap_or("");

            score_b
                .partial_cmp(&score_a)
                .unwrap()
                .then_with(|| tic_id_a.cmp(tic_id_b))
        });

        Ok(scored.into_iter().take(k).collect())
    }

    /// Return a deterministic ZK stub response
    pub fn zk_infer(&mut self, x: &Value) -> Result<Value> {
        let input_hash = sha256_bytes(&stable_json(x));
        let offset_ms = u64::from_str_radix(&input_hash[..12], 16)? % 86400000;

        let epoch = ISO_EPOCH.parse::<DateTime<Utc>>().unwrap();
        let timestamp = (epoch + Duration::milliseconds(offset_ms as i64)).to_rfc3339();

        let payload = serde_json::json!({
            "input_hash": input_hash,
            "timestamp": timestamp
        });

        let proof_stub = serde_json::json!({
            "hash": input_hash,
            "valid": true
        });

        let reference_vector = self.params.coordinates(0.0);
        let output_vector = self.params.coordinates(self.params.theta_step);
        let resonance = self.resonance.score(&output_vector, &reference_vector);
        let lzk = 1.0 - resonance; // Simplified from (1.0 - resonance) + self.zk_mu * (1.0 - 1.0) since (1.0 - 1.0) = 0

        let step_hash = self.register_step(
            "ZK_VERIFY",
            &serde_json::json!({
                "input_hash": input_hash,
                "lzk": lzk
            }),
        )?;

        self.state.pending_steps.push(step_hash);
        self.persist_state()?;

        Ok(serde_json::json!({
            "y": payload,
            "proof_stub": proof_stub,
            "lzk": lzk
        }))
    }

    /// Get a snapshot of the current state
    pub fn get_state_snapshot(&self) -> Value {
        serde_json::to_value(&self.state).unwrap_or(serde_json::json!({}))
    }

    // Helper methods

    fn override_params(&self, override_params: Option<&HashMap<String, f64>>) -> SpiralParameters {
        match override_params {
            None => self.params.clone(),
            Some(overrides) => SpiralParameters {
                a: overrides.get("a").copied().unwrap_or(self.params.a),
                b: overrides.get("b").copied().unwrap_or(self.params.b),
                c: overrides.get("c").copied().unwrap_or(self.params.c),
                theta_step: self.params.theta_step,
                alpha: self.params.alpha,
                beta: self.params.beta,
            },
        }
    }

    fn load_state(state_path: &Path) -> Result<CouplingState> {
        if state_path.exists() {
            let json = fs::read_to_string(state_path)?;
            Ok(serde_json::from_str(&json)?)
        } else {
            Ok(CouplingState::default())
        }
    }

    fn persist_state(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.state)?;
        fs::write(&self.state_path, json)?;
        Ok(())
    }

    fn register_step(&mut self, step: &str, payload: &Value) -> Result<String> {
        let material = serde_json::json!({
            "step": step,
            "payload": payload
        });

        let step_hash = sha256_bytes(&stable_json(&material));
        let timestamp = Self::deterministic_timestamp(&step_hash);

        let record = StepRecord {
            step: step.to_string(),
            payload: payload.clone(),
            hash: step_hash.clone(),
            timestamp,
        };

        self.state.steps.push(record);
        self.persist_state()?;

        Ok(step_hash)
    }

    fn flush_pending_steps(&mut self) -> Vec<String> {
        let pending = self.state.pending_steps.clone();
        self.state.pending_steps.clear();
        let _ = self.persist_state();
        pending
    }

    fn assemble_pipeline_proof(&self, steps: &[String]) -> String {
        let base = &self.state.pipeline_base;
        let material = format!(
            "{}{}{}{}{}",
            base.h_embed,
            base.h_solve,
            base.h_gate,
            base.h_index,
            steps.join("")
        );
        sha256_bytes(material.as_bytes())
    }

    fn compute_hdag_head(&self) -> String {
        let mut nodes: Vec<Value> = self
            .state
            .hdag
            .nodes
            .iter()
            .map(|(node_id, data)| {
                serde_json::json!({
                    "id": node_id,
                    "theta": data.get("theta"),
                    "tensor": data.get("tensor"),
                    "index": data.get("index")
                })
            })
            .collect();

        nodes.sort_by(|a, b| {
            let id_a = a.get("id").and_then(|i| i.as_str()).unwrap_or("");
            let id_b = b.get("id").and_then(|i| i.as_str()).unwrap_or("");
            id_a.cmp(id_b)
        });

        let edges: Vec<Value> = self
            .state
            .hdag
            .edges
            .iter()
            .map(|edge| {
                serde_json::json!({
                    "from": edge.from,
                    "to": edge.to,
                    "weight": edge.weight
                })
            })
            .collect();

        let payload = serde_json::json!({
            "nodes": nodes,
            "edges": edges
        });

        sha256_bytes(&stable_json(&payload))
    }

    fn seed_step_hashes(&self, histories: &[Vec<f64>]) -> Vec<String> {
        let mut lookup: HashMap<String, String> = HashMap::new();

        for entry in &self.state.steps {
            if entry.step != "SPIRAL_WRITE" {
                continue;
            }

            if let Some(coords) = entry.payload.get("x5").and_then(|x| x.as_array()) {
                let coords_vec: Vec<f64> = coords.iter().filter_map(|v| v.as_f64()).collect();
                let key = Self::coords_key(&coords_vec);
                lookup.entry(key).or_insert_with(|| entry.hash.clone());
            }
        }

        let mut seeds = Vec::new();
        for vector in histories {
            let key = Self::coords_key(vector);
            if let Some(step_hash) = lookup.get(&key) {
                seeds.push(step_hash.clone());
            }
        }

        seeds
    }

    fn coords_key(vector: &[f64]) -> String {
        vector
            .iter()
            .map(|v| format!("{:.12}", v))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn deterministic_timestamp(seed: &str) -> String {
        let digest = sha256_bytes(seed.as_bytes());
        let offset_ms = u64::from_str_radix(&digest[..12], 16).unwrap_or(0) % 86400000;

        let epoch = ISO_EPOCH.parse::<DateTime<Utc>>().unwrap();
        (epoch + Duration::milliseconds(offset_ms as i64)).to_rfc3339()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_spiral_parameters_coordinates() {
        let params = SpiralParameters::default();
        let coords = params.coordinates(0.0);

        assert_eq!(coords.len(), 5);
        assert_eq!(coords[0], 1.0); // a * cos(0) = 1.0
        assert_eq!(coords[1], 0.0); // a * sin(0) = 0.0
    }

    #[test]
    fn test_resonance_metric_cosine() {
        let metric = ResonanceMetric::default();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![1.0, 0.0, 0.0];

        let score = metric.score(&x, &y);
        assert_eq!(score, 1.0); // Perfect match
    }

    #[test]
    fn test_resonance_metric_l2sq() {
        let metric = ResonanceMetric {
            metric: "l2sq".to_string(),
        };
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let score = metric.score(&x, &y);
        assert_eq!(score, 0.0); // Perfect match (negative squared distance)
    }

    #[test]
    fn test_create_engine() {
        let result = SpiralCouplingEngine::new(
            Some(PathBuf::from("/tmp/coupling_test")),
            None,
            None,
            0.001,
            0.5,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_inject_seed() {
        let mut engine = SpiralCouplingEngine::new(
            Some(PathBuf::from("/tmp/coupling_test_inject")),
            None,
            None,
            0.001,
            0.5,
        )
        .unwrap();

        let event = json!({"test": "data", "value": 123});
        let result = engine.inject_seed(&event);

        assert!(result.is_ok());
        let seed_result = result.unwrap();
        assert!(seed_result.get("seed_id").is_some());
        assert!(seed_result.get("hdag_node_id").is_some());
        assert!(seed_result.get("resonance_seed").is_some());
    }

    #[test]
    fn test_sync_hdag() {
        let mut engine = SpiralCouplingEngine::new(
            Some(PathBuf::from("/tmp/coupling_test_sync")),
            None,
            None,
            0.001,
            0.5,
        )
        .unwrap();

        // Inject two seeds
        let event1 = json!({"test": "data1"});
        let event2 = json!({"test": "data2"});
        engine.inject_seed(&event1).unwrap();
        engine.inject_seed(&event2).unwrap();

        // Sync with threshold
        let result = engine.sync_hdag(0.5);

        assert!(result.is_ok());
        let sync_result = result.unwrap();
        assert!(sync_result.get("edges_added").is_some());
        assert!(sync_result.get("hdag_head").is_some());
    }

    #[test]
    fn test_navigate_spiral() {
        let mut engine = SpiralCouplingEngine::new(
            Some(PathBuf::from("/tmp/coupling_test_nav")),
            None,
            None,
            0.001,
            0.5,
        )
        .unwrap();

        let candidates = vec![0.1, 0.2, 0.3, 0.4];
        let result = engine.navigate_spiral(0.0, &candidates, None);

        assert!(result.is_ok());
        let nav_result = result.unwrap();
        assert!(nav_result.get("theta_next").is_some());
        assert!(nav_result.get("score").is_some());
    }

    #[test]
    fn test_condense_histories() {
        let mut engine = SpiralCouplingEngine::new(
            Some(PathBuf::from("/tmp/coupling_test_condense")),
            None,
            None,
            0.001,
            0.5,
        )
        .unwrap();

        let histories = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1],
            vec![0.9, 1.9, 2.9, 3.9, 4.9],
        ];

        let result = engine.condense_histories(&histories, "argmax_sumF");

        assert!(result.is_ok());
        let tic = result.unwrap();
        assert!(tic.get("tic_id").is_some());
        assert!(tic.get("vector").is_some());
        assert!(tic.get("invariants").is_some());
        assert!(tic.get("proof").is_some());
    }

    #[test]
    fn test_query_tics() {
        let mut engine = SpiralCouplingEngine::new(
            Some(PathBuf::from("/tmp/coupling_test_query")),
            None,
            None,
            0.001,
            0.5,
        )
        .unwrap();

        // Condense some histories to create TICs
        let histories = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        engine
            .condense_histories(&histories, "argmax_sumF")
            .unwrap();

        // Query
        let query_vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = engine.query_tics(&query_vector, 1);

        assert!(result.is_ok());
        let results = result.unwrap();
        assert!(!results.is_empty());
    }
}
