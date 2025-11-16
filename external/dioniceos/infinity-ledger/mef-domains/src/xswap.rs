/*!
 * Xswap - Cross-domain alignment module
 *
 * Implements HDAG-backed manifold alignment for cross-domain similarity
 * verification. The Xswap orchestrator processes source and target payloads
 * through the DomainLayer, aligns the resulting manifolds, obtains a
 * Merkaba-Gate proof, updates the HDAG, and optionally commits an audit block to
 * the MEF ledger.
 */

use crate::domain_layer::{DomainLayer, DomainProcessingResult};
use crate::meshholo::MeshHolo;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use mef_core::gates::merkaba_gate::{MerkabaDeCisionParams, MerkabaGate};
use mef_hdag::HDAG;
use mef_ledger::mef_block::MEFLedger;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Artifacts that describe a completed Xswap alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentArtifacts {
    /// Unique alignment identifier
    pub alignment_id: String,
    /// Alignment score (0.0 to 1.0)
    pub alignment_score: f64,
    /// Manifold gap metric
    pub manifold_gap: f64,
    /// Rotation matrix from alignment
    pub rotation_matrix: Vec<Vec<f64>>,
    /// Translation vector from alignment
    pub translation_vector: Vec<f64>,
    /// Merkaba gate event details
    pub gate_event: Value,
    /// HDAG update data
    pub hdag_data: Value,
    /// Optional ledger block
    pub ledger_block: Option<Value>,
}

/// Parameters for ledger commit
#[derive(Debug, Clone)]
struct LedgerCommitParams<'a> {
    alignment_id: &'a str,
    source_result: &'a DomainProcessingResult,
    target_result: &'a DomainProcessingResult,
    alignment_score: f64,
    manifold_gap: f64,
    gate_event: &'a Value,
    hdag_data: &'a Value,
}

impl AlignmentArtifacts {
    /// Convert artifacts to a serializable dictionary
    pub fn to_dict(&self) -> Value {
        json!({
            "alignment_id": self.alignment_id,
            "alignment_score": self.alignment_score,
            "manifold_gap": self.manifold_gap,
            "rotation_matrix": self.rotation_matrix,
            "translation_vector": self.translation_vector,
            "gate_event": self.gate_event,
            "hdag_data": self.hdag_data,
            "ledger_block": self.ledger_block,
        })
    }
}

/// HDAG-assisted manifold alignment for cross-domain similarity
pub struct Xswap {
    /// Domain layer for processing payloads
    pub domain_layer: Arc<Mutex<DomainLayer>>,
    /// Merkaba gate for validation
    pub merkaba_gate: Arc<Mutex<MerkabaGate>>,
    /// HDAG for tracking alignments
    pub hdag: Arc<Mutex<HDAG>>,
    /// MEF ledger for commits
    pub ledger: Arc<Mutex<MEFLedger>>,
    /// Audit log path
    pub audit_path: PathBuf,
}

impl Xswap {
    /// Create a new Xswap instance
    pub fn new(
        domain_layer: Arc<Mutex<DomainLayer>>,
        merkaba_gate: Arc<Mutex<MerkabaGate>>,
        hdag: Arc<Mutex<HDAG>>,
        ledger: Arc<Mutex<MEFLedger>>,
        audit_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let audit_path = audit_path.as_ref().to_path_buf();

        // Ensure audit path parent exists
        if let Some(parent) = audit_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(Self {
            domain_layer,
            merkaba_gate,
            hdag,
            ledger,
            audit_path,
        })
    }

    /// Perform cross-domain manifold alignment via Xswap
    ///
    /// # Arguments
    ///
    /// * `source_payload` - Raw payload from the source domain
    /// * `target_payload` - Raw payload from the target domain
    /// * `source_domain` - Identifier of the source domain adapter
    /// * `target_domain` - Identifier of the target domain adapter
    /// * `auto_commit` - Commit an audit block when the Merkaba decision is positive
    /// * `merkaba_thresholds` - Optional overrides for the Merkaba decision thresholds
    ///
    /// # Returns
    ///
    /// AlignmentArtifacts capturing metrics and persisted artifacts
    pub fn align(
        &mut self,
        source_payload: &Value,
        target_payload: &Value,
        source_domain: &str,
        target_domain: &str,
        auto_commit: bool,
        merkaba_thresholds: Option<HashMap<String, f64>>,
    ) -> Result<AlignmentArtifacts> {
        // Process source and target through domain layer
        let source_result = {
            let mut domain_layer = self.domain_layer.lock().unwrap();
            domain_layer.process_domain_data(source_payload, source_domain, None)?
        };

        let target_result = {
            let mut domain_layer = self.domain_layer.lock().unwrap();
            domain_layer.process_domain_data(target_payload, target_domain, None)?
        };

        let alignment_id = format!("XSWAP-{}", uuid::Uuid::new_v4());

        // Get meshes from results
        let source_mesh = self.get_mesh(&source_result)?;
        let target_mesh = self.get_mesh(&target_result)?;

        // Extract embeddings
        let embedding_source = self.mesh_embedding(&source_mesh);
        let embedding_target = self.mesh_embedding(&target_mesh);

        // Align embeddings
        let (rotation, translation, manifold_gap) =
            self.align_embeddings(&embedding_source, &embedding_target)?;

        let alignment_score = (1.0 - manifold_gap).clamp(0.0, 1.0);

        // Run Merkaba gate validation
        let gate_event = self.run_merkaba(
            &alignment_id,
            &source_result,
            &target_result,
            alignment_score,
            manifold_gap,
            merkaba_thresholds.unwrap_or_default(),
        )?;

        // Update HDAG
        let hdag_data = self.update_hdag(
            &alignment_id,
            &source_result,
            &target_result,
            alignment_score,
        )?;

        // Commit to ledger if appropriate
        let commit_params = LedgerCommitParams {
            alignment_id: &alignment_id,
            source_result: &source_result,
            target_result: &target_result,
            alignment_score,
            manifold_gap,
            gate_event: &gate_event,
            hdag_data: &hdag_data,
        };
        let ledger_block = self.commit_ledger(&commit_params, auto_commit)?;

        let artifacts = AlignmentArtifacts {
            alignment_id,
            alignment_score,
            manifold_gap,
            rotation_matrix: Self::matrix_to_vec(&rotation),
            translation_vector: translation.as_slice().to_vec(),
            gate_event,
            hdag_data,
            ledger_block,
        };

        self.write_audit_record(&artifacts)?;
        Ok(artifacts)
    }

    // ------------------------------------------------------------------
    // Alignment helpers
    // ------------------------------------------------------------------

    /// Get MeshHolo from processing result
    fn get_mesh(&self, result: &DomainProcessingResult) -> Result<MeshHolo> {
        let domain_layer = self.domain_layer.lock().unwrap();
        let meshes = domain_layer.meshes.lock().unwrap();

        meshes
            .get(&result.mesh_id)
            .cloned()
            .ok_or_else(|| anyhow!("Mesh not found: {}", result.mesh_id))
    }

    /// Extract Metatron embedding from mesh
    fn mesh_embedding(&self, mesh: &MeshHolo) -> DMatrix<f64> {
        let embedding_array = mesh.to_metatron_embedding();
        let (rows, cols) = embedding_array.dim();

        DMatrix::from_fn(rows, cols, |i, j| embedding_array[[i, j]])
    }

    /// Align two embeddings using a Procrustes-like fit
    ///
    /// Returns (rotation_matrix, translation_vector, manifold_gap)
    fn align_embeddings(
        &self,
        source_embedding: &DMatrix<f64>,
        target_embedding: &DMatrix<f64>,
    ) -> Result<(DMatrix<f64>, DVector<f64>, f64)> {
        if source_embedding.nrows() == 0 || target_embedding.nrows() == 0 {
            let identity = DMatrix::identity(13, 13);
            let zero_vec = DVector::zeros(13);
            return Ok((identity, zero_vec, 1.0));
        }

        let n = source_embedding.nrows().min(target_embedding.nrows());
        let m = source_embedding.ncols().min(target_embedding.ncols());
        let src = source_embedding.view((0, 0), (n, m));
        let tgt = target_embedding.view((0, 0), (n, m));

        // Calculate centroids (column means)
        let src_centroid = DVector::from_iterator(m, (0..m).map(|col| src.column(col).mean()));
        let tgt_centroid = DVector::from_iterator(m, (0..m).map(|col| tgt.column(col).mean()));

        // Center the data
        let mut src_centered = DMatrix::zeros(n, m);
        let mut tgt_centered = DMatrix::zeros(n, m);

        for i in 0..n {
            for j in 0..m {
                src_centered[(i, j)] = src[(i, j)] - src_centroid[j];
                tgt_centered[(i, j)] = tgt[(i, j)] - tgt_centroid[j];
            }
        }

        // Check if data is already aligned (centered data has very small norm)
        let src_norm = src_centered.norm();
        let tgt_norm = tgt_centered.norm();

        if src_norm < 1e-10 && tgt_norm < 1e-10 {
            // Data is already centered, perfect alignment
            let identity = DMatrix::identity(m, m);
            let zero_vec = DVector::zeros(m);
            return Ok((identity, zero_vec, 0.0));
        }

        // Compute covariance matrix
        let covariance = src_centered.transpose() * &tgt_centered;

        // Orthonormalize to get rotation matrix
        let rotation = self.orthonormalize_matrix(&covariance)?;

        // Apply rotation and compute residual
        let aligned = &src_centered * &rotation;
        let residual = (&aligned - &tgt_centered).norm();
        let normalizer = tgt_centered.norm() + 1e-10;
        let manifold_gap = residual / normalizer;

        // Compute translation
        let rotated_centroid = &rotation.transpose() * &src_centroid;
        let translation = &tgt_centroid - &rotated_centroid;

        Ok((rotation, translation, manifold_gap))
    }

    /// Produce an orthonormal matrix via Gram-Schmidt
    fn orthonormalize_matrix(&self, matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return Ok(DMatrix::identity(13, 13));
        }

        let mut ortho_columns: Vec<DVector<f64>> = Vec::new();

        // Process each column
        for col_idx in 0..matrix.ncols() {
            let mut vec = matrix.column(col_idx).into_owned();

            // Subtract projections onto previous orthonormal vectors
            for basis in &ortho_columns {
                let projection = vec.dot(basis);
                vec -= projection * basis;
            }

            // Normalize
            let norm = vec.norm();
            if norm > 1e-10 {
                ortho_columns.push(vec / norm);
            }
        }

        // Fill remaining columns with standard basis vectors if needed
        let size = matrix.ncols();
        for idx in ortho_columns.len()..size {
            let mut basis = DVector::zeros(size);
            if idx < size {
                basis[idx] = 1.0;
            }
            ortho_columns.push(basis);
        }

        // Build matrix from columns
        Ok(DMatrix::from_columns(&ortho_columns))
    }

    /// Convert DMatrix to Vec<Vec<f64>>
    fn matrix_to_vec(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
        (0..matrix.nrows())
            .map(|i| matrix.row(i).iter().copied().collect())
            .collect()
    }

    // ------------------------------------------------------------------
    // Merkaba Gate integration
    // ------------------------------------------------------------------

    /// Run Merkaba gate validation
    fn run_merkaba(
        &self,
        alignment_id: &str,
        source_result: &DomainProcessingResult,
        target_result: &DomainProcessingResult,
        alignment_score: f64,
        manifold_gap: f64,
        thresholds: HashMap<String, f64>,
    ) -> Result<Value> {
        // Extract fixpoints from TIC IDs (placeholder - would need actual TIC data)
        // For now, we'll use simplified logic
        let source_fixpoint = [1.0; 13];
        let target_fixpoint = [1.0; 13];

        let n = source_fixpoint.len().min(target_fixpoint.len());
        let src_fp: Vec<f64> = source_fixpoint.iter().take(n).copied().collect();
        let tgt_fp: Vec<f64> = target_fixpoint.iter().take(n).copied().collect();

        // Normalize fixpoints
        let src_norm = Self::normalize_vec(&src_fp);
        let tgt_norm = Self::normalize_vec(&tgt_fp);

        // Calculate Mirror Consistency Index (MCI)
        let mci: f64 = src_norm
            .iter()
            .zip(tgt_norm.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mci = (mci.clamp(-1.0, 1.0) + 1.0) / 2.0;

        // Get thresholds
        let merkaba_gate = self.merkaba_gate.lock().unwrap();
        let eps = *thresholds.get("epsilon").unwrap_or(&merkaba_gate.epsilon);
        let phi_star = *thresholds.get("phi_star").unwrap_or(&merkaba_gate.phi_star);
        let eta = thresholds.get("eta").copied().or(Some(merkaba_gate.eta));

        let delta_pi = manifold_gap;
        let phi = alignment_score.clamp(0.0, 1.0);
        let delta_v = -manifold_gap.abs() - 1e-6;

        // Determine PoR status
        let por_source = source_result.gate_validation.passed;
        let por_target = target_result.gate_validation.passed;
        let mut por = if por_source && por_target {
            "valid"
        } else {
            "invalid"
        };

        // Allow override if alignment is strong
        if por == "invalid" {
            let alignment_sufficient = phi >= phi_star && delta_pi.abs() <= eps;
            let mci_sufficient = eta.is_none_or(|e| mci >= e);
            if alignment_sufficient && mci_sufficient && delta_v < 0.0 {
                por = "valid";
            }
        }

        // Make decision
        let params = MerkabaDeCisionParams { eps, phi_star, eta };
        let (commit, reason) =
            merkaba_gate.merkaba_decide(por, delta_pi, phi, delta_v, Some(mci), &params);

        Ok(json!({
            "alignment_id": alignment_id,
            "checks": {
                "por": por,
                "delta_pi": delta_pi,
                "phi": phi,
                "delta_v": delta_v,
                "mci": mci,
            },
            "decision": {
                "commit": commit,
                "reason": reason,
                "thresholds": {
                    "epsilon": eps,
                    "phi_star": phi_star,
                    "eta": eta,
                },
            },
            "timestamp": Utc::now().to_rfc3339(),
        }))
    }

    /// Normalize a vector
    fn normalize_vec(vec: &[f64]) -> Vec<f64> {
        let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            vec.iter().map(|x| x / norm).collect()
        } else {
            vec.to_vec()
        }
    }

    // ------------------------------------------------------------------
    // HDAG and ledger integration
    // ------------------------------------------------------------------

    /// Update HDAG with alignment information
    fn update_hdag(
        &self,
        alignment_id: &str,
        source_result: &DomainProcessingResult,
        target_result: &DomainProcessingResult,
        weight: f64,
    ) -> Result<Value> {
        let now = Utc::now();
        let source_time = now;
        let target_time = now + Duration::seconds(1);

        let source_snapshot = self.snapshot_from_result(source_result, source_time);
        let target_snapshot = self.snapshot_from_result(target_result, target_time);

        let mut hdag = self.hdag.lock().unwrap();

        let source_node = hdag.create_node(
            source_snapshot["id"].as_str().unwrap(),
            source_snapshot["phase"].as_f64().unwrap(),
            Some(source_snapshot["timestamp"].as_str().unwrap().to_string()),
            Some(format!(
                "XSWAP-SRC-{}",
                source_snapshot["id"].as_str().unwrap()
            )),
        )?;

        let target_node = hdag.create_node(
            target_snapshot["id"].as_str().unwrap(),
            target_snapshot["phase"].as_f64().unwrap(),
            Some(target_snapshot["timestamp"].as_str().unwrap().to_string()),
            Some(format!(
                "XSWAP-TGT-{}",
                target_snapshot["id"].as_str().unwrap()
            )),
        )?;

        let mut edge_id = None;
        let mut path_info = json!({});

        if !source_node.is_empty() && !target_node.is_empty() {
            edge_id = hdag.create_edge(&source_node, &target_node, weight, "xswap")?;
            let path_result = hdag.verify_path_invariance(&source_node, &target_node);
            path_info = json!({
                "paths": path_result.paths,
                "weights": path_result.weights,
                "mean_weight": path_result.mean_weight,
                "std_weight": path_result.std_weight,
                "invariant": path_result.invariant,
            });
        }

        Ok(json!({
            "alignment_id": alignment_id,
            "source_node": source_node,
            "target_node": target_node,
            "edge_id": edge_id,
            "path": path_info,
        }))
    }

    /// Create snapshot from processing result
    fn snapshot_from_result(
        &self,
        result: &DomainProcessingResult,
        timestamp: DateTime<Utc>,
    ) -> Value {
        // Get mesh to extract spectral gap
        let domain_layer = self.domain_layer.lock().unwrap();
        let meshes = domain_layer.meshes.lock().unwrap();

        let phase = meshes
            .get(&result.mesh_id)
            .map(|mesh| mesh.invariants.lambda_gap)
            .unwrap_or(0.0);

        json!({
            "id": result.tic_id.as_ref().unwrap_or(&result.mesh_id),
            "phase": phase,
            "timestamp": timestamp.to_rfc3339(),
        })
    }

    /// Commit alignment to ledger
    fn commit_ledger(
        &self,
        params: &LedgerCommitParams,
        auto_commit: bool,
    ) -> Result<Option<Value>> {
        if !auto_commit
            || !params.gate_event["decision"]["commit"]
                .as_bool()
                .unwrap_or(false)
        {
            return Ok(None);
        }

        // Create combined fixpoint (placeholder)
        let source_fixpoint = vec![1.0; 13];
        let target_fixpoint = vec![1.0; 13];
        let combined_fixpoint = self.combine_fixpoints(&source_fixpoint, &target_fixpoint);

        let ledger_tic = json!({
            "tic_id": params.alignment_id,
            "seed": "xswap",
            "fixpoint": combined_fixpoint,
            "window": [0.0, 1.0],
            "invariants": {
                "alignment_score": params.alignment_score,
                "manifold_gap": params.manifold_gap,
                "hdag_invariant": params.hdag_data.get("path").and_then(|p| p.get("invariant")),
            },
            "sigma_bar": {
                "source": {},
                "target": {},
            },
            "proof": {
                "por": params.gate_event["checks"]["por"],
                "phi": params.gate_event["checks"]["phi"],
                "delta_pi": params.gate_event["checks"]["delta_pi"],
                "delta_v": params.gate_event["checks"]["delta_v"],
                "mci": params.gate_event["checks"]["mci"],
                "decision": params.gate_event["decision"],
            },
        });

        // Get mesh info for phase calculation
        let domain_layer = self.domain_layer.lock().unwrap();
        let meshes = domain_layer.meshes.lock().unwrap();

        let source_gap = meshes
            .get(&params.source_result.mesh_id)
            .map(|m| m.invariants.lambda_gap)
            .unwrap_or(0.0);
        let target_gap = meshes
            .get(&params.target_result.mesh_id)
            .map(|m| m.invariants.lambda_gap)
            .unwrap_or(0.0);

        let ledger_snapshot = json!({
            "id": params.alignment_id,
            "phase": (source_gap + target_gap) / 2.0,
            "timestamp": Utc::now().to_rfc3339(),
            "source_node": params.hdag_data.get("source_node"),
            "target_node": params.hdag_data.get("target_node"),
            "edge_id": params.hdag_data.get("edge_id"),
            "alignment": {
                "score": params.alignment_score,
                "gap": params.manifold_gap,
                "path": params.hdag_data.get("path"),
            },
        });

        let mut ledger = self.ledger.lock().unwrap();
        let block = ledger.append_block(&ledger_tic, &ledger_snapshot)?;

        Ok(Some(serde_json::to_value(block)?))
    }

    /// Combine two fixpoints by averaging
    fn combine_fixpoints(&self, source: &[f64], target: &[f64]) -> Vec<f64> {
        let n = source.len().min(target.len());
        if n == 0 {
            return vec![];
        }

        source
            .iter()
            .zip(target.iter())
            .take(n)
            .map(|(a, b)| (a + b) / 2.0)
            .collect()
    }

    // ------------------------------------------------------------------
    // Audit helpers
    // ------------------------------------------------------------------

    /// Write audit record to file
    fn write_audit_record(&self, artifacts: &AlignmentArtifacts) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_path)?;

        let json_str = serde_json::to_string(&artifacts.to_dict())?;
        writeln!(file, "{}", json_str)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mef_core::MEFCore;
    use mef_topology::MetatronRouter;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn test_alignment_artifacts_creation() {
        let artifacts = AlignmentArtifacts {
            alignment_id: "test-123".to_string(),
            alignment_score: 0.95,
            manifold_gap: 0.05,
            rotation_matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            translation_vector: vec![0.0, 0.0],
            gate_event: json!({"test": true}),
            hdag_data: json!({"test": true}),
            ledger_block: None,
        };

        let dict = artifacts.to_dict();
        assert_eq!(dict["alignment_id"], "test-123");
        assert_eq!(dict["alignment_score"], 0.95);
    }

    #[test]
    fn test_normalize_vec() {
        let vec = vec![3.0, 4.0];
        let normalized = Xswap::normalize_vec(&vec);
        let norm: f64 = normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_vec_zero() {
        let vec = vec![0.0, 0.0];
        let normalized = Xswap::normalize_vec(&vec);
        assert_eq!(normalized, vec![0.0, 0.0]);
    }

    #[test]
    fn test_matrix_to_vec() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let vec = Xswap::matrix_to_vec(&matrix);
        assert_eq!(vec, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn test_combine_fixpoints() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("storage");
        let hdag_path = temp_dir.path().join("hdag");
        let ledger_path = temp_dir.path().join("ledger");
        let audit_path = temp_dir.path().join("audit.jsonl");

        let mef_core = Arc::new(MEFCore::new("test-seed", None).unwrap());
        let router = Arc::new(Mutex::new(MetatronRouter::default()));
        let domain_layer = Arc::new(Mutex::new(
            DomainLayer::new(mef_core.clone(), router.clone(), &storage_path).unwrap(),
        ));
        let merkaba_gate = Arc::new(Mutex::new(MerkabaGate::new(
            temp_dir.path().join("merkaba_audit.jsonl"),
        )));
        let hdag = Arc::new(Mutex::new(HDAG::new(&hdag_path).unwrap()));
        let ledger = Arc::new(Mutex::new(MEFLedger::new(&ledger_path).unwrap()));

        let xswap = Xswap::new(domain_layer, merkaba_gate, hdag, ledger, &audit_path).unwrap();

        let source = vec![1.0, 2.0, 3.0];
        let target = vec![4.0, 5.0, 6.0];
        let combined = xswap.combine_fixpoints(&source, &target);

        assert_eq!(combined.len(), 3);
        assert_eq!(combined[0], 2.5);
        assert_eq!(combined[1], 3.5);
        assert_eq!(combined[2], 4.5);
    }

    #[test]
    fn test_combine_fixpoints_different_lengths() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("storage");
        let hdag_path = temp_dir.path().join("hdag");
        let ledger_path = temp_dir.path().join("ledger");
        let audit_path = temp_dir.path().join("audit.jsonl");

        let mef_core = Arc::new(MEFCore::new("test-seed", None).unwrap());
        let router = Arc::new(Mutex::new(MetatronRouter::default()));
        let domain_layer = Arc::new(Mutex::new(
            DomainLayer::new(mef_core.clone(), router.clone(), &storage_path).unwrap(),
        ));
        let merkaba_gate = Arc::new(Mutex::new(MerkabaGate::new(
            temp_dir.path().join("merkaba_audit.jsonl"),
        )));
        let hdag = Arc::new(Mutex::new(HDAG::new(&hdag_path).unwrap()));
        let ledger = Arc::new(Mutex::new(MEFLedger::new(&ledger_path).unwrap()));

        let xswap = Xswap::new(domain_layer, merkaba_gate, hdag, ledger, &audit_path).unwrap();

        let source = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![5.0, 6.0];
        let combined = xswap.combine_fixpoints(&source, &target);

        assert_eq!(combined.len(), 2);
        assert_eq!(combined[0], 3.0);
        assert_eq!(combined[1], 4.0);
    }

    #[test]
    fn test_orthonormalize_identity() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("storage");
        let hdag_path = temp_dir.path().join("hdag");
        let ledger_path = temp_dir.path().join("ledger");
        let audit_path = temp_dir.path().join("audit.jsonl");

        let mef_core = Arc::new(MEFCore::new("test-seed", None).unwrap());
        let router = Arc::new(Mutex::new(MetatronRouter::default()));
        let domain_layer = Arc::new(Mutex::new(
            DomainLayer::new(mef_core.clone(), router.clone(), &storage_path).unwrap(),
        ));
        let merkaba_gate = Arc::new(Mutex::new(MerkabaGate::new(
            temp_dir.path().join("merkaba_audit.jsonl"),
        )));
        let hdag = Arc::new(Mutex::new(HDAG::new(&hdag_path).unwrap()));
        let ledger = Arc::new(Mutex::new(MEFLedger::new(&ledger_path).unwrap()));

        let xswap = Xswap::new(domain_layer, merkaba_gate, hdag, ledger, &audit_path).unwrap();

        let identity = DMatrix::identity(3, 3);
        let ortho = xswap.orthonormalize_matrix(&identity).unwrap();

        // Check it's still close to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((ortho[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_align_embeddings_basic() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("storage");
        let hdag_path = temp_dir.path().join("hdag");
        let ledger_path = temp_dir.path().join("ledger");
        let audit_path = temp_dir.path().join("audit.jsonl");

        let mef_core = Arc::new(MEFCore::new("test-seed", None).unwrap());
        let router = Arc::new(Mutex::new(MetatronRouter::default()));
        let domain_layer = Arc::new(Mutex::new(
            DomainLayer::new(mef_core.clone(), router.clone(), &storage_path).unwrap(),
        ));
        let merkaba_gate = Arc::new(Mutex::new(MerkabaGate::new(
            temp_dir.path().join("merkaba_audit.jsonl"),
        )));
        let hdag = Arc::new(Mutex::new(HDAG::new(&hdag_path).unwrap()));
        let ledger = Arc::new(Mutex::new(MEFLedger::new(&ledger_path).unwrap()));

        let xswap = Xswap::new(domain_layer, merkaba_gate, hdag, ledger, &audit_path).unwrap();

        // Test alignment produces valid outputs
        let embedding1 =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let embedding2 =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let (rotation, translation, gap) =
            xswap.align_embeddings(&embedding1, &embedding2).unwrap();

        // Check outputs are valid
        assert_eq!(rotation.nrows(), 3);
        assert_eq!(rotation.ncols(), 3);
        assert_eq!(translation.len(), 3);
        assert!((0.0..=1.0).contains(&gap));
    }

    #[test]
    fn test_align_embeddings_empty() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("storage");
        let hdag_path = temp_dir.path().join("hdag");
        let ledger_path = temp_dir.path().join("ledger");
        let audit_path = temp_dir.path().join("audit.jsonl");

        let mef_core = Arc::new(MEFCore::new("test-seed", None).unwrap());
        let router = Arc::new(Mutex::new(MetatronRouter::default()));
        let domain_layer = Arc::new(Mutex::new(
            DomainLayer::new(mef_core.clone(), router.clone(), &storage_path).unwrap(),
        ));
        let merkaba_gate = Arc::new(Mutex::new(MerkabaGate::new(
            temp_dir.path().join("merkaba_audit.jsonl"),
        )));
        let hdag = Arc::new(Mutex::new(HDAG::new(&hdag_path).unwrap()));
        let ledger = Arc::new(Mutex::new(MEFLedger::new(&ledger_path).unwrap()));

        let xswap = Xswap::new(domain_layer, merkaba_gate, hdag, ledger, &audit_path).unwrap();

        let empty = DMatrix::from_row_slice(0, 0, &[]);
        let non_empty = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let (rotation, translation, gap) = xswap.align_embeddings(&empty, &non_empty).unwrap();

        assert_eq!(rotation.nrows(), 13);
        assert_eq!(rotation.ncols(), 13);
        assert_eq!(translation.len(), 13);
        assert_eq!(gap, 1.0);
    }
}
