//! # Knowledge Object Schema
//!
//! Defines the structure for derived knowledge objects that bind TICs with
//! routing information, seed paths, and proofs.
//!
//! ## SPEC-006 Reference
//!
//! From Part 2, Section 4:
//! - Knowledge = { tic, route, seed_path, mef_id, context }
//! - mef_id = HASH(canonical(TIC) || route_id || seed_path)

use serde::{Deserialize, Serialize};

/// Context information for knowledge derivation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeContext {
    /// HDAG node references (parent relationships)
    #[serde(default)]
    pub hdag_refs: Vec<String>,

    /// Parent knowledge IDs
    #[serde(default)]
    pub parents: Vec<String>,

    /// Child knowledge IDs
    #[serde(default)]
    pub children: Vec<String>,
}

/// TIC reference with minimal required fields
///
/// This is a lightweight reference to avoid duplicating full TIC structure.
/// The actual TIC is stored in the ledger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TicReference {
    /// TIC unique identifier
    pub tic_id: String,

    /// Associated snapshot ID
    pub snapshot_id: String,

    /// Timestamp of TIC crystallization
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Route reference with essential information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteReference {
    /// Route unique identifier
    pub route_id: String,

    /// Permutation indices
    pub sigma: Vec<u8>,

    /// Mesh metric score
    pub score: f64,
}

/// Knowledge object representing derived MEF knowledge
///
/// Binds a TIC with routing information and seed path to create a unique
/// knowledge identifier for semantic operations.
///
/// ## JSON Schema
///
/// ```json
/// {
///   "mef_id": "k-a1b2c3d4...",
///   "tic": {
///     "tic_id": "TIC-9f2a",
///     "snapshot_id": "SNAP-xyz",
///     "timestamp": "2025-10-16T22:00:00Z"
///   },
///   "route": {
///     "route_id": "r-abc123",
///     "sigma": [3, 1, 4, 2, 5, 7, 6],
///     "score": 0.842
///   },
///   "seed_path": "MEF/text/spiral/0001",
///   "context": {
///     "hdag_refs": [],
///     "parents": [],
///     "children": []
///   },
///   "ledger_block": 42
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeObject {
    /// Unique MEF knowledge identifier (content-addressed)
    /// mef_id = HASH(canonical(TIC) || route_id || seed_path)
    pub mef_id: String,

    /// TIC reference
    pub tic: TicReference,

    /// Route specification reference
    pub route: RouteReference,

    /// Hierarchical seed derivation path
    /// Format: "MEF/<domain>/<stage>/<index>"
    pub seed_path: String,

    /// Knowledge context and relationships
    pub context: KnowledgeContext,

    /// Ledger block number where this knowledge was committed
    pub ledger_block: u64,
}

impl KnowledgeObject {
    /// Create a new knowledge object
    pub fn new(
        mef_id: String,
        tic: TicReference,
        route: RouteReference,
        seed_path: String,
        ledger_block: u64,
    ) -> Self {
        Self {
            mef_id,
            tic,
            route,
            seed_path,
            context: KnowledgeContext {
                hdag_refs: Vec::new(),
                parents: Vec::new(),
                children: Vec::new(),
            },
            ledger_block,
        }
    }

    /// Add HDAG reference
    pub fn add_hdag_ref(&mut self, node_id: String) {
        self.context.hdag_refs.push(node_id);
    }

    /// Add parent relationship
    pub fn add_parent(&mut self, parent_id: String) {
        self.context.parents.push(parent_id);
    }

    /// Add child relationship
    pub fn add_child(&mut self, child_id: String) {
        self.context.children.push(child_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_object_creation() {
        let tic = TicReference {
            tic_id: "TIC-123".to_string(),
            snapshot_id: "SNAP-456".to_string(),
            timestamp: chrono::Utc::now(),
        };

        let route = RouteReference {
            route_id: "route-789".to_string(),
            sigma: vec![1, 2, 3, 4, 5, 6, 7],
            score: 0.5,
        };

        let knowledge = KnowledgeObject::new(
            "mef-k-abc".to_string(),
            tic,
            route,
            "MEF/test/spiral/0001".to_string(),
            42,
        );

        assert_eq!(knowledge.mef_id, "mef-k-abc");
        assert_eq!(knowledge.ledger_block, 42);
        assert!(knowledge.context.hdag_refs.is_empty());
    }

    #[test]
    fn test_knowledge_relationships() {
        let mut knowledge = KnowledgeObject::new(
            "mef-k-1".to_string(),
            TicReference {
                tic_id: "TIC-1".to_string(),
                snapshot_id: "SNAP-1".to_string(),
                timestamp: chrono::Utc::now(),
            },
            RouteReference {
                route_id: "r-1".to_string(),
                sigma: vec![1, 2, 3, 4, 5, 6, 7],
                score: 0.5,
            },
            "MEF/test/spiral/0001".to_string(),
            1,
        );

        knowledge.add_hdag_ref("HDAG-node-1".to_string());
        knowledge.add_parent("parent-k-1".to_string());
        knowledge.add_child("child-k-1".to_string());

        assert_eq!(knowledge.context.hdag_refs.len(), 1);
        assert_eq!(knowledge.context.parents.len(), 1);
        assert_eq!(knowledge.context.children.len(), 1);
    }
}
