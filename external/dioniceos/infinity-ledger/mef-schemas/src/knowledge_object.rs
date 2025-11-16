//! KnowledgeObject - TIC binding with route and seed path

use serde::{Deserialize, Serialize};

/// KnowledgeObject represents a TIC binding with route and seed derivation path
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeObject {
    /// Unique knowledge identifier (content-addressed via SHA256)
    pub mef_id: String,

    /// TIC binding identifier
    pub tic_id: String,

    /// Route specification ID
    pub route_id: String,

    /// HD-style seed derivation path (e.g., "MEF/domain/stage/0001")
    pub seed_path: String,

    /// Derived seed (not the root seed)
    pub derived_seed: Vec<u8>,

    /// Optional payload
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl KnowledgeObject {
    /// Create a new KnowledgeObject
    pub fn new(
        mef_id: String,
        tic_id: String,
        route_id: String,
        seed_path: String,
        derived_seed: Vec<u8>,
        payload: Option<serde_json::Value>,
    ) -> Self {
        Self {
            mef_id,
            tic_id,
            route_id,
            seed_path,
            derived_seed,
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_object_creation() {
        let obj = KnowledgeObject::new(
            "mef_001".to_string(),
            "tic_001".to_string(),
            "route_001".to_string(),
            "MEF/domain/stage/0001".to_string(),
            vec![1, 2, 3, 4],
            None,
        );

        assert_eq!(obj.mef_id, "mef_001");
        assert_eq!(obj.tic_id, "tic_001");
        assert_eq!(obj.seed_path, "MEF/domain/stage/0001");
    }

    #[test]
    fn test_serialization() {
        let obj = KnowledgeObject::new(
            "mef_001".to_string(),
            "tic_001".to_string(),
            "route_001".to_string(),
            "MEF/domain/stage/0001".to_string(),
            vec![1, 2, 3, 4],
            None,
        );

        let json = serde_json::to_string(&obj);
        assert!(json.is_ok());

        let deserialized: Result<KnowledgeObject, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }
}
