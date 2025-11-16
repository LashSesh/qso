use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionConfig {
    pub mef: MefConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MefConfig {
    pub extension: ExtensionSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionSettings {
    pub knowledge: KnowledgeConfig,
    pub memory: MemoryConfig,
    pub router: RouterConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    pub enabled: bool,
    pub inference: InferenceSettings,
    pub derivation: DerivationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSettings {
    pub threshold: f64,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivationSettings {
    pub root_seed_env: String,
    pub default_path_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub backend: String,
    pub backends: BackendConfigs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfigs {
    pub inmemory: InMemoryConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub faiss: Option<FaissConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw: Option<HnswConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryConfig {
    pub max_items: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissConfig {
    pub index_type: String,
    pub nlist: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub enabled: bool,
    pub mode: String,
    pub service: ServiceConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub url: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub s7_permutations: bool,
}

impl ExtensionConfig {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn load_from_env() -> anyhow::Result<Self> {
        let path = std::env::var("MEF_EXTENSION_CONFIG")
            .unwrap_or_else(|_| "config/extension.yaml".to_string());
        Self::load(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_load() {
        let config_yaml = r#"
mef:
  extension:
    knowledge:
      enabled: false
      inference:
        threshold: 0.5
        max_iterations: 100
      derivation:
        root_seed_env: "MEF_ROOT_SEED"
        default_path_prefix: "MEF"
    memory:
      enabled: false
      backend: inmemory
      backends:
        inmemory:
          max_items: 10000
        faiss:
          index_type: "IVF"
          nlist: 100
        hnsw:
          m: 16
          ef_construction: 200
    router:
      enabled: false
      mode: inproc
      service:
        url: "http://router-service:8080"
        timeout_ms: 5000
      cache:
        enabled: true
        s7_permutations: true
"#;
        let config: ExtensionConfig = serde_yaml::from_str(config_yaml).unwrap();
        assert!(!config.mef.extension.knowledge.enabled);
        assert!(!config.mef.extension.memory.enabled);
        assert!(!config.mef.extension.router.enabled);
        assert_eq!(config.mef.extension.memory.backend, "inmemory");
        assert_eq!(config.mef.extension.router.mode, "inproc");
    }
}
