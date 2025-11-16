use crate::config::ExtensionSettings;
use mef_memory::MemoryStore;
use mef_router::MetatronAdapter;
use mef_schemas::{KnowledgeObject, MemoryItem, RouteSpec};
use std::collections::HashMap;

pub struct ExtensionPipeline {
    config: ExtensionSettings,
    memory_store: Option<MemoryStore>,
    router: Option<MetatronAdapter>,
}

impl ExtensionPipeline {
    pub fn new(config: ExtensionSettings) -> Self {
        let memory_store = if config.memory.enabled {
            Some(MemoryStore::in_memory())
        } else {
            None
        };

        let router = if config.router.enabled {
            let mode = match config.router.mode.as_str() {
                "service" => mef_router::AdapterMode::Service,
                _ => mef_router::AdapterMode::InProcess,
            };
            Some(MetatronAdapter::new(mode))
        } else {
            None
        };

        Self {
            config,
            memory_store,
            router,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.knowledge.enabled || self.config.memory.enabled || self.config.router.enabled
    }

    pub fn process_knowledge(&mut self, _knowledge: KnowledgeObject) -> anyhow::Result<()> {
        if !self.config.knowledge.enabled {
            return Ok(());
        }

        // Knowledge processing logic (Phase 2 implementation)
        // For now, this is a placeholder that allows the extension to be wired up
        Ok(())
    }

    pub fn store_memory(&mut self, item: MemoryItem) -> anyhow::Result<()> {
        if let Some(store) = &mut self.memory_store {
            store.store(item)?;
        }
        Ok(())
    }

    pub fn select_route(
        &self,
        seed: &str,
        metrics: &HashMap<String, f64>,
    ) -> anyhow::Result<Option<RouteSpec>> {
        if let Some(router) = &self.router {
            Ok(Some(router.select_route(seed, metrics)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        BackendConfigs, CacheConfig, DerivationSettings, ExtensionSettings, InMemoryConfig,
        InferenceSettings, KnowledgeConfig, MemoryConfig, RouterConfig, ServiceConfig,
    };
    use mef_schemas::{MemoryItem, SpectralSignature};

    fn test_config() -> ExtensionSettings {
        ExtensionSettings {
            knowledge: KnowledgeConfig {
                enabled: false,
                inference: InferenceSettings {
                    threshold: 0.5,
                    max_iterations: 100,
                },
                derivation: DerivationSettings {
                    root_seed_env: "MEF_ROOT_SEED".to_string(),
                    default_path_prefix: "MEF".to_string(),
                },
            },
            memory: MemoryConfig {
                enabled: true,
                backend: "inmemory".to_string(),
                backends: BackendConfigs {
                    inmemory: InMemoryConfig { max_items: 10000 },
                    faiss: None,
                    hnsw: None,
                },
            },
            router: RouterConfig {
                enabled: true,
                mode: "inproc".to_string(),
                service: ServiceConfig {
                    url: "http://router-service:8080".to_string(),
                    timeout_ms: 5000,
                },
                cache: CacheConfig {
                    enabled: true,
                    s7_permutations: true,
                },
            },
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let config = test_config();
        let pipeline = ExtensionPipeline::new(config);
        assert!(pipeline.is_enabled());
    }

    #[test]
    fn test_pipeline_disabled() {
        let mut config = test_config();
        config.knowledge.enabled = false;
        config.memory.enabled = false;
        config.router.enabled = false;

        let pipeline = ExtensionPipeline::new(config);
        assert!(!pipeline.is_enabled());
    }

    #[test]
    fn test_memory_store() {
        let config = test_config();
        let mut pipeline = ExtensionPipeline::new(config);

        // Create a valid 8D normalized vector
        let val = 1.0 / (8.0_f64).sqrt();
        let vector = vec![val; 8];
        let spectral = SpectralSignature {
            psi: 0.3,
            rho: 0.3,
            omega: 0.4,
        };

        let item = MemoryItem::new("test_001".to_string(), vector, spectral, None).unwrap();

        // Store in memory
        pipeline.store_memory(item).unwrap();
    }

    #[test]
    fn test_route_selection() {
        let config = test_config();
        let pipeline = ExtensionPipeline::new(config);

        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route = pipeline.select_route("test_seed", &metrics).unwrap();
        assert!(route.is_some());
    }
}
