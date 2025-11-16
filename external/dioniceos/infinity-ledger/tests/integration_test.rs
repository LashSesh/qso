use mef_knowledge::{ExtensionConfig, ExtensionPipeline};
use mef_schemas::{MemoryItem, SpectralSignature};
use std::collections::HashMap;

#[test]
fn test_config_loading() {
    // Load config from fixtures
    let config = ExtensionConfig::load("tests/fixtures/test_config.yaml")
        .expect("Failed to load test config");

    // Verify configuration loaded correctly
    assert!(config.mef.extension.knowledge.enabled);
    assert!(config.mef.extension.memory.enabled);
    assert!(config.mef.extension.router.enabled);
    assert_eq!(config.mef.extension.memory.backend, "inmemory");
    assert_eq!(config.mef.extension.router.mode, "inproc");
}

#[test]
fn test_full_pipeline() {
    // Load config
    let config = ExtensionConfig::load("tests/fixtures/test_config.yaml")
        .expect("Failed to load test config");

    // Create pipeline
    let mut pipeline = ExtensionPipeline::new(config.mef.extension.clone());

    // Verify pipeline is enabled
    assert!(pipeline.is_enabled());

    // Test vector construction - create a valid 8D normalized vector
    let val = 1.0 / (8.0_f64).sqrt();
    let vector = vec![val; 8];
    let spectral = SpectralSignature {
        psi: 0.3,
        rho: 0.3,
        omega: 0.4,
    };

    let item = MemoryItem::new("test_001".to_string(), vector, spectral, None)
        .expect("Failed to create memory item");

    // Store in memory
    pipeline.store_memory(item).expect("Failed to store memory");

    // Test route selection
    let mut metrics = HashMap::new();
    metrics.insert("betti".to_string(), 2.0);
    metrics.insert("lambda_gap".to_string(), 0.5);
    metrics.insert("persistence".to_string(), 0.3);

    let route = pipeline
        .select_route("test_seed", &metrics)
        .expect("Failed to select route");
    assert!(route.is_some());

    // Verify route has valid permutation
    let route_spec = route.unwrap();
    assert_eq!(route_spec.permutation.len(), 7);
}

#[test]
fn test_disabled_pipeline() {
    use mef_knowledge::config::{
        BackendConfigs, CacheConfig, DerivationSettings, ExtensionSettings, InMemoryConfig,
        InferenceSettings, KnowledgeConfig, MemoryConfig, RouterConfig, ServiceConfig,
    };

    // Create config with all disabled
    let config = ExtensionSettings {
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
            enabled: false,
            backend: "inmemory".to_string(),
            backends: BackendConfigs {
                inmemory: InMemoryConfig { max_items: 10000 },
                faiss: None,
                hnsw: None,
            },
        },
        router: RouterConfig {
            enabled: false,
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
    };

    let pipeline = ExtensionPipeline::new(config);

    // Should not be enabled
    assert!(!pipeline.is_enabled());
}

#[test]
fn test_memory_only_pipeline() {
    use mef_knowledge::config::{
        BackendConfigs, CacheConfig, DerivationSettings, ExtensionSettings, InMemoryConfig,
        InferenceSettings, KnowledgeConfig, MemoryConfig, RouterConfig, ServiceConfig,
    };

    // Create config with only memory enabled
    let config = ExtensionSettings {
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
            enabled: false,
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
    };

    let mut pipeline = ExtensionPipeline::new(config);

    // Should be enabled (memory is on)
    assert!(pipeline.is_enabled());

    // Test memory storage
    let val = 1.0 / (8.0_f64).sqrt();
    let vector = vec![val; 8];
    let spectral = SpectralSignature {
        psi: 0.3,
        rho: 0.3,
        omega: 0.4,
    };

    let item = MemoryItem::new("test_memory_only".to_string(), vector, spectral, None)
        .expect("Failed to create memory item");

    // Should succeed
    pipeline.store_memory(item).expect("Failed to store memory");

    // Router should return None
    let mut metrics = HashMap::new();
    metrics.insert("betti".to_string(), 2.0);

    let route = pipeline
        .select_route("test_seed", &metrics)
        .expect("Route selection should not error");
    assert!(
        route.is_none(),
        "Route should be None when router is disabled"
    );
}
