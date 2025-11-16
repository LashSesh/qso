/*!
 * MEF-Core Benchmark Drivers
 *
 * This crate provides driver implementations for benchmarking vector stores.
 * Migrated from MEF-Core_v1.0/src/bench/drivers/
 */

pub mod base;
pub mod bench_runner;
pub mod datasets;
pub mod elastic_driver;
pub mod faiss_baseline;
pub mod mef_driver;
pub mod milvus_driver;
pub mod pinecone_driver;
pub mod qdrant_driver;
pub mod weaviate_driver;

// Re-export commonly used types
pub use base::{DriverUnavailable, UpsertItem, Vector, VectorStoreDriver};
pub use elastic_driver::ElasticDriver;
pub use faiss_baseline::FaissBaselineDriver;
pub use mef_driver::MEFDriver;
pub use milvus_driver::MilvusDriver;
pub use pinecone_driver::PineconeDriver;
pub use qdrant_driver::QdrantDriver;
pub use weaviate_driver::WeaviateDriver;

// Re-export dataset utilities
pub use datasets::{
    brute_force_top_k, build_spiral_corpus, chunked, cosine_similarity, generate_query_vectors,
    generate_spiral_points, iter_records, negative_l2_squared, Record,
};

// Re-export benchmark runner
pub use bench_runner::{
    BatchSettings, BenchmarkConfig, BenchmarkReport, BenchmarkRunner, LatencyMetrics,
    RetrySettings, TimeoutSettings,
};

use std::collections::HashMap;

/// Type alias for driver constructor function
type DriverConstructor = fn(Option<&str>) -> Box<dyn VectorStoreDriver>;

/// Driver registry mapping names to driver constructors
pub fn get_driver_registry() -> HashMap<String, DriverConstructor> {
    let mut registry: HashMap<String, DriverConstructor> = HashMap::new();

    registry.insert("mef".to_string(), |metric| {
        Box::new(MEFDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry.insert("faiss".to_string(), |metric| {
        Box::new(FaissBaselineDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry.insert("elastic".to_string(), |metric| {
        Box::new(ElasticDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry.insert("qdrant".to_string(), |metric| {
        Box::new(QdrantDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry.insert("milvus".to_string(), |metric| {
        Box::new(MilvusDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry.insert("weaviate".to_string(), |metric| {
        Box::new(WeaviateDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry.insert("pinecone".to_string(), |metric| {
        Box::new(PineconeDriver::new(metric)) as Box<dyn VectorStoreDriver>
    });

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_registry() {
        let registry = get_driver_registry();
        assert!(registry.contains_key("mef"));
        assert!(registry.contains_key("faiss"));
        assert!(registry.contains_key("elastic"));
        assert!(registry.contains_key("qdrant"));
        assert!(registry.contains_key("milvus"));
        assert!(registry.contains_key("weaviate"));
        assert!(registry.contains_key("pinecone"));
    }

    #[test]
    fn test_create_mef_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("mef").unwrap();
        let driver = constructor(Some("cosine"));
        assert_eq!(driver.name(), "MEF");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_create_faiss_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("faiss").unwrap();
        let driver = constructor(Some("l2"));
        assert_eq!(driver.name(), "faiss-baseline");
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_create_elastic_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("elastic").unwrap();
        let driver = constructor(Some("cosine"));
        assert_eq!(driver.name(), "Elastic");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_create_qdrant_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("qdrant").unwrap();
        let driver = constructor(Some("ip"));
        assert_eq!(driver.name(), "Qdrant");
        assert_eq!(driver.metric(), "ip");
    }

    #[test]
    fn test_create_milvus_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("milvus").unwrap();
        let driver = constructor(Some("cosine"));
        assert_eq!(driver.name(), "Milvus");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_create_weaviate_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("weaviate").unwrap();
        let driver = constructor(Some("l2"));
        assert_eq!(driver.name(), "Weaviate");
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_create_pinecone_driver_from_registry() {
        let registry = get_driver_registry();
        let constructor = registry.get("pinecone").unwrap();
        let driver = constructor(Some("ip"));
        assert_eq!(driver.name(), "Pinecone");
        assert_eq!(driver.metric(), "ip");
    }
}
