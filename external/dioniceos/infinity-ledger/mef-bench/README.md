# mef-bench

Benchmark driver infrastructure for MEF-Core vector store performance validation and cross-database comparison.

## Overview

The `mef-bench` crate provides a flexible driver abstraction for benchmarking vector database implementations. It includes drivers for the MEF-Core API and a baseline exact search implementation for recall validation.

## Features

- **VectorStoreDriver Trait**: Common interface for all benchmark targets
- **MEF Driver**: HTTP client for MEF-Core API endpoints
- **FAISS Baseline**: Brute-force exact nearest-neighbor search for ground truth
- **Elasticsearch Driver**: Elasticsearch/OpenSearch dense vector kNN API client
- **Qdrant Driver**: Qdrant HTTP API client for vector search
- **Milvus Driver**: Milvus HTTP API client for vector search
- **Weaviate Driver**: Weaviate HTTP API client for vector search
- **Pinecone Driver**: Pinecone managed vector database HTTP API client
- **Dataset Utilities**: Synthetic dataset generation for benchmarking
- **Benchmark Runner**: Configuration, execution, and reporting infrastructure
- **Driver Registry**: Dynamic driver instantiation by name
- **Comprehensive Error Handling**: Structured error types with actionable messages

## Usage

### Basic Example

```rust
use mef_bench::{VectorStoreDriver, MEFDriver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create and connect driver
    let mut driver = MEFDriver::new(Some("cosine"));
    driver.connect()?;

    // Prepare data
    let items = vec![
        ("doc1".to_string(), vec![1.0, 2.0, 3.0], None),
        ("doc2".to_string(), vec![4.0, 5.0, 6.0], None),
    ];

    // Upsert vectors
    driver.upsert(items, "my_collection", 1000)?;

    // Search
    let query = vec![1.0, 2.0, 3.0];
    let results = driver.search(&query, 10, "my_collection")?;
    
    for (id, score) in results {
        println!("{}: {}", id, score);
    }

    Ok(())
}
```

### Using the Driver Registry

```rust
use mef_bench::get_driver_registry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = get_driver_registry();
    
    // Create drivers dynamically
    let mut mef_driver = registry.get("mef").unwrap()(Some("cosine"));
    let mut baseline = registry.get("faiss").unwrap()(Some("cosine"));
    
    // Use drivers...
    
    Ok(())
}
```

### Recall Validation

```rust
use mef_bench::{FaissBaselineDriver, MEFDriver, VectorStoreDriver};

fn validate_recall() -> Result<(), Box<dyn std::error::Error>> {
    // Build ground truth with exact search
    let mut baseline = FaissBaselineDriver::new(Some("cosine"));
    let items = vec![
        ("doc1".to_string(), vec![1.0, 0.0, 0.0], None),
        ("doc2".to_string(), vec![0.0, 1.0, 0.0], None),
        ("doc3".to_string(), vec![1.0, 1.0, 0.0], None),
    ];
    baseline.upsert(items.clone(), "test", 1000)?;
    
    // Get exact results
    let query = vec![1.0, 0.5, 0.0];
    let exact = baseline.search(&query, 10, "test")?;
    
    // Compare with approximate search
    let mut mef = MEFDriver::new(Some("cosine"));
    mef.connect()?;
    mef.upsert(items, "test", 1000)?;
    let approx = mef.search(&query, 10, "test")?;
    
    // Calculate recall@k
    let k = 10;
    let matches = approx.iter()
        .take(k)
        .filter(|(id, _)| exact.iter().take(k).any(|(eid, _)| eid == id))
        .count();
    let recall = matches as f64 / k.min(exact.len()) as f64;
    
    println!("Recall@{}: {:.2}%", k, recall * 100.0);
    
    Ok(())
}
```

## Drivers

### MEFDriver

HTTP client for the MEF-Core REST API.

**Configuration**:
- `MEF_BASE_URL` or `QUALITY_BASE_URL` environment variable (default: `http://localhost:8080`)
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Health check validation
- Batched vector upsert
- Configurable timeouts
- JSON request/response handling

### FaissBaselineDriver

Brute-force exact nearest-neighbor search using ndarray.

**Configuration**:
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Exact search for ground truth
- Vector normalization for cosine similarity
- In-memory index
- No external dependencies required

### ElasticDriver

Elasticsearch/OpenSearch dense vector kNN API client.

**Configuration**:
- `ELASTIC_URL` environment variable (required)
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Bulk ingestion with NDJSON format
- Dense vector kNN search
- Automatic index creation with similarity configuration
- Configurable num_candidates for search quality

### QdrantDriver

Qdrant HTTP API client for vector search.

**Configuration**:
- `QDRANT_URL` environment variable (required)
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Collection management with distance metric configuration
- Batched point upsert
- HTTP-only implementation (no client library required)
- Wait confirmation for consistency

### MilvusDriver

Milvus HTTP API client for vector search.

**Configuration**:
- `MILVUS_HOST` environment variable (required)
- `MILVUS_PORT` environment variable (default: `19530`)
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Collection management with metric type configuration
- Batched vector insert via HTTP API
- Health check validation
- Support for COSINE, L2, and IP metrics

### WeaviateDriver

Weaviate HTTP API client for vector search.

**Configuration**:
- `WEAVIATE_URL` environment variable (required)
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Class management with distance metric configuration
- Batch object insertion via HTTP API
- GraphQL query support for search
- Automatic class name sanitization
- Support for cosine, dot, and l2-squared distance metrics

### PineconeDriver

Pinecone managed vector database HTTP API client.

**Configuration**:
- `PINECONE_API_KEY` environment variable (required)
- `PINECONE_ENV` environment variable (optional)
- Metric: `cosine` (default), `l2`, or `ip`

**Features**:
- Serverless and pod-based index support
- Batched vector upsert
- Index readiness waiting
- Support for cosine, dotproduct, and euclidean metrics
- Automatic index creation and deletion

## VectorStoreDriver Trait

All drivers implement the `VectorStoreDriver` trait:

```rust
pub trait VectorStoreDriver: Send + Sync {
    fn name(&self) -> &str;
    fn metric(&self) -> &str;
    fn connect(&mut self) -> Result<(), anyhow::Error>;
    fn clear(&mut self, namespace: &str) -> Result<(), anyhow::Error>;
    fn upsert(&mut self, items: Vec<UpsertItem>, namespace: &str, batch_size: usize) 
        -> Result<(), anyhow::Error>;
    fn search(&self, query: &Vector, k: usize, namespace: &str) 
        -> Result<Vec<(String, f64)>, anyhow::Error>;
}
```

## Error Handling

The crate uses structured error types:

```rust
use mef_bench::DriverUnavailable;

match driver.connect() {
    Err(e) => {
        if let Some(unavailable) = e.downcast_ref::<DriverUnavailable>() {
            eprintln!("Driver unavailable: {}", unavailable.reason);
            // Skip this driver in benchmark
        }
    }
    Ok(_) => {
        // Proceed with benchmark
    }
}
```

### Using Benchmark Runner

The `bench_runner` module provides infrastructure for executing benchmarks:

```rust
use mef_bench::{BenchmarkConfig, BenchmarkRunner};
use std::path::Path;

// Load configuration from file (or use defaults)
let config = BenchmarkRunner::load_config(Path::new("bench_config.json"))?;

// Create runner
let assets_dir = Path::new("assets/bench");
let base_url = "http://localhost:8080".to_string();
let mut runner = BenchmarkRunner::new(config, base_url, assets_dir)?;

// Execute benchmark
let report = runner.run()?;

println!("Status: {}", report.status);
println!("Latency p50: {:.2}ms", report.latency_ms.p50);
println!("Latency p95: {:.2}ms", report.latency_ms.p95);
```

#### Configuration Structure

```rust
use mef_bench::{BenchmarkConfig, TimeoutSettings, BatchSettings};

let config = BenchmarkConfig {
    collection: "spiral".to_string(),
    points: 100000,
    queries: 200,
    k: 10,
    warmup: 100,
    timeouts: TimeoutSettings {
        connect: 30.0,
        read: 60.0,
        bulk_operation: 120.0,
    },
    batch: BatchSettings {
        size: 2000,
        adaptive: true,
        min_size: 500,
        max_size: 5000,
    },
    ..Default::default()
};
```

### Using Dataset Utilities

The `datasets` module provides utilities for generating synthetic benchmark datasets:

```rust
use mef_bench::{
    build_spiral_corpus, 
    generate_query_vectors, 
    iter_records,
    chunked,
    brute_force_top_k,
};

// Generate a synthetic spiral corpus
let (ids, vectors) = build_spiral_corpus(1000, 123);

// Create records for bulk ingestion
let records: Vec<_> = iter_records(&ids, &vectors).collect();

// Process in batches
for batch in chunked(records, 100) {
    println!("Processing batch of {} records", batch.len());
    // Insert batch...
}

// Generate query vectors with jitter
let queries = generate_query_vectors(&vectors, 200, 321);

// Compute ground truth with brute force
let query = &queries[0];
let top_k = brute_force_top_k(query, &vectors, 10, "cosine");
println!("Top 10 matches: {:?}", top_k);
```

## Type Definitions

- `Vector`: `Vec<f64>` - A vector of floating-point numbers
- `UpsertItem`: `(String, Vector, Option<HashMap<String, serde_json::Value>>)` - ID, vector, and optional metadata
- `Record`: Bulk ingestion record with ID, vector, and metadata
- `BenchmarkConfig`: Complete benchmark configuration with timeouts, retry, and batch settings
- `BenchmarkReport`: Benchmark results with latency metrics and stage durations

## Dependencies

- `serde` - Serialization framework
- `serde_json` - JSON support
- `anyhow` - Error handling
- `thiserror` - Custom error types
- `reqwest` - HTTP client (with `blocking` feature)
- `ndarray` - NumPy-compatible arrays
- `tokio` - Async runtime
- `rand` - Random number generation
- `rand_distr` - Statistical distributions
- `chrono` - Date and time handling

## Testing

Run the test suite:

```bash
cargo test -p mef-bench
```

All 98 tests should pass, covering:
- Driver creation and configuration
- Connection management
- Upsert and search operations
- Error handling
- Driver registry
- All supported metrics (cosine, l2, ip)
- Environment variable configuration
- All 7 drivers (MEF, FAISS, Elasticsearch, Qdrant, Milvus, Weaviate, Pinecone)
- Dataset generation and utilities (spiral points, queries, ground truth)
- Benchmark runner configuration and execution

## License

MIT

## Contributing

This crate is part of the MEF-Core Python to Rust migration project. See `MIGRATION.md` for details.
