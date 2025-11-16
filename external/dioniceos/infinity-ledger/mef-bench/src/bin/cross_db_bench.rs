/*!
 * Cross-Database Benchmark CLI
 *
 * Runs comprehensive benchmarks across multiple vector database implementations
 * to compare performance, accuracy, and resource usage.
 */

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use mef_bench::{
    brute_force_top_k, build_spiral_corpus, generate_query_vectors, get_driver_registry,
    UpsertItem, VectorStoreDriver,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
struct DriverBenchmarkResult {
    driver_name: String,
    metric: String,
    status: String,
    error_message: Option<String>,

    // Timing metrics (in milliseconds)
    connect_time_ms: Option<f64>,
    clear_time_ms: Option<f64>,
    upsert_time_ms: Option<f64>,
    search_time_ms: Option<f64>,
    total_time_ms: Option<f64>,

    // Search performance
    avg_search_latency_ms: Option<f64>,
    p50_search_latency_ms: Option<f64>,
    p95_search_latency_ms: Option<f64>,
    p99_search_latency_ms: Option<f64>,

    // Accuracy metrics
    recall_at_10: Option<f64>,
    recall_at_100: Option<f64>,

    // Throughput metrics
    vectors_per_second: Option<f64>,
    queries_per_second: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CrossDBBenchmarkReport {
    timestamp: DateTime<Utc>,
    config: BenchmarkConfigSummary,
    results: Vec<DriverBenchmarkResult>,
    summary: BenchmarkSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkConfigSummary {
    num_vectors: usize,
    num_queries: usize,
    dimension: usize,
    k: usize,
    metric: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSummary {
    total_drivers_tested: usize,
    successful_drivers: usize,
    failed_drivers: usize,
    fastest_driver: Option<String>,
    highest_recall_driver: Option<String>,
    highest_throughput_driver: Option<String>,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let drivers_to_test = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        // Default: test all available drivers
        vec![
            "faiss".to_string(),
            "mef".to_string(),
            "elastic".to_string(),
            "qdrant".to_string(),
            "milvus".to_string(),
            "weaviate".to_string(),
            "pinecone".to_string(),
        ]
    };

    println!("🚀 MEF Cross-Database Benchmark Suite");
    println!("======================================\n");

    // Configuration
    let num_vectors = std::env::var("BENCH_NUM_VECTORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10000);

    let num_queries = std::env::var("BENCH_NUM_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let dimension = std::env::var("BENCH_DIMENSION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    let k = std::env::var("BENCH_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let metric = std::env::var("BENCH_METRIC").unwrap_or_else(|_| "cosine".to_string());

    let batch_size = std::env::var("BENCH_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);

    println!("📊 Configuration:");
    println!("  Vectors: {}", num_vectors);
    println!("  Queries: {}", num_queries);
    println!("  Dimension: {}", dimension);
    println!("  k: {}", k);
    println!("  Metric: {}", metric);
    println!("  Batch size: {}", batch_size);
    println!("  Drivers: {}\n", drivers_to_test.join(", "));

    // Generate benchmark dataset
    println!("📦 Generating benchmark dataset...");
    let (ids, vectors) = build_spiral_corpus(num_vectors, 42);
    let queries = generate_query_vectors(&vectors, num_queries, 123);
    println!(
        "  Generated {} vectors and {} queries\n",
        vectors.len(),
        queries.len()
    );

    // Compute ground truth with FAISS baseline
    println!("🎯 Computing ground truth with exact search...");
    let ground_truth: Vec<Vec<String>> = queries
        .iter()
        .map(|query| {
            let results = brute_force_top_k(query, &vectors, k, &metric);
            results
                .into_iter()
                .map(|(idx, _score)| ids[idx].clone())
                .collect()
        })
        .collect();
    println!("  Ground truth computed\n");

    // Prepare upsert items
    let items: Vec<UpsertItem> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| (id.clone(), vec.clone(), None))
        .collect();

    // Get driver registry
    let registry = get_driver_registry();

    // Run benchmarks for each driver
    let mut results = Vec::new();

    for driver_name in &drivers_to_test {
        println!("🔧 Testing driver: {}", driver_name);
        println!("  {}", "─".repeat(50));

        let result = match registry.get(driver_name.as_str()) {
            Some(constructor) => {
                let mut driver = constructor(Some(&metric));
                run_driver_benchmark(
                    driver.as_mut(),
                    &items,
                    &queries,
                    &ground_truth,
                    k,
                    batch_size,
                    &metric,
                )
            }
            None => {
                println!("  ❌ Driver not found in registry\n");
                DriverBenchmarkResult {
                    driver_name: driver_name.clone(),
                    metric: metric.clone(),
                    status: "not_found".to_string(),
                    error_message: Some(format!("Driver '{}' not found in registry", driver_name)),
                    connect_time_ms: None,
                    clear_time_ms: None,
                    upsert_time_ms: None,
                    search_time_ms: None,
                    total_time_ms: None,
                    avg_search_latency_ms: None,
                    p50_search_latency_ms: None,
                    p95_search_latency_ms: None,
                    p99_search_latency_ms: None,
                    recall_at_10: None,
                    recall_at_100: None,
                    vectors_per_second: None,
                    queries_per_second: None,
                }
            }
        };

        results.push(result);
    }

    // Generate summary
    let summary = generate_summary(&results);

    // Create report
    let report = CrossDBBenchmarkReport {
        timestamp: Utc::now(),
        config: BenchmarkConfigSummary {
            num_vectors: vectors.len(),
            num_queries: queries.len(),
            dimension,
            k,
            metric: metric.clone(),
        },
        results,
        summary,
    };

    // Print summary
    print_summary_table(&report);

    // Save report to JSON
    let output_path =
        std::env::var("BENCH_OUTPUT").unwrap_or_else(|_| "benchmark_results.json".to_string());

    // Create parent directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        fs::create_dir_all(parent)
            .context(format!("Failed to create directory for {}", output_path))?;
    }

    let json_output =
        serde_json::to_string_pretty(&report).context("Failed to serialize benchmark report")?;

    fs::write(&output_path, json_output)
        .context(format!("Failed to write report to {}", output_path))?;

    println!("\n💾 Full report saved to: {}", output_path);

    // Exit with error code if any benchmarks failed
    if report.summary.failed_drivers > 0 {
        println!("\n⚠️  Some benchmarks failed!");
        std::process::exit(1);
    }

    println!("\n✅ All benchmarks completed successfully!");
    Ok(())
}

fn run_driver_benchmark(
    driver: &mut dyn VectorStoreDriver,
    items: &[UpsertItem],
    queries: &[Vec<f64>],
    ground_truth: &[Vec<String>],
    k: usize,
    batch_size: usize,
    metric: &str,
) -> DriverBenchmarkResult {
    let driver_name = driver.name().to_string();
    let start_time = Instant::now();

    // Connect
    println!("  📡 Connecting...");
    let connect_start = Instant::now();
    let connect_result = driver.connect();
    let connect_time_ms = connect_start.elapsed().as_secs_f64() * 1000.0;

    if let Err(e) = connect_result {
        println!("  ❌ Connection failed: {}\n", e);
        return DriverBenchmarkResult {
            driver_name,
            metric: metric.to_string(),
            status: "connection_failed".to_string(),
            error_message: Some(e.to_string()),
            connect_time_ms: Some(connect_time_ms),
            clear_time_ms: None,
            upsert_time_ms: None,
            search_time_ms: None,
            total_time_ms: None,
            avg_search_latency_ms: None,
            p50_search_latency_ms: None,
            p95_search_latency_ms: None,
            p99_search_latency_ms: None,
            recall_at_10: None,
            recall_at_100: None,
            vectors_per_second: None,
            queries_per_second: None,
        };
    }
    println!("  ✓ Connected ({:.2}ms)", connect_time_ms);

    // Clear collection
    println!("  🧹 Clearing collection...");
    let clear_start = Instant::now();
    let namespace = "cross_db_bench";
    if let Err(e) = driver.clear(namespace) {
        println!("  ⚠️  Clear failed (continuing): {}", e);
    }
    let clear_time_ms = clear_start.elapsed().as_secs_f64() * 1000.0;
    println!("  ✓ Cleared ({:.2}ms)", clear_time_ms);

    // Upsert vectors
    println!("  📥 Upserting {} vectors...", items.len());
    let upsert_start = Instant::now();
    let upsert_result = driver.upsert(items.to_vec(), namespace, batch_size);
    let upsert_time_ms = upsert_start.elapsed().as_secs_f64() * 1000.0;

    if let Err(e) = upsert_result {
        println!("  ❌ Upsert failed: {}\n", e);
        return DriverBenchmarkResult {
            driver_name,
            metric: metric.to_string(),
            status: "upsert_failed".to_string(),
            error_message: Some(e.to_string()),
            connect_time_ms: Some(connect_time_ms),
            clear_time_ms: Some(clear_time_ms),
            upsert_time_ms: Some(upsert_time_ms),
            search_time_ms: None,
            total_time_ms: None,
            avg_search_latency_ms: None,
            p50_search_latency_ms: None,
            p95_search_latency_ms: None,
            p99_search_latency_ms: None,
            recall_at_10: None,
            recall_at_100: None,
            vectors_per_second: None,
            queries_per_second: None,
        };
    }

    let vectors_per_second = items.len() as f64 / (upsert_time_ms / 1000.0);
    println!(
        "  ✓ Upserted ({:.2}ms, {:.0} vectors/s)",
        upsert_time_ms, vectors_per_second
    );

    // Small delay to allow indexing to complete
    println!("  ⏳ Waiting for indexing...");
    std::thread::sleep(Duration::from_secs(2));

    // Run search queries
    println!("  🔍 Running {} search queries...", queries.len());
    let mut search_latencies = Vec::new();
    let mut recall_scores = Vec::new();

    let search_start = Instant::now();
    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();
        let search_result = driver.search(query, k, namespace);
        let query_latency = query_start.elapsed().as_secs_f64() * 1000.0;

        match search_result {
            Ok(results) => {
                search_latencies.push(query_latency);

                // Calculate recall
                let returned_ids: Vec<String> =
                    results.iter().map(|(id, _score)| id.clone()).collect();

                let matches = returned_ids
                    .iter()
                    .filter(|id| ground_truth[i].contains(id))
                    .count();

                let recall = matches as f64 / k.min(ground_truth[i].len()) as f64;
                recall_scores.push(recall);
            }
            Err(e) => {
                println!("  ❌ Search query {} failed: {}", i, e);
                // Continue with other queries
            }
        }
    }
    let search_time_ms = search_start.elapsed().as_secs_f64() * 1000.0;

    if search_latencies.is_empty() {
        println!("  ❌ All search queries failed\n");
        return DriverBenchmarkResult {
            driver_name,
            metric: metric.to_string(),
            status: "search_failed".to_string(),
            error_message: Some("All search queries failed".to_string()),
            connect_time_ms: Some(connect_time_ms),
            clear_time_ms: Some(clear_time_ms),
            upsert_time_ms: Some(upsert_time_ms),
            search_time_ms: Some(search_time_ms),
            total_time_ms: Some(start_time.elapsed().as_secs_f64() * 1000.0),
            avg_search_latency_ms: None,
            p50_search_latency_ms: None,
            p95_search_latency_ms: None,
            p99_search_latency_ms: None,
            recall_at_10: None,
            recall_at_100: None,
            vectors_per_second: Some(vectors_per_second),
            queries_per_second: None,
        };
    }

    // Calculate metrics
    let avg_latency = search_latencies.iter().sum::<f64>() / search_latencies.len() as f64;
    let queries_per_second = search_latencies.len() as f64 / (search_time_ms / 1000.0);

    let mut sorted_latencies = search_latencies.clone();
    sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = sorted_latencies[sorted_latencies.len() / 2];
    let p95 = sorted_latencies[sorted_latencies.len() * 95 / 100];
    let p99 = sorted_latencies[sorted_latencies.len() * 99 / 100];

    let avg_recall = recall_scores.iter().sum::<f64>() / recall_scores.len() as f64;

    let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    println!("  ✓ Searches completed");
    println!("    Avg latency: {:.2}ms", avg_latency);
    println!(
        "    p50: {:.2}ms, p95: {:.2}ms, p99: {:.2}ms",
        p50, p95, p99
    );
    println!("    Recall@{}: {:.2}%", k, avg_recall * 100.0);
    println!("    QPS: {:.0}", queries_per_second);
    println!(
        "  ✅ Benchmark completed ({:.2}s total)\n",
        total_time_ms / 1000.0
    );

    DriverBenchmarkResult {
        driver_name,
        metric: metric.to_string(),
        status: "success".to_string(),
        error_message: None,
        connect_time_ms: Some(connect_time_ms),
        clear_time_ms: Some(clear_time_ms),
        upsert_time_ms: Some(upsert_time_ms),
        search_time_ms: Some(search_time_ms),
        total_time_ms: Some(total_time_ms),
        avg_search_latency_ms: Some(avg_latency),
        p50_search_latency_ms: Some(p50),
        p95_search_latency_ms: Some(p95),
        p99_search_latency_ms: Some(p99),
        recall_at_10: Some(avg_recall),
        recall_at_100: None, // Could add this if k > 10
        vectors_per_second: Some(vectors_per_second),
        queries_per_second: Some(queries_per_second),
    }
}

fn generate_summary(results: &[DriverBenchmarkResult]) -> BenchmarkSummary {
    let successful_drivers = results.iter().filter(|r| r.status == "success").count();
    let failed_drivers = results.len() - successful_drivers;

    // Find fastest driver (lowest p50 latency)
    let fastest_driver = results
        .iter()
        .filter(|r| r.p50_search_latency_ms.is_some())
        .min_by(|a, b| {
            a.p50_search_latency_ms
                .unwrap()
                .partial_cmp(&b.p50_search_latency_ms.unwrap())
                .unwrap()
        })
        .map(|r| r.driver_name.clone());

    // Find highest recall driver
    let highest_recall_driver = results
        .iter()
        .filter(|r| r.recall_at_10.is_some())
        .max_by(|a, b| {
            a.recall_at_10
                .unwrap()
                .partial_cmp(&b.recall_at_10.unwrap())
                .unwrap()
        })
        .map(|r| r.driver_name.clone());

    // Find highest throughput driver
    let highest_throughput_driver = results
        .iter()
        .filter(|r| r.queries_per_second.is_some())
        .max_by(|a, b| {
            a.queries_per_second
                .unwrap()
                .partial_cmp(&b.queries_per_second.unwrap())
                .unwrap()
        })
        .map(|r| r.driver_name.clone());

    BenchmarkSummary {
        total_drivers_tested: results.len(),
        successful_drivers,
        failed_drivers,
        fastest_driver,
        highest_recall_driver,
        highest_throughput_driver,
    }
}

fn print_summary_table(report: &CrossDBBenchmarkReport) {
    println!("\n📊 Benchmark Results Summary");
    println!("============================\n");

    println!("Configuration:");
    println!("  Vectors: {}", report.config.num_vectors);
    println!("  Queries: {}", report.config.num_queries);
    println!("  Dimension: {}", report.config.dimension);
    println!("  k: {}", report.config.k);
    println!("  Metric: {}\n", report.config.metric);

    println!(
        "{:<15} {:<10} {:<12} {:<12} {:<12} {:<12}",
        "Driver", "Status", "P50 (ms)", "P95 (ms)", "Recall@10", "QPS"
    );
    println!("{}", "─".repeat(85));

    for result in &report.results {
        let status_icon = match result.status.as_str() {
            "success" => "✅",
            _ => "❌",
        };

        let p50 = result
            .p50_search_latency_ms
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "-".to_string());

        let p95 = result
            .p95_search_latency_ms
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "-".to_string());

        let recall = result
            .recall_at_10
            .map(|v| format!("{:.1}%", v * 100.0))
            .unwrap_or_else(|| "-".to_string());

        let qps = result
            .queries_per_second
            .map(|v| format!("{:.0}", v))
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<15} {:<10} {:<12} {:<12} {:<12} {:<12}",
            result.driver_name,
            format!("{} {}", status_icon, result.status),
            p50,
            p95,
            recall,
            qps
        );
    }

    println!("\n🏆 Winners:");
    if let Some(ref fastest) = report.summary.fastest_driver {
        println!("  ⚡ Fastest (P50): {}", fastest);
    }
    if let Some(ref accurate) = report.summary.highest_recall_driver {
        println!("  🎯 Most Accurate: {}", accurate);
    }
    if let Some(ref throughput) = report.summary.highest_throughput_driver {
        println!("  🚀 Highest Throughput: {}", throughput);
    }

    println!("\n📈 Overall:");
    println!("  Total: {}", report.summary.total_drivers_tested);
    println!("  ✅ Successful: {}", report.summary.successful_drivers);
    println!("  ❌ Failed: {}", report.summary.failed_drivers);
}
