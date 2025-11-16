//! Integration tests for MEF API endpoints
//!
//! This test suite validates the HTTP API endpoints and their interactions
//! with the underlying MEF services.

use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::time::Duration;

const API_BASE_URL: &str = "http://localhost:8080";

fn get_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

#[test]
#[ignore] // Requires API server to be running
fn test_health_endpoint() {
    let client = get_client();
    let response = client
        .get(format!("{}/healthz", API_BASE_URL))
        .send()
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().expect("Failed to parse response");
    assert_eq!(body["status"], "ok");
}

#[test]
#[ignore] // Requires API server to be running
fn test_metrics_endpoint() {
    let client = get_client();
    let response = client
        .get(format!("{}/metrics", API_BASE_URL))
        .send()
        .expect("Failed to send request");

    assert!(response.status().is_success());

    // Metrics should be in Prometheus format
    let body = response.text().expect("Failed to get response text");
    assert!(body.contains("# HELP"));
    assert!(body.contains("# TYPE"));
}

#[test]
#[ignore] // Requires API server to be running
fn test_search_endpoint() {
    let client = get_client();

    // Create a search request
    let search_request = json!({
        "query_vector": vec![0.1; 768],
        "top_k": 10,
        "filters": {}
    });

    let response = client
        .post(format!("{}/search", API_BASE_URL))
        .json(&search_request)
        .send()
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().expect("Failed to parse response");
    assert!(body.is_object());
    assert!(body["results"].is_array());
}

#[test]
#[ignore] // Requires API server to be running
fn test_spiral_snapshot_endpoint() {
    let client = get_client();

    // Create a snapshot request
    let snapshot_request = json!({
        "data": {
            "type": "test",
            "value": 42
        },
        "seed": "TEST_SEED_API_001"
    });

    let response = client
        .post(format!("{}/spiral/snapshot", API_BASE_URL))
        .json(&snapshot_request)
        .send()
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().expect("Failed to parse response");
    assert!(body["snapshot_id"].is_string());
    assert!(body["coordinates"].is_array());
}

#[test]
#[ignore] // Requires API server to be running
fn test_ledger_block_endpoint() {
    let client = get_client();

    // Create a block request
    let block_request = json!({
        "tic": {
            "tic_id": "tic-api-test-001",
            "seed": "API_TEST_SEED",
            "fixpoint": [0.1, 0.2, 0.3],
            "invariants": {"variance": 0.1},
            "sigma_bar": {"psi": 0.5},
            "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
            "proof": {"merkle_root": "test_root"}
        },
        "snapshot": {
            "id": "snap-api-test-001",
            "coordinates": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    });

    let response = client
        .post(format!("{}/ledger/block", API_BASE_URL))
        .json(&block_request)
        .send()
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().expect("Failed to parse response");
    assert!(body["block_index"].is_number());
    assert!(body["block_hash"].is_string());
}

#[test]
#[ignore] // Requires API server to be running
fn test_api_error_handling() {
    let client = get_client();

    // Send invalid request
    let invalid_request = json!({
        "invalid": "request"
    });

    let response = client
        .post(format!("{}/search", API_BASE_URL))
        .json(&invalid_request)
        .send()
        .expect("Failed to send request");

    // Should return error status
    assert!(response.status().is_client_error() || response.status().is_server_error());
}

#[test]
#[ignore] // Requires API server to be running
fn test_concurrent_requests() {
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let client = get_client();
                let response = client
                    .get(format!("{}/healthz", API_BASE_URL))
                    .send()
                    .expect("Failed to send request");

                assert!(response.status().is_success());
                i
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}
