/*!
 * Audit and Logging System for MEF-Core
 * Provides comprehensive logging and audit trail functionality.
 */

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for EventSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSeverity::Info => write!(f, "INFO"),
            EventSeverity::Warning => write!(f, "WARNING"),
            EventSeverity::Error => write!(f, "ERROR"),
        }
    }
}

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub timestamp: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub component: String,
    pub severity: EventSeverity,
    pub details: serde_json::Value,
}

/// Audit report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub generated: String,
    pub period: ReportPeriod,
    pub summary: ReportSummary,
    pub event_counts: HashMap<String, usize>,
    pub component_counts: HashMap<String, usize>,
    pub severity_counts: HashMap<String, usize>,
    pub recent_errors: Vec<AuditEvent>,
}

/// Report time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportPeriod {
    pub start: String,
    pub end: String,
}

/// Report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_events: usize,
    pub event_types: usize,
    pub active_components: usize,
    pub errors: usize,
    pub warnings: usize,
}

/// MEF Audit Logger
/// Tracks all operations and provides audit trail export
pub struct MEFAuditLogger {
    log_path: PathBuf,
    event_log_file: PathBuf,
    event_buffer: Vec<AuditEvent>,
    buffer_size: usize,
}

impl MEFAuditLogger {
    /// Initialize audit logger
    ///
    /// # Arguments
    /// * `log_path` - Directory for log storage
    pub fn new(log_path: impl AsRef<Path>) -> Result<Self> {
        let log_path = log_path.as_ref().to_path_buf();
        fs::create_dir_all(&log_path)?;

        let event_log_file = log_path.join("events.jsonl");

        Ok(Self {
            log_path,
            event_log_file,
            event_buffer: Vec::new(),
            buffer_size: 100,
        })
    }

    /// Log a system event
    ///
    /// # Arguments
    /// * `event_type` - Type of event (e.g., "SNAPSHOT_CREATED", "TIC_GENERATED")
    /// * `component` - Component generating the event
    /// * `details` - Event details
    /// * `severity` - Log severity level
    ///
    /// # Returns
    /// Event ID
    pub fn log_event(
        &mut self,
        event_type: &str,
        component: &str,
        details: serde_json::Value,
        severity: EventSeverity,
    ) -> Result<String> {
        let event_id = self.generate_event_id(event_type, &details);
        let timestamp = Utc::now().to_rfc3339();

        let event = AuditEvent {
            event_id: event_id.clone(),
            timestamp,
            event_type: event_type.to_string(),
            component: component.to_string(),
            severity,
            details,
        };

        // Add to buffer
        self.event_buffer.push(event);

        // Write to event log if buffer is full
        if self.event_buffer.len() >= self.buffer_size {
            self.flush_event_buffer()?;
        }

        // Log message
        log::log!(
            match severity {
                EventSeverity::Error => log::Level::Error,
                EventSeverity::Warning => log::Level::Warn,
                EventSeverity::Info => log::Level::Info,
            },
            "[{}] {}: {}",
            component,
            event_type,
            event_id
        );

        Ok(event_id)
    }

    /// Generate unique event ID
    fn generate_event_id(&self, event_type: &str, _details: &serde_json::Value) -> String {
        let timestamp = Utc::now().to_rfc3339();
        let content = format!("{}_{}", event_type, timestamp);
        let hash = Sha256::digest(content.as_bytes());
        format!("{:x}", hash)[..16].to_string()
    }

    /// Write buffered events to disk
    fn flush_event_buffer(&mut self) -> Result<()> {
        if self.event_buffer.is_empty() {
            return Ok(());
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.event_log_file)?;

        for event in &self.event_buffer {
            let json = serde_json::to_string(event)?;
            writeln!(file, "{}", json)?;
        }

        self.event_buffer.clear();
        Ok(())
    }

    /// Log snapshot creation event
    pub fn log_snapshot_creation(&mut self, snapshot: &serde_json::Value) -> Result<String> {
        let details = serde_json::json!({
            "snapshot_id": snapshot["id"],
            "seed": snapshot["seed"],
            "phase": snapshot["phase"],
            "por": snapshot["metrics"]["por"]
        });

        self.log_event("SNAPSHOT_CREATED", "Spiral", details, EventSeverity::Info)
    }

    /// Log TIC generation event
    pub fn log_tic_generation(
        &mut self,
        tic: &serde_json::Value,
        convergence_info: &serde_json::Value,
    ) -> Result<String> {
        let details = serde_json::json!({
            "tic_id": tic["tic_id"],
            "source_snapshot": tic["source_snapshot"],
            "converged": convergence_info.get("converged").and_then(|v| v.as_bool()).unwrap_or(false),
            "iterations": convergence_info.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0),
            "por": tic["proof"]["por"]
        });

        self.log_event(
            "TIC_GENERATED",
            "TIC-Crystallizer",
            details,
            EventSeverity::Info,
        )
    }

    /// Log ledger commit event
    pub fn log_ledger_commit(&mut self, block: &serde_json::Value) -> Result<String> {
        let details = serde_json::json!({
            "block_index": block["index"],
            "block_hash": block["hash"],
            "tic_id": block["tic_id"],
            "timestamp": block["timestamp"]
        });

        self.log_event("LEDGER_COMMIT", "MEF-Ledger", details, EventSeverity::Info)
    }

    /// Log validation result
    pub fn log_validation_result(
        &mut self,
        entity_type: &str,
        entity_id: &str,
        is_valid: bool,
        details: serde_json::Value,
    ) -> Result<String> {
        let event_type = format!("{}_VALIDATION", entity_type.to_uppercase());
        let severity = if is_valid {
            EventSeverity::Info
        } else {
            EventSeverity::Warning
        };

        let event_details = serde_json::json!({
            "entity_id": entity_id,
            "valid": is_valid,
            "details": details
        });

        self.log_event(&event_type, "Validator", event_details, severity)
    }

    /// Log error event
    pub fn log_error(
        &mut self,
        component: &str,
        error_type: &str,
        error_message: &str,
        context: serde_json::Value,
    ) -> Result<String> {
        let details = serde_json::json!({
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        });

        self.log_event("ERROR", component, details, EventSeverity::Error)
    }

    /// Retrieve events based on filters
    ///
    /// # Arguments
    /// * `event_type` - Filter by event type
    /// * `component` - Filter by component
    /// * `limit` - Maximum number of events to return
    ///
    /// # Returns
    /// List of matching events
    pub fn get_events(
        &mut self,
        event_type: Option<&str>,
        component: Option<&str>,
        limit: usize,
    ) -> Result<Vec<AuditEvent>> {
        // Flush buffer first
        self.flush_event_buffer()?;

        let mut events = Vec::new();

        if self.event_log_file.exists() {
            let file = File::open(&self.event_log_file)?;
            let reader = BufReader::new(file);

            for line in reader.lines().map_while(Result::ok) {
                if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                    // Apply filters
                    if let Some(et) = event_type {
                        if event.event_type != et {
                            continue;
                        }
                    }
                    if let Some(comp) = component {
                        if event.component != comp {
                            continue;
                        }
                    }

                    events.push(event);

                    if events.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(events)
    }

    /// Generate comprehensive audit report
    ///
    /// # Returns
    /// Audit report
    pub fn generate_audit_report(&mut self) -> Result<AuditReport> {
        // Get all events
        let events = self.get_events(None, None, 10000)?;

        // Analyze events
        let mut event_counts: HashMap<String, usize> = HashMap::new();
        let mut component_counts: HashMap<String, usize> = HashMap::new();
        let mut severity_counts: HashMap<String, usize> = HashMap::new();

        for event in &events {
            *event_counts.entry(event.event_type.clone()).or_insert(0) += 1;
            *component_counts.entry(event.component.clone()).or_insert(0) += 1;
            *severity_counts
                .entry(event.severity.to_string())
                .or_insert(0) += 1;
        }

        let errors = *severity_counts.get("ERROR").unwrap_or(&0);
        let warnings = *severity_counts.get("WARNING").unwrap_or(&0);

        // Get recent errors
        let recent_errors: Vec<AuditEvent> = events
            .iter()
            .rev()
            .filter(|e| e.severity == EventSeverity::Error)
            .take(10)
            .cloned()
            .collect();

        Ok(AuditReport {
            generated: Utc::now().to_rfc3339(),
            period: ReportPeriod {
                start: "beginning".to_string(),
                end: "current".to_string(),
            },
            summary: ReportSummary {
                total_events: events.len(),
                event_types: event_counts.len(),
                active_components: component_counts.len(),
                errors,
                warnings,
            },
            event_counts,
            component_counts,
            severity_counts,
            recent_errors,
        })
    }

    /// Get audit system statistics
    pub fn get_statistics(&mut self) -> Result<serde_json::Value> {
        // Flush buffer
        self.flush_event_buffer()?;

        // Count log files
        let log_files: Vec<_> = fs::read_dir(&self.log_path)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "log")
                    .unwrap_or(false)
            })
            .collect();

        // Calculate sizes
        let mut total_size = log_files
            .iter()
            .filter_map(|f| f.metadata().ok())
            .map(|m| m.len())
            .sum::<u64>();

        if self.event_log_file.exists() {
            if let Ok(metadata) = fs::metadata(&self.event_log_file) {
                total_size += metadata.len();
            }
        }

        // Get event count
        let event_count = if self.event_log_file.exists() {
            BufReader::new(File::open(&self.event_log_file)?)
                .lines()
                .count()
        } else {
            0
        };

        Ok(serde_json::json!({
            "log_files": log_files.len(),
            "event_count": event_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size as f64 / (1024.0 * 1024.0),
            "buffer_size": self.event_buffer.len()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_create_logger() {
        let temp_dir = env::temp_dir().join("test_audit_create");
        let logger = MEFAuditLogger::new(&temp_dir).unwrap();
        assert!(logger.log_path.exists());
    }

    #[test]
    fn test_log_event() {
        let temp_dir = env::temp_dir().join("test_audit_log");
        let mut logger = MEFAuditLogger::new(&temp_dir).unwrap();

        let details = serde_json::json!({
            "test_key": "test_value"
        });

        let event_id = logger
            .log_event("TEST_EVENT", "TestComponent", details, EventSeverity::Info)
            .unwrap();

        assert!(!event_id.is_empty());
        assert_eq!(event_id.len(), 16);
    }

    #[test]
    fn test_event_buffer_flush() {
        let temp_dir = env::temp_dir().join("test_audit_buffer");
        let mut logger = MEFAuditLogger::new(&temp_dir).unwrap();
        logger.buffer_size = 2; // Small buffer for testing

        // Add events to trigger flush
        for i in 0..3 {
            logger
                .log_event(
                    "TEST_EVENT",
                    "TestComponent",
                    serde_json::json!({"index": i}),
                    EventSeverity::Info,
                )
                .unwrap();
        }

        // Manually flush to ensure events are written
        logger.flush_event_buffer().unwrap();

        // Check that events were written
        let events = logger.get_events(None, None, 10).unwrap();
        assert!(events.len() >= 2);
    }

    #[test]
    fn test_get_events_with_filter() {
        let temp_dir = env::temp_dir().join("test_audit_filter");
        let mut logger = MEFAuditLogger::new(&temp_dir).unwrap();

        // Log different event types
        logger
            .log_event(
                "EVENT_A",
                "ComponentA",
                serde_json::json!({}),
                EventSeverity::Info,
            )
            .unwrap();
        logger
            .log_event(
                "EVENT_B",
                "ComponentB",
                serde_json::json!({}),
                EventSeverity::Info,
            )
            .unwrap();

        logger.flush_event_buffer().unwrap();

        // Filter by event type
        let events = logger.get_events(Some("EVENT_A"), None, 10).unwrap();
        assert!(events.iter().all(|e| e.event_type == "EVENT_A"));
    }

    #[test]
    fn test_generate_report() {
        let temp_dir = env::temp_dir().join("test_audit_report");
        let mut logger = MEFAuditLogger::new(&temp_dir).unwrap();

        // Log some events
        logger
            .log_event(
                "TEST_EVENT",
                "TestComponent",
                serde_json::json!({}),
                EventSeverity::Info,
            )
            .unwrap();
        logger
            .log_event(
                "ERROR_EVENT",
                "TestComponent",
                serde_json::json!({}),
                EventSeverity::Error,
            )
            .unwrap();

        logger.flush_event_buffer().unwrap();

        let report = logger.generate_audit_report().unwrap();
        assert!(report.summary.total_events >= 2);
        assert!(report.summary.errors >= 1);
    }

    #[test]
    fn test_log_snapshot_creation() {
        let temp_dir = env::temp_dir().join("test_audit_snapshot");
        let mut logger = MEFAuditLogger::new(&temp_dir).unwrap();

        let snapshot = serde_json::json!({
            "id": "snap-001",
            "seed": "TEST_SEED",
            "phase": 1.5,
            "metrics": {
                "por": "PASS"
            }
        });

        let event_id = logger.log_snapshot_creation(&snapshot).unwrap();
        assert!(!event_id.is_empty());
    }

    #[test]
    fn test_statistics() {
        let temp_dir = env::temp_dir().join("test_audit_stats");
        let mut logger = MEFAuditLogger::new(&temp_dir).unwrap();

        logger
            .log_event(
                "TEST_EVENT",
                "TestComponent",
                serde_json::json!({}),
                EventSeverity::Info,
            )
            .unwrap();
        logger.flush_event_buffer().unwrap();

        let stats = logger.get_statistics().unwrap();
        assert!(stats["event_count"].as_u64().unwrap() >= 1);
    }
}
