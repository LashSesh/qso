//! MetatronAdapter for routing integration

use crate::route_selection::select_route;
use mef_schemas::RouteSpec;
use std::collections::HashMap;

/// Adapter mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterMode {
    /// In-process mode (default)
    InProcess,

    /// Service mode (external routing service)
    Service,
}

/// MetatronAdapter for route selection
pub struct MetatronAdapter {
    mode: AdapterMode,
}

impl MetatronAdapter {
    /// Create a new adapter with the given mode
    pub fn new(mode: AdapterMode) -> Self {
        Self { mode }
    }

    /// Select a route
    pub fn select_route(
        &self,
        seed: &str,
        metrics: &HashMap<String, f64>,
    ) -> crate::Result<RouteSpec> {
        match self.mode {
            AdapterMode::InProcess => {
                // Use in-process route selection
                select_route(seed, metrics)
            }
            AdapterMode::Service => {
                // Scaffold for external service call
                // In Phase 2, this would make an HTTP/gRPC call to external service
                Err(crate::RouterError::Adapter(
                    "Service mode not yet implemented".to_string(),
                ))
            }
        }
    }
}

impl Default for MetatronAdapter {
    fn default() -> Self {
        Self::new(AdapterMode::InProcess)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_in_process() {
        let adapter = MetatronAdapter::new(AdapterMode::InProcess);

        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route = adapter.select_route("seed123", &metrics);
        assert!(route.is_ok());
    }

    #[test]
    fn test_adapter_service_not_implemented() {
        let adapter = MetatronAdapter::new(AdapterMode::Service);

        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route = adapter.select_route("seed123", &metrics);
        assert!(route.is_err());
    }

    #[test]
    fn test_default_mode() {
        let adapter = MetatronAdapter::default();

        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route = adapter.select_route("seed123", &metrics);
        assert!(route.is_ok());
    }
}
