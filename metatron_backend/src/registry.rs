//! Backend registry and selection logic
//!
//! Manages available quantum backends and selects appropriate backends
//! based on execution mode and circuit requirements.

use crate::backends::{BackendCapabilities, BoxedBackend};
use crate::circuit::MetatronCircuit;
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

/// Backend execution mode
///
/// Controls which backends are eligible for circuit execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendMode {
    /// Only use simulators (SAFE DEFAULT)
    ///
    /// This is the default mode. No QPU backends will be used.
    SimulationOnly,

    /// Prefer simulators, but allow QPUs if explicitly enabled
    ///
    /// QPU backends must be explicitly marked as available to be used.
    QpuEnabledAuto,

    /// Force a specific provider
    ///
    /// Only backends from the specified provider will be considered.
    /// Example: "local", "ibm", "azure"
    ForceProvider(String),
}

impl Default for BackendMode {
    fn default() -> Self {
        Self::SimulationOnly
    }
}

/// Registry of available quantum backends
///
/// Manages backend registration and provides selection logic based on
/// execution mode and circuit requirements.
pub struct BackendRegistry {
    /// Registered backends by name
    backends: HashMap<String, BoxedBackend>,

    /// Current execution mode
    mode: BackendMode,
}

impl BackendRegistry {
    /// Create a new backend registry with default mode (SimulationOnly)
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            mode: BackendMode::default(),
        }
    }

    /// Create a registry with a specific mode
    pub fn with_mode(mode: BackendMode) -> Self {
        Self {
            backends: HashMap::new(),
            mode,
        }
    }

    /// Register a backend
    pub fn register(&mut self, name: String, backend: BoxedBackend) -> Result<()> {
        let caps = backend.info();

        tracing::info!(
            "Registering backend '{}' (provider: {}, {} qubits, {})",
            name,
            caps.provider,
            caps.num_qubits,
            if caps.is_simulator {
                "simulator"
            } else {
                "QPU"
            }
        );

        if self.backends.contains_key(&name) {
            bail!("Backend '{}' is already registered", name);
        }

        self.backends.insert(name, backend);
        Ok(())
    }

    /// Set the execution mode
    pub fn set_mode(&mut self, mode: BackendMode) {
        tracing::info!("Backend mode changed: {:?}", mode);
        self.mode = mode;
    }

    /// Get the current execution mode
    pub fn mode(&self) -> &BackendMode {
        &self.mode
    }

    /// Get a backend by name
    pub fn get(&self, name: &str) -> Option<&BoxedBackend> {
        self.backends.get(name)
    }

    /// List all registered backends
    pub fn list_backends(&self) -> Vec<BackendCapabilities> {
        self.backends.values().map(|b| b.info()).collect()
    }

    /// Select the best backend for a given circuit
    ///
    /// Selection logic:
    /// - SimulationOnly: Only consider simulators
    /// - QpuEnabledAuto: Prefer simulators, use QPUs if available and no simulator works
    /// - ForceProvider: Only use backends from the specified provider
    pub fn select_backend_for(&self, circuit: &MetatronCircuit) -> Result<&BoxedBackend> {
        let num_qubits = circuit.num_qubits;

        match &self.mode {
            BackendMode::SimulationOnly => {
                // Only consider simulators
                self.find_best_simulator(num_qubits)
            }

            BackendMode::QpuEnabledAuto => {
                // Try simulators first
                if let Ok(backend) = self.find_best_simulator(num_qubits) {
                    return Ok(backend);
                }

                // Fall back to QPUs if available
                self.find_best_qpu(num_qubits)
            }

            BackendMode::ForceProvider(provider) => {
                // Only consider backends from the specified provider
                self.find_best_for_provider(provider, num_qubits)
            }
        }
    }

    /// Find the best simulator for a given qubit count
    fn find_best_simulator(&self, num_qubits: usize) -> Result<&BoxedBackend> {
        self.backends
            .values()
            .filter(|b| {
                let caps = b.info();
                caps.is_simulator && caps.available && b.can_run(num_qubits)
            })
            .min_by_key(|b| b.info().num_qubits) // Prefer smallest that fits
            .ok_or_else(|| anyhow!("No available simulator found for {} qubits", num_qubits))
    }

    /// Find the best QPU for a given qubit count
    fn find_best_qpu(&self, num_qubits: usize) -> Result<&BoxedBackend> {
        self.backends
            .values()
            .filter(|b| {
                let caps = b.info();
                !caps.is_simulator && caps.available && b.can_run(num_qubits)
            })
            .min_by_key(|b| b.info().num_qubits) // Prefer smallest that fits
            .ok_or_else(|| {
                anyhow!(
                    "No available QPU found for {} qubits (mode: {:?})",
                    num_qubits,
                    self.mode
                )
            })
    }

    /// Find the best backend for a specific provider
    fn find_best_for_provider(&self, provider: &str, num_qubits: usize) -> Result<&BoxedBackend> {
        self.backends
            .values()
            .filter(|b| {
                let caps = b.info();
                caps.provider == provider && caps.available && b.can_run(num_qubits)
            })
            .min_by_key(|b| {
                let caps = b.info();
                // Prefer simulators over QPUs, then prefer smallest that fits
                (if caps.is_simulator { 0 } else { 1 }, caps.num_qubits)
            })
            .ok_or_else(|| {
                anyhow!(
                    "No available backend from provider '{}' for {} qubits",
                    provider,
                    num_qubits
                )
            })
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::local::LocalSimulatorBackend;

    #[test]
    fn test_registry_creation() {
        let registry = BackendRegistry::new();
        assert_eq!(registry.mode(), &BackendMode::SimulationOnly);
        assert_eq!(registry.list_backends().len(), 0);
    }

    #[test]
    fn test_backend_registration() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(LocalSimulatorBackend::new());

        registry.register("local_sim".to_string(), backend).unwrap();
        assert_eq!(registry.list_backends().len(), 1);
        assert!(registry.get("local_sim").is_some());
    }

    #[test]
    fn test_mode_switching() {
        let mut registry = BackendRegistry::new();
        assert_eq!(registry.mode(), &BackendMode::SimulationOnly);

        registry.set_mode(BackendMode::QpuEnabledAuto);
        assert_eq!(registry.mode(), &BackendMode::QpuEnabledAuto);

        registry.set_mode(BackendMode::ForceProvider("ibm".to_string()));
        assert_eq!(
            registry.mode(),
            &BackendMode::ForceProvider("ibm".to_string())
        );
    }

    #[test]
    fn test_backend_selection_simulation_only() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(LocalSimulatorBackend::new());
        registry.register("local_sim".to_string(), backend).unwrap();

        let circuit = MetatronCircuit::new(2);
        let selected = registry.select_backend_for(&circuit).unwrap();

        assert!(selected.info().is_simulator);
        assert_eq!(selected.info().provider, "local");
    }
}
