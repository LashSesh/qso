//! Shadow mode and activation control for safe deployment.
//!
//! Implements SHADOW→ACTIVATE mechanism:
//! - Shadow mode: runs without side effects, only logs
//! - Activation: only when ΔF≤0 & W2_step↓ over 3 windows and Gate stable
//! - Auto-rollback on failure

use crate::metrics::{MetricsCollector, TickMetrics};
use mef_schemas::GateDecision;
use serde::{Deserialize, Serialize};

/// Shadow mode operational state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadowMode {
    /// Shadow mode - no side effects, only observation
    Shadow,
    /// Active mode - commits allowed
    Active,
    /// Rollback mode - reverting to shadow
    Rollback,
}

/// Criteria for activation from shadow to active mode
#[derive(Debug, Clone)]
pub struct ActivationCriteria {
    /// Number of consecutive windows required for activation
    pub window_count: usize,
    
    /// Maximum ΔF threshold (should be ≤ 0)
    pub max_delta_f: f64,
    
    /// Minimum gate stability (fraction of FIRE decisions)
    pub min_gate_stability: f64,
    
    /// Minimum coherence score
    pub min_coherence: f64,
}

impl Default for ActivationCriteria {
    fn default() -> Self {
        Self {
            window_count: 3,
            max_delta_f: 0.0,
            min_gate_stability: 0.8,
            min_coherence: 0.7,
        }
    }
}

/// Controller for shadow mode and activation
pub struct ShadowController {
    mode: ShadowMode,
    criteria: ActivationCriteria,
    metrics_collector: MetricsCollector,
    gate_decisions: Vec<GateDecision>,
    window_passes: usize,
}

impl ShadowController {
    /// Create new shadow controller starting in Shadow mode
    pub fn new(criteria: ActivationCriteria, window_size: usize) -> Self {
        Self {
            mode: ShadowMode::Shadow,
            criteria,
            metrics_collector: MetricsCollector::new(window_size),
            gate_decisions: Vec::new(),
            window_passes: 0,
        }
    }
    
    /// Get current mode
    pub fn mode(&self) -> ShadowMode {
        self.mode
    }
    
    /// Check if currently active (commits allowed)
    pub fn is_active(&self) -> bool {
        matches!(self.mode, ShadowMode::Active)
    }
    
    /// Update with new tick results
    pub fn update(&mut self, metrics: TickMetrics, gate_decision: GateDecision) {
        self.metrics_collector.add(metrics);
        self.gate_decisions.push(gate_decision);
        
        // Keep only recent gate decisions
        if self.gate_decisions.len() > self.metrics_collector.metrics().len() {
            self.gate_decisions.remove(0);
        }
        
        // Check activation/deactivation conditions
        match self.mode {
            ShadowMode::Shadow => {
                if self.check_activation_criteria() {
                    self.window_passes += 1;
                    if self.window_passes >= self.criteria.window_count {
                        self.activate();
                    }
                } else {
                    self.window_passes = 0;
                }
            }
            ShadowMode::Active => {
                if !self.check_stability() {
                    self.rollback();
                }
            }
            ShadowMode::Rollback => {
                // In rollback, go back to shadow
                self.mode = ShadowMode::Shadow;
                self.window_passes = 0;
            }
        }
    }
    
    /// Check if activation criteria are met
    fn check_activation_criteria(&self) -> bool {
        let metrics = self.metrics_collector.metrics();
        if metrics.is_empty() {
            return false;
        }
        
        // Check ΔF ≤ 0 (energy decreasing)
        let delta_f_ok = self.metrics_collector.delta_f_decreasing(3);
        
        // Check W2_step decreasing
        let w2_ok = self.metrics_collector.w2_step_decreasing(3);
        
        // Check gate stability
        let gate_stable = self.check_gate_stability();
        
        // Check coherence
        let coherence_ok = metrics.last().map_or(false, |m| m.s_mand >= self.criteria.min_coherence);
        
        delta_f_ok && w2_ok && gate_stable && coherence_ok
    }
    
    /// Check gate stability (no flicker)
    fn check_gate_stability(&self) -> bool {
        if self.gate_decisions.len() < 5 {
            return false;
        }
        
        // Count FIRE decisions in recent window
        let recent = &self.gate_decisions[self.gate_decisions.len().saturating_sub(10)..];
        let fire_count = recent.iter().filter(|d| matches!(d, GateDecision::FIRE)).count();
        let stability = fire_count as f64 / recent.len() as f64;
        
        stability >= self.criteria.min_gate_stability
    }
    
    /// Check if system remains stable in active mode
    fn check_stability(&self) -> bool {
        let metrics = self.metrics_collector.metrics();
        if metrics.len() < 3 {
            return true; // Not enough data, assume stable
        }
        
        // Check recent metrics for degradation
        let recent = &metrics[metrics.len().saturating_sub(3)..];
        
        // All recent ΔF should be ≤ threshold
        let delta_f_ok = recent.iter().all(|m| m.delta_f <= self.criteria.max_delta_f);
        
        // Coherence should remain high
        let coherence_ok = recent.iter().all(|m| m.s_mand >= self.criteria.min_coherence);
        
        // Gate should remain stable
        let gate_ok = self.check_gate_stability();
        
        delta_f_ok && coherence_ok && gate_ok
    }
    
    /// Activate from shadow to active mode
    fn activate(&mut self) {
        tracing::info!("ACTIVATION: Shadow → Active mode after {} windows", self.window_passes);
        self.mode = ShadowMode::Active;
        self.window_passes = 0;
    }
    
    /// Rollback from active to shadow mode
    fn rollback(&mut self) {
        tracing::warn!("ROLLBACK: Active → Shadow mode due to instability");
        self.mode = ShadowMode::Rollback;
    }
    
    /// Force set mode (for testing)
    pub fn set_mode(&mut self, mode: ShadowMode) {
        self.mode = mode;
    }
    
    /// Get metrics collector reference
    pub fn metrics_collector(&self) -> &MetricsCollector {
        &self.metrics_collector
    }
    
    /// Get status report
    pub fn status_report(&self) -> ShadowStatus {
        ShadowStatus {
            mode: self.mode,
            window_passes: self.window_passes,
            required_passes: self.criteria.window_count,
            gate_stability: if self.gate_decisions.len() >= 5 {
                let recent = &self.gate_decisions[self.gate_decisions.len().saturating_sub(10)..];
                let fire_count = recent.iter().filter(|d| matches!(d, GateDecision::FIRE)).count();
                fire_count as f64 / recent.len() as f64
            } else {
                0.0
            },
            avg_metrics: self.metrics_collector.average(),
        }
    }
}

/// Shadow status report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowStatus {
    pub mode: ShadowMode,
    pub window_passes: usize,
    pub required_passes: usize,
    pub gate_stability: f64,
    pub avg_metrics: Option<TickMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shadow_controller_creation() {
        let criteria = ActivationCriteria::default();
        let controller = ShadowController::new(criteria, 10);
        
        assert_eq!(controller.mode(), ShadowMode::Shadow);
        assert!(!controller.is_active());
    }
    
    #[test]
    fn test_shadow_to_active_transition() {
        let criteria = ActivationCriteria {
            window_count: 2,
            max_delta_f: 0.0,
            min_gate_stability: 0.5,
            min_coherence: 0.5,
        };
        let mut controller = ShadowController::new(criteria, 10);
        
        // Add improving metrics over multiple windows
        for _ in 0..6 {
            let m = TickMetrics {
                bi: 1.0,
                delta_f: -0.1, // Negative (improving)
                w2_step: 0.1,
                lambda_gap: 0.2,
                s_mand: 0.8,
                duty_por: 1.0,
                elapsed_ms: 10.0,
            };
            controller.update(m, GateDecision::FIRE);
        }
        
        // Should not activate yet (need decreasing trend)
    }
    
    #[test]
    fn test_status_report() {
        let criteria = ActivationCriteria::default();
        let mut controller = ShadowController::new(criteria, 10);
        
        let m = TickMetrics {
            bi: 1.0,
            delta_f: 0.0,
            w2_step: 0.1,
            lambda_gap: 0.2,
            s_mand: 0.8,
            duty_por: 1.0,
            elapsed_ms: 10.0,
        };
        controller.update(m, GateDecision::FIRE);
        
        let status = controller.status_report();
        assert_eq!(status.mode, ShadowMode::Shadow);
    }
}
