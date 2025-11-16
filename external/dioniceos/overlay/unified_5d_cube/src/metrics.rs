//! Metrics collection and export for the 5D Cube overlay.
//!
//! Tracks: BI, ΔF, W2_step, λ_gap, S_mand, Duty/PoR

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metrics collected during a single tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickMetrics {
    /// Betti number approximation (topological feature)
    pub bi: f64,
    
    /// Energy delta (ΔF) - Lyapunov function change
    pub delta_f: f64,
    
    /// Wasserstein-2 step distance (W2_step)
    pub w2_step: f64,
    
    /// Spectral gap (λ_gap)
    pub lambda_gap: f64,
    
    /// Mandorla score (S_mand) - coherence measure
    pub s_mand: f64,
    
    /// Duty cycle / PoR validity (1.0 = valid, 0.0 = invalid)
    pub duty_por: f64,
    
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
}

impl TickMetrics {
    /// Check if metrics indicate system is improving
    pub fn is_improving(&self, prev: &TickMetrics) -> bool {
        // ΔF should be decreasing (energy should go down)
        let delta_f_improving = self.delta_f < prev.delta_f;
        
        // W2_step should be decreasing (states should be closer)
        let w2_improving = self.w2_step < prev.w2_step;
        
        delta_f_improving && w2_improving
    }
    
    /// Check if system is stable
    pub fn is_stable(&self) -> bool {
        // Small energy change and high coherence
        self.delta_f.abs() < 0.01 && self.s_mand > 0.8
    }
}

/// Format for metrics export
#[derive(Debug, Clone, Copy)]
pub enum MetricsFormat {
    /// CSV format
    CSV,
    /// JSON format
    JSON,
}

/// Collector for accumulating metrics over multiple ticks
pub struct MetricsCollector {
    metrics: Vec<TickMetrics>,
    window_size: usize,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(window_size: usize) -> Self {
        Self {
            metrics: Vec::new(),
            window_size,
        }
    }
    
    /// Add metrics from a tick
    pub fn add(&mut self, metrics: TickMetrics) {
        self.metrics.push(metrics);
        
        // Keep only last window_size metrics
        if self.metrics.len() > self.window_size {
            self.metrics.remove(0);
        }
    }
    
    /// Get reference to collected metrics
    pub fn metrics(&self) -> &[TickMetrics] {
        &self.metrics
    }
    
    /// Check if system is improving over the window
    pub fn is_improving_trend(&self) -> bool {
        if self.metrics.len() < 2 {
            return false;
        }
        
        let mut improving_count = 0;
        for i in 1..self.metrics.len() {
            if self.metrics[i].is_improving(&self.metrics[i - 1]) {
                improving_count += 1;
            }
        }
        
        // Majority of transitions should be improving
        improving_count * 2 > self.metrics.len()
    }
    
    /// Check if ΔF is consistently decreasing
    pub fn delta_f_decreasing(&self, window: usize) -> bool {
        if self.metrics.len() < window {
            return false;
        }
        
        let start_idx = self.metrics.len() - window;
        for i in (start_idx + 1)..self.metrics.len() {
            if self.metrics[i].delta_f >= self.metrics[i - 1].delta_f {
                return false;
            }
        }
        
        true
    }
    
    /// Check if W2_step is consistently decreasing
    pub fn w2_step_decreasing(&self, window: usize) -> bool {
        if self.metrics.len() < window {
            return false;
        }
        
        let start_idx = self.metrics.len() - window;
        for i in (start_idx + 1)..self.metrics.len() {
            if self.metrics[i].w2_step >= self.metrics[i - 1].w2_step {
                return false;
            }
        }
        
        true
    }
    
    /// Export metrics to file
    pub fn export(&self, path: &Path, format: MetricsFormat) -> std::io::Result<()> {
        match format {
            MetricsFormat::CSV => self.export_csv(path),
            MetricsFormat::JSON => self.export_json(path),
        }
    }
    
    fn export_csv(&self, path: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        // Write header
        writeln!(
            file,
            "tick,bi,delta_f,w2_step,lambda_gap,s_mand,duty_por,elapsed_ms"
        )?;
        
        // Write data
        for (i, m) in self.metrics.iter().enumerate() {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                i, m.bi, m.delta_f, m.w2_step, m.lambda_gap, m.s_mand, m.duty_por, m.elapsed_ms
            )?;
        }
        
        Ok(())
    }
    
    fn export_json(&self, path: &Path) -> std::io::Result<()> {
        use std::fs::File;
        
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &self.metrics)?;
        
        Ok(())
    }
    
    /// Compute average metrics over the window
    pub fn average(&self) -> Option<TickMetrics> {
        if self.metrics.is_empty() {
            return None;
        }
        
        let n = self.metrics.len() as f64;
        let mut avg = TickMetrics {
            bi: 0.0,
            delta_f: 0.0,
            w2_step: 0.0,
            lambda_gap: 0.0,
            s_mand: 0.0,
            duty_por: 0.0,
            elapsed_ms: 0.0,
        };
        
        for m in &self.metrics {
            avg.bi += m.bi;
            avg.delta_f += m.delta_f;
            avg.w2_step += m.w2_step;
            avg.lambda_gap += m.lambda_gap;
            avg.s_mand += m.s_mand;
            avg.duty_por += m.duty_por;
            avg.elapsed_ms += m.elapsed_ms;
        }
        
        avg.bi /= n;
        avg.delta_f /= n;
        avg.w2_step /= n;
        avg.lambda_gap /= n;
        avg.s_mand /= n;
        avg.duty_por /= n;
        avg.elapsed_ms /= n;
        
        Some(avg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new(10);
        
        let m1 = TickMetrics {
            bi: 1.0,
            delta_f: 0.5,
            w2_step: 0.3,
            lambda_gap: 0.2,
            s_mand: 0.8,
            duty_por: 1.0,
            elapsed_ms: 10.0,
        };
        
        collector.add(m1);
        assert_eq!(collector.metrics().len(), 1);
    }
    
    #[test]
    fn test_improving_trend() {
        let mut collector = MetricsCollector::new(10);
        
        // Add improving metrics
        for i in 0..5 {
            let m = TickMetrics {
                bi: 1.0,
                delta_f: 0.5 - (i as f64 * 0.1),  // Decreasing
                w2_step: 0.3 - (i as f64 * 0.05), // Decreasing
                lambda_gap: 0.2,
                s_mand: 0.8,
                duty_por: 1.0,
                elapsed_ms: 10.0,
            };
            collector.add(m);
        }
        
        assert!(collector.is_improving_trend());
    }
}
