use crate::error::{EngineError, EngineResult};

#[derive(Debug, Clone)]
pub struct ResonanceTensorField {
    pub shape: (usize, usize, usize),
    pub initial_amplitude: f64,
    pub initial_frequency: f64,
    pub initial_phase: f64,
    pub gradient_threshold: f64,
    pub time: f64,
    amplitude: Vec<f64>,
    frequency: Vec<f64>,
    phase: Vec<f64>,
    prev_state: Option<Vec<f64>>,
}

impl ResonanceTensorField {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            shape,
            initial_amplitude: 1.0,
            initial_frequency: 1.0,
            initial_phase: 0.0,
            gradient_threshold: 1e-3,
            time: 0.0,
            amplitude: vec![1.0; size],
            frequency: vec![1.0; size],
            phase: vec![0.0; size],
            prev_state: None,
        }
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        let ny = self.shape.1;
        (x * ny * self.shape.2) + (y * self.shape.2) + z
    }

    pub fn get_state(&self) -> Vec<Vec<Vec<f64>>> {
        let mut result = vec![vec![vec![0.0; self.shape.2]; self.shape.1]; self.shape.0];
        for x in 0..self.shape.0 {
            for y in 0..self.shape.1 {
                for z in 0..self.shape.2 {
                    let idx = self.index(x, y, z);
                    let value = self.amplitude[idx]
                        * ((self.frequency[idx] * self.time + self.phase[idx]).sin());
                    result[x][y][z] = value;
                }
            }
        }
        result
    }

    pub fn step(
        &mut self,
        dt: f64,
        input_modulation: Option<&[f64]>,
    ) -> EngineResult<Vec<Vec<Vec<f64>>>> {
        if let Some(modulation) = input_modulation {
            if modulation.len() != self.phase.len() {
                return Err(EngineError::DimensionMismatch {
                    expected: self.phase.len(),
                    actual: modulation.len(),
                });
            }
            for (phase, delta) in self.phase.iter_mut().zip(modulation.iter()) {
                *phase += *delta;
            }
        }
        let current = self.flatten_state();
        self.prev_state = Some(current);
        self.time += dt;
        Ok(self.get_state())
    }

    fn flatten_state(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.phase.len());
        for x in 0..self.shape.0 {
            for y in 0..self.shape.1 {
                for z in 0..self.shape.2 {
                    let idx = self.index(x, y, z);
                    let value = self.amplitude[idx]
                        * ((self.frequency[idx] * self.time + self.phase[idx]).sin());
                    flat.push(value);
                }
            }
        }
        flat
    }

    pub fn coherence(&self) -> f64 {
        let flat = self.flatten_state();
        if flat.iter().all(|v| v.abs() < 1e-12) {
            return 0.0;
        }
        let mut similarities = Vec::new();
        for i in 0..flat.len() {
            for j in (i + 1)..flat.len() {
                let a = flat[i];
                let b = flat[j];
                let denom = (a.abs() + 1e-12) * (b.abs() + 1e-12);
                similarities.push((a * b) / denom);
            }
        }
        if similarities.is_empty() {
            0.0
        } else {
            similarities.iter().sum::<f64>() / similarities.len() as f64
        }
    }

    pub fn gradient_norm(&self) -> f64 {
        if let Some(prev) = &self.prev_state {
            let current = self.flatten_state();
            current
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        } else {
            0.0
        }
    }

    pub fn detect_singularity(&self) -> bool {
        self.gradient_norm() < self.gradient_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherence_non_negative() {
        let mut field = ResonanceTensorField::new((2, 2, 2));
        field.step(0.1, None).unwrap();
        assert!(field.coherence().is_finite());
    }
}
