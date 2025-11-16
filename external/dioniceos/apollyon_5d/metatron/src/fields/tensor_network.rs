use crate::error::EngineResult;
use crate::fields::tensor::ResonanceTensorField;

#[derive(Debug, Default)]
pub struct TensorNetwork {
    pub fields: Vec<ResonanceTensorField>,
}

impl TensorNetwork {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn with_fields(fields: Vec<ResonanceTensorField>) -> Self {
        Self { fields }
    }

    pub fn add_field(&mut self, field: ResonanceTensorField) {
        self.fields.push(field);
    }

    pub fn step(
        &mut self,
        dt: f64,
        input_modulations: Option<Vec<Vec<f64>>>,
    ) -> EngineResult<Vec<Vec<Vec<Vec<f64>>>>> {
        let mut states = Vec::new();
        for (idx, field) in self.fields.iter_mut().enumerate() {
            let modulation = input_modulations
                .as_ref()
                .and_then(|mods| mods.get(idx))
                .map(|v| v.as_slice());
            states.push(field.step(dt, modulation)?);
        }
        Ok(states)
    }

    pub fn coherence(&self) -> f64 {
        if self.fields.is_empty() {
            return 0.0;
        }
        self.fields.iter().map(|f| f.coherence()).sum::<f64>() / self.fields.len() as f64
    }

    pub fn cross_coherence(&self) -> f64 {
        if self.fields.len() < 2 {
            return 0.0;
        }
        let vectors: Vec<Vec<f64>> = self.fields.iter().map(|f| flatten(f)).collect();
        let mut sims = Vec::new();
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let a = &vectors[i];
                let b = &vectors[j];
                let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
                let norm_a = a.iter().map(|v| v * v).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
                let denom = norm_a * norm_b + 1e-12;
                sims.push(dot / denom);
            }
        }
        sims.iter().sum::<f64>() / sims.len() as f64
    }

    pub fn detect_singularities(&self) -> bool {
        self.fields.iter().any(|f| f.detect_singularity())
    }
}

fn flatten(field: &ResonanceTensorField) -> Vec<f64> {
    let state = field.get_state();
    let mut flat = Vec::new();
    for plane in state {
        for row in plane {
            for value in row {
                flat.push(value);
            }
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_coherence() {
        let mut net = TensorNetwork::new();
        net.add_field(ResonanceTensorField::new((2, 2, 2)));
        net.add_field(ResonanceTensorField::new((2, 2, 2)));
        net.step(0.1, None).unwrap();
        assert!(net.coherence().is_finite());
        assert!(net.cross_coherence().is_finite());
    }
}
