#[derive(Debug, Clone)]
pub struct SemanticField {
    prototypes: std::collections::HashMap<String, Vec<f64>>,
}

impl SemanticField {
    pub fn new() -> Self {
        Self {
            prototypes: std::collections::HashMap::new(),
        }
    }

    pub fn add_prototype(&mut self, name: &str, spectrum: &[f64]) {
        self.prototypes.insert(name.to_string(), spectrum.to_vec());
    }

    pub fn classify(&self, spectrum: &[f64], top_k: usize) -> Vec<(String, f64)> {
        let mut sims = Vec::new();
        let norm_s = norm(spectrum);
        if norm_s == 0.0 {
            return Vec::new();
        }
        for (name, proto) in &self.prototypes {
            if proto.len() != spectrum.len() {
                continue;
            }
            let dot = proto
                .iter()
                .zip(spectrum.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let similarity = dot / (norm(proto) * norm_s + 1e-12);
            sims.push((name.clone(), similarity));
        }
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(top_k);
        sims
    }
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[derive(Debug, Clone)]
pub struct ResonanceDiagnostics;

impl ResonanceDiagnostics {
    pub fn entropy(spectrum: &[f64]) -> f64 {
        let mut values: Vec<f64> = spectrum.iter().map(|v| v.abs()).collect();
        let sum: f64 = values.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        for v in &mut values {
            *v /= sum;
        }
        values.iter().map(|p| -p * (p + 1e-12).log2()).sum::<f64>()
    }

    pub fn spectral_centroid(spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() {
            return 0.0;
        }
        let mut weighted_sum = 0.0;
        let mut total = 0.0;
        for (i, value) in spectrum.iter().enumerate() {
            let mag = value.abs();
            weighted_sum += i as f64 * mag;
            total += mag;
        }
        if total == 0.0 {
            0.0
        } else {
            weighted_sum / total
        }
    }

    pub fn sparsity(spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() {
            return 0.0;
        }
        let non_zero = spectrum.iter().filter(|&&v| v.abs() > 1e-6).count();
        non_zero as f64 / spectrum.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify() {
        let mut field = SemanticField::new();
        field.add_prototype("alpha", &[1.0, 0.0, 0.0]);
        let res = field.classify(&[1.0, 0.0, 0.0], 1);
        assert_eq!(res.len(), 1);
    }
}
