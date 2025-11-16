#[derive(Debug, Clone)]
pub struct MandorlaField {
    pub threshold: f64,
    pub alpha: f64,
    pub beta: f64,
    pub resonance: f64,
    pub inputs: Vec<Vec<f64>>,
    pub history: Vec<f64>,
    pub current_theta: f64,
}

impl MandorlaField {
    pub fn new(threshold: f64, alpha: f64, beta: f64) -> Self {
        Self {
            threshold,
            alpha,
            beta,
            resonance: 0.0,
            inputs: Vec::new(),
            history: Vec::new(),
            current_theta: threshold,
        }
    }

    pub fn default() -> Self {
        Self::new(0.985, 0.5, 0.5)
    }

    pub fn add_input(&mut self, vec: impl Into<Vec<f64>>) {
        self.inputs.push(vec.into());
    }

    pub fn clear_inputs(&mut self) {
        self.inputs.clear();
    }

    pub fn calc_resonance(&mut self) -> f64 {
        if self.inputs.len() < 2 {
            self.resonance = 0.0;
            return 0.0;
        }
        let mut sims = Vec::new();
        for i in 0..self.inputs.len() {
            for j in (i + 1)..self.inputs.len() {
                let a = &self.inputs[i];
                let b = &self.inputs[j];
                let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
                let norm_a = a.iter().map(|v| v * v).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
                sims.push(dot / (norm_a * norm_b + 1e-12));
            }
        }
        self.resonance = sims.iter().sum::<f64>() / sims.len() as f64;
        self.history.push(self.resonance);
        self.resonance
    }

    pub fn calc_entropy(&self) -> f64 {
        if self.inputs.is_empty() {
            return 0.0;
        }
        let mut data = Vec::new();
        for vec in &self.inputs {
            data.extend(vec.iter().map(|v| v.abs()));
        }
        let sum: f64 = data.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        data.iter()
            .map(|v| {
                let p = v / sum;
                -p * (p + 1e-12).log2()
            })
            .sum()
    }

    pub fn calc_variance(&self) -> f64 {
        if self.inputs.is_empty() {
            return 0.0;
        }
        let mut data = Vec::new();
        for vec in &self.inputs {
            data.extend(vec.iter().cloned());
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64
    }

    pub fn decision_trigger(&mut self) -> bool {
        let res = self.calc_resonance();
        if self.alpha != 0.0 || self.beta != 0.0 {
            let entropy = self.calc_entropy();
            let variance = self.calc_variance();
            self.current_theta = self.alpha * entropy + self.beta * variance;
        } else {
            self.current_theta = self.threshold;
        }
        res > self.current_theta
    }
}

impl Default for MandorlaField {
    fn default() -> Self {
        Self::new(0.985, 0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision() {
        let mut mf = MandorlaField::default();
        mf.add_input(vec![1.0, 1.0, 1.0]);
        mf.add_input(vec![1.0, 1.0, 1.0]);
        assert!(mf.calc_resonance() > 0.9);
    }
}
