#[derive(Debug, Clone)]
pub struct SpiralMemory {
    pub memory: Vec<(Vec<Vec<f64>>, f64)>,
    pub alpha: f64,
    pub history: Vec<f64>,
}

impl SpiralMemory {
    pub fn new(alpha: f64) -> Self {
        Self {
            memory: Vec::new(),
            alpha,
            history: Vec::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(0.1)
    }

    pub fn embed(&self, sequence: &str) -> Vec<f64> {
        let base: Vec<f64> = sequence.chars().map(|c| c as u32 as f64).collect();
        let mut vec = vec![0.0; 5];
        for i in 0..5 {
            let mut acc = 0.0;
            for (idx, value) in base.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * (i as f64 + 1.0) * idx as f64
                    / (base.len() as f64 + 5.0);
                acc += value * angle.cos();
            }
            vec[i] = acc;
        }
        normalize(vec)
    }

    pub fn spiralize(&self, elements: &[String]) -> Vec<Vec<f64>> {
        elements.iter().map(|e| self.embed(e)).collect()
    }

    pub fn psi(&self, vi: &[f64], vj: &[f64]) -> f64 {
        let stab = cosine_similarity(vi, vj);
        let conv = 1.0 / (1.0 + l2_distance(vi, vj));
        let react = (vi.iter().zip(vj.iter()).map(|(a, b)| a - b).sum::<f64>())
            .sin()
            .abs();
        0.5 * stab + 0.3 * conv + 0.2 * react
    }

    pub fn psi_total(&self, points: &[Vec<f64>]) -> f64 {
        points
            .windows(2)
            .map(|pair| self.psi(&pair[0], &pair[1]))
            .sum::<f64>()
    }

    pub fn gradient(&self, points: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if points.len() < 2 {
            return vec![vec![0.0; 5]; points.len()];
        }
        let mut grads = Vec::new();
        for i in 0..(points.len() - 1) {
            let diff: Vec<f64> = points[i + 1]
                .iter()
                .zip(points[i].iter())
                .map(|(a, b)| a - b)
                .collect();
            grads.push(normalize(diff));
        }
        grads.push(grads.last().cloned().unwrap_or_else(|| vec![0.0; 5]));
        grads
    }

    pub fn mutate(&self, points: &[Vec<f64>], grads: &[Vec<f64>]) -> Vec<Vec<f64>> {
        points
            .iter()
            .zip(grads.iter())
            .map(|(p, g)| {
                p.iter()
                    .zip(g.iter())
                    .map(|(a, b)| a + self.alpha * b)
                    .collect()
            })
            .collect()
    }

    pub fn proof_of_resonance(&self, psi_old: f64, psi_new: f64, epsilon: f64) -> bool {
        (psi_new - psi_old).abs() < epsilon
    }

    pub fn step(&mut self, elements: &[String], max_iter: usize) -> (Vec<Vec<f64>>, f64) {
        let mut points = self.spiralize(elements);
        let mut psi_val = self.psi_total(&points);
        for _ in 0..max_iter {
            let grads = self.gradient(&points);
            let new_points = self.mutate(&points, &grads);
            let new_psi = self.psi_total(&new_points);
            self.history.push(new_psi);
            if self.proof_of_resonance(psi_val, new_psi, 1e-4) {
                self.memory.push((new_points.clone(), new_psi));
                return (new_points, new_psi);
            }
            points = new_points;
            psi_val = new_psi;
        }
        self.memory.push((points.clone(), psi_val));
        (points, psi_val)
    }
}

impl Default for SpiralMemory {
    fn default() -> Self {
        Self::new(0.1)
    }
}

fn normalize(mut vec: Vec<f64>) -> Vec<f64> {
    let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    dot / (norm_a * norm_b + 1e-12)
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_step() {
        let mut sm = SpiralMemory::default();
        let (points, psi) = sm.step(&["A".to_string(), "B".to_string()], 10);
        assert_eq!(points.len(), 2);
        assert!(psi.is_finite());
    }
}
