#[derive(Debug, Clone)]
pub struct FieldVector {
    pub vec: Vec<f64>,
    pub omega: f64,
    pub phi: f64,
    pub history: Vec<f64>,
}

impl FieldVector {
    pub fn new(data: impl Into<Vec<f64>>, omega: f64) -> Self {
        let vec = data.into();
        Self {
            vec,
            omega,
            phi: 0.0,
            history: Vec::new(),
        }
    }

    pub fn norm(&self) -> f64 {
        self.vec.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn normalize(&mut self) -> &Vec<f64> {
        let nrm = self.norm();
        if nrm > 0.0 {
            for val in &mut self.vec {
                *val /= nrm;
            }
        }
        &self.vec
    }

    pub fn similarity(&self, other: &[f64]) -> f64 {
        let denom = self.norm() * other.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-12;
        let dot = self
            .vec
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        dot / denom
    }

    pub fn add(&self, other: &[f64]) -> FieldVector {
        FieldVector::new(
            self.vec
                .iter()
                .zip(other.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<f64>>(),
            self.omega,
        )
    }

    pub fn scale(&self, factor: f64) -> FieldVector {
        FieldVector::new(
            self.vec.iter().map(|v| v * factor).collect::<Vec<f64>>(),
            self.omega,
        )
    }

    pub fn trm2_update(
        &mut self,
        inputs: &[f64],
        kappas: Option<&[f64]>,
        thetas: Option<&[f64]>,
        dt: f64,
    ) -> f64 {
        let n = self.vec.len();
        let kappas_buf;
        let kappas = match kappas {
            Some(values) => values,
            None => {
                kappas_buf = vec![1.0; n];
                &kappas_buf
            }
        };
        let thetas_buf;
        let thetas = match thetas {
            Some(values) => values,
            None => {
                thetas_buf = (0..n)
                    .map(|i| (2.0 * std::f64::consts::PI * i as f64) / n as f64)
                    .collect::<Vec<f64>>();
                &thetas_buf
            }
        };
        let mut dphi = self.omega;
        for i in 0..n {
            dphi += kappas[i] * inputs[i] * (thetas[i] - self.phi).sin();
        }
        self.phi += dphi * dt;
        self.history.push(self.phi);
        self.phi.sin()
    }

    pub fn as_array(&self) -> Vec<f64> {
        self.vec.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization() {
        let mut fv = FieldVector::new(vec![3.0, 0.0, 4.0], 0.0);
        fv.normalize();
        let norm = fv.norm();
        assert!((norm - 1.0).abs() < 1e-9);
    }
}
