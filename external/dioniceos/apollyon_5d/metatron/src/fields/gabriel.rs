#[derive(Debug)]
pub struct GabrielCell {
    pub psi: f64,
    pub rho: f64,
    pub omega: f64,
    pub learn_rate: f64,
    pub output: f64,
    neighbors: Vec<*mut GabrielCell>,
}

impl GabrielCell {
    pub fn new(psi: f64, rho: f64, omega: f64, learn_rate: f64) -> Self {
        let output = psi * rho * omega;
        Self {
            psi,
            rho,
            omega,
            learn_rate,
            output,
            neighbors: Vec::new(),
        }
    }

    pub fn simple() -> Self {
        Self::new(1.0, 1.0, 1.0, 0.12)
    }

    pub fn activate(&mut self, input: Option<f64>) -> f64 {
        if let Some(val) = input {
            self.psi = (1.0 - self.learn_rate) * self.psi + self.learn_rate * val;
        }
        self.output = self.psi * self.rho * self.omega;
        self.output
    }

    pub fn feedback(&mut self, target: f64) {
        let err = target - self.output;
        self.psi += self.learn_rate * err;
        self.rho += self.learn_rate * err.tanh();
        self.omega += self.learn_rate * err.sin();
        self.psi = self.psi.clamp(0.01, 10.0);
        self.rho = self.rho.clamp(0.01, 10.0);
        self.omega = self.omega.clamp(0.01, 10.0);
        self.output = self.psi * self.rho * self.omega;
    }

    pub fn couple_pair(a: &mut GabrielCell, b: &mut GabrielCell) {
        let pa: *mut GabrielCell = a;
        let pb: *mut GabrielCell = b;
        if !a.neighbors.iter().any(|&ptr| ptr == pb) {
            a.neighbors.push(pb);
        }
        if !b.neighbors.iter().any(|&ptr| ptr == pa) {
            b.neighbors.push(pa);
        }
    }

    pub fn neighbor_feedback(&mut self) {
        if self.neighbors.is_empty() {
            return;
        }
        let mut sum = 0.0;
        for &neighbor in &self.neighbors {
            unsafe {
                sum += (*neighbor).output;
            }
        }
        let avg = sum / self.neighbors.len() as f64;
        self.feedback(avg);
    }
}

impl Clone for GabrielCell {
    fn clone(&self) -> Self {
        Self {
            psi: self.psi,
            rho: self.rho,
            omega: self.omega,
            learn_rate: self.learn_rate,
            output: self.output,
            neighbors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_feedback() {
        let mut cell = GabrielCell::simple();
        let out1 = cell.activate(Some(0.8));
        cell.feedback(1.5);
        let out2 = cell.activate(None);
        assert!((out2 - out1).abs() > 1e-6);
    }

    #[test]
    fn coupling() {
        let mut c1 = GabrielCell::new(0.4, 1.0, 1.0, 0.12);
        let mut c2 = GabrielCell::new(0.9, 1.0, 1.0, 0.12);
        GabrielCell::couple_pair(&mut c1, &mut c2);
        c1.activate(Some(0.5));
        c2.activate(Some(1.2));
        c1.neighbor_feedback();
        c2.neighbor_feedback();
        assert!(c1.psi != 0.4 || c1.rho != 1.0);
        assert!(c2.psi != 0.9 || c2.rho != 1.0);
    }
}
