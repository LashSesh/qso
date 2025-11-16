use serde::Serialize;

#[derive(Debug, Clone)]
pub struct TripolarOperator {
    pub psi: f64,
    pub rho: f64,
    pub omega: f64,
}

impl TripolarOperator {
    pub fn new(psi: f64, rho: f64, omega: f64) -> Self {
        Self { psi, rho, omega }
    }

    pub fn value(&self) -> f64 {
        self.psi * self.rho * self.omega
    }
}

#[derive(Debug, Clone)]
pub struct OphanKernel {
    pub operators: Vec<TripolarOperator>,
}

impl OphanKernel {
    pub fn new(operators: Vec<TripolarOperator>) -> Self {
        Self { operators }
    }

    pub fn excalibration_ready(&self, threshold: f64) -> bool {
        self.total_resonance() > threshold
    }

    pub fn total_resonance(&self) -> f64 {
        self.operators.iter().map(|op| op.value()).product()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DecisionState {
    Pending,
    Excalibration,
}

#[derive(Debug, Clone)]
pub struct MonolithDecision {
    pub kernel: OphanKernel,
    threshold: f64,
}

impl MonolithDecision {
    pub fn new(kernel: OphanKernel, threshold: f64) -> Self {
        Self { kernel, threshold }
    }

    pub fn evaluate(&self) -> DecisionState {
        if self.kernel.excalibration_ready(self.threshold) {
            DecisionState::Excalibration
        } else {
            DecisionState::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_pending() {
        let kernel = OphanKernel::new(vec![TripolarOperator::new(0.1, 0.1, 0.1); 4]);
        let decision = MonolithDecision::new(kernel, 1.0);
        assert_eq!(decision.evaluate(), DecisionState::Pending);
    }
}
