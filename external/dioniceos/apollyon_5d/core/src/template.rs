//! Domain-Specific Templates
//!
//! Section 7 - pre-configured system specifications for specific domains.

use crate::coupling::{CouplingMatrix, CouplingType};
use crate::dynamics::{SystemParameters, VectorField};
use serde::{Deserialize, Serialize};

/// Template for domain-specific system instantiation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub name: String,
    pub description: String,
    pub variable_names: [String; 5],
    pub coupling_matrix: CouplingMatrix,
    pub parameters: SystemParameters,
}

impl Template {
    /// Create the SIR epidemiological model template (Section 7.2.1)
    ///
    /// Variables: (S, I, R, E, D)
    /// - S: Susceptible population
    /// - I: Infected population
    /// - R: Recovered population
    /// - E: Exposed population
    /// - D: Deceased population
    pub fn sir_model(beta: f64, gamma: f64, mu: f64) -> Self {
        let mut coupling = CouplingMatrix::zero();

        // S-I interaction: -β·S·I (transmission)
        coupling.set(0, 1, -beta, CouplingType::Product);

        // I dynamics: β·S·I - γ·I - μ·I (infection - recovery - death)
        coupling.set(1, 1, -gamma - mu, CouplingType::Linear);
        coupling.set(1, 0, beta, CouplingType::Product);

        // R dynamics: γ·I (recovery)
        coupling.set(2, 1, gamma, CouplingType::Linear);

        // D dynamics: μ·I (death)
        coupling.set(4, 1, mu, CouplingType::Linear);

        Template {
            name: "SIR Epidemiological Model".to_string(),
            description: "Extended SIR model with exposed and deceased compartments".to_string(),
            variable_names: [
                "Susceptible".to_string(),
                "Infected".to_string(),
                "Recovered".to_string(),
                "Exposed".to_string(),
                "Deceased".to_string(),
            ],
            coupling_matrix: coupling,
            parameters: SystemParameters::zero(),
        }
    }

    /// Create the financial market model template (Section 7.2.2)
    ///
    /// Variables: (P, V, M, L, R)
    /// - P: Price
    /// - V: Volume
    /// - M: Market sentiment
    /// - L: Liquidity
    /// - R: Risk factor
    pub fn financial_market(volatility: f64, momentum: f64, risk_aversion: f64) -> Self {
        let mut coupling = CouplingMatrix::zero();

        // Price dynamics: influenced by volume and sentiment
        coupling.set(0, 1, momentum, CouplingType::Linear);
        coupling.set(0, 2, volatility, CouplingType::Linear);

        // Volume dynamics: influenced by price changes
        coupling.set(1, 0, 0.5, CouplingType::Linear);

        // Sentiment dynamics: nonlinear (sigmoid)
        coupling.set(2, 0, 0.3, CouplingType::Sigmoid);
        coupling.set(2, 4, -risk_aversion, CouplingType::Linear);

        // Liquidity dynamics
        coupling.set(3, 1, 0.2, CouplingType::Linear);

        // Risk dynamics: quadratic in price
        coupling.set(4, 0, 0.1, CouplingType::Quadratic);

        let mut params = SystemParameters::zero();
        params.intrinsic_rates = [-0.1, -0.05, -0.2, 0.0, -0.15];

        Template {
            name: "Financial Market Model".to_string(),
            description:
                "5-variable market dynamics with price, volume, sentiment, liquidity, and risk"
                    .to_string(),
            variable_names: [
                "Price".to_string(),
                "Volume".to_string(),
                "Sentiment".to_string(),
                "Liquidity".to_string(),
                "Risk".to_string(),
            ],
            coupling_matrix: coupling,
            parameters: params,
        }
    }

    /// Create the predator-prey ecosystem template (Section 7.2.3)
    ///
    /// Variables: (X₁, X₂, X₃, R, C)
    /// - X₁: Prey population 1
    /// - X₂: Prey population 2
    /// - X₃: Predator population
    /// - R: Resource availability
    /// - C: Environmental capacity
    pub fn predator_prey(growth_rate: f64, predation_rate: f64, death_rate: f64) -> Self {
        let mut coupling = CouplingMatrix::zero();

        // Prey 1 dynamics: growth - predation
        coupling.set(0, 0, growth_rate, CouplingType::Linear);
        coupling.set(0, 2, -predation_rate, CouplingType::Product);
        coupling.set(0, 3, 0.3, CouplingType::Linear); // Resource influence

        // Prey 2 dynamics: similar to prey 1
        coupling.set(1, 1, growth_rate * 0.8, CouplingType::Linear);
        coupling.set(1, 2, -predation_rate * 0.7, CouplingType::Product);
        coupling.set(1, 3, 0.25, CouplingType::Linear);

        // Predator dynamics: gains from predation, natural death
        coupling.set(2, 0, predation_rate * 0.4, CouplingType::Product);
        coupling.set(2, 1, predation_rate * 0.3, CouplingType::Product);
        coupling.set(2, 2, -death_rate, CouplingType::Linear);

        // Resource dynamics: regeneration and consumption
        coupling.set(3, 0, -0.1, CouplingType::Linear);
        coupling.set(3, 1, -0.1, CouplingType::Linear);
        coupling.set(3, 3, 0.5, CouplingType::Linear); // Regeneration

        // Capacity: slowly changing
        coupling.set(4, 4, -0.01, CouplingType::Linear);

        Template {
            name: "Predator-Prey Ecosystem".to_string(),
            description: "Multi-species ecosystem with resource dynamics".to_string(),
            variable_names: [
                "Prey1".to_string(),
                "Prey2".to_string(),
                "Predator".to_string(),
                "Resources".to_string(),
                "Capacity".to_string(),
            ],
            coupling_matrix: coupling,
            parameters: SystemParameters::zero(),
        }
    }

    /// Create a vector field from this template
    pub fn to_vector_field(&self) -> VectorField {
        VectorField::new(self.coupling_matrix.clone(), self.parameters.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sir_template() {
        let template = Template::sir_model(0.3, 0.1, 0.01);
        assert_eq!(template.name, "SIR Epidemiological Model");
        assert_eq!(template.variable_names[0], "Susceptible");
    }

    #[test]
    fn test_financial_template() {
        let template = Template::financial_market(0.2, 0.1, 0.05);
        assert_eq!(template.name, "Financial Market Model");
        assert_eq!(template.variable_names[0], "Price");
    }

    #[test]
    fn test_predator_prey_template() {
        let template = Template::predator_prey(0.5, 0.3, 0.1);
        assert_eq!(template.name, "Predator-Prey Ecosystem");
        assert_eq!(template.variable_names[2], "Predator");
    }
}
