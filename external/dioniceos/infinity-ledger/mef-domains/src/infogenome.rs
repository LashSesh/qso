/*!
 * Infogene and Infogenome - Operator signatures and transformation behavior
 *
 * Infogenes define operator configurations with governance rules.
 * Infogenomes are collections of Infogenes defining complete transformation behavior.
 */

use mef_topology::metatron_router::OperatorType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Operator signature with governance rules for domain transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Infogene {
    /// Operator type (DK, SW, PI, WT)
    pub operator: OperatorType,
    /// Operator-specific parameters
    pub params: HashMap<String, f64>,
    /// Constraints on operator application
    pub constraints: Vec<String>,
    /// Weight for operator blending
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_weight() -> f64 {
    1.0
}

impl Infogene {
    /// Create a new Infogene
    pub fn new(
        operator: OperatorType,
        params: HashMap<String, f64>,
        constraints: Vec<String>,
        weight: f64,
    ) -> Self {
        Self {
            operator,
            params,
            constraints,
            weight,
        }
    }

    /// Create DoubleKick operator gene
    pub fn double_kick(alpha1: f64, alpha2: f64, weight: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("alpha1".to_string(), alpha1);
        params.insert("alpha2".to_string(), alpha2);

        Self::new(
            OperatorType::DK,
            params,
            vec!["non_expansive".to_string()],
            weight,
        )
    }

    /// Create Sweep operator gene
    pub fn sweep(tau: f64, beta: f64, weight: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("tau".to_string(), tau);
        params.insert("beta".to_string(), beta);

        Self::new(
            OperatorType::SW,
            params,
            vec!["adaptive".to_string()],
            weight,
        )
    }

    /// Create PathInvariance operator gene
    pub fn path_invariance(tolerance: f64, weight: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("tolerance".to_string(), tolerance);

        Self::new(
            OperatorType::PI,
            params,
            vec!["idempotent".to_string()],
            weight,
        )
    }

    /// Create WeightTransfer operator gene
    pub fn weight_transfer(gamma: f64, weight: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), gamma);

        Self::new(
            OperatorType::WT,
            params,
            vec!["conservative".to_string()],
            weight,
        )
    }
}

/// Collection of Infogenes defining transformation behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Infogenome {
    /// Unique identifier
    pub id: String,
    /// Constituent genes
    pub genes: Vec<Infogene>,
    /// Governance rules and constraints
    pub governance: HashMap<String, Vec<String>>,
    /// Fitness score [0, 1]
    #[serde(default)]
    pub fitness: f64,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Infogenome {
    /// Create a new Infogenome
    pub fn new(genes: Vec<Infogene>, governance: HashMap<String, Vec<String>>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            genes,
            governance,
            fitness: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a base Infogenome with standard operators
    pub fn base() -> Self {
        let genes = vec![
            Infogene::double_kick(0.05, -0.03, 1.0),
            Infogene::sweep(0.5, 0.1, 0.8),
            Infogene::path_invariance(1e-6, 0.9),
            Infogene::weight_transfer(0.1, 0.7),
        ];

        let mut governance = HashMap::new();
        governance.insert(
            "rules".to_string(),
            vec![
                "pfadinvarianz".to_string(),
                "resonanzvalidierung".to_string(),
            ],
        );
        governance.insert(
            "constraints".to_string(),
            vec!["contraction".to_string(), "convergence".to_string()],
        );

        Self::new(genes, governance)
    }

    /// Create mutated version of this Infogenome
    ///
    /// # Arguments
    ///
    /// * `mutation_rate` - Probability of mutating each gene (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// New mutated Infogenome
    pub fn mutate(&self, mutation_rate: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let new_genes = self
            .genes
            .iter()
            .map(|gene| {
                if rng.gen::<f64>() < mutation_rate {
                    // Mutate parameters
                    let mut new_params = gene.params.clone();
                    for (_key, value) in new_params.iter_mut() {
                        let noise: f64 = rng.gen_range(-0.1..0.1);
                        *value *= 1.0 + noise;
                    }

                    // Mutate weight
                    let noise: f64 = rng.gen_range(-0.05..0.05);
                    let new_weight = (gene.weight * (1.0 + noise)).clamp(0.0, 1.0);

                    Infogene::new(
                        gene.operator,
                        new_params,
                        gene.constraints.clone(),
                        new_weight,
                    )
                } else {
                    gene.clone()
                }
            })
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert(
            "parent".to_string(),
            serde_json::Value::String(self.id.clone()),
        );

        Self {
            id: Uuid::new_v4().to_string(),
            genes: new_genes,
            governance: self.governance.clone(),
            fitness: 0.0,
            metadata,
        }
    }

    /// Update fitness score
    pub fn update_fitness(&mut self, new_fitness: f64) {
        // Exponential moving average
        self.fitness = 0.9 * self.fitness + 0.1 * new_fitness.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infogene_creation() {
        let gene = Infogene::double_kick(0.05, -0.03, 1.0);

        assert_eq!(gene.operator, OperatorType::DK);
        assert_eq!(gene.weight, 1.0);
        assert_eq!(*gene.params.get("alpha1").unwrap(), 0.05);
        assert_eq!(*gene.params.get("alpha2").unwrap(), -0.03);
    }

    #[test]
    fn test_infogene_sweep() {
        let gene = Infogene::sweep(0.5, 0.1, 0.8);

        assert_eq!(gene.operator, OperatorType::SW);
        assert_eq!(gene.weight, 0.8);
    }

    #[test]
    fn test_infogenome_base() {
        let genome = Infogenome::base();

        assert_eq!(genome.genes.len(), 4);
        assert!(genome.governance.contains_key("rules"));
        assert!(genome.governance.contains_key("constraints"));
        assert_eq!(genome.fitness, 0.0);
    }

    #[test]
    fn test_infogenome_mutate() {
        let genome = Infogenome::base();
        let mutant = genome.mutate(0.3);

        assert_ne!(genome.id, mutant.id);
        assert_eq!(genome.genes.len(), mutant.genes.len());

        // Check parent metadata
        let parent_id = mutant.metadata.get("parent").unwrap();
        assert_eq!(parent_id.as_str().unwrap(), genome.id);
    }

    #[test]
    fn test_fitness_update() {
        let mut genome = Infogenome::base();

        genome.update_fitness(0.5);
        assert!((genome.fitness - 0.05).abs() < 1e-6); // 0.9*0 + 0.1*0.5

        genome.update_fitness(1.0);
        assert!((genome.fitness - 0.145).abs() < 1e-6); // 0.9*0.05 + 0.1*1.0
    }

    #[test]
    fn test_fitness_clamp() {
        let mut genome = Infogenome::base();

        genome.update_fitness(2.0); // Should clamp to 1.0
        assert!((genome.fitness - 0.1).abs() < 1e-6);

        genome.update_fitness(-1.0); // Should clamp to 0.0
        assert!(genome.fitness >= 0.0);
    }

    #[test]
    fn test_serialization() {
        let genome = Infogenome::base();

        let json = serde_json::to_string(&genome).unwrap();
        let deserialized: Infogenome = serde_json::from_str(&json).unwrap();

        assert_eq!(genome.id, deserialized.id);
        assert_eq!(genome.genes.len(), deserialized.genes.len());
    }
}
