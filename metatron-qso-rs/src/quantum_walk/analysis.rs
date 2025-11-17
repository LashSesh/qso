use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::graph::metatron::{AdjacencyMatrix, MetatronGraph};
use crate::hamiltonian::MetatronHamiltonian;
use crate::qso::QuantumStateOperator;
use crate::quantum::state::{METATRON_DIMENSION, QuantumState};

use super::continuous::{ContinuousTimeQuantumWalk, SpectralPropagator};

/// Time-series diagnostics for the mixing behaviour of a quantum walk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixingTimeResult {
    pub epsilon: f64,
    pub stationary_distribution: [f64; METATRON_DIMENSION],
    pub times: Vec<f64>,
    pub total_variation: Vec<f64>,
    pub mixing_time: Option<f64>,
    pub mixing_time_convergence: bool,
}

/// First-passage analytics for a given start/target configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumHittingResult {
    pub start: usize,
    pub target: usize,
    pub expected_time: f64,
    pub expected_steps: f64,
    pub success_probability: f64,
    pub first_passage_distribution: Vec<f64>,
}

/// Dense matrix of classical hitting times (expected steps).
pub type ClassicalHittingMatrix = [[f64; METATRON_DIMENSION]; METATRON_DIMENSION];

/// Aggregate benchmark results across all start/target pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HittingTimeBenchmark {
    pub dt: f64,
    pub steps: usize,
    pub quantum_average_time: f64,
    pub classical_average_steps: f64,
    pub quantum_average_steps: f64,
    pub speedup_factor: f64,
    pub mean_success_probability: f64,
    pub classical_matrix: ClassicalHittingMatrix,
    pub quantum_results: Vec<QuantumHittingResult>,
}

/// High-level driver connecting the QSO stack with benchmarking utilities.
pub struct QuantumWalkBenchmarker<'a> {
    qso: &'a QuantumStateOperator,
    ctqw: ContinuousTimeQuantumWalk<'a>,
    adjacency: AdjacencyMatrix,
}

impl<'a> QuantumWalkBenchmarker<'a> {
    pub fn new(qso: &'a QuantumStateOperator) -> Self {
        let adjacency = qso.graph().adjacency_matrix();
        let dephasing_rate = qso.parameters().dephasing_rate;
        let ctqw = if dephasing_rate > 0.0 {
            ContinuousTimeQuantumWalk::with_dephasing(qso.hamiltonian(), dephasing_rate)
        } else {
            ContinuousTimeQuantumWalk::new(qso.hamiltonian())
        };
        Self {
            qso,
            ctqw,
            adjacency,
        }
    }

    pub fn hamiltonian(&self) -> &'a MetatronHamiltonian {
        self.qso.hamiltonian()
    }

    pub fn graph(&self) -> &'a MetatronGraph {
        self.qso.graph()
    }

    /// Compute mixing time of a continuous-time quantum walk for an initial state.
    pub fn mixing_time(
        &self,
        initial: &QuantumState,
        times: &[f64],
        epsilon: f64,
    ) -> MixingTimeResult {
        let propagator = self.ctqw.propagator(initial);
        let stationary = propagator.time_average_distribution();

        let mut total_variation = Vec::with_capacity(times.len());
        let mut mixing_index = None;

        for (idx, &time) in times.iter().enumerate() {
            let probs = propagator.probabilities_at(time);
            let distance = total_variation_distance(&probs, &stationary);
            if mixing_index.is_none() && distance <= epsilon {
                mixing_index = Some(idx);
            }
            total_variation.push(distance);
        }

        let mixing_time = mixing_index.map(|idx| times[idx]);
        let mixing_time_convergence = mixing_time.is_some();

        MixingTimeResult {
            epsilon,
            stationary_distribution: stationary,
            times: times.to_vec(),
            total_variation,
            mixing_time,
            mixing_time_convergence,
        }
    }

    /// Evaluate quantum and classical hitting time statistics across all node pairs.
    #[allow(clippy::needless_range_loop)]
    pub fn hitting_time_benchmark(&self, dt: f64, steps: usize) -> HittingTimeBenchmark {
        let classical_matrix = classical_hitting_times(&self.adjacency);
        let mut quantum_results = Vec::new();

        let mut quantum_time_sum = 0.0;
        let mut quantum_steps_sum = 0.0;
        let mut classical_steps_sum = 0.0;
        let mut success_sum = 0.0;
        let mut count = 0.0;

        for start in 0..METATRON_DIMENSION {
            let initial = self.qso.basis_state(start);
            let propagator = self.ctqw.propagator(&initial);
            for target in 0..METATRON_DIMENSION {
                if start == target {
                    continue;
                }
                let classical_steps = classical_matrix[start][target];
                let quantum = quantum_hitting(start, &propagator, target, dt, steps);

                quantum_time_sum += quantum.expected_time;
                quantum_steps_sum += quantum.expected_steps;
                classical_steps_sum += classical_steps;
                success_sum += quantum.success_probability;
                count += 1.0;

                quantum_results.push(quantum);
            }
        }

        let quantum_average_time = if count > 0.0 {
            quantum_time_sum / count
        } else {
            0.0
        };
        let quantum_average_steps = if count > 0.0 {
            quantum_steps_sum / count
        } else {
            0.0
        };
        let classical_average_steps = if count > 0.0 {
            classical_steps_sum / count
        } else {
            0.0
        };
        let mean_success_probability = if count > 0.0 {
            success_sum / count
        } else {
            0.0
        };

        let speedup_factor = if quantum_average_steps > 0.0 {
            classical_average_steps / quantum_average_steps
        } else {
            f64::INFINITY
        };

        HittingTimeBenchmark {
            dt,
            steps,
            quantum_average_time,
            classical_average_steps,
            quantum_average_steps,
            speedup_factor,
            mean_success_probability,
            classical_matrix,
            quantum_results,
        }
    }

    /// Build a consolidated benchmarking suite including metadata for CI comparisons.
    pub fn benchmark_suite(
        &self,
        initial: &QuantumState,
        mixing_dt: f64,
        mixing_samples: usize,
        epsilon: f64,
        hitting_dt: f64,
        hitting_steps: usize,
    ) -> QuantumWalkBenchmarkSuite {
        let times: Vec<f64> = (0..mixing_samples).map(|k| k as f64 * mixing_dt).collect();
        let mixing_time = self.mixing_time(initial, &times, epsilon);
        let hitting_time = self.hitting_time_benchmark(hitting_dt, hitting_steps);

        QuantumWalkBenchmarkSuite {
            metadata: BenchmarkMetadata {
                epsilon,
                hitting_dt,
                hitting_steps,
                mixing_dt,
                mixing_samples,
                graph_nodes: METATRON_DIMENSION,
                dephasing_rate: self.qso.parameters().dephasing_rate,
            },
            mixing_time,
            hitting_time,
        }
    }
}

fn quantum_hitting(
    start: usize,
    propagator: &SpectralPropagator,
    target: usize,
    dt: f64,
    steps: usize,
) -> QuantumHittingResult {
    let mut survival = 1.0;
    let mut expected_time = 0.0;
    let mut expected_steps = 0.0;
    let mut first_passage = Vec::with_capacity(steps);

    for step in 0..steps {
        let time = (step + 1) as f64 * dt;
        let probs = propagator.probabilities_at(time);
        let p_hit = probs[target].clamp(0.0, 1.0);
        let first_prob = survival * p_hit;
        expected_time += first_prob * time;
        expected_steps += first_prob * (step as f64 + 1.0);
        survival *= 1.0 - p_hit;
        first_passage.push(first_prob);
    }

    QuantumHittingResult {
        start,
        target,
        expected_time,
        expected_steps,
        success_probability: 1.0 - survival,
        first_passage_distribution: first_passage,
    }
}

fn total_variation_distance(a: &[f64; METATRON_DIMENSION], b: &[f64; METATRON_DIMENSION]) -> f64 {
    0.5 * a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
}

/// Static metadata captured alongside benchmark suites for deterministic CI comparisons.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkMetadata {
    pub epsilon: f64,
    pub hitting_dt: f64,
    pub hitting_steps: usize,
    pub mixing_dt: f64,
    pub mixing_samples: usize,
    pub graph_nodes: usize,
    pub dephasing_rate: f64,
}

/// Aggregated benchmark outputs used for CI artifact generation and regression checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumWalkBenchmarkSuite {
    pub metadata: BenchmarkMetadata,
    pub mixing_time: MixingTimeResult,
    pub hitting_time: HittingTimeBenchmark,
}

#[allow(clippy::needless_range_loop)]
fn classical_hitting_times(adjacency: &AdjacencyMatrix) -> ClassicalHittingMatrix {
    let mut matrix = [[0.0; METATRON_DIMENSION]; METATRON_DIMENSION];
    let degrees: Vec<f64> = (0..METATRON_DIMENSION)
        .map(|i| adjacency.row(i).iter().copied().sum::<f64>())
        .collect();

    let mut transition = vec![vec![0.0; METATRON_DIMENSION]; METATRON_DIMENSION];
    for i in 0..METATRON_DIMENSION {
        if degrees[i] == 0.0 {
            continue;
        }
        for j in 0..METATRON_DIMENSION {
            transition[i][j] = adjacency[(i, j)] / degrees[i];
        }
    }

    for target in 0..METATRON_DIMENSION {
        let mut transient_indices = Vec::new();
        for idx in 0..METATRON_DIMENSION {
            if idx != target {
                transient_indices.push(idx);
            }
        }

        let dim = transient_indices.len();
        if dim == 0 {
            continue;
        }

        let mut system = DMatrix::<f64>::identity(dim, dim);
        let rhs = DVector::<f64>::from_element(dim, 1.0);

        for (row_pos, &i) in transient_indices.iter().enumerate() {
            for (col_pos, &j) in transient_indices.iter().enumerate() {
                system[(row_pos, col_pos)] -= transition[i][j];
            }
        }

        if let Some(solution) = system.lu().solve(&rhs) {
            for (row_pos, &i) in transient_indices.iter().enumerate() {
                matrix[i][target] = solution[row_pos];
            }
        }

        matrix[target][target] = 0.0;
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::QSOParameters;

    #[test]
    fn mixing_time_decays() {
        let params = QSOParameters::default();
        let qso = QuantumStateOperator::new(params);
        let benchmarker = QuantumWalkBenchmarker::new(&qso);
        let initial = qso.basis_state(0);
        let times: Vec<f64> = (0..20).map(|k| k as f64 * 0.5).collect();
        let result = benchmarker.mixing_time(&initial, &times, 0.05);
        assert_eq!(result.times.len(), result.total_variation.len());
        assert!(result.stationary_distribution.iter().all(|p| *p >= 0.0));
    }

    #[test]
    fn hitting_time_benchmark_runs() {
        let params = QSOParameters::default();
        let qso = QuantumStateOperator::new(params);
        let benchmarker = QuantumWalkBenchmarker::new(&qso);
        let report = benchmarker.hitting_time_benchmark(0.25, 12);
        assert!(!report.quantum_results.is_empty());
        assert!(report.speedup_factor.is_finite());
    }
}
