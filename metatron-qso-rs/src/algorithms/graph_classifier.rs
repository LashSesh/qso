//! Graph classifier with QAOA-type quantum layers.
//!
//! Every input graph `G` (`n` nodes = `n` qubits, edge weights `w`) defines
//! its own cut operator `C_G = Σ_{(u,v)} w_uv (1 − Z_u Z_v)/2`. The model
//! applies `layers` trainable blocks `e^{−iβ_l Σ X} e^{−iγ_l C_G}` to `|+⟩ⁿ`
//! — the same parameters for every graph — and reads out the normalised
//! expected cut
//!
//! ```text
//! f(G) = ⟨C_G⟩ / W(G),        W(G) = Σ w_uv,
//! p(y = 1 | G) = σ(a·(f(G) − ½) + b).
//! ```
//!
//! The feature is local: after `p` layers the contribution of an edge
//! depends only on the subgraph within distance `p` of it (the QAOA light
//! cone), so depth decides which cycles the model can see. With one layer
//! and unit weights, `f` depends only on degrees and triangles (Wang et al.,
//! Phys. Rev. A 97, 022304, 2018).
//!
//! Training minimises the mean binary cross-entropy (plus a small L2 penalty
//! on `a` and `b`, which keeps the readout finite on separable data) with
//! exact gradients from the adjoint QAOA pass.

use serde::{Deserialize, Serialize};

use super::classifier::TrainingReport;
use super::qaoa::MaxCutQaoa;
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::optimize::{Method, Objective, Options, minimize};
use crate::parallel::map_tasks;
use crate::rng::Rng;

/// L2 penalty on the readout weights `a` and `b`.
const READOUT_L2: f64 = 1e-4;

/// Quantum graph classifier (see the module documentation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphClassifier {
    /// Number of quantum layers `p`.
    pub layers: usize,
    /// `[γ₁ … γ_p, β₁ … β_p, a, b]` (empty before training).
    pub parameters: Vec<f64>,
}

impl GraphClassifier {
    pub fn new(layers: usize) -> Result<Self> {
        if layers == 0 {
            return Err(QsoError::invalid("layers", "need at least one layer"));
        }
        Ok(Self {
            layers,
            parameters: Vec::new(),
        })
    }

    /// `2·layers + 2`.
    pub fn num_parameters(&self) -> usize {
        2 * self.layers + 2
    }

    fn trained(&self) -> Result<&[f64]> {
        if self.parameters.len() != self.num_parameters() {
            return Err(QsoError::invalid(
                "model",
                "classifier has not been trained",
            ));
        }
        Ok(&self.parameters)
    }

    /// Feature `f(G) = ⟨C_G⟩ / W(G)` for the angles `[γ…, β…]`.
    pub fn feature_with(&self, graph: &Graph, angles: &[f64]) -> Result<f64> {
        if angles.len() != 2 * self.layers {
            return Err(QsoError::ParameterCount {
                expected: 2 * self.layers,
                actual: angles.len(),
            });
        }
        let instance = Instance::new(graph, self.layers)?;
        Ok(instance.qaoa.expectation(angles)? / instance.weight)
    }

    /// Feature `f(G)` of the trained model.
    pub fn feature(&self, graph: &Graph) -> Result<f64> {
        let p = self.trained()?;
        self.feature_with(graph, &p[..2 * self.layers])
    }

    /// `p(y = 1 | G)` of the trained model.
    pub fn predict_proba(&self, graph: &Graph) -> Result<f64> {
        let p = self.trained()?;
        let (a, b) = (p[2 * self.layers], p[2 * self.layers + 1]);
        Ok(sigmoid(a * (self.feature(graph)? - 0.5) + b))
    }

    /// Class prediction (`p ≥ ½`).
    pub fn predict(&self, graph: &Graph) -> Result<bool> {
        Ok(self.predict_proba(graph)? >= 0.5)
    }

    /// Fraction of correctly classified graphs.
    pub fn accuracy(&self, graphs: &[Graph], labels: &[bool]) -> Result<f64> {
        check_data(graphs, labels)?;
        let predictions = map_tasks(graphs.len(), |k| self.predict(&graphs[k]));
        let mut correct = 0usize;
        for (p, &y) in predictions.into_iter().zip(labels) {
            if p? == y {
                correct += 1;
            }
        }
        Ok(correct as f64 / graphs.len() as f64)
    }

    /// Train with L-BFGS from `restarts` seeded random starts (angles in
    /// `[0, π/2)`, readout at zero) and keep the lowest loss.
    pub fn fit(
        &mut self,
        graphs: &[Graph],
        labels: &[bool],
        options: &Options,
        restarts: usize,
        seed: u64,
    ) -> Result<TrainingReport> {
        check_data(graphs, labels)?;
        let restarts = restarts.max(1);
        let instances = map_tasks(graphs.len(), |k| Instance::new(&graphs[k], self.layers))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let loss = Loss {
            layers: self.layers,
            instances: &instances,
            labels,
        };
        let base = Rng::seed_from_u64(seed);
        let mut best: Option<(f64, Vec<f64>, usize, usize)> = None;
        for r in 0..restarts {
            let mut rng = base.fork(r as u64);
            let mut x0: Vec<f64> = (0..2 * self.layers)
                .map(|_| rng.uniform_range(0.0, std::f64::consts::FRAC_PI_2))
                .collect();
            x0.extend([0.0, 0.0]);
            let res = minimize(&loss, &x0, &Method::lbfgs(), options)?;
            if best.as_ref().is_none_or(|b| res.value < b.0) {
                best = Some((res.value, res.x, res.iterations, res.evaluations));
            }
        }
        let (value, x, iterations, evaluations) = best.expect("at least one restart");
        self.parameters = x;
        Ok(TrainingReport {
            loss: value,
            accuracy: self.accuracy(graphs, labels)?,
            iterations,
            evaluations,
        })
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn check_data(graphs: &[Graph], labels: &[bool]) -> Result<()> {
    if graphs.is_empty() || graphs.len() != labels.len() {
        return Err(QsoError::invalid(
            "data",
            "need equally many (non-zero) graphs and labels",
        ));
    }
    Ok(())
}

/// Precomputed QAOA instance of one graph.
struct Instance {
    qaoa: MaxCutQaoa,
    weight: f64,
}

impl Instance {
    fn new(graph: &Graph, layers: usize) -> Result<Self> {
        let weight: f64 = graph.edges().iter().map(|&(_, _, w)| w).sum();
        if weight <= 0.0 {
            return Err(QsoError::invalid(
                "graph",
                "every graph needs a positive total edge weight",
            ));
        }
        Ok(Self {
            qaoa: MaxCutQaoa::new(graph, layers)?,
            weight,
        })
    }
}

struct Loss<'a> {
    layers: usize,
    instances: &'a [Instance],
    labels: &'a [bool],
}

impl Objective for Loss<'_> {
    fn dim(&self) -> usize {
        2 * self.layers + 2
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.value_and_gradient(x)
            .map(|(v, _)| v)
            .unwrap_or(f64::INFINITY)
    }

    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        let k = 2 * self.layers;
        let (angles, a, b) = (&x[..k], x[k], x[k + 1]);
        let per_graph = map_tasks(self.instances.len(), |i| {
            let inst = &self.instances[i];
            inst.qaoa
                .expectation_and_gradient(angles)
                .map(|(c, dc)| (c / inst.weight, dc, inst.weight))
        });
        let m = self.instances.len() as f64;
        let mut loss = 0.5 * READOUT_L2 * (a * a + b * b);
        let mut grad = vec![0.0; k + 2];
        grad[k] = READOUT_L2 * a;
        grad[k + 1] = READOUT_L2 * b;
        for (result, &y) in per_graph.into_iter().zip(self.labels) {
            let (f, df, weight) = result.ok()?;
            let z = a * (f - 0.5) + b;
            let target = if y { 1.0 } else { 0.0 };
            // BCE with logits: softplus(z) − y·z, derivative σ(z) − y.
            loss += (softplus(z) - target * z) / m;
            let dl_dz = (sigmoid(z) - target) / m;
            for (g, d) in grad[..k].iter_mut().zip(&df) {
                *g += dl_dz * a * d / weight;
            }
            grad[k] += dl_dz * (f - 0.5);
            grad[k + 1] += dl_dz;
        }
        Some((loss, grad))
    }
}

/// `ln(1 + eᶻ)` without overflow.
fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;

    /// One-layer expected cut of edge (u, v) (Wang, Hadfield, Jiang &
    /// Rieffel 2018, Theorem 1): degrees d_u, d_v and λ triangles on the edge.
    fn wang_edge(du: usize, dv: usize, lambda: usize, gamma: f64, beta: f64) -> f64 {
        let (c, c2) = (gamma.cos(), (2.0 * gamma).cos());
        0.5 + 0.25
            * (4.0 * beta).sin()
            * gamma.sin()
            * (c.powi(du as i32 - 1) + c.powi(dv as i32 - 1))
            - 0.25
                * (2.0 * beta).sin().powi(2)
                * c.powi((du + dv) as i32 - 2 - 2 * lambda as i32)
                * (1.0 - c2.powi(lambda as i32))
    }

    fn wang_feature(g: &Graph, gamma: f64, beta: f64) -> f64 {
        let total: f64 = g
            .edges()
            .iter()
            .map(|&(u, v, _)| {
                let lambda = g
                    .neighbors(u)
                    .iter()
                    .filter(|&&(w, _)| g.has_edge(w, v))
                    .count();
                wang_edge(g.degree(u), g.degree(v), lambda, gamma, beta)
            })
            .sum();
        total / g.num_edges() as f64
    }

    #[test]
    fn one_layer_feature_matches_the_closed_form() {
        let model = GraphClassifier::new(1).unwrap();
        let mut rng = Rng::seed_from_u64(1);
        let graphs = [
            generators::petersen(),
            generators::complete(5),
            generators::cycle(7),
            generators::star(6),
            generators::random_regular(10, 3, &mut rng).unwrap(),
            generators::erdos_renyi(9, 0.4, &mut rng).unwrap(),
        ];
        for g in &graphs {
            for _ in 0..3 {
                let (gamma, beta) = (rng.uniform_range(-2.0, 2.0), rng.uniform_range(-2.0, 2.0));
                let f = model.feature_with(g, &[gamma, beta]).unwrap();
                let w = wang_feature(g, gamma, beta);
                assert!((f - w).abs() < 1e-12, "{}: {f} vs {w}", g.name());
            }
        }
    }

    /// Balanced set of random 3-regular graphs labelled by `label`.
    fn dataset(
        n: usize,
        per_class: usize,
        seed: u64,
        label: impl Fn(&Graph) -> Option<bool>,
    ) -> (Vec<Graph>, Vec<bool>) {
        let mut rng = Rng::seed_from_u64(seed);
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        while pos.len() < per_class || neg.len() < per_class {
            let g = generators::random_regular(n, 3, &mut rng).unwrap();
            match label(&g) {
                Some(true) if pos.len() < per_class => pos.push(g),
                Some(false) if neg.len() < per_class => neg.push(g),
                _ => {}
            }
        }
        let labels = [vec![true; per_class], vec![false; per_class]].concat();
        ([pos, neg].concat(), labels)
    }

    #[test]
    fn one_layer_learns_to_detect_triangles() {
        let has_triangle = |g: &Graph| Some(g.triangle_count() > 0);
        let (train, y_train) = dataset(10, 12, 2, has_triangle);
        let (test, y_test) = dataset(10, 12, 3, has_triangle);
        let mut model = GraphClassifier::new(1).unwrap();
        let options = Options {
            max_iterations: 300,
            ..Default::default()
        };
        let report = model.fit(&train, &y_train, &options, 4, 0).unwrap();
        assert_eq!(report.accuracy, 1.0, "{report:?}");
        assert_eq!(model.accuracy(&test, &y_test).unwrap(), 1.0);
    }

    #[test]
    fn light_cone() {
        // One layer sees the radius-1 neighbourhood of an edge, which is
        // the same tree for every triangle-free cubic graph: the feature is
        // constant across them. Two layers also see 4- and 5-cycles.
        let label = |g: &Graph| match g.girth() {
            Some(4) => Some(true),
            Some(g) if g >= 5 => Some(false),
            _ => None,
        };
        let (graphs, _) = dataset(14, 4, 4, label);
        let one = GraphClassifier::new(1).unwrap();
        let f: Vec<f64> = graphs
            .iter()
            .map(|g| one.feature_with(g, &[0.6, 0.4]).unwrap())
            .collect();
        assert!(f.iter().all(|x| (x - f[0]).abs() < 1e-12));
        let two = GraphClassifier::new(2).unwrap();
        let f2: Vec<f64> = graphs
            .iter()
            .map(|g| two.feature_with(g, &[0.5, 0.9, 0.6, 0.3]).unwrap())
            .collect();
        assert!(f2.iter().any(|x| (x - f2[0]).abs() > 1e-6));
    }

    #[test]
    fn validation() {
        assert!(GraphClassifier::new(0).is_err());
        let model = GraphClassifier::new(1).unwrap();
        assert!(model.predict(&generators::cycle(4)).is_err());
        assert!(model.feature_with(&generators::cycle(4), &[0.1]).is_err());
        assert!(model.feature_with(&Graph::new(3), &[0.1, 0.2]).is_err());
        let mut m = GraphClassifier::new(1).unwrap();
        assert!(
            m.fit(&[generators::cycle(4)], &[], &Options::default(), 1, 0)
                .is_err()
        );
    }
}
