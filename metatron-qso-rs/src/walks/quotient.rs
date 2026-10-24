//! Walks reduced by symmetry: quotients by equitable partitions.
//!
//! A partition of the nodes into cells `C₁, …, C_k` is *equitable* if every
//! node of `C_i` has the same total edge weight `B_ij` into `C_j`. The
//! normalised cell states `|C_i⟩ = |C_i|^{−1/2} Σ_{v∈C_i} |v⟩` then span a
//! subspace that the adjacency matrix and the Laplacian leave invariant; on
//! it they act as the symmetric `k × k` matrices
//!
//! ```text
//! A_red[i][j] = √(B_ij·B_ji),        L_red = diag(d_i) − A_red,
//! ```
//!
//! with `d_i` the weighted degree of the nodes of `C_i`. A walker that
//! starts in that subspace (at a node that is a cell of its own, or in the
//! uniform superposition) never leaves it, so its dynamics are exact in
//! `k` dimensions however many nodes the graph has: the hypercube `Q₃₀`
//! with 2³⁰ nodes reduces to 31 cells, Hamming, Johnson and complete
//! multipartite graphs likewise. The probability of a node is the
//! probability of its cell divided by the cell size.
//!
//! * [`Quotient`] — the partition and its quotient matrix: computed for a
//!   graph by colour refinement (the coarsest equitable partition in which
//!   given nodes are singletons), checked for a partition you supply, or
//!   written down directly for the distance partitions of hypercubes,
//!   Hamming, Johnson, complete and complete multipartite graphs, which
//!   are never built.
//! * [`QuotientWalk`] — continuous-time walks: amplitudes, cell
//!   probabilities, limiting and time-averaged distributions.
//! * [`QuotientSearch`] — Childs–Goldstone spatial search for marked cells.
//!
//! ```
//! use metatron_qso::walks::{Generator, Quotient, QuotientWalk};
//!
//! // Q₃₀ (2³⁰ nodes) in 31 dimensions: from 0…0 the walker reaches the
//! // antipode with certainty at γt = π/2 (perfect state transfer).
//! let q = Quotient::hypercube(30).unwrap();
//! let walk = QuotientWalk::new(&q, Generator::Laplacian, 1.0).unwrap();
//! let p = walk.cell_probabilities(0, std::f64::consts::FRAC_PI_2).unwrap();
//! assert!((p[30] - 1.0).abs() < 1e-10);
//! ```

use num_complex::Complex64 as C64;

use super::search::{SearchResult, optimize_peak, optimize_rate};
use super::{Generator, StartSpectrum};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::graph::canonical::refine;
use crate::linalg::{RMatrix, eigh_real};

/// An equitable partition and its quotient matrix (see the module
/// documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Quotient {
    name: String,
    sizes: Vec<f64>,
    /// `B[i][j]`: total weight from any node of cell `i` into cell `j`.
    matrix: RMatrix,
    /// Cell of every node, for partitions of an explicit graph.
    cell_of: Option<Vec<usize>>,
}

/// `C(n, k)` in floating point (exact up to 2⁵³, then correctly rounded to
/// about `k` ulps).
fn binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

impl Quotient {
    /// Quotient with cell sizes `sizes` and matrix `matrix[i][j]` = weight
    /// from any node of cell `i` into cell `j`.
    ///
    /// Counting the edges between two cells from either side gives
    /// `sizes[i]·B[i][j] = sizes[j]·B[j][i]`; this is checked, as are
    /// positive finite sizes, finite entries and the absence of edges
    /// inside single-node cells.
    pub fn from_matrix(sizes: Vec<f64>, matrix: RMatrix) -> Result<Self> {
        let k = sizes.len();
        if k == 0 {
            return Err(QsoError::invalid("sizes", "need at least one cell"));
        }
        if matrix.shape() != (k, k) {
            return Err(QsoError::DimensionMismatch {
                expected: k,
                actual: matrix.nrows(),
            });
        }
        if let Some(s) = sizes.iter().find(|s| !(s.is_finite() && **s >= 1.0)) {
            return Err(QsoError::invalid(
                "sizes",
                format!("cell sizes must be finite and at least 1, got {s}"),
            ));
        }
        if !sizes.iter().sum::<f64>().is_finite() {
            return Err(QsoError::invalid(
                "sizes",
                "total number of nodes overflows",
            ));
        }
        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid("matrix", "entries must be finite"));
        }
        for i in 0..k {
            if sizes[i] == 1.0 && matrix[(i, i)] != 0.0 {
                return Err(QsoError::invalid(
                    "matrix",
                    format!("cell {i} has one node but weight inside it"),
                ));
            }
            for j in i + 1..k {
                let (bij, bji) = (matrix[(i, j)], matrix[(j, i)]);
                // sizes[i]·bij = sizes[j]·bji, compared in logarithms
                // because sizes may be close to the largest float.
                let consistent = if bij == 0.0 || bji == 0.0 {
                    bij == bji
                } else {
                    bij.signum() == bji.signum()
                        && ((sizes[i].ln() + bij.abs().ln()) - (sizes[j].ln() + bji.abs().ln()))
                            .abs()
                            <= 1e-9
                };
                if !consistent {
                    return Err(QsoError::invalid(
                        "matrix",
                        format!(
                            "cells {i} and {j}: sizes[i]·B[i][j] = {} but sizes[j]·B[j][i] = {}",
                            sizes[i] * bij,
                            sizes[j] * bji
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            name: format!("quotient({k} cells)"),
            sizes,
            matrix,
            cell_of: None,
        })
    }

    /// Set a descriptive name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build from a graph and a partition known to be equitable.
    fn of_cells(graph: &Graph, cell_of: Vec<usize>, k: usize) -> Self {
        let mut sizes = vec![0.0; k];
        let mut representative = vec![usize::MAX; k];
        for (v, &c) in cell_of.iter().enumerate() {
            sizes[c] += 1.0;
            if representative[c] == usize::MAX {
                representative[c] = v;
            }
        }
        let mut matrix = RMatrix::zeros(k, k);
        for (i, &r) in representative.iter().enumerate() {
            for &(w, wt) in graph.neighbors(r) {
                matrix[(i, cell_of[w])] += wt;
            }
        }
        Self {
            name: format!("{} / equitable({k} cells)", graph.name()),
            sizes,
            matrix,
            cell_of: Some(cell_of),
        }
    }

    /// Cells numbered `0..k` in order of their smallest member.
    fn renumber(labels: &[usize]) -> (Vec<usize>, usize) {
        let mut index = std::collections::HashMap::new();
        let cells = labels
            .iter()
            .map(|&l| {
                let next = index.len();
                *index.entry(l).or_insert(next)
            })
            .collect();
        (cells, index.len())
    }

    /// Coarsest equitable partition refining the colouring `colours` (one
    /// colour per node), by colour refinement. Nodes of different colours
    /// stay in different cells; with weighted edges, nodes are kept
    /// together only if they have the same multiset of weights into every
    /// cell. Cells are numbered in order of their smallest node.
    pub fn equitable(graph: &Graph, colours: &[usize]) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 {
            return Err(QsoError::invalid("graph", "graph has no nodes"));
        }
        if colours.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: colours.len(),
            });
        }
        let mut distinct = colours.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let mut c: Vec<u32> = colours
            .iter()
            .map(|x| distinct.binary_search(x).expect("present") as u32)
            .collect();
        refine(graph, &mut c);
        let (cells, k) = Self::renumber(&c.iter().map(|&x| x as usize).collect::<Vec<_>>());
        Ok(Self::of_cells(graph, cells, k))
    }

    /// Coarsest equitable partition in which each of `nodes` is a cell of
    /// its own: the natural reduction for a walker started at (or a search
    /// for) those nodes.
    ///
    /// ```
    /// use metatron_qso::graph::generators;
    /// use metatron_qso::walks::Quotient;
    ///
    /// // From any node of the Petersen graph: itself, 3 neighbours, 6 others.
    /// let q = Quotient::with_singletons(&generators::petersen(), &[0]).unwrap();
    /// assert_eq!(q.sizes(), &[1.0, 3.0, 6.0]);
    /// ```
    pub fn with_singletons(graph: &Graph, nodes: &[usize]) -> Result<Self> {
        let n = graph.num_nodes();
        let mut colours = vec![0; n];
        for (k, &v) in nodes.iter().enumerate() {
            if v >= n {
                return Err(QsoError::NodeOutOfRange {
                    node: v,
                    n_nodes: n,
                });
            }
            colours[v] = k + 1;
        }
        Self::equitable(graph, &colours)
    }

    /// The partition `cell_of` (node → cell label) of `graph`, after
    /// checking that it is equitable (to a relative tolerance of 1e-9).
    pub fn from_cells(graph: &Graph, cell_of: &[usize]) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 {
            return Err(QsoError::invalid("graph", "graph has no nodes"));
        }
        if cell_of.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: cell_of.len(),
            });
        }
        let (cells, k) = Self::renumber(cell_of);
        let q = Self::of_cells(graph, cells, k);
        let cells = q.cell_of.as_ref().expect("explicit");
        let mut row = vec![0.0; k];
        for v in 0..n {
            row.iter_mut().for_each(|x| *x = 0.0);
            for &(w, wt) in graph.neighbors(v) {
                row[cells[w]] += wt;
            }
            let i = cells[v];
            for (j, &x) in row.iter().enumerate() {
                let expected = q.matrix[(i, j)];
                if (x - expected).abs() > 1e-9 * expected.abs().max(1.0) {
                    return Err(QsoError::invalid(
                        "cell_of",
                        format!(
                            "partition is not equitable: node {v} has weight {x} into cell \
                             {j}, other nodes of its cell {expected}"
                        ),
                    ));
                }
            }
        }
        Ok(q)
    }

    /// Distance partition of the complete graph `K_n` from one node: the
    /// node and the `n − 1` others.
    pub fn complete(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(QsoError::invalid("n", "need at least one node"));
        }
        let q = if n == 1 {
            Self::from_matrix(vec![1.0], RMatrix::zeros(1, 1))?
        } else {
            let m = (n - 1) as f64;
            Self::from_matrix(
                vec![1.0, m],
                RMatrix::from_row_slice(2, 2, &[0.0, m, 1.0, m - 1.0]),
            )?
        };
        Ok(q.with_name(format!("complete({n}) from a node")))
    }

    /// Complete multipartite graph with parts of the given sizes, seen from
    /// a node of the first part: that node, the rest of its part (if any),
    /// and every other part.
    pub fn complete_multipartite(parts: &[usize]) -> Result<Self> {
        if parts.is_empty() || parts.contains(&0) {
            return Err(QsoError::invalid("parts", "need non-empty parts"));
        }
        // Cells: the start node, the rest of part 0, parts 1.. .
        let mut sizes = vec![1.0];
        let mut part_of_cell = vec![0];
        if parts[0] > 1 {
            sizes.push((parts[0] - 1) as f64);
            part_of_cell.push(0);
        }
        for (p, &size) in parts.iter().enumerate().skip(1) {
            sizes.push(size as f64);
            part_of_cell.push(p);
        }
        let k = sizes.len();
        let matrix = RMatrix::from_fn(k, k, |i, j| {
            if part_of_cell[i] == part_of_cell[j] {
                0.0
            } else {
                sizes[j]
            }
        });
        Ok(Self::from_matrix(sizes, matrix)?.with_name(format!(
            "complete_multipartite({parts:?}) from a node of part 0"
        )))
    }

    /// Distance partition of the Hamming graph `H(d, q)` (words of length
    /// `d` over `q` letters, adjacent when they differ in one letter) from
    /// a word: cell `k` holds the `C(d,k)(q−1)^k` words at distance `k`.
    pub fn hamming(d: usize, q: usize) -> Result<Self> {
        if d == 0 || q < 2 {
            return Err(QsoError::invalid("hamming", "need d ≥ 1 and q ≥ 2"));
        }
        if d as f64 * (q as f64).ln() > 709.0 {
            return Err(QsoError::invalid(
                "hamming",
                "q^d nodes exceed the floating-point range",
            ));
        }
        let qm = (q - 1) as f64;
        let sizes: Vec<f64> = (0..=d)
            .map(|k| binomial(d, k) * qm.powi(k as i32))
            .collect();
        let matrix = RMatrix::from_fn(d + 1, d + 1, |i, j| {
            if j + 1 == i {
                i as f64
            } else if j == i {
                i as f64 * (q as f64 - 2.0)
            } else if j == i + 1 {
                (d - i) as f64 * qm
            } else {
                0.0
            }
        });
        Ok(Self::from_matrix(sizes, matrix)?.with_name(format!("hamming({d},{q}) from a word")))
    }

    /// Distance partition of the hypercube `Q_d = H(d, 2)` from a node:
    /// cell `k` holds the `C(d,k)` nodes of Hamming weight `k`.
    pub fn hypercube(d: usize) -> Result<Self> {
        Ok(Self::hamming(d, 2)?.with_name(format!("hypercube({d}) from a node")))
    }

    /// Distance partition of the Johnson graph `J(n, k)` (`k`-subsets of
    /// `n` elements, adjacent when they share `k − 1`) from a subset `S`:
    /// cell `i` holds the `C(k,i)·C(n−k,i)` subsets with `|X \ S| = i`.
    pub fn johnson(n: usize, k: usize) -> Result<Self> {
        if k == 0 || k >= n {
            return Err(QsoError::invalid("johnson", "need 0 < k < n"));
        }
        let m = k.min(n - k);
        let sizes: Vec<f64> = (0..=m)
            .map(|i| binomial(k, i) * binomial(n - k, i))
            .collect();
        let matrix = RMatrix::from_fn(m + 1, m + 1, |i, j| {
            let (fi, ff, fk) = (i as f64, n as f64, k as f64);
            if j + 1 == i {
                fi * fi
            } else if j == i {
                fi * (ff - 2.0 * fi)
            } else if j == i + 1 {
                (fk - fi) * (ff - fk - fi)
            } else {
                0.0
            }
        });
        Ok(Self::from_matrix(sizes, matrix)?.with_name(format!("johnson({n},{k}) from a subset")))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_cells(&self) -> usize {
        self.sizes.len()
    }

    /// Number of nodes in each cell.
    pub fn sizes(&self) -> &[f64] {
        &self.sizes
    }

    /// Number of nodes of the graph (a float: implicit families may be far
    /// larger than any index type).
    pub fn num_nodes(&self) -> f64 {
        self.sizes.iter().sum()
    }

    /// The quotient matrix `B` (weight from a node of cell `i` into cell `j`).
    pub fn matrix(&self) -> &RMatrix {
        &self.matrix
    }

    /// Cell of every node, for partitions of an explicit graph.
    pub fn cell_of(&self) -> Option<&[usize]> {
        self.cell_of.as_deref()
    }

    /// Weighted degree of the nodes of each cell.
    pub fn degrees(&self) -> Vec<f64> {
        (0..self.num_cells())
            .map(|i| self.matrix.row(i).sum())
            .collect()
    }

    /// The adjacency matrix on the cell states (symmetric).
    pub fn adjacency(&self) -> RMatrix {
        let k = self.num_cells();
        RMatrix::from_fn(k, k, |i, j| {
            let (bij, bji) = (self.matrix[(i, j)], self.matrix[(j, i)]);
            if i == j {
                bij
            } else {
                bij.signum() * (bij * bji).abs().sqrt()
            }
        })
    }

    /// The Laplacian on the cell states (symmetric).
    pub fn laplacian(&self) -> RMatrix {
        let mut l = -self.adjacency();
        for (i, d) in self.degrees().into_iter().enumerate() {
            l[(i, i)] += d;
        }
        l
    }

    /// `γ·L_red` or `−γ·A_red`.
    pub fn hamiltonian(&self, generator: Generator, gamma: f64) -> RMatrix {
        match generator {
            Generator::Laplacian => self.laplacian() * gamma,
            Generator::Adjacency => self.adjacency() * -gamma,
        }
    }

    /// The uniform superposition over all nodes, in the cell basis:
    /// `√(|C_i|/N)`.
    pub fn uniform_state(&self) -> Vec<f64> {
        let n = self.num_nodes();
        self.sizes.iter().map(|s| (s / n).sqrt()).collect()
    }

    /// Per-node values from per-cell probabilities: node `v` gets
    /// `per_cell[cell(v)] / |cell(v)|`. Only for partitions of an explicit
    /// graph.
    pub fn node_probabilities(&self, per_cell: &[f64]) -> Result<Vec<f64>> {
        let cells = self.cell_of.as_ref().ok_or_else(|| {
            QsoError::invalid("quotient", "node probabilities need an explicit graph")
        })?;
        self.check_len(per_cell.len())?;
        Ok(cells.iter().map(|&c| per_cell[c] / self.sizes[c]).collect())
    }

    fn check_len(&self, len: usize) -> Result<()> {
        if len != self.num_cells() {
            return Err(QsoError::DimensionMismatch {
                expected: self.num_cells(),
                actual: len,
            });
        }
        Ok(())
    }

    fn check_cell(&self, cell: usize) -> Result<()> {
        if cell >= self.num_cells() {
            return Err(QsoError::invalid(
                "cell",
                format!("cell {cell} out of range for {} cells", self.num_cells()),
            ));
        }
        Ok(())
    }
}

/// Continuous-time walk on a [`Quotient`]. A walk "started at cell `c`"
/// starts in `|C⟩`: the node itself for a single-node cell, otherwise the
/// uniform superposition over the cell.
#[derive(Clone, Debug)]
pub struct QuotientWalk {
    quotient: Quotient,
    generator: Generator,
    gamma: f64,
    values: Vec<f64>,
    vectors: RMatrix,
}

impl QuotientWalk {
    /// Walk with hopping rate `gamma > 0` (`H = γL` or `H = −γA`, as for
    /// [`ContinuousWalk`](super::ContinuousWalk)).
    pub fn new(quotient: &Quotient, generator: Generator, gamma: f64) -> Result<Self> {
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        let (values, vectors) = eigh_real(&quotient.hamiltonian(generator, gamma));
        Ok(Self {
            quotient: quotient.clone(),
            generator,
            gamma,
            values,
            vectors,
        })
    }

    pub fn quotient(&self) -> &Quotient {
        &self.quotient
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn generator(&self) -> Generator {
        self.generator
    }

    /// Eigenvalues of the reduced Hamiltonian (ascending): the eigenvalues
    /// of `H` the walker can see.
    pub fn spectrum(&self) -> &[f64] {
        &self.values
    }

    /// `exp(−iHt)` applied to cell-basis amplitudes `c0`.
    pub fn evolve(&self, c0: &[C64], t: f64) -> Result<Vec<C64>> {
        self.quotient.check_len(c0.len())?;
        if !t.is_finite() {
            return Err(QsoError::invalid("t", "evolution time must be finite"));
        }
        let k = c0.len();
        let c: Vec<C64> = (0..k)
            .map(|a| {
                let overlap: C64 = (0..k).map(|j| c0[j] * self.vectors[(j, a)]).sum();
                overlap * C64::from_polar(1.0, -self.values[a] * t)
            })
            .collect();
        Ok((0..k)
            .map(|j| (0..k).map(|a| c[a] * self.vectors[(j, a)]).sum())
            .collect())
    }

    /// `⟨C_to| exp(−iHt) |C_from⟩`; for single-node cells, the amplitude
    /// between the two nodes.
    pub fn amplitude(&self, from: usize, to: usize, t: f64) -> Result<C64> {
        self.quotient.check_cell(from)?;
        self.quotient.check_cell(to)?;
        if !t.is_finite() {
            return Err(QsoError::invalid("t", "evolution time must be finite"));
        }
        Ok((0..self.values.len())
            .map(|a| {
                C64::from_polar(1.0, -self.values[a] * t)
                    * (self.vectors[(from, a)] * self.vectors[(to, a)])
            })
            .sum())
    }

    /// Probability of each cell at time `t` for the walk started at cell
    /// `start`. Divide by the cell sizes for single nodes (see
    /// [`Quotient::node_probabilities`]).
    pub fn cell_probabilities(&self, start: usize, t: f64) -> Result<Vec<f64>> {
        self.quotient.check_cell(start)?;
        let mut c0 = vec![C64::new(0.0, 0.0); self.values.len()];
        c0[start] = C64::new(1.0, 0.0);
        Ok(self.evolve(&c0, t)?.iter().map(|a| a.norm_sqr()).collect())
    }

    fn start_spectrum(&self, start: usize) -> Result<StartSpectrum<'_>> {
        self.quotient.check_cell(start)?;
        Ok(StartSpectrum {
            values: std::borrow::Cow::Borrowed(&self.values),
            coeffs: (0..self.values.len())
                .map(|a| self.vectors[(start, a)])
                .collect(),
            vectors: std::borrow::Cow::Borrowed(&self.vectors),
        })
    }

    /// Limiting (infinite-time-averaged) probability of each cell.
    pub fn limiting_distribution(&self, start: usize) -> Result<Vec<f64>> {
        Ok(self.start_spectrum(start)?.limiting_distribution())
    }

    /// Exact time average `(1/T)∫₀ᵀ P(cell, t) dt`.
    pub fn average_distribution(&self, start: usize, t_max: f64) -> Result<Vec<f64>> {
        if t_max.is_nan() || t_max <= 0.0 {
            return self.cell_probabilities(start, 0.0);
        }
        Ok(self.start_spectrum(start)?.average_distribution(t_max))
    }
}

/// Childs–Goldstone spatial search on a [`Quotient`]: the walker starts in
/// the uniform superposition and evolves under `H = γL − Σ_{C∈M} Π_C` (or
/// `−γA − Σ Π_C`), where `Π_C` projects onto the nodes of a marked cell.
/// For a single marked node, make it a cell of its own (for example with
/// [`Quotient::with_singletons`], or cell 0 of the implicit families).
#[derive(Clone, Debug)]
pub struct QuotientSearch {
    marked: Vec<usize>,
    gamma: f64,
    values: Vec<f64>,
    overlaps: Vec<f64>,
    vectors: RMatrix,
}

impl QuotientSearch {
    pub fn new(
        quotient: &Quotient,
        marked_cells: &[usize],
        generator: Generator,
        gamma: f64,
    ) -> Result<Self> {
        if marked_cells.is_empty() {
            return Err(QsoError::invalid(
                "marked",
                "at least one marked cell required",
            ));
        }
        for &c in marked_cells {
            quotient.check_cell(c)?;
        }
        if !(gamma.is_finite() && gamma >= 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "must be finite and non-negative",
            ));
        }
        let mut marked = marked_cells.to_vec();
        marked.sort_unstable();
        marked.dedup();
        let mut h = quotient.hamiltonian(generator, gamma);
        for &c in &marked {
            h[(c, c)] -= 1.0;
        }
        let (values, vectors) = eigh_real(&h);
        let s = quotient.uniform_state();
        let k = s.len();
        let overlaps = (0..k)
            .map(|a| (0..k).map(|j| vectors[(j, a)] * s[j]).sum())
            .collect();
        Ok(Self {
            marked,
            gamma,
            values,
            overlaps,
            vectors,
        })
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn marked(&self) -> &[usize] {
        &self.marked
    }

    /// Probability of finding a node of a marked cell at time `t`.
    pub fn success_probability(&self, t: f64) -> f64 {
        let phases: Vec<C64> = self
            .values
            .iter()
            .zip(&self.overlaps)
            .map(|(&e, &o)| C64::from_polar(o, -e * t))
            .collect();
        self.marked
            .iter()
            .map(|&c| {
                let amp: C64 = (0..phases.len())
                    .map(|a| phases[a] * self.vectors[(c, a)])
                    .sum();
                amp.norm_sqr()
            })
            .sum()
    }

    /// Maximise the success probability over `t ∈ [0, t_max]` (as
    /// [`SpatialSearch::optimize_time`](super::SpatialSearch::optimize_time)).
    pub fn optimize_time(&self, t_max: f64, resolution: usize) -> SearchResult {
        optimize_peak(self.gamma, t_max, resolution, |t| {
            self.success_probability(t)
        })
    }

    /// Critical hopping rate `γ_c = ⟨C|L⁺|C⟩` for a marked cell `C`, with
    /// `L⁺` the pseudo-inverse of the Laplacian. For one marked node of a
    /// vertex-transitive graph this is the Childs–Goldstone rate
    /// `(1/N)·Σ_{λ>0} 1/λ` of
    /// [`SpatialSearch::critical_gamma`](super::SpatialSearch::critical_gamma),
    /// obtained here without the full spectrum.
    pub fn critical_gamma(quotient: &Quotient, cell: usize) -> Result<f64> {
        quotient.check_cell(cell)?;
        if quotient.num_cells() < 2 {
            return Err(QsoError::invalid("quotient", "need at least two cells"));
        }
        let (values, vectors) = eigh_real(&quotient.laplacian());
        let scale = values.last().copied().unwrap_or(1.0).abs().max(1.0);
        let tol = 1e-9 * scale;
        if values.iter().filter(|&&l| l.abs() <= tol).count() != 1 {
            return Err(QsoError::invalid("quotient", "graph must be connected"));
        }
        Ok(values
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l > tol)
            .map(|(a, &l)| vectors[(cell, a)].powi(2) / l)
            .sum())
    }

    /// Optimise the hopping rate (around the
    /// [`critical_gamma`](Self::critical_gamma) of the first marked cell)
    /// and the measurement time together, as
    /// [`SpatialSearch::optimize`](super::SpatialSearch::optimize).
    pub fn optimize(
        quotient: &Quotient,
        marked_cells: &[usize],
        generator: Generator,
        t_max: f64,
        resolution: usize,
    ) -> Result<SearchResult> {
        let first = *marked_cells
            .first()
            .ok_or_else(|| QsoError::invalid("marked", "at least one marked cell required"))?;
        let gc = Self::critical_gamma(quotient, first)?;
        optimize_rate(gc, |gamma| {
            Ok(Self::new(quotient, marked_cells, generator, gamma)?
                .optimize_time(t_max, resolution))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{generators, metatron};
    use crate::rng::Rng;
    use crate::walks::{ContinuousWalk, SpatialSearch};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn reduced_walks_match_full_walks() {
        let mut rng = Rng::seed_from_u64(3);
        let graphs = vec![
            generators::petersen(),
            metatron::metatron_cube(),
            metatron::fruit_of_life(),
            generators::hypercube(5),
            generators::grid_2d(4, 5, false),
            generators::erdos_renyi(18, 0.3, &mut rng).unwrap(),
            Graph::from_weighted_edges(5, &[(0, 1, 2.0), (1, 2, 0.5), (2, 3, 1.0), (3, 4, 2.0)])
                .unwrap(),
        ];
        for g in &graphs {
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                let start = g.num_nodes() / 3;
                let q = Quotient::with_singletons(g, &[start]).unwrap();
                let full = ContinuousWalk::new(g, generator, 0.8).unwrap();
                let reduced = QuotientWalk::new(&q, generator, 0.8).unwrap();
                let cell = q.cell_of().unwrap()[start];
                assert_eq!(q.sizes()[cell], 1.0);
                for t in [0.0, 0.3, 1.7, 6.0] {
                    let p = q
                        .node_probabilities(&reduced.cell_probabilities(cell, t).unwrap())
                        .unwrap();
                    assert!(close(&p, &full.probabilities(start, t).unwrap(), 1e-12));
                }
                let limit = q
                    .node_probabilities(&reduced.limiting_distribution(cell).unwrap())
                    .unwrap();
                assert!(close(
                    &limit,
                    &full.limiting_distribution(start).unwrap(),
                    1e-12
                ));
                let avg = q
                    .node_probabilities(&reduced.average_distribution(cell, 4.0).unwrap())
                    .unwrap();
                assert!(close(
                    &avg,
                    &full.average_distribution(start, 4.0).unwrap(),
                    1e-12
                ));
                // A reduced amplitude between singletons is the node amplitude.
                let u = full.propagator(1.3).unwrap();
                let amp = reduced.amplitude(cell, cell, 1.3).unwrap();
                assert!((amp - u[(start, start)]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn equitable_partitions_of_symmetric_graphs_are_small() {
        let g = generators::hypercube(10);
        let q = Quotient::with_singletons(&g, &[0]).unwrap();
        assert_eq!(q.num_cells(), 11);
        assert_eq!(q.num_nodes(), 1024.0);
        // The same as the implicit distance partition, up to cell order.
        let implicit = Quotient::hypercube(10).unwrap();
        let mut a: Vec<f64> = q.sizes().to_vec();
        let mut b: Vec<f64> = implicit.sizes().to_vec();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(a, b);
        let mut ev_a = eigh_real(&q.laplacian()).0;
        let ev_b = eigh_real(&implicit.laplacian()).0;
        ev_a.sort_by(f64::total_cmp);
        assert!(close(&ev_a, &ev_b, 1e-10));
        // Every vertex of a vertex-transitive graph looks the same; fixing
        // an edge {0, 1} of the Petersen graph leaves the cells {0}, {1},
        // the other neighbours of 0, those of 1, and the remaining four.
        let p = generators::petersen();
        let q = Quotient::with_singletons(&p, &[0, 1]).unwrap();
        assert_eq!(q.sizes(), &[1.0, 1.0, 2.0, 4.0, 2.0]);
        // The trivial partition is equitable exactly for regular graphs.
        assert!(Quotient::from_cells(&p, &[0; 10]).is_ok());
        assert!(Quotient::from_cells(&generators::path(4), &[0; 4]).is_err());
        let orbit_cells: Vec<usize> = (0..10).map(|v| usize::from(v >= 5)).collect();
        let q = Quotient::from_cells(&p, &orbit_cells).unwrap();
        assert_eq!(
            q.matrix(),
            &RMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0])
        );
    }

    /// Probability of each distance class from node 0 under the full walk.
    fn by_distance(g: &Graph, walk: &ContinuousWalk, t: f64) -> Vec<f64> {
        let d = g.distances_from(0);
        let p = walk.probabilities(0, t).unwrap();
        let mut out = vec![0.0; d.iter().flatten().max().unwrap() + 1];
        for (v, &dv) in d.iter().enumerate() {
            out[dv.unwrap()] += p[v];
        }
        out
    }

    #[test]
    fn implicit_families_match_explicit_graphs() {
        let cases = vec![
            (
                Quotient::hamming(3, 3).unwrap(),
                generators::hamming(3, 3).unwrap(),
            ),
            (Quotient::hamming(4, 2).unwrap(), generators::hypercube(4)),
            (
                Quotient::johnson(7, 3).unwrap(),
                generators::johnson(7, 3).unwrap(),
            ),
            (
                Quotient::johnson(6, 4).unwrap(),
                generators::johnson(6, 4).unwrap(),
            ),
            (Quotient::complete(9).unwrap(), generators::complete(9)),
        ];
        for (q, g) in &cases {
            assert_eq!(q.num_nodes(), g.num_nodes() as f64, "{}", q.name());
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                let full = ContinuousWalk::new(g, generator, 0.6).unwrap();
                let reduced = QuotientWalk::new(q, generator, 0.6).unwrap();
                for t in [0.4, 2.2] {
                    assert!(
                        close(
                            &reduced.cell_probabilities(0, t).unwrap(),
                            &by_distance(g, &full, t),
                            1e-12
                        ),
                        "{}",
                        q.name()
                    );
                }
            }
        }
        // Complete multipartite K_{2,3,4} from a node of the first part.
        let parts = [2usize, 3, 4];
        let label: Vec<usize> = parts
            .iter()
            .enumerate()
            .flat_map(|(p, &s)| std::iter::repeat_n(p, s))
            .collect();
        let mut edges = Vec::new();
        for u in 0..9 {
            for v in u + 1..9 {
                if label[u] != label[v] {
                    edges.push((u, v));
                }
            }
        }
        let g = Graph::from_edges(9, &edges).unwrap();
        let q = Quotient::complete_multipartite(&parts).unwrap();
        let full = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        let reduced = QuotientWalk::new(&q, Generator::Laplacian, 1.0).unwrap();
        let p = full.probabilities(0, 0.9).unwrap();
        let expected = [p[0], p[1], p[2..5].iter().sum(), p[5..].iter().sum()];
        assert!(close(
            &reduced.cell_probabilities(0, 0.9).unwrap(),
            &expected,
            1e-12
        ));
    }

    #[test]
    fn hypercube_of_dimension_thirty_has_perfect_state_transfer() {
        // Q_d factorises into d two-level systems: from 0…0 the probability
        // of the cell of weight k is C(d,k)·sin^{2k}(γt)·cos^{2(d−k)}(γt).
        let d = 30;
        let q = Quotient::hypercube(d).unwrap();
        assert_eq!(q.num_nodes(), 2f64.powi(30));
        let walk = QuotientWalk::new(&q, Generator::Laplacian, 1.0).unwrap();
        for t in [0.2, 0.7, 1.1, FRAC_PI_2] {
            let p = walk.cell_probabilities(0, t).unwrap();
            let (s, c) = (t.sin().powi(2), t.cos().powi(2));
            for (k, &pk) in p.iter().enumerate() {
                let expected = binomial(d, k) * s.powi(k as i32) * c.powi((d - k) as i32);
                assert!((pk - expected).abs() < 1e-10, "t={t} k={k}");
            }
            let amp = walk.amplitude(0, d, t).unwrap();
            assert!((amp.norm() - t.sin().abs().powi(d as i32)).abs() < 1e-10);
        }
        let p = walk.cell_probabilities(0, FRAC_PI_2).unwrap();
        assert!((p[d] - 1.0).abs() < 1e-10);
        // Far larger instances are still 1001-dimensional problems.
        let big = Quotient::hypercube(1000).unwrap();
        let walk = QuotientWalk::new(&big, Generator::Adjacency, 1.0).unwrap();
        let p = walk.cell_probabilities(0, FRAC_PI_2).unwrap();
        assert!((p[1000] - 1.0).abs() < 1e-8);
        assert!(Quotient::hypercube(1100).is_err());
    }

    #[test]
    fn reduced_search_matches_full_search() {
        let g = generators::hypercube(8);
        let q = Quotient::hypercube(8).unwrap();
        let gc = SpatialSearch::critical_gamma(&g).unwrap();
        assert!((QuotientSearch::critical_gamma(&q, 0).unwrap() - gc).abs() < 1e-12);
        for generator in [Generator::Laplacian, Generator::Adjacency] {
            let full = SpatialSearch::new(&g, &[0], generator, gc).unwrap();
            let reduced = QuotientSearch::new(&q, &[0], generator, gc).unwrap();
            for t in [0.0, 5.0, 27.5, 40.0] {
                assert!(
                    (full.success_probability(t) - reduced.success_probability(t)).abs() < 1e-12
                );
            }
        }
        // Explicit partition of a graph without vertex transitivity.
        let g = generators::grid_2d(5, 5, false);
        let q = Quotient::with_singletons(&g, &[12]).unwrap();
        let cell = q.cell_of().unwrap()[12];
        let full = SpatialSearch::new(&g, &[12], Generator::Laplacian, 0.3).unwrap();
        let reduced = QuotientSearch::new(&q, &[cell], Generator::Laplacian, 0.3).unwrap();
        for t in [1.0, 3.0, 9.0] {
            assert!((full.success_probability(t) - reduced.success_probability(t)).abs() < 1e-12);
        }
        // K_N with a million nodes: certainty at t = π√N/2 for γ = 1/N.
        let n = 1_000_000;
        let k = Quotient::complete(n).unwrap();
        let s = QuotientSearch::new(&k, &[0], Generator::Laplacian, 1.0 / n as f64).unwrap();
        assert!((s.success_probability(PI * (n as f64).sqrt() / 2.0) - 1.0).abs() < 1e-9);
        assert!(
            (QuotientSearch::critical_gamma(&k, 0).unwrap() - (n - 1) as f64 / (n as f64).powi(2))
                .abs()
                < 1e-15
        );
        // Search on Q_20 (about a million nodes) at the critical rate.
        let q20 = Quotient::hypercube(20).unwrap();
        let best =
            QuotientSearch::optimize(&q20, &[0], Generator::Laplacian, 3000.0, 3000).unwrap();
        assert!(best.success_probability > 0.85, "{best:?}");
    }

    #[test]
    fn invalid_quotients_are_rejected() {
        // Edge counts between cells must agree from both sides.
        let bad = RMatrix::from_row_slice(2, 2, &[0.0, 3.0, 2.0, 0.0]);
        assert!(Quotient::from_matrix(vec![1.0, 3.0], bad).is_err());
        let loop_in_singleton = RMatrix::from_row_slice(1, 1, &[1.0]);
        assert!(Quotient::from_matrix(vec![1.0], loop_in_singleton).is_err());
        assert!(Quotient::from_matrix(vec![0.5], RMatrix::zeros(1, 1)).is_err());
        assert!(Quotient::from_matrix(vec![], RMatrix::zeros(0, 0)).is_err());
        assert!(Quotient::johnson(4, 4).is_err());
        assert!(Quotient::complete_multipartite(&[2, 0]).is_err());
        let q = Quotient::complete(5).unwrap();
        let walk = QuotientWalk::new(&q, Generator::Laplacian, 1.0).unwrap();
        assert!(walk.cell_probabilities(2, 1.0).is_err());
        assert!(walk.evolve(&[C64::new(1.0, 0.0)], 1.0).is_err());
        assert!(q.node_probabilities(&[1.0, 0.0]).is_err());
        let disconnected = Graph::from_edges(4, &[(0, 1), (2, 3)]).unwrap();
        let q = Quotient::with_singletons(&disconnected, &[0]).unwrap();
        assert!(QuotientSearch::critical_gamma(&q, 0).is_err());
    }
}
