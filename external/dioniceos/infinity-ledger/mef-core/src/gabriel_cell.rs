/*!
 * Gabriel Cell Module - Modular Feedback Resonator
 *
 * This module provides the GabrielCell class, a minimalistic feedback cell
 * that implements a modular resonator with three parameters: psi (activation),
 * rho (coherence), and omega (rhythm/oscillation). The cell can respond
 * dynamically to feedback and inputs, implement Hebbian learning, and couple
 * with neighboring cells.
 */

/// Minimalistic feedback cell: modular resonator with (psi, rho, omega)
///
/// A GabrielCell can respond dynamically to feedback and inputs, implementing
/// Hebbian learning and coupling with neighbors. The cell maintains three
/// primary parameters:
/// - psi: Activation level
/// - rho: Coherence
/// - omega: Rhythm/Oscillation
///
/// The output is computed as: output = psi * rho * omega
#[derive(Debug, Clone)]
pub struct GabrielCell {
    /// Activation level
    pub psi: f64,
    /// Coherence
    pub rho: f64,
    /// Rhythm/Oscillation
    pub omega: f64,
    /// Learning rate for Hebbian updates
    pub learn_rate: f64,
    /// Current output value
    pub output: f64,
    /// References to neighbor cells (indices in a collection)
    pub neighbors: Vec<usize>,
}

impl GabrielCell {
    /// Create a new GabrielCell with the given parameters
    ///
    /// # Arguments
    ///
    /// * `psi` - Initial activation level (default: 1.0)
    /// * `rho` - Initial coherence (default: 1.0)
    /// * `omega` - Initial rhythm/oscillation (default: 1.0)
    /// * `learn_rate` - Learning rate for updates (default: 0.12)
    ///
    /// # Returns
    ///
    /// A new GabrielCell instance
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

    /// Activate the cell with optional input
    ///
    /// If an input is provided, psi is updated using a weighted combination:
    /// psi = (1 - learn_rate) * psi + learn_rate * input
    ///
    /// The output is then recomputed as psi * rho * omega.
    ///
    /// # Arguments
    ///
    /// * `input` - Optional input value to modulate activation
    ///
    /// # Returns
    ///
    /// The new output value
    pub fn activate(&mut self, input: Option<f64>) -> f64 {
        if let Some(inp) = input {
            self.psi = (1.0 - self.learn_rate) * self.psi + self.learn_rate * inp;
        }
        self.output = self.psi * self.rho * self.omega;
        self.output
    }

    /// Apply feedback to adjust cell parameters
    ///
    /// Adjusts psi, rho, and omega based on the error between target and output:
    /// - psi += learn_rate * error
    /// - rho += learn_rate * tanh(error)
    /// - omega += learn_rate * sin(error)
    ///
    /// All parameters are clipped to [0.01, 10.0] to maintain stability.
    ///
    /// # Arguments
    ///
    /// * `target` - Target output value for feedback
    pub fn feedback(&mut self, target: f64) {
        let error = target - self.output;

        self.psi += self.learn_rate * error;
        self.rho += self.learn_rate * error.tanh();
        self.omega += self.learn_rate * error.sin();

        // Clip parameters to maintain stability
        self.psi = self.psi.clamp(0.01, 10.0);
        self.rho = self.rho.clamp(0.01, 10.0);
        self.omega = self.omega.clamp(0.01, 10.0);
    }

    /// Add a neighbor reference
    ///
    /// # Arguments
    ///
    /// * `neighbor_idx` - Index of the neighbor cell
    pub fn add_neighbor(&mut self, neighbor_idx: usize) {
        if !self.neighbors.contains(&neighbor_idx) {
            self.neighbors.push(neighbor_idx);
        }
    }

    /// Remove a neighbor reference
    ///
    /// # Arguments
    ///
    /// * `neighbor_idx` - Index of the neighbor cell to remove
    pub fn remove_neighbor(&mut self, neighbor_idx: usize) {
        self.neighbors.retain(|&idx| idx != neighbor_idx);
    }

    /// Check if this cell has a specific neighbor
    ///
    /// # Arguments
    ///
    /// * `neighbor_idx` - Index of the neighbor cell to check
    ///
    /// # Returns
    ///
    /// True if the cell is a neighbor, false otherwise
    pub fn has_neighbor(&self, neighbor_idx: usize) -> bool {
        self.neighbors.contains(&neighbor_idx)
    }
}

impl Default for GabrielCell {
    /// Create a default GabrielCell with standard parameters
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0, 0.12)
    }
}

/// Helper function to couple two cells bidirectionally
///
/// This is a convenience function for coupling cells in a collection.
/// It ensures both cells have each other as neighbors.
///
/// # Arguments
///
/// * `cells` - Mutable reference to a collection of cells
/// * `idx1` - Index of the first cell
/// * `idx2` - Index of the second cell
pub fn couple_cells(cells: &mut [GabrielCell], idx1: usize, idx2: usize) {
    if idx1 < cells.len() && idx2 < cells.len() && idx1 != idx2 {
        cells[idx1].add_neighbor(idx2);
        cells[idx2].add_neighbor(idx1);
    }
}

/// Apply neighbor feedback to a cell in a collection
///
/// Computes the average output of all neighbors and uses it as feedback
/// target for the specified cell.
///
/// # Arguments
///
/// * `cells` - Mutable reference to a collection of cells
/// * `idx` - Index of the cell to update
pub fn neighbor_feedback(cells: &mut [GabrielCell], idx: usize) {
    if idx >= cells.len() {
        return;
    }

    let neighbors = cells[idx].neighbors.clone();
    if neighbors.is_empty() {
        return;
    }

    // Compute average output of neighbors
    let mut sum = 0.0;
    for &neighbor_idx in &neighbors {
        if neighbor_idx < cells.len() {
            sum += cells[neighbor_idx].output;
        }
    }
    let avg = sum / neighbors.len() as f64;

    // Apply feedback
    cells[idx].feedback(avg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_cell() {
        let cell = GabrielCell::default();
        assert_eq!(cell.psi, 1.0);
        assert_eq!(cell.rho, 1.0);
        assert_eq!(cell.omega, 1.0);
        assert_eq!(cell.learn_rate, 0.12);
        assert_eq!(cell.output, 1.0);
        assert!(cell.neighbors.is_empty());
    }

    #[test]
    fn test_create_custom_cell() {
        let cell = GabrielCell::new(0.5, 2.0, 1.5, 0.2);
        assert_eq!(cell.psi, 0.5);
        assert_eq!(cell.rho, 2.0);
        assert_eq!(cell.omega, 1.5);
        assert_eq!(cell.learn_rate, 0.2);
        assert_eq!(cell.output, 0.5 * 2.0 * 1.5);
    }

    #[test]
    fn test_activate_without_input() {
        let mut cell = GabrielCell::new(2.0, 3.0, 4.0, 0.1);
        let initial_psi = cell.psi;

        let output = cell.activate(None);
        assert_eq!(cell.psi, initial_psi);
        assert_eq!(output, 2.0 * 3.0 * 4.0);
    }

    #[test]
    fn test_activate_with_input() {
        let mut cell = GabrielCell::new(1.0, 1.0, 1.0, 0.1);

        let output = cell.activate(Some(2.0));
        // psi = (1 - 0.1) * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
        assert!((cell.psi - 1.1).abs() < 1e-10);
        assert_eq!(output, 1.1 * 1.0 * 1.0);
    }

    #[test]
    fn test_activate_multiple_steps() {
        let mut cell = GabrielCell::new(0.0, 1.0, 1.0, 0.5);

        cell.activate(Some(1.0));
        assert!((cell.psi - 0.5).abs() < 1e-10);

        cell.activate(Some(1.0));
        assert!((cell.psi - 0.75).abs() < 1e-10);

        cell.activate(Some(1.0));
        assert!((cell.psi - 0.875).abs() < 1e-10);
    }

    #[test]
    fn test_feedback_positive_error() {
        let mut cell = GabrielCell::new(1.0, 1.0, 1.0, 0.1);
        let target = 2.0;

        cell.feedback(target);

        // Error = 2.0 - 1.0 = 1.0
        // psi = 1.0 + 0.1 * 1.0 = 1.1
        assert!((cell.psi - 1.1).abs() < 1e-10);
        // rho = 1.0 + 0.1 * tanh(1.0) ≈ 1.0 + 0.0762 = 1.0762
        assert!((cell.rho - (1.0 + 0.1 * 1.0_f64.tanh())).abs() < 1e-10);
        // omega = 1.0 + 0.1 * sin(1.0) ≈ 1.0 + 0.0841 = 1.0841
        assert!((cell.omega - (1.0 + 0.1 * 1.0_f64.sin())).abs() < 1e-10);
    }

    #[test]
    fn test_feedback_negative_error() {
        let mut cell = GabrielCell::new(2.0, 1.0, 1.0, 0.1);
        let target = 1.0;

        let initial_output = cell.output;
        cell.feedback(target);

        // Error = 1.0 - 2.0 = -1.0
        // psi should decrease
        assert!(cell.psi < 2.0);
    }

    #[test]
    fn test_feedback_clipping_lower_bound() {
        let mut cell = GabrielCell::new(0.05, 0.05, 0.05, 0.5);
        cell.feedback(0.0);

        // All parameters should be clipped to at least 0.01
        assert!(cell.psi >= 0.01);
        assert!(cell.rho >= 0.01);
        assert!(cell.omega >= 0.01);
    }

    #[test]
    fn test_feedback_clipping_upper_bound() {
        let mut cell = GabrielCell::new(9.5, 9.5, 9.5, 0.5);
        cell.feedback(1000.0);

        // All parameters should be clipped to at most 10.0
        assert!(cell.psi <= 10.0);
        assert!(cell.rho <= 10.0);
        assert!(cell.omega <= 10.0);
    }

    #[test]
    fn test_add_neighbor() {
        let mut cell = GabrielCell::default();

        cell.add_neighbor(1);
        assert_eq!(cell.neighbors.len(), 1);
        assert!(cell.has_neighbor(1));

        cell.add_neighbor(2);
        assert_eq!(cell.neighbors.len(), 2);
        assert!(cell.has_neighbor(2));

        // Adding same neighbor twice should not duplicate
        cell.add_neighbor(1);
        assert_eq!(cell.neighbors.len(), 2);
    }

    #[test]
    fn test_remove_neighbor() {
        let mut cell = GabrielCell::default();

        cell.add_neighbor(1);
        cell.add_neighbor(2);
        cell.add_neighbor(3);

        cell.remove_neighbor(2);
        assert_eq!(cell.neighbors.len(), 2);
        assert!(!cell.has_neighbor(2));
        assert!(cell.has_neighbor(1));
        assert!(cell.has_neighbor(3));
    }

    #[test]
    fn test_couple_cells() {
        let mut cells = vec![GabrielCell::default(), GabrielCell::default()];

        couple_cells(&mut cells, 0, 1);

        assert!(cells[0].has_neighbor(1));
        assert!(cells[1].has_neighbor(0));
    }

    #[test]
    fn test_couple_cells_invalid_indices() {
        let mut cells = vec![GabrielCell::default(), GabrielCell::default()];

        // Should not panic with out-of-bounds indices
        couple_cells(&mut cells, 0, 5);
        assert!(cells[0].neighbors.is_empty());
    }

    #[test]
    fn test_couple_cells_self() {
        let mut cells = vec![GabrielCell::default()];

        // Should not couple a cell to itself
        couple_cells(&mut cells, 0, 0);
        assert!(cells[0].neighbors.is_empty());
    }

    #[test]
    fn test_neighbor_feedback() {
        let mut cells = vec![
            GabrielCell::new(1.0, 1.0, 1.0, 0.5),
            GabrielCell::new(2.0, 1.0, 1.0, 0.5),
            GabrielCell::new(3.0, 1.0, 1.0, 0.5),
        ];

        // Cell 0 has neighbors 1 and 2
        couple_cells(&mut cells, 0, 1);
        couple_cells(&mut cells, 0, 2);

        let avg_neighbor_output = (cells[1].output + cells[2].output) / 2.0;

        neighbor_feedback(&mut cells, 0);

        // Cell 0 should have been adjusted based on average neighbor output
        // The exact values depend on the feedback function, but we can check
        // that the cell was updated
        assert_ne!(cells[0].psi, 1.0);
    }

    #[test]
    fn test_neighbor_feedback_no_neighbors() {
        let mut cells = vec![GabrielCell::new(1.0, 1.0, 1.0, 0.5)];

        let initial_psi = cells[0].psi;
        neighbor_feedback(&mut cells, 0);

        // Should not change if no neighbors
        assert_eq!(cells[0].psi, initial_psi);
    }

    #[test]
    fn test_neighbor_feedback_invalid_index() {
        let mut cells = vec![GabrielCell::default()];

        // Should not panic with invalid index
        neighbor_feedback(&mut cells, 5);
    }
}
