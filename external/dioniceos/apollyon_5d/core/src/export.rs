//! Data Export Functionality
//!
//! Section 10 - CSV and JSON export formats.

use crate::state::State5D;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

/// Trajectory data for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub times: Vec<f64>,
    pub states: Vec<State5D>,
}

impl Trajectory {
    /// Create a new trajectory from time-state pairs
    pub fn from_pairs(pairs: Vec<(f64, State5D)>) -> Self {
        let (times, states) = pairs.into_iter().unzip();
        Trajectory { times, states }
    }

    /// Export trajectory to CSV file
    ///
    /// Format (Section 10.2):
    /// time,sigma_1,sigma_2,sigma_3,sigma_4,sigma_5
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = csv::Writer::from_writer(file);

        // Write header
        writer.write_record(&[
            "time", "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5",
        ])?;

        // Write data rows
        for (t, state) in self.times.iter().zip(self.states.iter()) {
            writer.write_record(&[
                t.to_string(),
                state.get(0).to_string(),
                state.get(1).to_string(),
                state.get(2).to_string(),
                state.get(3).to_string(),
                state.get(4).to_string(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Export trajectory to JSON file
    ///
    /// Format (Section 10.3):
    /// {"times": [...], "states": [[...], ...]}
    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = std::io::BufWriter::new(file);

        // Convert states to arrays for JSON
        let states_as_arrays: Vec<[f64; 5]> = self.states.iter().map(|s| s.to_array()).collect();

        let export_data = serde_json::json!({
            "times": &self.times,
            "states": states_as_arrays,
        });

        serde_json::to_writer_pretty(writer, &export_data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_trajectory_creation() {
        let pairs = vec![
            (0.0, State5D::new(1.0, 2.0, 3.0, 4.0, 5.0)),
            (1.0, State5D::new(2.0, 3.0, 4.0, 5.0, 6.0)),
        ];

        let traj = Trajectory::from_pairs(pairs);
        assert_eq!(traj.times.len(), 2);
        assert_eq!(traj.states.len(), 2);
    }

    #[test]
    fn test_csv_export() {
        let pairs = vec![
            (0.0, State5D::new(1.0, 2.0, 3.0, 4.0, 5.0)),
            (1.0, State5D::new(2.0, 3.0, 4.0, 5.0, 6.0)),
        ];

        let traj = Trajectory::from_pairs(pairs);
        let path = "/tmp/test_trajectory.csv";

        traj.export_csv(path).unwrap();
        assert!(Path::new(path).exists());

        // Clean up
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_json_export() {
        let pairs = vec![
            (0.0, State5D::new(1.0, 2.0, 3.0, 4.0, 5.0)),
            (1.0, State5D::new(2.0, 3.0, 4.0, 5.0, 6.0)),
        ];

        let traj = Trajectory::from_pairs(pairs);
        let path = "/tmp/test_trajectory.json";

        traj.export_json(path).unwrap();
        assert!(Path::new(path).exists());

        // Clean up
        fs::remove_file(path).ok();
    }
}
