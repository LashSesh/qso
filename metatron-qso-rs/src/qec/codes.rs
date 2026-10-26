//! Memory experiments: repetition and rotated surface codes with
//! circuit-level noise.

use super::circuit::{DetectorCircuit, Noise};
use crate::error::{QsoError, Result};

/// Largest code distance the generators build.
pub const MAX_DISTANCE: usize = 101;
/// Most rounds of syndrome extraction.
pub const MAX_ROUNDS: usize = 10_000;

/// Circuit-level noise, as in the standard depolarizing model: each
/// probability may be zero.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CircuitNoise {
    /// Depolarizing noise after each single-qubit gate.
    pub after_gate1: f64,
    /// Two-qubit depolarizing noise after each CNOT.
    pub after_gate2: f64,
    /// Flip probability of each measurement result.
    pub measurement: f64,
    /// X error after each reset.
    pub after_reset: f64,
    /// Depolarizing noise on the data qubits before each round.
    pub before_round: f64,
}

impl CircuitNoise {
    /// Every kind of noise with the same probability `p`.
    pub fn uniform(p: f64) -> Self {
        CircuitNoise {
            after_gate1: p,
            after_gate2: p,
            measurement: p,
            after_reset: p,
            before_round: p,
        }
    }

    /// Only depolarizing noise on the data qubits before each round, with
    /// perfect gates and measurements ("code capacity" for one round).
    pub fn data_only(p: f64) -> Self {
        CircuitNoise {
            before_round: p,
            ..Default::default()
        }
    }
}

fn check(distance: usize, rounds: usize, odd: bool) -> Result<()> {
    if !(2..=MAX_DISTANCE).contains(&distance) || (odd && distance.is_multiple_of(2)) {
        return Err(QsoError::invalid(
            "distance",
            format!(
                "the distance must be {} between 2 and {MAX_DISTANCE}",
                if odd { "odd and" } else { "" }
            ),
        ));
    }
    if rounds == 0 || rounds > MAX_ROUNDS {
        return Err(QsoError::invalid(
            "rounds",
            format!("between 1 and {MAX_ROUNDS} rounds of syndrome extraction"),
        ));
    }
    Ok(())
}

/// A builder that adds the noise of [`CircuitNoise`] around operations.
struct Noisy<'a> {
    c: DetectorCircuit,
    noise: &'a CircuitNoise,
}

impl Noisy<'_> {
    fn reset(&mut self, qs: &[usize]) -> Result<()> {
        for &q in qs {
            self.c.reset(q)?;
        }
        self.c.noise(Noise::XError(self.noise.after_reset), qs)?;
        Ok(())
    }

    fn h(&mut self, qs: &[usize]) -> Result<()> {
        for &q in qs {
            self.c.h(q)?;
        }
        self.c
            .noise(Noise::Depolarize1(self.noise.after_gate1), qs)?;
        Ok(())
    }

    /// CNOTs on disjoint pairs (control, target).
    fn cx(&mut self, pairs: &[(usize, usize)]) -> Result<()> {
        let mut flat = Vec::with_capacity(2 * pairs.len());
        for &(a, b) in pairs {
            self.c.cx(a, b)?;
            flat.extend([a, b]);
        }
        self.c
            .noise(Noise::Depolarize2(self.noise.after_gate2), &flat)?;
        Ok(())
    }

    fn measure(&mut self, qs: &[usize]) -> Result<Vec<usize>> {
        qs.iter()
            .map(|&q| self.c.measure(q, self.noise.measurement))
            .collect()
    }
}

/// A repetition-code memory experiment protecting `|0⟩` against bit
/// flips: `distance` data qubits (`0..d`) and `d − 1` ancillas (`d..2d−1`)
/// measuring `Z_i Z_{i+1}` for `rounds` rounds, then every data qubit.
///
/// Detectors compare each parity with its previous round (the first round
/// with 0, the last with the final data measurements). Observable 0 is the
/// final value of data qubit 0.
pub fn repetition_code_memory(
    distance: usize,
    rounds: usize,
    noise: &CircuitNoise,
) -> Result<DetectorCircuit> {
    check(distance, rounds, false)?;
    let d = distance;
    let data: Vec<usize> = (0..d).collect();
    let anc: Vec<usize> = (d..2 * d - 1).collect();
    let mut b = Noisy {
        c: DetectorCircuit::new(2 * d - 1)?,
        noise,
    };
    b.reset(&(0..2 * d - 1).collect::<Vec<_>>())?;
    let mut previous: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        b.c.noise(Noise::Depolarize1(noise.before_round), &data)?;
        b.cx(&(0..d - 1).map(|i| (data[i], anc[i])).collect::<Vec<_>>())?;
        b.cx(&(0..d - 1)
            .map(|i| (data[i + 1], anc[i]))
            .collect::<Vec<_>>())?;
        let m = b.measure(&anc)?;
        for (i, &r) in m.iter().enumerate() {
            match &previous {
                None => b.c.detector(&[r])?,
                Some(p) => b.c.detector(&[r, p[i]])?,
            };
        }
        b.reset(&anc)?;
        previous = Some(m);
    }
    let last = previous.expect("at least one round");
    let m = b.measure(&data)?;
    for i in 0..d - 1 {
        b.c.detector(&[last[i], m[i], m[i + 1]])?;
    }
    b.c.observable(0, &[m[0]])?;
    Ok(b.c)
}

/// A rotated surface-code memory experiment protecting `|0⟩_L`: `d × d`
/// data qubits (row-major, `0..d²`) and `d² − 1` ancillas, `rounds`
/// rounds of syndrome extraction with the hook-safe CNOT order, then every
/// data qubit measured.
///
/// Detectors: every Z-type stabilizer, compared with its previous round
/// (the first round with +1, the last with the final data measurements).
/// The X-type stabilizers are measured too but not used, so the model is a
/// matching graph for bit flips. Observable 0 is the logical Z, the
/// product of Z on the first row.
pub fn surface_code_memory(
    distance: usize,
    rounds: usize,
    noise: &CircuitNoise,
) -> Result<DetectorCircuit> {
    check(distance, rounds, true)?;
    let d = distance as i64;
    let data = |r: i64, c: i64| (r * d + c) as usize;
    // Plaquette (i, j) covers data (i, j), (i, j+1), (i+1, j), (i+1, j+1)
    // where they exist; X-type when i + j is even. Interior plaquettes, X
    // plaquettes on the top and bottom edges, Z plaquettes on the left and
    // right edges.
    let mut plaquettes: Vec<(i64, i64, bool)> = Vec::new(); // (i, j, is_x)
    for i in -1..d {
        for j in -1..d {
            let is_x = (i + j).rem_euclid(2) == 0;
            let interior = (0..d - 1).contains(&i) && (0..d - 1).contains(&j);
            let top_bottom = (i == -1 || i == d - 1) && (0..d - 1).contains(&j) && is_x;
            let left_right = (j == -1 || j == d - 1) && (0..d - 1).contains(&i) && !is_x;
            if interior || top_bottom || left_right {
                plaquettes.push((i, j, is_x));
            }
        }
    }
    let n_data = (d * d) as usize;
    let ancilla = |k: usize| n_data + k;
    let n = n_data + plaquettes.len();
    // CNOT layers: corner offsets (row, column) of the data qubit; X
    // ancillas go NW, NE, SW, SE, Z ancillas NW, SW, NE, SE, so that hook
    // errors run perpendicular to the logical operators.
    let order_x = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let order_z = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let mut b = Noisy {
        c: DetectorCircuit::new(n)?,
        noise,
    };
    let all: Vec<usize> = (0..n).collect();
    let data_qubits: Vec<usize> = (0..n_data).collect();
    let x_anc: Vec<usize> = (0..plaquettes.len())
        .filter(|&k| plaquettes[k].2)
        .map(ancilla)
        .collect();
    let all_anc: Vec<usize> = (0..plaquettes.len()).map(ancilla).collect();
    b.reset(&all)?;
    let mut previous: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        b.c.noise(Noise::Depolarize1(noise.before_round), &data_qubits)?;
        b.h(&x_anc)?;
        for layer in 0..4 {
            let mut pairs = Vec::new();
            for (k, &(i, j, is_x)) in plaquettes.iter().enumerate() {
                let (dr, dc) = if is_x { order_x[layer] } else { order_z[layer] };
                let (r, c) = (i + dr, j + dc);
                if !(0..d).contains(&r) || !(0..d).contains(&c) {
                    continue;
                }
                let q = data(r, c);
                // X ancillas control the data; Z ancillas are targets.
                pairs.push(if is_x {
                    (ancilla(k), q)
                } else {
                    (q, ancilla(k))
                });
            }
            b.cx(&pairs)?;
        }
        b.h(&x_anc)?;
        let m = b.measure(&all_anc)?;
        for (k, &(_, _, is_x)) in plaquettes.iter().enumerate() {
            if is_x {
                continue;
            }
            match &previous {
                None => b.c.detector(&[m[k]])?,
                Some(p) => b.c.detector(&[m[k], p[k]])?,
            };
        }
        b.reset(&all_anc)?;
        previous = Some(m);
    }
    let last = previous.expect("at least one round");
    let m = b.measure(&data_qubits)?;
    for (k, &(i, j, is_x)) in plaquettes.iter().enumerate() {
        if is_x {
            continue;
        }
        let mut recs = vec![last[k]];
        for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let (r, c) = (i + dr, j + dc);
            if (0..d).contains(&r) && (0..d).contains(&c) {
                recs.push(m[data(r, c)]);
            }
        }
        b.c.detector(&recs)?;
    }
    let first_row: Vec<usize> = (0..d).map(|c| m[data(0, c)]).collect();
    b.c.observable(0, &first_row)?;
    Ok(b.c)
}
