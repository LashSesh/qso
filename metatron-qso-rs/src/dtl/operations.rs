use super::state::{DTLState, TripolarStateKind};

/// Dynamic Tripolar Logic operations implementing functional completeness.
pub struct DTLOperations;

impl DTLOperations {
    /// Tripolar conjunction x ∧_DTL y.
    pub fn and(x: &DTLState, y: &DTLState) -> DTLState {
        match (x.kind(), y.kind()) {
            (TripolarStateKind::L0, _) | (_, TripolarStateKind::L0) => DTLState::l0(),
            (TripolarStateKind::L1, TripolarStateKind::L1) => DTLState::l1(),
            _ => {
                let x_clone = x.clone();
                let y_clone = y.clone();
                DTLState::ld_from_function(move |t| x_clone.evaluate(t).min(y_clone.evaluate(t)))
            }
        }
    }

    /// Tripolar disjunction x ∨_DTL y.
    pub fn or(x: &DTLState, y: &DTLState) -> DTLState {
        match (x.kind(), y.kind()) {
            (TripolarStateKind::L1, _) | (_, TripolarStateKind::L1) => DTLState::l1(),
            (TripolarStateKind::L0, TripolarStateKind::L0) => DTLState::l0(),
            _ => {
                let x_clone = x.clone();
                let y_clone = y.clone();
                DTLState::ld_from_function(move |t| x_clone.evaluate(t).max(y_clone.evaluate(t)))
            }
        }
    }

    /// Tripolar negation ¬_DTL x.
    pub fn not(x: &DTLState) -> DTLState {
        match x.kind() {
            TripolarStateKind::L0 => DTLState::l1(),
            TripolarStateKind::L1 => DTLState::l0(),
            TripolarStateKind::Ld => {
                let x_clone = x.clone();
                DTLState::ld_from_function(move |t| 1.0 - x_clone.evaluate(t))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_reduction_matches_expectations() {
        let l0 = DTLState::l0();
        let l1 = DTLState::l1();
        assert_eq!(DTLOperations::and(&l0, &l1).kind(), TripolarStateKind::L0);
        assert_eq!(DTLOperations::or(&l0, &l1).kind(), TripolarStateKind::L1);
        assert_eq!(DTLOperations::not(&l0).kind(), TripolarStateKind::L1);
    }
}
