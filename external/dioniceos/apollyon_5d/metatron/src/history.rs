use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct HistoryEvent {
    pub event_type: String,
    pub payload: serde_json::Value,
    pub info: Option<serde_json::Value>,
}

#[derive(Debug, Default)]
pub struct HistoryLogger {
    pub events: Vec<HistoryEvent>,
}

impl HistoryLogger {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn log<P, I>(&mut self, event_type: &str, payload: P, info: Option<I>)
    where
        P: serde::Serialize,
        I: serde::Serialize,
    {
        let payload = serde_json::to_value(payload).unwrap();
        let info = info.map(|value| serde_json::to_value(value).unwrap());
        self.events.push(HistoryEvent {
            event_type: event_type.to_string(),
            payload,
            info,
        });
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn filter(&self, event_type: &str) -> Vec<&HistoryEvent> {
        self.events
            .iter()
            .filter(|evt| evt.event_type == event_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logging() {
        let mut logger = HistoryLogger::new();
        logger.log("activation", serde_json::json!({"psi": 1.0}), None::<()>);
        logger.log("decision", serde_json::json!({"value": 1.0}), None::<()>);
        assert_eq!(logger.events.len(), 2);
        assert_eq!(logger.filter("activation").len(), 1);
        logger.clear();
        assert!(logger.events.is_empty());
    }
}
