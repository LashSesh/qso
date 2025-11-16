use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum SeraphicValue {
    Text(String),
    Number(f64),
    Sequence(Vec<f64>),
}

impl From<&str> for SeraphicValue {
    fn from(value: &str) -> Self {
        SeraphicValue::Text(value.to_string())
    }
}

impl From<String> for SeraphicValue {
    fn from(value: String) -> Self {
        SeraphicValue::Text(value)
    }
}

impl From<f64> for SeraphicValue {
    fn from(value: f64) -> Self {
        SeraphicValue::Number(value)
    }
}

impl From<i32> for SeraphicValue {
    fn from(value: i32) -> Self {
        SeraphicValue::Number(value as f64)
    }
}

impl From<Vec<f64>> for SeraphicValue {
    fn from(value: Vec<f64>) -> Self {
        SeraphicValue::Sequence(value)
    }
}

impl From<Vec<i32>> for SeraphicValue {
    fn from(value: Vec<i32>) -> Self {
        SeraphicValue::Sequence(value.into_iter().map(|v| v as f64).collect())
    }
}

impl From<&[f64]> for SeraphicValue {
    fn from(value: &[f64]) -> Self {
        SeraphicValue::Sequence(value.to_vec())
    }
}

pub struct SeraphicFeedbackModule {
    filter: Arc<dyn Fn(&SeraphicValue) -> Vec<f64> + Send + Sync>,
}

impl std::fmt::Debug for SeraphicFeedbackModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeraphicFeedbackModule").finish()
    }
}

impl SeraphicFeedbackModule {
    pub fn new() -> Self {
        Self {
            filter: Arc::new(default_filter),
        }
    }

    pub fn with_filter<F>(filter: F) -> Self
    where
        F: Fn(&SeraphicValue) -> Vec<f64> + Send + Sync + 'static,
    {
        Self {
            filter: Arc::new(filter),
        }
    }

    pub fn map_inputs<V>(&self, inputs: impl IntoIterator<Item = V>) -> Vec<Vec<f64>>
    where
        V: Into<SeraphicValue>,
    {
        inputs
            .into_iter()
            .map(|item| (self.filter)(&item.into()))
            .collect()
    }
}

fn default_filter(value: &SeraphicValue) -> Vec<f64> {
    match value {
        SeraphicValue::Text(text) => {
            let mut base: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
            if base.is_empty() {
                base.push(0.0);
            }
            let mut result = vec![0.0; 5];
            for i in 0..5 {
                let mut acc = 0.0;
                for (idx, val) in base.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64 + 1.0) * idx as f64
                        / (base.len() as f64 + 5.0);
                    acc += val * angle.cos();
                }
                result[i] = acc;
            }
            normalize(result)
        }
        SeraphicValue::Number(num) => {
            let result = vec![*num; 5];
            normalize(result)
        }
        SeraphicValue::Sequence(seq) => {
            let mean = if seq.is_empty() {
                0.0
            } else {
                seq.iter().sum::<f64>() / seq.len() as f64
            };
            let result = vec![mean; 5];
            normalize(result)
        }
    }
}

fn normalize(mut vec: Vec<f64>) -> Vec<f64> {
    let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_mapping() {
        let module = SeraphicFeedbackModule::new();
        let res = module.map_inputs(["TEST"]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].len(), 5);
    }
}
