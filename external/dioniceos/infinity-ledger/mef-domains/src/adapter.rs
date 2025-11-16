/*!
 * Domain Adapters - Transform raw domain data into Resonits
 *
 * Provides trait-based adapter system for different data domains
 * (text, signal, etc.) with domain-specific feature extraction.
 */

use crate::resonit::{Resonit, Sigma};
use anyhow::Result;
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Abstract interface for domain-specific data adapters
///
/// Each domain implements this trait to transform raw data into Resonits
/// with domain-appropriate feature extraction and tripolar signatures.
pub trait DomainAdapter: Send + Sync {
    /// Transform raw domain data into a list of Resonits
    fn transform(&self, raw_data: &Value) -> Result<Vec<Resonit>>;

    /// Extract feature vector from raw data
    fn extract_features(&self, raw_data: &Value) -> Result<Vec<f64>>;

    /// Return domain identifier
    fn domain_name(&self) -> &str;
}

/// Adapter for text/NLP domain
///
/// Transforms text into Resonits by splitting into semantic units (sentences)
/// and calculating tripolar signatures from text features.
pub struct TextDomainAdapter;

impl Default for TextDomainAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextDomainAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Split text into sentences (simplified)
    fn split_sentences(text: &str) -> Vec<String> {
        text.replace(&['!', '?'][..], ".")
            .split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Calculate mean of first n values
    fn mean_first_n(values: &[f64], n: usize) -> f64 {
        let count = n.min(values.len());
        if count == 0 {
            return 0.0;
        }
        values.iter().take(count).sum::<f64>() / count as f64
    }

    /// Calculate standard deviation
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Calculate mean of absolute differences
    fn mean_abs_diff(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let diffs: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        diffs.iter().sum::<f64>() / diffs.len() as f64
    }
}

impl DomainAdapter for TextDomainAdapter {
    fn transform(&self, raw_data: &Value) -> Result<Vec<Resonit>> {
        // Convert to text
        let text = match raw_data {
            Value::String(s) => s.clone(),
            _ => raw_data.to_string().trim_matches('"').to_string(),
        };

        // Split into sentences
        let sentences = Self::split_sentences(&text);

        let mut resonits = Vec::new();
        for (i, sentence) in sentences.iter().enumerate() {
            // Extract features
            let features = self.extract_features(&Value::String(sentence.clone()))?;

            // Map features to tripolar signature
            let psi = Self::mean_first_n(&features, 3); // Activation from first features
            let rho = Self::std_dev(&features); // Coherence from variation
            let omega = Self::mean_abs_diff(&features); // Rhythm from changes

            let sigma = Sigma::new(psi, rho, omega);

            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("content".to_string(), Value::String(sentence.clone()));
            metadata.insert("position".to_string(), Value::Number(i.into()));

            let resonit = Resonit {
                id: Uuid::new_v4().to_string(),
                sigma,
                src: self.domain_name().to_string(),
                ts: Utc::now().timestamp(),
                coordinates: None,
                metadata,
            };

            resonits.push(resonit);
        }

        Ok(resonits)
    }

    fn extract_features(&self, raw_data: &Value) -> Result<Vec<f64>> {
        let text = match raw_data {
            Value::String(s) => s.clone(),
            _ => raw_data.to_string().trim_matches('"').to_string(),
        };

        let mut features = Vec::new();

        // Simple feature extraction
        features.push(text.len() as f64 / 100.0); // Length normalized

        // Unique characters (normalized by alphabet size)
        let unique_chars = text
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect::<std::collections::HashSet<_>>()
            .len();
        features.push(unique_chars as f64 / 26.0);

        // Word density (spaces per character)
        let space_count = text.matches(' ').count();
        features.push(space_count as f64 / (text.len() + 1) as f64);

        // Capital letters
        let capitals = text.chars().filter(|c| c.is_uppercase()).count();
        features.push(capitals as f64 / (text.len() + 1) as f64);

        // Digits
        let digits = text.chars().filter(|c| c.is_numeric()).count();
        features.push(digits as f64 / (text.len() + 1) as f64);

        // Pad to fixed size (10 features)
        while features.len() < 10 {
            features.push(0.0);
        }

        // Truncate to 10
        features.truncate(10);

        Ok(features)
    }

    fn domain_name(&self) -> &str {
        "text"
    }
}

/// Adapter for signal/time-series domain
///
/// Transforms signals into Resonits by windowing and calculating
/// tripolar signatures from time-domain and frequency-domain features.
pub struct SignalDomainAdapter {
    window_size: usize,
}

impl Default for SignalDomainAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalDomainAdapter {
    pub fn new() -> Self {
        Self { window_size: 100 }
    }

    pub fn with_window_size(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Convert JSON value to signal array
    fn extract_signal(raw_data: &Value) -> Result<Vec<f64>> {
        match raw_data {
            Value::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
                    _ => Ok(0.0),
                })
                .collect(),
            Value::Number(n) => Ok(vec![n.as_f64().unwrap_or(0.0)]),
            _ => Ok(vec![]),
        }
    }

    /// Calculate mean
    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate standard deviation
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = Self::mean(values);
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Simple FFT-like frequency analysis (dominant frequency)
    fn dominant_frequency(window: &[f64]) -> f64 {
        if window.len() < 2 {
            return 0.5;
        }

        // Simplified: use zero-crossing rate as frequency proxy
        let mut crossings = 0;
        let mean = Self::mean(window);
        for i in 0..window.len() - 1 {
            if (window[i] - mean) * (window[i + 1] - mean) < 0.0 {
                crossings += 1;
            }
        }

        // Normalize to [0, 1]
        (crossings as f64 / window.len() as f64).min(1.0)
    }
}

impl DomainAdapter for SignalDomainAdapter {
    fn transform(&self, raw_data: &Value) -> Result<Vec<Resonit>> {
        let signal = Self::extract_signal(raw_data)?;

        if signal.is_empty() {
            return Ok(vec![]);
        }

        // Segment signal into windows
        let window_size = self.window_size.min(signal.len());
        let stride = window_size / 2;

        let mut resonits = Vec::new();

        let mut i = 0;
        while i + window_size <= signal.len() {
            let window = &signal[i..i + window_size];
            let _features = self.extract_features(&Value::Array(
                window.iter().map(|&x| Value::from(x)).collect(),
            ))?;

            // Map to tripolar signature
            let psi = window.iter().map(|x| x.abs()).sum::<f64>() / window.len() as f64; // Amplitude
            let rho = 1.0 / (1.0 + Self::std_dev(window)); // Inverse variance for coherence
            let omega = Self::dominant_frequency(window); // Frequency content

            let sigma = Sigma::new(psi, rho, omega);

            // Create coordinates (position in signal)
            let coordinates = vec![i as f64 / signal.len() as f64, 0.0, 0.0];

            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("window_start".to_string(), Value::Number(i.into()));
            metadata.insert("window_size".to_string(), Value::Number(window_size.into()));

            let resonit = Resonit {
                id: Uuid::new_v4().to_string(),
                sigma,
                src: self.domain_name().to_string(),
                ts: Utc::now().timestamp(),
                coordinates: Some(coordinates),
                metadata,
            };

            resonits.push(resonit);
            i += stride;
        }

        Ok(resonits)
    }

    fn extract_features(&self, raw_data: &Value) -> Result<Vec<f64>> {
        let signal = Self::extract_signal(raw_data)?;

        let mut features = Vec::new();

        if signal.is_empty() {
            // Return zero features
            return Ok(vec![0.0; 10]);
        }

        // Time-domain features
        features.push(Self::mean(&signal));
        features.push(Self::std_dev(&signal));
        features.push(signal.iter().cloned().fold(f64::INFINITY, f64::min));
        features.push(signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        // Spectral features (simplified)
        if signal.len() > 1 {
            // Use variance as proxy for spectral content
            features.push(Self::std_dev(&signal));
            features.push(Self::dominant_frequency(&signal));
        } else {
            features.push(0.0);
            features.push(0.0);
        }

        // Pad to fixed size (10 features)
        while features.len() < 10 {
            features.push(0.0);
        }

        // Truncate to 10
        features.truncate(10);

        Ok(features)
    }

    fn domain_name(&self) -> &str {
        "signal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_adapter_transform() {
        let adapter = TextDomainAdapter::new();
        let data = Value::String("Hello world. This is a test.".to_string());

        let resonits = adapter.transform(&data).unwrap();

        assert_eq!(resonits.len(), 2); // Two sentences
        assert_eq!(resonits[0].src, "text");
        assert!(resonits[0].sigma.psi >= 0.0);
        assert!(resonits[0].sigma.rho >= 0.0);
        assert!(resonits[0].sigma.omega >= 0.0);
    }

    #[test]
    fn test_text_adapter_extract_features() {
        let adapter = TextDomainAdapter::new();
        let data = Value::String("Test".to_string());

        let features = adapter.extract_features(&data).unwrap();

        assert_eq!(features.len(), 10);
        assert!(features[0] > 0.0); // Length feature
    }

    #[test]
    fn test_text_adapter_domain_name() {
        let adapter = TextDomainAdapter::new();
        assert_eq!(adapter.domain_name(), "text");
    }

    #[test]
    fn test_signal_adapter_transform() {
        let adapter = SignalDomainAdapter::new();
        let data = Value::Array((0..150).map(|i| Value::from(i as f64)).collect());

        let resonits = adapter.transform(&data).unwrap();

        assert!(!resonits.is_empty());
        assert_eq!(resonits[0].src, "signal");
        assert!(resonits[0].coordinates.is_some());
    }

    #[test]
    fn test_signal_adapter_extract_features() {
        let adapter = SignalDomainAdapter::new();
        let data = Value::Array(vec![Value::from(1.0), Value::from(2.0), Value::from(3.0)]);

        let features = adapter.extract_features(&data).unwrap();

        assert_eq!(features.len(), 10);
        assert!(features[0] > 0.0); // Mean
    }

    #[test]
    fn test_signal_adapter_domain_name() {
        let adapter = SignalDomainAdapter::new();
        assert_eq!(adapter.domain_name(), "signal");
    }

    #[test]
    fn test_signal_adapter_empty() {
        let adapter = SignalDomainAdapter::new();
        let data = Value::Array(vec![]);

        let resonits = adapter.transform(&data).unwrap();
        assert_eq!(resonits.len(), 0);
    }

    #[test]
    fn test_text_adapter_empty() {
        let adapter = TextDomainAdapter::new();
        let data = Value::String("".to_string());

        let resonits = adapter.transform(&data).unwrap();
        assert_eq!(resonits.len(), 0);
    }

    #[test]
    fn test_text_split_sentences() {
        let sentences = TextDomainAdapter::split_sentences("One. Two! Three?");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "One");
        assert_eq!(sentences[1], "Two");
        assert_eq!(sentences[2], "Three");
    }
}
