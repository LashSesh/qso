//! Canonical JSON serialization with stable key ordering and fixed precision

use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Convert a serializable value to canonical JSON
/// - Keys are sorted alphabetically
/// - Floating point numbers are rounded to 6 decimal places
/// - Output is deterministic and reproducible
pub fn canonical_json<T: Serialize>(value: &T) -> crate::Result<String> {
    let json_value = serde_json::to_value(value)?;
    let canonical = canonicalize_value(&json_value);
    Ok(serde_json::to_string(&canonical)?)
}

/// Recursively canonicalize a JSON value
fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut canonical_map = BTreeMap::new();
            for (k, v) in map {
                canonical_map.insert(k.clone(), canonicalize_value(v));
            }
            Value::Object(canonical_map.into_iter().collect())
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_value).collect()),
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                // Round to 6 decimal places for determinism
                let rounded = (f * 1_000_000.0).round() / 1_000_000.0;
                Value::Number(serde_json::Number::from_f64(rounded).unwrap_or(n.clone()))
            } else {
                value.clone()
            }
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_key_ordering() {
        let data = json!({
            "zebra": 1,
            "apple": 2,
            "monkey": 3
        });

        let canonical = canonical_json(&data).unwrap();
        // Keys should be in alphabetical order
        assert!(canonical.contains(r#""apple":2"#));
        let apple_pos = canonical.find("apple").unwrap();
        let monkey_pos = canonical.find("monkey").unwrap();
        let zebra_pos = canonical.find("zebra").unwrap();
        assert!(apple_pos < monkey_pos);
        assert!(monkey_pos < zebra_pos);
    }

    #[test]
    fn test_float_precision() {
        let data = json!({
            "value": 0.123456789
        });

        let canonical = canonical_json(&data).unwrap();
        // Should be rounded to 6 decimals
        assert!(canonical.contains("0.123457") || canonical.contains("0.123456"));
    }

    #[test]
    fn test_determinism() {
        let data = json!({
            "z": 1.111111111,
            "a": 2.222222222,
            "m": 3.333333333
        });

        let canonical1 = canonical_json(&data).unwrap();
        let canonical2 = canonical_json(&data).unwrap();
        assert_eq!(canonical1, canonical2);
    }
}
