/*!
 * Dataclass representations of the SPEC-002 blueprint.
 */

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Metadata describing the blueprint specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Spec {
    pub id: String,
    pub title: String,
    pub version: String,
    pub date: String,
    #[serde(default)]
    pub owners: Vec<String>,
    #[serde(default)]
    pub goals: Vec<String>,
}

impl Spec {
    pub fn from_dict(data: &HashMap<String, Value>) -> Self {
        Self {
            id: data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            title: data
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            version: data
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            date: data
                .get("date")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            owners: data
                .get("owners")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            goals: data
                .get("goals")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

/// Component entry from the blueprint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Component {
    pub name: String,
    #[serde(rename = "type")]
    pub component_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deps: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responsibilities: Option<Vec<String>>,
    #[serde(flatten)]
    pub extras: HashMap<String, Value>,
}

impl Component {
    pub fn from_dict(data: &HashMap<String, Value>) -> Self {
        let known_keys = vec!["name", "type", "deps", "responsibilities"];
        let mut extras = HashMap::new();
        for (key, value) in data.iter() {
            if !known_keys.contains(&key.as_str()) {
                extras.insert(key.clone(), value.clone());
            }
        }

        let deps = data.get("deps").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });

        let responsibilities = data
            .get("responsibilities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });

        Self {
            name: data
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            component_type: data
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            deps,
            responsibilities,
            extras,
        }
    }
}

/// API surface description
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct API {
    pub rest: Vec<String>,
    pub grpc: HashMap<String, Value>,
}

impl API {
    pub fn from_dict(data: &HashMap<String, Value>) -> Self {
        let rest = data
            .get("rest")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let grpc = data
            .get("grpc")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Self { rest, grpc }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Storage {
    pub fs_root: String,
    pub s3: HashMap<String, Value>,
    pub layout: HashMap<String, Value>,
}

impl Storage {
    pub fn from_dict(data: &HashMap<String, Value>) -> Self {
        let fs_root = data
            .get("fs_root")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let s3 = data
            .get("s3")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let layout = data
            .get("layout")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Self {
            fs_root,
            s3,
            layout,
        }
    }
}

/// Top-level blueprint model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Blueprint {
    pub spec: Spec,
    pub priorities: HashMap<String, Vec<String>>,
    pub components: Vec<Component>,
    pub storage: Storage,
    pub distance: HashMap<String, Value>,
    pub schemas: HashMap<String, Value>,
    pub api: API,
    pub index_backends: HashMap<String, Value>,
    pub consistency: HashMap<String, Value>,
    pub merkaba_gate: HashMap<String, Value>,
    pub workflows: HashMap<String, Value>,
    pub security: HashMap<String, Value>,
    pub observability: HashMap<String, Value>,
    pub config: HashMap<String, Value>,
    #[serde(flatten)]
    pub extras: HashMap<String, Value>,
}

impl Blueprint {
    pub fn from_dict(data: &HashMap<String, Value>) -> Self {
        let known_keys = vec![
            "spec",
            "priorities",
            "components",
            "storage",
            "distance",
            "schemas",
            "api",
            "index_backends",
            "consistency",
            "merkaba_gate",
            "workflows",
            "security",
            "observability",
            "config",
        ];

        let mut extras = HashMap::new();
        for (key, value) in data.iter() {
            if !known_keys.contains(&key.as_str()) {
                extras.insert(key.clone(), value.clone());
            }
        }

        let spec = data
            .get("spec")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let map: HashMap<String, Value> =
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Spec::from_dict(&map)
            })
            .unwrap_or_else(|| Spec::from_dict(&HashMap::new()));

        let priorities = data
            .get("priorities")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| {
                        let vec = v
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        (k.clone(), vec)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let components = data
            .get("components")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_object())
                    .map(|obj| {
                        let map: HashMap<String, Value> =
                            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        Component::from_dict(&map)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let storage = data
            .get("storage")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let map: HashMap<String, Value> =
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Storage::from_dict(&map)
            })
            .unwrap_or_else(|| Storage::from_dict(&HashMap::new()));

        let api = data
            .get("api")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let map: HashMap<String, Value> =
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                API::from_dict(&map)
            })
            .unwrap_or_else(|| API::from_dict(&HashMap::new()));

        // Helper function to extract HashMap<String, Value>
        let extract_map = |key: &str| -> HashMap<String, Value> {
            data.get(key)
                .and_then(|v| v.as_object())
                .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default()
        };

        Self {
            spec,
            priorities,
            components,
            storage,
            distance: extract_map("distance"),
            schemas: extract_map("schemas"),
            api,
            index_backends: extract_map("index_backends"),
            consistency: extract_map("consistency"),
            merkaba_gate: extract_map("merkaba_gate"),
            workflows: extract_map("workflows"),
            security: extract_map("security"),
            observability: extract_map("observability"),
            config: extract_map("config"),
            extras,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_from_dict() {
        let mut data = HashMap::new();
        data.insert("id".to_string(), Value::String("SPEC-002".to_string()));
        data.insert("title".to_string(), Value::String("Test Spec".to_string()));
        data.insert("version".to_string(), Value::String("1.0".to_string()));
        data.insert("date".to_string(), Value::String("2025-01-01".to_string()));
        data.insert(
            "owners".to_string(),
            serde_json::json!(["owner1", "owner2"]),
        );
        data.insert("goals".to_string(), serde_json::json!(["goal1", "goal2"]));

        let spec = Spec::from_dict(&data);
        assert_eq!(spec.id, "SPEC-002");
        assert_eq!(spec.title, "Test Spec");
        assert_eq!(spec.version, "1.0");
        assert_eq!(spec.date, "2025-01-01");
        assert_eq!(spec.owners, vec!["owner1", "owner2"]);
        assert_eq!(spec.goals, vec!["goal1", "goal2"]);
    }

    #[test]
    fn test_component_from_dict() {
        let mut data = HashMap::new();
        data.insert(
            "name".to_string(),
            Value::String("test-component".to_string()),
        );
        data.insert("type".to_string(), Value::String("service".to_string()));
        data.insert("deps".to_string(), serde_json::json!(["dep1", "dep2"]));
        data.insert(
            "custom_field".to_string(),
            Value::String("custom_value".to_string()),
        );

        let component = Component::from_dict(&data);
        assert_eq!(component.name, "test-component");
        assert_eq!(component.component_type, "service");
        assert_eq!(
            component.deps,
            Some(vec!["dep1".to_string(), "dep2".to_string()])
        );
        assert_eq!(
            component.extras.get("custom_field").unwrap(),
            "custom_value"
        );
    }

    #[test]
    fn test_api_from_dict() {
        let mut data = HashMap::new();
        data.insert("rest".to_string(), serde_json::json!(["/v1/endpoint"]));
        data.insert(
            "grpc".to_string(),
            serde_json::json!({"service": "TestService"}),
        );

        let api = API::from_dict(&data);
        assert_eq!(api.rest, vec!["/v1/endpoint"]);
        assert!(api.grpc.contains_key("service"));
    }

    #[test]
    fn test_storage_from_dict() {
        let mut data = HashMap::new();
        data.insert("fs_root".to_string(), Value::String("/data".to_string()));
        data.insert("s3".to_string(), serde_json::json!({"bucket": "my-bucket"}));
        data.insert(
            "layout".to_string(),
            serde_json::json!({"type": "versioned"}),
        );

        let storage = Storage::from_dict(&data);
        assert_eq!(storage.fs_root, "/data");
        assert!(storage.s3.contains_key("bucket"));
        assert!(storage.layout.contains_key("type"));
    }

    #[test]
    fn test_blueprint_from_dict() {
        let data: HashMap<String, Value> = serde_json::from_value(serde_json::json!({
            "spec": {
                "id": "SPEC-002",
                "title": "Test",
                "version": "1.0",
                "date": "2025-01-01"
            },
            "priorities": {
                "must": ["feature1"],
                "should": ["feature2"],
                "could": ["feature3"]
            },
            "components": [],
            "storage": {
                "fs_root": "/data",
                "s3": {},
                "layout": {}
            },
            "distance": {},
            "schemas": {},
            "api": {
                "rest": [],
                "grpc": {}
            },
            "index_backends": {},
            "consistency": {},
            "merkaba_gate": {},
            "workflows": {},
            "security": {},
            "observability": {},
            "config": {}
        }))
        .unwrap();

        let blueprint = Blueprint::from_dict(&data);
        assert_eq!(blueprint.spec.id, "SPEC-002");
        assert_eq!(blueprint.priorities.get("must").unwrap(), &vec!["feature1"]);
        assert_eq!(blueprint.storage.fs_root, "/data");
    }
}
