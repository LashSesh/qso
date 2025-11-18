//! Configuration management

use figment::{
    providers::{Env, Format, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// HTTP server configuration
    pub server: ServerConfig,
    /// Static files directory
    pub static_dir: String,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
}

impl Config {
    /// Load configuration from file and environment
    pub fn load() -> Result<Self, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::file("metatron_telemetry.toml").nested())
            .merge(Env::prefixed("METATRON_").split("__"))
            .extract()
            .map_err(Box::new)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8080,
            },
            static_dir: "metatron_telemetry/static".to_string(),
        }
    }
}
