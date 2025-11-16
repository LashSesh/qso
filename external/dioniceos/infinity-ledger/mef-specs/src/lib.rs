/*!
 * MEF-Core Specs Module
 *
 * Provides blueprint loading and validation for SPEC-002 compliant configurations.
 */

pub mod blueprint_loader;
pub mod blueprint_models;

pub use blueprint_loader::{
    load_blueprint, BlueprintDocument, BlueprintSchemaError, BlueprintValidationError,
    REQUIRED_TOP_LEVEL_KEYS,
};
pub use blueprint_models::{Blueprint, Component, Spec, Storage, API};
