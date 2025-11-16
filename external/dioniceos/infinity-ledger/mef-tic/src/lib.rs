/*!
 * MEF-TIC - Temporal Information Crystal module
 * Crystallizes Solve-Coagula fixpoints into stable crystals
 */

pub mod crystallizer;

pub use crystallizer::{
    GateConfig, Invariants, MacroGating, MesoGating, MicroGating, MultiscaleGating, Proof,
    SigmaBar, TICConfig, TICCrystallizer, TIC,
};
