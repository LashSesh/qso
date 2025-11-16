//! Quantum walk solvers, diagnostics, and high-end benchmarking utilities.

pub mod analysis;
pub mod continuous;
pub mod krylov;
pub mod scattering;

pub use analysis::{
    BenchmarkMetadata, ClassicalHittingMatrix, HittingTimeBenchmark, MixingTimeResult,
    QuantumHittingResult, QuantumWalkBenchmarkSuite, QuantumWalkBenchmarker,
};
pub use continuous::{ContinuousTimeQuantumWalk, SpectralPropagator};
pub use krylov::{KrylovEvolution, KrylovProjection, LanczosResult};
pub use scattering::{DensityOfStates, ScatteringAnalysis, ScatteringChannel};
