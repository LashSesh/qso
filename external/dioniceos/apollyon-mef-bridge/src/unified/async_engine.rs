//! Async version of the Unified Cognitive Engine
//!
//! Provides asynchronous processing capabilities for the cognitive engine,
//! enabling parallel batch processing and non-blocking operations.

use super::cognitive_engine::{CognitiveError, UnifiedCognitiveEngine};
use super::types::{BatchResult, CognitiveInput, CognitiveOutput};
use tokio::task;

/// Async wrapper around UnifiedCognitiveEngine
///
/// This provides asynchronous versions of the processing methods,
/// allowing for concurrent processing and integration with async runtimes.
///
/// # Examples
/// ```ignore
/// use tokio;
///
/// #[tokio::main]
/// async fn main() {
///     let engine = AsyncUnifiedCognitiveEngine::new();
///     let result = engine.process_async(input).await;
/// }
/// ```
pub struct AsyncUnifiedCognitiveEngine {
    engine: UnifiedCognitiveEngine,
}

impl AsyncUnifiedCognitiveEngine {
    /// Create a new async cognitive engine
    pub fn new() -> Self {
        Self {
            engine: UnifiedCognitiveEngine::new(),
        }
    }

    /// Create from an existing synchronous engine
    pub fn from_engine(engine: UnifiedCognitiveEngine) -> Self {
        Self { engine }
    }

    /// Get a reference to the underlying synchronous engine
    pub fn inner(&self) -> &UnifiedCognitiveEngine {
        &self.engine
    }

    /// Get a mutable reference to the underlying synchronous engine
    pub fn inner_mut(&mut self) -> &mut UnifiedCognitiveEngine {
        &mut self.engine
    }

    /// Process input asynchronously
    ///
    /// # Arguments
    /// * `input` - Cognitive input to process
    ///
    /// # Returns
    /// Result containing CognitiveOutput or error
    ///
    /// # Performance
    /// Runs the processing on a blocking task pool to avoid blocking
    /// the async runtime.
    pub async fn process_async(
        &mut self,
        input: CognitiveInput,
    ) -> Result<CognitiveOutput, CognitiveError> {
        // Clone the engine for the blocking task
        // Note: This is a simplified approach. In production, you might want
        // to use a different strategy (e.g., channels, Arc<Mutex<>>)
        let mut engine = self.engine.clone();

        task::spawn_blocking(move || engine.process(input))
            .await
            .map_err(|e| CognitiveError::IntegrationError(format!("Task join error: {}", e)))?
    }

    /// Process multiple inputs in parallel batches
    ///
    /// # Arguments
    /// * `inputs` - Vector of cognitive inputs to process
    /// * `parallelism` - Maximum number of concurrent tasks (default: CPU cores)
    ///
    /// # Returns
    /// BatchResult containing successful outputs and failures
    ///
    /// # Performance
    /// Processes inputs in parallel using Tokio's task pool.
    /// Each input is processed independently on a blocking task.
    ///
    /// # Examples
    /// ```ignore
    /// let inputs = vec![input1, input2, input3];
    /// let result = engine.process_batch_parallel(inputs, None).await;
    /// println!("Success rate: {:.1}%", result.success_rate());
    /// ```
    pub async fn process_batch_parallel(
        &self,
        inputs: Vec<CognitiveInput>,
        parallelism: Option<usize>,
    ) -> BatchResult {
        let start_time = std::time::Instant::now();

        // Determine parallelism level
        let max_parallel = parallelism.unwrap_or_else(num_cpus::get);

        // Process inputs in chunks to respect parallelism limit
        let mut successes = Vec::new();
        let mut failures = Vec::new();

        for chunk in inputs.chunks(max_parallel) {
            let mut handles = Vec::new();

            for (local_idx, input) in chunk.iter().enumerate() {
                let engine = self.engine.clone();
                let input = input.clone();

                let handle = task::spawn_blocking(move || {
                    let mut eng = engine;
                    eng.process(input)
                });

                handles.push((local_idx, handle));
            }

            // Await all tasks in this chunk
            for (_local_idx, handle) in handles {
                let global_idx = successes.len() + failures.len();
                match handle.await {
                    Ok(Ok(output)) => successes.push(output),
                    Ok(Err(e)) => failures.push((global_idx, e.to_string())),
                    Err(e) => failures.push((global_idx, format!("Task join error: {}", e))),
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        BatchResult::new(successes, failures, total_time)
    }

    /// Process batch sequentially in async context
    ///
    /// Similar to the sync version but runs in async context
    pub async fn process_batch_async(&mut self, inputs: Vec<CognitiveInput>) -> BatchResult {
        let start_time = std::time::Instant::now();
        let mut successes = Vec::new();
        let mut failures = Vec::new();

        for (index, input) in inputs.into_iter().enumerate() {
            match self.process_async(input).await {
                Ok(output) => successes.push(output),
                Err(e) => failures.push((index, e.to_string())),
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        BatchResult::new(successes, failures, total_time)
    }
}

impl Default for AsyncUnifiedCognitiveEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Note: We can't derive Clone for UnifiedCognitiveEngine because Box<dyn ResonanceField>
// doesn't implement Clone. We'll need a custom implementation.
impl Clone for UnifiedCognitiveEngine {
    fn clone(&self) -> Self {
        // For cloning, we create a new engine with the same configuration
        // but a new (default) resonance field
        Self::new_with_config(self.gate_config().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_5d::{State5D, SystemParameters};

    #[tokio::test]
    async fn test_async_process() {
        let mut engine = AsyncUnifiedCognitiveEngine::new();

        let input = CognitiveInput {
            initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
            parameters: SystemParameters::default(),
            t_final: 0.5,
            tic_id: "TIC-ASYNC-001".to_string(),
            seed: "async_test".to_string(),
            seed_path: "MEF/test/async/0001".to_string(),
        };

        let result = engine.process_async(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.trajectory.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_batch_processing() {
        let engine = AsyncUnifiedCognitiveEngine::new();

        let inputs = vec![
            CognitiveInput {
                initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
                parameters: SystemParameters::default(),
                t_final: 0.5,
                tic_id: "TIC-PARALLEL-1".to_string(),
                seed: "parallel_1".to_string(),
                seed_path: "MEF/test/parallel/0001".to_string(),
            },
            CognitiveInput {
                initial_state: State5D::new(0.8, 0.4, 0.2, 0.1, 0.05),
                parameters: SystemParameters::default(),
                t_final: 0.3,
                tic_id: "TIC-PARALLEL-2".to_string(),
                seed: "parallel_2".to_string(),
                seed_path: "MEF/test/parallel/0002".to_string(),
            },
            CognitiveInput {
                initial_state: State5D::new(1.2, 0.6, 0.4, 0.3, 0.2),
                parameters: SystemParameters::default(),
                t_final: 0.7,
                tic_id: "TIC-PARALLEL-3".to_string(),
                seed: "parallel_3".to_string(),
                seed_path: "MEF/test/parallel/0003".to_string(),
            },
        ];

        let batch_result = engine.process_batch_parallel(inputs, Some(2)).await;

        assert_eq!(batch_result.success_count(), 3);
        assert_eq!(batch_result.failure_count(), 0);
        assert!(batch_result.all_succeeded());
    }

    #[tokio::test]
    async fn test_sequential_async_batch() {
        let mut engine = AsyncUnifiedCognitiveEngine::new();

        let inputs = vec![
            CognitiveInput {
                initial_state: State5D::new(1.0, 0.0, 0.0, 0.0, 0.0),
                parameters: SystemParameters::default(),
                t_final: 0.1,
                tic_id: "TIC-SEQ-1".to_string(),
                seed: "seq_1".to_string(),
                seed_path: "MEF/test/seq/0001".to_string(),
            },
            CognitiveInput {
                initial_state: State5D::new(0.5, 0.0, 0.0, 0.0, 0.0),
                parameters: SystemParameters::default(),
                t_final: 0.1,
                tic_id: "TIC-SEQ-2".to_string(),
                seed: "seq_2".to_string(),
                seed_path: "MEF/test/seq/0002".to_string(),
            },
        ];

        let batch_result = engine.process_batch_async(inputs).await;

        assert_eq!(batch_result.total_count(), 2);
        assert!(batch_result.all_succeeded());
    }

    #[tokio::test]
    async fn test_from_engine() {
        let sync_engine = UnifiedCognitiveEngine::new();
        let mut async_engine = AsyncUnifiedCognitiveEngine::from_engine(sync_engine);

        let input = CognitiveInput {
            initial_state: State5D::new(1.0, 0.0, 0.0, 0.0, 0.0),
            parameters: SystemParameters::default(),
            t_final: 0.2,
            tic_id: "TIC-FROM".to_string(),
            seed: "from_test".to_string(),
            seed_path: "MEF/test/from/0001".to_string(),
        };

        let result = async_engine.process_async(input).await;
        assert!(result.is_ok());
    }
}
