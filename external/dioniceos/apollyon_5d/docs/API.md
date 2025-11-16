# API Documentation - 5D System Framework

Complete API reference for all public types and functions.

## Module: `state`

### `State5D`

5-dimensional state vector σ ∈ ℝ⁵.

**Fields:**
- `data: SVector<f64, 5>` - Internal 5D vector storage

**Methods:**

#### `new(s1: f64, s2: f64, s3: f64, s4: f64, s5: f64) -> State5D`
Create state from five components.
Panics if any component is NaN or infinite.

#### `from_array(arr: [f64; 5]) -> State5D`
Create state from array.

#### `zero() -> State5D`
Create zero state (all components 0.0).

#### `get(&self, i: usize) -> f64`
Get component by index (0-4).

#### `set(&mut self, i: usize, value: f64) -> bool`
Set component. Returns false if value is not finite.

#### `is_valid(&self) -> bool`
Check if all components are finite.

#### `to_array(&self) -> [f64; 5]`
Convert to array.

#### `norm(&self) -> f64`
Compute Euclidean norm ‖σ‖₂.

#### `dot(&self, other: &State5D) -> f64`
Compute inner product ⟨σ, other⟩.

#### `add(&self, other: &State5D) -> State5D`
Element-wise addition.

#### `sub(&self, other: &State5D) -> State5D`
Element-wise subtraction.

#### `scale(&self, scalar: f64) -> State5D`
Scalar multiplication.

**Operators:**
- `s1 + s2` - Addition
- `s1 - s2` - Subtraction  
- `s * scalar` - Multiplication
- `scalar * s` - Multiplication

---

## Module: `coupling`

### `CouplingType`

Enum defining interaction types.

**Variants:**
- `Linear` - Cᵢⱼ · σⱼ
- `Quadratic` - Cᵢⱼ · σⱼ²
- `Product` - Cᵢⱼ · σᵢ · σⱼ
- `Sigmoid` - Cᵢⱼ · tanh(σⱼ)

**Methods:**

#### `apply(&self, si: f64, sj: f64, cij: f64) -> f64`
Apply coupling function to values.

#### `derivative_wrt_sj(&self, si: f64, sj: f64, cij: f64) -> f64`
Compute ∂/∂σⱼ of coupling term (for Jacobian).

#### `derivative_wrt_si(&self, si: f64, sj: f64, cij: f64) -> f64`
Compute ∂/∂σᵢ of coupling term (relevant for Product type).

### `CouplingMatrix`

5×5 coupling matrix with types.

**Fields:**
- `strengths: [[f64; 5]; 5]` - Coupling strengths Cᵢⱼ
- `types: [[CouplingType; 5]; 5]` - Coupling types τᵢⱼ

**Methods:**

#### `new(strengths: [[f64; 5]; 5]) -> CouplingMatrix`
Create with specified strengths (all linear coupling).

#### `zero() -> CouplingMatrix`
Create zero matrix (no interactions).

#### `identity() -> CouplingMatrix`
Create identity matrix.

#### `set(&mut self, i: usize, j: usize, strength: f64, coupling_type: CouplingType)`
Set coupling strength and type.

#### `get_strength(&self, i: usize, j: usize) -> f64`
Get coupling strength.

#### `get_type(&self, i: usize, j: usize) -> CouplingType`
Get coupling type.

#### `apply_to_variable(&self, i: usize, state: &State5D) -> f64`
Compute total coupling contribution to variable i.

---

## Module: `dynamics`

### `SystemParameters`

Parameters for system dynamics.

**Fields:**
- `intrinsic_rates: [f64; 5]` - Rates αᵢ
- `external_forcing: [f64; 5]` - Forcing fᵢ(t)

**Methods:**

#### `new(intrinsic_rates: [f64; 5], external_forcing: [f64; 5]) -> SystemParameters`
Create parameters.

#### `zero() -> SystemParameters`
Create zero parameters.

### `VectorField`

Vector field F(σ) defining system dynamics.

**Fields:**
- `coupling: CouplingMatrix`
- `parameters: SystemParameters`

**Methods:**

#### `new(coupling: CouplingMatrix, parameters: SystemParameters) -> VectorField`
Create vector field.

#### `from_coupling(coupling: CouplingMatrix) -> VectorField`
Create with zero parameters.

#### `evaluate(&self, state: &State5D) -> State5D`
Evaluate F(σ) at given state.

#### `jacobian(&self, state: &State5D) -> [[f64; 5]; 5]`
Compute Jacobian matrix Jᵢⱼ = ∂Fᵢ/∂σⱼ.

---

## Module: `integration`

### `TimeConfig`

Time discretization configuration.

**Fields:**
- `dt: f64` - Time step Δt
- `t0: f64` - Initial time
- `t_final: f64` - Final time

**Methods:**

#### `new(dt: f64, t0: f64, t_final: f64) -> TimeConfig`
Create time configuration.

#### `num_steps(&self) -> usize`
Get number of time steps.

#### `time_at_step(&self, n: usize) -> f64`
Get time at step n.

### `Integrator`

Numerical integrator using Heun's method.

**Fields:**
- `vector_field: VectorField`
- `time_config: TimeConfig`

**Methods:**

#### `new(vector_field: VectorField, time_config: TimeConfig) -> Integrator`
Create integrator.

#### `step(&self, state: &State5D) -> State5D`
Perform single Heun step.

#### `integrate(&self, initial_state: State5D) -> Vec<(f64, State5D)>`
Integrate from initial state, return trajectory with times.

#### `integrate_states(&self, initial_state: State5D) -> Vec<State5D>`
Integrate and return states only.

#### `integrate_final(&self, initial_state: State5D) -> State5D`
Integrate and return only final state.

---

## Module: `stability`

### `FixedPointFinder`

Find fixed points where F(σ*) = 0.

**Methods:**

#### `new(vector_field: VectorField, tolerance: f64) -> FixedPointFinder`
Create finder with tolerance.

#### `is_fixed_point(&self, state: &State5D) -> bool`
Check if state is fixed point.

### `StabilityAnalyzer`

Analyze stability of fixed points.

**Methods:**

#### `eigenvalues(jacobian: &[[f64; 5]; 5]) -> Vec<f64>`
Compute real parts of eigenvalues (sorted descending).

#### `classify_stability(eigenvalues: &[f64]) -> StabilityType`
Determine stability type from eigenvalues.

### `StabilityType`

Enum for stability classification.

**Variants:**
- `Stable` - All ℜ(λᵢ) < 0
- `Unstable` - Any ℜ(λᵢ) > 0
- `Marginal` - Eigenvalues on imaginary axis

---

## Module: `projection`

### `Point2D`

2D point for visualization.

**Fields:**
- `x: f64`
- `y: f64`

### `Point3D`

3D point for visualization.

**Fields:**
- `x: f64`
- `y: f64`
- `z: f64`

### `ProjectionMethod`

Enum defining projection types.

**Variants:**
- `Orthogonal(usize, usize)` - Project onto two dimensions
- `Isometric` - Isometric projection
- `PCA` - Principal component analysis

### `Projector`

Projects 5D states to 2D.

**Methods:**

#### `new(method: ProjectionMethod) -> Projector`
Create projector.

#### `orthogonal(dim1: usize, dim2: usize) -> Projector`
Create orthogonal projector.

#### `isometric() -> Projector`
Create isometric projector.

#### `fit_pca(&mut self, states: &[State5D])`
Fit PCA to data.

#### `project(&self, state: &State5D) -> Point2D`
Project single state.

#### `project_many(&self, states: &[State5D]) -> Vec<Point2D>`
Project multiple states.

---

## Module: `template`

### `Template`

Pre-configured system for specific domain.

**Fields:**
- `name: String`
- `description: String`
- `variable_names: [String; 5]`
- `coupling_matrix: CouplingMatrix`
- `parameters: SystemParameters`

**Methods:**

#### `sir_model(beta: f64, gamma: f64, mu: f64) -> Template`
Create SIR epidemiological model.
- β: transmission rate
- γ: recovery rate
- μ: death rate

#### `financial_market(volatility: f64, momentum: f64, risk_aversion: f64) -> Template`
Create financial market model.

#### `predator_prey(growth_rate: f64, predation_rate: f64, death_rate: f64) -> Template`
Create predator-prey ecosystem model.

#### `to_vector_field(&self) -> VectorField`
Convert template to vector field.

---

## Module: `export`

### `Trajectory`

Trajectory data for export.

**Fields:**
- `times: Vec<f64>`
- `states: Vec<State5D>`

**Methods:**

#### `from_pairs(pairs: Vec<(f64, State5D)>) -> Trajectory`
Create from time-state pairs.

#### `export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>>`
Export to CSV file.

Format:
```
time,sigma_1,sigma_2,sigma_3,sigma_4,sigma_5
0.0,1.0,2.0,3.0,4.0,5.0
...
```

#### `export_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>>`
Export to JSON file.

Format:
```json
{
  "times": [0.0, 0.1, ...],
  "states": [[1.0,2.0,3.0,4.0,5.0], ...]
}
```

---

## Module: `validation`

### Functions

#### `test_linear_decoupled() -> bool`
Test 1: Linear decoupled system.
Tests exponential decay with analytical solution.

#### `test_harmonic_oscillator() -> bool`
Test 2: Harmonic oscillator.
Tests oscillation with analytical solution.

#### `test_fixed_point_convergence() -> bool`
Test 3: Fixed point convergence.
Tests convergence to stable equilibrium.

#### `run_all_tests() -> bool`
Run all validation tests. Returns true if all pass.

---

## Module: `ensemble`

### `EnsembleConfig`

Configuration for ensemble simulation.

**Fields:**
- `num_runs: usize`
- `initial_state_mean: State5D`
- `initial_state_std: f64`

**Methods:**

#### `new(num_runs: usize, initial_state_mean: State5D, initial_state_std: f64) -> EnsembleConfig`
Create configuration.

### `EnsembleResult`

Result of ensemble simulation.

**Fields:**
- `trajectories: Vec<Vec<State5D>>`
- `mean_trajectory: Vec<State5D>`
- `std_trajectory: Vec<State5D>`

**Methods:**

#### `from_trajectories(trajectories: Vec<Vec<State5D>>) -> EnsembleResult`
Compute statistics from trajectories.

### `ParameterSweep`

Parameter sweep configuration.

**Fields:**
- `parameter_name: String`
- `values: Vec<f64>`

**Methods:**

#### `new(parameter_name: String, start: f64, end: f64, num_points: usize) -> ParameterSweep`
Create linear parameter sweep.

### Functions

#### `run_ensemble(config: &EnsembleConfig, vf: &VectorField, tc: &TimeConfig) -> EnsembleResult`
Run Monte Carlo ensemble simulation with randomized initial conditions.

**Arguments:**
- `config`: Ensemble configuration (number of runs, mean state, std deviation)
- `vf`: Vector field defining the dynamics
- `tc`: Time configuration for integration

**Returns:** `EnsembleResult` containing all trajectories and statistics

#### `run_parameter_sweep(sweep: &ParameterSweep, base_template: &Template, initial: State5D, tc: &TimeConfig) -> Vec<Vec<State5D>>`
Run parameter sweep over a range of values.

**Arguments:**
- `sweep`: Parameter sweep configuration
- `base_template`: Base template to modify
- `initial`: Initial state for all runs
- `tc`: Time configuration for integration

**Returns:** Vector of trajectories, one for each parameter value

---

## Usage Examples

### Basic Integration
```rust
use system_5d::*;

let coupling = CouplingMatrix::identity();
let vf = VectorField::from_coupling(coupling);
let tc = TimeConfig::new(0.01, 0.0, 10.0);
let integrator = Integrator::new(vf, tc);
let initial = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
let trajectory = integrator.integrate(initial);
```

### Using Templates
```rust
let template = Template::sir_model(0.3, 0.1, 0.01);
let vf = template.to_vector_field();
let tc = TimeConfig::new(0.1, 0.0, 100.0);
let integrator = Integrator::new(vf, tc);
let initial = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0);
let trajectory = integrator.integrate(initial);
```

### Stability Analysis
```rust
let state = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
let jac = vector_field.jacobian(&state);
let eigs = StabilityAnalyzer::eigenvalues(&jac);
let stability = StabilityAnalyzer::classify_stability(&eigs);
```

### Data Export
```rust
let traj = Trajectory::from_pairs(trajectory);
traj.export_csv("output.csv")?;
traj.export_json("output.json")?;
```

### Projection
```rust
let mut projector = Projector::orthogonal(0, 1);
let points = projector.project_many(&states);

// Or use PCA
let mut projector = Projector::new(ProjectionMethod::PCA);
projector.fit_pca(&states);
let points = projector.project_many(&states);
```

---

## Error Handling

Most functions that can fail return `Result<T, E>`:
- File I/O operations return `Result<(), Box<dyn Error>>`
- State validation panics on NaN/Inf (by design)
- Integration handles non-finite values gracefully

---

## Type Aliases

None currently defined, but could add:
```rust
type State = State5D;
type Matrix5 = [[f64; 5]; 5];
type TrajectoryPoint = (f64, State5D);
```
