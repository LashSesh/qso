# Q⊗DASH 
# (Metatron VM) 
**Quantum State Operator Framework**

[![Rust](https://img.shields.io/badge/rust-1.85.0-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Hochmodernes Quantencomputing-Framework in pure Rust**

QDash implementiert den Metatron Quantum State Operator (QSO) - ein 13-dimensionales Quantensystem basierend auf der Heiligen Geometrie des Metatron-Würfels mit dynamischer tripolarer Logik (DTL).

## 🌟 Kernfeatures

### Quantenalgorithmen
- ✅ **Variational Quantum Eigensolver (VQE)** - Grundzustandsberechnung
- ✅ **QAOA** - Kombinatorische Optimierung (MaxCut, Graph Coloring)
- ✅ **VQC** - Variational Quantum Classifier (ML)
- ✅ **Quantum Walks** - CTQW, Krylov-Methoden, Scattering-Analyse

### Metatron-Geometrie (13 Knoten)
- **1 Zentralknoten** + **6 Hexagon-Knoten** + **6 Würfel-Knoten**
- **78 Kanten** mit vollständiger Konnektivität
- Einbettung aller 5 Platonischen Körper
- Symmetriegruppe G_M für fehlerresistente Operationen

### Dynamic Tripolar Logic (DTL)
- **58,5% Informationsvorteil** über binäre Systeme
- Zustände: L+ (aktiv), L- (inaktiv), Ld (dynamisch/unbestimmt)
- Kuramoto-Synchronisationsnetzwerke
- Resonator-Dynamik

### Performance & CI/CD
- **6 umfassende Benchmark-Suites** mit automatischer Regression-Detection
- Baseline-Tracking für alle Algorithmen
- Parallele CI/CD-Pipeline mit GitHub Actions
- Tägliche Performance-Metriken

## 🚀 Quick Start

### Installation

```bash
# Repository klonen
git clone https://github.com/LashSesh/qdash.git
cd qdash/metatron-qso-rs

# Build & Test
cargo build --release
cargo test --lib

# Benchmarks ausführen
cargo run --release --bin quantum_walk_bench
cargo run --release --bin vqe_bench
```

### Ihr erstes Quantenprogramm

```rust
use metatron_qso_rs::prelude::*;
use nalgebra::DVector;

fn main() -> Result<()> {
    // Metatron QSO initialisieren
    let params = QSOParameters::default();
    let qso = QSO::new(params)?;

    // Quantum Walk von Zentrumsknoten starten
    let initial_state = QuantumState::basis_state(0); // Node 0 = Zentrum
    let time = 1.0;
    let evolved = qso.evolve_state(&initial_state, time)?;

    // Wahrscheinlichkeitsverteilung ausgeben
    let probs = evolved.probabilities();
    println!("Probability distribution after t=1.0:");
    for (node, prob) in probs.iter().enumerate() {
        println!("  Node {}: {:.4}", node, prob);
    }

    Ok(())
}
```

### VQE Grundzustandsberechnung

```rust
use metatron_qso_rs::prelude::*;
use metatron_qso_rs::vqa::{VQE, AnsatzType};

fn main() -> Result<()> {
    let qso = QSO::new(QSOParameters::default())?;

    // VQE konfigurieren
    let vqe = VQE::builder()
        .hamiltonian(qso.hamiltonian().clone())
        .ansatz_type(AnsatzType::HardwareEfficient)
        .depth(2)
        .optimizer_name("ADAM")
        .max_iterations(1000)
        .build()?;

    // Optimierung starten
    let result = vqe.run()?;

    println!("Ground State Energy: {:.10}", result.ground_energy);
    println!("Converged in {} iterations", result.iterations);

    Ok(())
}
```

### QAOA für MaxCut

```rust
use metatron_qso_rs::vqa::{QAOA, MaxCutProblem};

fn main() -> Result<()> {
    // MaxCut Problem auf Metatron-Graph definieren
    let graph = MetatronGraph::new();
    let problem = MaxCutProblem::from_graph(&graph);

    // QAOA mit depth=3
    let qaoa = QAOA::new(problem.hamiltonian(), 3);
    let result = qaoa.run()?;

    println!("Best cut value: {:.2}", result.best_value);
    println!("Approximation ratio: {:.4}", result.approximation_ratio);

    Ok(())
}
```

## 🐍 Python SDK (metatron_qso)

**High-Performance Quantum Computing in Python** - Powered by Rust

Das Python SDK bietet eine benutzerfreundliche API für Data Scientists, ML-Researcher und Entwickler:

```python
import metatron_qso

# Metatron Cube Graph erstellen
graph = metatron_qso.MetatronGraph()

# Quantum Walk ausführen
result = metatron_qso.run_quantum_walk(
    graph=graph,
    source_nodes=[0],  # Zentralknoten
    t_max=5.0,
    dt=0.1
)

# MaxCut mit QAOA lösen
qaoa_result = metatron_qso.solve_maxcut_qaoa(
    graph=graph,
    depth=3,
    max_iters=100
)

# VQE für Grundzustand
vqe_result = metatron_qso.run_vqe(
    graph=graph,
    depth=2,
    max_iters=150,
    ansatz_type="hardware_efficient"
)
```

### Installation

```bash
# Rust Toolchain installieren (falls nicht vorhanden)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Maturin installieren
pip install maturin

# Python SDK bauen und installieren
cd metatron_qso_py
maturin develop --release
```

### Features

- ✅ **Python-idiomatische API** - dict returns, list parameters
- ✅ **Rust-Performance** - Zero-cost Python bindings via PyO3
- ✅ **Jupyter-ready** - Interactive notebooks mit Visualisierungen
- ✅ **Type Safety** - Klare Fehlerbehandlung ohne Panics

### Beispiele & Notebooks

```bash
# Beispiele ausführen
python metatron_qso_py/examples/01_quantum_walk_basic.py
python metatron_qso_py/examples/02_qaoa_maxcut_basic.py
python metatron_qso_py/examples/03_vqe_ground_state.py

# Jupyter Notebook starten
jupyter notebook metatron_qso_py/notebooks/QuantumWalk_Intro.ipynb
```

### Dokumentation

- [Python SDK Guide](docs/PYTHON_SDK_GUIDE.md) - Vollständige API-Referenz
- [Python SDK README](metatron_qso_py/README.md) - Quick Start Guide
- [Jupyter Notebooks](metatron_qso_py/notebooks/) - Interaktive Tutorials

## 🔧 Seraphic Calibration Shell (SCS) - Auto-Tuner

**Automatische Hyperparameter-Optimierung für Quantenalgorithmen**

Die Seraphic Calibration Shell ist ein Meta-Algorithmus, der Quantenalgorithmen automatisch optimiert. SCS nutzt field-theoretisches Feedback und Fixpoint-Dynamiken, um die beste Konfiguration für Ihre Algorithmen zu finden.

### Kernkonzepte

**Performance Triplet Φ(c) = (ψ, ρ, ω)**
- **ψ** (Quality): Algorithmen-spezifische Qualität (z.B. Approximation Ratio bei QAOA)
- **ρ** (Stability): Robustheit über mehrere Runs
- **ω** (Efficiency): Recheneffizienz (Evaluationen/Sekunde)

**Mandorla Field M(t)**
- 16-dimensionales Resonanzfeld für Feedback-Akkumulation
- Speichert historische Performance-Muster
- Leitet Konfigurationsänderungen

**Double-Kick Operator T = Φ_V ∘ Φ_U**
- Update-Kick Φ_U: Verbessert Qualität
- Stabilization-Kick Φ_V: Optimiert Stabilität & Effizienz
- Konvergiert zu Fixpoint-Attraktoren

**Proof-of-Resonance (PoR)**
- Akzeptanzkriterium für neue Konfigurationen
- Garantiert monotone Qualitätsverbesserung
- Validiert Field-Resonanz

**CRI (Calibration Regime Initialization)**
- Erkennt Stagnation im lokalen Optimum
- Wechselt automatisch zu neuem Regime (z.B. VQE → QAOA)
- Ermöglicht globale Exploration

### Quick Start

**Mit Python SDK:**
```python
import metatron_qso

graph = metatron_qso.MetatronGraph()

# QAOA mit Auto-Calibration
result, proposal = metatron_qso.solve_maxcut_qaoa_with_tuning(
    graph=graph,
    depth=3,
    max_iters=100,
    auto_calibrate=True
)

print(f"Approximation ratio: {result['approximation_ratio']:.3f}")
if proposal.por_accepted:
    print(f"SCS schlägt neue Konfiguration vor: depth={proposal.config.ansatz_depth}")
```

**Mit CLI:**
```bash
# SCS initialisieren
python -m scs.cli init

# 5 Calibration-Schritte ausführen
python -m scs.cli step -n 5

# Status anzeigen
python -m scs.cli status

# Beste Konfiguration exportieren
python -m scs.cli export -o best_config.json
```

**Auto-Tuning Loop:**
```python
from scs import AutoTuner

tuner = AutoTuner(benchmark_dir="benchmarks", enabled=True)
tuner.initialize()

for iteration in range(10):
    # Algorithmus ausführen
    result = run_your_algorithm()

    # Metrics berechnen
    metrics = {"psi": 0.85, "rho": 0.80, "omega": 0.72}

    # In SCS einspeisen
    tuner.ingest_benchmark("qaoa", config, metrics, result)

    # Neue Konfiguration vorschlagen
    proposal = tuner.propose_new_config()

    if proposal.por_accepted:
        # Neue Config anwenden
        config = proposal.config
```

### Features

- ✅ **Opt-in Design** - SCS ist optional und stört bestehende Workflows nicht
- ✅ **Generisches Benchmark-Schema** - Unterstützt alle Quantenalgorithmen
- ✅ **Persistenter Zustand** - Speichert Field-State und History
- ✅ **CLI & Python API** - Flexible Nutzung
- ✅ **Integration mit QW & QAOA** - Native Auto-Tuning-Hooks
- ✅ **Nachvollziehbar** - Alle Schritte dokumentiert und erklärbar

### Dokumentation

- [SCS Core Design](docs/SCS_CORE_DESIGN.md) - Architektur & Datenfluss
- [SCS Benchmark Schema](docs/SCS_BENCHMARK_SCHEMA.md) - JSON-Schema Spezifikation
- [SCS Usage Guide](docs/SCS_USAGE_GUIDE.md) - Workflows & Best Practices

---

## 📊 Architektur

```
qdash/
├── metatron-qso-rs/          # Rust Core Library
│   ├── src/
│   │   ├── lib.rs            # Library entry point
│   │   ├── qso.rs            # Quantum State Operator (Haupt-API)
│   │   ├── graph/            # Metatron-Geometrie
│   │   ├── quantum/          # Quantenzustände & Operatoren
│   │   ├── dtl/              # Dynamic Tripolar Logic
│   │   ├── quantum_walk/     # Quantum Walk Algorithmen
│   │   └── vqa/              # Variational Quantum Algorithms
│   ├── bins/                 # 8 Benchmark-Executables
│   ├── ci/                   # Baseline-Daten für CI/CD
│   └── docs/                 # Detaillierte Dokumentation
├── metatron_qso_py/          # Python SDK (PyO3/Maturin)
│   ├── src/lib.rs            # Python bindings
│   ├── python/               # Pure Python helpers
│   │   └── metatron_qso/
│   │       ├── __init__.py   # Public API
│   │       └── auto_tuning.py # SCS integration
│   ├── examples/             # Python-Beispiele
│   ├── notebooks/            # Jupyter Notebooks
│   ├── Cargo.toml            # cdylib configuration
│   └── pyproject.toml        # Maturin build config
├── scs/                      # Seraphic Calibration Shell (Auto-Tuner)
│   ├── config.py             # Configuration space
│   ├── performance.py        # Performance triplet (ψ, ρ, ω)
│   ├── field.py              # Mandorla field M(t)
│   ├── operators.py          # Double-kick operator T
│   ├── por.py                # Proof-of-Resonance
│   ├── cri.py                # CRI regime switching
│   ├── calibrator.py         # Main orchestrator
│   ├── benchmark.py          # Benchmark system
│   ├── core.py               # Auto-tuner API
│   └── cli.py                # CLI interface
├── docs/                     # Globale Dokumentation
│   ├── PYTHON_SDK_GUIDE.md   # Python API Guide
│   ├── SCS_CORE_DESIGN.md    # SCS Architecture
│   ├── SCS_BENCHMARK_SCHEMA.md # Benchmark JSON Schema
│   ├── SCS_USAGE_GUIDE.md    # SCS Workflows
│   ├── QUANTENINFORMATIONSVERARBEITUNG_DOKUMENTATION.md
│   ├── VQA_IMPLEMENTATION_GUIDE.md (aktualisiert für Rust)
│   └── BENCHMARK_*.md
├── CHANGELOG.md              # Version history
├── RELEASE_PLAN.md           # Packaging strategy
└── .github/workflows/        # CI/CD Pipelines
```

## 📖 Dokumentation

### Deutsch
- [Quanteninformationsverarbeitung Dokumentation](QUANTENINFORMATIONSVERARBEITUNG_DOKUMENTATION.md)
- [VQA Implementierungsleitfaden](VQA_IMPLEMENTATION_GUIDE.md)
- [Projekt-Roadmap](PROJECT_ROADMAP.md)

### Englisch
- [Architecture Overview](metatron-qso-rs/docs/ARCHITECTURE.md)
- [Benchmark Suite Documentation](BENCHMARK_SUITE_DOCUMENTATION.md)
- [CI/CD Upgrade Summary](CI_BENCHMARK_UPGRADE_SUMMARY.md)

### API-Dokumentation
```bash
# Rustdoc generieren und öffnen
cargo doc --open
```

## 🧪 Testen & Benchmarking

```bash
# Alle Unit-Tests
cargo test --lib

# Spezifischen Benchmark
cargo run --release --bin vqe_bench
cargo run --release --bin qaoa_bench
cargo run --release --bin vqc_bench
cargo run --release --bin quantum_walk_bench

# Integration-Tests
cargo run --release --bin integration_bench

# Cross-Framework Vergleich
cargo run --release --bin cross_system_bench
```

## 🔬 Wissenschaftlicher Hintergrund

### Informationstheoretischer Vorteil

```
Metatron-System (13 Knoten):
├─ Binär:     13,0 Bit (klassisch)
├─ Tripolar:  20,6 Bit (+58,5%)
└─ Mit Phase: 46,6 Bit (+258% über binär)
```

### Quantenalgorithmen auf Metatron-Graph

| Algorithmus | Komplexität | Speedup vs. Klassisch |
|-------------|-------------|----------------------|
| Quantum Walk Search | O(√N) | ~3.6× |
| VQE Ground State | O(poly(n)) | Exponentiell |
| QAOA MaxCut | O(p·M) | >0.75 approximation |
| Boson Sampling | #P-hard | Klassisch intraktabel |

### Graph-Eigenschaften

- **Knoten:** 13 (1 Zentrum + 6 Hexagon + 6 Würfel)
- **Kanten:** 78
- **Durchschnittsgrad:** 12
- **Algebraische Konnektivität:** λ₁ > 0 (hoch)
- **Code-Distanz:** d ≥ 6 (für topologische Fehlerkorrektur)

## 🛠️ Entwicklung

### Voraussetzungen
- Rust 1.85.0+ (Edition 2024)
- Cargo
- Optional: Just (für Task-Automatisierung)

### Projekt aufbauen
```bash
cd metatron-qso-rs
cargo build --release
```

### Tests ausführen
```bash
cargo test --lib          # Unit-Tests
cargo test --bins         # Binary-Tests
```

### Formatierung & Linting
```bash
cargo fmt                 # Code formatieren
cargo clippy              # Linter
```

## 📈 Performance-Baselines

| Benchmark | Operationen/Sek | Konvergenz |
|-----------|----------------|------------|
| Quantum Walk | 31,941 | 100% |
| VQE (HardwareEfficient) | ~50 iters | E₀ = -12.9997 |
| QAOA (depth=3) | ~100 iters | ratio = 1.0 |
| VQC (binary) | ~200 epochs | acc = 50-90% |

## 🎯 Roadmap

### ✅ Phase 1: Core Implementation (Abgeschlossen)
- [x] Metatron-Geometrie (13 Knoten, 78 Kanten)
- [x] Quantum State & Operator Primitives
- [x] DTL System
- [x] Quantum Walks (CTQW, Krylov, Scattering)

### ✅ Phase 2: Variational Algorithms (Abgeschlossen)
- [x] VQE mit 3 Ansatz-Typen
- [x] QAOA für kombinatorische Optimierung
- [x] VQC für Klassifikation
- [x] Parameter Shift Rule Gradienten

### ✅ Phase 3: Benchmarking & CI/CD (Abgeschlossen)
- [x] 6 umfassende Benchmark-Suites
- [x] Automatische Baseline-Vergleiche
- [x] GitHub Actions Integration
- [x] Performance Regression Detection

### 🚧 Phase 4: Advanced Features (In Arbeit)
- [ ] Metatron-spezifische Grover-Search-Variante
- [ ] Boson-Sampling mit Platonic-Solid-Interferenz
- [ ] Quantum Machine Learning auf Graph-Struktur
- [ ] Symmetrie-geschützte Quantencodes (G_M)
- [ ] GPU-Beschleunigung
- [ ] Visualisierungstools

### 🔮 Phase 5: Hardware-Integration (Geplant)
- [ ] IBM Qiskit Backend
- [ ] AWS Braket Integration
- [ ] IonQ/Rigetti Support
- [ ] Photonisches Chip-Design

## 🤝 Contributing

Beiträge sind willkommen! Bitte beachten Sie:

1. Fork des Repositories
2. Feature-Branch erstellen (`git checkout -b feature/amazing-feature`)
3. Tests hinzufügen (`cargo test`)
4. Committen (`git commit -m 'Add amazing feature'`)
5. Push zum Branch (`git push origin feature/amazing-feature`)
6. Pull Request öffnen

## 📝 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe [LICENSE](LICENSE) für Details.

## 🙏 Danksagungen

- **Heilige Geometrie:** Metatron's Cube als fundamentale Struktur
- **Quanteninformatik:** VQE/QAOA/VQC Forschung
- **Rust Community:** nalgebra, petgraph, rayon

## 📧 Kontakt

Bei Fragen oder Anregungen öffnen Sie bitte ein [GitHub Issue](https://github.com/LashSesh/qdash/issues).

---

**Made with ❤️ in Rust** | **Powered by Quantum Geometry** | **© 2025 QDash Project**
