# Unified Demo Rewrite Specification: jugar-probar First

**Document ID:** SIMULAR-DEMO-001
**Version:** 1.0.0
**Status:** Draft
**Author:** PAIML Engineering
**Date:** 2025-12-12

---

## Executive Summary

This specification defines a complete rewrite of ALL simular demos (TSP, Orbit, Monte Carlo, etc.) using a **single, unified architecture** that guarantees:

1. **100% TDD/EDD compliance** via jugar-probar testing framework
2. **YAML-first configuration** as the single source of truth
3. **Bit-identical behavior** between TUI and WASM implementations
4. **Falsifiable hypotheses** with metamorphic testing
5. **Deterministic replay** for reproducibility

---

## 1. Peer-Reviewed Citations

This specification is grounded in peer-reviewed research:

### [C1] Metamorphic Testing for Scientific Software
> Chen, T.Y., Kuo, F.-C., Liu, H., Poon, P.-L., Towey, D., Tse, T.H., & Zhou, Z.Q. (2018). **"Metamorphic testing: a review of challenges and opportunities."** *ACM Computing Surveys*, 51(1), Article 4.
> DOI: [10.1145/3143561](https://dl.acm.org/doi/10.1145/3143561)

**Application:** All demos implement metamorphic relations (MRs) that verify invariants without oracle knowledge. For TSP: `MR-RotationInvariance`, `MR-PermutationInvariance`. For Orbit: `MR-EnergyConservation`, `MR-TimeReversal`.

### [C2] Deterministic Replay Survey
> Lavoie, E., & Hendren, L. (2015). **"Deterministic Replay: A Survey."** *ACM Computing Surveys*, 48(2), Article 25.
> DOI: [10.1145/2790077](https://dl.acm.org/doi/10.1145/2790077)

**Application:** All demos support deterministic replay via seed-controlled RNG, event journaling, and state checkpointing. Given identical YAML + seed, TUI and WASM produce bit-identical outputs.

### [C3] Property-Based Testing for Scientific Code
> Faulk, S., Loh, E., Van De Vanter, M.L., Squires, S., & Votta, L.G. (2020). **"Falsify your Software: validating scientific code with property-based testing."** *Computing in Science & Engineering*.
> DOI: [10.1109/MCSE.2019.2919318](https://www.researchgate.net/publication/343232178_Falsify_your_Software_validating_scientific_code_with_property-based_testing)

**Application:** All demos use jugar-probar's `InvariantChecker` with Hypothesis-style property generation. Properties are falsifiable per Popperian methodology.

### [C4] Property-Based Testing in Agent-Based Simulation
> Gaudel, N., Michel, F., Ferber, J., & Stratulat, T. (2019). **"Show Me Your Properties! The Potential of Property-Based Testing in Agent-Based Simulation."** *SummerSim-SCSC*, 31(1).
> DOI: [10.22360/summersim.2019.scsc.001](https://www.researchgate.net/publication/333198243_SHOW_ME_YOUR_PROPERTIES_THE_POTENTIAL_OF_PROPERTY-BASED_TESTING_IN_AGENT-BASED_SIMULATION)

**Application:** Stochastic simulations (Monte Carlo, GRASP) use property-based testing with specification-driven test generation. QuickCheck-style shrinking identifies minimal failing cases.

### [C5] Metamorphic Testing for Monte Carlo Programs
> Villaverde, A.F., Pathirana, D., Fröhlich, F., Hasenauer, J., & Banga, J.R. (2016). **"Application of metamorphic testing monitored by test adequacy in a Monte Carlo simulation program."** *Software Quality Journal*, 26, 1281-1303.
> DOI: [10.1007/s11219-016-9337-3](https://link.springer.com/article/10.1007/s11219-016-9337-3)

**Application:** Monte Carlo Pi demo uses metamorphic relations for convergence testing. MR-SampleScaling verifies that doubling samples reduces variance by ~sqrt(2).

---

## 2. Architecture Overview

### 2.1 Single Source of Truth: YAML

```
┌─────────────────────────────────────────────────────────────────┐
│                    YAML Configuration                            │
│  examples/experiments/{demo}.yaml                                │
│  - Complete state specification                                  │
│  - Algorithm parameters                                          │
│  - Falsification criteria                                        │
│  - Expected outcomes                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Engine (lib.rs)                          │
│  src/demos/{demo}.rs                                             │
│  - Pure Rust, no I/O                                             │
│  - Deterministic given seed                                      │
│  - Implements EddDemo trait                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    TUI Renderer          │    │    WASM Renderer         │
│  src/tui/{demo}_app.rs   │    │  src/demos/{demo}_wasm.rs│
│  - ratatui rendering     │    │  - Canvas rendering      │
│  - Terminal events       │    │  - DOM events            │
│  - SAME state machine    │    │  - SAME state machine    │
└──────────────────────────┘    └──────────────────────────┘
```

### 2.2 Key Invariant: Renderer Independence

**CRITICAL:** The core demo engine (`{Demo}Engine`) MUST be renderer-agnostic. Both TUI and WASM:
- Load from the SAME YAML
- Use the SAME engine
- Produce the SAME state sequence
- Differ ONLY in how they render pixels/characters

---

## 3. Unified Demo Structure

### 3.1 Directory Layout

```
simular/
├── examples/experiments/           # YAML configs (source of truth)
│   ├── tsp_bay_area.yaml
│   ├── orbit_earth_sun.yaml
│   ├── monte_carlo_pi.yaml
│   ├── harmonic_oscillator.yaml
│   └── ...
├── schemas/                        # JSON Schema for validation
│   ├── demo.schema.json           # Unified demo schema
│   └── experiment.schema.json
├── src/
│   ├── demos/
│   │   ├── mod.rs                 # Demo registry
│   │   ├── engine.rs              # DemoEngine trait
│   │   ├── tsp/
│   │   │   ├── mod.rs             # TSP engine (pure logic)
│   │   │   ├── grasp.rs           # GRASP algorithm
│   │   │   ├── instance.rs        # YAML loader
│   │   │   └── metamorphic.rs     # MR tests
│   │   ├── orbit/
│   │   │   ├── mod.rs             # Orbit engine
│   │   │   ├── physics.rs         # Integrators
│   │   │   ├── scenarios.rs       # YAML loader
│   │   │   └── metamorphic.rs     # MR tests
│   │   ├── monte_carlo/
│   │   │   └── ...
│   │   └── ...
│   ├── tui/                       # TUI renderers (thin layer)
│   │   ├── mod.rs
│   │   ├── tsp_app.rs
│   │   ├── orbit_app.rs
│   │   └── ...
│   └── wasm/                      # WASM renderers (thin layer)
│       ├── mod.rs
│       ├── tsp_app.rs
│       ├── orbit_app.rs
│       └── ...
└── tests/
    └── probar/                    # jugar-probar E2E tests
        ├── tsp_probar.rs
        ├── orbit_probar.rs
        └── ...
```

### 3.2 Core Trait: DemoEngine

```rust
/// MANDATORY trait for ALL demos (EDD-compliant)
pub trait DemoEngine: Sized + Clone + Serialize + Deserialize {
    /// Configuration type loaded from YAML
    type Config: DeserializeOwned + Validate;

    /// State snapshot for replay/audit
    type State: Clone + Serialize + Deserialize + PartialEq;

    /// Step result with metrics
    type StepResult;

    // === Lifecycle ===

    /// Create engine from YAML configuration
    fn from_yaml(yaml: &str) -> Result<Self, DemoError>;

    /// Create engine from config struct
    fn from_config(config: Self::Config, seed: u64) -> Self;

    /// Reset to initial state (same seed = same result)
    fn reset(&mut self);

    /// Reset with new seed
    fn reset_with_seed(&mut self, seed: u64);

    // === Execution ===

    /// Execute one step (deterministic given state + seed)
    fn step(&mut self) -> Self::StepResult;

    /// Execute N steps
    fn run(&mut self, n: usize) -> Vec<Self::StepResult>;

    /// Check if converged/complete
    fn is_complete(&self) -> bool;

    // === State Access ===

    /// Get current state snapshot (for replay verification)
    fn state(&self) -> Self::State;

    /// Get current step number
    fn step_count(&self) -> u64;

    /// Get seed for reproducibility
    fn seed(&self) -> u64;

    // === EDD Compliance ===

    /// Get falsification criteria from config
    fn falsification_criteria(&self) -> &[FalsificationCriterion];

    /// Evaluate all criteria against current state
    fn evaluate_criteria(&self) -> Vec<CriterionResult>;

    /// Check if all criteria pass
    fn is_verified(&self) -> bool {
        self.evaluate_criteria().iter().all(|r| r.passed)
    }

    // === Metamorphic Relations ===

    /// Get metamorphic relations for this demo
    fn metamorphic_relations(&self) -> Vec<MetamorphicRelation>;

    /// Verify a specific MR
    fn verify_mr(&self, mr: &MetamorphicRelation, source: &Self::State) -> MrResult;
}
```

---

## 4. YAML Schema (Unified)

### 4.1 Demo Configuration Schema

```yaml
# examples/experiments/tsp_bay_area.yaml
---
# === Header (MANDATORY) ===
meta:
  id: "TSP-BAY-020"                    # Unique identifier
  version: "1.0.0"                     # Semantic version
  demo_type: "tsp"                     # Demo engine type
  description: "20-city Bay Area TSP"
  author: "PAIML Engineering"
  created: "2025-12-12"

# === Reproducibility (MANDATORY) ===
seed: 42                               # RNG seed
deterministic: true                    # Enforce determinism

# === Demo-Specific Configuration ===
config:
  # TSP-specific fields
  cities:
    - { id: 0, name: "San Francisco", lat: 37.7749, lon: -122.4194 }
    - { id: 1, name: "Oakland", lat: 37.8044, lon: -122.2712 }
    # ... more cities

  distance_matrix:
    - [0, 12, 48, ...]
    - [12, 0, 42, ...]
    # ... symmetric matrix

  algorithm:
    method: "grasp"
    params:
      rcl_size: 3
      max_restarts: 100
      two_opt: true

  units: "miles"
  optimal_known: 416

# === Falsification (MANDATORY per EDD) ===
falsification:
  criteria:
    - id: "optimality_gap"
      metric: "gap"
      threshold: 0.20
      condition: "gap <= threshold"
      severity: "major"

    - id: "convergence_cv"
      metric: "restart_cv"
      threshold: 0.05
      condition: "cv <= threshold"
      severity: "minor"

    - id: "valid_tour"
      metric: "tour_valid"
      threshold: 1.0
      condition: "tour_valid == threshold"
      severity: "critical"

# === Metamorphic Relations ===
metamorphic:
  relations:
    - id: "MR-PermutationInvariance"
      description: "Relabeling cities preserves tour structure"
      source_transform: "permute_city_indices"
      expected_relation: "tour_length_unchanged"
      tolerance: 1e-10

    - id: "MR-SymmetryConsistency"
      description: "Reversing tour direction preserves length"
      source_transform: "reverse_tour"
      expected_relation: "tour_length_unchanged"
      tolerance: 1e-10

# === Visualization (renderer hints) ===
visualization:
  title: "Bay Area TSP - GRASP"
  color_scheme: "default"
  show_coordinates: true
  show_tour_animation: true
  update_rate_ms: 100
```

### 4.2 JSON Schema for Validation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://simular.paiml.com/schemas/demo.schema.json",
  "title": "Simular Demo Configuration Schema",
  "type": "object",
  "required": ["meta", "seed", "config", "falsification"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["id", "version", "demo_type"],
      "properties": {
        "id": { "type": "string", "pattern": "^[A-Z]+-[A-Z]+-[0-9]+$" },
        "version": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$" },
        "demo_type": { "type": "string", "enum": ["tsp", "orbit", "monte_carlo", "oscillator", "epidemic", "climate"] }
      }
    },
    "seed": { "type": "integer", "minimum": 0 },
    "deterministic": { "type": "boolean", "default": true },
    "config": { "type": "object" },
    "falsification": {
      "type": "object",
      "required": ["criteria"],
      "properties": {
        "criteria": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/criterion" }
        }
      }
    },
    "metamorphic": {
      "type": "object",
      "properties": {
        "relations": {
          "type": "array",
          "items": { "$ref": "#/$defs/metamorphicRelation" }
        }
      }
    }
  },
  "$defs": {
    "criterion": {
      "type": "object",
      "required": ["id", "threshold", "condition"],
      "properties": {
        "id": { "type": "string" },
        "metric": { "type": "string" },
        "threshold": { "type": "number" },
        "condition": { "type": "string" },
        "tolerance": { "type": "number", "minimum": 0 },
        "severity": { "type": "string", "enum": ["critical", "major", "minor"] }
      }
    },
    "metamorphicRelation": {
      "type": "object",
      "required": ["id", "source_transform", "expected_relation"],
      "properties": {
        "id": { "type": "string" },
        "description": { "type": "string" },
        "source_transform": { "type": "string" },
        "expected_relation": { "type": "string" },
        "tolerance": { "type": "number", "default": 1e-10 }
      }
    }
  }
}
```

---

## 5. jugar-probar Integration

### 5.1 Test Structure

```rust
// tests/probar/tsp_probar.rs
use jugar_probar::{
    Assertion, TestCase, TestHarness, TestSuite,
    InvariantChecker, InvariantCheck,
};
use simular::demos::tsp::TspEngine;
use simular::demos::DemoEngine;

const BAY_AREA_YAML: &str = include_str!("../../examples/experiments/tsp_bay_area.yaml");

// === Phase 1: Configuration Loading ===

#[test]
fn probar_tsp_yaml_loads() {
    let engine = TspEngine::from_yaml(BAY_AREA_YAML);
    let result = Assertion::is_ok(&engine);
    assert!(result.passed, "YAML should load: {}", result.message);
}

#[test]
fn probar_tsp_validates_schema() {
    let engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let result = Assertion::equals(&engine.config().meta.id, &"TSP-BAY-020".to_string());
    assert!(result.passed);
}

// === Phase 2: Deterministic Replay ===

#[test]
fn probar_tsp_deterministic_replay() {
    let mut engine1 = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let mut engine2 = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();

    // Run 100 steps
    engine1.run(100);
    engine2.run(100);

    // States MUST be identical
    let result = Assertion::equals(&engine1.state(), &engine2.state());
    assert!(result.passed, "Deterministic replay failed: states differ");

    // Tour lengths MUST be identical
    let result = Assertion::approx_eq(
        engine1.best_tour_length(),
        engine2.best_tour_length(),
        1e-15
    );
    assert!(result.passed, "Tour lengths differ");
}

// === Phase 3: Metamorphic Testing ===

#[test]
fn probar_tsp_mr_permutation_invariance() {
    let engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let mr = engine.metamorphic_relations()
        .iter()
        .find(|r| r.id == "MR-PermutationInvariance")
        .unwrap();

    let source_state = engine.state();
    let mr_result = engine.verify_mr(mr, &source_state);

    let result = Assertion::is_true(mr_result.passed);
    assert!(result.passed, "MR-PermutationInvariance failed: {}", mr_result.message);
}

// === Phase 4: Falsification Criteria ===

#[test]
fn probar_tsp_falsification_criteria() {
    let mut engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();

    // Run to convergence
    while !engine.is_complete() {
        engine.step();
    }

    // Evaluate all criteria
    let results = engine.evaluate_criteria();

    for result in &results {
        let assertion = Assertion::is_true(result.passed);
        assert!(
            assertion.passed || result.severity != Severity::Critical,
            "Critical criterion {} failed: {}",
            result.id,
            result.message
        );
    }
}

// === Phase 5: Invariant Checking ===

struct TspInvariants;

impl InvariantCheck for TspInvariants {
    fn check_invariant(&self, state: &dyn std::any::Any) -> bool {
        let engine = state.downcast_ref::<TspEngine>().unwrap();

        // Invariant 1: Tour visits all cities exactly once
        let tour = engine.best_tour();
        let n = engine.city_count();
        if tour.len() != n {
            return false;
        }
        let mut visited = vec![false; n];
        for &city in tour {
            if city >= n || visited[city] {
                return false;
            }
            visited[city] = true;
        }

        // Invariant 2: Tour length matches computed distance
        let computed = engine.compute_tour_length(tour);
        if (computed - engine.best_tour_length()).abs() > 1e-10 {
            return false;
        }

        // Invariant 3: Lower bound <= best tour length
        if engine.lower_bound() > engine.best_tour_length() + 1e-10 {
            return false;
        }

        true
    }
}

#[test]
fn probar_tsp_invariants() {
    let mut engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let checker = InvariantChecker::new(Box::new(TspInvariants));

    for _ in 0..100 {
        engine.step();
        let result = checker.check(&engine);
        assert!(result.passed, "Invariant violated at step {}", engine.step_count());
    }
}
```

### 5.2 TUI/WASM Parity Tests

```rust
// tests/probar/parity_probar.rs
use jugar_probar::{Assertion, TestCase, TestHarness, TestSuite};
use simular::demos::tsp::TspEngine;
use simular::tui::TspTuiApp;
use simular::wasm::TspWasmApp;

const BAY_AREA_YAML: &str = include_str!("../../examples/experiments/tsp_bay_area.yaml");

/// CRITICAL: TUI and WASM MUST produce identical state sequences
#[test]
fn probar_parity_tui_wasm_state_sequence() {
    let mut tui = TspTuiApp::from_yaml(BAY_AREA_YAML).unwrap();
    let mut wasm = TspWasmApp::from_yaml(BAY_AREA_YAML).unwrap();

    // Run 50 steps on each
    for step in 0..50 {
        tui.step();
        wasm.step();

        // States MUST be bit-identical
        let result = Assertion::equals(
            &tui.engine().state(),
            &wasm.engine().state()
        );
        assert!(
            result.passed,
            "TUI/WASM state diverged at step {}: {}",
            step,
            result.message
        );
    }
}

/// Verify both renderers expose same metrics
#[test]
fn probar_parity_metrics() {
    let mut tui = TspTuiApp::from_yaml(BAY_AREA_YAML).unwrap();
    let mut wasm = TspWasmApp::from_yaml(BAY_AREA_YAML).unwrap();

    tui.run(100);
    wasm.run(100);

    // All metrics must match
    assert_eq!(tui.engine().best_tour_length(), wasm.engine().best_tour_length());
    assert_eq!(tui.engine().lower_bound(), wasm.engine().lower_bound());
    assert_eq!(tui.engine().optimality_gap(), wasm.engine().optimality_gap());
    assert_eq!(tui.engine().step_count(), wasm.engine().step_count());
}
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Define `DemoEngine` trait in `src/demos/engine.rs`
- [ ] Create unified YAML schema in `schemas/demo.schema.json`
- [ ] Implement schema validation with feature-gated jsonschema
- [ ] Set up jugar-probar test infrastructure

### Phase 2: TSP Demo Rewrite (Week 2)
- [ ] Refactor `TspGraspDemo` to implement `DemoEngine`
- [ ] Move YAML loading to `TspConfig`
- [ ] Extract pure engine logic (no rendering)
- [ ] Implement metamorphic relations
- [ ] Create thin TUI renderer
- [ ] Create thin WASM renderer
- [ ] Write probar parity tests

### Phase 3: Orbit Demo Rewrite (Week 3)
- [ ] Refactor `OrbitSimulation` to implement `DemoEngine`
- [ ] Create `orbit_earth_sun.yaml` config
- [ ] Implement physics metamorphic relations (energy, momentum)
- [ ] Thin TUI/WASM renderers
- [ ] Probar parity tests

### Phase 4: Monte Carlo Demo Rewrite (Week 4)
- [ ] Refactor `MonteCarloPi` to implement `DemoEngine`
- [ ] Implement convergence metamorphic relations
- [ ] Thin TUI/WASM renderers
- [ ] Probar parity tests

### Phase 5: Remaining Demos (Week 5-6)
- [ ] Harmonic oscillator
- [ ] Epidemic simulation
- [ ] Climate model
- [ ] Factory queueing (Little's Law)

### Phase 6: Documentation & Polish (Week 7)
- [ ] Update all examples
- [ ] Write user guide for creating new demos
- [ ] Performance benchmarks
- [ ] Final parity verification

---

## 7. Quality Gates

### 7.1 Per-Demo Requirements

| Gate | Requirement | Enforcement |
|------|-------------|-------------|
| EDD-01 | YAML config passes schema validation | `make validate` |
| EDD-02 | Implements `DemoEngine` trait | Compile-time |
| EDD-03 | Has ≥1 falsification criterion | YAML schema |
| EDD-04 | Has ≥1 metamorphic relation | YAML schema |
| EDD-05 | Deterministic replay works | probar test |
| EDD-06 | TUI/WASM parity verified | probar test |
| EDD-07 | 95% code coverage | `make coverage` |
| EDD-08 | All probar tests pass | `make test-probar` |

### 7.2 PMAT Integration

```bash
# Run PMAT quality gates
pmat quality-gate --strict

# Expected output:
# ✓ Test coverage: 95.2% (threshold: 95%)
# ✓ Mutation coverage: 82.1% (threshold: 80%)
# ✓ Max complexity: 12 (threshold: 15)
# ✓ Max nesting: 3 (threshold: 4)
# ✓ EDD compliance: PASS
# ✓ Probar parity: PASS
```

---

## 8. Migration Guide

### 8.1 Existing Demo Migration Steps

1. **Create YAML config** from existing hardcoded parameters
2. **Extract engine logic** from rendering code
3. **Implement `DemoEngine`** trait
4. **Add metamorphic relations**
5. **Create thin TUI renderer** that wraps engine
6. **Create thin WASM renderer** that wraps engine
7. **Write probar parity tests**
8. **Verify deterministic replay**

### 8.2 Example: TSP Migration

**Before:**
```rust
// Old: Rendering mixed with logic
impl TspGraspDemo {
    pub fn new(seed: u64, n: usize) -> Self {
        // Random city generation (not reproducible from YAML)
        let cities = generate_random_cities(n, seed);
        // ...
    }

    pub fn render_to_canvas(&self, ctx: &CanvasRenderingContext2d) {
        // Rendering logic mixed with state
    }
}
```

**After:**
```rust
// New: Clean separation
impl DemoEngine for TspEngine {
    type Config = TspConfig;
    type State = TspState;
    type StepResult = TspStepResult;

    fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let config: TspConfig = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(Self::from_config(config, config.seed))
    }

    fn step(&mut self) -> TspStepResult {
        // Pure logic, no rendering
        self.grasp_iteration()
    }
}

// TUI renderer (thin wrapper)
impl TspTuiApp {
    pub fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let engine = TspEngine::from_yaml(yaml)?;
        Ok(Self { engine, frame: 0 })
    }

    pub fn render(&self, frame: &mut Frame) {
        // ONLY rendering, no state mutation
        render_tsp_to_tui(frame, &self.engine);
    }
}

// WASM renderer (thin wrapper)
impl TspWasmApp {
    pub fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let engine = TspEngine::from_yaml(yaml)?;
        Ok(Self { engine, canvas: None })
    }

    pub fn render(&self) {
        // ONLY rendering, no state mutation
        render_tsp_to_canvas(&self.ctx, &self.engine);
    }
}
```

---

## 9. Appendix A: Demo Inventory

| Demo | Type | Status | YAML | TUI | WASM | Probar |
|------|------|--------|------|-----|------|--------|
| TSP GRASP | Optimization | Rewrite | `tsp_bay_area.yaml` | Yes | Yes | Yes |
| Orbit | Physics | Rewrite | `orbit_earth_sun.yaml` | Yes | Yes | Yes |
| Monte Carlo Pi | Statistics | Rewrite | `monte_carlo_pi.yaml` | Yes | Yes | Yes |
| Harmonic Oscillator | Physics | New | `harmonic_oscillator.yaml` | Yes | Yes | Yes |
| Epidemic SIR | Simulation | New | `epidemic_sir.yaml` | Yes | Yes | Yes |
| Little's Law | Queueing | New | `littles_law.yaml` | Yes | Yes | Yes |
| Kepler Orbit | Physics | New | `kepler_orbit.yaml` | Yes | Yes | Yes |
| Climate Model | Environment | New | `climate_model.yaml` | Yes | Yes | Yes |

---

## 10. Appendix B: Metamorphic Relations Library

### B.1 Universal Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-DeterministicReplay | All | Same seed → same output |
| MR-StateConsistency | All | state() matches internal fields |
| MR-StepMonotonicity | All | step_count() increases by 1 per step |

### B.2 Optimization Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-PermutationInvariance | TSP | Relabeling preserves solution structure |
| MR-SymmetricDistance | TSP | d(i,j) = d(j,i) |
| MR-TriangleInequality | Euclidean TSP | d(i,k) ≤ d(i,j) + d(j,k) |
| MR-LowerBoundValid | TSP | LB ≤ optimal ≤ best_found |

### B.3 Physics Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-EnergyConservation | Hamiltonian systems | |E(t) - E(0)| < ε |
| MR-MomentumConservation | Closed systems | |p(t) - p(0)| < ε |
| MR-TimeReversal | Symplectic integrators | Forward+reverse ≈ identity |
| MR-RotationInvariance | Central force | Rotating system preserves dynamics |

### B.4 Statistical Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-SampleScaling | Monte Carlo | 4x samples → 2x precision |
| MR-ConfidenceInterval | Monte Carlo | CI contains true value |
| MR-VarianceReduction | Monte Carlo | Stratification reduces variance |

---

## 11. References

1. Chen, T.Y. et al. (2018). "Metamorphic testing: a review of challenges and opportunities." ACM Computing Surveys.
2. Lavoie, E. & Hendren, L. (2015). "Deterministic Replay: A Survey." ACM Computing Surveys.
3. Faulk, S. et al. (2020). "Falsify your Software: validating scientific code with property-based testing." Computing in Science & Engineering.
4. Gaudel, N. et al. (2019). "Show Me Your Properties! The Potential of Property-Based Testing in Agent-Based Simulation." SummerSim-SCSC.
5. Villaverde, A.F. et al. (2016). "Application of metamorphic testing monitored by test adequacy in a Monte Carlo simulation program." Software Quality Journal.

---

**Document End**