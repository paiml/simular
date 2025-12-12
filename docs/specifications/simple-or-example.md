---
title: "Simple OR Example: Bay Area TSP Ground Truth"
issue: OR-001
status: Draft
created: 2025-12-12
updated: 2025-12-12
---

# OR-001: Simple Operations Research Example Specification

**Ticket ID**: OR-001
**Status**: Draft (Awaiting Review)
**Methodology**: Toyota Production System (TPS) + Popperian Falsification + Equation Driven Development (EDD)
**Quality**: EXTREME TDD | 100% Coverage | Probar E2E

## 1. Executive Summary

This specification defines a minimal, verifiable Traveling Salesman Problem (TSP) instance using Bay Area cities with known driving distances. Adhering to **Equation Driven Development (EDD)**, the problem is rigorously defined via an Equation Model Card (EMC) before implementation.

**Core Principle**: YAML-first architecture where users can download, modify, and re-run experiments without touching code.

### 1.1 Key Deliverables

| Deliverable | Description | Deployment |
|-------------|-------------|------------|
| `bay_area_tsp.yaml` | Ground truth instance (6 cities) | `examples/experiments/` |
| WASM Module | Browser-executable TSP solver | `interactive.paiml.com` |
| User Editor | In-browser YAML editor | Web UI component |
| Probar E2E | Deterministic replay tests | `tests/probar_tsp.rs` |

## 2. Equation Model Card (EMC)

**ID**: EMC-TSP-MTZ-001
**Name**: Miller-Tucker-Zemlin (MTZ) Formulation for TSP
**Type**: Integer Linear Programming (ILP)

### 2.1 Core Equations

**Objective Function (Minimization):**
$$ Z = \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij} $$
*Goal: Minimize total travel distance $Z$.*

**Constraints:**

1.  **Topology Constraint (Out-degree)**: Leave each city exactly once.
    $$ \sum_{j=1, j \neq i}^{n} x_{ij} = 1 \quad \forall i \in \{1, \dots, n \} $$

2.  **Topology Constraint (In-degree)**: Enter each city exactly once.
    $$ \sum_{i=1, i \neq j}^{n} x_{ij} = 1 \quad \forall j \in \{1, \dots, n \} $$

3.  **Subtour Elimination (MTZ)**: Prevent disconnected loops.
    $$ u_i - u_j + n x_{ij} \le n - 1 \quad \forall i, j \in \{2, \dots, n \}, i \neq j $$

4.  **Integrality**:
    $$ x_{ij} \in \{0, 1 \} $$

### 2.2 Variable Definitions

| Variable | Symbol | Type | Description |
|----------|--------|------|-------------|
| **Cost Matrix** | $c_{ij}$ | Constant ($u32$) | Driving distance from city $i$ to $j$ (miles). |
| **Decision** | $x_{ij}$ | Binary | 1 if path goes $i \to j$, 0 otherwise. |
| **Auxiliary** | $u_i$ | Integer | Order of visitation for city $i$ (MTZ). |
| **Size** | $n$ | Integer | Number of cities (user-configurable). |

## 3. YAML-First Architecture

### 3.1 Design Principle

**All configuration is YAML. Code reads YAML. Users modify YAML.**

```
User Experience:
┌──────────────────────────────────────────────────────────┐
│  1. Download bay_area_tsp.yaml                           │
│  2. Edit cities/distances in any text editor             │
│  3. Upload modified YAML (or paste in web editor)        │
│  4. Run solver → See results                             │
│  5. Compare different configurations                     │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Ground Truth Instance (YAML)

```yaml
# bay_area_tsp.yaml
# User-editable TSP instance - modify cities and distances freely
meta:
  id: "TSP-BAY-006"
  version: "1.0.0"
  description: "6-city Bay Area ground truth instance"
  source: "Google Maps (Dec 2024)"
  units: "miles"
  optimal_known: 115  # Known optimal for verification

cities:
  - id: 0
    name: "San Francisco"
    alias: "SF"
    coords: { lat: 37.7749, lon: -122.4194 }
  - id: 1
    name: "Oakland"
    alias: "OAK"
    coords: { lat: 37.8044, lon: -122.2712 }
  - id: 2
    name: "San Jose"
    alias: "SJ"
    coords: { lat: 37.3382, lon: -121.8863 }
  - id: 3
    name: "Palo Alto"
    alias: "PA"
    coords: { lat: 37.4419, lon: -122.1430 }
  - id: 4
    name: "Berkeley"
    alias: "BRK"
    coords: { lat: 37.8716, lon: -122.2727 }
  - id: 5
    name: "Fremont"
    alias: "FRE"
    coords: { lat: 37.5485, lon: -121.9886 }

# Distance Matrix (Row=From, Col=To)
# Symmetric matrix - users can modify any value
# Order: SF, OAK, SJ, PA, BRK, FRE
matrix:
  - [ 0, 12, 48, 35, 14, 42]  # From SF
  - [12,  0, 42, 30,  4, 30]  # From OAK
  - [48, 42,  0, 15, 46, 17]  # From SJ
  - [35, 30, 15,  0, 32, 18]  # From PA
  - [14,  4, 46, 32,  0, 32]  # From BRK
  - [42, 30, 17, 18, 32,  0]  # From FRE

# Algorithm configuration (user-selectable)
algorithm:
  method: "grasp"  # Options: greedy, grasp, brute_force
  params:
    rcl_size: 3        # Restricted Candidate List size
    restarts: 10       # Number of GRASP restarts
    two_opt: true      # Enable 2-opt local search
    seed: 42           # For reproducibility
```

### 3.3 User-Modifiable Components

| Component | File/Section | User Action |
|-----------|--------------|-------------|
| **Cities** | `cities:` array | Add/remove/rename cities |
| **Distances** | `matrix:` | Edit driving distances |
| **Algorithm** | `algorithm.method` | Switch: greedy/grasp/brute_force |
| **Parameters** | `algorithm.params` | Tune RCL size, restarts, seed |
| **Equations** | `docs/emc/*.yaml` | Modify EMC definitions |

## 4. Web Deployment (interactive.paiml.com)

### 4.1 User Interface Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│  TSP GRASP Demo - Bay Area                          [Run] [Reset]│
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│   YAML Editor   │           Visualization                       │
│   ───────────   │           ─────────────                       │
│   [Download]    │           ┌─────────────────────┐             │
│   [Upload]      │           │     Map View        │             │
│   [Examples ▼]  │           │   with Tour Path    │             │
│                 │           └─────────────────────┘             │
│   cities:       │                                               │
│     - SF        │           Tour: SF→OAK→BRK→FRE→SJ→PA→SF       │
│     - Oakland   │           Distance: 115 miles                 │
│     ...         │           Optimal: ✓ (verified)               │
│                 │                                               │
│   matrix:       │           ┌─────────────────────┐             │
│     [edit]      │           │  Convergence Plot   │             │
│                 │           └─────────────────────┘             │
├─────────────────┴───────────────────────────────────────────────┤
│  Equations (EMC-TSP-MTZ-001)                     [Show/Hide]    │
│  Z = Σᵢ Σⱼ cᵢⱼ xᵢⱼ   |   Constraints: In/Out degree = 1        │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Interactive Features

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **YAML Editor** | Monaco/CodeMirror with YAML syntax | Web component |
| **Download** | Export current config as .yaml file | Blob download |
| **Upload** | Load user's .yaml file | FileReader API |
| **Examples** | Dropdown: Bay Area, Random 10, Random 50 | Preset configs |
| **Equation Toggle** | Show/hide LaTeX-rendered EMC | MathJax/KaTeX |
| **Algorithm Switch** | Radio: Greedy / GRASP / Brute Force | Config update |
| **Live Validation** | Red border on invalid YAML | Schema check |

## 5. Implementation Tasks (pmat work)

### 5.1 Task Breakdown

Each task follows EXTREME TDD: Write failing test → Implement → Verify 100% coverage.

| Task ID | Description | Est. | Dependencies |
|---------|-------------|------|--------------|
| **OR-001-01** | Create `TspInstanceYaml` struct + serde | S | None |
| **OR-001-02** | Implement `TspGraspDemo::from_yaml()` | M | OR-001-01 |
| **OR-001-03** | Create `bay_area_tsp.yaml` ground truth | S | OR-001-01 |
| **OR-001-04** | Add Jidoka validators (triangle ineq, symmetry) | M | OR-001-02 |
| **OR-001-05** | Probar E2E: deterministic replay test | M | OR-001-03 |
| **OR-001-06** | WASM: expose `from_yaml()` to JS | M | OR-001-02 |
| **OR-001-07** | Web UI: YAML editor component | L | OR-001-06 |
| **OR-001-08** | Web UI: Download/Upload handlers | S | OR-001-07 |
| **OR-001-09** | Web UI: Algorithm selector | S | OR-001-07 |
| **OR-001-10** | Integration test: full user flow | M | OR-001-08, OR-001-09 |

### 5.2 Task Details

#### OR-001-01: TspInstanceYaml Struct

```rust
// src/demos/tsp_instance.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspInstanceYaml {
    pub meta: TspMeta,
    pub cities: Vec<TspCity>,
    pub matrix: Vec<Vec<u32>>,
    pub algorithm: TspAlgorithmConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspMeta {
    pub id: String,
    pub version: String,
    pub description: String,
    pub units: String,
    pub optimal_known: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspCity {
    pub id: usize,
    pub name: String,
    pub alias: String,
    pub coords: Coords,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspAlgorithmConfig {
    pub method: String,  // "greedy" | "grasp" | "brute_force"
    pub params: TspParams,
}
```

**Tests Required (100% coverage):**
- `test_deserialize_valid_yaml`
- `test_deserialize_minimal_yaml`
- `test_deserialize_invalid_missing_cities`
- `test_deserialize_invalid_matrix_size`
- `test_serialize_roundtrip`
- `test_default_algorithm_params`

#### OR-001-02: from_yaml() Implementation

```rust
impl TspGraspDemo {
    /// Load TSP instance from YAML configuration.
    ///
    /// # Errors
    /// Returns error if:
    /// - YAML parse fails
    /// - Matrix dimensions don't match city count
    /// - Triangle inequality violated (Jidoka)
    pub fn from_yaml(yaml: &str) -> Result<Self, TspError> {
        let config: TspInstanceYaml = serde_yaml::from_str(yaml)
            .map_err(|e| TspError::ParseError(e.to_string()))?;

        // Jidoka: Validate before proceeding
        Self::validate_instance(&config)?;

        Self::from_config(config)
    }

    /// Load from file path.
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, TspError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| TspError::IoError(e.to_string()))?;
        Self::from_yaml(&content)
    }
}
```

**Tests Required:**
- `test_from_yaml_bay_area_instance`
- `test_from_yaml_returns_optimal_115`
- `test_from_yaml_invalid_matrix_dimension`
- `test_from_yaml_triangle_inequality_violation`
- `test_from_yaml_asymmetric_matrix_warning`
- `test_from_yaml_file_not_found`
- `test_from_yaml_file_success`

#### OR-001-05: Probar E2E Test

```rust
// tests/probar_tsp.rs
//! Probar E2E tests for TSP YAML loading and deterministic replay.

use simular::demos::TspGraspDemo;

const BAY_AREA_YAML: &str = include_str!("../examples/experiments/bay_area_tsp.yaml");

#[test]
fn probar_tsp_bay_area_optimal() {
    // Load from YAML
    let demo = TspGraspDemo::from_yaml(BAY_AREA_YAML)
        .expect("YAML should parse");

    // Run to completion
    let result = demo.run_to_completion();

    // Verify optimal
    assert_eq!(result.best_tour_length, 115.0,
        "Bay Area optimal tour should be 115 miles");

    // Verify tour visits all cities exactly once
    assert_eq!(result.best_tour.len(), 6);
    let mut visited: Vec<usize> = result.best_tour.clone();
    visited.sort();
    assert_eq!(visited, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn probar_tsp_deterministic_replay() {
    // Two runs with same seed must produce identical results
    let demo1 = TspGraspDemo::from_yaml(BAY_AREA_YAML).unwrap();
    let demo2 = TspGraspDemo::from_yaml(BAY_AREA_YAML).unwrap();

    let result1 = demo1.run_to_completion();
    let result2 = demo2.run_to_completion();

    assert_eq!(result1.best_tour, result2.best_tour,
        "Deterministic replay failed");
    assert_eq!(result1.best_tour_length, result2.best_tour_length);
}

#[test]
fn probar_tsp_user_modified_yaml() {
    // Simulate user editing: change SF→Oakland distance from 12 to 5
    let modified_yaml = BAY_AREA_YAML.replace(
        "[ 0, 12, 48, 35, 14, 43]",
        "[ 0,  5, 48, 35, 14, 43]"
    );

    let demo = TspGraspDemo::from_yaml(&modified_yaml).unwrap();
    let result = demo.run_to_completion();

    // New optimal should be different (shorter via SF-Oakland)
    assert!(result.best_tour_length < 115.0,
        "Modified distances should yield shorter tour");
}
```

## 6. Toyota Production System Alignment

### 6.1 Jidoka (Stop-on-Error)

| Check | Condition | Action |
|-------|-----------|--------|
| Matrix Size | `matrix.len() != cities.len()` | ERROR: Halt |
| Triangle Inequality | `c[i][k] > c[i][j] + c[j][k]` | WARN: Log |
| Symmetry | `c[i][j] != c[j][i]` | WARN: Log (unless asymmetric flag) |
| Tour Validity | City visited twice | ERROR: Halt |
| NaN/Inf | Distance is not finite | ERROR: Halt |

### 6.2 Poka-Yoke (Mistake-Proofing)

| Guard | Implementation |
|-------|----------------|
| Type-safe city IDs | `CityId(usize)` newtype |
| Validated YAML schema | JSON Schema + runtime check |
| Immutable config after load | `TspInstanceYaml` is `Clone` but not `&mut` |

### 6.3 Genchi Genbutsu (Go and See)

Every run produces step-by-step verification output:

```
Step 1: San Francisco → Oakland = 12 miles (total: 12)
Step 2: Oakland → Berkeley = 4 miles (total: 16)
Step 3: Berkeley → Fremont = 32 miles (total: 48)
Step 4: Fremont → San Jose = 17 miles (total: 65)
Step 5: San Jose → Palo Alto = 15 miles (total: 80)
Step 6: Palo Alto → San Francisco = 35 miles (total: 115)

VERIFIED: Tour length 115 matches sum of edges ✓
VERIFIED: All 6 cities visited exactly once ✓
VERIFIED: Optimal known (115) achieved ✓
```

## 7. Falsification Criteria (Popperian)

| Criterion | Equation / Logic | Failure Mode |
|-----------|------------------|--------------|
| **F1: Validity** | `|tour| = n` and all unique | Disconnected tour |
| **F2: Cost** | `Z_calc = Σ c[edge]` | Arithmetic bug |
| **F3: Optimality** | `Z_algo ≤ optimal_known` | Impossible claim |
| **F4: Determinism** | `run(seed) = run(seed)` | RNG leak |
| **F5: YAML Integrity** | Modified YAML → different result | Config not used |

## 8. Peer-Reviewed References

1.  **Miller, C. E., Tucker, A. W., & Zemlin, R. A.** (1960). "Integer Programming Formulation of Traveling Salesman Problems." *Journal of the ACM*, 7(4), 326–329.
    *Relevance*: MTZ subtour elimination constraints.

2.  **Karp, R. M.** (1972). "Reducibility among combinatorial problems." In *Complexity of Computer Computations* (pp. 85-103). Springer.
    *Relevance*: TSP NP-completeness proof.

3.  **Feo, T. A., & Resende, M. G. C.** (1995). "Greedy Randomized Adaptive Search Procedures." *Journal of Global Optimization*, 6(2), 109-133.
    *Relevance*: GRASP metaheuristic foundation.

4.  **Lin, S., & Kernighan, B. W.** (1973). "An Effective Heuristic Algorithm for the Traveling-Salesman Problem." *Operations Research*, 21(2), 498-516.
    *Relevance*: k-opt local search methodology.

5.  **Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M.** (1977). "An Analysis of Several Heuristics for the Traveling Salesman Problem." *SIAM Journal on Computing*, 6(3), 563-581.
    *Relevance*: Nearest Neighbor performance bounds.

## 9. Acceptance Criteria

### 9.1 Functional

- [ ] **AC-1**: `TspGraspDemo::from_yaml()` parses `bay_area_tsp.yaml` correctly
- [ ] **AC-2**: Bay Area instance returns optimal 115 miles
- [ ] **AC-3**: User can modify YAML cities → solver uses modified data
- [ ] **AC-4**: User can switch algorithm via YAML → different behavior
- [ ] **AC-5**: Probar E2E tests pass with deterministic replay

### 9.2 Quality Gates

- [ ] **QG-1**: 100% test coverage on `tsp_instance.rs`
- [ ] **QG-2**: 100% test coverage on `from_yaml()` code paths
- [ ] **QG-3**: Zero clippy warnings
- [ ] **QG-4**: Probar E2E: 3+ deterministic replay scenarios
- [ ] **QG-5**: WASM bundle includes YAML parsing (<100KB overhead)

### 9.3 Web Deployment

- [ ] **WD-1**: YAML editor renders and validates
- [ ] **WD-2**: Download button exports current config
- [ ] **WD-3**: Upload accepts user .yaml files
- [ ] **WD-4**: Algorithm selector updates config
- [ ] **WD-5**: Equations display toggleable

## 10. Roadmap Integration

Add to `docs/roadmaps/roadmap.yaml`:

```yaml
- id: OR-001
  github_issue: null
  item_type: epic
  title: "Simple OR Example: Bay Area TSP Ground Truth"
  status: planning
  priority: high
  spec: docs/specifications/simple-or-example.md
  acceptance_criteria:
    - "AC-1: YAML loading works"
    - "AC-2: Optimal 115 miles verified"
    - "AC-3: User-modifiable cities"
    - "AC-4: Algorithm selection"
    - "AC-5: Probar E2E passes"
    - "QG-1: 100% coverage"
  subtasks:
    - id: OR-001-01
      title: "TspInstanceYaml struct + serde"
      status: pending
    - id: OR-001-02
      title: "from_yaml() implementation"
      status: pending
    - id: OR-001-03
      title: "bay_area_tsp.yaml ground truth"
      status: pending
    - id: OR-001-04
      title: "Jidoka validators"
      status: pending
    - id: OR-001-05
      title: "Probar E2E tests"
      status: pending
    - id: OR-001-06
      title: "WASM from_yaml() binding"
      status: pending
    - id: OR-001-07
      title: "Web UI: YAML editor"
      status: pending
    - id: OR-001-08
      title: "Web UI: Download/Upload"
      status: pending
    - id: OR-001-09
      title: "Web UI: Algorithm selector"
      status: pending
    - id: OR-001-10
      title: "Integration test: full flow"
      status: pending
  labels:
    - or
    - tsp
    - yaml-first
    - wasm
```

---

*"Make your workplace into a showcase that can be understood by everyone at a glance."*
— Taiichi Ohno, Toyota Production System