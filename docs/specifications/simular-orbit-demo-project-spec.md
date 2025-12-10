# Simular Orbit Demo: Technical Specification

**Document ID:** SIMULAR-SPEC-ORBIT-001
**Version:** 1.0.0
**Status:** Draft
**Classification:** Open Source
**Author:** PAIML Engineering
**Date:** 2025-12-10

---

## Executive Summary

This specification defines the **Simular Orbit Demo**, a canonical demonstration of the Simular simulation engine showcasing deterministic orbital mechanics with dual deployment targets: Terminal User Interface (TUI) for CLI environments and WebAssembly (WASM) for browser-based visualization. The implementation adheres to Toyota Production System (TPS) quality principles and NASA/JPL mission-critical software standards.

---

## 1. Introduction

### 1.1 Purpose

The Simular Orbit Demo serves as the reference implementation demonstrating:

1. **Deterministic Reproducibility** — Bit-identical results across platforms given identical seeds
2. **Jidoka Quality Gates** — Automatic anomaly detection (energy drift, numerical instability)
3. **Dual-Target Architecture** — Single codebase compiling to native TUI and WASM
4. **Physics Fidelity** — Symplectic integration preserving orbital invariants

### 1.2 Scope

This specification covers:

- Mathematical model of N-body gravitational dynamics
- Numerical integration methods with error bounds
- Visualization requirements for TUI and WebGL/Canvas
- Quality assurance through Jidoka guards
- Build and deployment pipeline

### 1.3 Design Philosophy

#### 1.3.1 Toyota Production System (TPS) Principles

| Principle | Application |
|-----------|-------------|
| **Jidoka** (自働化) | Automatic halt on numerical anomaly detection |
| **Poka-Yoke** (ポカヨケ) | Type-safe units preventing dimensional errors |
| **Heijunka** (平準化) | Load-balanced computation across simulation steps |
| **Genchi Genbutsu** (現地現物) | Direct observation via replay system |
| **Kaizen** (改善) | Continuous improvement through metrics |

#### 1.3.2 NASA/JPL Power of 10 Rules [1]

1. Simple control flow (max cyclomatic complexity: 15)
2. Fixed upper bounds on loops
3. No dynamic memory allocation after initialization
4. Maximum function length: 60 lines
5. Minimum 2 assertions per function
6. Data hiding at smallest scope
7. All return values checked
8. Limited preprocessor/macro usage
9. Restrict pointer arithmetic (enforced by Safe Rust)
10. Compile with all warnings enabled, treat as errors

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     simular-orbit-demo                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scenarios │  │   Physics   │  │      Visualization      │  │
│  │             │  │   Engine    │  │                         │  │
│  │ - Kepler    │  │             │  │  ┌─────────┐ ┌───────┐  │  │
│  │ - N-body    │  │ - Verlet    │  │  │   TUI   │ │ WASM  │  │  │
│  │ - Hohmann   │  │ - RK4       │  │  │(ratatui)│ │(wasm- │  │  │
│  │ - Lagrange  │  │ - Yoshida   │  │  │         │ │bindgen│  │  │
│  │             │  │             │  │  └─────────┘ └───────┘  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Core Engine (simular)                    ││
│  │  SimState │ SimRng │ JidokaGuard │ ReplaySystem │ Clock    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
simular-orbit-demo/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Core simulation (no_std compatible)
│   ├── scenarios/
│   │   ├── mod.rs
│   │   ├── kepler.rs       # Two-body Keplerian orbits
│   │   ├── n_body.rs       # N-body gravitational simulation
│   │   ├── hohmann.rs      # Hohmann transfer maneuvers
│   │   └── lagrange.rs     # Lagrange point dynamics
│   ├── physics/
│   │   ├── mod.rs
│   │   ├── gravity.rs      # Gravitational force computation
│   │   ├── integrators.rs  # Symplectic integrators
│   │   └── units.rs        # Type-safe physical units
│   ├── jidoka/
│   │   ├── mod.rs
│   │   ├── energy.rs       # Energy conservation monitor
│   │   ├── angular.rs      # Angular momentum conservation
│   │   └── stability.rs    # Numerical stability checks
│   └── render/
│       ├── mod.rs
│       ├── commands.rs     # Platform-agnostic render commands
│       └── camera.rs       # View transformation
├── src/bin/
│   └── orbit-tui.rs        # TUI binary entry point
├── src/wasm/
│   └── lib.rs              # WASM entry point
└── tests/
    ├── determinism.rs      # Reproducibility verification
    ├── conservation.rs     # Physics invariant tests
    └── integration.rs      # End-to-end scenarios
```

### 2.3 Compilation Targets

| Target | Toolchain | Output | Use Case |
|--------|-----------|--------|----------|
| `x86_64-unknown-linux-gnu` | stable | Native binary | CLI/TUI |
| `x86_64-apple-darwin` | stable | Native binary | CLI/TUI |
| `wasm32-unknown-unknown` | stable | WASM module | Browser |

---

## 3. Mathematical Model

### 3.1 Gravitational N-Body Problem

The gravitational N-body problem describes the motion of N point masses under mutual gravitational attraction [2, 3].

#### 3.1.1 Equations of Motion

For body $i$ with mass $m_i$ at position $\mathbf{r}_i$:

$$\ddot{\mathbf{r}}_i = -G \sum_{j \neq i} \frac{m_j (\mathbf{r}_i - \mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$$

Where $G = 6.67430 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$ is the gravitational constant [4].

#### 3.1.2 Hamiltonian Formulation

The system Hamiltonian [5]:

$$H = \sum_{i=1}^{N} \frac{|\mathbf{p}_i|^2}{2m_i} - G \sum_{i<j} \frac{m_i m_j}{|\mathbf{r}_i - \mathbf{r}_j|}$$

Conservation of $H$ serves as a Jidoka invariant for numerical validation.

### 3.2 Orbital Elements

Keplerian orbital elements [6, 7]:

| Element | Symbol | Description |
|---------|--------|-------------|
| Semi-major axis | $a$ | Size of orbit |
| Eccentricity | $e$ | Shape (0=circle, <1=ellipse) |
| Inclination | $i$ | Tilt from reference plane |
| RAAN | $\Omega$ | Right ascension of ascending node |
| Argument of periapsis | $\omega$ | Orientation in orbital plane |
| True anomaly | $\nu$ | Position along orbit |

### 3.3 Kepler's Laws

1. **First Law**: Orbits are conic sections with primary at focus
2. **Second Law**: Equal areas swept in equal times (angular momentum conservation)
3. **Third Law**: $T^2 \propto a^3$ (period-semimajor axis relation)

---

## 4. Numerical Methods

### 4.1 Integrator Selection Criteria

Per NASA/JPL standards for orbital mechanics [8, 9]:

| Criterion | Requirement |
|-----------|-------------|
| Symplecticity | Required for long-term stability |
| Time-reversibility | Required for energy conservation |
| Error order | Minimum 4th order |
| Adaptive stepping | Optional (fixed step for reproducibility) |

### 4.2 Velocity Verlet (Störmer-Verlet)

The Störmer-Verlet method [10, 11] is symplectic and time-reversible:

```
x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt²
a_{n+1} = acceleration(x_{n+1})
v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
```

**Properties:**
- Order: 2 (position), 2 (velocity)
- Symplectic: Yes
- Time-reversible: Yes
- Energy drift: $O(dt^2)$ bounded oscillation [12]

### 4.3 4th-Order Yoshida Integrator

The Yoshida 4th-order method [13] extends Störmer-Verlet:

```rust
const W0: f64 = -1.702414383919315;  // -(2^(1/3))/(2 - 2^(1/3))
const W1: f64 = 1.351207191959657;   // 1/(2 - 2^(1/3))
const C: [f64; 4] = [W1/2.0, (W0+W1)/2.0, (W0+W1)/2.0, W1/2.0];
const D: [f64; 3] = [W1, W0, W1];
```

**Properties:**
- Order: 4
- Symplectic: Yes
- Time-reversible: Yes
- Computational cost: 3× Verlet per step

### 4.4 Regularization for Close Encounters

For handling close approaches (softening parameter) [14]:

$$F_{ij} = \frac{G m_i m_j}{(|\mathbf{r}_{ij}|^2 + \epsilon^2)^{3/2}} \hat{\mathbf{r}}_{ij}$$

Where $\epsilon$ is the softening length, typically $\epsilon \ll$ minimum separation.

---

## 5. Jidoka Quality Gates

### 5.1 Energy Conservation Monitor

Total mechanical energy must remain bounded [15]:

```rust
pub struct EnergyGuard {
    initial_energy: f64,
    tolerance: f64,      // Default: 1e-6 (relative)
    max_drift: f64,      // Accumulated drift threshold
}

impl EnergyGuard {
    pub fn check(&mut self, state: &SimState) -> Result<(), JidokaViolation> {
        let current = state.total_energy();
        let relative_error = (current - self.initial_energy).abs() / self.initial_energy.abs();

        if relative_error > self.tolerance {
            return Err(JidokaViolation::EnergyDrift {
                initial: self.initial_energy,
                current,
                relative_error,
                threshold: self.tolerance,
            });
        }
        Ok(())
    }
}
```

### 5.2 Angular Momentum Conservation

Total angular momentum $\mathbf{L} = \sum_i m_i \mathbf{r}_i \times \mathbf{v}_i$ is conserved [16]:

```rust
pub struct AngularMomentumGuard {
    initial_momentum: Vec3,
    tolerance: f64,
}
```

### 5.3 Numerical Stability Checks

Per IEEE 754-2019 [17]:

```rust
pub fn check_finite(state: &SimState) -> Result<(), JidokaViolation> {
    for (i, pos) in state.positions().iter().enumerate() {
        if !pos.is_finite() {
            return Err(JidokaViolation::NonFinite {
                body_index: i,
                field: "position",
                value: *pos,
            });
        }
    }
    // ... check velocities, accelerations
    Ok(())
}
```

### 5.4 Collision Detection

Minimum separation threshold [18]:

```rust
pub fn check_collision(state: &SimState, min_separation: f64) -> Result<(), JidokaViolation> {
    for i in 0..state.num_bodies() {
        for j in (i+1)..state.num_bodies() {
            let separation = (state.positions()[i] - state.positions()[j]).magnitude();
            if separation < min_separation {
                return Err(JidokaViolation::Collision { body_i: i, body_j: j, separation });
            }
        }
    }
    Ok(())
}
```

---

## 6. Visualization

### 6.1 Platform-Agnostic Render Commands

Following the command pattern [19] for platform abstraction:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RenderCommand {
    Clear { color: Color },
    DrawCircle { x: f64, y: f64, radius: f64, color: Color },
    DrawLine { x1: f64, y1: f64, x2: f64, y2: f64, color: Color, width: f64 },
    DrawOrbitPath { points: Vec<(f64, f64)>, color: Color },
    DrawText { x: f64, y: f64, text: String, font_size: f64, color: Color },
    DrawVelocityVector { x: f64, y: f64, vx: f64, vy: f64, scale: f64, color: Color },
    SetCamera { center_x: f64, center_y: f64, zoom: f64 },
}
```

### 6.2 TUI Visualization (ratatui)

Terminal rendering using Braille characters for sub-cell precision [20]:

```rust
// Each terminal cell = 2×4 Braille dots = 8 sub-pixels
// Resolution: (cols * 2) × (rows * 4) virtual pixels

pub struct TuiRenderer {
    width: u16,
    height: u16,
    buffer: Vec<Vec<bool>>,  // Braille dot buffer
}

impl TuiRenderer {
    pub fn render_to_braille(&self) -> Vec<char> {
        // Convert 2×4 dot patterns to Unicode Braille (U+2800-U+28FF)
    }
}
```

**TUI Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  SIMULAR ORBIT DEMO                              [P]ause [R]eset│
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                           ⣿⣿                                    │
│                      ⣤⣤⣤⣿⣿⣿⣤⣤⣤                                │
│                   ⣤⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⣤                              │
│                ⣤⣤⣿⣿⣿⣿⣿⣿  ☉  ⣿⣿⣿⣿⣿⣤⣤                          │
│               ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿                           │
│                ⣤⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⣤    🌍                       │
│                   ⣤⣤⣿⣿⣿⣿⣿⣿⣿⣤⣤                                │
│                      ⣤⣤⣤⣿⣿⣤⣤⣤                                 │
│                           ⣿⣿                                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Time: 365.24d  Energy: -2.341e9 J (Δ: 1.2e-9)  Step: 0.001s    │
│ Bodies: Earth (1.0 AU, e=0.017)  Seed: 42  FPS: 60             │
│ Jidoka: ✓ Energy ✓ Angular Momentum ✓ Finite ✓ Separation      │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 WASM Visualization (Canvas/WebGL)

Browser rendering via wasm-bindgen [21]:

```rust
#[wasm_bindgen]
pub struct OrbitDemo {
    simulation: Simulation,
    render_commands: Vec<RenderCommand>,
}

#[wasm_bindgen]
impl OrbitDemo {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<OrbitDemo, JsValue> {
        // Parse config, initialize simulation
    }

    #[wasm_bindgen]
    pub fn tick(&mut self, dt: f64) -> Result<JsValue, JsValue> {
        // Step simulation, return render commands as JSON
        self.simulation.step(dt)?;
        self.render_commands = self.simulation.render();
        Ok(serde_wasm_bindgen::to_value(&self.render_commands)?)
    }

    #[wasm_bindgen]
    pub fn get_state(&self) -> JsValue {
        // Return simulation state for UI display
    }
}
```

---

## 7. Scenarios

### 7.1 Kepler Two-Body

Classic Earth-Sun system demonstrating Kepler's laws:

```rust
pub struct KeplerScenario {
    sun_mass: f64,      // 1.989e30 kg
    planet_mass: f64,   // 5.972e24 kg
    semi_major_axis: f64,
    eccentricity: f64,
}
```

### 7.2 N-Body Solar System

Inner solar system (Sun + 4 planets) [22]:

| Body | Mass (kg) | Semi-major axis (AU) | Eccentricity |
|------|-----------|---------------------|--------------|
| Sun | 1.989×10³⁰ | — | — |
| Mercury | 3.301×10²³ | 0.387 | 0.206 |
| Venus | 4.867×10²⁴ | 0.723 | 0.007 |
| Earth | 5.972×10²⁴ | 1.000 | 0.017 |
| Mars | 6.417×10²³ | 1.524 | 0.093 |

### 7.3 Hohmann Transfer

Minimum-energy transfer between circular orbits [23]:

```rust
pub struct HohmannTransfer {
    r1: f64,  // Initial orbit radius
    r2: f64,  // Target orbit radius
    mu: f64,  // Standard gravitational parameter
}

impl HohmannTransfer {
    pub fn delta_v1(&self) -> f64 {
        // First burn: raise apoapsis
        let v_circular = (self.mu / self.r1).sqrt();
        let v_transfer = (2.0 * self.mu * self.r2 / (self.r1 * (self.r1 + self.r2))).sqrt();
        v_transfer - v_circular
    }

    pub fn delta_v2(&self) -> f64 {
        // Second burn: circularize
        let v_transfer = (2.0 * self.mu * self.r1 / (self.r2 * (self.r1 + self.r2))).sqrt();
        let v_circular = (self.mu / self.r2).sqrt();
        v_circular - v_transfer
    }
}
```

### 7.4 Lagrange Points

L1-L5 dynamics in restricted three-body problem [24]:

```rust
pub enum LagrangePoint { L1, L2, L3, L4, L5 }

pub struct LagrangeScenario {
    primary_mass: f64,
    secondary_mass: f64,
    point: LagrangePoint,
    perturbation: Vec3,  // Initial displacement for stability demo
}
```

---

## 8. Determinism Requirements

### 8.1 Reproducibility Guarantees

Per Simular core requirements:

1. **Seed Determinism**: Identical seed → bit-identical trajectory
2. **Platform Independence**: Same results on x86_64, ARM64, WASM
3. **Compiler Independence**: Same results across Rust compiler versions

### 8.2 Implementation Constraints

```rust
// REQUIRED: Use SimRng for all randomness
let mut rng = SimRng::new(seed);

// REQUIRED: Fixed-point or strictly ordered floating-point operations
// FORBIDDEN: HashMap iteration order dependency
// FORBIDDEN: Parallel execution with race conditions
// FORBIDDEN: System time dependency in simulation logic
```

### 8.3 Verification Tests

```rust
#[test]
fn test_reproducibility_across_runs() {
    let seed = 42u64;

    let trajectory_1 = run_simulation(seed, 1000);
    let trajectory_2 = run_simulation(seed, 1000);

    assert_eq!(trajectory_1.len(), trajectory_2.len());
    for (state_1, state_2) in trajectory_1.iter().zip(trajectory_2.iter()) {
        assert_eq!(state_1.positions(), state_2.positions());
        assert_eq!(state_1.velocities(), state_2.velocities());
    }
}

#[test]
fn test_determinism_wasm_native_parity() {
    // Compare WASM and native execution for identical seeds
}
```

---

## 9. API Reference

### 9.1 Core Types

```rust
/// Simulation configuration
pub struct OrbitConfig {
    pub scenario: ScenarioType,
    pub integrator: IntegratorType,
    pub dt: f64,                    // Time step (seconds)
    pub seed: u64,                  // RNG seed for reproducibility
    pub jidoka: JidokaConfig,       // Quality guard configuration
}

/// Scenario selection
pub enum ScenarioType {
    Kepler { eccentricity: f64 },
    NBody { bodies: Vec<BodyConfig> },
    Hohmann { r1: f64, r2: f64 },
    Lagrange { point: LagrangePoint },
}

/// Integrator selection
pub enum IntegratorType {
    Verlet,
    Yoshida4,
    RK4,  // Non-symplectic, for comparison
}
```

### 9.2 Simulation Interface

```rust
pub trait OrbitSimulation {
    fn new(config: OrbitConfig) -> SimResult<Self> where Self: Sized;
    fn step(&mut self, dt: f64) -> SimResult<()>;
    fn state(&self) -> &SimState;
    fn time(&self) -> f64;
    fn render(&self) -> Vec<RenderCommand>;
    fn jidoka_status(&self) -> JidokaStatus;
}
```

---

## 10. Build and Deployment

### 10.1 Build Commands

```makefile
# Native TUI build
build-tui:
    cargo build --release --bin orbit-tui --features tui

# WASM build
build-wasm:
    wasm-pack build --target web --out-dir pkg src/wasm

# Run TUI demo
run-tui:
    cargo run --release --bin orbit-tui --features tui

# Serve WASM demo locally
serve-wasm:
    python3 -m http.server 8080 --directory web/
```

### 10.2 Feature Flags

```toml
[features]
default = []
tui = ["ratatui", "crossterm"]
wasm = ["wasm-bindgen", "serde-wasm-bindgen", "console_error_panic_hook"]
full = ["tui", "wasm"]
```

### 10.3 Deployment

| Target | Location | Method |
|--------|----------|--------|
| TUI | crates.io | `cargo install simular-orbit-demo` |
| WASM | interactive.paiml.com | S3 + CloudFront |

---

## 11. Quality Assurance

### 11.1 Test Coverage Requirements

| Category | Coverage Target |
|----------|-----------------|
| Unit tests | ≥95% |
| Integration tests | ≥90% |
| Property tests | 100 cases minimum |
| Mutation testing | ≥80% kill rate |

### 11.2 Continuous Integration

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        target: [native, wasm32-unknown-unknown]
    steps:
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo llvm-cov --fail-under 95
```

### 11.3 Acceptance Criteria

1. **AC-1**: Earth orbit completes in 365.25 ± 0.01 simulation days
2. **AC-2**: Energy drift < 1e-9 over 100 orbits (Yoshida integrator)
3. **AC-3**: Angular momentum conserved to 1e-12 relative precision
4. **AC-4**: WASM bundle size < 500KB (gzipped)
5. **AC-5**: TUI renders at ≥30 FPS on standard terminal
6. **AC-6**: Identical trajectories on native and WASM targets

---

## 12. References

[1] G. J. Holzmann, "The Power of 10: Rules for Developing Safety-Critical Code," *IEEE Computer*, vol. 39, no. 6, pp. 95-97, June 2006. doi:10.1109/MC.2006.212

[2] V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed. New York: Springer-Verlag, 1989. ISBN: 978-0-387-96890-2

[3] S. J. Aarseth, *Gravitational N-Body Simulations: Tools and Algorithms*. Cambridge University Press, 2003. doi:10.1017/CBO9780511535246

[4] P. J. Mohr, D. B. Newell, and B. N. Taylor, "CODATA Recommended Values of the Fundamental Physical Constants: 2014," *Reviews of Modern Physics*, vol. 88, no. 3, 035009, 2016. doi:10.1103/RevModPhys.88.035009

[5] H. Goldstein, C. Poole, and J. Safko, *Classical Mechanics*, 3rd ed. San Francisco: Addison Wesley, 2002. ISBN: 978-0-201-65702-9

[6] R. R. Bate, D. D. Mueller, and J. E. White, *Fundamentals of Astrodynamics*. New York: Dover Publications, 1971. ISBN: 978-0-486-60061-1

[7] D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. Microcosm Press, 2013. ISBN: 978-1-881883-18-0

[8] E. Hairer, C. Lubich, and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. Berlin: Springer, 2006. doi:10.1007/3-540-30666-8

[9] J. Laskar and M. Gastineau, "Existence of collisional trajectories of Mercury, Mars and Venus with the Earth," *Nature*, vol. 459, pp. 817-819, 2009. doi:10.1038/nature08096

[10] L. Verlet, "Computer 'Experiments' on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules," *Physical Review*, vol. 159, no. 1, pp. 98-103, 1967. doi:10.1103/PhysRev.159.98

[11] W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, "A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules," *Journal of Chemical Physics*, vol. 76, no. 1, pp. 637-649, 1982. doi:10.1063/1.442716

[12] R. D. Ruth, "A Canonical Integration Technique," *IEEE Transactions on Nuclear Science*, vol. 30, no. 4, pp. 2669-2671, 1983. doi:10.1109/TNS.1983.4332919

[13] H. Yoshida, "Construction of higher order symplectic integrators," *Physics Letters A*, vol. 150, no. 5-7, pp. 262-268, 1990. doi:10.1016/0375-9601(90)90092-3

[14] S. J. Aarseth, "Direct N-body Calculations," in *Multiple Time Scales*, J. U. Brackbill and B. I. Cohen, Eds. Academic Press, 1985, pp. 377-418.

[15] J. M. Sanz-Serna and M. P. Calvo, *Numerical Hamiltonian Problems*. London: Chapman and Hall, 1994. ISBN: 978-0-412-54290-4

[16] L. D. Landau and E. M. Lifshitz, *Mechanics*, 3rd ed. Oxford: Butterworth-Heinemann, 1976. ISBN: 978-0-7506-2896-9

[17] IEEE Computer Society, "IEEE Standard for Floating-Point Arithmetic," *IEEE Std 754-2019*, 2019. doi:10.1109/IEEESTD.2019.8766229

[18] P. Hut and J. N. Bahcall, "Binary-single star scattering. I. Numerical experiments for equal masses," *The Astrophysical Journal*, vol. 268, pp. 319-341, 1983. doi:10.1086/160956

[19] E. Gamma, R. Helm, R. Johnson, and J. Vlissides, *Design Patterns: Elements of Reusable Object-Oriented Software*. Reading, MA: Addison-Wesley, 1994. ISBN: 978-0-201-63361-0

[20] D. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, 3rd ed. Addison-Wesley, 1997. ISBN: 978-0-201-89684-8

[21] A. Crichton et al., "The `wasm-bindgen` Guide," Rust and WebAssembly Working Group, 2023. [Online]. Available: https://rustwasm.github.io/docs/wasm-bindgen/

[22] E. V. Pitjeva and N. P. Pitjev, "Masses of the Main Asteroid Belt and the Kuiper Belt from the Motions of Planets and Spacecraft," *Solar System Research*, vol. 52, no. 2, pp. 145-159, 2018. doi:10.1134/S0038094618020058

[23] W. Hohmann, *Die Erreichbarkeit der Himmelskörper* (The Attainability of Heavenly Bodies). Munich: Oldenbourg, 1925. NASA Technical Translation F-44, 1960.

[24] V. Szebehely, *Theory of Orbits: The Restricted Problem of Three Bodies*. New York: Academic Press, 1967. ISBN: 978-0-12-680650-9

[25] J. Wisdom and M. Holman, "Symplectic maps for the n-body problem," *The Astronomical Journal*, vol. 102, no. 4, pp. 1528-1538, 1991. doi:10.1086/115978

---

## Appendix A: Physical Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Gravitational constant | $G$ | 6.67430×10⁻¹¹ | m³ kg⁻¹ s⁻² |
| Astronomical unit | AU | 1.495978707×10¹¹ | m |
| Solar mass | $M_☉$ | 1.98892×10³⁰ | kg |
| Earth mass | $M_⊕$ | 5.9722×10²⁴ | kg |
| Earth orbital period | $T_⊕$ | 365.256363004 | days |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Jidoka** | Toyota's "automation with a human touch" - stopping production when defects detected |
| **Poka-Yoke** | Mistake-proofing through design constraints |
| **Symplectic** | Structure-preserving in phase space, conserves volume |
| **WASM** | WebAssembly - portable binary instruction format |
| **TUI** | Terminal User Interface |

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-10 | PAIML Engineering | Initial specification |

---

*This specification is released under MIT License. Contributions welcome via pull request.*
