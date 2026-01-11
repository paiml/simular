//! `ComputeBlock` integration for simular TUI
//!
//! Provides SIMD-optimized visualization widgets from presentar-terminal
//! for simulation metrics. Integrates with probar's `ComputeBlockAssertion`
//! for testability.
//!
//! # Architecture (SIMULAR-CB-001)
//!
//! ```text
//! OrbitApp State
//!     │
//!     ├─► EnergySparkline (SparklineBlock)
//!     │   └─► Normalized energy history graph
//!     │
//!     ├─► MomentumSparkline (SparklineBlock)
//!     │   └─► Normalized angular momentum history
//!     │
//!     └─► FrameBudgetTrend (LoadTrendBlock)
//!         └─► Heijunka utilization over time
//! ```
//!
//! # Toyota Way Integration
//!
//! - **Jidoka**: `SparklineBlocks` show conservation violations
//! - **Mieruka**: Visual management of simulation health
//! - **Heijunka**: Frame budget visualization

use presentar_terminal::{ComputeBlock, LoadTrendBlock, SimdInstructionSet, SparklineBlock};

/// Maximum history length for sparklines (SPEC-024 compliant)
const SPARKLINE_HISTORY: usize = 60;

/// Default render width for sparklines
const SPARKLINE_WIDTH: usize = 40;

/// Energy conservation sparkline using presentar-terminal's SIMD-optimized `SparklineBlock`
pub struct EnergySparkline {
    block: SparklineBlock,
    initial_energy: Option<f64>,
}

impl EnergySparkline {
    /// Create a new energy sparkline
    #[must_use]
    pub fn new() -> Self {
        Self {
            block: SparklineBlock::new(SPARKLINE_HISTORY),
            initial_energy: None,
        }
    }

    /// Push a new energy value
    pub fn push(&mut self, energy: f64) {
        // Store initial energy for normalization
        if self.initial_energy.is_none() {
            self.initial_energy = Some(energy);
        }

        // Compute relative energy drift (should stay near 0 for good conservation)
        let initial = self.initial_energy.unwrap_or(energy);
        #[allow(clippy::cast_possible_truncation)]
        let drift = if initial.abs() > f64::EPSILON {
            ((energy - initial) / initial.abs() * 1e6) as f32 // Parts per million
        } else {
            0.0
        };

        self.block.compute(&drift);
    }

    /// Get the rendered sparkline characters
    #[must_use]
    pub fn render(&self) -> Vec<char> {
        self.block.render(SPARKLINE_WIDTH)
    }

    /// Get the min/max range for display
    #[must_use]
    pub fn range(&self) -> (f32, f32) {
        let history = self.block.history();
        if history.is_empty() {
            return (0.0, 0.0);
        }
        let min = history.iter().copied().fold(f32::INFINITY, f32::min);
        let max = history.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Get current value
    #[must_use]
    pub fn current(&self) -> Option<f32> {
        self.block.history().last().copied()
    }

    /// Get history length
    #[must_use]
    pub fn len(&self) -> usize {
        self.block.history().len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block.history().is_empty()
    }

    /// Get SIMD instruction set being used
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.block.simd_instruction_set()
    }

    /// Reset the sparkline
    pub fn reset(&mut self) {
        self.block = SparklineBlock::new(SPARKLINE_HISTORY);
        self.initial_energy = None;
    }
}

impl Default for EnergySparkline {
    fn default() -> Self {
        Self::new()
    }
}

/// Angular momentum conservation sparkline
pub struct MomentumSparkline {
    block: SparklineBlock,
    initial_momentum: Option<f64>,
}

impl MomentumSparkline {
    /// Create a new momentum sparkline
    #[must_use]
    pub fn new() -> Self {
        Self {
            block: SparklineBlock::new(SPARKLINE_HISTORY),
            initial_momentum: None,
        }
    }

    /// Push a new angular momentum value
    pub fn push(&mut self, momentum: f64) {
        if self.initial_momentum.is_none() {
            self.initial_momentum = Some(momentum);
        }

        let initial = self.initial_momentum.unwrap_or(momentum);
        #[allow(clippy::cast_possible_truncation)]
        let drift = if initial.abs() > f64::EPSILON {
            ((momentum - initial) / initial.abs() * 1e6) as f32
        } else {
            0.0
        };

        self.block.compute(&drift);
    }

    /// Get the rendered sparkline characters
    #[must_use]
    pub fn render(&self) -> Vec<char> {
        self.block.render(SPARKLINE_WIDTH)
    }

    /// Get the min/max range
    #[must_use]
    pub fn range(&self) -> (f32, f32) {
        let history = self.block.history();
        if history.is_empty() {
            return (0.0, 0.0);
        }
        let min = history.iter().copied().fold(f32::INFINITY, f32::min);
        let max = history.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Get current value
    #[must_use]
    pub fn current(&self) -> Option<f32> {
        self.block.history().last().copied()
    }

    /// Get history length
    #[must_use]
    pub fn len(&self) -> usize {
        self.block.history().len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block.history().is_empty()
    }

    /// Get SIMD instruction set
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.block.simd_instruction_set()
    }

    /// Reset the sparkline
    pub fn reset(&mut self) {
        self.block = SparklineBlock::new(SPARKLINE_HISTORY);
        self.initial_momentum = None;
    }
}

impl Default for MomentumSparkline {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export `TrendDirection` from presentar-terminal
pub use presentar_terminal::compute_block::TrendDirection;

/// Frame budget utilization trend (Heijunka visualization)
pub struct FrameBudgetTrend {
    block: LoadTrendBlock,
    /// History of utilization values for sparkline rendering
    history: Vec<f32>,
    /// Sparkline for visual rendering
    sparkline: SparklineBlock,
}

impl FrameBudgetTrend {
    /// Create a new frame budget trend
    #[must_use]
    pub fn new() -> Self {
        Self {
            block: LoadTrendBlock::new(SPARKLINE_HISTORY),
            history: Vec::with_capacity(SPARKLINE_HISTORY),
            sparkline: SparklineBlock::new(SPARKLINE_HISTORY),
        }
    }

    /// Push a new utilization value (0.0 - 1.0+)
    pub fn push(&mut self, utilization: f64) {
        // LoadTrendBlock expects 0-100 scale
        #[allow(clippy::cast_possible_truncation)]
        let pct = (utilization * 100.0).clamp(0.0, 200.0) as f32;
        self.block.compute(&pct);
        self.sparkline.compute(&pct);

        // Track history for average calculation
        if self.history.len() >= SPARKLINE_HISTORY {
            self.history.remove(0);
        }
        self.history.push(pct);
    }

    /// Get the rendered trend characters (using sparkline)
    #[must_use]
    pub fn render(&self) -> Vec<char> {
        self.sparkline.render(SPARKLINE_WIDTH)
    }

    /// Get the average utilization
    #[must_use]
    pub fn average(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f32>() / self.history.len() as f32
    }

    /// Get trend direction
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        self.block.trend()
    }

    /// Get history length
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get SIMD instruction set
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.sparkline.simd_instruction_set()
    }

    /// Reset the trend
    pub fn reset(&mut self) {
        self.block = LoadTrendBlock::new(SPARKLINE_HISTORY);
        self.sparkline = SparklineBlock::new(SPARKLINE_HISTORY);
        self.history.clear();
    }
}

impl Default for FrameBudgetTrend {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined simulation metrics using `ComputeBlocks`
pub struct SimulationMetrics {
    /// Energy conservation sparkline
    pub energy: EnergySparkline,
    /// Angular momentum conservation sparkline
    pub momentum: MomentumSparkline,
    /// Frame budget utilization trend
    pub frame_budget: FrameBudgetTrend,
}

impl SimulationMetrics {
    /// Create new simulation metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            energy: EnergySparkline::new(),
            momentum: MomentumSparkline::new(),
            frame_budget: FrameBudgetTrend::new(),
        }
    }

    /// Update all metrics from simulation state
    pub fn update(&mut self, energy: f64, momentum: f64, frame_utilization: f64) {
        self.energy.push(energy);
        self.momentum.push(momentum);
        self.frame_budget.push(frame_utilization);
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.energy.reset();
        self.momentum.reset();
        self.frame_budget.reset();
    }

    /// Get the SIMD instruction set being used
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        // All blocks use the same detection
        self.energy.simd_instruction_set()
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_sparkline_creation() {
        let sparkline = EnergySparkline::new();
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_energy_sparkline_push() {
        let mut sparkline = EnergySparkline::new();
        sparkline.push(-1e30);
        assert!(!sparkline.is_empty());
        assert_eq!(sparkline.len(), 1);
    }

    #[test]
    fn test_energy_sparkline_drift_calculation() {
        let mut sparkline = EnergySparkline::new();
        sparkline.push(-1e30);
        sparkline.push(-1e30 * 1.000001); // 1 ppm drift

        let current = sparkline.current();
        assert!(current.is_some());
    }

    #[test]
    fn test_momentum_sparkline_creation() {
        let sparkline = MomentumSparkline::new();
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_momentum_sparkline_push() {
        let mut sparkline = MomentumSparkline::new();
        sparkline.push(1e40);
        assert_eq!(sparkline.len(), 1);
    }

    #[test]
    fn test_frame_budget_trend_creation() {
        let trend = FrameBudgetTrend::new();
        assert!(trend.is_empty());
    }

    #[test]
    fn test_frame_budget_trend_push() {
        let mut trend = FrameBudgetTrend::new();
        trend.push(0.5); // 50% utilization
        assert_eq!(trend.len(), 1);
    }

    #[test]
    fn test_simulation_metrics_creation() {
        let metrics = SimulationMetrics::new();
        assert!(metrics.energy.is_empty());
        assert!(metrics.momentum.is_empty());
        assert!(metrics.frame_budget.is_empty());
    }

    #[test]
    fn test_simulation_metrics_update() {
        let mut metrics = SimulationMetrics::new();
        metrics.update(-1e30, 1e40, 0.5);

        assert_eq!(metrics.energy.len(), 1);
        assert_eq!(metrics.momentum.len(), 1);
        assert_eq!(metrics.frame_budget.len(), 1);
    }

    #[test]
    fn test_simulation_metrics_reset() {
        let mut metrics = SimulationMetrics::new();
        metrics.update(-1e30, 1e40, 0.5);
        metrics.reset();

        assert!(metrics.energy.is_empty());
        assert!(metrics.momentum.is_empty());
        assert!(metrics.frame_budget.is_empty());
    }

    #[test]
    fn test_simd_detection() {
        let metrics = SimulationMetrics::new();
        let simd = metrics.simd_instruction_set();
        // Should detect something (at least Scalar)
        assert!(simd.vector_width() >= 1);
    }

    #[test]
    fn test_energy_sparkline_render() {
        let mut sparkline = EnergySparkline::new();
        for i in 0..10 {
            sparkline.push(-1e30 + (i as f64) * 1e20);
        }
        let chars = sparkline.render();
        assert!(!chars.is_empty());
    }

    #[test]
    fn test_defaults() {
        let _ = EnergySparkline::default();
        let _ = MomentumSparkline::default();
        let _ = FrameBudgetTrend::default();
        let _ = SimulationMetrics::default();
    }
}
