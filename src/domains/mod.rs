//! Domain-specific simulation engines.
//!
//! Each domain implements a specific type of simulation:
//! - Physics: Rigid body, orbital, fluid dynamics
//! - Monte Carlo: Stochastic sampling with variance reduction
//! - Optimization: Bayesian, evolutionary, gradient-based
//! - ML: Training simulation, prediction, multi-turn evaluation

pub mod physics;
pub mod monte_carlo;
pub mod optimization;
pub mod ml;

pub use physics::{PhysicsEngine, Integrator, VerletIntegrator, RK4Integrator, EulerIntegrator};
pub use monte_carlo::{MonteCarloEngine, MonteCarloResult, VarianceReduction};
pub use optimization::{BayesianOptimizer, OptimizerConfig, AcquisitionFunction, GaussianProcess};
pub use ml::{
    TrainingSimulation, TrainingConfig, TrainingState, TrainingMetrics, TrainingTrajectory,
    PredictionSimulation, PredictionState, InferenceConfig,
    MultiTurnSimulation, Turn, TurnMetrics, MultiTurnEvaluation, ParetoPoint, ParetoAnalysis,
    AnomalyDetector, TrainingAnomaly, RollingStats,
    JidokaMLFeedback, RulePatch, RuleType, AnomalyPattern, AnomalyType,
};
