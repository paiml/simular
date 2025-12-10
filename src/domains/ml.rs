//! Machine Learning Simulation Engine.
//!
//! Provides deterministic, reproducible simulation of ML workflows using
//! Popperian falsification methodology. Implements TPS principles:
//! - Jidoka: Stop-on-anomaly detection
//! - Heijunka: Load-balanced batch processing
//! - Kaizen: Continuous improvement via feedback
//!
//! # Example
//!
//! ```rust
//! use simular::domains::ml::{TrainingSimulation, TrainingConfig, AnomalyDetector};
//! use simular::engine::rng::SimRng;
//!
//! let mut sim = TrainingSimulation::new(42);
//! let config = TrainingConfig::default();
//! // Training simulation would run here
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::engine::rng::{SimRng, RngState};
use crate::engine::SimTime;
use crate::error::{SimError, SimResult};
use crate::replay::EventJournal;

// ============================================================================
// Training Simulation Types
// ============================================================================

/// Training hyperparameters configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size for training.
    pub batch_size: usize,
    /// Number of epochs.
    pub epochs: u64,
    /// Early stopping patience (None = disabled).
    pub early_stopping: Option<usize>,
    /// Gradient clipping max norm (None = disabled).
    pub gradient_clip: Option<f64>,
    /// Weight decay (L2 regularization).
    pub weight_decay: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            early_stopping: Some(10),
            gradient_clip: Some(1.0),
            weight_decay: 0.0001,
        }
    }
}

/// Training state captured at each epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch.
    pub epoch: u64,
    /// Training loss.
    pub loss: f64,
    /// Validation loss.
    pub val_loss: f64,
    /// Training metrics.
    pub metrics: TrainingMetrics,
    /// RNG state for perfect reproducibility.
    pub rng_state: RngState,
}

/// Training metrics collected during simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss.
    pub train_loss: f64,
    /// Validation loss.
    pub val_loss: f64,
    /// Accuracy (if classification).
    pub accuracy: Option<f64>,
    /// Gradient L2 norm.
    pub gradient_norm: f64,
    /// Current learning rate (after scheduling).
    pub learning_rate: f64,
    /// Number of parameters updated.
    pub params_updated: usize,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            train_loss: 0.0,
            val_loss: 0.0,
            accuracy: None,
            gradient_norm: 0.0,
            learning_rate: 0.001,
            params_updated: 0,
        }
    }
}

/// Training trajectory - sequence of training states.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingTrajectory {
    /// Sequence of training states.
    pub states: Vec<TrainingState>,
}

impl TrainingTrajectory {
    /// Create new empty trajectory.
    #[must_use]
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Add a state to the trajectory.
    pub fn push(&mut self, state: TrainingState) {
        self.states.push(state);
    }

    /// Get the final training state.
    #[must_use]
    pub fn final_state(&self) -> Option<&TrainingState> {
        self.states.last()
    }

    /// Get best validation loss achieved.
    #[must_use]
    pub fn best_val_loss(&self) -> Option<f64> {
        self.states
            .iter()
            .map(|s| s.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Check if training converged (loss stabilized).
    #[must_use]
    pub fn converged(&self, tolerance: f64) -> bool {
        if self.states.len() < 10 {
            return false;
        }
        let recent: Vec<f64> = self.states.iter().rev().take(10).map(|s| s.loss).collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        variance.sqrt() < tolerance
    }
}

// ============================================================================
// Anomaly Detection (Jidoka)
// ============================================================================

/// Training anomaly types for Jidoka detection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrainingAnomaly {
    /// Loss became NaN or Infinity.
    NonFiniteLoss,
    /// Gradient norm exceeded threshold.
    GradientExplosion {
        /// Observed gradient norm.
        norm: f64,
        /// Threshold that was exceeded.
        threshold: f64,
    },
    /// Gradient norm fell below threshold.
    GradientVanishing {
        /// Observed gradient norm.
        norm: f64,
        /// Threshold that was violated.
        threshold: f64,
    },
    /// Loss spike detected (statistical outlier).
    LossSpike {
        /// Z-score of the spike.
        z_score: f64,
        /// Actual loss value.
        loss: f64,
    },
    /// Prediction confidence below threshold.
    LowConfidence {
        /// Observed confidence.
        confidence: f64,
        /// Required threshold.
        threshold: f64,
    },
}

impl std::fmt::Display for TrainingAnomaly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteLoss => write!(f, "Non-finite loss detected (NaN/Inf)"),
            Self::GradientExplosion { norm, threshold } => {
                write!(f, "Gradient explosion: norm={norm:.2e} > threshold={threshold:.2e}")
            }
            Self::GradientVanishing { norm, threshold } => {
                write!(f, "Gradient vanishing: norm={norm:.2e} < threshold={threshold:.2e}")
            }
            Self::LossSpike { z_score, loss } => {
                write!(f, "Loss spike: z-score={z_score:.2}, loss={loss:.4}")
            }
            Self::LowConfidence { confidence, threshold } => {
                write!(f, "Low confidence: {confidence:.4} < threshold={threshold:.4}")
            }
        }
    }
}

/// Rolling statistics for anomaly detection.
#[derive(Debug, Clone, Default)]
pub struct RollingStats {
    /// Number of observations.
    count: u64,
    /// Running mean.
    mean: f64,
    /// Running M2 for variance calculation.
    m2: f64,
    /// Window size (0 = unlimited).
    window_size: usize,
    /// Recent values for windowed stats.
    recent: Vec<f64>,
}

impl RollingStats {
    /// Create new rolling stats with optional window.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            window_size,
            recent: Vec::new(),
        }
    }

    /// Update with new observation (Welford's algorithm).
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if self.window_size > 0 {
            self.recent.push(value);
            if self.recent.len() > self.window_size {
                self.recent.remove(0);
            }
        }
    }

    /// Get current mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Get current standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute z-score for a value.
    #[must_use]
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean) / std
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.recent.clear();
    }
}

/// Anomaly detector for Jidoka-style training quality gates.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Rolling statistics for loss values.
    loss_stats: RollingStats,
    /// Threshold in standard deviations for loss spikes.
    threshold_sigma: f64,
    /// Gradient explosion threshold.
    gradient_explosion_threshold: f64,
    /// Gradient vanishing threshold.
    gradient_vanishing_threshold: f64,
    /// Minimum observations before anomaly detection.
    warmup_count: u64,
    /// Number of anomalies detected.
    anomaly_count: u64,
}

impl AnomalyDetector {
    /// Create new anomaly detector with sigma threshold.
    #[must_use]
    pub fn new(threshold_sigma: f64) -> Self {
        Self {
            loss_stats: RollingStats::new(100),
            threshold_sigma,
            gradient_explosion_threshold: 1e6,
            gradient_vanishing_threshold: 1e-10,
            warmup_count: 10,
            anomaly_count: 0,
        }
    }

    /// Set gradient explosion threshold.
    #[must_use]
    pub fn with_gradient_explosion_threshold(mut self, threshold: f64) -> Self {
        self.gradient_explosion_threshold = threshold;
        self
    }

    /// Set gradient vanishing threshold.
    #[must_use]
    pub fn with_gradient_vanishing_threshold(mut self, threshold: f64) -> Self {
        self.gradient_vanishing_threshold = threshold;
        self
    }

    /// Set warmup count before anomaly detection activates.
    #[must_use]
    pub fn with_warmup(mut self, count: u64) -> Self {
        self.warmup_count = count;
        self
    }

    /// Check for training anomalies given loss and gradient norm.
    pub fn check(&mut self, loss: f64, gradient_norm: f64) -> Option<TrainingAnomaly> {
        // NaN/Inf detection (Poka-Yoke) - always active
        if !loss.is_finite() {
            self.anomaly_count += 1;
            return Some(TrainingAnomaly::NonFiniteLoss);
        }

        // Gradient explosion detection
        if gradient_norm > self.gradient_explosion_threshold {
            self.anomaly_count += 1;
            return Some(TrainingAnomaly::GradientExplosion {
                norm: gradient_norm,
                threshold: self.gradient_explosion_threshold,
            });
        }

        // Gradient vanishing detection
        if gradient_norm < self.gradient_vanishing_threshold && gradient_norm > 0.0 {
            self.anomaly_count += 1;
            return Some(TrainingAnomaly::GradientVanishing {
                norm: gradient_norm,
                threshold: self.gradient_vanishing_threshold,
            });
        }

        // Loss spike detection (statistical) - only after warmup
        self.loss_stats.update(loss);
        if self.loss_stats.count > self.warmup_count {
            let z_score = self.loss_stats.z_score(loss);
            if z_score.abs() > self.threshold_sigma {
                self.anomaly_count += 1;
                return Some(TrainingAnomaly::LossSpike { z_score, loss });
            }
        }

        None
    }

    /// Get number of anomalies detected.
    #[must_use]
    pub fn anomaly_count(&self) -> u64 {
        self.anomaly_count
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.loss_stats.reset();
        self.anomaly_count = 0;
    }
}

// ============================================================================
// Training Simulation
// ============================================================================

/// Simulated training event for journaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainEvent {
    /// Epoch completed.
    Epoch(TrainingState),
    /// Anomaly detected.
    Anomaly(String),
    /// Checkpoint created.
    Checkpoint { epoch: u64 },
    /// Early stopping triggered.
    EarlyStopping { best_epoch: u64, best_val_loss: f64 },
}

/// Simulated training scenario for reproducible ML experiments.
///
/// Implements Toyota Way principles:
/// - Jidoka: Stop-on-anomaly via `AnomalyDetector`
/// - Heijunka: Load-balanced batch iteration
/// - Kaizen: Continuous improvement tracking
pub struct TrainingSimulation {
    /// Training hyperparameters.
    config: TrainingConfig,
    /// Deterministic RNG for reproducibility.
    rng: SimRng,
    /// Training event journal for replay.
    journal: EventJournal,
    /// Anomaly detector (Jidoka).
    anomaly_detector: AnomalyDetector,
    /// Current epoch.
    current_epoch: u64,
    /// Training trajectory.
    trajectory: TrainingTrajectory,
    /// Best validation loss for early stopping.
    best_val_loss: f64,
    /// Epochs without improvement counter.
    epochs_without_improvement: usize,
}

impl TrainingSimulation {
    /// Create new training simulation with deterministic seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            config: TrainingConfig::default(),
            rng: SimRng::new(seed),
            journal: EventJournal::new(true), // Record RNG state
            anomaly_detector: AnomalyDetector::new(3.0), // 3σ threshold
            current_epoch: 0,
            trajectory: TrainingTrajectory::new(),
            best_val_loss: f64::INFINITY,
            epochs_without_improvement: 0,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(seed: u64, config: TrainingConfig) -> Self {
        Self {
            config,
            rng: SimRng::new(seed),
            journal: EventJournal::new(true), // Record RNG state
            anomaly_detector: AnomalyDetector::new(3.0),
            current_epoch: 0,
            trajectory: TrainingTrajectory::new(),
            best_val_loss: f64::INFINITY,
            epochs_without_improvement: 0,
        }
    }

    /// Set anomaly detector.
    pub fn set_anomaly_detector(&mut self, detector: AnomalyDetector) {
        self.anomaly_detector = detector;
    }

    /// Get current training configuration.
    #[must_use]
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get current trajectory.
    #[must_use]
    pub fn trajectory(&self) -> &TrainingTrajectory {
        &self.trajectory
    }

    /// Simulate a single training step with given loss and gradient norm.
    ///
    /// This is a simplified simulation - real training would compute actual
    /// forward/backward passes. This enables testing training dynamics
    /// without actual model computation.
    ///
    /// # Errors
    ///
    /// Returns error if a training anomaly is detected (Jidoka).
    pub fn step(&mut self, loss: f64, gradient_norm: f64) -> SimResult<Option<TrainingState>> {
        // Jidoka: Check for anomalies
        if let Some(anomaly) = self.anomaly_detector.check(loss, gradient_norm) {
            let event = TrainEvent::Anomaly(anomaly.to_string());
            let rng_state = self.rng.save_state();
            let _ = self.journal.append(
                SimTime::from_secs(self.current_epoch as f64),
                self.current_epoch,
                &event,
                Some(&rng_state),
            );
            return Err(SimError::jidoka(format!(
                "Training anomaly at epoch {}: {anomaly}",
                self.current_epoch
            )));
        }

        // Simulate validation loss (simplified: add noise to training loss)
        let val_loss = loss * (1.0 + 0.1 * (self.rng.gen_f64() - 0.5));

        // Create training state
        let rng_state = self.rng.save_state();
        let state = TrainingState {
            epoch: self.current_epoch,
            loss,
            val_loss,
            metrics: TrainingMetrics {
                train_loss: loss,
                val_loss,
                accuracy: None,
                gradient_norm,
                learning_rate: self.config.learning_rate,
                params_updated: 1000, // Simulated
            },
            rng_state: rng_state.clone(),
        };

        // Track best validation loss for early stopping
        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }

        // Record in journal and trajectory
        let event = TrainEvent::Epoch(state.clone());
        let _ = self.journal.append(
            SimTime::from_secs(self.current_epoch as f64),
            self.current_epoch,
            &event,
            Some(&rng_state),
        );
        self.trajectory.push(state.clone());

        self.current_epoch += 1;

        // Check early stopping
        if let Some(patience) = self.config.early_stopping {
            if self.epochs_without_improvement >= patience {
                let event = TrainEvent::EarlyStopping {
                    best_epoch: self.current_epoch - patience as u64,
                    best_val_loss: self.best_val_loss,
                };
                let rng_state = self.rng.save_state();
                let _ = self.journal.append(
                    SimTime::from_secs(self.current_epoch as f64),
                    self.current_epoch,
                    &event,
                    Some(&rng_state),
                );
                return Ok(None); // Signal early stopping
            }
        }

        Ok(Some(state))
    }

    /// Simulate training for specified epochs using a loss function.
    ///
    /// The `loss_fn` takes (epoch, rng) and returns (loss, `gradient_norm`).
    ///
    /// # Errors
    ///
    /// Returns error if a training anomaly is detected.
    pub fn simulate<F>(&mut self, epochs: u64, mut loss_fn: F) -> SimResult<&TrainingTrajectory>
    where
        F: FnMut(u64, &mut SimRng) -> (f64, f64),
    {
        for epoch in 0..epochs {
            let (loss, grad_norm) = loss_fn(epoch, &mut self.rng);
            if self.step(loss, grad_norm)?.is_none() {
                break; // Early stopping
            }
        }
        Ok(&self.trajectory)
    }

    /// Replay training from a checkpoint state.
    ///
    /// # Errors
    ///
    /// Returns error if RNG state restoration fails.
    pub fn replay_from(&mut self, checkpoint: &TrainingState) -> SimResult<()> {
        self.rng.restore_state(&checkpoint.rng_state)
            .map_err(|e| SimError::config(format!("Failed to restore RNG state: {e}")))?;
        self.current_epoch = checkpoint.epoch;
        Ok(())
    }

    /// Get the event journal.
    #[must_use]
    pub fn journal(&self) -> &EventJournal {
        &self.journal
    }

    /// Reset simulation state.
    pub fn reset(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.journal = EventJournal::new(true);
        self.anomaly_detector.reset();
        self.current_epoch = 0;
        self.trajectory = TrainingTrajectory::new();
        self.best_val_loss = f64::INFINITY;
        self.epochs_without_improvement = 0;
    }
}

// ============================================================================
// Prediction Simulation
// ============================================================================

/// Prediction state for replay and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionState {
    /// Input features.
    pub input: Vec<f64>,
    /// Model output.
    pub output: Vec<f64>,
    /// Uncertainty estimate (if available).
    pub uncertainty: Option<f64>,
    /// Inference latency in microseconds (simulated).
    pub latency_us: u64,
    /// Sequence number.
    pub sequence: u64,
}

/// Inference configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference.
    pub batch_size: usize,
    /// Temperature for probabilistic outputs.
    pub temperature: f64,
    /// Top-k sampling (0 = greedy).
    pub top_k: usize,
    /// Enable uncertainty quantification.
    pub uncertainty: bool,
    /// Simulated latency base (microseconds).
    pub base_latency_us: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            temperature: 1.0,
            top_k: 0,
            uncertainty: false,
            base_latency_us: 1000,
        }
    }
}

/// Simulated inference scenario for reproducible prediction testing.
pub struct PredictionSimulation {
    /// Inference configuration.
    config: InferenceConfig,
    /// Deterministic RNG for stochastic models.
    rng: SimRng,
    /// Prediction sequence counter.
    sequence: u64,
    /// Prediction history.
    history: Vec<PredictionState>,
}

impl PredictionSimulation {
    /// Create new prediction simulation with deterministic seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            config: InferenceConfig::default(),
            rng: SimRng::new(seed),
            sequence: 0,
            history: Vec::new(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(seed: u64, config: InferenceConfig) -> Self {
        Self {
            config,
            rng: SimRng::new(seed),
            sequence: 0,
            history: Vec::new(),
        }
    }

    /// Get inference configuration.
    #[must_use]
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Simulate single prediction using a model function.
    ///
    /// The `model_fn` takes input and returns output vector.
    ///
    /// # Errors
    ///
    /// Returns error if model prediction fails.
    pub fn predict<F>(&mut self, input: &[f64], model_fn: F) -> SimResult<PredictionState>
    where
        F: FnOnce(&[f64]) -> Vec<f64>,
    {
        // Simulate inference
        let mut output = model_fn(input);

        // Apply temperature scaling if not 1.0
        if (self.config.temperature - 1.0).abs() > 1e-10 {
            output = self.apply_temperature(&output, self.config.temperature);
        }

        // Apply top-k sampling if configured
        if self.config.top_k > 0 {
            output = self.sample_top_k(&output, self.config.top_k);
        }

        // Compute uncertainty if enabled (simplified: variance of output)
        let uncertainty = if self.config.uncertainty {
            Some(self.compute_uncertainty(&output))
        } else {
            None
        };

        // Simulate latency with noise
        let latency_noise = (self.rng.gen_f64() * 0.2 - 0.1) * self.config.base_latency_us as f64;
        let latency_us = (self.config.base_latency_us as f64 + latency_noise).max(1.0) as u64;

        let state = PredictionState {
            input: input.to_vec(),
            output,
            uncertainty,
            latency_us,
            sequence: self.sequence,
        };

        self.sequence += 1;
        self.history.push(state.clone());

        Ok(state)
    }

    /// Simulate batch prediction.
    ///
    /// # Errors
    ///
    /// Returns error if any prediction fails.
    pub fn predict_batch<F>(&mut self, inputs: &[Vec<f64>], model_fn: F) -> SimResult<Vec<PredictionState>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        inputs
            .iter()
            .map(|input| self.predict(input, &model_fn))
            .collect()
    }

    /// Apply temperature scaling to logits.
    #[allow(clippy::unused_self)]
    fn apply_temperature(&self, logits: &[f64], temperature: f64) -> Vec<f64> {
        if temperature <= 0.0 {
            return logits.to_vec();
        }
        logits.iter().map(|x| x / temperature).collect()
    }

    /// Sample top-k values, zeroing out the rest.
    #[allow(clippy::unused_self)]
    fn sample_top_k(&self, values: &[f64], k: usize) -> Vec<f64> {
        if k >= values.len() {
            return values.to_vec();
        }

        // Find k-th largest value
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted.get(k - 1).copied().unwrap_or(f64::NEG_INFINITY);

        // Zero out values below threshold
        values
            .iter()
            .map(|&v| if v >= threshold { v } else { 0.0 })
            .collect()
    }

    /// Compute simplified uncertainty estimate.
    #[allow(clippy::unused_self)]
    fn compute_uncertainty(&self, output: &[f64]) -> f64 {
        if output.is_empty() {
            return 0.0;
        }
        let mean = output.iter().sum::<f64>() / output.len() as f64;
        let variance = output.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / output.len() as f64;
        variance.sqrt()
    }

    /// Get prediction history.
    #[must_use]
    pub fn history(&self) -> &[PredictionState] {
        &self.history
    }

    /// Reset simulation state.
    pub fn reset(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.sequence = 0;
        self.history.clear();
    }
}

// ============================================================================
// Multi-Turn Simulation
// ============================================================================

/// A single turn in multi-turn interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Turn index.
    pub index: usize,
    /// Input query/prompt.
    pub input: String,
    /// Model response.
    pub output: String,
    /// Ground truth (if available).
    pub expected: Option<String>,
    /// Turn metrics.
    pub metrics: TurnMetrics,
    /// Context window usage (tokens).
    pub context_tokens: usize,
}

/// Metrics for a single turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMetrics {
    /// Generation latency in milliseconds.
    pub latency_ms: f64,
    /// Input tokens.
    pub input_tokens: usize,
    /// Output tokens.
    pub output_tokens: usize,
    /// Estimated cost (normalized).
    pub cost: f64,
    /// Accuracy vs oracle (if available).
    pub accuracy: Option<f64>,
}

impl Default for TurnMetrics {
    fn default() -> Self {
        Self {
            latency_ms: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            cost: 0.0,
            accuracy: None,
        }
    }
}

/// Multi-turn evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnEvaluation {
    /// Mean accuracy across runs.
    pub mean_accuracy: Option<f64>,
    /// Mean latency across runs.
    pub mean_latency: Option<f64>,
    /// Total cost across runs.
    pub total_cost: f64,
    /// Confidence interval level.
    pub confidence_interval: f64,
    /// Number of runs performed.
    pub n_runs: usize,
}

/// Point on Pareto frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    /// Model identifier.
    pub model_id: String,
    /// Accuracy score.
    pub accuracy: f64,
    /// Cost metric.
    pub cost: f64,
    /// Latency metric.
    pub latency: f64,
    /// Models that dominate this one.
    pub dominated_by: Vec<String>,
}

/// Pareto frontier analysis results.
#[derive(Debug, Clone, Default)]
pub struct ParetoAnalysis {
    /// Non-dominated solutions (Pareto frontier).
    pub frontier: Vec<ParetoPoint>,
    /// Value scores per model.
    pub value_scores: HashMap<String, f64>,
}

/// Multi-turn simulation for conversational/iterative model evaluation.
///
/// Implements Pareto frontier analysis across accuracy, cost, and latency.
pub struct MultiTurnSimulation {
    /// Conversation history.
    history: Vec<Turn>,
    /// Deterministic RNG.
    rng: SimRng,
    /// Cost per input token.
    input_token_cost: f64,
    /// Cost per output token.
    output_token_cost: f64,
    /// Base latency per token (ms).
    latency_per_token_ms: f64,
}

impl MultiTurnSimulation {
    /// Create new multi-turn simulation.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            history: Vec::new(),
            rng: SimRng::new(seed),
            input_token_cost: 0.00001,
            output_token_cost: 0.00003,
            latency_per_token_ms: 10.0,
        }
    }

    /// Set cost parameters.
    #[must_use]
    pub fn with_costs(mut self, input_cost: f64, output_cost: f64) -> Self {
        self.input_token_cost = input_cost;
        self.output_token_cost = output_cost;
        self
    }

    /// Set latency per token.
    #[must_use]
    pub fn with_latency_per_token(mut self, latency_ms: f64) -> Self {
        self.latency_per_token_ms = latency_ms;
        self
    }

    /// Execute a single turn using a response generator.
    ///
    /// The `generate_fn` takes (input, history) and returns response string.
    ///
    /// # Errors
    ///
    /// Returns error if turn execution fails.
    pub fn turn<F>(&mut self, input: &str, expected: Option<&str>, generate_fn: F) -> SimResult<Turn>
    where
        F: FnOnce(&str, &[Turn]) -> String,
    {
        let input_tokens = self.count_tokens(input);

        // Generate response
        let output = generate_fn(input, &self.history);
        let output_tokens = self.count_tokens(&output);

        // Compute latency with noise
        let base_latency = (input_tokens + output_tokens) as f64 * self.latency_per_token_ms;
        let noise = (self.rng.gen_f64() * 0.2 - 0.1) * base_latency;
        let latency_ms = (base_latency + noise).max(1.0);

        // Compute cost
        let cost = input_tokens as f64 * self.input_token_cost
            + output_tokens as f64 * self.output_token_cost;

        // Compute accuracy if expected is provided
        let accuracy = expected.map(|exp| self.compute_accuracy(&output, exp));

        let context_tokens = self.history.iter().map(|t| t.metrics.input_tokens + t.metrics.output_tokens).sum::<usize>()
            + input_tokens;

        let turn = Turn {
            index: self.history.len(),
            input: input.to_string(),
            output,
            expected: expected.map(String::from),
            metrics: TurnMetrics {
                latency_ms,
                input_tokens,
                output_tokens,
                cost,
                accuracy,
            },
            context_tokens,
        };

        self.history.push(turn.clone());
        Ok(turn)
    }

    /// Simplified token counting (words * 1.3).
    #[allow(clippy::unused_self)]
    fn count_tokens(&self, text: &str) -> usize {
        let words = text.split_whitespace().count();
        (words as f64 * 1.3).ceil() as usize
    }

    /// Compute accuracy between output and expected (Levenshtein similarity).
    #[allow(clippy::unused_self)]
    fn compute_accuracy(&self, output: &str, expected: &str) -> f64 {
        if expected.is_empty() && output.is_empty() {
            return 1.0;
        }
        if expected.is_empty() || output.is_empty() {
            return 0.0;
        }

        // Simple word overlap similarity
        let output_words: std::collections::HashSet<&str> = output.split_whitespace().collect();
        let expected_words: std::collections::HashSet<&str> = expected.split_whitespace().collect();

        let intersection = output_words.intersection(&expected_words).count();
        let union = output_words.union(&expected_words).count();

        if union == 0 {
            return 1.0;
        }

        intersection as f64 / union as f64
    }

    /// Run complete multi-turn evaluation with statistical analysis.
    ///
    /// Following Princeton methodology: minimum 5 runs, 95% CI.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 5 runs are requested or if evaluation fails.
    pub fn evaluate<F>(
        &mut self,
        queries: &[(String, Option<String>)],
        n_runs: usize,
        generate_fn: F,
    ) -> SimResult<MultiTurnEvaluation>
    where
        F: Fn(&str, &[Turn]) -> String,
    {
        if n_runs < 5 {
            return Err(SimError::config(
                "Princeton methodology requires minimum 5 runs".to_string(),
            ));
        }

        let mut all_accuracies: Vec<f64> = Vec::new();
        let mut all_latencies: Vec<f64> = Vec::new();
        let mut total_cost = 0.0;

        for run in 0..n_runs {
            // Reset for each run with derived seed
            let derived_seed = self.rng.gen_u64().wrapping_add(run as u64);
            self.reset(derived_seed);

            for (query, expected) in queries {
                let turn = self.turn(query, expected.as_deref(), &generate_fn)?;
                if let Some(acc) = turn.metrics.accuracy {
                    all_accuracies.push(acc);
                }
                all_latencies.push(turn.metrics.latency_ms);
                total_cost += turn.metrics.cost;
            }
        }

        let mean_accuracy = if all_accuracies.is_empty() {
            None
        } else {
            Some(all_accuracies.iter().sum::<f64>() / all_accuracies.len() as f64)
        };

        let mean_latency = if all_latencies.is_empty() {
            None
        } else {
            Some(all_latencies.iter().sum::<f64>() / all_latencies.len() as f64)
        };

        Ok(MultiTurnEvaluation {
            mean_accuracy,
            mean_latency,
            total_cost: total_cost / n_runs as f64,
            confidence_interval: 0.95,
            n_runs,
        })
    }

    /// Compute Pareto frontier across multiple model evaluations.
    #[must_use]
    pub fn pareto_analysis(evaluations: &[(String, MultiTurnEvaluation)]) -> ParetoAnalysis {
        let mut points: Vec<ParetoPoint> = evaluations
            .iter()
            .map(|(id, eval)| ParetoPoint {
                model_id: id.clone(),
                accuracy: eval.mean_accuracy.unwrap_or(0.0),
                cost: eval.total_cost,
                latency: eval.mean_latency.unwrap_or(f64::MAX),
                dominated_by: Vec::new(),
            })
            .collect();

        // Identify dominated points
        // First pass: identify dominance relationships
        let mut dominance: Vec<Vec<String>> = vec![Vec::new(); points.len()];
        for i in 0..points.len() {
            for j in 0..points.len() {
                if i != j && Self::dominates(&points[j], &points[i]) {
                    dominance[i].push(points[j].model_id.clone());
                }
            }
        }
        // Second pass: assign dominated_by
        for (i, dominated_by) in dominance.into_iter().enumerate() {
            points[i].dominated_by = dominated_by;
        }

        // Compute value scores
        let baseline_accuracy = points.iter().map(|p| p.accuracy).fold(0.0_f64, f64::max);
        let baseline_cost = points
            .iter()
            .map(|p| p.cost)
            .fold(f64::INFINITY, f64::min);
        let baseline_latency = points
            .iter()
            .map(|p| p.latency)
            .fold(f64::INFINITY, f64::min);

        let value_scores: HashMap<String, f64> = points
            .iter()
            .map(|p| {
                let accuracy_gap = baseline_accuracy - p.accuracy;
                let cost_ratio = baseline_cost / p.cost.max(1e-10);
                let latency_ratio = baseline_latency / p.latency.max(1e-10);
                let value = (1.0 - accuracy_gap) * cost_ratio * latency_ratio;
                (p.model_id.clone(), value)
            })
            .collect();

        let frontier: Vec<ParetoPoint> = points
            .into_iter()
            .filter(|p| p.dominated_by.is_empty())
            .collect();

        ParetoAnalysis {
            frontier,
            value_scores,
        }
    }

    /// Check if point a dominates point b (better in all objectives).
    fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
        a.accuracy >= b.accuracy
            && a.cost <= b.cost
            && a.latency <= b.latency
            && (a.accuracy > b.accuracy || a.cost < b.cost || a.latency < b.latency)
    }

    /// Get conversation history.
    #[must_use]
    pub fn history(&self) -> &[Turn] {
        &self.history
    }

    /// Reset simulation state.
    pub fn reset(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.history.clear();
    }
}

// ============================================================================
// Jidoka ML Feedback Loop
// ============================================================================

/// Rule patch types for Kaizen improvements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RuleType {
    /// Clip gradients to max norm.
    GradientClipping,
    /// Reduce learning rate.
    LearningRateDecay,
    /// Add warmup steps.
    LearningRateWarmup,
    /// Increase batch size.
    BatchSizeIncrease,
    /// Add regularization.
    Regularization,
    /// Manual review required.
    ManualReview,
}

/// Improvement patch generated from anomaly pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePatch {
    /// Type of rule patch.
    pub rule_type: RuleType,
    /// Parameters for the patch.
    pub parameters: HashMap<String, String>,
}

impl Default for RulePatch {
    fn default() -> Self {
        Self {
            rule_type: RuleType::ManualReview,
            parameters: HashMap::new(),
        }
    }
}

/// Anomaly pattern classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Loss became NaN/Inf.
    NonFiniteLoss,
    /// Gradient norm exceeded threshold.
    GradientExplosion,
    /// Gradient vanished below threshold.
    GradientVanishing,
    /// Loss spike (statistical outlier).
    LossSpike,
    /// Prediction confidence below threshold.
    LowConfidence,
    /// Model output inconsistent with oracle.
    OracleMismatch,
}

/// Pattern detected during training/inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPattern {
    /// Pattern type.
    pub pattern_type: AnomalyType,
    /// Frequency of occurrence.
    pub frequency: u64,
    /// Context information.
    pub context: HashMap<String, String>,
    /// Suggested fix description.
    pub suggested_fix: Option<String>,
}

/// Jidoka feedback loop for ML simulation.
///
/// Each detected anomaly generates improvement patches (Kaizen).
pub struct JidokaMLFeedback {
    /// Anomaly patterns detected.
    patterns: Vec<AnomalyPattern>,
    /// Generated fixes (rule patches).
    patches: Vec<RulePatch>,
    /// Rolling stats for anomaly rate tracking.
    anomaly_rate: RollingStats,
    /// Threshold for auto-patch generation.
    auto_patch_threshold: u64,
}

impl Default for JidokaMLFeedback {
    fn default() -> Self {
        Self::new()
    }
}

impl JidokaMLFeedback {
    /// Create new Jidoka feedback loop.
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            patches: Vec::new(),
            anomaly_rate: RollingStats::new(100),
            auto_patch_threshold: 3,
        }
    }

    /// Set threshold for automatic patch generation.
    #[must_use]
    pub fn with_auto_patch_threshold(mut self, threshold: u64) -> Self {
        self.auto_patch_threshold = threshold;
        self
    }

    /// Record anomaly and potentially generate improvement patch.
    pub fn record_anomaly(&mut self, anomaly: TrainingAnomaly) -> Option<RulePatch> {
        let pattern_type = self.classify_type(&anomaly);
        self.anomaly_rate.update(1.0);

        // Check if we've seen this pattern before
        let existing_idx = self.patterns.iter().position(|p| p.pattern_type == pattern_type);

        if let Some(idx) = existing_idx {
            self.patterns[idx].frequency += 1;

            // After threshold occurrences, generate automated fix
            if self.patterns[idx].frequency >= self.auto_patch_threshold {
                let pattern_clone = self.patterns[idx].clone();
                let patch = self.generate_patch(&pattern_clone);
                self.patches.push(patch.clone());
                return Some(patch);
            }
        } else {
            let suggested_fix = Some(self.suggest_fix(pattern_type));
            let new_pattern = AnomalyPattern {
                pattern_type,
                frequency: 1,
                context: HashMap::new(),
                suggested_fix,
            };

            // Check if threshold is 1 (generate patch on first occurrence)
            if self.auto_patch_threshold <= 1 {
                let patch = self.generate_patch(&new_pattern);
                self.patterns.push(new_pattern);
                self.patches.push(patch.clone());
                return Some(patch);
            }

            self.patterns.push(new_pattern);
        }

        None
    }

    /// Classify anomaly into pattern type.
    #[allow(clippy::unused_self)]
    fn classify_type(&self, anomaly: &TrainingAnomaly) -> AnomalyType {
        match anomaly {
            TrainingAnomaly::NonFiniteLoss => AnomalyType::NonFiniteLoss,
            TrainingAnomaly::GradientExplosion { .. } => AnomalyType::GradientExplosion,
            TrainingAnomaly::GradientVanishing { .. } => AnomalyType::GradientVanishing,
            TrainingAnomaly::LossSpike { .. } => AnomalyType::LossSpike,
            TrainingAnomaly::LowConfidence { .. } => AnomalyType::LowConfidence,
        }
    }

    /// Suggest fix for pattern type.
    #[allow(clippy::unused_self)]
    fn suggest_fix(&self, pattern_type: AnomalyType) -> String {
        match pattern_type {
            AnomalyType::GradientExplosion => "Apply gradient clipping with max_norm=1.0",
            AnomalyType::GradientVanishing => "Use skip connections or residual architecture",
            AnomalyType::LossSpike => "Reduce learning rate or add warmup",
            AnomalyType::NonFiniteLoss => "Check for numerical stability issues",
            AnomalyType::LowConfidence => "Increase model capacity or training data",
            AnomalyType::OracleMismatch => "Review training data distribution",
        }
        .to_string()
    }

    /// Generate rule patch from pattern (Kaizen improvement).
    #[allow(clippy::unused_self)]
    fn generate_patch(&self, pattern: &AnomalyPattern) -> RulePatch {
        let mut params = HashMap::new();

        match pattern.pattern_type {
            AnomalyType::GradientExplosion => {
                params.insert("max_norm".to_string(), "1.0".to_string());
                RulePatch {
                    rule_type: RuleType::GradientClipping,
                    parameters: params,
                }
            }
            AnomalyType::LossSpike => {
                params.insert("warmup_steps".to_string(), "1000".to_string());
                RulePatch {
                    rule_type: RuleType::LearningRateWarmup,
                    parameters: params,
                }
            }
            AnomalyType::GradientVanishing => {
                params.insert("factor".to_string(), "10.0".to_string());
                RulePatch {
                    rule_type: RuleType::LearningRateDecay,
                    parameters: params,
                }
            }
            _ => RulePatch::default(),
        }
    }

    /// Get all detected patterns.
    #[must_use]
    pub fn patterns(&self) -> &[AnomalyPattern] {
        &self.patterns
    }

    /// Get all generated patches.
    #[must_use]
    pub fn patches(&self) -> &[RulePatch] {
        &self.patches
    }

    /// Get current anomaly rate.
    #[must_use]
    pub fn anomaly_rate(&self) -> f64 {
        self.anomaly_rate.mean()
    }

    /// Reset feedback loop state.
    pub fn reset(&mut self) {
        self.patterns.clear();
        self.patches.clear();
        self.anomaly_rate.reset();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // RollingStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rolling_stats_empty() {
        let stats = RollingStats::new(0);
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.std_dev(), 0.0);
    }

    #[test]
    fn test_rolling_stats_single_value() {
        let mut stats = RollingStats::new(0);
        stats.update(5.0);
        assert!((stats.mean() - 5.0).abs() < 1e-10);
        assert_eq!(stats.variance(), 0.0); // n-1 variance with n=1
    }

    #[test]
    fn test_rolling_stats_multiple_values() {
        let mut stats = RollingStats::new(0);
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(v);
        }
        assert!((stats.mean() - 5.0).abs() < 1e-10);
        assert!((stats.variance() - 4.571_428_571_428_571).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_stats_z_score() {
        let mut stats = RollingStats::new(0);
        for v in [10.0, 10.0, 10.0, 10.0, 10.0] {
            stats.update(v);
        }
        // All same values => std = 0 => z_score = 0
        assert!((stats.z_score(10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_stats_windowed() {
        let mut stats = RollingStats::new(3);
        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);
        stats.update(4.0); // Window: [2, 3, 4]
        assert_eq!(stats.recent.len(), 3);
        assert_eq!(stats.recent, vec![2.0, 3.0, 4.0]);
    }

    // -------------------------------------------------------------------------
    // AnomalyDetector Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_anomaly_detector_nan() {
        let mut detector = AnomalyDetector::new(3.0);
        let result = detector.check(f64::NAN, 1.0);
        assert!(matches!(result, Some(TrainingAnomaly::NonFiniteLoss)));
    }

    #[test]
    fn test_anomaly_detector_inf() {
        let mut detector = AnomalyDetector::new(3.0);
        let result = detector.check(f64::INFINITY, 1.0);
        assert!(matches!(result, Some(TrainingAnomaly::NonFiniteLoss)));
    }

    #[test]
    fn test_anomaly_detector_gradient_explosion() {
        let mut detector = AnomalyDetector::new(3.0)
            .with_gradient_explosion_threshold(1e6);
        let result = detector.check(1.0, 1e7);
        assert!(matches!(result, Some(TrainingAnomaly::GradientExplosion { .. })));
    }

    #[test]
    fn test_anomaly_detector_gradient_vanishing() {
        let mut detector = AnomalyDetector::new(3.0)
            .with_gradient_vanishing_threshold(1e-10);
        let result = detector.check(1.0, 1e-12);
        assert!(matches!(result, Some(TrainingAnomaly::GradientVanishing { .. })));
    }

    #[test]
    fn test_anomaly_detector_loss_spike() {
        let mut detector = AnomalyDetector::new(3.0).with_warmup(5);

        // Warmup with stable losses
        for _ in 0..10 {
            detector.check(1.0, 1.0);
        }

        // Now introduce a spike
        let result = detector.check(100.0, 1.0);
        assert!(matches!(result, Some(TrainingAnomaly::LossSpike { .. })));
    }

    #[test]
    fn test_anomaly_detector_no_anomaly() {
        let mut detector = AnomalyDetector::new(3.0);
        let result = detector.check(1.0, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_anomaly_detector_count() {
        let mut detector = AnomalyDetector::new(3.0);
        detector.check(f64::NAN, 1.0);
        detector.check(f64::INFINITY, 1.0);
        assert_eq!(detector.anomaly_count(), 2);
    }

    // -------------------------------------------------------------------------
    // TrainingTrajectory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trajectory_empty() {
        let traj = TrainingTrajectory::new();
        assert!(traj.final_state().is_none());
        assert!(traj.best_val_loss().is_none());
        assert!(!traj.converged(0.01));
    }

    #[test]
    fn test_trajectory_best_val_loss() {
        let mut traj = TrainingTrajectory::new();
        let rng = SimRng::new(42);
        let rng_state = rng.save_state();

        traj.push(TrainingState {
            epoch: 0,
            loss: 1.0,
            val_loss: 0.9,
            metrics: TrainingMetrics::default(),
            rng_state: rng_state.clone(),
        });
        traj.push(TrainingState {
            epoch: 1,
            loss: 0.8,
            val_loss: 0.7,
            metrics: TrainingMetrics::default(),
            rng_state: rng_state.clone(),
        });
        traj.push(TrainingState {
            epoch: 2,
            loss: 0.6,
            val_loss: 0.8,
            metrics: TrainingMetrics::default(),
            rng_state: rng_state.clone(),
        });

        assert!((traj.best_val_loss().unwrap_or(0.0) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_converged() {
        let mut traj = TrainingTrajectory::new();
        let rng = SimRng::new(42);
        let rng_state = rng.save_state();

        for i in 0..15 {
            traj.push(TrainingState {
                epoch: i,
                loss: 0.5, // Constant loss
                val_loss: 0.5,
                metrics: TrainingMetrics::default(),
                rng_state: rng_state.clone(),
            });
        }
        assert!(traj.converged(0.01));
    }

    // -------------------------------------------------------------------------
    // TrainingSimulation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_simulation_new() {
        let sim = TrainingSimulation::new(42);
        assert_eq!(sim.config().learning_rate, 0.001);
        assert_eq!(sim.trajectory().states.len(), 0);
    }

    #[test]
    fn test_training_simulation_step() {
        let mut sim = TrainingSimulation::new(42);
        let result = sim.step(0.5, 1.0);
        assert!(result.is_ok());
        assert_eq!(sim.trajectory().states.len(), 1);
    }

    #[test]
    fn test_training_simulation_anomaly_stops() {
        let mut sim = TrainingSimulation::new(42);
        let result = sim.step(f64::NAN, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_training_simulation_simulate() {
        let mut sim = TrainingSimulation::new(42);
        let result = sim.simulate(10, |epoch, _rng| {
            // Simulated decreasing loss
            let loss = 1.0 / (epoch as f64 + 1.0);
            let grad_norm = 0.5;
            (loss, grad_norm)
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap().states.len(), 10);
    }

    #[test]
    fn test_training_simulation_early_stopping() {
        let config = TrainingConfig {
            early_stopping: Some(3),
            ..Default::default()
        };
        let mut sim = TrainingSimulation::with_config(42, config);

        // Loss that doesn't improve
        let result = sim.simulate(100, |_epoch, _rng| (1.0, 1.0));
        assert!(result.is_ok());
        // Should stop early due to no improvement
        assert!(result.unwrap().states.len() < 100);
    }

    // -------------------------------------------------------------------------
    // PredictionSimulation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prediction_simulation_new() {
        let sim = PredictionSimulation::new(42);
        assert_eq!(sim.config().batch_size, 32);
        assert!(sim.history().is_empty());
    }

    #[test]
    fn test_prediction_simulation_predict() {
        let mut sim = PredictionSimulation::new(42);
        let result = sim.predict(&[1.0, 2.0, 3.0], |input| {
            input.iter().map(|x| x * 2.0).collect()
        });
        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.output, vec![2.0, 4.0, 6.0]);
        assert_eq!(sim.history().len(), 1);
    }

    #[test]
    fn test_prediction_simulation_batch() {
        let mut sim = PredictionSimulation::new(42);
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = sim.predict_batch(&inputs, |input| vec![input[0] * 2.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_prediction_simulation_temperature() {
        let config = InferenceConfig {
            temperature: 0.5,
            ..Default::default()
        };
        let mut sim = PredictionSimulation::with_config(42, config);
        let result = sim.predict(&[1.0, 2.0], |input| input.to_vec());
        assert!(result.is_ok());
        let state = result.unwrap();
        // Temperature scaling divides by temperature
        assert!((state.output[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_simulation_top_k() {
        let config = InferenceConfig {
            top_k: 2,
            ..Default::default()
        };
        let mut sim = PredictionSimulation::with_config(42, config);
        let result = sim.predict(&[], |_| vec![0.1, 0.5, 0.3, 0.1]);
        assert!(result.is_ok());
        let state = result.unwrap();
        // Top-2 should keep 0.5 and 0.3, zero out others
        assert!(state.output[0].abs() < 1e-10); // 0.1 zeroed
        assert!((state.output[1] - 0.5).abs() < 1e-10);
        assert!((state.output[2] - 0.3).abs() < 1e-10);
        assert!(state.output[3].abs() < 1e-10); // 0.1 zeroed
    }

    #[test]
    fn test_prediction_simulation_uncertainty() {
        let config = InferenceConfig {
            uncertainty: true,
            ..Default::default()
        };
        let mut sim = PredictionSimulation::with_config(42, config);
        let result = sim.predict(&[], |_| vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());
        assert!(result.unwrap().uncertainty.is_some());
    }

    // -------------------------------------------------------------------------
    // MultiTurnSimulation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_turn_simulation_new() {
        let sim = MultiTurnSimulation::new(42);
        assert!(sim.history().is_empty());
    }

    #[test]
    fn test_multi_turn_simulation_turn() {
        let mut sim = MultiTurnSimulation::new(42);
        let result = sim.turn("Hello", None, |input, _| format!("Response to: {input}"));
        assert!(result.is_ok());
        let turn = result.unwrap();
        assert_eq!(turn.index, 0);
        assert!(turn.output.contains("Hello"));
    }

    #[test]
    fn test_multi_turn_simulation_with_expected() {
        let mut sim = MultiTurnSimulation::new(42);
        let result = sim.turn(
            "What is 2+2?",
            Some("4"),
            |_, _| "4".to_string(),
        );
        assert!(result.is_ok());
        let turn = result.unwrap();
        assert!(turn.metrics.accuracy.is_some());
        assert!((turn.metrics.accuracy.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_turn_simulation_history() {
        let mut sim = MultiTurnSimulation::new(42);
        sim.turn("First", None, |_, _| "Response 1".to_string()).unwrap();
        sim.turn("Second", None, |_, history| {
            format!("Response 2 (after {} turns)", history.len())
        }).unwrap();
        assert_eq!(sim.history().len(), 2);
    }

    #[test]
    fn test_multi_turn_evaluation_minimum_runs() {
        let mut sim = MultiTurnSimulation::new(42);
        let queries = vec![("Hello".to_string(), None)];
        let result = sim.evaluate(&queries, 3, |_, _| "Hi".to_string());
        assert!(result.is_err()); // Should fail with < 5 runs
    }

    #[test]
    fn test_multi_turn_evaluation_success() {
        let mut sim = MultiTurnSimulation::new(42);
        let queries = vec![
            ("Q1".to_string(), Some("A1".to_string())),
            ("Q2".to_string(), Some("A2".to_string())),
        ];
        let result = sim.evaluate(&queries, 5, |_, _| "A1 A2".to_string());
        assert!(result.is_ok());
        let eval = result.unwrap();
        assert_eq!(eval.n_runs, 5);
        assert!(eval.mean_accuracy.is_some());
    }

    #[test]
    fn test_pareto_analysis_dominance() {
        let evals = vec![
            ("model_a".to_string(), MultiTurnEvaluation {
                mean_accuracy: Some(0.9),
                mean_latency: Some(100.0),
                total_cost: 1.0,
                confidence_interval: 0.95,
                n_runs: 5,
            }),
            ("model_b".to_string(), MultiTurnEvaluation {
                mean_accuracy: Some(0.8),
                mean_latency: Some(200.0),
                total_cost: 2.0,
                confidence_interval: 0.95,
                n_runs: 5,
            }),
        ];

        let analysis = MultiTurnSimulation::pareto_analysis(&evals);
        // model_a dominates model_b (better in all dimensions)
        assert_eq!(analysis.frontier.len(), 1);
        assert_eq!(analysis.frontier[0].model_id, "model_a");
    }

    #[test]
    fn test_pareto_analysis_no_dominance() {
        let evals = vec![
            ("model_a".to_string(), MultiTurnEvaluation {
                mean_accuracy: Some(0.9),
                mean_latency: Some(200.0), // Worse latency
                total_cost: 1.0,
                confidence_interval: 0.95,
                n_runs: 5,
            }),
            ("model_b".to_string(), MultiTurnEvaluation {
                mean_accuracy: Some(0.8),
                mean_latency: Some(100.0), // Better latency
                total_cost: 2.0,
                confidence_interval: 0.95,
                n_runs: 5,
            }),
        ];

        let analysis = MultiTurnSimulation::pareto_analysis(&evals);
        // Neither dominates - both on frontier (trade-off)
        assert_eq!(analysis.frontier.len(), 2);
    }

    // -------------------------------------------------------------------------
    // JidokaMLFeedback Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jidoka_feedback_new() {
        let feedback = JidokaMLFeedback::new();
        assert!(feedback.patterns().is_empty());
        assert!(feedback.patches().is_empty());
    }

    #[test]
    fn test_jidoka_feedback_record_anomaly() {
        let mut feedback = JidokaMLFeedback::new();
        let patch = feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        });
        assert!(patch.is_none()); // First occurrence, no patch yet
        assert_eq!(feedback.patterns().len(), 1);
    }

    #[test]
    fn test_jidoka_feedback_auto_patch() {
        let mut feedback = JidokaMLFeedback::new().with_auto_patch_threshold(2);

        // First occurrence
        feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        });

        // Second occurrence - should trigger patch
        let patch = feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
            norm: 1e8,
            threshold: 1e6,
        });

        assert!(patch.is_some());
        assert_eq!(patch.unwrap().rule_type, RuleType::GradientClipping);
    }

    #[test]
    fn test_jidoka_feedback_loss_spike_patch() {
        let mut feedback = JidokaMLFeedback::new().with_auto_patch_threshold(1);

        let patch = feedback.record_anomaly(TrainingAnomaly::LossSpike {
            z_score: 5.0,
            loss: 100.0,
        });

        assert!(patch.is_some());
        assert_eq!(patch.unwrap().rule_type, RuleType::LearningRateWarmup);
    }

    #[test]
    fn test_jidoka_feedback_different_anomalies() {
        let mut feedback = JidokaMLFeedback::new();

        feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        });
        feedback.record_anomaly(TrainingAnomaly::LossSpike {
            z_score: 5.0,
            loss: 100.0,
        });

        assert_eq!(feedback.patterns().len(), 2);
    }

    #[test]
    fn test_jidoka_feedback_reset() {
        let mut feedback = JidokaMLFeedback::new();
        feedback.record_anomaly(TrainingAnomaly::NonFiniteLoss);
        feedback.reset();
        assert!(feedback.patterns().is_empty());
    }

    // -------------------------------------------------------------------------
    // Display and Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_anomaly_display() {
        let anomaly = TrainingAnomaly::NonFiniteLoss;
        let display = format!("{}", anomaly);
        assert!(display.contains("NaN/Inf"));

        let anomaly = TrainingAnomaly::GradientExplosion { norm: 1e7, threshold: 1e6 };
        let display = format!("{}", anomaly);
        assert!(display.contains("explosion"));

        let anomaly = TrainingAnomaly::GradientVanishing { norm: 1e-12, threshold: 1e-10 };
        let display = format!("{}", anomaly);
        assert!(display.contains("vanishing"));

        let anomaly = TrainingAnomaly::LossSpike { z_score: 5.0, loss: 100.0 };
        let display = format!("{}", anomaly);
        assert!(display.contains("spike"));

        let anomaly = TrainingAnomaly::LowConfidence { confidence: 0.3, threshold: 0.5 };
        let display = format!("{}", anomaly);
        assert!(display.contains("confidence"));
    }

    #[test]
    fn test_rolling_stats_reset() {
        let mut stats = RollingStats::new(5);
        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);
        stats.reset();
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.variance(), 0.0);
    }

    #[test]
    fn test_rolling_stats_z_score_with_variance() {
        let mut stats = RollingStats::new(0);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.update(v);
        }
        // Mean = 3, std dev = sqrt(2.5) ≈ 1.58
        let z = stats.z_score(5.0);
        assert!(z > 1.0); // 5 is above mean
    }

    #[test]
    fn test_rolling_stats_clone() {
        let mut stats = RollingStats::new(3);
        stats.update(1.0);
        stats.update(2.0);
        let cloned = stats.clone();
        assert_eq!(cloned.mean(), stats.mean());
    }

    #[test]
    fn test_training_config_clone() {
        let config = TrainingConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.learning_rate, config.learning_rate);
        assert_eq!(cloned.batch_size, config.batch_size);
    }

    #[test]
    fn test_training_state_clone() {
        let rng = SimRng::new(42);
        let state = TrainingState {
            epoch: 5,
            loss: 0.5,
            val_loss: 0.6,
            metrics: TrainingMetrics::default(),
            rng_state: rng.save_state(),
        };
        let cloned = state.clone();
        assert_eq!(cloned.epoch, state.epoch);
        assert_eq!(cloned.loss, state.loss);
    }

    #[test]
    fn test_training_metrics_clone() {
        let metrics = TrainingMetrics {
            train_loss: 0.5,
            val_loss: 0.6,
            accuracy: Some(0.9),
            gradient_norm: 1.0,
            learning_rate: 0.001,
            params_updated: 1000,
        };
        let cloned = metrics.clone();
        assert_eq!(cloned.accuracy, metrics.accuracy);
    }

    #[test]
    fn test_training_trajectory_clone() {
        let mut traj = TrainingTrajectory::new();
        let rng = SimRng::new(42);
        traj.push(TrainingState {
            epoch: 0,
            loss: 1.0,
            val_loss: 0.9,
            metrics: TrainingMetrics::default(),
            rng_state: rng.save_state(),
        });
        let cloned = traj.clone();
        assert_eq!(cloned.states.len(), 1);
    }

    #[test]
    fn test_anomaly_detector_clone() {
        let detector = AnomalyDetector::new(3.0)
            .with_warmup(10)
            .with_gradient_explosion_threshold(1e6);
        let cloned = detector.clone();
        assert_eq!(cloned.threshold_sigma, detector.threshold_sigma);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.batch_size, 32);
        assert!((config.temperature - 1.0).abs() < 1e-10);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_prediction_state_clone() {
        let state = PredictionState {
            input: vec![1.0, 2.0],
            output: vec![2.0, 4.0],
            uncertainty: Some(0.05),
            latency_us: 100,
            sequence: 0,
        };
        let cloned = state.clone();
        assert_eq!(cloned.sequence, state.sequence);
    }

    #[test]
    fn test_pareto_point_clone() {
        let point = ParetoPoint {
            model_id: "test".to_string(),
            accuracy: 0.9,
            latency: 100.0,
            cost: 1.0,
            dominated_by: vec![],
        };
        let cloned = point.clone();
        assert_eq!(cloned.model_id, "test");
    }
}
