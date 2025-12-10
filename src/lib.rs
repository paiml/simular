//! # simular
//!
//! Unified Simulation Engine for the Sovereign AI Stack.
//!
//! A falsifiable, reproducible simulation framework implementing:
//! - Toyota Production System (TPS): Jidoka, Poka-Yoke, Heijunka, Kaizen
//! - JPL Mission-Critical Verification: Power of 10 rules
//! - Popperian Falsification: Null hypothesis testing
//!
//! ## Example
//!
//! ```rust
//! use simular::prelude::*;
//!
//! // Create a deterministic simulation
//! let config = SimConfig::builder()
//!     .seed(42)
//!     .build();
//! ```

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suspicious_operation_groupings,  // False positive for variance = E[X²] - E[X]²
    clippy::suboptimal_flops,  // Manual Horner's method is intentional
    clippy::imprecise_flops,   // Numerical code choices are intentional
    clippy::no_effect_underscore_binding,
    clippy::too_many_lines,
    clippy::missing_const_for_fn,  // Many functions can't be const in stable Rust
    clippy::needless_range_loop,   // Sometimes range loops are clearer
    clippy::manual_midpoint,       // Manual midpoint is intentional in numerical code
)]

pub mod config;
pub mod engine;
pub mod domains;
pub mod replay;
pub mod error;
pub mod falsification;
pub mod scenarios;
pub mod discovery;
pub mod visualization;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::config::{SimConfig, SimConfigBuilder};
    pub use crate::engine::{SimEngine, SimState, SimTime};
    pub use crate::engine::rng::SimRng;
    pub use crate::engine::jidoka::{JidokaGuard, JidokaViolation};
    pub use crate::error::{SimError, SimResult};
    pub use crate::falsification::{FalsifiableHypothesis, NHSTResult};
}

/// Re-export for public API
pub use error::{SimError, SimResult};
