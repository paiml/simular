//! Pre-built simulation scenarios.
//!
//! Provides ready-to-use scenario templates for common simulation types:
//! - Rocket launch and stage separation
//! - Satellite orbital mechanics
//! - Pendulum systems (canonical physics example)
//! - Climate models (energy balance)
//! - Portfolio risk (Monte Carlo `VaR`)
//! - Epidemic models (SIR/SEIR)

pub mod rocket;
pub mod satellite;
pub mod pendulum;
pub mod climate;
pub mod portfolio;
pub mod epidemic;

pub use rocket::{RocketScenario, RocketConfig, StageSeparation};
pub use satellite::{SatelliteScenario, OrbitalElements};
pub use pendulum::{PendulumScenario, PendulumConfig};
pub use climate::{ClimateScenario, ClimateConfig};
pub use portfolio::{PortfolioScenario, PortfolioConfig, VaRResult};
pub use epidemic::{SIRScenario, SIRConfig, SEIRScenario, SEIRConfig};
