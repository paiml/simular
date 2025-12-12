//! TUI module for simular.
//!
//! This module contains reusable TUI application state and logic
//! extracted from bin/*.rs to enable testing.
//!
//! The actual terminal I/O remains in the binaries, but all testable
//! state management and business logic lives here.

#[cfg(feature = "tui")]
pub mod orbit_app;
#[cfg(feature = "tui")]
pub mod tsp_app;

#[cfg(test)]
#[cfg(feature = "tui")]
mod tests;
