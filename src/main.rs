//! simular CLI - Unified Simulation Engine
//!
//! Command-line interface for running simulations.

use std::process::ExitCode;

fn main() -> ExitCode {
    println!("simular v{}", env!("CARGO_PKG_VERSION"));
    println!("Unified Simulation Engine for the Sovereign AI Stack");
    println!();
    println!("Usage: simular <config.yaml>");
    println!();
    println!("See documentation at https://github.com/paiml/simular");

    ExitCode::SUCCESS
}
