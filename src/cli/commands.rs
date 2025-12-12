//! CLI command handlers.
//!
//! This module contains the execution logic for each CLI command.
//! Extracted to enable comprehensive testing of command behavior.

use crate::edd::v2::{validate_emc_yaml, validate_experiment_yaml, SchemaValidationError};
use crate::edd::{ExperimentRunner, RunnerConfig};
use std::path::Path;
use std::process::ExitCode;

use super::output::{
    print_emc_report, print_emc_validation_results, print_experiment_result, print_help,
    print_version,
};
use super::schema::validate_emc_schema;
use super::{Args, Command};

/// Main CLI entry point.
///
/// Dispatches to the appropriate command handler based on parsed arguments.
#[must_use]
pub fn run_cli(args: Args) -> ExitCode {
    match args.command {
        Command::Run {
            experiment_path,
            seed_override,
            verbose,
        } => run_experiment(&experiment_path, seed_override, verbose),
        Command::Validate { experiment_path } => validate_experiment(&experiment_path),
        Command::Verify {
            experiment_path,
            runs,
        } => verify_reproducibility(&experiment_path, runs),
        Command::EmcCheck { experiment_path } => emc_check(&experiment_path),
        Command::EmcValidate { emc_path } => emc_validate(&emc_path),
        Command::ListEmc => list_emc(),
        Command::Help => {
            print_help();
            ExitCode::SUCCESS
        }
        Command::Version => {
            print_version();
            ExitCode::SUCCESS
        }
    }
}

/// Run an experiment from a YAML file.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
/// * `seed_override` - Optional seed to override the experiment's configured seed
/// * `verbose` - Whether to enable verbose output
#[must_use]
pub fn run_experiment(path: &Path, seed_override: Option<u64>, verbose: bool) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           simular - EDD Experiment Runner                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let config = RunnerConfig {
        seed_override,
        verbose,
        ..RunnerConfig::default()
    };

    let mut runner = ExperimentRunner::with_config(config);

    // Initialize EMC registry
    match runner.initialize() {
        Ok(count) => {
            if verbose {
                println!("Loaded {count} EMCs from library");
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to scan EMC library: {e}");
        }
    }

    println!("Running experiment: {}\n", path.display());

    match runner.run(path) {
        Ok(result) => {
            print_experiment_result(&result, verbose);
            if result.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

/// Validate an experiment YAML file against the EDD v2 schema.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
#[must_use]
pub fn validate_experiment(path: &Path) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       simular - EDD v2 Experiment Schema Validation           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Validating: {}\n", path.display());

    // Read file
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("✗ Error reading file: {e}");
            return ExitCode::from(1);
        }
    };

    // Validate against EDD v2 schema
    match validate_experiment_yaml(&contents) {
        Ok(()) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✓ Schema validation PASSED");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            println!("EDD v2 Compliance:");
            println!("  ✓ Required fields present (id, seed, emc_ref, simulation, falsification)");
            println!("  ✓ Falsification criteria defined");
            println!("  ✓ No prohibited custom code fields");
            println!("\nNext steps:");
            println!("  • Run: simular run {}", path.display());
            println!("  • Verify: simular verify {}", path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✗ Schema validation FAILED");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            match e {
                SchemaValidationError::ValidationFailed(errors) => {
                    println!("Validation errors:");
                    for (i, err) in errors.iter().enumerate() {
                        println!("  {}. {err}", i + 1);
                    }
                }
                other => {
                    println!("Error: {other}");
                }
            }
            println!("\nSee: docs/specifications/EDD-spec-unified.md for schema requirements");
            ExitCode::from(1)
        }
    }
}

/// Verify reproducibility of an experiment across multiple runs.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
/// * `runs` - Number of verification runs to perform
#[must_use]
pub fn verify_reproducibility(path: &Path, runs: usize) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        simular - Reproducibility Verification                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let config = RunnerConfig {
        verify_reproducibility: true,
        reproducibility_runs: runs,
        ..RunnerConfig::default()
    };

    let mut runner = ExperimentRunner::with_config(config);

    if let Err(e) = runner.initialize() {
        eprintln!("Warning: Failed to scan EMC library: {e}");
    }

    println!("Verifying reproducibility: {}", path.display());
    println!("Runs: {runs}\n");

    match runner.verify(path) {
        Ok(summary) => {
            let status = if summary.passed { "PASSED" } else { "FAILED" };
            let sym = if summary.passed { "✓" } else { "✗" };

            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Reproducibility Check");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            println!("  Runs:      {}", summary.runs);
            println!("  Identical: {}", summary.identical);
            println!("  Platform:  {}", summary.platform);
            println!("\n  Reference Hash: {}", summary.reference_hash);

            if summary.run_hashes.len() > 1 {
                println!("\n  Run Hashes:");
                for (i, hash) in summary.run_hashes.iter().enumerate() {
                    let match_sym = if hash == &summary.reference_hash {
                        "="
                    } else {
                        "!"
                    };
                    println!("    Run {}: {} {}", i + 1, hash, match_sym);
                }
            }

            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("{sym} Result: {status}");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            if summary.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

/// Check EMC compliance for an experiment.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
#[must_use]
pub fn emc_check(path: &Path) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          simular - EMC Compliance Check                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let mut runner = ExperimentRunner::new();

    if let Err(e) = runner.initialize() {
        eprintln!("Warning: Failed to scan EMC library: {e}");
    }

    println!("Checking EMC compliance: {}\n", path.display());

    match runner.emc_check(path) {
        Ok(report) => {
            print_emc_report(&report);
            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

/// List all EMCs available in the library.
#[must_use]
pub fn list_emc() -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             simular - EMC Library                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let mut runner = ExperimentRunner::new();

    match runner.initialize() {
        Ok(count) => {
            println!("Found {count} EMCs in library:\n");

            let references = runner.registry_mut().list_references();
            let mut sorted: Vec<_> = references.into_iter().collect();
            sorted.sort_unstable();

            let mut current_domain = String::new();

            for reference in &sorted {
                let parts: Vec<&str> = reference.split('/').collect();
                if parts.len() >= 2 {
                    let domain = parts[0];
                    if domain != current_domain {
                        if !current_domain.is_empty() {
                            println!();
                        }
                        println!("{}:", domain.to_uppercase());
                        current_domain = domain.to_string();
                    }
                }
                println!("  - {reference}");
            }

            println!("\nUsage: simular run <experiment.yaml>");
            println!(
                "Reference EMCs using: equation_model_card.emc_ref: \"{}\"",
                if sorted.is_empty() {
                    "domain/name"
                } else {
                    sorted[0]
                }
            );

            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error scanning EMC library: {e}");
            ExitCode::from(1)
        }
    }
}

/// Validate an EMC YAML file against the EDD v2 schema.
///
/// # Arguments
///
/// * `path` - Path to the EMC file
#[must_use]
pub fn emc_validate(path: &Path) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        simular - EDD v2 EMC Schema Validation                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    println!("Validating EMC: {}\n", path.display());

    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("✗ Error reading file: {e}");
            return ExitCode::from(1);
        }
    };

    // First, validate against EDD v2 JSON schema
    match validate_emc_yaml(&contents) {
        Ok(()) => {
            println!("✓ EDD v2 JSON Schema validation PASSED\n");
        }
        Err(e) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✗ EDD v2 JSON Schema validation FAILED");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            match e {
                SchemaValidationError::ValidationFailed(errors) => {
                    println!("Validation errors:");
                    for (i, err) in errors.iter().enumerate() {
                        println!("  {}. {err}", i + 1);
                    }
                }
                other => {
                    println!("Error: {other}");
                }
            }
            return ExitCode::from(1);
        }
    }

    // Then run the semantic validation
    let yaml: serde_yaml::Value = match serde_yaml::from_str(&contents) {
        Ok(y) => y,
        Err(e) => {
            eprintln!("YAML Syntax Error: {e}");
            return ExitCode::from(1);
        }
    };

    let (errors, warnings) = validate_emc_schema(&yaml);
    print_emc_validation_results(&yaml, &errors, &warnings);

    if errors.is_empty() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✓ All EMC validations PASSED");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
