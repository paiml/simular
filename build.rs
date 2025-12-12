/// Build script for simular
/// Captures build environment for reproducibility

fn main() {
    // Capture build metadata for reproducibility verification
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=rust-toolchain.toml");

    // Embed version information
    if let Ok(version) = std::env::var("CARGO_PKG_VERSION") {
        println!("cargo:rustc-env=SIMULAR_VERSION={version}");
    }

    // Capture git hash for reproducibility
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if let Ok(hash) = String::from_utf8(output.stdout) {
            println!("cargo:rustc-env=GIT_HASH={}", hash.trim());
        }
    }

    // Capture build timestamp (ISO 8601)
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        chrono_lite_timestamp()
    );
}

/// Simple ISO 8601 timestamp without external crate
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Approximate UTC timestamp (not leap-second accurate, but sufficient)
    format!("{secs}")
}
