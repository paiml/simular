//! TSP Instance YAML Configuration
//!
//! YAML-first architecture for TSP instances. Users can download, modify, and re-run
//! experiments without touching code.
//!
//! # Example YAML
//!
//! ```yaml
//! meta:
//!   id: "TSP-BAY-006"
//!   version: "1.0.0"
//!   description: "6-city Bay Area ground truth instance"
//!   source: "Google Maps (Dec 2024)"
//!   units: "miles"
//!   optimal_known: 115
//!
//! cities:
//!   - id: 0
//!     name: "San Francisco"
//!     alias: "SF"
//!     coords: { lat: 37.7749, lon: -122.4194 }
//!
//! matrix:
//!   - [0, 12, 48]
//!   - [12, 0, 42]
//!   - [48, 42, 0]
//!
//! algorithm:
//!   method: "grasp"
//!   params:
//!     rcl_size: 3
//!     restarts: 10
//!     two_opt: true
//!     seed: 42
//! ```

use serde::{Deserialize, Serialize};

/// Geographic coordinates (lat/lon).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coords {
    /// Latitude in degrees.
    pub lat: f64,
    /// Longitude in degrees.
    pub lon: f64,
}

impl Default for Coords {
    fn default() -> Self {
        Self { lat: 0.0, lon: 0.0 }
    }
}

/// A city in the TSP instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspCity {
    /// Unique city identifier (0-indexed).
    pub id: usize,
    /// Full city name.
    pub name: String,
    /// Short alias (2-4 chars).
    pub alias: String,
    /// Geographic coordinates.
    pub coords: Coords,
}

impl TspCity {
    /// Create a new city.
    #[must_use]
    pub fn new(id: usize, name: impl Into<String>, alias: impl Into<String>, lat: f64, lon: f64) -> Self {
        Self {
            id,
            name: name.into(),
            alias: alias.into(),
            coords: Coords { lat, lon },
        }
    }
}

/// Metadata about the TSP instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TspMeta {
    /// Unique instance identifier.
    pub id: String,
    /// Version string (semver).
    #[serde(default = "default_version")]
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Data source (e.g., "Google Maps").
    #[serde(default)]
    pub source: String,
    /// Distance units (e.g., "miles", "km").
    #[serde(default = "default_units")]
    pub units: String,
    /// Known optimal solution (for verification).
    #[serde(default)]
    pub optimal_known: Option<u32>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

fn default_units() -> String {
    "miles".to_string()
}

impl Default for TspMeta {
    fn default() -> Self {
        Self {
            id: "TSP-UNNAMED".to_string(),
            version: default_version(),
            description: "Unnamed TSP instance".to_string(),
            source: String::new(),
            units: default_units(),
            optimal_known: None,
        }
    }
}

/// Algorithm parameters for TSP solving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TspParams {
    /// RCL size for GRASP (restricted candidate list).
    #[serde(default = "default_rcl_size")]
    pub rcl_size: usize,
    /// Number of GRASP restarts.
    #[serde(default = "default_restarts")]
    pub restarts: usize,
    /// Enable 2-opt local search.
    #[serde(default = "default_two_opt")]
    pub two_opt: bool,
    /// Random seed for reproducibility.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_rcl_size() -> usize {
    3
}

fn default_restarts() -> usize {
    10
}

fn default_two_opt() -> bool {
    true
}

fn default_seed() -> u64 {
    42
}

impl Default for TspParams {
    fn default() -> Self {
        Self {
            rcl_size: default_rcl_size(),
            restarts: default_restarts(),
            two_opt: default_two_opt(),
            seed: default_seed(),
        }
    }
}

/// Algorithm configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TspAlgorithmConfig {
    /// Algorithm method: `greedy`, `grasp`, or `brute_force`.
    #[serde(default = "default_method")]
    pub method: String,
    /// Algorithm-specific parameters.
    #[serde(default)]
    pub params: TspParams,
}

fn default_method() -> String {
    "grasp".to_string()
}

impl Default for TspAlgorithmConfig {
    fn default() -> Self {
        Self {
            method: default_method(),
            params: TspParams::default(),
        }
    }
}

/// Complete TSP instance configuration (YAML-first).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspInstanceYaml {
    /// Instance metadata.
    pub meta: TspMeta,
    /// List of cities.
    pub cities: Vec<TspCity>,
    /// Distance matrix (n x n).
    pub matrix: Vec<Vec<u32>>,
    /// Algorithm configuration.
    #[serde(default)]
    pub algorithm: TspAlgorithmConfig,
}

/// Error types for TSP instance parsing/validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TspInstanceError {
    /// YAML parsing failed.
    ParseError(String),
    /// Matrix dimensions don't match city count.
    MatrixDimensionMismatch { expected: usize, got_rows: usize },
    /// Matrix row has wrong number of columns.
    MatrixRowMismatch { row: usize, expected: usize, got: usize },
    /// Triangle inequality violated.
    TriangleInequalityViolation { i: usize, j: usize, k: usize },
    /// Matrix is not symmetric.
    AsymmetricMatrix { i: usize, j: usize, forward: u32, backward: u32 },
    /// Invalid city ID.
    InvalidCityId { id: usize, max: usize },
    /// IO error.
    IoError(String),
}

impl std::fmt::Display for TspInstanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "YAML parse error: {msg}"),
            Self::MatrixDimensionMismatch { expected, got_rows } => {
                write!(f, "Matrix dimension mismatch: expected {expected}x{expected}, got {got_rows} rows")
            }
            Self::MatrixRowMismatch { row, expected, got } => {
                write!(f, "Matrix row {row} has {got} columns, expected {expected}")
            }
            Self::TriangleInequalityViolation { i, j, k } => {
                write!(f, "Triangle inequality violated: d({i},{k}) > d({i},{j}) + d({j},{k})")
            }
            Self::AsymmetricMatrix { i, j, forward, backward } => {
                write!(f, "Asymmetric matrix: d({i},{j})={forward} != d({j},{i})={backward}")
            }
            Self::InvalidCityId { id, max } => {
                write!(f, "Invalid city ID {id}, max is {max}")
            }
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
        }
    }
}

impl std::error::Error for TspInstanceError {}

impl TspInstanceYaml {
    /// Parse TSP instance from YAML string.
    ///
    /// # Errors
    ///
    /// Returns `TspInstanceError::ParseError` if YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self, TspInstanceError> {
        serde_yaml::from_str(yaml).map_err(|e| TspInstanceError::ParseError(e.to_string()))
    }

    /// Load TSP instance from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or YAML is invalid.
    pub fn from_yaml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TspInstanceError> {
        let content = std::fs::read_to_string(path).map_err(|e| TspInstanceError::IoError(e.to_string()))?;
        Self::from_yaml(&content)
    }

    /// Serialize to YAML string.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_yaml(&self) -> Result<String, TspInstanceError> {
        serde_yaml::to_string(self).map_err(|e| TspInstanceError::ParseError(e.to_string()))
    }

    /// Number of cities in the instance.
    #[must_use]
    pub fn city_count(&self) -> usize {
        self.cities.len()
    }

    /// Get distance between two cities.
    ///
    /// # Panics
    ///
    /// Panics if city indices are out of bounds.
    #[must_use]
    pub fn distance(&self, from: usize, to: usize) -> u32 {
        self.matrix[from][to]
    }

    /// Validate the instance (Jidoka).
    ///
    /// # Errors
    ///
    /// Returns error if validation fails.
    pub fn validate(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();

        // Check matrix dimensions
        if self.matrix.len() != n {
            return Err(TspInstanceError::MatrixDimensionMismatch {
                expected: n,
                got_rows: self.matrix.len(),
            });
        }

        // Check each row has correct length
        for (i, row) in self.matrix.iter().enumerate() {
            if row.len() != n {
                return Err(TspInstanceError::MatrixRowMismatch {
                    row: i,
                    expected: n,
                    got: row.len(),
                });
            }
        }

        // Check city IDs are valid
        for city in &self.cities {
            if city.id >= n {
                return Err(TspInstanceError::InvalidCityId { id: city.id, max: n - 1 });
            }
        }

        Ok(())
    }

    /// Check if matrix is symmetric.
    ///
    /// # Errors
    ///
    /// Returns error with first asymmetry found.
    pub fn check_symmetry(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let forward = self.matrix[i][j];
                let backward = self.matrix[j][i];
                if forward != backward {
                    return Err(TspInstanceError::AsymmetricMatrix { i, j, forward, backward });
                }
            }
        }
        Ok(())
    }

    /// Check triangle inequality for all city triples.
    ///
    /// # Errors
    ///
    /// Returns error with first violation found.
    pub fn check_triangle_inequality(&self) -> Result<(), TspInstanceError> {
        let n = self.cities.len();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if i != j && j != k && i != k {
                        let direct = self.matrix[i][k];
                        let via_j = self.matrix[i][j].saturating_add(self.matrix[j][k]);
                        if direct > via_j {
                            return Err(TspInstanceError::TriangleInequalityViolation { i, j, k });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute tour length for given city sequence.
    ///
    /// # Arguments
    ///
    /// * `tour` - Sequence of city indices (must visit each city once)
    ///
    /// # Returns
    ///
    /// Total tour length including return to start.
    #[must_use]
    pub fn tour_length(&self, tour: &[usize]) -> u32 {
        if tour.is_empty() {
            return 0;
        }
        let mut total = 0u32;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total = total.saturating_add(self.matrix[from][to]);
        }
        total
    }

    /// Compute tour length with step-by-step verification (Genchi Genbutsu).
    ///
    /// # Returns
    ///
    /// Tuple of (`total_length`, `step_descriptions`).
    #[must_use]
    pub fn tour_length_verified(&self, tour: &[usize]) -> (u32, Vec<String>) {
        let mut total = 0u32;
        let mut steps = Vec::new();

        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            let dist = self.matrix[from][to];
            total = total.saturating_add(dist);

            let from_name = self.cities.get(from).map_or("?", |c| &c.name);
            let to_name = self.cities.get(to).map_or("?", |c| &c.name);
            steps.push(format!(
                "Step {}: {} → {} = {} {} (total: {})",
                i + 1,
                from_name,
                to_name,
                dist,
                self.meta.units,
                total
            ));
        }

        (total, steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_YAML: &str = r#"
meta:
  id: "TEST-001"
  description: "Minimal test instance"
cities:
  - id: 0
    name: "City A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "City B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
"#;

    const BAY_AREA_YAML: &str = r#"
meta:
  id: "TSP-BAY-006"
  version: "1.0.0"
  description: "6-city Bay Area ground truth instance"
  source: "Google Maps (Dec 2024)"
  units: "miles"
  optimal_known: 115
cities:
  - id: 0
    name: "San Francisco"
    alias: "SF"
    coords: { lat: 37.7749, lon: -122.4194 }
  - id: 1
    name: "Oakland"
    alias: "OAK"
    coords: { lat: 37.8044, lon: -122.2712 }
  - id: 2
    name: "San Jose"
    alias: "SJ"
    coords: { lat: 37.3382, lon: -121.8863 }
  - id: 3
    name: "Palo Alto"
    alias: "PA"
    coords: { lat: 37.4419, lon: -122.1430 }
  - id: 4
    name: "Berkeley"
    alias: "BRK"
    coords: { lat: 37.8716, lon: -122.2727 }
  - id: 5
    name: "Fremont"
    alias: "FRE"
    coords: { lat: 37.5485, lon: -121.9886 }
matrix:
  - [ 0, 12, 48, 35, 14, 42]
  - [12,  0, 42, 30,  4, 30]
  - [48, 42,  0, 15, 46, 17]
  - [35, 30, 15,  0, 32, 18]
  - [14,  4, 46, 32,  0, 32]
  - [42, 30, 17, 18, 32,  0]
algorithm:
  method: "grasp"
  params:
    rcl_size: 3
    restarts: 10
    two_opt: true
    seed: 42
"#;

    // =========================================================================
    // OR-001-01: TspInstanceYaml struct + serde tests
    // =========================================================================

    #[test]
    fn test_deserialize_valid_yaml() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Should parse valid YAML");
        assert_eq!(instance.meta.id, "TSP-BAY-006");
        assert_eq!(instance.cities.len(), 6);
        assert_eq!(instance.matrix.len(), 6);
        assert_eq!(instance.algorithm.method, "grasp");
    }

    #[test]
    fn test_deserialize_minimal_yaml() {
        let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Should parse minimal YAML");
        assert_eq!(instance.meta.id, "TEST-001");
        assert_eq!(instance.cities.len(), 2);
        assert_eq!(instance.matrix.len(), 2);
        // Defaults should apply
        assert_eq!(instance.algorithm.method, "grasp");
        assert_eq!(instance.algorithm.params.rcl_size, 3);
    }

    #[test]
    fn test_deserialize_invalid_yaml() {
        let invalid = "this is not valid yaml: [[[";
        let result = TspInstanceYaml::from_yaml(invalid);
        assert!(result.is_err());
        if let Err(TspInstanceError::ParseError(msg)) = result {
            assert!(!msg.is_empty());
        } else {
            panic!("Expected ParseError");
        }
    }

    #[test]
    fn test_deserialize_missing_required_fields() {
        let missing_cities = r#"
meta:
  id: "TEST"
  description: "No cities"
matrix: []
"#;
        let result = TspInstanceYaml::from_yaml(missing_cities);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let original = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        let yaml = original.to_yaml().expect("Serialize");
        let restored = TspInstanceYaml::from_yaml(&yaml).expect("Reparse");
        assert_eq!(original.meta.id, restored.meta.id);
        assert_eq!(original.cities.len(), restored.cities.len());
        assert_eq!(original.matrix, restored.matrix);
    }

    #[test]
    fn test_default_algorithm_params() {
        let params = TspParams::default();
        assert_eq!(params.rcl_size, 3);
        assert_eq!(params.restarts, 10);
        assert!(params.two_opt);
        assert_eq!(params.seed, 42);
    }

    #[test]
    fn test_default_meta() {
        let meta = TspMeta::default();
        assert_eq!(meta.id, "TSP-UNNAMED");
        assert_eq!(meta.version, "1.0.0");
        assert_eq!(meta.units, "miles");
        assert!(meta.optimal_known.is_none());
    }

    #[test]
    fn test_default_algorithm_config() {
        let config = TspAlgorithmConfig::default();
        assert_eq!(config.method, "grasp");
        assert_eq!(config.params.rcl_size, 3);
    }

    #[test]
    fn test_coords_default() {
        let coords = Coords::default();
        assert_eq!(coords.lat, 0.0);
        assert_eq!(coords.lon, 0.0);
    }

    #[test]
    fn test_tsp_city_new() {
        let city = TspCity::new(0, "San Francisco", "SF", 37.7749, -122.4194);
        assert_eq!(city.id, 0);
        assert_eq!(city.name, "San Francisco");
        assert_eq!(city.alias, "SF");
        assert!((city.coords.lat - 37.7749).abs() < 0.0001);
    }

    // =========================================================================
    // Validation tests (Jidoka)
    // =========================================================================

    #[test]
    fn test_validate_valid_instance() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        assert!(instance.validate().is_ok());
    }

    #[test]
    fn test_validate_matrix_dimension_mismatch() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Bad matrix"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10, 20]
  - [10, 0, 30]
  - [20, 30, 0]
"#;
        let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
        let result = instance.validate();
        assert!(matches!(result, Err(TspInstanceError::MatrixDimensionMismatch { .. })));
    }

    #[test]
    fn test_validate_matrix_row_mismatch() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Bad row"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10]
"#;
        let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
        let result = instance.validate();
        assert!(matches!(result, Err(TspInstanceError::MatrixRowMismatch { .. })));
    }

    #[test]
    fn test_validate_invalid_city_id() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Bad city ID"
cities:
  - id: 5
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
"#;
        let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
        let result = instance.validate();
        assert!(matches!(result, Err(TspInstanceError::InvalidCityId { .. })));
    }

    #[test]
    fn test_check_symmetry_valid() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        assert!(instance.check_symmetry().is_ok());
    }

    #[test]
    fn test_check_symmetry_invalid() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Asymmetric"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [20, 0]
"#;
        let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
        let result = instance.check_symmetry();
        assert!(matches!(result, Err(TspInstanceError::AsymmetricMatrix { .. })));
    }

    #[test]
    fn test_check_triangle_inequality_valid() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        assert!(instance.check_triangle_inequality().is_ok());
    }

    #[test]
    fn test_check_triangle_inequality_violation() {
        // d(0,2) = 100 > d(0,1) + d(1,2) = 10 + 10 = 20
        let yaml = r#"
meta:
  id: "TEST"
  description: "Triangle violation"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
  - id: 2
    name: "C"
    alias: "C"
    coords: { lat: 2.0, lon: 2.0 }
matrix:
  - [0, 10, 100]
  - [10, 0, 10]
  - [100, 10, 0]
"#;
        let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
        let result = instance.check_triangle_inequality();
        assert!(matches!(result, Err(TspInstanceError::TriangleInequalityViolation { .. })));
    }

    // =========================================================================
    // Tour length tests
    // =========================================================================

    #[test]
    fn test_tour_length_bay_area_optimal() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        // Optimal tour: SF(0) → OAK(1) → BRK(4) → FRE(5) → SJ(2) → PA(3) → SF(0)
        let tour = vec![0, 1, 4, 5, 2, 3];
        let length = instance.tour_length(&tour);
        // 12 + 4 + 32 + 17 + 15 + 35 = 115
        assert_eq!(length, 115);
    }

    #[test]
    fn test_tour_length_empty() {
        let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Parse");
        let length = instance.tour_length(&[]);
        assert_eq!(length, 0);
    }

    #[test]
    fn test_tour_length_single_city() {
        let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Parse");
        let length = instance.tour_length(&[0]);
        assert_eq!(length, 0); // d(0,0) = 0
    }

    #[test]
    fn test_tour_length_verified() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        let tour = vec![0, 1, 4, 5, 2, 3];
        let (length, steps) = instance.tour_length_verified(&tour);
        assert_eq!(length, 115);
        assert_eq!(steps.len(), 6);
        assert!(steps[0].contains("San Francisco"));
        assert!(steps[0].contains("Oakland"));
        assert!(steps[0].contains("12"));
    }

    // =========================================================================
    // Accessor tests
    // =========================================================================

    #[test]
    fn test_city_count() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        assert_eq!(instance.city_count(), 6);
    }

    #[test]
    fn test_distance() {
        let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
        assert_eq!(instance.distance(0, 1), 12); // SF to Oakland
        assert_eq!(instance.distance(1, 4), 4);  // Oakland to Berkeley
    }

    // =========================================================================
    // Error display tests
    // =========================================================================

    #[test]
    fn test_error_display_parse() {
        let err = TspInstanceError::ParseError("test error".to_string());
        assert!(err.to_string().contains("YAML parse error"));
    }

    #[test]
    fn test_error_display_matrix_dimension() {
        let err = TspInstanceError::MatrixDimensionMismatch { expected: 6, got_rows: 4 };
        let msg = err.to_string();
        assert!(msg.contains("6x6"));
        assert!(msg.contains("4 rows"));
    }

    #[test]
    fn test_error_display_matrix_row() {
        let err = TspInstanceError::MatrixRowMismatch { row: 2, expected: 6, got: 4 };
        let msg = err.to_string();
        assert!(msg.contains("row 2"));
        assert!(msg.contains("4 columns"));
    }

    #[test]
    fn test_error_display_triangle() {
        let err = TspInstanceError::TriangleInequalityViolation { i: 0, j: 1, k: 2 };
        assert!(err.to_string().contains("Triangle inequality"));
    }

    #[test]
    fn test_error_display_asymmetric() {
        let err = TspInstanceError::AsymmetricMatrix { i: 0, j: 1, forward: 10, backward: 20 };
        let msg = err.to_string();
        assert!(msg.contains("Asymmetric"));
        assert!(msg.contains("10"));
        assert!(msg.contains("20"));
    }

    #[test]
    fn test_error_display_invalid_city() {
        let err = TspInstanceError::InvalidCityId { id: 10, max: 5 };
        assert!(err.to_string().contains("Invalid city ID"));
    }

    #[test]
    fn test_error_display_io() {
        let err = TspInstanceError::IoError("file not found".to_string());
        assert!(err.to_string().contains("IO error"));
    }

    // =========================================================================
    // File loading tests
    // =========================================================================

    #[test]
    fn test_from_yaml_file_not_found() {
        let result = TspInstanceYaml::from_yaml_file("/nonexistent/path/file.yaml");
        assert!(matches!(result, Err(TspInstanceError::IoError(_))));
    }

    // =========================================================================
    // Trait implementations
    // =========================================================================

    #[test]
    fn test_coords_clone_and_copy() {
        let coords = Coords { lat: 37.0, lon: -122.0 };
        let cloned = coords.clone();
        let copied = coords;
        assert_eq!(coords, cloned);
        assert_eq!(coords, copied);
    }

    #[test]
    fn test_coords_partial_eq() {
        let c1 = Coords { lat: 37.0, lon: -122.0 };
        let c2 = Coords { lat: 37.0, lon: -122.0 };
        let c3 = Coords { lat: 38.0, lon: -122.0 };
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_tsp_city_clone() {
        let city = TspCity::new(0, "SF", "SF", 37.0, -122.0);
        let cloned = city.clone();
        assert_eq!(city, cloned);
    }

    #[test]
    fn test_tsp_meta_clone() {
        let meta = TspMeta::default();
        let cloned = meta.clone();
        assert_eq!(meta, cloned);
    }

    #[test]
    fn test_tsp_params_clone() {
        let params = TspParams::default();
        let cloned = params.clone();
        assert_eq!(params, cloned);
    }

    #[test]
    fn test_tsp_algorithm_config_clone() {
        let config = TspAlgorithmConfig::default();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_tsp_instance_yaml_clone() {
        let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Parse");
        let cloned = instance.clone();
        assert_eq!(instance, cloned);
    }

    #[test]
    fn test_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(TspInstanceError::ParseError("test".to_string()));
        assert!(!err.to_string().is_empty());
    }

    // =========================================================================
    // Debug trait tests
    // =========================================================================

    #[test]
    fn test_coords_debug() {
        let coords = Coords { lat: 37.0, lon: -122.0 };
        let debug = format!("{:?}", coords);
        assert!(debug.contains("Coords"));
        assert!(debug.contains("37"));
    }

    #[test]
    fn test_tsp_instance_error_debug() {
        let err = TspInstanceError::ParseError("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("ParseError"));
    }
}
