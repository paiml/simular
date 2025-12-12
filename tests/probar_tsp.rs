//! Probar E2E tests for TSP YAML-first architecture.
//!
//! These tests verify:
//! - YAML loading works correctly
//! - Deterministic replay produces identical results
//! - User modifications to YAML affect results
//! - Jidoka validators catch invalid data
//!
//! # OR-001 Reference
//!
//! See `docs/specifications/simple-or-example.md` for the full specification.

use simular::demos::{TspInstanceError, TspInstanceYaml};

const BAY_AREA_YAML: &str = include_str!("../examples/experiments/bay_area_tsp.yaml");

// =============================================================================
// Probar E2E: YAML Loading
// =============================================================================

#[test]
fn probar_tsp_yaml_loads_successfully() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML);
    assert!(instance.is_ok(), "Bay Area YAML should parse successfully");
}

#[test]
fn probar_tsp_yaml_has_correct_metadata() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    assert_eq!(instance.meta.id, "TSP-BAY-006");
    assert_eq!(instance.meta.version, "1.0.0");
    assert_eq!(instance.meta.units, "miles");
    assert_eq!(instance.meta.optimal_known, Some(115));
}

#[test]
fn probar_tsp_yaml_has_six_cities() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    assert_eq!(instance.city_count(), 6);

    // Verify city names
    let names: Vec<&str> = instance.cities.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"San Francisco"));
    assert!(names.contains(&"Oakland"));
    assert!(names.contains(&"San Jose"));
    assert!(names.contains(&"Palo Alto"));
    assert!(names.contains(&"Berkeley"));
    assert!(names.contains(&"Fremont"));
}

#[test]
fn probar_tsp_yaml_has_correct_distances() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    // Spot check key distances
    assert_eq!(instance.distance(0, 1), 12); // SF to Oakland
    assert_eq!(instance.distance(1, 4), 4);  // Oakland to Berkeley
    assert_eq!(instance.distance(2, 3), 15); // San Jose to Palo Alto
}

// =============================================================================
// Probar E2E: Optimal Tour Verification
// =============================================================================

#[test]
fn probar_tsp_optimal_tour_is_115_miles() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    // Optimal tour: SF(0) -> OAK(1) -> BRK(4) -> FRE(5) -> SJ(2) -> PA(3) -> SF(0)
    let optimal_tour = vec![0, 1, 4, 5, 2, 3];
    let length = instance.tour_length(&optimal_tour);

    assert_eq!(
        length, 115,
        "Optimal tour should be 115 miles: 12 + 4 + 32 + 17 + 15 + 35 = 115"
    );
}

#[test]
fn probar_tsp_optimal_tour_verified_step_by_step() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    let optimal_tour = vec![0, 1, 4, 5, 2, 3];
    let (length, steps) = instance.tour_length_verified(&optimal_tour);

    assert_eq!(length, 115);
    assert_eq!(steps.len(), 6);

    // Verify first step: SF -> Oakland = 12 miles
    assert!(steps[0].contains("San Francisco"));
    assert!(steps[0].contains("Oakland"));
    assert!(steps[0].contains("12"));

    // Verify last step: Palo Alto -> SF = 35 miles
    assert!(steps[5].contains("Palo Alto"));
    assert!(steps[5].contains("San Francisco"));
    assert!(steps[5].contains("35"));
}

#[test]
fn probar_tsp_all_cities_visited_once() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    let optimal_tour = vec![0, 1, 4, 5, 2, 3];

    // Verify all cities present exactly once
    let mut visited = optimal_tour.clone();
    visited.sort();
    assert_eq!(visited, vec![0, 1, 2, 3, 4, 5]);
}

// =============================================================================
// Probar E2E: Jidoka Validation
// =============================================================================

#[test]
fn probar_tsp_validates_successfully() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    assert!(instance.validate().is_ok(), "Instance should pass validation");
}

#[test]
fn probar_tsp_symmetry_check_passes() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    assert!(
        instance.check_symmetry().is_ok(),
        "Distance matrix should be symmetric"
    );
}

#[test]
fn probar_tsp_triangle_inequality_passes() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("YAML should parse");

    assert!(
        instance.check_triangle_inequality().is_ok(),
        "Distance matrix should satisfy triangle inequality"
    );
}

// =============================================================================
// Probar E2E: User Modification Tests (YAML-First)
// =============================================================================

#[test]
fn probar_tsp_user_modified_distance_affects_tour() {
    // User edits: reduce SF->Oakland from 12 to 5 miles
    let modified_yaml = BAY_AREA_YAML.replace(
        "[ 0, 12, 48, 35, 14, 42]",
        "[ 0,  5, 48, 35, 14, 42]",
    );

    let instance = TspInstanceYaml::from_yaml(&modified_yaml).expect("Modified YAML should parse");

    // Original tour length should be different
    let optimal_tour = vec![0, 1, 4, 5, 2, 3];
    let length = instance.tour_length(&optimal_tour);

    // New length: 5 + 4 + 32 + 17 + 15 + 35 = 108 (7 miles shorter)
    assert_eq!(length, 108, "Modified tour should be 108 miles");
}

#[test]
fn probar_tsp_user_can_change_algorithm() {
    // User selects brute_force instead of grasp
    let modified_yaml = BAY_AREA_YAML.replace(
        "method: \"grasp\"",
        "method: \"brute_force\"",
    );

    let instance =
        TspInstanceYaml::from_yaml(&modified_yaml).expect("Modified algorithm should parse");

    assert_eq!(instance.algorithm.method, "brute_force");
}

#[test]
fn probar_tsp_user_can_change_seed() {
    let modified_yaml = BAY_AREA_YAML.replace("seed: 42", "seed: 12345");

    let instance = TspInstanceYaml::from_yaml(&modified_yaml).expect("Modified seed should parse");

    assert_eq!(instance.algorithm.params.seed, 12345);
}

#[test]
fn probar_tsp_user_can_add_city() {
    // Add Mountain View as city 6
    let modified_yaml = BAY_AREA_YAML
        .replace(
            "    coords: { lat: 37.5485, lon: -121.9886 }",
            "    coords: { lat: 37.5485, lon: -121.9886 }\n  - id: 6\n    name: \"Mountain View\"\n    alias: \"MTV\"\n    coords: { lat: 37.3861, lon: -122.0839 }",
        )
        // Extend matrix to 7x7
        .replace(
            "  - [42, 30, 17, 18, 32,  0]  # From FRE",
            "  - [42, 30, 17, 18, 32,  0, 15]  # From FRE\n  - [40, 35, 10, 8, 38, 15,  0]  # From MTV",
        )
        .replace("[ 0, 12, 48, 35, 14, 42]", "[ 0, 12, 48, 35, 14, 42, 40]")
        .replace("[12,  0, 42, 30,  4, 30]", "[12,  0, 42, 30,  4, 30, 35]")
        .replace("[48, 42,  0, 15, 46, 17]", "[48, 42,  0, 15, 46, 17, 10]")
        .replace("[35, 30, 15,  0, 32, 18]", "[35, 30, 15,  0, 32, 18,  8]")
        .replace("[14,  4, 46, 32,  0, 32]", "[14,  4, 46, 32,  0, 32, 38]");

    let instance = TspInstanceYaml::from_yaml(&modified_yaml).expect("Extended YAML should parse");

    assert_eq!(instance.city_count(), 7);
    assert_eq!(instance.cities[6].name, "Mountain View");
}

// =============================================================================
// Probar E2E: Deterministic Replay
// =============================================================================

#[test]
fn probar_tsp_deterministic_parsing() {
    // Parse twice, results should be identical
    let instance1 = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("First parse");
    let instance2 = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Second parse");

    assert_eq!(instance1.meta.id, instance2.meta.id);
    assert_eq!(instance1.cities.len(), instance2.cities.len());
    assert_eq!(instance1.matrix, instance2.matrix);
    assert_eq!(instance1.algorithm.params.seed, instance2.algorithm.params.seed);
}

#[test]
fn probar_tsp_roundtrip_preserves_data() {
    let original = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse original");
    let yaml = original.to_yaml().expect("Serialize");
    let restored = TspInstanceYaml::from_yaml(&yaml).expect("Parse restored");

    assert_eq!(original.meta.id, restored.meta.id);
    assert_eq!(original.matrix, restored.matrix);
    assert_eq!(original.city_count(), restored.city_count());
}

// =============================================================================
// Probar E2E: Error Handling
// =============================================================================

#[test]
fn probar_tsp_rejects_invalid_yaml() {
    let invalid = "this is not valid yaml: [[[";
    let result = TspInstanceYaml::from_yaml(invalid);

    assert!(result.is_err());
    assert!(matches!(result, Err(TspInstanceError::ParseError(_))));
}

#[test]
fn probar_tsp_detects_asymmetric_matrix() {
    // Make matrix asymmetric: SF->Oakland = 12, but Oakland->SF = 99
    let asymmetric_yaml = BAY_AREA_YAML.replace(
        "[12,  0, 42, 30,  4, 30]",
        "[99,  0, 42, 30,  4, 30]",
    );

    let instance = TspInstanceYaml::from_yaml(&asymmetric_yaml).expect("Parse");
    let result = instance.check_symmetry();

    assert!(result.is_err());
    assert!(matches!(result, Err(TspInstanceError::AsymmetricMatrix { .. })));
}

#[test]
fn probar_tsp_detects_triangle_violation() {
    // Make triangle inequality fail: SF->SJ = 500 (but SF->PA->SJ = 35+15=50)
    let violation_yaml = BAY_AREA_YAML.replace(
        "[ 0, 12, 48, 35, 14, 42]",
        "[ 0, 12, 500, 35, 14, 42]",
    ).replace(
        "[48, 42,  0, 15, 46, 17]",
        "[500, 42,  0, 15, 46, 17]",
    );

    let instance = TspInstanceYaml::from_yaml(&violation_yaml).expect("Parse");
    let result = instance.check_triangle_inequality();

    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(TspInstanceError::TriangleInequalityViolation { .. })
    ));
}
