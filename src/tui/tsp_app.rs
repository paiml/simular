//! TSP GRASP TUI application state and logic.
//!
//! This module contains the testable state and logic for the TSP GRASP TUI demo.
//! Terminal I/O is handled by the binary, but all state management lives here.
//!
//! # YAML-First Architecture
//!
//! The TUI supports loading TSP instances from YAML files:
//!
//! ```bash
//! # Load Bay Area TSP instance
//! cargo run --bin tsp_tui -- examples/experiments/bay_area_tsp.yaml
//! ```
//!
//! Or use the 'L' key to trigger file loading (handled by binary).

use crate::demos::tsp_grasp::{ConstructionMethod, TspGraspDemo};
use crate::demos::tsp_instance::{TspInstanceError, TspInstanceYaml};
use crate::demos::EddDemo;
use crossterm::event::KeyCode;
use std::path::Path;

/// Application state for the TSP GRASP TUI demo.
pub struct TspApp {
    /// The underlying GRASP demo.
    pub demo: TspGraspDemo,
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Whether auto-run is enabled.
    pub auto_run: bool,
    /// Frame counter.
    pub frame_count: u64,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// Convergence history (scaled tour lengths).
    pub convergence_history: Vec<u64>,
    /// Maximum history length.
    pub max_history: usize,
    /// Loaded YAML instance (if any).
    pub loaded_instance: Option<TspInstanceYaml>,
    /// Path to loaded file (if any).
    pub loaded_path: Option<String>,
}

impl TspApp {
    /// Create a new TSP application.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of cities
    /// * `seed` - Random seed for reproducibility
    #[must_use]
    pub fn new(n: usize, seed: u64) -> Self {
        let mut demo = TspGraspDemo::new(seed, n);
        // Run initial GRASP iteration so we have something to display
        demo.grasp_iteration();

        let mut convergence_history = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tour_len = (demo.best_tour_length * 1000.0).max(0.0) as u64;
        convergence_history.push(tour_len);

        Self {
            demo,
            paused: false,
            auto_run: true, // Start running automatically
            frame_count: 0,
            should_quit: false,
            convergence_history,
            max_history: 50,
            loaded_instance: None,
            loaded_path: None,
        }
    }

    /// Create application from YAML string.
    ///
    /// # Errors
    ///
    /// Returns error if YAML parsing or validation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use simular::tui::tsp_app::TspApp;
    ///
    /// let yaml = include_str!("../../examples/experiments/bay_area_tsp.yaml");
    /// let app = TspApp::from_yaml(yaml).expect("YAML should parse");
    /// assert_eq!(app.demo.n, 6);
    /// ```
    pub fn from_yaml(yaml: &str) -> Result<Self, TspInstanceError> {
        let instance = TspInstanceYaml::from_yaml(yaml)?;
        instance.validate()?;

        let mut demo = TspGraspDemo::from_instance(&instance);
        demo.grasp_iteration();

        let mut convergence_history = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tour_len = (demo.best_tour_length * 1000.0).max(0.0) as u64;
        convergence_history.push(tour_len);

        Ok(Self {
            demo,
            paused: false,
            auto_run: true,
            frame_count: 0,
            should_quit: false,
            convergence_history,
            max_history: 50,
            loaded_instance: Some(instance),
            loaded_path: None,
        })
    }

    /// Create application from YAML file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or YAML is invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use simular::tui::tsp_app::TspApp;
    ///
    /// let app = TspApp::from_yaml_file("examples/experiments/bay_area_tsp.yaml")
    ///     .expect("File should load");
    /// assert_eq!(app.demo.n, 6);
    /// assert!(app.loaded_path.is_some());
    /// ```
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, TspInstanceError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let instance = TspInstanceYaml::from_yaml_file(&path)?;
        instance.validate()?;

        let mut demo = TspGraspDemo::from_instance(&instance);
        demo.grasp_iteration();

        let mut convergence_history = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tour_len = (demo.best_tour_length * 1000.0).max(0.0) as u64;
        convergence_history.push(tour_len);

        Ok(Self {
            demo,
            paused: false,
            auto_run: true,
            frame_count: 0,
            should_quit: false,
            convergence_history,
            max_history: 50,
            loaded_instance: Some(instance),
            loaded_path: Some(path_str),
        })
    }

    /// Get the instance ID if loaded from YAML.
    #[must_use]
    pub fn instance_id(&self) -> Option<&str> {
        self.loaded_instance.as_ref().map(|i| i.meta.id.as_str())
    }

    /// Get the optimal known value if loaded from YAML.
    #[must_use]
    pub fn optimal_known(&self) -> Option<u32> {
        self.loaded_instance.as_ref().and_then(|i| i.meta.optimal_known)
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        let seed = self.demo.seed;
        let n = self.demo.n;
        self.demo = TspGraspDemo::new(seed, n);
        self.convergence_history.clear();
        self.frame_count = 0;
    }

    /// Execute one step of the simulation.
    pub fn step(&mut self) {
        if !self.paused || self.auto_run {
            self.demo.grasp_iteration();

            // Track convergence history
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let tour_len = (self.demo.best_tour_length * 1000.0).max(0.0) as u64;
            self.convergence_history.push(tour_len);
            if self.convergence_history.len() > self.max_history {
                self.convergence_history.remove(0);
            }
        }
        self.frame_count += 1;
    }

    /// Handle a key press.
    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char(' ') => {
                self.auto_run = !self.auto_run;
                self.paused = !self.auto_run;
            }
            KeyCode::Char('g') => {
                // Single GRASP iteration
                self.paused = false;
                self.step();
                self.paused = true;
                self.auto_run = false;
            }
            KeyCode::Char('r') => self.reset(),
            KeyCode::Char('+' | '=') => {
                let new_size = (self.demo.rcl_size + 1).min(10);
                self.demo.set_rcl_size(new_size);
            }
            KeyCode::Char('-') => {
                let new_size = self.demo.rcl_size.saturating_sub(1).max(1);
                self.demo.set_rcl_size(new_size);
            }
            KeyCode::Char('m') => {
                // Cycle construction method
                let new_method = match self.demo.construction_method {
                    ConstructionMethod::RandomizedGreedy => ConstructionMethod::NearestNeighbor,
                    ConstructionMethod::NearestNeighbor => ConstructionMethod::Random,
                    ConstructionMethod::Random => ConstructionMethod::RandomizedGreedy,
                };
                self.demo.set_construction_method(new_method);
            }
            _ => {}
        }
    }

    /// Check if the app should quit.
    #[must_use]
    pub const fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Get the optimality gap.
    #[must_use]
    pub fn optimality_gap(&self) -> f64 {
        self.demo.optimality_gap()
    }

    /// Get the construction method name.
    #[must_use]
    pub fn construction_method_name(&self) -> &'static str {
        match self.demo.construction_method {
            ConstructionMethod::RandomizedGreedy => "Randomized Greedy",
            ConstructionMethod::NearestNeighbor => "Nearest Neighbor",
            ConstructionMethod::Random => "Random",
        }
    }

    /// Get the falsification status.
    #[must_use]
    pub fn falsification_status(&self) -> crate::demos::FalsificationStatus {
        self.demo.get_falsification_status()
    }

    /// Verify the governing equation.
    #[must_use]
    pub fn verify_equation(&self) -> bool {
        self.demo.verify_equation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_app() {
        let app = TspApp::new(10, 42);
        assert!(!app.paused);
        assert!(app.auto_run);
        assert!(!app.should_quit);
        assert_eq!(app.frame_count, 0);
        assert_eq!(app.demo.n, 10);
    }

    #[test]
    fn test_reset() {
        let mut app = TspApp::new(10, 42);
        app.frame_count = 100;
        app.convergence_history.push(500);
        app.reset();
        assert_eq!(app.frame_count, 0);
        assert!(app.convergence_history.is_empty());
    }

    #[test]
    fn test_handle_key_quit() {
        let mut app = TspApp::new(10, 42);
        assert!(!app.should_quit());
        app.handle_key(KeyCode::Char('q'));
        assert!(app.should_quit());
    }

    #[test]
    fn test_handle_key_esc() {
        let mut app = TspApp::new(10, 42);
        app.handle_key(KeyCode::Esc);
        assert!(app.should_quit());
    }

    #[test]
    fn test_handle_key_pause_toggle() {
        let mut app = TspApp::new(10, 42);
        assert!(app.auto_run);
        assert!(!app.paused);

        app.handle_key(KeyCode::Char(' '));
        assert!(!app.auto_run);
        assert!(app.paused);

        app.handle_key(KeyCode::Char(' '));
        assert!(app.auto_run);
        assert!(!app.paused);
    }

    #[test]
    fn test_handle_key_single_step() {
        let mut app = TspApp::new(10, 42);
        let initial_restarts = app.demo.restarts;
        app.handle_key(KeyCode::Char('g'));
        assert!(app.paused);
        assert!(!app.auto_run);
        assert!(app.demo.restarts > initial_restarts);
    }

    #[test]
    fn test_handle_key_reset() {
        let mut app = TspApp::new(10, 42);
        app.frame_count = 50;
        app.handle_key(KeyCode::Char('r'));
        assert_eq!(app.frame_count, 0);
    }

    #[test]
    fn test_handle_key_rcl_increase() {
        let mut app = TspApp::new(10, 42);
        let initial_size = app.demo.rcl_size;
        app.handle_key(KeyCode::Char('+'));
        assert_eq!(app.demo.rcl_size, initial_size + 1);
    }

    #[test]
    fn test_handle_key_rcl_increase_equals() {
        let mut app = TspApp::new(10, 42);
        let initial_size = app.demo.rcl_size;
        app.handle_key(KeyCode::Char('='));
        assert_eq!(app.demo.rcl_size, initial_size + 1);
    }

    #[test]
    fn test_handle_key_rcl_decrease() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(5);
        app.handle_key(KeyCode::Char('-'));
        assert_eq!(app.demo.rcl_size, 4);
    }

    #[test]
    fn test_rcl_size_max_limit() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(10);
        app.handle_key(KeyCode::Char('+'));
        assert_eq!(app.demo.rcl_size, 10);
    }

    #[test]
    fn test_rcl_size_min_limit() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(1);
        app.handle_key(KeyCode::Char('-'));
        assert_eq!(app.demo.rcl_size, 1);
    }

    #[test]
    fn test_handle_key_cycle_method() {
        let mut app = TspApp::new(10, 42);
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::RandomizedGreedy
        ));

        app.handle_key(KeyCode::Char('m'));
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::NearestNeighbor
        ));

        app.handle_key(KeyCode::Char('m'));
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::Random
        ));

        app.handle_key(KeyCode::Char('m'));
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::RandomizedGreedy
        ));
    }

    #[test]
    fn test_step_increments_frame() {
        let mut app = TspApp::new(10, 42);
        assert_eq!(app.frame_count, 0);
        app.step();
        assert_eq!(app.frame_count, 1);
    }

    #[test]
    fn test_step_tracks_convergence() {
        let mut app = TspApp::new(10, 42);
        let initial_len = app.convergence_history.len();
        app.step();
        assert_eq!(app.convergence_history.len(), initial_len + 1);
    }

    #[test]
    fn test_convergence_history_max_limit() {
        let mut app = TspApp::new(10, 42);
        app.max_history = 5;
        for _ in 0..10 {
            app.step();
        }
        assert!(app.convergence_history.len() <= 5);
    }

    #[test]
    fn test_optimality_gap() {
        let app = TspApp::new(10, 42);
        let gap = app.optimality_gap();
        assert!(gap >= 0.0);
    }

    #[test]
    fn test_construction_method_name() {
        let mut app = TspApp::new(10, 42);
        assert_eq!(app.construction_method_name(), "Randomized Greedy");

        app.demo.set_construction_method(ConstructionMethod::NearestNeighbor);
        assert_eq!(app.construction_method_name(), "Nearest Neighbor");

        app.demo.set_construction_method(ConstructionMethod::Random);
        assert_eq!(app.construction_method_name(), "Random");
    }

    #[test]
    fn test_falsification_status() {
        let app = TspApp::new(10, 42);
        let status = app.falsification_status();
        // Just verify it returns something
        assert!(!status.message.is_empty());
    }

    #[test]
    fn test_verify_equation() {
        let app = TspApp::new(10, 42);
        // Should verify equation (tour length matches computed length)
        let verified = app.verify_equation();
        assert!(verified);
    }

    #[test]
    fn test_unknown_key_ignored() {
        let mut app = TspApp::new(10, 42);
        let paused_before = app.paused;
        let quit_before = app.should_quit;
        app.handle_key(KeyCode::Char('x')); // Unknown key
        assert_eq!(app.paused, paused_before);
        assert_eq!(app.should_quit, quit_before);
    }

    #[test]
    fn test_step_when_paused() {
        let mut app = TspApp::new(10, 42);
        app.paused = true;
        app.auto_run = false;
        let restarts_before = app.demo.restarts;
        app.step();
        // Should still increment frame count
        assert_eq!(app.frame_count, 1);
        // But should not run GRASP iteration
        assert_eq!(app.demo.restarts, restarts_before);
    }

    #[test]
    fn test_initial_convergence_history() {
        let app = TspApp::new(10, 42);
        // Should have initial entry from constructor
        assert!(!app.convergence_history.is_empty());
    }

    // =========================================================================
    // YAML Loading Tests (OR-001-08)
    // =========================================================================

    const BAY_AREA_YAML: &str = include_str!("../../examples/experiments/bay_area_tsp.yaml");

    #[test]
    fn test_from_yaml_bay_area() {
        let app = TspApp::from_yaml(BAY_AREA_YAML).expect("YAML should parse");
        assert_eq!(app.demo.n, 6);
        assert!(app.loaded_instance.is_some());
        assert!(app.loaded_path.is_none()); // Not from file
    }

    #[test]
    fn test_from_yaml_instance_id() {
        let app = TspApp::from_yaml(BAY_AREA_YAML).expect("YAML should parse");
        assert_eq!(app.instance_id(), Some("TSP-BAY-006"));
    }

    #[test]
    fn test_from_yaml_optimal_known() {
        let app = TspApp::from_yaml(BAY_AREA_YAML).expect("YAML should parse");
        assert_eq!(app.optimal_known(), Some(115));
    }

    #[test]
    fn test_from_yaml_no_optimal_known() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Test"
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
  - [10, 0]
"#;
        let app = TspApp::from_yaml(yaml).expect("YAML should parse");
        assert_eq!(app.optimal_known(), None);
    }

    #[test]
    fn test_from_yaml_file_success() {
        let app = TspApp::from_yaml_file("examples/experiments/bay_area_tsp.yaml")
            .expect("File should load");
        assert_eq!(app.demo.n, 6);
        assert!(app.loaded_path.is_some());
        assert!(app.loaded_path.as_ref().unwrap().contains("bay_area"));
    }

    #[test]
    fn test_from_yaml_file_not_found() {
        let result = TspApp::from_yaml_file("/nonexistent/path.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_yaml_invalid() {
        let result = TspApp::from_yaml("invalid yaml: [[[");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_yaml_has_convergence_history() {
        let app = TspApp::from_yaml(BAY_AREA_YAML).expect("YAML should parse");
        assert!(!app.convergence_history.is_empty());
    }

    #[test]
    fn test_from_yaml_can_step() {
        let mut app = TspApp::from_yaml(BAY_AREA_YAML).expect("YAML should parse");
        let initial_restarts = app.demo.restarts;
        app.step();
        assert!(app.demo.restarts > initial_restarts);
    }

    #[test]
    fn test_instance_id_none_when_not_loaded() {
        let app = TspApp::new(10, 42);
        assert!(app.instance_id().is_none());
    }

    #[test]
    fn test_optimal_known_none_when_not_loaded() {
        let app = TspApp::new(10, 42);
        assert!(app.optimal_known().is_none());
    }
}
