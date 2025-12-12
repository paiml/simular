//! TSP GRASP TUI application state and logic.
//!
//! This module contains the testable state and logic for the TSP GRASP TUI demo.
//! Terminal I/O is handled by the binary, but all state management lives here.

use crate::demos::tsp_grasp::{ConstructionMethod, TspGraspDemo};
use crate::demos::EddDemo;
use crossterm::event::KeyCode;

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
        }
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
}
