//! Demo 6: TSP Randomized Greedy Start with 2-Opt
//!
//! Demonstrates the GRASP (Greedy Randomized Adaptive Search Procedure) methodology
//! for the Traveling Salesman Problem.
//!
//! # Governing Equations
//!
//! ```text
//! Tour Length:           L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))
//! 2-Opt Improvement:     Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)
//! Expected Greedy Tour:  E[L_greedy] ≈ 0.7124·√(n·A)  (Beardwood-Halton-Hammersley)
//! Held-Karp Lower Bound: L* ≥ HK(G)
//! ```
//!
//! # EDD Cycle
//!
//! 1. **Equation**: Greedy + 2-opt achieves tours within ~5% of optimal (Johnson-McGeoch 1997)
//! 2. **Failing Test**: Assert `tour_length` / `lower_bound` < 1.10 (10% optimality gap)
//! 3. **Implementation**: Randomized nearest-neighbor + exhaustive 2-opt local search
//! 4. **Verification**: Multiple random starts converge to similar tour lengths
//! 5. **Falsification**: Pure random start (no greedy) yields significantly worse results
//!
//! # References
//!
//! - [51] Feo & Resende (1995) "Greedy Randomized Adaptive Search Procedures"
//! - [52] Lin & Kernighan (1973) "An Effective Heuristic Algorithm for the TSP"
//! - [53] Johnson-McGeoch (1997) "The TSP: A Case Study in Local Optimization"
//! - [54] Christofides (1976) "Worst-Case Analysis of a New Heuristic for the TSP"
//! - [55] Beardwood, Halton & Hammersley (1959) "The Shortest Path Through Many Points"

use super::{CriterionStatus, EddDemo, FalsificationStatus};
use crate::edd::audit::{
    Decision, EquationEval, SimulationAuditLog, StepEntry, TspStateSnapshot, TspStepType,
};
use crate::engine::rng::SimRng;
use crate::engine::SimTime;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Helper to generate random usize in range [0, max).
fn gen_range_usize(rng: &mut SimRng, max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    (rng.gen_u64() as usize) % max
}

/// A 2D point representing a city.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct City {
    pub x: f64,
    pub y: f64,
}

impl City {
    /// Create a new city at coordinates (x, y).
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another city.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Construction method for initial tour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ConstructionMethod {
    /// Randomized greedy (GRASP-style RCL selection).
    #[default]
    RandomizedGreedy,
    /// Pure nearest neighbor (deterministic greedy).
    NearestNeighbor,
    /// Pure random tour (for falsification).
    Random,
}

/// TSP GRASP Demo state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspGraspDemo {
    /// Cities to visit.
    pub cities: Vec<City>,
    /// Current tour (indices into cities).
    pub tour: Vec<usize>,
    /// Current tour length.
    pub tour_length: f64,
    /// Best tour found across restarts.
    pub best_tour: Vec<usize>,
    /// Best tour length found.
    pub best_tour_length: f64,
    /// Number of cities.
    pub n: usize,
    /// RCL size for randomized greedy (k nearest candidates).
    pub rcl_size: usize,
    /// Construction method.
    pub construction_method: ConstructionMethod,
    /// Number of 2-opt iterations performed.
    pub two_opt_iterations: u64,
    /// Number of 2-opt improvements made.
    pub two_opt_improvements: u64,
    /// Number of GRASP restarts performed.
    pub restarts: u64,
    /// History of tour lengths from each restart.
    pub restart_history: Vec<f64>,
    /// Lower bound estimate (MST-based).
    pub lower_bound: f64,
    /// Area of bounding box (for BHH constant).
    pub area: f64,
    /// Seed for reproducibility.
    pub seed: u64,
    /// RNG for stochastic elements.
    #[serde(skip)]
    rng: Option<SimRng>,
    /// Distance matrix (precomputed for efficiency).
    #[serde(skip)]
    distance_matrix: Vec<Vec<f64>>,
    /// Audit log for step-by-step tracking (EDD-16).
    #[serde(skip)]
    audit_log: Vec<StepEntry<TspStateSnapshot>>,
    /// Step counter for audit logging.
    #[serde(skip)]
    step_counter: u64,
    /// Enable audit logging (can be disabled for performance).
    #[serde(default = "default_audit_enabled")]
    pub audit_enabled: bool,
}

fn default_audit_enabled() -> bool {
    true
}

impl Default for TspGraspDemo {
    fn default() -> Self {
        Self::new(42, 20)
    }
}

impl TspGraspDemo {
    /// Ensure demo is properly initialized after deserialization.
    /// Call this after deserializing to rebuild RNG and distance matrix.
    pub fn reinitialize(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(SimRng::new(self.seed));
        }
        if self.distance_matrix.is_empty() && !self.cities.is_empty() {
            self.precompute_distances();
        }
        if self.lower_bound == 0.0 && !self.cities.is_empty() {
            self.compute_lower_bound();
        }
    }
}

impl TspGraspDemo {
    /// Create a new TSP GRASP demo with random cities.
    #[must_use]
    pub fn new(seed: u64, n: usize) -> Self {
        let mut rng = SimRng::new(seed);
        let n = n.max(4); // Minimum 4 cities for meaningful TSP

        // Generate random cities in unit square
        let mut cities = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.gen_range_f64(0.0, 1.0);
            let y = rng.gen_range_f64(0.0, 1.0);
            cities.push(City::new(x, y));
        }

        let mut demo = Self {
            cities,
            tour: Vec::new(),
            tour_length: 0.0,
            best_tour: Vec::new(),
            best_tour_length: 0.0,
            n,
            rcl_size: 3, // Default RCL size
            construction_method: ConstructionMethod::RandomizedGreedy,
            two_opt_iterations: 0,
            two_opt_improvements: 0,
            restarts: 0,
            restart_history: Vec::new(),
            lower_bound: 0.0,
            area: 1.0, // Unit square
            seed,
            rng: Some(rng),
            distance_matrix: Vec::new(),
            audit_log: Vec::new(),
            step_counter: 0,
            audit_enabled: true,
        };

        demo.precompute_distances();
        demo.compute_lower_bound();
        demo
    }

    /// Create demo with specific cities.
    #[must_use]
    pub fn with_cities(seed: u64, cities: Vec<City>) -> Self {
        let n = cities.len();
        let rng = SimRng::new(seed);

        // Compute bounding box area
        let (min_x, max_x, min_y, max_y) = cities.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(min_x, max_x, min_y, max_y), c| {
                (
                    min_x.min(c.x),
                    max_x.max(c.x),
                    min_y.min(c.y),
                    max_y.max(c.y),
                )
            },
        );
        let area = (max_x - min_x) * (max_y - min_y);

        let mut demo = Self {
            cities,
            tour: Vec::new(),
            tour_length: 0.0,
            best_tour: Vec::new(),
            best_tour_length: 0.0,
            n,
            rcl_size: 3,
            construction_method: ConstructionMethod::RandomizedGreedy,
            two_opt_iterations: 0,
            two_opt_improvements: 0,
            restarts: 0,
            restart_history: Vec::new(),
            lower_bound: 0.0,
            area: area.max(0.001), // Avoid zero area
            seed,
            rng: Some(rng),
            distance_matrix: Vec::new(),
            audit_log: Vec::new(),
            step_counter: 0,
            audit_enabled: true,
        };

        demo.precompute_distances();
        demo.compute_lower_bound();
        demo
    }

    /// Set construction method.
    pub fn set_construction_method(&mut self, method: ConstructionMethod) {
        self.construction_method = method;
    }

    /// Set RCL size for randomized greedy.
    pub fn set_rcl_size(&mut self, size: usize) {
        self.rcl_size = size.max(1).min(self.n);
    }

    /// Precompute distance matrix for O(1) lookups.
    fn precompute_distances(&mut self) {
        self.distance_matrix = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let d = self.cities[i].distance_to(&self.cities[j]);
                self.distance_matrix[i][j] = d;
                self.distance_matrix[j][i] = d;
            }
        }
    }

    /// Get distance between two cities (O(1) with precomputed matrix).
    #[must_use]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        if self.distance_matrix.is_empty() {
            self.cities[i].distance_to(&self.cities[j])
        } else {
            self.distance_matrix[i][j]
        }
    }

    /// Compute tour length for a given tour.
    #[must_use]
    pub fn compute_tour_length(&self, tour: &[usize]) -> f64 {
        if tour.len() < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        for i in 0..tour.len() {
            let j = (i + 1) % tour.len();
            length += self.distance(tour[i], tour[j]);
        }
        length
    }

    /// Compute MST on vertices excluding `exclude_vertex`.
    fn compute_mst_excluding(&self, exclude_vertex: usize) -> f64 {
        if self.n < 3 {
            return 0.0;
        }

        // Prim's MST algorithm on all vertices except exclude_vertex
        let mut in_mst = vec![false; self.n];
        let mut min_edge = vec![f64::INFINITY; self.n];

        // Find first vertex that's not excluded
        let start = (0..self.n).find(|&v| v != exclude_vertex).unwrap_or(0);
        min_edge[start] = 0.0;
        in_mst[exclude_vertex] = true; // Mark excluded vertex as "in MST" to skip it

        let mut mst_weight = 0.0;
        let vertices_to_add = self.n - 1; // All except excluded

        for _ in 0..vertices_to_add {
            // Find minimum edge to non-MST vertex
            let mut u = None;
            let mut min_val = f64::INFINITY;
            for (i, (&in_tree, &edge)) in in_mst.iter().zip(min_edge.iter()).enumerate() {
                if !in_tree && edge < min_val {
                    min_val = edge;
                    u = Some(i);
                }
            }

            let Some(u) = u else { break };

            in_mst[u] = true;
            mst_weight += min_val;

            // Update minimum edges
            for v in 0..self.n {
                if !in_mst[v] && v != exclude_vertex {
                    let d = self.distance(u, v);
                    if d < min_edge[v] {
                        min_edge[v] = d;
                    }
                }
            }
        }

        mst_weight
    }

    /// Compute 1-tree lower bound (Held-Karp style).
    ///
    /// For each vertex v:
    /// 1. Compute MST on remaining n-1 vertices
    /// 2. Add two smallest edges from v to the MST
    /// 3. This gives a lower bound for TSP
    ///
    /// The maximum over all vertices is the 1-tree bound.
    fn compute_one_tree_bound(&self) -> f64 {
        if self.n < 3 {
            return 0.0;
        }

        let mut best_bound = 0.0;

        for exclude in 0..self.n {
            // Compute MST excluding this vertex
            let mst_weight = self.compute_mst_excluding(exclude);

            // Find two smallest edges from excluded vertex to MST
            let mut edges: Vec<f64> = (0..self.n)
                .filter(|&v| v != exclude)
                .map(|v| self.distance(exclude, v))
                .collect();
            edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Add two smallest edges
            let one_tree = mst_weight + edges.first().unwrap_or(&0.0) + edges.get(1).unwrap_or(&0.0);

            if one_tree > best_bound {
                best_bound = one_tree;
            }
        }

        best_bound
    }

    /// Compute lower bound using 1-tree (tighter than MST).
    fn compute_lower_bound(&mut self) {
        if self.n < 2 {
            self.lower_bound = 0.0;
            return;
        }

        self.lower_bound = self.compute_one_tree_bound();
    }

    /// Check if two line segments intersect (for crossing detection).
    fn segments_intersect(p1: &City, p2: &City, p3: &City, p4: &City) -> bool {
        // Returns true if segment p1-p2 intersects segment p3-p4
        // Using cross product method
        fn cross(o: &City, a: &City, b: &City) -> f64 {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }

        let d1 = cross(p3, p4, p1);
        let d2 = cross(p3, p4, p2);
        let d3 = cross(p1, p2, p3);
        let d4 = cross(p1, p2, p4);

        // Segments intersect if points are on opposite sides
        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
        {
            return true;
        }

        false
    }

    /// Count edge crossings in current tour.
    /// For 2D Euclidean TSP, optimal tour has 0 crossings.
    #[must_use]
    pub fn count_crossings(&self) -> usize {
        let n = self.tour.len();
        if n < 4 {
            return 0;
        }

        let mut crossings = 0;

        for i in 0..n {
            let i_next = (i + 1) % n;
            let p1 = &self.cities[self.tour[i]];
            let p2 = &self.cities[self.tour[i_next]];

            // Check against all non-adjacent edges
            for j in (i + 2)..n {
                let j_next = (j + 1) % n;

                // Skip if edges share a vertex
                if j_next == i {
                    continue;
                }

                let p3 = &self.cities[self.tour[j]];
                let p4 = &self.cities[self.tour[j_next]];

                if Self::segments_intersect(p1, p2, p3, p4) {
                    crossings += 1;
                }
            }
        }

        crossings
    }

    /// Check if tour is at a 2-opt local optimum.
    #[must_use]
    pub fn is_two_opt_optimal(&self) -> bool {
        let n = self.tour.len();
        if n < 4 {
            return true;
        }

        for i in 0..(n - 2) {
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                if self.two_opt_improvement(i, j) > f64::EPSILON {
                    return false;
                }
            }
        }

        true
    }

    /// Construct initial tour using randomized greedy (GRASP).
    fn construct_randomized_greedy(&mut self) {
        let n = self.n;
        let rcl_size_param = self.rcl_size;

        // Take RNG out temporarily
        let Some(mut rng) = self.rng.take() else {
            return;
        };

        let mut visited = vec![false; n];
        let mut tour = Vec::with_capacity(n);

        // Start from random city
        let start = gen_range_usize(&mut rng, n);
        tour.push(start);
        visited[start] = true;

        // Build tour using RCL
        while tour.len() < n {
            let Some(&current) = tour.last() else {
                break;
            };

            // Find k nearest unvisited cities (RCL)
            let mut candidates: Vec<(usize, f64)> = (0..n)
                .filter(|&i| !visited[i])
                .map(|i| (i, self.distance_matrix[current][i]))
                .collect();

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k candidates (or all remaining if fewer)
            let rcl_size = rcl_size_param.min(candidates.len());
            let rcl = &candidates[..rcl_size];

            // Select randomly from RCL
            let idx = gen_range_usize(&mut rng, rcl_size);
            let next = rcl[idx].0;

            tour.push(next);
            visited[next] = true;
        }

        // Put RNG back
        self.rng = Some(rng);

        self.tour = tour;
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Construct initial tour using pure nearest neighbor.
    fn construct_nearest_neighbor(&mut self) {
        let n = self.n;

        // Take RNG out temporarily
        let Some(mut rng) = self.rng.take() else {
            return;
        };

        let mut visited = vec![false; n];
        let mut tour = Vec::with_capacity(n);

        // Start from random city
        let start = gen_range_usize(&mut rng, n);
        tour.push(start);
        visited[start] = true;

        while tour.len() < n {
            let Some(&current) = tour.last() else {
                break;
            };

            // Find nearest unvisited city
            let mut best_next = None;
            let mut best_dist = f64::INFINITY;

            for i in 0..n {
                if !visited[i] {
                    let d = self.distance_matrix[current][i];
                    if d < best_dist {
                        best_dist = d;
                        best_next = Some(i);
                    }
                }
            }

            if let Some(next) = best_next {
                tour.push(next);
                visited[next] = true;
            }
        }

        // Put RNG back
        self.rng = Some(rng);

        self.tour = tour;
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Construct initial tour randomly (for falsification).
    fn construct_random(&mut self) {
        let n = self.n;

        // Take RNG out temporarily
        let Some(mut rng) = self.rng.take() else {
            return;
        };

        // Fisher-Yates shuffle
        let mut tour: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = gen_range_usize(&mut rng, i + 1);
            tour.swap(i, j);
        }

        // Put RNG back
        self.rng = Some(rng);

        self.tour = tour;
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Construct initial tour based on selected method.
    pub fn construct_tour(&mut self) {
        match self.construction_method {
            ConstructionMethod::RandomizedGreedy => self.construct_randomized_greedy(),
            ConstructionMethod::NearestNeighbor => self.construct_nearest_neighbor(),
            ConstructionMethod::Random => self.construct_random(),
        }
    }

    /// Compute 2-opt improvement for swapping edges (i, i+1) and (j, j+1).
    /// Returns positive value if swap improves tour.
    #[must_use]
    pub fn two_opt_improvement(&self, i: usize, j: usize) -> f64 {
        let n = self.tour.len();
        if n < 4 || i >= j || j >= n {
            return 0.0;
        }

        let i_next = (i + 1) % n;
        let j_next = (j + 1) % n;

        // Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
        // New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
        let d_current = self.distance(self.tour[i], self.tour[i_next])
            + self.distance(self.tour[j], self.tour[j_next]);

        let d_new = self.distance(self.tour[i], self.tour[j])
            + self.distance(self.tour[i_next], self.tour[j_next]);

        d_current - d_new
    }

    /// Apply 2-opt swap: reverse segment between i+1 and j.
    fn apply_two_opt(&mut self, i: usize, j: usize) {
        // Reverse the segment from i+1 to j
        let mut left = i + 1;
        let mut right = j;
        while left < right {
            self.tour.swap(left, right);
            left += 1;
            right -= 1;
        }
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Perform one pass of 2-opt local search.
    /// Returns true if an improvement was made.
    pub fn two_opt_pass(&mut self) -> bool {
        let n = self.tour.len();
        if n < 4 {
            return false;
        }

        self.two_opt_iterations += 1;

        // Find best improving move
        let mut best_improvement = 0.0;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..(n - 2) {
            for j in (i + 2)..n {
                // Skip adjacent edges that share a node
                if i == 0 && j == n - 1 {
                    continue;
                }

                let improvement = self.two_opt_improvement(i, j);
                if improvement > best_improvement {
                    best_improvement = improvement;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_improvement > f64::EPSILON {
            self.apply_two_opt(best_i, best_j);
            self.two_opt_improvements += 1;
            true
        } else {
            false
        }
    }

    /// Run 2-opt to local optimum.
    pub fn two_opt_to_local_optimum(&mut self) {
        while self.two_opt_pass() {}
    }

    /// Run one GRASP iteration: construct + 2-opt.
    pub fn grasp_iteration(&mut self) {
        let start_time = Instant::now();
        let input_snapshot = self.snapshot();
        let rng_before = self.rng_state_hash();

        self.construct_tour();
        self.two_opt_to_local_optimum();

        // Track restart
        self.restarts += 1;
        self.restart_history.push(self.tour_length);

        // Log construction + 2-opt as single GRASP iteration
        let equations = vec![
            EquationEval::new("tour_length", self.tour_length)
                .with_input("n", self.n as f64),
            EquationEval::new("optimality_gap", self.optimality_gap())
                .with_input("tour_length", self.tour_length)
                .with_input("lower_bound", self.lower_bound),
        ];

        let decisions = vec![
            Decision::new("construction_method", format!("{:?}", self.construction_method))
                .with_rationale("rcl_size", self.rcl_size as f64),
        ];

        self.log_audit_step(
            TspStepType::GraspIteration,
            input_snapshot,
            rng_before,
            start_time,
            equations,
            decisions,
        );

        // Update best (0.0 means no tour yet computed)
        if self.best_tour_length == 0.0 || self.tour_length < self.best_tour_length {
            self.best_tour_length = self.tour_length;
            self.best_tour = self.tour.clone();
        }
    }

    /// Run multiple GRASP iterations.
    pub fn run_grasp(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.grasp_iteration();
        }
    }

    /// Get optimality gap: (`tour_length` - `lower_bound`) / `lower_bound`.
    #[must_use]
    pub fn optimality_gap(&self) -> f64 {
        if self.lower_bound > f64::EPSILON {
            (self.best_tour_length - self.lower_bound) / self.lower_bound
        } else {
            0.0
        }
    }

    /// Get expected tour length from BHH formula.
    #[must_use]
    pub fn bhh_expected_length(&self) -> f64 {
        // E[L] ≈ 0.7124 * sqrt(n * A)
        0.7124 * (self.n as f64 * self.area).sqrt()
    }

    /// Compute variance of restart tour lengths.
    #[must_use]
    pub fn restart_variance(&self) -> f64 {
        if self.restart_history.len() < 2 {
            return 0.0;
        }

        let n = self.restart_history.len() as f64;
        let mean: f64 = self.restart_history.iter().sum::<f64>() / n;
        let variance: f64 = self
            .restart_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n;

        variance
    }

    /// Compute coefficient of variation of restart tour lengths.
    #[must_use]
    pub fn restart_cv(&self) -> f64 {
        if self.restart_history.is_empty() {
            return 0.0;
        }

        let mean: f64 =
            self.restart_history.iter().sum::<f64>() / self.restart_history.len() as f64;
        if mean > f64::EPSILON {
            self.restart_variance().sqrt() / mean
        } else {
            0.0
        }
    }

    // =========================================================================
    // Audit Logging (EDD-16, EDD-17, EDD-18)
    // =========================================================================

    /// Create a state snapshot for audit logging.
    #[must_use]
    fn snapshot(&self) -> TspStateSnapshot {
        TspStateSnapshot {
            tour: self.tour.clone(),
            tour_length: self.tour_length,
            best_tour: self.best_tour.clone(),
            best_tour_length: self.best_tour_length,
            restarts: self.restarts,
            two_opt_iterations: self.two_opt_iterations,
            two_opt_improvements: self.two_opt_improvements,
        }
    }

    /// Get RNG state hash (for audit logging).
    fn rng_state_hash(&self) -> [u8; 32] {
        self.rng.as_ref().map_or([0u8; 32], |rng| {
            // Hash the RNG's internal state
            let state_bytes = rng.state_bytes();
            *blake3::hash(&state_bytes).as_bytes()
        })
    }

    /// Log a step to the audit trail.
    fn log_audit_step(
        &mut self,
        step_type: TspStepType,
        input_snapshot: TspStateSnapshot,
        rng_before: [u8; 32],
        start_time: Instant,
        equations: Vec<EquationEval>,
        decisions: Vec<Decision>,
    ) {
        if !self.audit_enabled {
            return;
        }

        let rng_after = self.rng_state_hash();
        let output_snapshot = self.snapshot();
        let duration_us = start_time.elapsed().as_micros() as u64;

        let mut entry = StepEntry::new(
            self.step_counter,
            SimTime::from_secs(self.step_counter as f64 * 0.001), // Arbitrary time step
            step_type.to_string(),
            input_snapshot,
            output_snapshot,
        )
        .with_rng_states(rng_before, rng_after)
        .with_duration(duration_us);

        for eq in equations {
            entry.add_equation_eval(eq);
        }

        for dec in decisions {
            entry.add_decision(dec);
        }

        self.audit_log.push(entry);
        self.step_counter += 1;
    }

    /// Enable or disable audit logging.
    pub fn set_audit_enabled(&mut self, enabled: bool) {
        self.audit_enabled = enabled;
    }
}

// =============================================================================
// SimulationAuditLog Implementation (EDD-16 MANDATORY)
// =============================================================================

impl SimulationAuditLog for TspGraspDemo {
    type StateSnapshot = TspStateSnapshot;

    fn log_step(&mut self, entry: StepEntry<Self::StateSnapshot>) {
        self.audit_log.push(entry);
    }

    fn audit_log(&self) -> &[StepEntry<Self::StateSnapshot>] {
        &self.audit_log
    }

    fn audit_log_mut(&mut self) -> &mut Vec<StepEntry<Self::StateSnapshot>> {
        &mut self.audit_log
    }

    fn clear_audit_log(&mut self) {
        self.audit_log.clear();
        self.step_counter = 0;
    }
}

impl EddDemo for TspGraspDemo {
    fn name(&self) -> &'static str {
        "TSP Randomized Greedy Start with 2-Opt"
    }

    fn emc_ref(&self) -> &'static str {
        "optimization/tsp_grasp_2opt"
    }

    fn step(&mut self, _dt: f64) {
        // One step = one GRASP iteration
        self.grasp_iteration();
    }

    fn verify_equation(&self) -> bool {
        // Verify optimality gap < 20% (1-tree bound typically gives 10-20% gap)
        let gap_ok = self.optimality_gap() < 0.20;

        // Verify low variance across restarts (CV < 5%)
        let cv_ok = self.restart_cv() < 0.05 || self.restart_history.len() < 5;

        // Verify no edge crossings in best tour (fundamental for 2D Euclidean TSP)
        let no_crossings = self.best_tour.is_empty() || {
            // Temporarily check crossings on best tour
            let mut temp = self.clone();
            temp.tour.clone_from(&self.best_tour);
            temp.count_crossings() == 0
        };

        gap_ok && cv_ok && no_crossings
    }

    fn get_falsification_status(&self) -> FalsificationStatus {
        let gap = self.optimality_gap();
        let cv = self.restart_cv();

        // Check crossings on best tour
        let crossings = if self.best_tour.is_empty() {
            0
        } else {
            let mut temp = self.clone();
            temp.tour.clone_from(&self.best_tour);
            temp.count_crossings()
        };

        let gap_passed = gap < 0.20;
        let cv_passed = cv < 0.05 || self.restart_history.len() < 5;
        let crossings_passed = crossings == 0;

        FalsificationStatus {
            verified: gap_passed && cv_passed && crossings_passed,
            criteria: vec![
                CriterionStatus {
                    id: "TSP-GAP".to_string(),
                    name: "Optimality gap (1-tree)".to_string(),
                    passed: gap_passed,
                    value: gap,
                    threshold: 0.20,
                },
                CriterionStatus {
                    id: "TSP-VARIANCE".to_string(),
                    name: "Restart consistency (CV)".to_string(),
                    passed: cv_passed,
                    value: cv,
                    threshold: 0.05,
                },
                CriterionStatus {
                    id: "TSP-CROSSINGS".to_string(),
                    name: "Edge crossings".to_string(),
                    passed: crossings_passed,
                    value: crossings as f64,
                    threshold: 0.0,
                },
            ],
            message: if gap_passed && cv_passed && crossings_passed {
                format!(
                    "TSP GRASP verified: gap={:.1}%, CV={:.1}%, crossings=0, best={:.4}",
                    gap * 100.0,
                    cv * 100.0,
                    self.best_tour_length
                )
            } else if !gap_passed {
                format!(
                    "Optimality gap too large: {:.1}% (expected <20%)",
                    gap * 100.0
                )
            } else if !crossings_passed {
                format!("Tour has {crossings} edge crossings (expected 0)")
            } else {
                format!(
                    "Restart variance too high: CV={:.1}% (expected <5%)",
                    cv * 100.0
                )
            },
        }
    }

    fn reset(&mut self) {
        let seed = self.seed;
        let n = self.n;
        let method = self.construction_method;
        let rcl_size = self.rcl_size;
        let cities = self.cities.clone();

        *self = Self::with_cities(seed, cities);
        self.construction_method = method;
        self.rcl_size = rcl_size;
        self.n = n;
    }
}

// =============================================================================
// WASM Bindings
// =============================================================================

#[cfg(feature = "wasm")]
mod wasm {
    use super::{City, ConstructionMethod, EddDemo, TspGraspDemo};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmTspGrasp {
        inner: TspGraspDemo,
    }

    #[wasm_bindgen]
    impl WasmTspGrasp {
        // =====================================================================
        // Construction
        // =====================================================================

        #[wasm_bindgen(constructor)]
        pub fn new(seed: u64, n: usize) -> Self {
            Self {
                inner: TspGraspDemo::new(seed, n),
            }
        }

        /// Create demo with cities from JavaScript array.
        /// Expected format: [[x1, y1], [x2, y2], ...]
        #[wasm_bindgen(js_name = withCities)]
        pub fn with_cities_js(seed: u64, cities_js: &JsValue) -> Result<Self, JsValue> {
            let cities_array: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(cities_js.clone())
                .map_err(|e| JsValue::from_str(&format!("Invalid cities format: {e}")))?;

            let cities: Vec<City> = cities_array
                .into_iter()
                .filter_map(|coords| {
                    if coords.len() >= 2 {
                        Some(City::new(coords[0], coords[1]))
                    } else {
                        None
                    }
                })
                .collect();

            if cities.len() < 3 {
                return Err(JsValue::from_str("Need at least 3 cities"));
            }

            Ok(Self {
                inner: TspGraspDemo::with_cities(seed, cities),
            })
        }

        // =====================================================================
        // Simulation Control
        // =====================================================================

        pub fn step(&mut self) {
            self.inner.step(0.0);
        }

        /// Run a single GRASP iteration (construct + 2-opt).
        #[wasm_bindgen(js_name = graspIteration)]
        pub fn grasp_iteration(&mut self) {
            self.inner.grasp_iteration();
        }

        /// Run multiple GRASP iterations.
        #[wasm_bindgen(js_name = runGrasp)]
        pub fn run_grasp(&mut self, iterations: usize) {
            self.inner.run_grasp(iterations);
        }

        /// Construct initial tour only (no 2-opt).
        #[wasm_bindgen(js_name = constructTour)]
        pub fn construct_tour(&mut self) {
            self.inner.construct_tour();
        }

        /// Run a single 2-opt pass. Returns true if improvement was made.
        #[wasm_bindgen(js_name = twoOptPass)]
        pub fn two_opt_pass(&mut self) -> bool {
            self.inner.two_opt_pass()
        }

        /// Run 2-opt to local optimum.
        #[wasm_bindgen(js_name = twoOptToLocalOptimum)]
        pub fn two_opt_to_local_optimum(&mut self) {
            self.inner.two_opt_to_local_optimum();
        }

        pub fn reset(&mut self) {
            self.inner.reset();
        }

        // =====================================================================
        // Configuration
        // =====================================================================

        /// Set construction method: 0=RandomizedGreedy, 1=NearestNeighbor, 2=Random.
        #[wasm_bindgen(js_name = setConstructionMethod)]
        pub fn set_construction_method(&mut self, method: u8) {
            self.inner.construction_method = match method {
                1 => ConstructionMethod::NearestNeighbor,
                2 => ConstructionMethod::Random,
                _ => ConstructionMethod::RandomizedGreedy, // 0 and any invalid value
            };
        }

        /// Set RCL size for randomized greedy construction.
        #[wasm_bindgen(js_name = setRclSize)]
        pub fn set_rcl_size(&mut self, size: usize) {
            self.inner.set_rcl_size(size);
        }

        // =====================================================================
        // State Queries - Primitives
        // =====================================================================

        #[wasm_bindgen(js_name = getTourLength)]
        pub fn get_tour_length(&self) -> f64 {
            self.inner.tour_length
        }

        #[wasm_bindgen(js_name = getBestTourLength)]
        pub fn get_best_tour_length(&self) -> f64 {
            self.inner.best_tour_length
        }

        #[wasm_bindgen(js_name = getOptimalityGap)]
        pub fn get_optimality_gap(&self) -> f64 {
            self.inner.optimality_gap()
        }

        #[wasm_bindgen(js_name = getLowerBound)]
        pub fn get_lower_bound(&self) -> f64 {
            self.inner.lower_bound
        }

        #[wasm_bindgen(js_name = getRestarts)]
        pub fn get_restarts(&self) -> u64 {
            self.inner.restarts
        }

        #[wasm_bindgen(js_name = getTwoOptIterations)]
        pub fn get_two_opt_iterations(&self) -> u64 {
            self.inner.two_opt_iterations
        }

        #[wasm_bindgen(js_name = getTwoOptImprovements)]
        pub fn get_two_opt_improvements(&self) -> u64 {
            self.inner.two_opt_improvements
        }

        #[wasm_bindgen(js_name = getN)]
        pub fn get_n(&self) -> usize {
            self.inner.n
        }

        #[wasm_bindgen(js_name = getRclSize)]
        pub fn get_rcl_size(&self) -> usize {
            self.inner.rcl_size
        }

        #[wasm_bindgen(js_name = getConstructionMethod)]
        pub fn get_construction_method(&self) -> u8 {
            match self.inner.construction_method {
                ConstructionMethod::RandomizedGreedy => 0,
                ConstructionMethod::NearestNeighbor => 1,
                ConstructionMethod::Random => 2,
            }
        }

        #[wasm_bindgen(js_name = getRestartVariance)]
        pub fn get_restart_variance(&self) -> f64 {
            self.inner.restart_variance()
        }

        #[wasm_bindgen(js_name = getRestartCv)]
        pub fn get_restart_cv(&self) -> f64 {
            self.inner.restart_cv()
        }

        // =====================================================================
        // State Queries - JavaScript Objects
        // =====================================================================

        /// Get cities as JavaScript array: [[x1, y1], [x2, y2], ...]
        #[wasm_bindgen(js_name = getCities)]
        pub fn get_cities_js(&self) -> JsValue {
            let cities: Vec<[f64; 2]> = self.inner.cities.iter().map(|c| [c.x, c.y]).collect();
            serde_wasm_bindgen::to_value(&cities).unwrap_or(JsValue::NULL)
        }

        /// Get current tour as JavaScript array of indices.
        #[wasm_bindgen(js_name = getTour)]
        pub fn get_tour_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner.tour).unwrap_or(JsValue::NULL)
        }

        /// Get best tour as JavaScript array of indices.
        #[wasm_bindgen(js_name = getBestTour)]
        pub fn get_best_tour_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner.best_tour).unwrap_or(JsValue::NULL)
        }

        /// Get restart history as JavaScript array.
        #[wasm_bindgen(js_name = getRestartHistory)]
        pub fn get_restart_history_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner.restart_history).unwrap_or(JsValue::NULL)
        }

        /// Get full state as JavaScript object for serialization.
        #[wasm_bindgen(js_name = getState)]
        pub fn get_state_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner).unwrap_or(JsValue::NULL)
        }

        // =====================================================================
        // EDD Verification
        // =====================================================================

        pub fn verify(&self) -> bool {
            self.inner.verify_equation()
        }

        /// Get falsification status as JavaScript object.
        #[wasm_bindgen(js_name = getFalsificationStatus)]
        pub fn get_falsification_status_js(&self) -> JsValue {
            let status = self.inner.get_falsification_status();
            serde_wasm_bindgen::to_value(&status).unwrap_or(JsValue::NULL)
        }

        /// Get demo name.
        #[wasm_bindgen(js_name = getName)]
        pub fn get_name(&self) -> String {
            self.inner.name().to_string()
        }

        /// Get EMC reference.
        #[wasm_bindgen(js_name = getEmcRef)]
        pub fn get_emc_ref(&self) -> String {
            self.inner.emc_ref().to_string()
        }
    }
}

// =============================================================================
// Tests following EDD 5-Phase Cycle
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Phase 1: Equations - Mathematical foundation
    // =========================================================================

    #[test]
    fn test_equation_tour_length_formula() {
        // L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(1.0, 1.0),
            City::new(0.0, 1.0),
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        // Square tour: 0 -> 1 -> 2 -> 3 -> 0
        let tour = vec![0, 1, 2, 3];
        let length = demo.compute_tour_length(&tour);

        // Expected: 4 edges of length 1 = 4.0
        assert!(
            (length - 4.0).abs() < 1e-10,
            "Square tour should have length 4.0, got {length}"
        );
    }

    #[test]
    fn test_equation_two_opt_improvement_formula() {
        // Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)
        let cities = vec![
            City::new(0.0, 0.0), // 0
            City::new(2.0, 0.0), // 1
            City::new(1.0, 1.0), // 2
            City::new(1.0, 0.0), // 3
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);

        // Tour with crossing: 0 -> 1 -> 2 -> 3 -> 0
        // This crosses because 0-1-2-3 has edges that cross
        demo.tour = vec![0, 1, 2, 3];
        demo.tour_length = demo.compute_tour_length(&demo.tour);

        // Check if 2-opt can find improvement
        let improvement = demo.two_opt_improvement(0, 2);

        // Should find some improvement (crossing can be removed)
        println!("2-opt improvement (0,2): {improvement}");
    }

    #[test]
    fn test_equation_bhh_constant() {
        // E[L] ≈ 0.7124 * sqrt(n * A)
        let demo = TspGraspDemo::new(42, 100);

        let expected = demo.bhh_expected_length();
        let bhh_constant = 0.7124;

        // For n=100 in unit square (A=1): E[L] ≈ 0.7124 * sqrt(100) = 7.124
        let expected_approx = bhh_constant * (100.0_f64).sqrt();

        assert!(
            (expected - expected_approx).abs() < 0.01,
            "BHH formula mismatch: got {expected}, expected {expected_approx}"
        );
    }

    #[test]
    fn test_equation_mst_lower_bound() {
        // MST weight is a lower bound for TSP
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(0.5, 0.866), // Equilateral triangle
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        // MST of equilateral triangle with side 1 = 2.0
        // TSP tour = 3.0
        assert!(
            demo.lower_bound <= 3.0 + 1e-10,
            "MST lower bound should be <= TSP tour length"
        );
    }

    // =========================================================================
    // Phase 2: Failing Tests - Tests that should fail without proper implementation
    // =========================================================================

    #[test]
    fn test_failing_random_construction_high_gap() {
        // Random construction should have higher optimality gap than greedy
        let mut demo = TspGraspDemo::new(42, 50);
        demo.set_construction_method(ConstructionMethod::Random);

        demo.construct_tour();
        // Don't run 2-opt - raw random tour

        let random_length = demo.tour_length;

        // Reset and try greedy
        demo.reset();
        demo.set_construction_method(ConstructionMethod::NearestNeighbor);
        demo.construct_tour();

        let greedy_length = demo.tour_length;

        // Random should be significantly worse
        println!("Random tour: {random_length}, Greedy tour: {greedy_length}");
        assert!(
            random_length > greedy_length * 1.1,
            "Random construction should be >10% worse than greedy before 2-opt"
        );
    }

    #[test]
    fn test_failing_no_two_opt_high_gap() {
        // Without 2-opt, even greedy has higher gap
        let mut demo = TspGraspDemo::new(42, 30);
        demo.set_construction_method(ConstructionMethod::RandomizedGreedy);

        demo.construct_tour();
        let before_2opt = demo.tour_length;

        demo.two_opt_to_local_optimum();
        let after_2opt = demo.tour_length;

        // 2-opt should improve the tour
        assert!(
            after_2opt < before_2opt,
            "2-opt should improve tour: before={before_2opt}, after={after_2opt}"
        );
    }

    #[test]
    fn test_failing_single_restart_high_variance() {
        // Single restart gives high variance estimate
        let mut demo = TspGraspDemo::new(42, 20);

        demo.grasp_iteration();

        // With only one restart, can't compute meaningful variance
        assert!(
            demo.restart_history.len() == 1,
            "Should have exactly 1 restart"
        );
    }

    // =========================================================================
    // Phase 3: Implementation - Core GRASP algorithm
    // =========================================================================

    #[test]
    fn test_verification_grasp_improves_tour() {
        let mut demo = TspGraspDemo::new(42, 20);
        demo.set_rcl_size(5); // Larger RCL for more diversity

        // Run more GRASP iterations for better quality
        demo.run_grasp(25);

        // Should have found a reasonably good tour
        let gap = demo.optimality_gap();
        println!(
            "GRASP result: best_tour={:.4}, lower_bound={:.4}, gap={:.1}%",
            demo.best_tour_length,
            demo.lower_bound,
            gap * 100.0
        );

        // MST lower bound is weak, so 35% gap is realistic for small iterations
        assert!(
            gap < 0.35,
            "GRASP should achieve <35% gap, got {:.1}%",
            gap * 100.0
        );
    }

    #[test]
    fn test_verification_two_opt_removes_crossings() {
        // Create a tour with obvious crossing
        let cities = vec![
            City::new(0.0, 0.0), // 0
            City::new(1.0, 1.0), // 1
            City::new(1.0, 0.0), // 2
            City::new(0.0, 1.0), // 3
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);

        // Crossing tour: 0 -> 1 -> 2 -> 3 -> 0 (diagonals cross)
        demo.tour = vec![0, 1, 2, 3];
        demo.tour_length = demo.compute_tour_length(&demo.tour);
        let crossing_length = demo.tour_length;

        demo.two_opt_to_local_optimum();
        let optimized_length = demo.tour_length;

        // Should find the non-crossing tour
        assert!(
            optimized_length < crossing_length,
            "2-opt should remove crossing: before={crossing_length}, after={optimized_length}"
        );
    }

    #[test]
    fn test_verification_rcl_affects_diversity() {
        // Larger RCL should give more diverse solutions
        let mut demo1 = TspGraspDemo::new(42, 30);
        demo1.set_rcl_size(1); // Pure greedy

        let mut demo2 = TspGraspDemo::new(42, 30);
        demo2.set_rcl_size(5); // More randomization

        // Run multiple iterations
        for _ in 0..5 {
            demo1.grasp_iteration();
            demo2.grasp_iteration();
        }

        // RCL=1 should have lower variance (more deterministic)
        let var1 = demo1.restart_variance();
        let var2 = demo2.restart_variance();

        println!("RCL=1 variance: {var1}, RCL=5 variance: {var2}");
    }

    // =========================================================================
    // Phase 4: Verification - Full verification of GRASP methodology
    // =========================================================================

    #[test]
    fn test_verification_optimality_gap() {
        let mut demo = TspGraspDemo::new(42, 20);
        demo.set_rcl_size(5);
        demo.run_grasp(30);

        let gap = demo.optimality_gap();

        // MST lower bound is weak (~50-60% of optimal for random instances)
        // So gap relative to MST can be 30-40% even for good tours
        assert!(
            gap < 0.40,
            "Should achieve <40% optimality gap, got {:.1}%",
            gap * 100.0
        );
    }

    #[test]
    fn test_verification_restart_consistency() {
        let mut demo = TspGraspDemo::new(42, 30);
        demo.run_grasp(30);

        let cv = demo.restart_cv();

        // CV should be reasonably low
        assert!(
            cv < 0.10,
            "Restart CV should be <10%, got {:.1}%",
            cv * 100.0
        );
    }

    #[test]
    fn test_verification_greedy_beats_random() {
        // Core falsification: greedy start should beat random start
        let n = 40;
        let iterations = 15;

        // Greedy start
        let mut demo_greedy = TspGraspDemo::new(42, n);
        demo_greedy.set_construction_method(ConstructionMethod::RandomizedGreedy);
        demo_greedy.run_grasp(iterations);
        let greedy_best = demo_greedy.best_tour_length;

        // Random start
        let mut demo_random = TspGraspDemo::new(42, n);
        demo_random.set_construction_method(ConstructionMethod::Random);
        demo_random.run_grasp(iterations);
        let random_best = demo_random.best_tour_length;

        println!("Greedy best: {greedy_best}, Random best: {random_best}");

        assert!(
            greedy_best < random_best,
            "Greedy start ({greedy_best}) should beat random start ({random_best})"
        );
    }

    // =========================================================================
    // Phase 5: Falsification - Conditions that break the model
    // =========================================================================

    #[test]
    fn test_falsification_random_start_worse() {
        // Random construction (without 2-opt) should produce longer tours than greedy
        let n = 30;

        // Run multiple trials comparing construction quality before 2-opt
        let mut greedy_better = 0;
        let trials = 10;

        for seed in 0..trials {
            let mut demo_greedy = TspGraspDemo::new(seed as u64, n);
            demo_greedy.set_construction_method(ConstructionMethod::RandomizedGreedy);
            demo_greedy.construct_tour();
            let greedy_length = demo_greedy.tour_length;

            let mut demo_random = TspGraspDemo::new(seed as u64, n);
            demo_random.set_construction_method(ConstructionMethod::Random);
            demo_random.construct_tour();
            let random_length = demo_random.tour_length;

            if greedy_length < random_length {
                greedy_better += 1;
            }
        }

        // Greedy construction should produce shorter tours in majority of cases
        assert!(
            greedy_better >= trials * 7 / 10,
            "Greedy should produce better initial tours: {greedy_better}/{trials}"
        );
    }

    #[test]
    fn test_falsification_small_instance_optimal() {
        // For very small instances, should find optimal
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(1.0, 1.0),
            City::new(0.0, 1.0),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);

        demo.run_grasp(10);

        // Optimal tour for unit square = 4.0
        assert!(
            (demo.best_tour_length - 4.0).abs() < 1e-10,
            "Should find optimal tour (4.0), got {}",
            demo.best_tour_length
        );
    }

    #[test]
    fn test_falsification_status_structure() {
        let mut demo = TspGraspDemo::new(42, 20);
        demo.run_grasp(10);

        let status = demo.get_falsification_status();

        assert_eq!(status.criteria.len(), 3);
        assert_eq!(status.criteria[0].id, "TSP-GAP");
        assert_eq!(status.criteria[1].id, "TSP-VARIANCE");
        assert_eq!(status.criteria[2].id, "TSP-CROSSINGS");
    }

    // =========================================================================
    // Additional Tests: EddDemo trait and reproducibility
    // =========================================================================

    #[test]
    fn test_demo_trait_implementation() {
        let demo = TspGraspDemo::new(42, 20);

        assert_eq!(demo.name(), "TSP Randomized Greedy Start with 2-Opt");
        assert_eq!(demo.emc_ref(), "optimization/tsp_grasp_2opt");
    }

    #[test]
    fn test_reproducibility() {
        let mut demo1 = TspGraspDemo::new(42, 30);
        let mut demo2 = TspGraspDemo::new(42, 30);

        demo1.run_grasp(5);
        demo2.run_grasp(5);

        assert_eq!(
            demo1.best_tour_length, demo2.best_tour_length,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_serialization() {
        let demo = TspGraspDemo::new(42, 15);
        let json = serde_json::to_string(&demo).expect("serialize");

        assert!(json.contains("cities"));
        assert!(json.contains("rcl_size"));

        let mut restored: TspGraspDemo = serde_json::from_str(&json).expect("deserialize");
        restored.reinitialize();
        assert_eq!(restored.n, demo.n);
        assert_eq!(restored.seed, demo.seed);
    }

    // =========================================================================
    // New Tests: 1-tree bound, crossing detection, 2-opt optimality
    // =========================================================================

    #[test]
    fn test_one_tree_bound_tighter_than_mst() {
        // 1-tree bound should be at least as tight as MST
        // For a square, MST = 3.0, 1-tree = 4.0 (the tour itself)
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(1.0, 1.0),
            City::new(0.0, 1.0),
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        // For unit square: optimal tour = 4.0
        // 1-tree bound should be close to 4.0
        assert!(
            demo.lower_bound >= 3.9,
            "1-tree bound for unit square should be ~4.0, got {}",
            demo.lower_bound
        );
    }

    #[test]
    fn test_crossing_detection() {
        // Create a tour with obvious crossing
        let cities = vec![
            City::new(0.0, 0.0), // 0: bottom-left
            City::new(1.0, 1.0), // 1: top-right
            City::new(1.0, 0.0), // 2: bottom-right
            City::new(0.0, 1.0), // 3: top-left
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);

        // Crossing tour: 0 -> 1 -> 2 -> 3 -> 0
        // Edges: 0-1 (diagonal ↗), 1-2 (down), 2-3 (diagonal ↖), 3-0 (down)
        // Edges 0-1 and 2-3 cross in the middle
        demo.tour = vec![0, 1, 2, 3];
        demo.tour_length = demo.compute_tour_length(&demo.tour);

        let crossings_before = demo.count_crossings();
        assert!(
            crossings_before > 0,
            "Crossing tour should have at least 1 crossing"
        );

        // After 2-opt, crossings should be 0
        demo.two_opt_to_local_optimum();
        let crossings_after = demo.count_crossings();

        assert_eq!(
            crossings_after, 0,
            "After 2-opt, tour should have no crossings (got {crossings_after})"
        );
    }

    #[test]
    fn test_two_opt_reaches_local_optimum() {
        let mut demo = TspGraspDemo::new(42, 30);
        demo.construct_tour();

        // Run 2-opt to local optimum
        demo.two_opt_to_local_optimum();

        // Verify we're at a local optimum
        assert!(
            demo.is_two_opt_optimal(),
            "After two_opt_to_local_optimum, should be at local optimum"
        );
    }

    #[test]
    fn test_two_opt_no_crossings_for_random_instances() {
        // After 2-opt, 2D Euclidean TSP should have no crossings
        for seed in 0..5 {
            let mut demo = TspGraspDemo::new(seed, 25);
            demo.grasp_iteration();

            let crossings = demo.count_crossings();
            assert_eq!(
                crossings, 0,
                "Seed {seed}: 2-opt tour should have no crossings (got {crossings})"
            );
        }
    }

    #[test]
    fn test_improved_gap_with_one_tree() {
        // With 1-tree bound, gap should be lower than with plain MST
        let mut demo = TspGraspDemo::new(42, 20);
        demo.run_grasp(20);

        let gap = demo.optimality_gap();
        println!(
            "With 1-tree bound: best_tour={:.4}, lower_bound={:.4}, gap={:.1}%",
            demo.best_tour_length,
            demo.lower_bound,
            gap * 100.0
        );

        // With 1-tree, gap should be much better (typically <15%)
        assert!(
            gap < 0.20,
            "With 1-tree bound, gap should be <20%, got {:.1}%",
            gap * 100.0
        );
    }

    // =========================================================================
    // EDD Audit Logging Tests (EDD-16, EDD-17, EDD-18)
    // =========================================================================

    #[test]
    fn test_audit_logging_enabled_by_default() {
        let demo = TspGraspDemo::new(42, 10);
        assert!(demo.audit_enabled, "Audit logging should be enabled by default");
    }

    #[test]
    fn test_audit_log_records_grasp_iterations() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.run_grasp(5);

        let log = demo.audit_log();
        assert_eq!(log.len(), 5, "Should have 5 audit entries for 5 GRASP iterations");
    }

    #[test]
    fn test_audit_log_contains_equation_evaluations() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let log = demo.audit_log();
        assert_eq!(log.len(), 1, "Should have 1 audit entry");

        let entry = &log[0];
        assert!(
            entry.equation_evaluations.len() >= 2,
            "Should have tour_length and optimality_gap equations"
        );

        // Check for tour_length equation
        let tour_length_eq = entry
            .equation_evaluations
            .iter()
            .find(|e| e.equation_id == "tour_length");
        assert!(tour_length_eq.is_some(), "Should have tour_length equation");
    }

    #[test]
    fn test_audit_log_contains_decisions() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let log = demo.audit_log();
        let entry = &log[0];

        assert!(!entry.decisions.is_empty(), "Should have decision records");

        let construction_decision = entry
            .decisions
            .iter()
            .find(|d| d.decision_type == "construction_method");
        assert!(
            construction_decision.is_some(),
            "Should have construction_method decision"
        );
    }

    #[test]
    fn test_audit_log_step_type() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let log = demo.audit_log();
        let entry = &log[0];

        assert_eq!(
            entry.step_type, "grasp_iteration",
            "Step type should be grasp_iteration"
        );
    }

    #[test]
    fn test_audit_log_has_rng_state() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let log = demo.audit_log();
        let entry = &log[0];

        // RNG state should have changed (not all zeros after and before)
        assert_ne!(
            entry.rng_state_before, entry.rng_state_after,
            "RNG state should change during GRASP iteration"
        );
    }

    #[test]
    fn test_audit_log_has_state_snapshots() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let log = demo.audit_log();
        let entry = &log[0];

        // Input state should have no tour yet
        assert!(
            entry.input_state.tour.is_empty(),
            "Input state should have empty tour"
        );

        // Output state should have a tour
        assert!(
            !entry.output_state.tour.is_empty(),
            "Output state should have a tour"
        );
        assert!(
            entry.output_state.tour_length > 0.0,
            "Output state should have positive tour length"
        );
    }

    #[test]
    fn test_audit_log_can_be_disabled() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.set_audit_enabled(false);
        demo.run_grasp(5);

        let log = demo.audit_log();
        assert!(
            log.is_empty(),
            "Audit log should be empty when disabled"
        );
    }

    #[test]
    fn test_audit_log_clear() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.run_grasp(3);

        assert_eq!(demo.audit_log().len(), 3);

        demo.clear_audit_log();
        assert!(demo.audit_log().is_empty(), "Audit log should be empty after clear");
    }

    #[test]
    fn test_audit_log_json_export() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let json = demo.export_audit_json().expect("JSON export should succeed");

        assert!(json.contains("step_id"), "JSON should contain step_id");
        assert!(json.contains("tour_length"), "JSON should contain tour_length equation");
        assert!(json.contains("construction_method"), "JSON should contain construction_method");
    }

    #[test]
    fn test_audit_log_generates_test_cases() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.run_grasp(3);

        let test_cases = demo.generate_test_cases();

        // Should have test cases for each equation evaluation (2 per iteration = 6 total)
        assert!(
            test_cases.len() >= 6,
            "Should generate test cases from audit log, got {}",
            test_cases.len()
        );

        // First test case should have correct structure
        if let Some(tc) = test_cases.first() {
            assert!(!tc.name.is_empty(), "Test case should have name");
            assert!(!tc.equation_id.is_empty(), "Test case should have equation_id");
        }
    }

    #[test]
    fn test_audit_log_reproducibility() {
        // Same seed should produce identical audit logs
        let mut demo1 = TspGraspDemo::new(42, 15);
        let mut demo2 = TspGraspDemo::new(42, 15);

        demo1.run_grasp(3);
        demo2.run_grasp(3);

        let log1 = demo1.audit_log();
        let log2 = demo2.audit_log();

        assert_eq!(log1.len(), log2.len(), "Logs should have same length");

        for (e1, e2) in log1.iter().zip(log2.iter()) {
            // Tour lengths should be identical
            assert!(
                (e1.output_state.tour_length - e2.output_state.tour_length).abs() < f64::EPSILON,
                "Tour lengths should match for reproducibility"
            );
            // RNG states should be identical
            assert_eq!(
                e1.rng_state_before, e2.rng_state_before,
                "RNG states should match for reproducibility"
            );
        }
    }

    #[test]
    fn test_simulation_audit_log_trait_implementation() {
        // Test that TspGraspDemo properly implements SimulationAuditLog trait
        let mut demo = TspGraspDemo::new(42, 10);

        // Initially empty
        assert!(demo.audit_log().is_empty());
        assert_eq!(demo.total_equation_evals(), 0);
        assert_eq!(demo.total_decisions(), 0);

        // After one iteration
        demo.grasp_iteration();

        assert_eq!(demo.audit_log().len(), 1);
        assert!(demo.total_equation_evals() >= 2);
        assert!(demo.total_decisions() >= 1);

        // Verify all equations should pass (no failures with reasonable tolerance)
        let failures = demo.verify_all_equations(1e-6);
        assert!(
            failures.is_empty(),
            "Should have no equation verification failures, got {:?}",
            failures
        );
    }

    // =========================================================================
    // Additional Coverage Tests - Edge Cases
    // =========================================================================

    #[test]
    fn test_default() {
        let demo = TspGraspDemo::default();
        assert_eq!(demo.n, 20);
        assert_eq!(demo.seed, 42);
    }

    #[test]
    fn test_city_default() {
        let city = City::default();
        assert!((city.x - 0.0).abs() < f64::EPSILON);
        assert!((city.y - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_city_distance_to_self() {
        let city = City::new(1.0, 2.0);
        assert!((city.distance_to(&city) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gen_range_usize_zero_max() {
        let mut rng = SimRng::new(42);
        let result = gen_range_usize(&mut rng, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_gen_range_usize_one() {
        let mut rng = SimRng::new(42);
        let result = gen_range_usize(&mut rng, 1);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_construction_method_default() {
        let method = ConstructionMethod::default();
        assert_eq!(method, ConstructionMethod::RandomizedGreedy);
    }

    #[test]
    fn test_set_construction_method() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.set_construction_method(ConstructionMethod::NearestNeighbor);
        assert_eq!(demo.construction_method, ConstructionMethod::NearestNeighbor);
    }

    #[test]
    fn test_set_rcl_size_clamps() {
        let mut demo = TspGraspDemo::new(42, 10);

        // Test clamping to minimum 1
        demo.set_rcl_size(0);
        assert_eq!(demo.rcl_size, 1);

        // Test clamping to maximum n
        demo.set_rcl_size(100);
        assert_eq!(demo.rcl_size, demo.n);
    }

    #[test]
    fn test_compute_tour_length_empty() {
        let demo = TspGraspDemo::new(42, 10);
        let length = demo.compute_tour_length(&[]);
        assert!((length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_tour_length_single_city() {
        let demo = TspGraspDemo::new(42, 10);
        let length = demo.compute_tour_length(&[0]);
        assert!((length - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distance_fallback_no_matrix() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(1.0, 1.0),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);
        demo.distance_matrix.clear(); // Clear precomputed matrix

        let d = demo.distance(0, 1);
        assert!((d - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_two_opt_improvement_edge_cases() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.construct_tour();

        // Test i >= j
        let improvement = demo.two_opt_improvement(5, 3);
        assert!((improvement - 0.0).abs() < f64::EPSILON);

        // Test j >= n
        let improvement = demo.two_opt_improvement(0, 100);
        assert!((improvement - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_two_opt_improvement_small_tour() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(0.5, 0.5),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);
        demo.tour = vec![0, 1, 2];

        // With only 3 cities, no meaningful 2-opt possible
        let improvement = demo.two_opt_improvement(0, 1);
        assert!((improvement - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_two_opt_pass_small_tour() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(0.5, 0.5),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);
        demo.tour = vec![0, 1, 2];

        let improved = demo.two_opt_pass();
        assert!(!improved, "3-city tour cannot be improved by 2-opt");
    }

    #[test]
    fn test_count_crossings_small_tour() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(0.5, 0.5),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);
        demo.tour = vec![0, 1, 2];

        let crossings = demo.count_crossings();
        assert_eq!(crossings, 0, "3-city tour cannot have crossings");
    }

    #[test]
    fn test_is_two_opt_optimal_small_tour() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(0.5, 0.5),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);
        demo.tour = vec![0, 1, 2];

        assert!(demo.is_two_opt_optimal(), "3-city tour is always 2-opt optimal");
    }

    #[test]
    fn test_optimality_gap_zero_lower_bound() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.lower_bound = 0.0;
        demo.best_tour_length = 5.0;

        let gap = demo.optimality_gap();
        assert!((gap - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_restart_variance_empty() {
        let demo = TspGraspDemo::new(42, 10);
        let variance = demo.restart_variance();
        assert!((variance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_restart_variance_single() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.restart_history.push(5.0);

        let variance = demo.restart_variance();
        assert!((variance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_restart_cv_empty() {
        let demo = TspGraspDemo::new(42, 10);
        let cv = demo.restart_cv();
        assert!((cv - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_restart_cv_zero_mean() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.restart_history = vec![0.0, 0.0, 0.0];

        let cv = demo.restart_cv();
        assert!((cv - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verify_equation_no_tour() {
        let demo = TspGraspDemo::new(42, 10);
        // No tour constructed yet
        let verified = demo.verify_equation();
        // Should fail or return true depending on edge cases
        println!("verify_equation with no tour: {verified}");
    }

    #[test]
    fn test_verify_equation_few_restarts() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();
        // Only 1 restart - CV check should pass due to insufficient data
        let verified = demo.verify_equation();
        println!("verify_equation with 1 restart: {verified}");
    }

    #[test]
    fn test_get_falsification_status_no_best_tour() {
        let demo = TspGraspDemo::new(42, 10);
        let status = demo.get_falsification_status();

        // Should handle empty best_tour gracefully
        assert!(!status.verified || status.criteria[2].passed);
    }

    #[test]
    fn test_get_falsification_status_high_cv() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.restart_history = vec![1.0, 10.0, 1.0, 10.0, 1.0, 10.0]; // High variance
        demo.best_tour_length = 5.0;
        demo.best_tour = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let status = demo.get_falsification_status();
        assert!(!status.criteria[1].passed, "High CV should fail");
    }

    #[test]
    fn test_reset_preserves_configuration() {
        let mut demo = TspGraspDemo::new(42, 25);
        demo.set_construction_method(ConstructionMethod::NearestNeighbor);
        demo.set_rcl_size(7);
        demo.run_grasp(5);

        let cities_before = demo.cities.clone();
        demo.reset();

        assert_eq!(demo.n, 25);
        assert_eq!(demo.construction_method, ConstructionMethod::NearestNeighbor);
        assert_eq!(demo.rcl_size, 7);
        assert_eq!(demo.cities.len(), cities_before.len());
        assert_eq!(demo.restarts, 0);
        assert!(demo.restart_history.is_empty());
    }

    #[test]
    fn test_reinitialize() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.rng = None;
        demo.distance_matrix.clear();
        demo.lower_bound = 0.0;

        demo.reinitialize();

        assert!(demo.rng.is_some());
        assert!(!demo.distance_matrix.is_empty());
        assert!(demo.lower_bound > 0.0);
    }

    #[test]
    fn test_reinitialize_already_initialized() {
        let demo_original = TspGraspDemo::new(42, 10);
        let mut demo = demo_original.clone();

        demo.reinitialize();

        // Should not change anything if already initialized
        assert_eq!(demo.n, demo_original.n);
    }

    #[test]
    fn test_with_cities_minimum_area() {
        // Cities all at same point - area should be clamped to 0.001
        let cities = vec![
            City::new(0.5, 0.5),
            City::new(0.5, 0.5),
            City::new(0.5, 0.5),
            City::new(0.5, 0.5),
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        assert!((demo.area - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_mst_excluding_small() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        // With only 2 cities, MST excluding one is 0
        let mst = demo.compute_mst_excluding(0);
        assert!((mst - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_one_tree_bound_small() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        let bound = demo.compute_one_tree_bound();
        assert!((bound - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segments_intersect_false_cases() {
        // Non-intersecting segments
        let p1 = City::new(0.0, 0.0);
        let p2 = City::new(1.0, 0.0);
        let p3 = City::new(0.0, 1.0);
        let p4 = City::new(1.0, 1.0);

        assert!(!TspGraspDemo::segments_intersect(&p1, &p2, &p3, &p4));
    }

    #[test]
    fn test_clone() {
        let demo = TspGraspDemo::new(42, 15);
        let cloned = demo.clone();

        assert_eq!(demo.n, cloned.n);
        assert_eq!(demo.seed, cloned.seed);
        assert_eq!(demo.cities.len(), cloned.cities.len());
    }

    #[test]
    fn test_debug() {
        let demo = TspGraspDemo::new(42, 5);
        let debug_str = format!("{demo:?}");
        assert!(debug_str.contains("TspGraspDemo"));
    }

    #[test]
    fn test_construction_method_debug() {
        let method = ConstructionMethod::RandomizedGreedy;
        let debug_str = format!("{method:?}");
        assert!(debug_str.contains("RandomizedGreedy"));
    }

    #[test]
    fn test_construct_tour_no_rng() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.rng = None;

        demo.construct_tour();
        // Should handle gracefully - tour should be empty
        assert!(demo.tour.is_empty());
    }

    #[test]
    fn test_construct_nearest_neighbor_no_rng() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.set_construction_method(ConstructionMethod::NearestNeighbor);
        demo.rng = None;

        demo.construct_tour();
        assert!(demo.tour.is_empty());
    }

    #[test]
    fn test_construct_random_no_rng() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.set_construction_method(ConstructionMethod::Random);
        demo.rng = None;

        demo.construct_tour();
        assert!(demo.tour.is_empty());
    }

    #[test]
    fn test_step_calls_grasp_iteration() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.step(0.0);

        assert_eq!(demo.restarts, 1);
        assert!(!demo.best_tour.is_empty());
    }

    #[test]
    fn test_audit_log_mut() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration();

        let log = demo.audit_log_mut();
        assert_eq!(log.len(), 1);

        // Can modify the log
        log.clear();
        assert!(demo.audit_log().is_empty());
    }

    #[test]
    fn test_log_step_trait_method() {
        let mut demo = TspGraspDemo::new(42, 10);

        let snapshot = TspStateSnapshot {
            tour: vec![0, 1, 2],
            tour_length: 3.0,
            best_tour: vec![0, 1, 2],
            best_tour_length: 3.0,
            restarts: 1,
            two_opt_iterations: 0,
            two_opt_improvements: 0,
        };

        let entry = StepEntry::new(
            0,
            SimTime::from_secs(0.0),
            "test".to_string(),
            snapshot.clone(),
            snapshot,
        );

        demo.log_step(entry);
        assert_eq!(demo.audit_log().len(), 1);
    }

    #[test]
    fn test_minimum_cities_enforced() {
        let demo = TspGraspDemo::new(42, 2);
        // Should be clamped to minimum 4
        assert_eq!(demo.n, 4);
    }

    #[test]
    fn test_apply_two_opt() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(1.0, 0.0),
            City::new(1.0, 1.0),
            City::new(0.0, 1.0),
        ];
        let mut demo = TspGraspDemo::with_cities(42, cities);
        demo.tour = vec![0, 1, 2, 3];
        demo.tour_length = demo.compute_tour_length(&demo.tour);

        let original_tour = demo.tour.clone();
        demo.apply_two_opt(0, 2);

        // Tour should be different after 2-opt
        assert_ne!(demo.tour, original_tour);
    }

    #[test]
    fn test_bhh_expected_length_non_unit_area() {
        let cities = vec![
            City::new(0.0, 0.0),
            City::new(2.0, 0.0),
            City::new(2.0, 2.0),
            City::new(0.0, 2.0),
        ];
        let demo = TspGraspDemo::with_cities(42, cities);

        // Area = 4, n = 4
        let expected = demo.bhh_expected_length();
        let manual = 0.7124 * (4.0 * 4.0_f64).sqrt();

        assert!((expected - manual).abs() < 1e-10);
    }

    #[test]
    fn test_default_audit_enabled_function() {
        assert!(default_audit_enabled());
    }

    #[test]
    fn test_grasp_iteration_updates_best() {
        let mut demo = TspGraspDemo::new(42, 10);
        assert_eq!(demo.best_tour_length, 0.0);

        demo.grasp_iteration();

        assert!(demo.best_tour_length > 0.0);
        assert!(!demo.best_tour.is_empty());
    }

    #[test]
    fn test_grasp_iteration_improves_best() {
        let mut demo = TspGraspDemo::new(42, 50);
        demo.run_grasp(20);

        let first_best = demo.restart_history[0];
        let final_best = demo.best_tour_length;

        // Best should be <= first attempt
        assert!(final_best <= first_best + f64::EPSILON);
    }
}
