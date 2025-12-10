//! Simular Orbit Demo - Terminal User Interface
//!
//! A TUI demonstration of orbital mechanics using ratatui.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin orbit-tui --features tui
//! ```
//!
//! # Controls
//!
//! - Space: Pause/Resume
//! - R: Reset simulation
//! - +/-: Adjust time scale
//! - Q: Quit

#![forbid(unsafe_code)]

#[cfg(feature = "tui")]
mod app {
    use std::io;
    use std::time::{Duration, Instant};
    use crossterm::{
        event::{self, Event, KeyCode, KeyEventKind},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout, Rect},
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{
            Block, Borders, Gauge, Paragraph,
            canvas::{Canvas, Points},
        },
        Frame, Terminal,
    };

    use simular::orbit::prelude::*;
    use simular::orbit::physics::YoshidaIntegrator;
    use simular::orbit::render::OrbitTrail;

    /// Application state.
    pub struct App {
        state: NBodyState,
        jidoka: OrbitJidokaGuard,
        heijunka: HeijunkaScheduler,
        _integrator: YoshidaIntegrator,
        trails: Vec<OrbitTrail>,
        config: KeplerConfig,
        paused: bool,
        time_scale: f64,
        sim_time_days: f64,
        frame_count: u64,
        should_quit: bool,
    }

    impl App {
        pub fn new() -> Self {
            let config = KeplerConfig::earth_sun();
            let state = config.build(1e6);

            let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
            jidoka.initialize(&state);

            let heijunka_config = HeijunkaConfig {
                frame_budget_ms: 33.0, // 30 FPS target
                physics_budget_fraction: 0.5,
                base_dt: 3600.0, // 1 hour per physics step
                max_substeps: 24,
                ..HeijunkaConfig::default()
            };
            let heijunka = HeijunkaScheduler::new(heijunka_config);

            let trails = vec![
                OrbitTrail::new(0),    // Sun doesn't need trail
                OrbitTrail::new(500),  // Earth trail
            ];

            Self {
                state,
                jidoka,
                heijunka,
                _integrator: YoshidaIntegrator::new(),
                trails,
                config,
                paused: false,
                time_scale: 1.0,
                sim_time_days: 0.0,
                frame_count: 0,
                should_quit: false,
            }
        }

        pub fn reset(&mut self) {
            self.state = self.config.build(1e6);
            self.jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
            self.jidoka.initialize(&self.state);
            self.sim_time_days = 0.0;
            self.frame_count = 0;

            for trail in &mut self.trails {
                trail.clear();
            }
        }

        pub fn update(&mut self) {
            if self.paused {
                return;
            }

            // Execute physics with Heijunka time budget
            if let Ok(result) = self.heijunka.execute_frame(&mut self.state) {
                self.sim_time_days += result.sim_time_advanced / 86400.0;

                // Update trails
                for (i, body) in self.state.bodies.iter().enumerate() {
                    if i < self.trails.len() {
                        let (x, y, _) = body.position.as_meters();
                        self.trails[i].push(x, y);
                    }
                }
            }

            // Check Jidoka guards
            let response = self.jidoka.check(&self.state);
            if response.should_pause() || response.should_halt() {
                self.paused = true;
            }

            self.frame_count += 1;
        }

        pub fn handle_key(&mut self, key: KeyCode) {
            match key {
                KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                KeyCode::Char(' ') => self.paused = !self.paused,
                KeyCode::Char('r') => self.reset(),
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    self.time_scale = (self.time_scale * 2.0).min(1000.0);
                }
                KeyCode::Char('-') => {
                    self.time_scale = (self.time_scale / 2.0).max(0.1);
                }
                _ => {}
            }
        }
    }

    /// Draw the UI.
    pub fn ui(f: &mut Frame, app: &App) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title
                Constraint::Min(10),     // Main orbit view
                Constraint::Length(3),   // Status bar
                Constraint::Length(5),   // Jidoka/Heijunka status
            ])
            .split(f.area());

        // Title
        let title = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    " SIMULAR ORBIT DEMO ",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" | "),
                Span::styled(
                    if app.paused { "[PAUSED]" } else { "[RUNNING]" },
                    Style::default().fg(if app.paused { Color::Yellow } else { Color::Green }),
                ),
                Span::raw(" | "),
                Span::styled(
                    format!("Time: {:.1} days", app.sim_time_days),
                    Style::default().fg(Color::White),
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("Controls: [Space] Pause  [R] Reset  [+/-] Speed  [Q] Quit"));
        f.render_widget(title, chunks[0]);

        // Main orbit canvas
        draw_orbit_canvas(f, chunks[1], app);

        // Status bar
        let energy = app.state.total_energy();
        let status = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Energy: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.6e} J", energy),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" | "),
                Span::styled("Frame: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", app.frame_count),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" | "),
                Span::styled("Scale: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}x", app.time_scale),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL));
        f.render_widget(status, chunks[2]);

        // Jidoka and Heijunka status
        draw_status_panel(f, chunks[3], app);
    }

    fn draw_orbit_canvas(f: &mut Frame, area: Rect, app: &App) {
        // Scale factor: 1 AU should fit in roughly half the canvas
        let scale = 2.0 / AU; // Normalized to ~2 units for 1 AU

        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title("Orbit View"))
            .x_bounds([-2.0, 2.0])
            .y_bounds([-2.0, 2.0])
            .paint(|ctx| {
                // Draw orbit trail for Earth
                if app.trails.len() > 1 {
                    let trail_points: Vec<(f64, f64)> = app.trails[1]
                        .points()
                        .iter()
                        .map(|(x, y)| (x * scale, y * scale))
                        .collect();

                    if !trail_points.is_empty() {
                        ctx.draw(&Points {
                            coords: &trail_points,
                            color: Color::Blue,
                        });
                    }
                }

                // Draw Sun at origin
                let (sun_x, sun_y, _) = app.state.bodies[0].position.as_meters();
                ctx.print(sun_x * scale, sun_y * scale, Span::styled("☉", Style::default().fg(Color::Yellow)));

                // Draw Earth
                if app.state.bodies.len() > 1 {
                    let (earth_x, earth_y, _) = app.state.bodies[1].position.as_meters();
                    ctx.print(earth_x * scale, earth_y * scale, Span::styled("🌍", Style::default().fg(Color::Cyan)));
                }
            });

        f.render_widget(canvas, area);
    }

    fn draw_status_panel(f: &mut Frame, area: Rect, app: &App) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Jidoka status
        let jidoka_status = app.jidoka.status();
        let jidoka_color = if jidoka_status.energy_ok && jidoka_status.angular_momentum_ok {
            Color::Green
        } else if jidoka_status.warning_count > 0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let jidoka_widget = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Jidoka: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    if jidoka_status.energy_ok { "✓" } else { "✗" },
                    Style::default().fg(jidoka_color),
                ),
                Span::raw(" Energy "),
                Span::styled(
                    if jidoka_status.angular_momentum_ok { "✓" } else { "✗" },
                    Style::default().fg(jidoka_color),
                ),
                Span::raw(" L "),
                Span::styled(
                    if jidoka_status.finite_ok { "✓" } else { "✗" },
                    Style::default().fg(jidoka_color),
                ),
                Span::raw(" Finite"),
            ]),
            Line::from(vec![
                Span::styled(
                    format!("ΔE: {:.2e}  ΔL: {:.2e}", jidoka_status.energy_error, jidoka_status.angular_momentum_error),
                    Style::default().fg(Color::Gray),
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("Jidoka"));
        f.render_widget(jidoka_widget, chunks[0]);

        // Heijunka status
        let heijunka_status = app.heijunka.status();
        let budget_ratio = (heijunka_status.utilization * 100.0).min(100.0) as u16;

        let heijunka_widget = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Heijunka Budget"))
            .gauge_style(
                Style::default()
                    .fg(if heijunka_status.utilization <= 1.0 { Color::Green } else { Color::Red })
            )
            .percent(budget_ratio)
            .label(format!(
                "{:.1}ms/{:.1}ms {:?}",
                heijunka_status.used_ms,
                heijunka_status.budget_ms,
                heijunka_status.quality,
            ));
        f.render_widget(heijunka_widget, chunks[1]);
    }

    pub fn run() -> io::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Create app state
        let mut app = App::new();
        let tick_rate = Duration::from_millis(33); // ~30 FPS

        loop {
            let start = Instant::now();

            // Draw UI
            terminal.draw(|f| ui(f, &app))?;

            // Handle input
            let timeout = tick_rate.saturating_sub(start.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        app.handle_key(key.code);
                    }
                }
            }

            if app.should_quit {
                break;
            }

            // Update simulation
            app.update();
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }
}

fn main() {
    #[cfg(feature = "tui")]
    {
        if let Err(e) = app::run() {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "tui"))]
    {
        eprintln!("TUI feature not enabled. Run with --features tui");
        std::process::exit(1);
    }
}
