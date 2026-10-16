//! The GreenLens desktop window.
//!
//! Layout: a gradient hero header with the worker status and the file picker,
//! a left card with the image preview, a right card with the analysis result
//! (confidence gauge and ranked candidates), and a status line at the bottom.
//! Inference runs in the worker process, so the UI never blocks on it;
//! responses arrive through the channel in worker.rs and are only accepted
//! when they match the pending request.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui::{self, Align, Color32, FontFamily, FontId, RichText, Stroke};

use crate::worker::{Prediction, Worker, WorkerEvent, WorkerInfo};

// Deep, quiet greens for the room of the app, one vivid accent for plants.
const BG_TOP: Color32 = Color32::from_rgb(8, 15, 11);
const BG_BOTTOM: Color32 = Color32::from_rgb(12, 22, 16);
const HERO_TOP: Color32 = Color32::from_rgb(10, 31, 19);
const HERO_BOTTOM: Color32 = Color32::from_rgb(16, 45, 27);
const CARD: Color32 = Color32::from_rgb(17, 28, 21);
const CARD_STROKE: Color32 = Color32::from_rgb(34, 54, 40);
const TRACK: Color32 = Color32::from_rgb(29, 44, 34);
const ACCENT: Color32 = Color32::from_rgb(64, 224, 128);
const ACCENT_SOFT: Color32 = Color32::from_rgb(38, 126, 76);
const TEXT_BRIGHT: Color32 = Color32::from_rgb(230, 245, 235);
const TEXT_DIM: Color32 = Color32::from_rgb(138, 162, 146);
const TEXT_FAINT: Color32 = Color32::from_rgb(95, 115, 100);
const INK: Color32 = Color32::from_rgb(6, 28, 16);
const AMBER: Color32 = Color32::from_rgb(255, 179, 0);
const RED: Color32 = Color32::from_rgb(255, 105, 97);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

pub enum WorkerState {
    Starting,
    Ready(WorkerInfo),
    Failed(String),
}

struct AnalysisOutcome {
    predictions: Vec<Prediction>,
    elapsed: f32,
}

enum Analysis {
    Idle,
    Running { id: u64, started: Instant },
    Done(AnalysisOutcome),
    Failed(String),
}

pub struct GreenLensApp {
    worker: Option<Worker>,
    worker_state: WorkerState,
    selected: Option<PathBuf>,
    preview: Option<egui::TextureHandle>,
    image_dimensions: Option<(u32, u32)>,
    analysis: Analysis,
    dialog_open: bool,
    dialog_rx: Option<Receiver<Option<PathBuf>>>,
    status: String,
}

impl GreenLensApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_style(&cc.egui_ctx);
        let (worker, worker_state, status) = match start_worker() {
            Ok(worker) => (
                Some(worker),
                WorkerState::Starting,
                "Loading the model...".to_owned(),
            ),
            Err(err) => {
                let message = format!("Could not start the model worker: {err}");
                (None, WorkerState::Failed(message.clone()), message)
            }
        };
        Self {
            worker,
            worker_state,
            selected: None,
            preview: None,
            image_dimensions: None,
            analysis: Analysis::Idle,
            dialog_open: false,
            dialog_rx: None,
            status,
        }
    }

    fn poll_worker(&mut self) {
        let mut events = Vec::new();
        if let Some(worker) = &mut self.worker {
            while let Some(event) = worker.poll() {
                events.push(event);
            }
        }
        for event in events {
            match event {
                WorkerEvent::Ready(info) => {
                    self.status = format!(
                        "Model ready: {} classes, input {}x{}.",
                        info.num_classes,
                        info.input_size.first().copied().unwrap_or(224),
                        info.input_size.get(1).copied().unwrap_or(224)
                    );
                    self.worker_state = WorkerState::Ready(info);
                }
                WorkerEvent::Predictions { id, predictions, .. } => {
                    let elapsed = match &self.analysis {
                        Analysis::Running { id: expected, started } if *expected == id => {
                            Some(started.elapsed())
                        }
                        _ => None,
                    };
                    if let Some(elapsed) = elapsed {
                        let finished = predictions.is_empty();
                        self.analysis = Analysis::Done(AnalysisOutcome {
                            predictions,
                            elapsed: elapsed.as_secs_f32(),
                        });
                        self.status = if finished {
                            "The worker returned no predictions.".to_owned()
                        } else {
                            "Analysis complete.".to_owned()
                        };
                    }
                }
                WorkerEvent::Error { id, code, message } => {
                    let text = describe_error(&code, &message);
                    match (id, &self.analysis) {
                        (Some(id), Analysis::Running { id: expected, .. }) if *expected == id => {
                            self.status = "Analysis failed.".to_owned();
                            self.analysis = Analysis::Failed(text);
                        }
                        (Some(_), _) => {
                            // Response to an old request; the user already
                            // moved on to a different image.
                        }
                        (None, _) => {
                            self.status = "The model worker reported an error.".to_owned();
                            self.worker_state = WorkerState::Failed(text);
                            if matches!(self.analysis, Analysis::Running { .. }) {
                                self.analysis =
                                    Analysis::Failed("The model worker reported an error.".to_owned());
                            }
                        }
                    }
                }
                WorkerEvent::Protocol(problem) => {
                    self.status = format!("Unexpected message from the worker: {problem}");
                }
                WorkerEvent::Exited => {
                    self.worker = None;
                    if !matches!(self.worker_state, WorkerState::Failed(_)) {
                        let text = "The model worker exited unexpectedly.".to_owned();
                        self.status = text.clone();
                        self.worker_state = WorkerState::Failed(text);
                    }
                    if matches!(self.analysis, Analysis::Running { .. }) {
                        self.analysis =
                            Analysis::Failed("The model worker stopped during the analysis.".to_owned());
                    }
                }
            }
        }
    }

    fn poll_dialog(&mut self, ctx: &egui::Context) {
        if !self.dialog_open {
            return;
        }
        let answer = self
            .dialog_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(file) = answer {
            self.dialog_open = false;
            self.dialog_rx = None;
            match file {
                Some(path) => self.load_selected(ctx, path),
                None => self.status = "No file was selected.".to_owned(),
            }
        }
    }

    fn load_selected(&mut self, ctx: &egui::Context, path: PathBuf) {
        match load_preview(ctx, &path) {
            Ok((texture, dimensions)) => {
                let name = file_name_of(&path);
                self.selected = Some(path);
                self.preview = Some(texture);
                self.image_dimensions = Some(dimensions);
                self.analysis = Analysis::Idle;
                self.status = format!("Selected {name}.");
                self.start_analysis();
            }
            Err(err) => {
                self.status = format!("Could not read that image: {err}");
            }
        }
    }

    fn start_analysis(&mut self) {
        let Some(path) = self.selected.clone() else {
            return;
        };
        if !matches!(self.worker_state, WorkerState::Ready(_)) {
            return;
        }
        let Some(worker) = self.worker.as_mut() else {
            return;
        };
        match worker.send_predict(&path) {
            Ok(id) => {
                self.analysis = Analysis::Running {
                    id,
                    started: Instant::now(),
                };
                self.status = "Running inference...".to_owned();
            }
            Err(err) => {
                self.analysis = Analysis::Failed(format!("Could not send the request: {err}"));
            }
        }
    }

    fn open_dialog(&mut self) {
        if self.dialog_open {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.dialog_rx = Some(rx);
        self.dialog_open = true;
        self.status = "Waiting for the file dialog...".to_owned();
        thread::spawn(move || {
            let file = rfd::FileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png", "webp", "bmp"])
                .set_title("Open a leaf image")
                .pick_file();
            let _ = tx.send(file);
        });
    }

    fn check_timeout(&mut self, ctx: &egui::Context) {
        if let Analysis::Running { started, .. } = &self.analysis {
            if started.elapsed() > REQUEST_TIMEOUT {
                self.analysis = Analysis::Failed(
                    "The worker did not answer within 180 seconds.".to_owned(),
                );
                self.status = "Analysis timed out.".to_owned();
            } else {
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }
    }

    fn status_pill(&self, ui: &mut egui::Ui) {
        let (color, text) = match &self.worker_state {
            WorkerState::Starting => (AMBER, "loading model"),
            WorkerState::Ready(_) => (ACCENT, "model ready"),
            WorkerState::Failed(_) => (RED, "worker failed"),
        };
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(0, 0, 0, 110))
            .stroke(Stroke::new(1.0, color.gamma_multiply(0.55)))
            .corner_radius(100)
            .inner_margin(egui::Margin::symmetric(14, 7))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (dot, _) =
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot.center(), 4.5, color);
                    ui.label(RichText::new(text).size(14.0).color(color));
                });
            });
    }

    fn header(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("header")
            .frame(egui::Frame::new())
            .show(ui, |ui| {
                let rect = ui.max_rect();
                paint_gradient(ui, rect, HERO_TOP, HERO_BOTTOM);
                ui.painter().hline(
                    rect.x_range(),
                    rect.bottom() - 0.5,
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(64, 224, 128, 36)),
                );
                egui::Frame::new()
                    .inner_margin(egui::Margin::symmetric(28, 18))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 0.0;
                                    ui.label(
                                        RichText::new("Green")
                                            .font(FontId::new(40.0, FontFamily::Proportional))
                                            .strong()
                                            .color(ACCENT)
                                            .extra_letter_spacing(2.0),
                                    );
                                    ui.label(
                                        RichText::new("Lens")
                                            .font(FontId::new(40.0, FontFamily::Proportional))
                                            .strong()
                                            .color(TEXT_BRIGHT)
                                            .extra_letter_spacing(2.0),
                                    );
                                });
                                ui.add_space(1.0);
                                ui.label(
                                    RichText::new("leaf disease identification")
                                        .size(14.0)
                                        .color(TEXT_DIM),
                                );
                            });
                            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                                let open = ui
                                    .add_enabled_ui(!self.dialog_open, |ui| {
                                        ui.add_sized(
                                            egui::vec2(170.0, 42.0),
                                            egui::Button::new(
                                                RichText::new("Open image...")
                                                    .size(16.0)
                                                    .strong()
                                                    .color(INK),
                                            )
                                            .fill(ACCENT)
                                            .corner_radius(11),
                                        )
                                    })
                                    .inner;
                                if open.clicked() {
                                    self.open_dialog();
                                }
                                ui.add_space(16.0);
                                self.status_pill(ui);
                            });
                        });
                    });
            });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status")
            .frame(egui::Frame::new().fill(Color32::from_rgb(9, 16, 12)))
            .show(ui, |ui| {
                let rect = ui.max_rect();
                ui.painter().hline(
                    rect.x_range(),
                    rect.top() + 0.5,
                    Stroke::new(1.0, CARD_STROKE),
                );
                egui::Frame::new()
                    .inner_margin(egui::Margin::symmetric(28, 9))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&self.status).size(13.0).color(TEXT_DIM));
                            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                                if let WorkerState::Ready(info) = &self.worker_state {
                                    let device = if info.gpus.is_empty() { "cpu" } else { "gpu" };
                                    chip(ui, device);
                                    chip(
                                        ui,
                                        format!(
                                            "input {}x{}",
                                            info.input_size.first().copied().unwrap_or(224),
                                            info.input_size.get(1).copied().unwrap_or(224)
                                        ),
                                    );
                                    chip(ui, format!("{} classes", info.num_classes));
                                }
                            });
                        });
                    });
            });
    }

    fn image_column(&mut self, ui: &mut egui::Ui) {
        section_title(ui, "IMAGE", None);
        ui.add_space(8.0);
        card(ui, |ui| {
            let max = egui::vec2((ui.available_width() - 10.0).max(120.0), 380.0);
            ui.set_min_height(300.0);
            if let Some(texture) = &self.preview {
                ui.vertical_centered(|ui| {
                    ui.add(
                        egui::Image::new(texture)
                            .maintain_aspect_ratio(true)
                            .max_size(max)
                            .corner_radius(10),
                    );
                });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(92.0);
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(64.0, 64.0), egui::Sense::hover());
                    let center = rect.center();
                    ui.painter()
                        .circle_stroke(center, 31.0, Stroke::new(1.5, CARD_STROKE));
                    ui.painter()
                        .circle_filled(center + egui::vec2(11.0, -10.0), 4.0, TEXT_FAINT);
                    ui.painter().add(egui::Shape::line(
                        vec![
                            center + egui::vec2(-15.0, 9.0),
                            center + egui::vec2(-1.0, -7.0),
                            center + egui::vec2(15.0, 9.0),
                        ],
                        Stroke::new(2.0, TEXT_FAINT),
                    ));
                    ui.add_space(14.0);
                    ui.label(RichText::new("No image selected").size(19.0).color(TEXT_DIM));
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Choose a leaf photo to begin")
                            .size(13.5)
                            .color(TEXT_FAINT),
                    );
                    ui.add_space(92.0);
                });
            }
        });
        if let Some(path) = &self.selected {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(file_name_of(path))
                        .size(15.0)
                        .strong()
                        .color(TEXT_BRIGHT),
                );
                if let Some((w, h)) = self.image_dimensions {
                    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{w} x {h}"))
                                .size(13.0)
                                .color(TEXT_FAINT),
                        );
                    });
                }
            });
        }
    }

    fn results_column(&mut self, ui: &mut egui::Ui) {
        let elapsed = match &self.analysis {
            Analysis::Done(outcome) => Some(outcome.elapsed),
            _ => None,
        };
        section_title(ui, "ANALYSIS", elapsed);
        ui.add_space(8.0);
        card(ui, |ui| {
            if let WorkerState::Failed(message) = &self.worker_state {
                ui.label(
                    RichText::new("The model is not available")
                        .size(17.0)
                        .strong()
                        .color(RED),
                );
                ui.add_space(6.0);
                ui.label(RichText::new(message).size(13.5).color(TEXT_DIM));
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Check that the python environment and artifacts/model.keras \
                         exist; see the Model section of the README.",
                    )
                    .size(13.0)
                    .color(TEXT_FAINT),
                );
                return;
            }

            let can_analyze = matches!(self.worker_state, WorkerState::Ready(_))
                && self.selected.is_some()
                && !matches!(self.analysis, Analysis::Running { .. });
            let again = ui.add_enabled(
                can_analyze,
                egui::Button::new(RichText::new("Analyze again").size(15.0))
                    .fill(Color32::from_rgb(20, 34, 25))
                    .corner_radius(10),
            );
            if again.clicked() {
                self.start_analysis();
            }
            ui.add_space(14.0);

            match &self.analysis {
                Analysis::Idle => {
                    ui.label(
                        RichText::new("Pick a leaf photo to begin.")
                            .size(15.0)
                            .color(TEXT_DIM),
                    );
                    ui.add_space(16.0);
                    for fraction in [0.72, 0.5, 0.32] {
                        skeleton_row(ui, fraction);
                    }
                }
                Analysis::Running { .. } => {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new().size(26.0));
                        ui.add_space(8.0);
                        ui.vertical(|ui| {
                            ui.label(
                                RichText::new("Running inference...")
                                    .size(16.0)
                                    .color(TEXT_BRIGHT),
                            );
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new("The model is looking at the leaf")
                                    .size(13.0)
                                    .color(TEXT_FAINT),
                            );
                        });
                    });
                }
                Analysis::Failed(message) => {
                    ui.label(
                        RichText::new("Analysis failed")
                            .size(17.0)
                            .strong()
                            .color(RED),
                    );
                    ui.add_space(4.0);
                    ui.label(RichText::new(message).size(13.5).color(TEXT_DIM));
                }
                Analysis::Done(outcome) => {
                    if outcome.predictions.is_empty() {
                        ui.colored_label(AMBER, "The worker returned no predictions.");
                    } else {
                        prediction_detail(ui, &outcome.predictions);
                    }
                }
            }

            ui.add_space(18.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(
                RichText::new(
                    "Preliminary, image-based identification only, not a diagnosis. \
                     Image quality, lighting, and plant variety all affect the result.",
                )
                .size(12.0)
                .color(TEXT_FAINT),
            );
        });
    }
}

impl eframe::App for GreenLensApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_worker();
        self.poll_dialog(&ctx);
        self.check_timeout(&ctx);
        self.header(ui);
        self.status_bar(ui);
        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ui, |ui| {
                let rect = ui.max_rect();
                paint_gradient(ui, rect, BG_TOP, BG_BOTTOM);
                egui::Frame::new()
                    .inner_margin(egui::Margin::symmetric(28, 24))
                    .show(ui, |ui| {
                        let spacing = 26.0;
                        let col_width = (ui.available_width() - spacing) / 2.0;
                        ui.spacing_mut().item_spacing.x = spacing;
                        ui.horizontal_top(|ui| {
                            ui.allocate_ui_with_layout(
                                egui::vec2(col_width, 0.0),
                                egui::Layout::top_down(Align::Min),
                                |ui| {
                                    ui.set_width(col_width);
                                    self.image_column(ui);
                                },
                            );
                            ui.allocate_ui_with_layout(
                                egui::vec2(col_width, 0.0),
                                egui::Layout::top_down(Align::Min),
                                |ui| {
                                    ui.set_width(col_width);
                                    self.results_column(ui);
                                },
                            );
                        });
                    });
            });
    }
}

fn prediction_detail(ui: &mut egui::Ui, predictions: &[Prediction]) {
    let top = &predictions[0];
    ui.horizontal(|ui| {
        confidence_gauge(ui, top.probability as f32);
        ui.add_space(18.0);
        ui.vertical(|ui| {
            ui.add_space(8.0);
            ui.label(
                RichText::new("MOST LIKELY")
                    .size(11.5)
                    .strong()
                    .color(TEXT_FAINT)
                    .extra_letter_spacing(2.0),
            );
            ui.add_space(4.0);
            ui.label(RichText::new(&top.label).size(25.0).strong().color(ACCENT));
            ui.add_space(3.0);
            ui.label(RichText::new(&top.class).size(13.0).color(TEXT_DIM));
        });
    });

    if predictions.len() > 1 {
        ui.add_space(20.0);
        ui.label(
            RichText::new("OTHER CANDIDATES")
                .size(11.5)
                .strong()
                .color(TEXT_FAINT)
                .extra_letter_spacing(2.0),
        );
        for prediction in &predictions[1..] {
            ui.add_space(11.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(&prediction.label)
                        .size(14.5)
                        .color(TEXT_BRIGHT),
                );
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{:.1}%", prediction.probability * 100.0))
                            .size(13.5)
                            .color(TEXT_DIM),
                    );
                });
            });
            ui.add_space(5.0);
            bar(ui, prediction.probability as f32, ACCENT_SOFT);
        }
    }
}

fn confidence_gauge(ui: &mut egui::Ui, value: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(136.0, 136.0), egui::Sense::hover());
    let center = rect.center();
    let radius = rect.width().min(rect.height()) * 0.5 - 12.0;
    let thickness = 11.0;
    let painter = ui.painter();
    painter.circle_stroke(center, radius, Stroke::new(thickness, TRACK));

    let sweep = value.clamp(0.0, 1.0) * std::f32::consts::TAU;
    if sweep > 0.02 {
        let segments = 72;
        let start = -std::f32::consts::FRAC_PI_2;
        let mut points = Vec::with_capacity(segments + 1);
        for i in 0..=segments {
            let angle = start + sweep * (i as f32 / segments as f32);
            points.push(center + egui::vec2(angle.cos(), angle.sin()) * radius);
        }
        painter.add(egui::Shape::line(points, Stroke::new(thickness, ACCENT)));
    }

    painter.text(
        center + egui::vec2(0.0, -7.0),
        egui::Align2::CENTER_CENTER,
        format!("{:.0}%", value * 100.0),
        FontId::new(27.0, FontFamily::Proportional),
        TEXT_BRIGHT,
    );
    painter.text(
        center + egui::vec2(0.0, 16.0),
        egui::Align2::CENTER_CENTER,
        "confidence",
        FontId::new(11.5, FontFamily::Proportional),
        TEXT_FAINT,
    );
}

fn bar(ui: &mut egui::Ui, fraction: f32, color: Color32) {
    let width = ui.available_width().max(40.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 9.0), egui::Sense::hover());
    let radius = egui::CornerRadius::same(4);
    ui.painter().rect_filled(rect, radius, TRACK);
    let filled = fraction.clamp(0.0, 1.0);
    if filled > 0.01 {
        let partial =
            egui::Rect::from_min_size(rect.min, egui::vec2((rect.width() * filled).max(6.0), rect.height()));
        ui.painter().rect_filled(partial, radius, color);
    }
}

fn skeleton_row(ui: &mut egui::Ui, fraction: f32) {
    let width = ui.available_width().max(60.0) * fraction;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 9.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(4), TRACK);
    ui.add_space(9.0);
}

fn section_title(ui: &mut egui::Ui, text: &str, trailing: Option<f32>) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(text)
                .size(12.0)
                .strong()
                .color(TEXT_FAINT)
                .extra_letter_spacing(2.2),
        );
        if let Some(seconds) = trailing {
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                chip(ui, format!("{seconds:.1} s"));
            });
        }
    });
}

fn chip(ui: &mut egui::Ui, text: impl Into<String>) {
    egui::Frame::new()
        .fill(Color32::from_rgb(17, 28, 21))
        .corner_radius(9)
        .inner_margin(egui::Margin::symmetric(10, 4))
        .show(ui, |ui| {
            ui.label(RichText::new(text.into()).size(12.0).color(TEXT_DIM));
        });
}

fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, CARD_STROKE))
        .corner_radius(16)
        .shadow(egui::epaint::Shadow {
            offset: [0, 6],
            blur: 22,
            spread: 0,
            color: Color32::from_black_alpha(90),
        })
        .inner_margin(egui::Margin::symmetric(18, 16))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add(ui)
        })
        .inner
}

fn paint_gradient(ui: &egui::Ui, rect: egui::Rect, top: Color32, bottom: Color32) {
    let steps = 36;
    let painter = ui.painter();
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let y0 = rect.top() + (rect.bottom() - rect.top()) * t0;
        let y1 = rect.top() + (rect.bottom() - rect.top()) * t1;
        let color = mix(top, bottom, (t0 + t1) * 0.5);
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(rect.left(), y0), egui::pos2(rect.right(), y1)),
            0,
            color,
        );
    }
}

fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let channel = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgb(
        channel(a.r(), b.r()),
        channel(a.g(), b.g()),
        channel(a.b(), b.b()),
    )
}

fn describe_error(code: &str, message: &str) -> String {
    match code {
        "startup_failed" => format!(
            "The model could not be loaded: {message}. Check that artifacts/model.keras \
             exists and was trained for this python environment."
        ),
        "file_not_found" => format!("The file was not found: {message}"),
        "invalid_image" => format!("The image could not be read: {message}"),
        "inference_failed" => format!("Inference failed: {message}"),
        _ => format!("Worker error ({code}): {message}"),
    }
}

fn apply_style(ctx: &egui::Context) {
    install_fonts(ctx);

    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG_TOP;
    visuals.window_fill = CARD;
    visuals.extreme_bg_color = Color32::from_rgb(10, 18, 13);
    visuals.faint_bg_color = Color32::from_rgb(15, 25, 18);
    visuals.window_corner_radius = egui::CornerRadius::same(14);
    visuals.window_stroke = Stroke::new(1.0, CARD_STROKE);
    visuals.selection.bg_fill = ACCENT_SOFT;
    visuals.hyperlink_color = ACCENT;

    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, CARD_STROKE);
    widgets.noninteractive.corner_radius = egui::CornerRadius::same(10);
    widgets.inactive.corner_radius = egui::CornerRadius::same(10);
    widgets.inactive.bg_fill = CARD;
    widgets.inactive.weak_bg_fill = CARD;
    widgets.inactive.bg_stroke = Stroke::new(1.0, CARD_STROKE);
    widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    widgets.hovered.corner_radius = egui::CornerRadius::same(10);
    widgets.hovered.bg_fill = Color32::from_rgb(26, 42, 32);
    widgets.hovered.weak_bg_fill = Color32::from_rgb(26, 42, 32);
    widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_SOFT);
    widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    widgets.active.corner_radius = egui::CornerRadius::same(10);
    widgets.active.bg_fill = ACCENT_SOFT;
    widgets.active.weak_bg_fill = ACCENT_SOFT;
    widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    widgets.active.fg_stroke = Stroke::new(1.0, INK);
    ctx.set_visuals(visuals);

    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(10.0, 8.0);
        style.spacing.button_padding = egui::vec2(14.0, 8.0);
        style.text_styles = [
            (
                egui::TextStyle::Heading,
                FontId::new(28.0, FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Body,
                FontId::new(15.5, FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Button,
                FontId::new(15.5, FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Small,
                FontId::new(12.5, FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Monospace,
                FontId::new(13.0, FontFamily::Monospace),
            ),
        ]
        .into();
    });
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "inter".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/fonts/InterVariable.ttf"
        ))),
    );
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "inter".to_owned());
    ctx.set_fonts(fonts);
}

fn start_worker() -> std::io::Result<Worker> {
    let root = project_root();
    let python = std::env::var("GREENLENS_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join(".venv/bin/python"));
    let script = root.join("ml/inference_worker.py");
    let model = root.join("artifacts/model.keras");
    let model = model.is_file().then_some(model);
    Worker::spawn(&python, &script, model.as_deref())
}

/// Locate the project so the worker script and model can be found no matter
/// where the binary is started from. GREENLENS_HOME overrides everything.
fn project_root() -> PathBuf {
    if let Ok(home) = std::env::var("GREENLENS_HOME") {
        return PathBuf::from(home);
    }
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("ml/inference_worker.py").is_file() {
            return cwd;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
            if dir.join("ml/inference_worker.py").is_file() {
                return dir.to_path_buf();
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn load_preview(
    ctx: &egui::Context,
    path: &std::path::Path,
) -> Result<(egui::TextureHandle, (u32, u32)), String> {
    let image = image::open(path).map_err(|err| err.to_string())?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let color = egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        rgba.as_raw(),
    );
    let texture = ctx.load_texture("selected-image", color, egui::TextureOptions::default());
    Ok((texture, (width, height)))
}

fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
