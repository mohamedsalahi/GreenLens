//! The GreenLens desktop window.
//!
//! Layout: a header with the worker status and the file picker, a left
//! column with the image preview, a right column with ranked results,
//! and a status line at the bottom. Inference runs in the worker process,
//! so the UI never blocks on it; responses arrive through the channel in
//! worker.rs and are only accepted when they match the pending request.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText};

use crate::worker::{Prediction, Worker, WorkerEvent, WorkerInfo};

const GREEN: Color32 = Color32::from_rgb(46, 125, 50);
const GREEN_BRIGHT: Color32 = Color32::from_rgb(129, 199, 132);
const GRAY: Color32 = Color32::from_rgb(120, 125, 125);
const AMBER: Color32 = Color32::from_rgb(255, 179, 0);
const RED: Color32 = Color32::from_rgb(211, 47, 47);
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
        if let Some(worker) = &self.worker {
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
                            self.status = text.clone();
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

    fn header(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("header").show(ui, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("GreenLens")
                        .size(26.0)
                        .strong()
                        .color(GREEN_BRIGHT),
                );
                ui.label(RichText::new("leaf disease identification").color(GRAY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let open = ui.add_enabled(!self.dialog_open, egui::Button::new("Open image..."));
                    if open.clicked() {
                        self.open_dialog();
                    }
                    ui.add_space(12.0);
                    let (color, text) = match &self.worker_state {
                        WorkerState::Starting => (AMBER, "loading model..."),
                        WorkerState::Ready(_) => (GREEN_BRIGHT, "model ready"),
                        WorkerState::Failed(_) => (RED, "worker failed"),
                    };
                    ui.label(RichText::new(text).color(color));
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, color);
                });
            });
            ui.add_space(8.0);
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(&self.status).small().color(GRAY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let WorkerState::Ready(info) = &self.worker_state {
                        if let (Some(w), Some(h)) = (
                            info.input_size.first(),
                            info.input_size.get(1),
                        ) {
                            let gpu = if info.gpus.is_empty() { "CPU" } else { "GPU" };
                            ui.label(
                                RichText::new(format!(
                                    "{} classes | input {w}x{h} | {gpu}",
                                    info.num_classes
                                ))
                                .small()
                                .color(GRAY),
                            );
                        }
                    }
                });
            });
            ui.add_space(3.0);
        });
    }

    fn image_column(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Image").strong());
        ui.add_space(6.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            let max = egui::vec2(ui.available_width() - 12.0, 420.0);
            ui.set_min_height(300.0);
            if let Some(texture) = &self.preview {
                ui.vertical_centered(|ui| {
                    ui.add(
                        egui::Image::new(texture)
                            .maintain_aspect_ratio(true)
                            .max_size(max),
                    );
                });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(130.0);
                    ui.label(RichText::new("No image selected").color(GRAY));
                    ui.label(
                        RichText::new("Use \"Open image...\" to choose a leaf photo.")
                            .small()
                            .color(GRAY),
                    );
                });
            }
        });
        if let Some(path) = &self.selected {
            ui.add_space(6.0);
            ui.label(RichText::new(file_name_of(path)).strong());
            if let Some((w, h)) = self.image_dimensions {
                ui.label(RichText::new(format!("{w} x {h} pixels")).small().color(GRAY));
            }
        }
    }

    fn results_column(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Results").strong());
        ui.add_space(6.0);

        if let WorkerState::Failed(message) = &self.worker_state {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.colored_label(RED, "The model is not available.");
                ui.label(RichText::new(message).small());
                ui.label(
                    RichText::new(
                        "Check that the python environment and artifacts/model.keras \
                         exist; see the Model section of the README.",
                    )
                    .small()
                    .color(GRAY),
                );
            });
            return;
        }

        let can_analyze = matches!(self.worker_state, WorkerState::Ready(_))
            && self.selected.is_some()
            && !matches!(self.analysis, Analysis::Running { .. });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_analyze, egui::Button::new("Analyze again"))
                .clicked()
            {
                self.start_analysis();
            }
            if let Analysis::Done(outcome) = &self.analysis {
                ui.label(
                    RichText::new(format!("in {:.1} s", outcome.elapsed))
                        .small()
                        .color(GRAY),
                );
            }
        });
        ui.add_space(10.0);

        match &self.analysis {
            Analysis::Idle => {
                ui.label(RichText::new("Select an image to analyze it.").color(GRAY));
            }
            Analysis::Running { .. } => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Running inference...");
                });
            }
            Analysis::Failed(message) => {
                ui.colored_label(RED, message);
            }
            Analysis::Done(outcome) => {
                if outcome.predictions.is_empty() {
                    ui.colored_label(AMBER, "The worker returned no predictions.");
                } else {
                    prediction_list(ui, &outcome.predictions);
                }
            }
        }

        ui.add_space(14.0);
        ui.separator();
        ui.add_space(6.0);
        ui.label(
            RichText::new(
                "Preliminary, image-based identification only, not a diagnosis. \
                 Image quality, lighting, and plant variety all affect the result.",
            )
            .small()
            .color(GRAY),
        );
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
        egui::CentralPanel::default().show(ui, |ui| {
            ui.add_space(8.0);
            ui.columns(2, |columns| {
                self.image_column(&mut columns[0]);
                self.results_column(&mut columns[1]);
            });
        });
    }
}

fn prediction_list(ui: &mut egui::Ui, predictions: &[Prediction]) {
    let top = &predictions[0];
    ui.label(RichText::new("Most likely").small().color(GRAY));
    ui.label(RichText::new(&top.label).size(21.0).strong().color(GREEN_BRIGHT));
    ui.add(
        egui::ProgressBar::new(top.probability as f32)
            .fill(GREEN)
            .text(format!("{:.1} percent", top.probability * 100.0))
            .desired_width(ui.available_width()),
    );
    if predictions.len() > 1 {
        ui.add_space(12.0);
        ui.label(RichText::new("Other candidates").small().color(GRAY));
        for prediction in &predictions[1..] {
            ui.add_space(2.0);
            ui.label(&prediction.label);
            ui.add(
                egui::ProgressBar::new(prediction.probability as f32)
                    .fill(GRAY)
                    .text(format!("{:.1} percent", prediction.probability * 100.0))
                    .desired_width(ui.available_width()),
            );
        }
    }
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
    let mut visuals = egui::Visuals::dark();
    visuals.selection.bg_fill = GREEN;
    visuals.hyperlink_color = GREEN_BRIGHT;
    ctx.set_visuals(visuals);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
    });
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
