use eframe::egui;

pub struct GreenLensApp {
    status: String,
}

impl GreenLensApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            status: "No image selected.".to_owned(),
        }
    }
}

impl eframe::App for GreenLensApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("GreenLens");
            ui.label("Plant disease classification from leaf images.");
            ui.separator();
            ui.label(&self.status);
        });
    }
}
