use greenlens::app::GreenLensApp;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("GreenLens")
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "GreenLens",
        options,
        Box::new(|cc| Ok(Box::new(GreenLensApp::new(cc)))),
    )
}
