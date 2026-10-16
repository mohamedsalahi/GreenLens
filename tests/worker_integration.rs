//! End-to-end test of the worker process from Rust.
//!
//! Starts the real python worker (same code path the application uses),
//! waits for the ready handshake, and checks predictions for a real
//! dataset image. Skips with a message when the python environment,
//! model artifacts, or dataset are not built yet.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use greenlens::worker::{Worker, WorkerEvent};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn first_dataset_image() -> Option<PathBuf> {
    fn walk(dir: &Path) -> Option<PathBuf> {
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = walk(&path) {
                    return Some(found);
                }
            } else {
                let is_image = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_lowercase)
                    .is_some_and(|ext| matches!(ext.as_str(), "jpg" | "jpeg" | "png"));
                if is_image {
                    return Some(path);
                }
            }
        }
        None
    }
    walk(&project_root().join("data/raw/plantvillage"))
}

fn wait_for_event(worker: &Worker, mut predicate: impl FnMut(&WorkerEvent) -> bool) -> Option<WorkerEvent> {
    let deadline = Instant::now() + Duration::from_secs(240);
    while Instant::now() < deadline {
        if let Some(event) = worker.poll() {
            if predicate(&event) {
                return Some(event);
            }
            if matches!(event, WorkerEvent::Exited) {
                return None;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

fn start_worker() -> Option<Worker> {
    let root = project_root();
    let python = root.join(".venv/bin/python");
    let script = root.join("ml/inference_worker.py");
    let model = root.join("artifacts/model.keras");
    if !python.is_file() || !model.is_file() {
        eprintln!("skipping worker integration test: no .venv or no model artifact");
        return None;
    }
    Some(Worker::spawn(&python, &script, Some(&model)).expect("worker should spawn"))
}

#[test]
fn worker_handshake_and_prediction() {
    let Some(mut worker) = start_worker() else {
        return;
    };

    let ready = wait_for_event(&worker, |event| {
        matches!(event, WorkerEvent::Ready(_))
    });
    let Some(WorkerEvent::Ready(info)) = ready else {
        panic!("worker did not become ready");
    };
    assert!(!info.classes.is_empty(), "ready message has no classes");
    assert_eq!(info.classes.len(), info.labels.len());

    let Some(image) = first_dataset_image() else {
        eprintln!("skipping prediction check: dataset not present");
        worker.shutdown();
        return;
    };

    let id = worker.send_predict(&image).expect("request should send");
    let event = wait_for_event(&worker, |event| {
        matches!(event, WorkerEvent::Predictions { id: event_id, .. } if *event_id == id)
    });
    let Some(WorkerEvent::Predictions { predictions, .. }) = event else {
        panic!("no prediction response for request {id}");
    };

    assert_eq!(predictions.len(), 3, "top_k should be 3");
    for pair in predictions.windows(2) {
        assert!(
            pair[0].probability >= pair[1].probability,
            "predictions must be ordered by probability"
        );
    }
    for prediction in &predictions {
        assert!(
            (0.0..=1.0).contains(&prediction.probability),
            "probability out of range: {}",
            prediction.probability
        );
        assert!(!prediction.label.is_empty());
        assert!(info.classes.contains(&prediction.class));
    }
    let total: f64 = predictions.iter().map(|p| p.probability).sum();
    assert!(total <= 1.0001, "top-3 probabilities exceed 1: {total}");

    worker.shutdown();
}

#[test]
fn worker_reports_missing_file_as_structured_error() {
    let Some(mut worker) = start_worker() else {
        return;
    };
    let ready = wait_for_event(
        &worker,
        |event| matches!(event, WorkerEvent::Ready(_) | WorkerEvent::Error { .. }),
    );
    assert!(
        matches!(ready, Some(WorkerEvent::Ready(_))),
        "worker failed to start"
    );

    let missing = project_root().join("data/does-not-exist-42.jpg");
    let id = worker.send_predict(&missing).expect("request should send");
    let event = wait_for_event(&worker, |event| {
        matches!(event, WorkerEvent::Error { id: Some(event_id), .. } if *event_id == id)
    });
    let Some(WorkerEvent::Error { code, .. }) = event else {
        panic!("expected a structured error for a missing file");
    };
    assert_eq!(code, "file_not_found");

    worker.shutdown();
}
