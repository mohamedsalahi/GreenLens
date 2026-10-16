//! Management of the Python inference worker process.
//!
//! The worker is a separate Python process (ml/inference_worker.py) that
//! loads the trained model once and answers JSON lines on stdin/stdout.
//! This module owns the child process, the request ids, and a background
//! reader thread that forwards every response to the GUI as an event.

use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    pub index: u32,
    pub class: String,
    pub label: String,
    pub probability: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OriginalSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WorkerInfo {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub num_classes: usize,
    #[serde(default)]
    pub classes: Vec<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub input_size: Vec<u32>,
    #[serde(default)]
    pub preprocessing: Option<String>,
    #[serde(default)]
    pub gpus: Vec<String>,
}

#[derive(Debug)]
pub enum WorkerEvent {
    Ready(WorkerInfo),
    Predictions {
        id: u64,
        path: PathBuf,
        original_size: OriginalSize,
        predictions: Vec<Prediction>,
    },
    Error {
        id: Option<u64>,
        code: String,
        message: String,
    },
    Protocol(String),
    Exited,
}

/// One line of the worker protocol, decoded.
#[derive(Debug)]
enum WireMessage {
    Ready(WorkerInfo),
    Pong,
    Result {
        id: u64,
        path: String,
        original_size: OriginalSize,
        predictions: Vec<Prediction>,
    },
    Error {
        id: Option<u64>,
        code: String,
        message: String,
    },
    Bye,
}

fn decode<T: DeserializeOwned>(value: &serde_json::Value, target: &str) -> Result<T, String> {
    serde_json::from_value(value.clone()).map_err(|e| format!("bad {target} message: {e}"))
}

fn parse_line(line: &str) -> Result<WireMessage, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("not valid JSON: {e}"))?;
    let kind = value
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "message has no type".to_string())?;
    match kind {
        "ready" => decode(&value, "ready").map(WireMessage::Ready),
        "pong" => Ok(WireMessage::Pong),
        "result" => decode(&value, "result").map(
            |m: ResultMessage| WireMessage::Result {
                id: m.id,
                path: m.path,
                original_size: m.original_size,
                predictions: m.predictions,
            },
        ),
        "error" => decode(&value, "error").map(|m: ErrorMessage| WireMessage::Error {
            id: m.id,
            code: m.code,
            message: m.message,
        }),
        "bye" => Ok(WireMessage::Bye),
        other => Err(format!("unknown message type {other:?}")),
    }
}

#[derive(Deserialize)]
struct ResultMessage {
    id: u64,
    path: String,
    original_size: OriginalSize,
    #[serde(default)]
    predictions: Vec<Prediction>,
}

#[derive(Deserialize)]
struct ErrorMessage {
    #[serde(default)]
    id: Option<u64>,
    code: String,
    message: String,
}

pub struct Worker {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    events: Receiver<WorkerEvent>,
    next_id: u64,
}

impl Worker {
    /// Start the worker process. `model` is passed only when explicitly set,
    /// otherwise the worker falls back to its own default path.
    pub fn spawn(python: &Path, script: &Path, model: Option<&Path>) -> std::io::Result<Worker> {
        let mut command = Command::new(python);
        command
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(model) = model {
            command.arg("--model").arg(model);
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| {
            Error::new(ErrorKind::BrokenPipe, "worker stdin was not piped")
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            Error::new(ErrorKind::BrokenPipe, "worker stdout was not piped")
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            Error::new(ErrorKind::BrokenPipe, "worker stderr was not piped")
        })?;

        let (tx, rx) = mpsc::channel();
        spawn_stdout_reader(stdout, tx);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("[worker] {line}");
            }
        });

        Ok(Worker {
            child: Some(child),
            stdin: Some(stdin),
            events: rx,
            next_id: 1,
        })
    }

    pub fn send_predict(&mut self, path: &Path) -> std::io::Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&serde_json::json!({
            "id": id,
            "type": "predict",
            "path": path,
            "top_k": 3,
        }))?;
        Ok(id)
    }

    fn send(&mut self, value: &serde_json::Value) -> std::io::Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            Error::new(ErrorKind::BrokenPipe, "worker stdin is closed")
        })?;
        writeln!(stdin, "{value}")?;
        stdin.flush()
    }

    /// Next event, if any, without blocking.
    pub fn poll(&self) -> Option<WorkerEvent> {
        match self.events.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(WorkerEvent::Exited),
        }
    }

    /// Ask the worker to exit, then make sure the process is gone.
    pub fn shutdown(&mut self) {
        let _ = self.send(&serde_json::json!({"id": null, "type": "shutdown"}));
        self.stdin = None;
        if let Some(mut child) = self.child.take() {
            for _ in 0..20 {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => thread::sleep(Duration::from_millis(100)),
                    Err(_) => break,
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn spawn_stdout_reader(stdout: std::process::ChildStdout, tx: Sender<WorkerEvent>) {
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = match line {
                Ok(line) => line,
                Err(_) => break,
            };
            if line.trim().is_empty() {
                continue;
            }
            let event = match parse_line(&line) {
                Ok(WireMessage::Ready(info)) => WorkerEvent::Ready(info),
                Ok(WireMessage::Pong) => continue,
                Ok(WireMessage::Result {
                    id,
                    path,
                    original_size,
                    predictions,
                }) => WorkerEvent::Predictions {
                    id,
                    path: PathBuf::from(path),
                    original_size,
                    predictions,
                },
                Ok(WireMessage::Error { id, code, message }) => {
                    WorkerEvent::Error { id, code, message }
                }
                Ok(WireMessage::Bye) => continue,
                Err(problem) => WorkerEvent::Protocol(problem),
            };
            if tx.send(event).is_err() {
                break;
            }
        }
        let _ = tx.send(WorkerEvent::Exited);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ready_message() {
        let line = r#"{"type":"ready","model":"model.keras","num_classes":2,
            "classes":["A","B"],"labels":["a","b"],"input_size":[224,224],
            "preprocessing":"vgg16","gpus":["/physical_device:GPU:0"]}"#;
        match parse_line(line).unwrap() {
            WireMessage::Ready(info) => {
                assert_eq!(info.num_classes, 2);
                assert_eq!(info.classes, vec!["A", "B"]);
                assert_eq!(info.input_size, vec![224, 224]);
                assert_eq!(info.gpus.len(), 1);
            }
            other => panic!("expected ready, got {other:?}"),
        }
    }

    #[test]
    fn parses_result_with_ranked_predictions() {
        let line = r#"{"id":7,"type":"result","path":"/tmp/leaf.jpg",
            "original_size":{"width":256,"height":256},
            "predictions":[
                {"index":3,"class":"Tomato___healthy","label":"Tomato: healthy","probability":0.91},
                {"index":1,"class":"Tomato___Late_blight","label":"Tomato: Late blight","probability":0.05}]}"#;
        match parse_line(line).unwrap() {
            WireMessage::Result { id, predictions, .. } => {
                assert_eq!(id, 7);
                assert_eq!(predictions.len(), 2);
                assert!(predictions[0].probability > predictions[1].probability);
            }
            other => panic!("expected result, got {other:?}"),
        }
    }

    #[test]
    fn parses_error_without_id() {
        let line = r#"{"type":"error","code":"startup_failed","message":"no model"}"#;
        match parse_line(line).unwrap() {
            WireMessage::Error { id, code, .. } => {
                assert_eq!(id, None);
                assert_eq!(code, "startup_failed");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_and_malformed_messages() {
        assert!(parse_line("not json").is_err());
        assert!(parse_line(r#"{"no":"type"}"#).is_err());
        assert!(parse_line(r#"{"type":"whatever"}"#).is_err());
    }
}
