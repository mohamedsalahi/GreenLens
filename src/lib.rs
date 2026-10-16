//! GreenLens: plant disease identification from leaf images.
//!
//! The desktop application lives in `app`, the python worker process
//! handling in `worker`. The binary in main.rs is a thin wrapper so tests
//! can exercise the same code the application runs.

pub mod app;
pub mod worker;
