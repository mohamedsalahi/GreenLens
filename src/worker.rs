//! Management of the Python inference worker process.
//!
//! The worker is a separate Python process that loads the trained model once
//! and answers JSON-lines requests on stdin/stdout.
