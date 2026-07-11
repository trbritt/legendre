//! Output backends.
//!
//! Writers implement both the synchronous
//! [`crate::core::observer::Observer`] contract (small runs) and
//! [`crate::core::observer::SnapshotSink`] for the async pipeline
//! (snapshot ring → bounded mpsc → tokio runtime → sinks).

use std::fmt::Debug;

pub mod parquet;
pub mod progress;

/// Errors constructing or running an output backend.
#[derive(Debug, thiserror::Error)]
pub enum ObserverError {
    /// The backend does not support this spatial dimension.
    #[error("invalid dimension {0}")]
    InvalidDimension(usize),
    /// No fields were given to observe.
    #[error("no fields to observe")]
    NoValidFields,
    /// Filesystem failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Parquet encoding failure.
    #[error(transparent)]
    Parquet(#[from] ::parquet::errors::ParquetError),
    /// An internal invariant was violated.
    #[error("internal error: {0:?}")]
    Internal(&'static str),
}
