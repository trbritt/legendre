//! Observation of simulation state.
//!
//! Two layers:
//!
//! - [`Observer`] is the solver-facing contract: called synchronously after
//!   every step, must return fast.
//! - [`AsyncObserver`] implements `Observer` by copying the state into a
//!   **preallocated snapshot ring** and handing it through a **bounded mpsc**
//!   to a **background tokio runtime**, where [`SnapshotSink`]s (Parquet,
//!   statistics, logging) do the slow work:
//!
//! ```text
//! solver thread                    background thread (tokio)
//! ─────────────                    ─────────────────────────
//! observe(step, t, &state)
//!   ├── free ring buffer? ──no──► skip (dropped_snapshots += 1)
//!   ├── copy_from(state)                 ▼
//!   └── try_send ──────bounded──► sink₁.consume, sink₂.consume, …
//!                 ◄───buffer──────┘ (returned to the free ring)
//! ```
//!
//! **The solver never blocks**: if all ring buffers are in flight (sinks
//! slower than snapshot cadence), the snapshot is *dropped* and counted —
//! backpressure degrades observation, never simulation. Buffers cycle
//! through the two channels, so after construction the pipeline allocates
//! nothing. A future synchronous checkpoint sink can opt into blocking
//! without changing this contract.

use crate::core::{
    state::State,
    storage::{Real, Scalar, StorageBackend},
};

/// The solver-facing observation contract.
///
/// `observe` receives the *current* grid and its epoch (0 at construction,
/// +1 per adaptive regrid), so geometry-writing observers can follow grid
/// changes by re-emitting statics when the epoch bumps. Observers that
/// don't care about geometry (progress bars, the async snapshot ring)
/// implement the trait generically over `G` and ignore both arguments.
pub trait Observer<G, T: Scalar, S: StorageBackend<T>>: Send {
    /// Called after each completed step. Must not block on IO; copy what you
    /// need and hand off.
    fn observe(&mut self, step: u64, t: f64, epoch: u64, grid: &G, state: &State<T, S>);
}

/// A consumer of snapshots, run on the observer runtime — free to do slow,
/// blocking work (file IO, statistics, rendering).
pub trait SnapshotSink<T: Scalar, S: StorageBackend<T>>: Send {
    /// Process one snapshot (a deep copy of the state at `step`, time `t`).
    fn consume(&mut self, step: u64, t: f64, state: &State<T, S>);

    /// Called once after the last snapshot, when the pipeline shuts down.
    fn finish(&mut self) {}
}

struct SnapshotMsg<T: Scalar, S: StorageBackend<T>> {
    step: u64,
    t: f64,
    state: State<T, S>,
}

/// [`Observer`] implementation that moves snapshots through a preallocated
/// ring and a bounded channel to a background runtime (see module docs).
pub struct AsyncObserver<T: Scalar, S: StorageBackend<T> + 'static> {
    every: u64,
    work_tx: Option<tokio::sync::mpsc::Sender<SnapshotMsg<T, S>>>,
    free_rx: tokio::sync::mpsc::Receiver<State<T, S>>,
    worker: Option<std::thread::JoinHandle<()>>,
    dropped: u64,
}

impl<T: Real, S: StorageBackend<T> + 'static> AsyncObserver<T, S> {
    /// Build the pipeline. `buffers` is the snapshot ring (e.g. from
    /// [`crate::core::simulation::Simulation::snapshot_buffers`]); its
    /// length bounds both channels and thus the peak memory and lag of the
    /// pipeline. Snapshots fire on step 1 and every `every` steps.
    ///
    /// Dynamic dispatch on the type here is fine since this is meant to be
    /// an out of band (out of hot loop) code path for observability.
    ///
    /// # Panics
    ///
    /// Panics if the background observer thread cannot be spawned.
    #[must_use]
    pub fn new(
        every: u64,
        buffers: Vec<State<T, S>>,
        mut sinks: Vec<Box<dyn SnapshotSink<T, S>>>,
    ) -> Self {
        let depth = buffers.len().max(1);
        let (work_tx, mut work_rx) = tokio::sync::mpsc::channel::<SnapshotMsg<T, S>>(depth);
        let (free_tx, free_rx) = tokio::sync::mpsc::channel::<State<T, S>>(depth);
        for buf in buffers {
            free_tx
                .try_send(buf)
                .unwrap_or_else(|_| unreachable!("free ring sized to buffer count"));
        }

        // Put this on a managed OS thread to track hardware specs of observability
        // more easily than dissecting tokio for it
        let worker = std::thread::Builder::new()
            .name("legendre-observer".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .expect("observer runtime");
                rt.block_on(async move {
                    while let Some(msg) = work_rx.recv().await {
                        for sink in &mut sinks {
                            sink.consume(msg.step, msg.t, &msg.state);
                        }
                        // Ring invariant: total buffers == channel capacity,
                        // so returning one can never fail.
                        let _ = free_tx.send(msg.state).await;
                    }
                    for sink in &mut sinks {
                        sink.finish();
                    }
                });
            })
            .expect("spawning observer thread");

        Self {
            every,
            work_tx: Some(work_tx),
            free_rx,
            worker: Some(worker),
            dropped: 0,
        }
    }

    /// Snapshots skipped because every ring buffer was still in flight.
    #[must_use]
    pub const fn dropped_snapshots(&self) -> u64 {
        self.dropped
    }
}

/// Grid-agnostic: the preallocated snapshot ring assumes a fixed block
/// structure, so the async pipeline serves uniform/static grids; adaptive
/// runs use synchronous grid-aware observers instead.
impl<G, T: Real, S: StorageBackend<T> + 'static> Observer<G, T, S> for AsyncObserver<T, S> {
    fn observe(&mut self, step: u64, t: f64, _epoch: u64, _grid: &G, state: &State<T, S>) {
        if step != 1 && !step.is_multiple_of(self.every) {
            return;
        }
        let Ok(mut buf) = self.free_rx.try_recv() else {
            self.dropped += 1;
            return;
        };
        buf.copy_from(state);
        let msg = SnapshotMsg {
            step,
            t,
            state: buf,
        };
        if let Some(tx) = &self.work_tx
            && tx.try_send(msg).is_err()
        {
            // Can only happen if the worker died; count it and move on.
            self.dropped += 1;
        }
    }
}

impl<T: Scalar, S: StorageBackend<T> + 'static> Drop for AsyncObserver<T, S> {
    fn drop(&mut self) {
        // Closing the work channel lets the worker drain in-flight
        // snapshots, run sink finishers, and exit.
        self.work_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if self.dropped > 0 {
            tracing::warn!(
                dropped = self.dropped,
                "observer ring was saturated; snapshots were skipped"
            );
        }
    }
}
