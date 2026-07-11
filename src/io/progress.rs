//! Live progress and statistics via indicatif.
//!
//! Two cooperating pieces sharing one [`ProgressBar`] handle (indicatif
//! bars are `Clone` + thread-safe):
//!
//! - [`ProgressObserver`] — a synchronous [`Observer`] that bumps the bar
//!   position every step. This is one atomic store per step (indicatif
//!   rate-limits terminal redraws internally), so it costs nothing even when
//!   the solver is floored.
//! - [`FieldStatsSink`] — a [`SnapshotSink`] running on the observer runtime
//!   that computes per-field statistics (min/max/mean, fraction of cells above
//!   a threshold) at snapshot cadence and writes them into the bar's message —
//!   so the expensive full-state reduction never touches the solver thread.
//!
//! For a dendritic-solidification nucleation run the bar reads like:
//!
//! ```text
//! ⠁ [00:01:12] ████████░░ 45231/187000 (628 steps/s, eta 3m 46s)
//!   t=723.7 | phi∈[-1.00,1.00] solid 23.4% | u∈[-0.70,0.09] ⟨u⟩=-0.512
//! ```

use crate::{
    core::{
        observer::{Observer, SnapshotSink},
        state::{FieldHandle, State},
        storage::StorageBackend,
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
};
use indicatif::{ProgressBar, ProgressStyle};

/// Build the shared progress bar for a run of `total_steps`.
///
/// # Panics
///
/// Panics only if the compiled-in style template is invalid (a bug, not a
/// runtime condition).
#[must_use]
pub fn progress_bar(total_steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(total_steps);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner} [{elapsed_precise}] {bar:32.cyan/dim} {pos}/{len} ({per_sec}, eta {eta})\n  {msg}",
        )
        .expect("static template"),
    );
    bar
}

/// Advances the bar one tick per solver step.
pub struct ProgressObserver {
    bar: ProgressBar,
}

impl ProgressObserver {
    /// Wrap a shared progress bar.
    #[must_use]
    pub const fn new(bar: ProgressBar) -> Self {
        Self { bar }
    }
}

impl<T: crate::core::storage::Scalar, S: StorageBackend<T>> Observer<T, S> for ProgressObserver {
    fn observe(&mut self, step: u64, _t: f64, _state: &State<T, S>) {
        self.bar.set_position(step);
    }
}

/// Which statistics to report for one field.
pub struct FieldStat {
    /// Display name of the field.
    pub name: &'static str,
    /// Handle of the field to reduce over.
    pub handle: FieldHandle<f64>,
    /// If set, additionally report the fraction of cells strictly above
    /// this threshold (e.g. `Some(0.0)` on a phase field = solid fraction).
    pub fraction_above: Option<f64>,
}

/// Computes interior-cell statistics per snapshot and publishes them to the
/// progress bar message (and to `tracing` at info level, for headless runs).
pub struct FieldStatsSink<const D: usize = 2> {
    grid: CartesianGrid<D>,
    stats: Vec<FieldStat>,
    bar: ProgressBar,
}

impl<const D: usize> FieldStatsSink<D> {
    /// Reduce `stats` over `grid` at snapshot cadence, publishing to `bar`.
    #[must_use]
    pub const fn new(grid: CartesianGrid<D>, stats: Vec<FieldStat>, bar: ProgressBar) -> Self {
        Self { grid, stats, bar }
    }
}

impl<S: StorageBackend<f64>, const D: usize> SnapshotSink<f64, S> for FieldStatsSink<D> {
    fn consume(&mut self, _step: u64, t: f64, state: &State<f64, S>) {
        use std::fmt::Write as _;
        let mut msg = format!("t={t:.1}");
        for stat in &self.stats {
            let (mut lo, mut hi, mut sum, mut above, mut n) =
                (f64::MAX, f64::MIN, 0.0f64, 0u64, 0u64);
            for b in 0..self.grid.num_blocks() {
                let v = state.view(&self.grid, BlockId(b as u32), stat.handle);
                for_each_interior(self.grid.block_cells(), |idx| {
                    let x = v.get(idx);
                    lo = lo.min(x);
                    hi = hi.max(x);
                    sum += x;
                    n += 1;
                    if let Some(thr) = stat.fraction_above
                        && x > thr
                    {
                        above += 1;
                    }
                });
            }
            let mean = sum / n.max(1) as f64;
            let _ = write!(
                msg,
                " | {name}∈[{lo:.2},{hi:.2}] ⟨{name}⟩={mean:.3}",
                name = stat.name
            );
            if let Some(thr) = stat.fraction_above {
                let _ = write!(
                    msg,
                    " frac>{thr:.0}: {:.1}%",
                    100.0 * above as f64 / n.max(1) as f64
                );
            }
            tracing::info!(field = stat.name, t, lo, hi, mean, "snapshot stats");
        }
        self.bar.set_message(msg);
    }

    fn finish(&mut self) {
        self.bar.finish_with_message("done");
    }
}
