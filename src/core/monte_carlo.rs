//! A generic Monte Carlo path ensemble.
//!
//! A Monte Carlo ensemble is just a [`Simulation`] in which every interior
//! cell is an **independent sample path** — a 0-dimensional SDE per cell, no
//! spatial coupling (see [`crate::physics::stochastic_vol`]). This module adds
//! the two things such a run needs on top of the ordinary simulation loop, and
//! nothing model-specific:
//!
//! - **[`MonteCarlo`]** wraps any simulation — any model, integrator,
//!   scheduler, and allocator — and drives it forward, leaving field setup and
//!   observation to the wrapped [`Simulation`] (reached through
//!   [`MonteCarlo::simulation_mut`]).
//! - **[`ensemble_stats`]** reduces a per-path scalar functional over the
//!   whole ensemble to summary [`Stats`] (mean, variance, extrema) — the
//!   terminal-payoff reduction, expressed once for every model.
//!
//! The ensemble is entirely trait-generic; the only structural commitment is
//! "one path per cell", so it is fixed to the [`CartesianGrid`] family and
//! generic over everything else.

use crate::{
    core::{
        scheduler::Scheduler,
        simulation::Simulation,
        state::{FieldHandle, State},
        storage::{Allocator, Scalar, StorageBackend},
    },
    geometry::{
        GridError,
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::Integrator,
    physics::model::Model,
};

/// A 1-D grid of `paths` independent Monte Carlo paths, tiled by `block`-sized
/// blocks (the parallel work unit). The domain coordinates are arbitrary —
/// paths do not couple — so it uses the unit interval.
///
/// # Errors
///
/// Returns a [`GridError`] if `paths` is not a positive multiple of `block`.
pub fn path_grid(paths: usize, block: usize) -> Result<CartesianGrid<1>, GridError> {
    CartesianGrid::new([paths], [block], [0.0], [1.0])
}

/// Summary statistics of a per-path functional over an ensemble.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Number of paths reduced.
    pub count: u64,
    /// Ensemble mean.
    pub mean: f64,
    /// Ensemble variance (population, i.e. divided by `count`).
    pub variance: f64,
    /// Smallest per-path value.
    pub min: f64,
    /// Largest per-path value.
    pub max: f64,
}

impl Stats {
    /// Ensemble standard deviation.
    #[must_use]
    pub fn std(&self) -> f64 {
        self.variance.sqrt()
    }
}

/// One path's terminal state: field access at a single interior cell, handed
/// to the functional reduced by [`ensemble_stats`].
pub struct PathSample<'a, T: Scalar, S: StorageBackend<T>, const D: usize> {
    grid: &'a CartesianGrid<D>,
    state: &'a State<T, S>,
    block: BlockId,
    idx: [isize; D],
}

impl<T: Scalar, S: StorageBackend<T>, const D: usize> PathSample<'_, T, S, D> {
    /// This path's value of the field `handle`.
    #[inline]
    #[must_use]
    pub fn get(&self, handle: FieldHandle<T>) -> T {
        self.state.view(self.grid, self.block, handle).get(self.idx)
    }
}

/// Reduce a per-path scalar functional over an ensemble to summary [`Stats`].
///
/// `f` is evaluated once per interior cell (path) of `state`; it reads that
/// path's fields through its [`PathSample`] and returns the payoff whose
/// distribution is summarized — e.g. terminal wealth `cash + q·S − α·q²`.
pub fn ensemble_stats<T, S, const D: usize>(
    grid: &CartesianGrid<D>,
    state: &State<T, S>,
    f: impl Fn(&PathSample<'_, T, S, D>) -> f64,
) -> Stats
where
    T: Scalar,
    S: StorageBackend<T>,
{
    let (mut sum, mut sum2, mut count) = (0.0f64, 0.0f64, 0u64);
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        for_each_interior(grid.block_cells(), |idx| {
            let value = f(&PathSample {
                grid,
                state,
                block,
                idx,
            });
            sum += value;
            sum2 = value.mul_add(value, sum2);
            count += 1;
            min = min.min(value);
            max = max.max(value);
        });
    }
    let mean = sum / count as f64;
    // max(0) guards the tiny negative from catastrophic cancellation when the
    // variance is ~0 (e.g. a degenerate, deterministic functional).
    let variance = mean.mul_add(-mean, sum2 / count as f64).max(0.0);
    Stats {
        count,
        mean,
        variance,
        min,
        max,
    }
}

/// A Monte Carlo ensemble: a thin driver around a [`Simulation`] whose cells
/// are independent paths.
///
/// Generic over the model, discretization, integrator, scheduler, and
/// allocator; construct the simulation as usual (typically over a
/// [`path_grid`]) and wrap it.
pub struct MonteCarlo<const D: usize, Disc, M, I, Sch, A>
where
    M: Model<CartesianGrid<D>, Disc>,
    I: Integrator<CartesianGrid<D>, Disc, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
{
    sim: Simulation<CartesianGrid<D>, Disc, M, I, Sch, A>,
}

impl<const D: usize, Disc, M, I, Sch, A> MonteCarlo<D, Disc, M, I, Sch, A>
where
    M: Model<CartesianGrid<D>, Disc>,
    I: Integrator<CartesianGrid<D>, Disc, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
{
    /// Wrap a constructed simulation as a path ensemble.
    #[must_use]
    pub const fn new(sim: Simulation<CartesianGrid<D>, Disc, M, I, Sch, A>) -> Self {
        Self { sim }
    }

    /// The wrapped simulation (read access: grid, state, model).
    #[must_use]
    pub const fn simulation(&self) -> &Simulation<CartesianGrid<D>, Disc, M, I, Sch, A> {
        &self.sim
    }

    /// The wrapped simulation mutably — for initial conditions, observers, and
    /// anything else the ensemble does not itself provide.
    pub const fn simulation_mut(
        &mut self,
    ) -> &mut Simulation<CartesianGrid<D>, Disc, M, I, Sch, A> {
        &mut self.sim
    }

    /// Advance every path `steps` steps of size `dt`.
    pub fn run(&mut self, steps: usize, dt: f64) {
        for _ in 0..steps {
            self.sim.step(dt);
        }
    }

    /// Reduce a per-path functional over the ensemble to summary [`Stats`]
    /// (see [`ensemble_stats`]).
    pub fn stats(&self, f: impl Fn(&PathSample<'_, M::Scalar, A::Storage, D>) -> f64) -> Stats {
        ensemble_stats(self.sim.grid(), self.sim.state(), f)
    }
}
