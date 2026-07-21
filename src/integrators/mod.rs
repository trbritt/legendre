//! Time integration.
//!
//! - The integrator owns the timestep: it requests ghost fills, drives
//!   `Model::vector_field_block` through the scheduler, and updates state.
//!   Models never see dt — each driver's kernel
//!   ([`crate::core::driver::DriverKind`]) applies the measure-correct
//!   scaling: dt for the time field, √dt·ξ per Wiener field.
//! - **Integrators are implemented per driver set.** The trait is
//!   `Integrator<G, D, N: DriverSet>`; deterministic schemes implement it
//!   for [`NoNoise`](crate::physics::model::NoNoise) only, so handing RK4 a
//!   stochastic model is a compile error rather than a silently dropped
//!   noise term. [`EulerMaruyama`] is implemented for every driver set —
//!   and, because application is delegated to the driver kernels, it never
//!   names a driver kind: new kinds work through it unchanged.
//! - **Integrators never index space.** Every state-shaped buffer is
//!   slab-congruent with the state (see [`crate::core::state`]), so stage
//!   combinations are pure vector-space operations (`axpy_with`,
//!   `copy_from_with`, `apply_driver_with`) — themselves scheduler-dispatched,
//!   since at large volume they are memory-bound over the whole state. An
//!   integrator therefore works unchanged on any grid, dimension, or
//!   discretization.
//! - Stage buffers are declared via [`Integrator::stage_layout`] and
//!   allocated once by the simulation; scratch comes from the worker-pinned
//!   [`ScratchPool`]. Integrators allocate nothing.
//! - **Itô convention**: Wiener fields are evaluated at the pre-update
//!   state, and each driver's increment uses one deviate per cell,
//!   broadcast across the fields that driver moves — counter-based and
//!   schedule-independent.

mod euler;
mod euler_maruyama;
mod rk4;
mod subcycling;

use crate::{
    core::{scheduler::Scheduler, scratch::ScratchPool, state::State, storage::StorageBackend},
    geometry::grid::Grid,
    physics::model::{Driver, DriverSet, Model, RhsContext},
};

pub use euler::ForwardEuler;
pub use euler_maruyama::EulerMaruyama;
pub use rk4::RungeKutta4;
pub use subcycling::Subcycling;

/// One stage buffer a scheme requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// A tendency buffer conjugate to one driver: carries storage for
    /// exactly the fields that driver moves (declared via
    /// [`crate::core::state::StateBuilder::register_driven`]).
    /// `Tendency(Driver::Time)` is the classic dY/dt buffer; a stochastic
    /// driver's buffer holds its amplitudes.
    Tendency(Driver),
    /// A full stage-state buffer (an intermediate state models read like
    /// the real state, e.g. RK4's `y_tmp`): carries every field.
    State,
}

/// The stage buffers a scheme needs, in [`Integrator::step`] order.
///
/// `Simulation` maps each kind to its allocation mechanically, so a new
/// driver kind needs no allocation plumbing. The per-kind storage split is
/// what keeps a static field (e.g. a grain-orientation map θ₀) out of
/// k-buffers, and a single-field noise term from paying whole-state
/// traffic, at scale.
#[derive(Debug, Clone)]
pub struct StageLayout {
    /// Requested buffers, in `stages`-slice order.
    pub stages: Vec<StageKind>,
}

/// A time-integration scheme for models with driver set `N`; see the
/// module docs for the contract.
pub trait Integrator<G: Grid, D, N: DriverSet>: Send + Sync {
    /// Buffers this scheme needs; called at setup and after each regrid.
    /// Receives the grid so grid-dependent schemes (subcycling needs one
    /// time-interpolation snapshot per refinement level) can size their
    /// buffers to the current hierarchy.
    fn stage_layout(&self, grid: &G) -> StageLayout;

    /// The timestep this scheme would take on `grid`, from the model's
    /// stability law. The default is the most restrictive cell's stable dt
    /// (the finest level under AMR) — correct for a global-dt scheme that
    /// applies one dt everywhere. A subcycling scheme overrides this to
    /// return the *coarsest* level's dt and refine in time per level.
    fn suggested_dt<M>(&self, model: &M, grid: &G) -> Option<f64>
    where
        M: Model<G, D, Drivers = N>,
    {
        model.stable_dt(grid.finest_spacing())
    }

    /// Advance `state` from `t` to `t + dt`. `stages` are the pre-allocated
    /// buffers from [`Integrator::stage_layout`]; `pool` is the
    /// worker-pinned scratch.
    #[allow(clippy::too_many_arguments)]
    fn step<M, S, Sch>(
        &self,
        model: &M,
        grid: &G,
        disc: &D,
        scheduler: &Sch,
        pool: &ScratchPool<M::Scalar, S>,
        state: &mut State<M::Scalar, S>,
        stages: &mut [State<M::Scalar, S>],
        t: f64,
        dt: f64,
    ) where
        M: Model<G, D, Drivers = N>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler;
}

/// Dispatch one driver's vector field into `out`. If `level` is `Some`,
/// only blocks at that refinement level are evaluated (the rest are left
/// untouched) — the primitive Berger–Oliger subcycling steps one level at
/// a time on; `None` evaluates every block, the global-dt path.
#[allow(clippy::too_many_arguments)]
fn dispatch_driver<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    out: &mut State<M::Scalar, S>,
    t: f64,
    driver: Driver,
    level: Option<u8>,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    let (layout, blocks) = out.split_blocks_mut();
    scheduler.for_each_block_mut(
        blocks,
        || pool.checkout(),
        |block, storage, sc| {
            if level.is_some_and(|l| grid.level(block) != l) {
                return;
            }
            let ctx = RhsContext {
                grid,
                disc,
                block,
                t,
            };
            model.vector_field_block(driver, &ctx, state, &mut storage.bind_mut(layout), &mut *sc);
        },
    );
}

/// Fill ghosts of `state`, then evaluate the time (drift) field into `out`,
/// both dispatched per block. The building block shared by all explicit
/// schemes.
#[allow(clippy::too_many_arguments)]
fn eval_drift<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &mut State<M::Scalar, S>,
    out: &mut State<M::Scalar, S>,
    t: f64,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    // Halo exchange + physical boundary conditions, model-directed.
    // This is the point at which we have global and internally consistent
    // state for the current time at which we could evaluate nonlocal terms
    // like convolutions, spectral range couplings, etc.
    model.fill_ghosts(grid, state, t);
    dispatch_driver(
        model,
        grid,
        disc,
        scheduler,
        pool,
        state,
        out,
        t,
        Driver::Time,
        None,
    );
}

/// Evaluate the drift on **one level's** blocks into `out`, reading the
/// already-ghost-filled `state`. Unlike [`eval_drift`] this does *not*
/// fill ghosts — under subcycling the driver fills them separately, with
/// time interpolation from the coarser level. The Berger–Oliger per-level
/// RHS evaluation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_drift_level<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    out: &mut State<M::Scalar, S>,
    t: f64,
    level: u8,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    dispatch_driver(
        model,
        grid,
        disc,
        scheduler,
        pool,
        state,
        out,
        t,
        Driver::Time,
        Some(level),
    );
}

/// [`eval_tendency`] restricted to one level's blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_tendency_level<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scalar, S>,
    t: f64,
    driver: Driver,
    level: u8,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    eval_tendency_impl(
        model,
        grid,
        disc,
        scheduler,
        pool,
        state,
        amp,
        t,
        driver,
        Some(level),
    );
}

/// Zero `amp` and let the model write `driver`'s amplitude field into it,
/// reading `state` — which must be the pre-update, ghost-filled state
/// (Itô: [`eval_drift`] has already run at the same `t`). `amp` carries
/// storage only for the fields `driver` moves, so the zero pass and the
/// evaluation touch exactly that memory.
#[allow(clippy::too_many_arguments)]
fn eval_tendency<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scalar, S>,
    t: f64,
    driver: Driver,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    eval_tendency_impl(
        model, grid, disc, scheduler, pool, state, amp, t, driver, None,
    );
}

#[allow(clippy::too_many_arguments)]
fn eval_tendency_impl<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scalar, S>,
    t: f64,
    driver: Driver,
    level: Option<u8>,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    // Zeroing is fused into the evaluation dispatch: each work item resets
    // its own block's slabs just before the model fills them, so the
    // buffer is touched once (cache-hot), with one barrier instead of two.
    let (layout, blocks) = amp.split_blocks_mut();
    scheduler.for_each_block_mut(
        blocks,
        || pool.checkout(),
        |block, storage, sc| {
            if level.is_some_and(|l| grid.level(block) != l) {
                return;
            }
            storage.fill_zero();
            let ctx = RhsContext {
                grid,
                disc,
                block,
                t,
            };
            model.vector_field_block(driver, &ctx, state, &mut storage.bind_mut(layout), &mut *sc);
        },
    );
}
