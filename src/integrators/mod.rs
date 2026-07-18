//! Time integration.
//!
//! - The integrator owns the timestep: it requests ghost fills, drives
//!   `Model::vector_field_block` through the scheduler, and updates state.
//!   Models never see dt — the integrator applies the measure-correct
//!   scaling per [`Driver`]: dt for the time field, √dt·ξ per Wiener field.
//! - **Integrators are implemented per driver set.** The trait is
//!   `Integrator<G, D, N: NoiseSpec>`; deterministic schemes implement it
//!   for [`NoNoise`](crate::physics::model::NoNoise) only, so handing RK4 a
//!   stochastic model is a compile error rather than a silently dropped
//!   noise term. [`EulerMaruyama`] is implemented for every driver set and
//!   degenerates to forward Euler at `NoNoise`.
//! - **Integrators never index space.** Every state-shaped buffer is
//!   slab-congruent with the state (see [`crate::core::state`]), so stage
//!   combinations are pure vector-space operations (`axpy_with`,
//!   `copy_from_with`, `add_wiener_with`) — themselves scheduler-dispatched,
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

use crate::{
    core::{scheduler::Scheduler, scratch::ScratchPool, state::State, storage::StorageBackend},
    geometry::grid::Grid,
    physics::model::{Driver, Model, NoiseSpec, RhsContext},
};

pub use euler::ForwardEuler;
pub use euler_maruyama::EulerMaruyama;
pub use rk4::RungeKutta4;

/// How many state-shaped buffers a scheme needs, split by role.
///
/// `tendency` buffers hold vector-field evaluations (dY/dt, Wiener
/// amplitudes) and carry **no storage for static fields**; `stage_state`
/// buffers hold intermediate *states* (RK4's `y_tmp`) that models read like
/// the real state, so they carry every field. The split is what keeps a
/// static field (e.g. a phase-field model's grain orientation θ₀) from
/// being replicated across k-buffers at scale.
#[derive(Debug, Clone, Copy)]
pub struct StageLayout {
    /// Number of tendency buffers (dY/dt, Wiener amplitudes).
    pub tendency: usize,
    /// Number of full stage-state buffers (intermediate states).
    pub stage_state: usize,
}

/// A time-integration scheme for models with driver set `N`; see the
/// module docs for the contract.
pub trait Integrator<G: Grid, D, N: NoiseSpec>: Send + Sync {
    /// Buffers this scheme needs. `stages` passed to [`Integrator::step`]
    /// holds the tendency buffers first, stage-state buffers after.
    fn stage_layout(&self) -> StageLayout;

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
        M: Model<G, D, Noise = N>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler;
}

/// Dispatch one driver's vector field over all blocks into `out`.
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
    );
}

/// Zero `amp` and let the model write Wiener field `j`'s amplitude into it,
/// reading `state` — which must be the pre-update, ghost-filled state
/// (Itô: [`eval_drift`] has already run at the same `t`).
#[allow(clippy::too_many_arguments)]
fn eval_wiener<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scalar, S>,
    t: f64,
    j: usize,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    amp.fill_zero_with(scheduler);
    dispatch_driver(
        model,
        grid,
        disc,
        scheduler,
        pool,
        state,
        amp,
        t,
        Driver::Wiener(j),
    );
}
