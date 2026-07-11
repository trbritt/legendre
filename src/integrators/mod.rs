//! Time integration.
//!
//! - The integrator owns the timestep: it requests ghost fills, drives
//!   `Model::rhs_block` through the scheduler, and updates state. Models never
//!   see dt.
//! - **Integrators never index space.** Every state-shaped buffer is
//!   slab-congruent with the state (see [`crate::core::state`]), so stage
//!   combinations are pure vector-space operations (`axpy_with`,
//!   `copy_from_with`) — themselves scheduler-dispatched, since at large volume
//!   they are memory-bound over the whole state. An integrator therefore works
//!   unchanged on any grid, dimension, or discretization.
//! - Stage buffers are declared via [`Integrator::stage_layout`] and
//!   allocated once by the simulation; scratch comes from the worker-pinned
//!   [`ScratchPool`]. Integrators allocate nothing.
//! - **Noise composes with any explicit scheme**: the model's amplitude b(Y) is
//!   evaluated at the pre-update state, and `√dt·b∘ξ` is added after the drift
//!   update (exact for additive noise; Euler–Maruyama drift coupling
//!   otherwise). ξ is counter-based and schedule-independent if the model needs
//!   it.

mod euler;
mod euler_maruyama;
mod rk4;

use crate::{
    core::{scheduler::Scheduler, scratch::ScratchPool, state::State, storage::StorageBackend},
    geometry::grid::Grid,
    physics::model::{Model, RhsContext},
};

pub use euler::ForwardEuler;
pub use euler_maruyama::EulerMaruyama;
pub use rk4::RungeKutta4;

/// How many state-shaped buffers a scheme needs, split by role.
///
/// `tendency` buffers hold dY/dt or noise amplitudes and carry **no storage
/// for static fields**; `stage_state` buffers hold intermediate *states*
/// (RK4's `y_tmp`) that models read like the real state, so they carry every
/// field. The split is what keeps a static field (e.g. a phase-field model's
/// grain orientation θ₀) from being replicated across k-buffers at scale.
#[derive(Debug, Clone, Copy)]
pub struct StageLayout {
    /// Number of tendency buffers (dY/dt, noise amplitudes).
    pub tendency: usize,
    /// Number of full stage-state buffers (intermediate states).
    pub stage_state: usize,
}

/// A time-integration scheme; see the module docs for the contract.
pub trait Integrator<G: Grid, D>: Send + Sync {
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
        M: Model<G, D>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler;
}

/// Fill ghosts of `state`, then evaluate the model RHS into `out`, both
/// dispatched per block. The building block shared by all explicit schemes.
#[allow(clippy::too_many_arguments)]
fn eval_rhs<G, D, M, S, Sch>(
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

    // dY/dt, one block per work item; `state` is shared read-only.
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
            model.rhs_block(&ctx, state, &mut storage.bind_mut(layout), &mut *sc);
        },
    );
}

/// Zero `amp` and let the model write its noise amplitude b(Y) into it,
/// reading `state` (which must be the pre-update state).
#[allow(clippy::too_many_arguments)]
fn eval_noise<G, D, M, S, Sch>(
    model: &M,
    grid: &G,
    disc: &D,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &State<M::Scalar, S>,
    amp: &mut State<M::Scalar, S>,
    t: f64,
) where
    G: Grid,
    M: Model<G, D>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D: Sync,
{
    amp.fill_zero_with(scheduler);
    let (layout, blocks) = amp.split_blocks_mut();
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
            model.noise_block(&ctx, state, &mut storage.bind_mut(layout), &mut *sc);
        },
    );
}
