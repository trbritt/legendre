//! The [`Model`] trait: pure mathematics over discretization policies.
//!
//! - **`Model::rhs`, not `Model::step`**. A model evaluates dY/dt into an
//!   output buffer and never mutates simulation state; the integrator owns
//!   state updates, which is what makes multi-stage schemes trivial.
//!
//! - **Per-block evaluation.** The scheduler drives `rhs_block` once per block;
//!   the model sees one block's output and the whole (read-only) state. This is
//!   the contract that makes uniform grids and AMR identical to the model.
//!
//! - **`Model<G, D>` with bounds at the impl.** The trait puts no requirements
//!   on the discretization `D`; each model impl demands exactly the operators
//!   it uses (e.g. `where D: Discretizes<G, Laplacian>`), so models are generic
//!   over schemes and schemes over models.
//!
//! - **Stochastic terms are split from the drift.** `noise_block` writes the
//!   noise *amplitude* b(Y); the integrator supplies the random increment with
//!   the correct dt-scaling (√dt for Euler–Maruyama). This keeps the timestep
//!   scaling out of model code.
//!
//! - **Boundary conditions belong to the model.** `fill_ghosts` runs once per
//!   RHS evaluation over the whole state; models delegate the interior halo
//!   exchange and standard physical conditions to grid-family helpers (e.g.
//!   [`crate::geometry::cartesian::fill_ghosts_mirror`]). Grid helpers own the
//!   cross-block copies because they are topology, not physics.

use crate::{
    core::{
        scratch::{Scratch, ScratchSpec},
        state::{BlockStateMut, State, StateBuilder},
        storage::{Real, StorageBackend},
    },
    geometry::grid::{BlockId, Grid},
};

/// Everything a model may consult while evaluating one block's RHS.
/// Deliberately read-only and allocation-free.
pub struct RhsContext<'a, G: Grid, D> {
    /// The grid the simulation runs on.
    pub grid: &'a G,
    /// The discretization policy (build stencils from it).
    pub disc: &'a D,
    /// The block being evaluated.
    pub block: BlockId,
    /// Evaluation time (the stage time for multi-stage schemes).
    pub t: f64,
}

/// A system of PDEs; see the module docs for the contract.
pub trait Model<G: Grid, D>: Send + Sync {
    /// Arithmetic type of this model's fields.
    type Scalar: Real;

    /// Declare fields (name + ghost width = max stencil support) and stash
    /// the returned handles. Called exactly once, before allocation; the
    /// only mutating model method.
    fn register_fields(&mut self, builder: &mut StateBuilder<Self::Scalar>);

    /// Per-worker scratch requirements (block-sized slabs).
    fn scratch_spec(&self, _grid: &G) -> ScratchSpec {
        ScratchSpec::NONE
    }

    /// Make every ghost cell of every field consistent: interior halos and
    /// physical boundary conditions. Called by the integrator before each
    /// RHS evaluation with the evaluation time `t` (stage time for
    /// multi-stage schemes), so time-dependent boundary forcing is
    /// expressible. Models typically forward to a grid-family helper per
    /// field (e.g. `fill_ghosts_mirror(grid, state, self.phi)`).
    fn fill_ghosts<S: StorageBackend<Self::Scalar>>(
        &self,
        _grid: &G,
        _state: &mut State<Self::Scalar, S>,
        _t: f64,
    ) {
    }

    /// Evaluate dY/dt on one block into `out` (interior cells only), reading
    /// the ghost-filled `state`. Must not touch any other block's output.
    fn rhs_block<S: StorageBackend<Self::Scalar>>(
        &self,
        ctx: &RhsContext<'_, G, D>,
        state: &State<Self::Scalar, S>,
        out: &mut BlockStateMut<'_, Self::Scalar, S>,
        scratch: &mut Scratch<Self::Scalar, S>,
    );

    /// Whether this model has a stochastic term. Gates the integrator's
    /// noise pass; when `true`, [`Model::noise_block`] must write amplitudes.
    fn has_noise(&self) -> bool {
        false
    }

    /// Write the noise *amplitude* b(Y) on one block into `out` (interior
    /// cells of noisy fields only; `out` arrives zeroed). The integrator
    /// turns this into b(Y)·√dt·ξ with deterministic, schedule-independent ξ.
    fn noise_block<S: StorageBackend<Self::Scalar>>(
        &self,
        _ctx: &RhsContext<'_, G, D>,
        _state: &State<Self::Scalar, S>,
        _out: &mut BlockStateMut<'_, Self::Scalar, S>,
        _scratch: &mut Scratch<Self::Scalar, S>,
    ) {
    }

    /// Largest stable explicit timestep, if the model knows one (e.g.
    /// `0.25·h²/D` for phase fields). Advisory; drivers may use it to pick dt.
    fn stable_dt(&self, _grid: &G) -> Option<f64> {
        None
    }
}
