//! The [`Stencil`] abstraction.
//!
//! A stencil is the *executable* form of a discretized operator on one
//! block: it reads a ghost-filled input view and writes interior cells of an
//! output view. Operators (Laplacian, gradient, …) remain purely
//! mathematical tags; different
//! [`Discretizes`](crate::discretization::operators::Discretizes) policies
//! swap in different stencil implementations without any operator or model
//! interface changing.
//!
//! **Ownership/bounds decisions:**
//! - `Stencil<G>` is generic over the grid so an implementation can be written
//!   per grid family (a Cartesian 5-point Laplacian, a quadtree FV Laplacian).
//!   Under AMR the stencil implementation is where coarse–fine interface
//!   handling lives; the operator tag never learns about it.
//! - `apply` is generic over the arithmetic type, bounded by
//!   [`crate::core::storage::Real`], and monomorphizes into a flat loop over
//!   the block. Stencils hold no buffers and cannot allocate; anything
//!   temporary comes from the caller's scratch.
//! - Grid geometry (spacing) is read *at apply time* from `(grid, block)`,
//!   never baked in at construction, so one stencil value serves all refinement
//!   levels.
//!
//! The meat of the design is really in the stencil here. The width of the
//! ghost is not only relevant for boundary conditions of the simulation region,
//! but it is a direct manifestation of the physics required and is a measure
//! of the locality of the data. For example, central finite difference
//! calculations require a neighbor in each direction, so the ghost is 1 for
//! that stencil. If an operator requires next nearest neighbors, e.g.
//! Metropolis algorithms for spin flips on a lattice, then the stencil's ghost
//! width must be 2.
//!
//! [`Model::fill_ghosts`](crate::physics::model::Model::fill_ghosts) is not
//! only a boundary-condition hook — it's a serial, exclusive-access, pre-RHS
//! phase (`&mut State`, called by the integrator before every block-parallel
//! evaluation) that runs at exactly the right cadence for a nonlocal term.
//! The pattern is: materialize the nonlocal quantity into an auxiliary field
//! during that phase, then read it locally in `rhs_block` like any other
//! field.
//!
//! The design of this local stencil means that any data required for evaluating
//! operators or the RHS is guaranteed to be duplicated in a given slab, so by
//! the time that the scheduler dispatches work (perhaps in parallel), all
//! necessary data for evaluating a slab is present in that slab, and no
//! communications or queries are required to fetch grid data from within an
//! individual slab.
//!
//! That being said, nonlocal terms, e.g. ∂φ/∂t = (local stuff) + (K∗φ)(x),
//! can actually still be expressed in this framework!! The trick is that before
//! scheduled (potentially parallel) work, there is a single global pre-RHS
//! phase that has view of _all_ data and as such is a natural fit for
//! evaluating global terms.
//!
//! 1. Register an auxiliary field for the materialized term: self.conv =
//!    `builder.register_static("conv`", 0). `register_static` is correct here; it
//!    means the field gets zero-length tendency slabs, so the integrator's axpy
//!    never touches it. It's not "static" in the time-invariant sense; it's
//!    "carries no tendency," which is exactly what a diagnostic/materialized
//!    field is.
//! 2. In `fill_ghosts`, after the usual halo exchange, compute the global term
//!    from φ across all blocks — one has &mut State, exclusive, every block
//!    readable and writable — and scatter the result into conv's slabs.
//! 3. In `rhs_block`, read state.view(ctx.block, self.conv) — a purely local
//!    read. The block-parallel phase never knows the term was nonlocal.
//!
//! This covers global scalar functionals, convolutions, and spectral elliptic
//! solves, but is limited on implicit solves (CG/multigrid, implicit
//! timestepping), which requires more work.

use crate::{
    core::storage::Real,
    geometry::grid::{BlockId, Grid},
};

/// The executable form of a discretized operator on one block.
pub trait Stencil<G: Grid>: Send + Sync {
    /// Ghost-ring width this stencil reads. Fields it is applied to must be
    /// registered with at least this ghost width; the framework checks at
    /// setup, not in the hot loop.
    fn ghost_width(&self) -> u32;

    /// Apply on one block: read `input` (interior + ghosts), write the
    /// interior of `output`.
    fn apply<T: Real>(
        &self,
        grid: &G,
        block: BlockId,
        input: G::View<'_, T>,
        output: &mut G::ViewMut<'_, T>,
    );
}
