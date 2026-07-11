//! # legendre — a PDE simulation framework
//!
//! `legendre` solves systems of time-dependent partial differential
//! equations — deterministic or stochastic, in any spatial dimension — on
//! block-decomposed structured grids. Heat transport, reaction–diffusion,
//! and phase-field solidification are *models* here, not the framework:
//! they are all expressed against the same small trait surface.
//!
//! The architecture follows four principles:
//!
//! 1. **Mathematical objects own no execution.** A [`physics::model::Model`]
//!    cannot spawn threads, a [`geometry::grid::Grid`] cannot write files, a
//!    [`discretization::stencil::Stencil`] cannot allocate.
//! 2. **Execution is scheduler-driven.** Everything runs because the
//!    [`core::scheduler::Scheduler`] requests it. No trait in this crate
//!    mentions Rayon; only concrete schedulers do.
//! 3. **Storage is separate from views.** Fields are typed views into a
//!    [`core::storage::StorageBackend`]; allocation happens once, up front,
//!    through an [`core::storage::Allocator`].
//! 4. **Numerical methods are policies.** A grid plus a discretization
//!    policy yields operator realizations
//!    ([`discretization::stencil::Stencil`]s); models state *what* operators
//!    they need and never learn *how* they were realized.
//!
//! The fundamental unit of computation is the **block**, not the grid. Even a
//! uniform Cartesian grid is a collection of fixed-size blocks: this gives
//! cache locality, natural parallel work units, localized halo exchange, and
//! an execution model that is unchanged when adaptive refinement arrives.
//!
//! ## Ownership graph
//!
//! ```text
//! Simulation
//! ├── Scheduler        (how blocks are dispatched)
//! ├── State            (Storage + field views, ghost-inclusive slabs)
//! ├── Grid             (topology: blocks, indices, views)
//! ├── Discretization   (policy: builds Stencils for Operators)
//! ├── Model            (mathematics: rhs, never mutates state)
//! ├── Integrator       (advances State using Model::rhs)
//! └── Observers        (async output; never block the solver)
//! ```
//!
//! ## A complete model
//!
//! The dimension-generic heat equation `∂u/∂t = κ∇²u` with no-flux
//! boundaries, wired to a grid, integrator, and scheduler:
//!
//! ```
//! use legendre::core::{
//!     scheduler::SerialScheduler,
//!     scratch::Scratch,
//!     simulation::Simulation,
//!     state::{BlockStateMut, FieldHandle, State, StateBuilder},
//!     storage::{StorageBackend, SystemAllocator},
//! };
//! use legendre::discretization::{
//!     finite_difference::FiniteDifference,
//!     operators::{Discretizes, Laplacian},
//!     stencil::Stencil,
//! };
//! use legendre::geometry::cartesian::{CartesianGrid, fill_ghosts_mirror, for_each_interior};
//! use legendre::integrators::ForwardEuler;
//! use legendre::physics::model::{Model, RhsContext};
//!
//! struct Heat<const D: usize> {
//!     kappa: f64,
//!     u: Option<FieldHandle<f64>>,
//! }
//!
//! impl<const D: usize, P> Model<CartesianGrid<D>, P> for Heat<D>
//! where
//!     P: Discretizes<CartesianGrid<D>, Laplacian>,
//! {
//!     type Scalar = f64;
//!
//!     fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
//!         self.u = Some(builder.register("u", 1)); // name + ghost width
//!     }
//!
//!     fn fill_ghosts<S: StorageBackend<f64>>(
//!         &self,
//!         grid: &CartesianGrid<D>,
//!         state: &mut State<f64, S>,
//!         _t: f64,
//!     ) {
//!         fill_ghosts_mirror(grid, state, self.u.unwrap());
//!     }
//!
//!     fn rhs_block<S: StorageBackend<f64>>(
//!         &self,
//!         ctx: &RhsContext<'_, CartesianGrid<D>, P>,
//!         state: &State<f64, S>,
//!         out: &mut BlockStateMut<'_, f64, S>,
//!         _scratch: &mut Scratch<f64, S>,
//!     ) {
//!         let u = self.u.unwrap();
//!         let lap = ctx.disc.build(ctx.grid, Laplacian);
//!         let input = state.view(ctx.grid, ctx.block, u);
//!         let mut output = out.view_mut(ctx.grid, ctx.block, u);
//!         lap.apply(ctx.grid, ctx.block, input, &mut output);
//!         for_each_interior(input.interior(), |i| {
//!             output.set(i, output.get(i) * self.kappa);
//!         });
//!     }
//! }
//!
//! let grid = CartesianGrid::new([32; 2], [16; 2], [0.0; 2], [0.1; 2])?;
//! let heat = Heat::<2> { kappa: 0.7, u: None };
//! let mut sim = Simulation::new(
//!     grid,
//!     FiniteDifference,
//!     heat,
//!     ForwardEuler,
//!     SerialScheduler,
//!     SystemAllocator,
//! );
//! for _ in 0..10 {
//!     sim.step(1e-3);
//! }
//! # Ok::<(), legendre::geometry::GridError>(())
//! ```
//!
//! Swap [`core::scheduler::SerialScheduler`] for
//! [`core::scheduler::RayonScheduler`] and the run parallelizes over blocks
//! with **bitwise-identical** results; the `examples/` directory adds the
//! async observation pipeline (Parquet snapshots, live statistics) on top.

pub mod core;
pub mod discretization;
pub mod geometry;
pub mod integrators;
pub mod io;
pub mod physics;
pub mod util;
