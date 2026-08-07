//! The [`Model`] trait: dynamics as driver-indexed vector fields.
//!
//! A model is the system
//!
//! ```text
//! dY = V₀(Y, t)·dt + Σⱼ Vⱼ(Y, t)·dWⱼ,   j = 1..M
//! ```
//!
//! — a *family of vector fields*, one per [`Driver`]. A deterministic PDE is
//! the M = 0 member of the family, not a different kind of object.
//!
//! - **`vector_field_block`, not `Model::step`**. A model evaluates the
//!   vector field conjugate to one driver into an output buffer and never
//!   mutates simulation state; the integrator owns state updates and the
//!   measure-correct scaling (dt for [`Driver::Time`], √dt·ξ for
//!   [`Driver::Wiener`]). Stochastic calculus convention is **Itô**:
//!   Wiener fields are evaluated at the pre-update state.
//!
//! - **The driver set is a type.** [`Model::Drivers`] names it ([`NoNoise`]
//!   or [`Wiener<M>`]); integrators are implemented per driver set, so
//!   pairing a deterministic-only scheme with a stochastic model is a
//!   *compile error*, not a silently dropped term. Fields declare at
//!   registration which drivers move them, and per-driver buffers carry
//!   storage for exactly those fields.
//!
//! - **Correlated noise is model mathematics.** Drivers are independent by
//!   construction (the framework draws one i.i.d. increment per cell per
//!   driver); correlation between components is expressed by how the model's
//!   Wiener fields mix drivers across its fields — i.e. the Cholesky factor
//!   lives in the model, where it belongs.
//!
//! - **Per-block evaluation.** The scheduler drives `vector_field_block`
//!   once per block; the model sees one block's output and the whole
//!   (read-only) state. This is the contract that makes uniform grids and
//!   AMR identical to the model.
//!
//! - **`Model<G, D>` with bounds at the impl.** The trait puts no
//!   requirements on the discretization `D`; each model impl demands exactly
//!   the operators it uses (e.g. `where D: Discretizes<G, Laplacian>`), so
//!   models are generic over schemes and schemes over models.
//!
//! - **Boundary conditions belong to the model.** `fill_ghosts` runs once
//!   per RHS evaluation over the whole state; models delegate the interior
//!   halo exchange and standard physical conditions to grid-family helpers
//!   (e.g. [`crate::geometry::cartesian::fill_ghosts_mirror`]). Grid helpers
//!   own the cross-block copies because they are topology, not physics.

use crate::{
    core::{
        scratch::{Scratch, ScratchSpec},
        state::{BlockStateMut, State, StateBuilder},
        storage::{Real, StorageBackend},
    },
    geometry::grid::{BlockId, Grid},
};

pub use crate::core::driver::{Driver, DriverKind, DriverSet, NoNoise, Wiener, WienerJump};

/// One row of a stiff tridiagonal operator along dimension 0:
/// `(L·y)ᵢ = row[0]·yᵢ₋₁ + row[1]·yᵢ + row[2]·yᵢ₊₁`.
///
/// Written by [`Model::stiff_rows`] and consumed by IMEX schemes
/// ([`crate::integrators::ImexEuler`]); coefficients are scheme-level reals
/// (like `dt`), independent of the model's scalar type.
pub type StiffRow = [f64; 3];

/// Everything a model may consult while evaluating one block's vector
/// field. Deliberately read-only and allocation-free.
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

/// A system of differential equations — deterministic or stochastic; see
/// the module docs for the contract.
pub trait Model<G: Grid, D>: Send + Sync {
    /// Arithmetic type of this model's fields.
    type Scalar: Real;

    /// The stochastic driver set of this model's dynamics (time is always
    /// implicit): [`NoNoise`] for a deterministic system, [`Wiener<M>`]
    /// for one driven by `M` independent Wiener processes.
    type Drivers: DriverSet;

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
    /// evaluation sweep with the evaluation time `t` (stage time for
    /// multi-stage schemes), so time-dependent boundary forcing is
    /// expressible. Models forward to a grid-family helper per field — a fixed
    /// condition with `fill_ghosts_mirror`, or a per-face rule with
    /// `fill_ghosts_bc(grid, state, self.phi, |dim, side| …)`.
    fn fill_ghosts<S: StorageBackend<Self::Scalar>>(
        &self,
        _grid: &G,
        _state: &mut State<Self::Scalar, S>,
        _t: f64,
    ) {
    }

    /// Evaluate the vector field conjugate to `driver` on one block into
    /// `out`, reading the ghost-filled `state`. Must not touch any other
    /// block's output.
    ///
    /// For [`Driver::Time`], write dY/dt into the interior cells of every
    /// time-driven field (`out` arrives with unspecified contents, as
    /// models overwrite it). For a [`Driver::Wiener`] driver, write the
    /// *amplitude* Vⱼ(Y) into the interior cells of the fields registered as
    /// driven by it (`out` arrives zeroed and carries storage only for those
    /// fields; see [`StateBuilder::register_driven`]); the integrator applies
    /// the driver's measure with one increment per cell, broadcast across
    /// fields, so a driver shared by several fields moves them with the
    /// *same* increment.
    ///
    /// For a [`Driver::Jump`] driver, write the per-field *increment* `ΔJ(Y)`
    /// into those same driven-field slabs **and** the per-cell firing
    /// intensity λ(Y) into `out.intensity_mut(grid, block)`; the integrator
    /// draws one fire per cell against λ (`1 − e^{−λ·dt}`) and, on cells that
    /// fire, applies every field's increment together. The `rate·dt →
    /// probability` conversion is the kernel's, never the model's — mirroring
    /// the `√dt` rule for Wiener amplitudes.
    ///
    /// Models with `Drivers = NoNoise` only ever receive [`Driver::Time`].
    fn vector_field_block<S: StorageBackend<Self::Scalar>>(
        &self,
        driver: Driver,
        ctx: &RhsContext<'_, G, D>,
        state: &State<Self::Scalar, S>,
        out: &mut BlockStateMut<'_, Self::Scalar, S>,
        scratch: &mut Scratch<Self::Scalar, S>,
    );

    /// Enforce pointwise state constraints after each completed step (e.g.
    /// positivity of a CIR variance, clamping an order parameter). Called by
    /// the simulation over the whole state between the integrator update and
    /// observer notification — once per step, *after* the full advance, so
    /// under subcycling it sees the synchronized state, not intermediate
    /// substages. This is an explicit post-hoc correction, not part of any
    /// scheme's stage combination; multi-stage intermediate states are never
    /// projected.
    ///
    /// Default: no-op.
    fn project<S: StorageBackend<Self::Scalar>>(
        &self,
        _grid: &G,
        _state: &mut State<Self::Scalar, S>,
    ) {
    }

    /// The stiff linear tridiagonal part `L` (along dimension 0) of this
    /// model's time vector field on one block of `field`.
    ///
    /// Write one [`StiffRow`] per interior cell into `rows` and return
    /// `true`. Rows at *physical domain boundaries* must fold the model's
    /// boundary condition into the coefficients — the implicit solve sees no
    /// ghosts. Returning `false` (the default) declares the field has no
    /// stiff part: IMEX schemes then treat it fully explicitly, and a model
    /// with no stiff field anywhere degenerates
    /// [`ImexEuler`](crate::integrators::ImexEuler) to forward Euler bit for
    /// bit.
    ///
    /// Only consulted by IMEX schemes. [`Model::vector_field_block`] remains
    /// the *complete* dynamics — an IMEX scheme subtracts `L·Y` itself, so
    /// this split can never drift out of sync with the physics.
    fn stiff_rows(
        &self,
        _grid: &G,
        _block: BlockId,
        _field: crate::core::state::FieldHandle<Self::Scalar>,
        _rows: &mut [StiffRow],
    ) -> bool {
        false
    }

    /// Stable timestep of the **nonstiff remainder** `N = V₀ − L·Y` alone —
    /// the dt bound an IMEX scheme must respect once the stiff part is
    /// integrated implicitly (independent of grid spacing for a
    /// diffusion-stiff model). `None` (the default) falls back to
    /// [`Model::stable_dt`].
    fn stable_dt_nonstiff(&self, _spacing: G::Point) -> Option<f64> {
        None
    }

    /// Largest stable explicit timestep for a cell of the given `spacing`,
    /// if the model knows one (e.g. `0.25·h²/D` for a diffusion term).
    ///
    /// A *pure function of spacing* — not the whole grid — so an integrator
    /// can evaluate it at whichever resolution it needs: the finest cell
    /// for a global-dt scheme, or per level for subcycling, where the ratio
    /// `stable_dt(h_coarse) / stable_dt(h_fine)` gives the substep count
    /// directly (parabolic ⇒ `r²`, hyperbolic ⇒ `r`, no assumption baked
    /// in). Advisory; drivers may use it to pick dt.
    fn stable_dt(&self, _spacing: G::Point) -> Option<f64> {
        None
    }
}
