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
//! - **Noise dimension is a type.** [`Model::Noise`] names the driver set
//!   ([`NoNoise`] or [`Wiener<M>`]); integrators are implemented per driver
//!   set, so pairing a deterministic-only scheme with a stochastic model is
//!   a *compile error*, not a silently dropped term.
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

/// What multiplies a vector field in the dynamics: the deterministic clock
/// or one of the model's independent Wiener processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driver {
    /// The deterministic clock; this vector field is scaled by `dt`.
    Time,
    /// The `j`-th independent Wiener process (`j < M` for a model with
    /// `Noise = Wiener<M>`); this vector field is scaled by
    /// `ΔWⱼ = √dt·ξⱼ`, with ξⱼ drawn per cell from the counter-based
    /// generator (see [`crate::util::rng`]).
    Wiener(usize),
}

/// Type-level description of a model's driver set beyond time.
///
/// Implemented by the marker types [`NoNoise`] and [`Wiener<M>`]; models
/// name one as [`Model::Noise`] and integrators are implemented per spec,
/// which is what turns a model/integrator mismatch into a compile error.
///
/// The supertraits make every spec a trivial marker, so generic models can
/// `derive(Clone, Debug)` while carrying a spec in `PhantomData`.
pub trait NoiseSpec:
    Copy + Clone + std::fmt::Debug + Default + Send + Sync + 'static
{
    /// Number of independent Wiener drivers.
    const WIENER_DIM: usize;
}

/// Driver set of a deterministic model: time only.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoNoise;

impl NoiseSpec for NoNoise {
    const WIENER_DIM: usize = 0;
}

/// Driver set with `M` independent Wiener processes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wiener<const M: usize>;

impl<const M: usize> NoiseSpec for Wiener<M> {
    const WIENER_DIM: usize = M;
}

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

    /// The driver set of this model's dynamics: [`NoNoise`] for a
    /// deterministic system, [`Wiener<M>`] for one driven by `M`
    /// independent Wiener processes.
    type Noise: NoiseSpec;

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
    /// expressible. Models typically forward to a grid-family helper per
    /// field (e.g. `fill_ghosts_mirror(grid, state, self.phi)`).
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
    /// dynamic field (`out` arrives with unspecified contents, as models
    /// overwrite it). For [`Driver::Wiener`], write the noise *amplitude*
    /// Vⱼ(Y) into the interior cells of the driven fields only (`out`
    /// arrives zeroed); the integrator supplies √dt·ξ with one deviate per
    /// cell, broadcast across fields, so a driver shared by several fields
    /// moves them with the *same* increment.
    ///
    /// Models with `Noise = NoNoise` only ever receive [`Driver::Time`].
    fn vector_field_block<S: StorageBackend<Self::Scalar>>(
        &self,
        driver: Driver,
        ctx: &RhsContext<'_, G, D>,
        state: &State<Self::Scalar, S>,
        out: &mut BlockStateMut<'_, Self::Scalar, S>,
        scratch: &mut Scratch<Self::Scalar, S>,
    );

    /// Largest stable explicit timestep, if the model knows one (e.g.
    /// `0.25·h²/D` for phase fields). Advisory; drivers may use it to pick dt.
    fn stable_dt(&self, _grid: &G) -> Option<f64> {
        None
    }
}
