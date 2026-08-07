//! Drivers: the measures that multiply a model's vector fields.
//!
//! Dynamics are `dY = Σ_d V_d(Y, t)·dμ_d` over a set of drivers `d` — the
//! deterministic clock (`dμ = dt`) and any number of independent stochastic
//! processes (`dμ = dWⱼ` for Wiener drivers). Everything in the framework
//! that must know "which terms exist and what moves them" speaks this one
//! vocabulary:
//!
//! - **Fields** declare at registration which drivers move them
//!   ([`crate::core::state::StateBuilder`]); a *static* field is one moved
//!   by no driver.
//! - **Buffers** are allocated per driver
//!   ([`crate::core::state::State::like_for`]): storage for exactly the
//!   fields that driver moves.
//! - **Models** evaluate one vector field per driver
//!   ([`crate::physics::model::Model::vector_field_block`]) and name their
//!   driver set as a type ([`DriverSet`]).
//! - **Integrators** request one tendency buffer per driver
//!   ([`crate::integrators::StageKind`]) and apply each through
//!   [`DriverKind::apply_slab`], which owns the measure-correct scaling.
//!
//! [`Driver`] is a `Copy` enum and [`DriverKind`] is implemented *on the
//! enum* by a single match hoisted outside the per-cell loops, so driver
//! dispatch is static and per-slab, never per-cell and never `dyn`. Adding
//! a new driver kind (e.g. a Poisson jump measure) is a new variant plus a
//! kernel arm — no new registration, allocation, or `Simulation` surface.

use crate::{
    core::storage::Real,
    geometry::grid::{BlockId, Grid},
    util::rng,
};

/// What multiplies a vector field in the dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driver {
    /// The deterministic clock; this vector field is scaled by `dt`.
    Time,
    /// The `j`-th independent Wiener process; this vector field is scaled
    /// by `ΔWⱼ = √dt·ξⱼ`, with ξⱼ drawn per cell from the counter-based
    /// generator (see [`crate::util::rng`]).
    Wiener(usize),
    /// The `j`-th jump (point-process) driver; this vector field is the
    /// per-cell increment `ΔJ` applied on the cells where the process fires
    /// this step. Firing is drawn per cell from the counter-based generator
    /// as `Bernoulli(1 − e^{−λ·dt})` against the model-supplied per-cell
    /// intensity λ (see [`crate::util::rng`]); the `rate·dt → probability`
    /// conversion lives in the kernel, never in model code.
    Jump(usize),
}

impl Driver {
    /// RNG stream tag: the driver kind in the high bits, its index in the
    /// low bits, so streams of different kinds can never collide.
    const fn stream(self) -> u64 {
        match self {
            Self::Time => 0,
            Self::Wiener(j) => (1 << 32) | j as u64,
            Self::Jump(j) => (2 << 32) | j as u64,
        }
    }
}

/// One field a driver moves, as handed to [`DriverKind::apply_slab`].
///
/// Bundles its ghost width, the driver's vector field on this block (`amp`),
/// and the state slab to update. All ghost-inclusive; ghost entries are
/// skipped by the stochastic kernels via [`Grid::cell_key`].
pub struct DriverField<'a, T> {
    /// Ghost-ring width of this field.
    pub ghost: u32,
    /// The driver's vector field on this block: `b(Y)` for a Wiener driver,
    /// the increment `ΔJ(Y)` for a jump driver, `dY/dt` for time.
    pub amp: &'a [T],
    /// The state slab this driver updates in place.
    pub state: &'a mut [T],
}

/// The behavior of a driver: its measure applied across the fields it moves.
///
/// One implementation on [`Driver`] delegates to the kernel of each variant —
/// static dispatch, resolved once per block. A driver moves a *set* of fields
/// with **one shared per-cell random draw** (the same Wiener deviate, or the
/// same jump fire), so the kernel is multi-field: correlated components move
/// together by construction, and a jump moves all its fields on the same fire.
pub trait DriverKind {
    /// Apply this driver's measure over one block: for each field in
    /// `fields`, `state += dμ ∘ amp`, where `dμ` is the measure over a step
    /// of size `dt` —
    ///
    /// - [`Driver::Time`]: `dt` (a uniform axpy, no RNG);
    /// - [`Driver::Wiener`]: `√dt·ξ` per cell, ξ standard-normal;
    /// - [`Driver::Jump`]: the field's increment on cells where the process
    ///   fires, each firing `Bernoulli(1 − e^{−λ·dt})` against the per-cell
    ///   intensity in `rate` (which is `Some` only for jump drivers).
    ///
    /// The per-cell draw is keyed by `(seed, salt, stream, block, cell)` —
    /// **shared across every field** the driver moves and independent of
    /// ghost width, so correlated diffusion and multi-field jumps are
    /// consistent by construction. Stochastic kernels skip zero-amplitude
    /// entries and never touch ghost entries (gated by [`Grid::cell_key`]).
    #[allow(clippy::too_many_arguments)]
    fn apply_slab<T: Real, G: Grid>(
        &self,
        grid: &G,
        block: BlockId,
        dt: f64,
        seed: u64,
        salt: u64,
        rate: Option<&[T]>,
        fields: &mut [DriverField<'_, T>],
    );
}

impl DriverKind for Driver {
    #[inline]
    fn apply_slab<T: Real, G: Grid>(
        &self,
        grid: &G,
        block: BlockId,
        dt: f64,
        seed: u64,
        salt: u64,
        rate: Option<&[T]>,
        fields: &mut [DriverField<'_, T>],
    ) {
        match *self {
            // dt is uniform over the slab: a pure axpy per field, no RNG, no
            // cell-key arithmetic (ghost garbage is refilled before any
            // stencil reads it, exactly as with stage combination).
            Self::Time => {
                let a = T::from_f64(dt);
                for f in fields {
                    for (x, v) in f.state.iter_mut().zip(f.amp) {
                        *x += a * *v;
                    }
                }
            }
            Self::Wiener(_) => {
                let scale = T::from_f64(dt.sqrt());
                let block_key = rng::mix_key(seed, &[salt, self.stream(), block.index() as u64]);
                for f in fields {
                    for (i, (x, v)) in f.state.iter_mut().zip(f.amp).enumerate() {
                        if *v == T::ZERO {
                            continue;
                        }
                        // One deviate per (cell, driver, step), broadcast
                        // across every field the driver moves: the cell id is
                        // ghost-independent, so correlated multi-component
                        // dynamics see the same increment on every field.
                        let Some(cell) = grid.cell_key(block, f.ghost, i) else {
                            continue;
                        };
                        let key = rng::splitmix64(block_key ^ cell);
                        *x += scale * *v * T::from_f64(rng::standard_normal(key));
                    }
                }
            }
            Self::Jump(_) => {
                let rate = rate.expect("a jump driver must be given an intensity slab");
                let block_key = rng::mix_key(seed, &[salt, self.stream(), block.index() as u64]);
                for f in fields {
                    for (i, (x, v)) in f.state.iter_mut().zip(f.amp).enumerate() {
                        let Some(cell) = grid.cell_key(block, f.ghost, i) else {
                            continue;
                        };
                        let lambda = rate[cell as usize].to_f64();
                        if lambda <= 0.0 {
                            continue;
                        }
                        // One fire per (cell, driver, step), shared across
                        // every field the driver moves: the event fires once
                        // and applies each field's increment together. The
                        // rate·dt → probability conversion lives here.
                        let key = rng::splitmix64(block_key ^ cell);
                        let p = -(-lambda * dt).exp_m1();
                        if rng::unit_open(key) < p {
                            *x += *v;
                        }
                    }
                }
            }
        }
    }
}

/// Type-level description of a model's stochastic driver set (the time
/// driver is always implicit).
///
/// Implemented by the marker types [`NoNoise`] and [`Wiener<M>`]; models
/// name one as [`crate::physics::model::Model::Drivers`] and integrators
/// are implemented per set, which is what turns a model/integrator
/// mismatch into a compile error. `LEN` and `driver(i)` are kind-agnostic,
/// so a future mixed set (Wiener + jump drivers) is a new marker type, not
/// a trait change.
///
/// The supertraits make every set a trivial marker, so generic models can
/// `derive(Clone, Debug)` while carrying one in `PhantomData`.
pub trait DriverSet: Copy + Clone + std::fmt::Debug + Default + Send + Sync + 'static {
    /// Number of stochastic drivers in the set.
    const LEN: usize;

    /// The `i`-th stochastic driver (`i < LEN`).
    fn driver(i: usize) -> Driver;
}

/// Driver set of a deterministic model: time only.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoNoise;

impl DriverSet for NoNoise {
    const LEN: usize = 0;

    fn driver(_i: usize) -> Driver {
        unreachable!("NoNoise has no stochastic drivers")
    }
}

/// Driver set with `M` independent Wiener processes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wiener<const M: usize>;

impl<const M: usize> DriverSet for Wiener<M> {
    const LEN: usize = M;

    fn driver(i: usize) -> Driver {
        debug_assert!(i < M);
        Driver::Wiener(i)
    }
}

/// Driver set with `W` independent Wiener processes and `J` jump drivers.
///
/// The Wiener drivers occupy indices `0..W`, the jump drivers `W..W+J`, so
/// `driver(i)` yields `Wiener(i)` then `Jump(i−W)`. Mixed diffusion + jump
/// dynamics — a jump-diffusion asset, a piecewise-deterministic Markov
/// process, a controlled point process — name this set.
#[derive(Debug, Clone, Copy, Default)]
pub struct WienerJump<const W: usize, const J: usize>;

impl<const W: usize, const J: usize> DriverSet for WienerJump<W, J> {
    const LEN: usize = W + J;

    fn driver(i: usize) -> Driver {
        debug_assert!(i < W + J);
        if i < W {
            Driver::Wiener(i)
        } else {
            Driver::Jump(i - W)
        }
    }
}
