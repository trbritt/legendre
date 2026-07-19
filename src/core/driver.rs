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
}

impl Driver {
    /// RNG stream tag: the driver kind in the high bits, its index in the
    /// low bits, so streams of different kinds can never collide.
    const fn stream(self) -> u64 {
        match self {
            Self::Time => 0,
            Self::Wiener(j) => (1 << 32) | j as u64,
        }
    }
}

/// The behavior of a driver: its measure as a slab kernel.
///
/// One implementation on [`Driver`] delegates to the kernel of each
/// variant — static dispatch, resolved once per slab.
pub trait DriverKind {
    /// Apply this driver's increment over one block's slab of one field:
    /// `state += dμ ∘ amp`, where `dμ` is the driver's measure over a step
    /// of size `dt` (`dt` itself for [`Driver::Time`]; `√dt·ξ` per cell,
    /// keyed by `(seed, salt, stream, block, cell)`, for
    /// [`Driver::Wiener`]). Stochastic kernels skip zero-amplitude entries
    /// and never touch ghost entries (gated by [`Grid::cell_key`]).
    #[allow(clippy::too_many_arguments)]
    fn apply_slab<T: Real, G: Grid>(
        &self,
        grid: &G,
        block: BlockId,
        ghost: u32,
        dt: f64,
        seed: u64,
        salt: u64,
        amp: &[T],
        state: &mut [T],
    );
}

impl DriverKind for Driver {
    #[inline]
    fn apply_slab<T: Real, G: Grid>(
        &self,
        grid: &G,
        block: BlockId,
        ghost: u32,
        dt: f64,
        seed: u64,
        salt: u64,
        amp: &[T],
        state: &mut [T],
    ) {
        match *self {
            // dt is uniform over the slab: a pure axpy, no RNG, no
            // cell-key arithmetic (ghost garbage is refilled before any
            // stencil reads it, exactly as with stage combination).
            Self::Time => {
                let a = T::from_f64(dt);
                for (x, v) in state.iter_mut().zip(amp) {
                    *x += a * *v;
                }
            }
            Self::Wiener(_) => {
                let scale = T::from_f64(dt.sqrt());
                let block_key = rng::mix_key(seed, &[salt, self.stream(), block.index() as u64]);
                for (i, (x, v)) in state.iter_mut().zip(amp).enumerate() {
                    if *v == T::ZERO {
                        continue;
                    }
                    // One deviate per (cell, driver, step), broadcast
                    // across every field the driver moves: the cell id is
                    // ghost-independent, so correlated multi-component
                    // dynamics see the same increment on every field.
                    let Some(cell) = grid.cell_key(block, ghost, i) else {
                        continue;
                    };
                    let key = rng::splitmix64(block_key ^ cell);
                    *x += scale * *v * T::from_f64(rng::standard_normal(key));
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
