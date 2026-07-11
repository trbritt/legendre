//! Storage backends and allocation.
//!
//! Storage is separate from views.
//! A field never owns a `Vec<f64>`; it owns an opaque [`StorageBackend`]
//! produced by an [`Allocator`] exactly once, at setup. This keeps the door
//! open for memory pools, mmap-backed checkpoint restarts, pinned host memory
//! for GPU transfer, and per-block AMR storage — none of which change a single
//! downstream trait bound.
//!
//! Two numeric traits are deliberately distinct:
//! - [`Scalar`] is the *storage* contract: plain-old-data that can live in a
//!   backend and cross thread boundaries. Backends and allocators bound on this
//!   only.
//! - [`Real`] is the *arithmetic* contract used by stencils, models, and
//!   integrators. Every `Real` is a `Scalar`, but a backend never needs to know
//!   arithmetic exists.

use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Plain-old-data that can be held by a [`StorageBackend`].
pub trait Scalar: Copy + Send + Sync + Debug + 'static {
    /// The additive identity, used to zero-initialize storage.
    const ZERO: Self;
}

/// Field arithmetic required by stencils, models, and integrators.
///
/// Kept minimal and dependency-free on purpose: it names exactly the
/// operations the framework uses, so adding `f32` (or a dual number type for
/// sensitivity analysis) means implementing this trait, nothing more.
pub trait Real:
    Scalar
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The multiplicative identity.
    const ONE: Self;
    /// Convert from `f64` (used for scheme coefficients like `dt`).
    #[must_use]
    fn from_f64(x: f64) -> Self;
    /// Convert to `f64` (used for diagnostics and output).
    fn to_f64(self) -> f64;
    /// Square root.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Absolute value.
    #[must_use]
    fn abs(self) -> Self;
    /// Hyperbolic tangent.
    #[must_use]
    fn tanh(self) -> Self;
    /// Integer power.
    #[must_use]
    fn powi(self, n: i32) -> Self;
    /// Elementwise maximum (NaN-propagation per the underlying type).
    #[must_use]
    fn max(self, other: Self) -> Self;
    /// Elementwise minimum (NaN-propagation per the underlying type).
    #[must_use]
    fn min(self, other: Self) -> Self;
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Scalar for $t {
            const ZERO: Self = 0.0;
        }
        impl Real for $t {
            const ONE: Self = 1.0;
            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline(always)]
            fn abs(self) -> Self {
                self.abs()
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                self.tanh()
            }
            #[inline(always)]
            fn powi(self, n: i32) -> Self {
                self.powi(n)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                <$t>::min(self, other)
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

/// A contiguous, typed slab of memory.
///
/// **Ownership semantics:** a backend owns its bytes. Everything above it —
/// fields, views, snapshots — borrows. Backends are `Send + Sync` so blocks
/// can be dispatched across a thread pool; *disjointness* of concurrent
/// writes is the scheduler's obligation, expressed through `&mut` splitting,
/// never through interior mutability here.
pub trait StorageBackend<T: Scalar>: Send + Sync {
    /// Number of elements in the slab.
    fn len(&self) -> usize;
    /// Whether the slab holds no elements (zero-length slabs mark fields a
    /// buffer does not carry, e.g. static fields in tendency buffers).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The slab as a contiguous slice.
    fn as_slice(&self) -> &[T];
    /// The slab as a contiguous mutable slice.
    fn as_mut_slice(&mut self) -> &mut [T];
}

/// The single point where memory comes from.
///
/// Called during `Simulation` construction and never again (RFC 0001,
/// "no allocations after initialization"). Scratch arenas, field storage,
/// and integrator stage buffers all draw from an allocator so a memory pool
/// or GPU allocator can be swapped in without touching solver code.
/// `Send + Sync` because per-worker scratch factories carry a reference
/// across the scheduler's threads.
pub trait Allocator<T: Scalar>: Send + Sync {
    /// The backend this allocator produces.
    type Storage: StorageBackend<T>;

    /// Allocate zero-initialized storage for `len` elements.
    fn allocate(&self, len: usize) -> Self::Storage;
}

/// Heap-backed dense storage: the v1 CPU workhorse.
#[derive(Debug, Clone)]
pub struct DenseStorage<T: Scalar> {
    data: Box<[T]>,
}

impl<T: Scalar> StorageBackend<T> for DenseStorage<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        &self.data
    }
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Allocates [`DenseStorage`] from the global heap.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemAllocator;

impl<T: Scalar> Allocator<T> for SystemAllocator {
    type Storage = DenseStorage<T>;

    fn allocate(&self, len: usize) -> Self::Storage {
        DenseStorage {
            data: vec![T::ZERO; len].into_boxed_slice(),
        }
    }
}
