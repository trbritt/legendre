//! Per-worker scratch memory.
//!
//! Models declare their scratch needs up front as a [`ScratchSpec`];
//! the simulation turns that into a factory the scheduler
//! calls once per worker. Inside the hot loop nothing allocates — a model
//! borrows pre-sized slabs by index. Slabs are block-sized (ghost-inclusive
//! for the largest ghost width the model uses) so a scratch slab can be
//! viewed through the grid exactly like a field slab.

use crate::core::storage::{Allocator, Scalar, StorageBackend};

/// How many block-sized scratch slabs a model needs, and how long each must
/// be. `slab_len` is grid-dependent, so models compute it via
/// `Grid::block_len` for their widest-ghost field.
#[derive(Debug, Clone, Copy)]
pub struct ScratchSpec {
    /// Number of block-sized scratch slabs required.
    pub slabs: usize,
    /// Length of each slab in elements (ghost-inclusive block length).
    pub slab_len: usize,
}

impl ScratchSpec {
    /// No scratch required.
    pub const NONE: Self = Self {
        slabs: 0,
        slab_len: 0,
    };
}

/// A worker-private set of scratch slabs.
pub struct Scratch<T: Scalar, S: StorageBackend<T>> {
    slabs: Vec<S>,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: Scalar, S: StorageBackend<T>> Scratch<T, S> {
    /// Allocate the slabs described by `spec` (setup-time only).
    pub fn allocate<A: Allocator<T, Storage = S>>(spec: ScratchSpec, alloc: &A) -> Self {
        Self {
            slabs: (0..spec.slabs)
                .map(|_| alloc.allocate(spec.slab_len))
                .collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Borrow one slab. Contents are unspecified between uses; callers
    /// overwrite before reading.
    #[inline(always)]
    pub fn slab_mut(&mut self, i: usize) -> &mut [T] {
        self.slabs[i].as_mut_slice()
    }

    /// Borrow all slabs at once (each independently mutable).
    pub fn slabs_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.slabs
            .iter_mut()
            .map(super::storage::StorageBackend::as_mut_slice)
    }
}

/// A fixed pool of [`Scratch`] instances for worker checkout.
///
/// Allocated once at simulation setup and checked out by workers per
/// block-batch, upholding "no allocations after initialization".
///
/// Sized to the scheduler's
/// [`crate::core::scheduler::Scheduler::max_concurrency`], so a `try_lock`
/// sweep always finds a free slot (at most `max_concurrency` checkouts are live
/// at once); the terminal blocking `lock` is a safety net, not an expected
/// path.
pub struct ScratchPool<T: Scalar, S: StorageBackend<T>> {
    slots: Vec<parking_lot::Mutex<Scratch<T, S>>>,
}

impl<T: Scalar, S: StorageBackend<T>> ScratchPool<T, S> {
    /// Allocate `concurrency` scratch instances of the given spec.
    pub fn allocate<A: Allocator<T, Storage = S>>(
        spec: ScratchSpec,
        alloc: &A,
        concurrency: usize,
    ) -> Self {
        Self {
            slots: (0..concurrency.max(1))
                .map(|_| parking_lot::Mutex::new(Scratch::allocate(spec, alloc)))
                .collect(),
        }
    }

    /// Check out a scratch instance for the duration of the guard.
    pub fn checkout(&self) -> parking_lot::MutexGuard<'_, Scratch<T, S>> {
        for slot in &self.slots {
            if let Some(guard) = slot.try_lock() {
                return guard;
            }
        }
        self.slots[0].lock()
    }
}
