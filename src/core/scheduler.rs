//! Block dispatch. The **only** module in the crate allowed to name Rayon.
//!
//! Execution is scheduler-driven.
//! The scheduler's contract is deliberately narrow — "run this function over
//! these disjoint block work-items, giving each worker private scratch" —
//! because that is the whole execution model for block-structured PDEs.
//! Uniform grids, AMR level sweeps, and (later) GPU streams or MPI ranks are
//! all implementations of this one trait.
//!
//! **Determinism:** work items are disjoint `&mut` blocks, and results land
//! at fixed slab locations, so answers are independent of scheduling order.
//! The remaining determinism hazards — RNG and floating-point reductions —
//! are handled where they arise (counter-based noise keyed by (block, cell,
//! step); reductions defined as fixed-order over blocks).

use crate::geometry::grid::BlockId;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// Dispatches per-block work.
///
/// `Item` is whatever the caller split per block (typically `BlockStorage`
/// from an output `State`); `scratch` is a factory invoked once per worker,
/// never per block, so scratch memory is reused across the blocks a worker
/// processes.
pub trait Scheduler: Send + Sync {
    /// Run `f` once per item, with disjoint `&mut` access and worker-private
    /// scratch. Implementations may order and parallelize freely; results
    /// must not depend on that ordering (block writes are disjoint).
    fn for_each_block_mut<Item, Sc, Init, F>(&self, items: &mut [Item], scratch: Init, f: F)
    where
        Item: Send,
        Init: Fn() -> Sc + Send + Sync,
        F: Fn(BlockId, &mut Item, &mut Sc) + Send + Sync;

    /// Upper bound on concurrently executing block closures. Sizes the
    /// worker-pinned scratch pool at setup.
    fn max_concurrency(&self) -> usize;
}

/// Single-threaded reference scheduler: the semantics oracle every parallel
/// scheduler must reproduce bit-for-bit.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialScheduler;

impl Scheduler for SerialScheduler {
    fn for_each_block_mut<Item, Sc, Init, F>(&self, items: &mut [Item], scratch: Init, f: F)
    where
        Item: Send,
        Init: Fn() -> Sc + Send + Sync,
        F: Fn(BlockId, &mut Item, &mut Sc) + Send + Sync,
    {
        let mut sc = scratch();
        for (i, item) in items.iter_mut().enumerate() {
            f(BlockId(i as u32), item, &mut sc);
        }
    }

    fn max_concurrency(&self) -> usize {
        1
    }
}

/// Rayon-backed work-stealing scheduler.
#[derive(Debug, Clone, Copy, Default)]
pub struct RayonScheduler;

impl Scheduler for RayonScheduler {
    fn for_each_block_mut<Item, Sc, Init, F>(&self, items: &mut [Item], scratch: Init, f: F)
    where
        Item: Send,
        Init: Fn() -> Sc + Send + Sync,
        F: Fn(BlockId, &mut Item, &mut Sc) + Send + Sync,
    {
        items
            .par_iter_mut()
            .panic_fuse()
            .enumerate()
            .for_each_init(&scratch, |sc, (i, item)| f(BlockId(i as u32), item, sc));
    }

    fn max_concurrency(&self) -> usize {
        rayon::current_num_threads()
    }
}
