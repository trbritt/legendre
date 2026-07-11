//! The [`Grid`] trait: topology, blocks, and block views.
//!
//! **Design decisions:**
//!
//! - **Dimension is an associated fact, not a trait parameter.** Concrete grids
//!   carry `const D: usize`; the trait exposes dimension through `DIM` and
//!   through associated `Point`/`Index` types (`[f64; D]`, `[isize; D]` for
//!   Cartesian). Generic solver code never needs `D`; only concrete stencils
//!   do, and they are written per grid family anyway.
//!
//! - **The grid is a collection of blocks**. Block identity is a flat
//!   [`BlockId`]; refinement level and geometry are queried per block so a
//!   uniform grid (one level) and an adaptive grid (many levels) present the
//!   same execution surface.
//!
//! - **Views are grid-associated types (GATs).** Storage hands the grid a raw
//!   `&[T]` slab for one block; the grid wraps it in a typed view that knows
//!   the block's interior extent and ghost width. This is what makes "storage
//!   separate from views" zero-cost: the view is a slice + a few integers,
//!   constructed inline, and index arithmetic monomorphizes into the stencil
//!   loop. Stencils are written against `G::View<'_, T>`, so an AMR grid can
//!   hand out views that transparently handle coarse–fine interpolation later
//!   without any stencil signature changing.
//!
//! - **Grids own no field data.** They translate (block, index) into offsets
//!   and coordinates; storage lives in [`crate::core::state::State`].

use crate::core::storage::Scalar;
use std::fmt::Debug;

/// Flat identifier of a block.
///
/// Stable for the lifetime of a grid; an AMR regrid produces a *new* grid
/// (and a state migration), never mutates one in place — which is what
/// keeps `Grid: Sync` trivially sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// The block id as a `usize` index.
    #[inline(always)]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Spatial topology and block-structured layout.
pub trait Grid: Send + Sync + 'static {
    /// Spatial dimension.
    const DIM: usize;

    /// A physical coordinate, e.g. `[f64; D]`.
    type Point: Copy + Send + Sync + Debug;

    /// A cell index *within one block*, relative to the block's interior
    /// origin, e.g. `[isize; D]`. Signed so ghost cells are addressable as
    /// `-1`, `-2`, … without offset gymnastics in stencil code.
    type Index: Copy + Send + Sync + Debug;

    /// Read-only view of one block's slab (interior + ghost ring).
    type View<'a, T: Scalar>: Copy + Send + Sync
    where
        Self: 'a;

    /// Mutable view of one block's slab.
    type ViewMut<'a, T: Scalar>: Send
    where
        Self: 'a;

    /// Number of blocks tiling the domain.
    fn num_blocks(&self) -> usize;

    /// Number of scalar entries in one block's slab, ghosts included, for a
    /// field stored with ghost ring width `ghost`. This is the allocation
    /// contract between grid and storage.
    fn block_len(&self, block: BlockId, ghost: u32) -> usize;

    /// Refinement level of a block (always 0 on a uniform grid).
    fn level(&self, block: BlockId) -> u8;

    /// Cell spacing on this block (level-dependent under AMR).
    fn spacing(&self, block: BlockId) -> Self::Point;

    /// Physical coordinates of a cell center.
    fn cell_center(&self, block: BlockId, idx: Self::Index) -> Self::Point;

    /// Wrap one block's raw slab in a typed read view.
    ///
    /// `data` must be exactly `block_len(block, ghost)` long; implementations
    /// debug-assert this. Callers (only `State`) uphold it by construction.
    fn view<'a, T: Scalar>(
        &'a self,
        block: BlockId,
        ghost: u32,
        data: &'a [T],
    ) -> Self::View<'a, T>;

    /// Wrap one block's raw slab in a typed mutable view.
    fn view_mut<'a, T: Scalar>(
        &'a self,
        block: BlockId,
        ghost: u32,
        data: &'a mut [T],
    ) -> Self::ViewMut<'a, T>;
}
