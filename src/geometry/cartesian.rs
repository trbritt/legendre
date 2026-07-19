//! Uniform block-structured Cartesian grid, generic over dimension.
//!
//! The global domain of `cells[d]` interior cells per dimension is tiled by
//! equal blocks of `block_cells[d]` cells. Every field slab for a block is
//! stored dense, dimension-0-fastest, with a ghost ring of the field's
//! declared width:
//!
//! ```text
//! linear(idx) = Σ_d (idx[d] + ghost) · stride[d],   stride[0] = 1,
//! stride[d+1] = stride[d] · (block_cells[d] + 2·ghost)
//! ```
//!
//! Views do this arithmetic inline; after monomorphization a stencil read
//! `u.get([1, 0])` is a single indexed load.

use super::grid::{BlockId, Grid};
use crate::{
    core::{
        state::{FieldHandle, State},
        storage::{Scalar, StorageBackend},
    },
    geometry::GridError,
};

/// Uniform Cartesian grid of `cells` interior cells tiled by congruent
/// blocks; see the module docs for the storage layout.
#[derive(Debug, Clone)]
pub struct CartesianGrid<const D: usize> {
    cells: [usize; D],
    block_cells: [usize; D],
    blocks_per_dim: [usize; D],
    origin: [f64; D],
    spacing: [f64; D],
    periodic: [bool; D],
}

impl<const D: usize> CartesianGrid<D> {
    /// A domain of `cells` interior cells tiled by `block_cells`-sized
    /// blocks. `cells[d]` must be divisible by `block_cells[d]` — v1 keeps
    /// all blocks congruent, which makes storage layout and halo exchange
    /// uniform.
    ///
    /// # Errors
    ///
    /// Returns a [`GridError`] if any dimension is empty, `cells[d]` is not
    /// divisible by `block_cells[d]`, or `spacing[d]` is not a positive
    /// finite number.
    pub fn new(
        cells: [usize; D],
        block_cells: [usize; D],
        origin: [f64; D],
        spacing: [f64; D],
    ) -> Result<Self, GridError> {
        let mut blocks_per_dim = [0usize; D];
        for d in 0..D {
            if !(cells[d] > 0 && block_cells[d] > 0) {
                return Err(GridError::EmptyDimension(d));
            }
            if !cells[d].is_multiple_of(block_cells[d]) {
                return Err(GridError::IndivisibleDimension {
                    dimension: d,
                    cells: cells[d],
                    block_cells: block_cells[d],
                });
            }
            if spacing[d] <= 0.0 || !spacing[d].is_finite() {
                return Err(GridError::InvalidSpacing {
                    dimension: d,
                    spacing: spacing[d],
                });
            }
            blocks_per_dim[d] = cells[d] / block_cells[d];
        }
        Ok(Self {
            cells,
            block_cells,
            blocks_per_dim,
            origin,
            spacing,
            periodic: [false; D],
        })
    }

    /// Mark dimensions as periodic: the domain wraps, so the last block's
    /// high face neighbors the first block's low face (possibly the same
    /// block when one block spans the dimension).
    ///
    /// Periodicity is **topology, not physics**: [`Self::face_neighbor`]
    /// wraps in periodic dimensions, so every ghost-fill helper built on it
    /// (e.g. [`fill_ghosts_mirror`]) exchanges halos across the wrap
    /// automatically and applies its physical boundary rule only on the
    /// remaining non-periodic faces. Models never mention periodicity.
    #[must_use]
    pub const fn with_periodic(mut self, periodic: [bool; D]) -> Self {
        self.periodic = periodic;
        self
    }

    /// Which dimensions are periodic.
    #[must_use]
    pub const fn periodic(&self) -> [bool; D] {
        self.periodic
    }

    /// Whole domain as one block — convenient for small runs and tests.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::new`].
    pub fn single_block(
        cells: [usize; D],
        origin: [f64; D],
        spacing: [f64; D],
    ) -> Result<Self, GridError> {
        Self::new(cells, cells, origin, spacing)
    }

    /// Interior cells per dimension over the whole domain.
    #[must_use]
    pub const fn cells(&self) -> [usize; D] {
        self.cells
    }

    /// Physical coordinate of the domain's lower corner.
    #[must_use]
    pub const fn origin(&self) -> [f64; D] {
        self.origin
    }

    /// Interior cells per dimension of one block.
    #[must_use]
    pub const fn block_cells(&self) -> [usize; D] {
        self.block_cells
    }

    /// Number of blocks per dimension.
    #[must_use]
    pub const fn blocks_per_dim(&self) -> [usize; D] {
        self.blocks_per_dim
    }

    /// Multi-index of a block, dimension 0 fastest.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn block_coords(&self, block: BlockId) -> [usize; D] {
        let mut rem = block.index();
        let mut out = [0usize; D];
        for d in 0..D {
            out[d] = rem % self.blocks_per_dim[d];
            rem /= self.blocks_per_dim[d];
        }
        debug_assert_eq!(rem, 0, "block id out of range");
        out
    }

    /// Neighbor block across face `(dim, +1/-1)`: `None` at a non-periodic
    /// domain boundary, the wrapped block in a periodic dimension (which is
    /// `block` itself when one block spans it). This is the topology query
    /// halo exchange is built on.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn face_neighbor(&self, block: BlockId, dim: usize, dir: isize) -> Option<BlockId> {
        let coords = self.block_coords(block);
        let stepped = coords[dim] as isize + dir.signum();
        let nb = self.blocks_per_dim[dim] as isize;
        let next = if (0..nb).contains(&stepped) {
            stepped as usize
        } else if self.periodic[dim] {
            stepped.rem_euclid(nb) as usize
        } else {
            return None;
        };
        let mut id = 0usize;
        let mut stride = 1usize;
        for d in 0..D {
            let c = if d == dim { next } else { coords[d] };
            id += c * stride;
            stride *= self.blocks_per_dim[d];
        }
        Some(BlockId(id as u32))
    }

    fn dims_with_ghosts(&self, ghost: u32) -> [usize; D] {
        let g = ghost as usize;
        std::array::from_fn(|d| self.block_cells[d] + 2 * g)
    }
}

#[inline(always)]
fn linearize<const D: usize>(idx: [isize; D], interior: [usize; D], ghost: u32) -> usize {
    let g = ghost as isize;
    let mut offset = 0usize;
    let mut stride = 1usize;
    for d in 0..D {
        let shifted = idx[d] + g;
        debug_assert!(
            shifted >= 0 && (shifted as usize) < interior[d] + 2 * ghost as usize,
            "index {idx:?} outside block extent {interior:?} with ghost {ghost}"
        );
        offset += shifted as usize * stride;
        stride *= interior[d] + 2 * ghost as usize;
    }
    offset
}

/// Read view of one block's slab. Indices are relative to the interior
/// origin; ghosts sit at `-1..-ghost` and `n..n+ghost-1`.
#[derive(Debug, Clone, Copy)]
pub struct CartesianView<'a, T: Scalar, const D: usize> {
    data: &'a [T],
    interior: [usize; D],
    ghost: u32,
}

impl<'a, T: Scalar, const D: usize> CartesianView<'a, T, D> {
    /// Wrap a ghost-inclusive slab of a uniform box (used by every grid
    /// family whose blocks are uniform boxes — Cartesian and AMR patches).
    #[inline(always)]
    pub(crate) const fn from_raw(data: &'a [T], interior: [usize; D], ghost: u32) -> Self {
        Self {
            data,
            interior,
            ghost,
        }
    }
}

impl<T: Scalar, const D: usize> CartesianView<'_, T, D> {
    /// Value at `idx` (ghost cells addressable with negative indices).
    #[inline(always)]
    #[must_use]
    pub fn get(&self, idx: [isize; D]) -> T {
        self.data[linearize(idx, self.interior, self.ghost)]
    }

    /// Interior extent of the block.
    #[must_use]
    pub const fn interior(&self) -> [usize; D] {
        self.interior
    }

    /// Ghost-ring width of the viewed field.
    #[must_use]
    pub const fn ghost(&self) -> u32 {
        self.ghost
    }
}

/// Mutable view of one block's slab.
#[derive(Debug)]
pub struct CartesianViewMut<'a, T: Scalar, const D: usize> {
    data: &'a mut [T],
    interior: [usize; D],
    ghost: u32,
}

impl<'a, T: Scalar, const D: usize> CartesianViewMut<'a, T, D> {
    /// Mutable counterpart of [`CartesianView::from_raw`].
    #[inline(always)]
    pub(crate) const fn from_raw_mut(data: &'a mut [T], interior: [usize; D], ghost: u32) -> Self {
        Self {
            data,
            interior,
            ghost,
        }
    }
}

impl<T: Scalar, const D: usize> CartesianViewMut<'_, T, D> {
    /// Value at `idx` (ghost cells addressable with negative indices).
    #[inline(always)]
    #[must_use]
    pub fn get(&self, idx: [isize; D]) -> T {
        self.data[linearize(idx, self.interior, self.ghost)]
    }

    /// Write `value` at `idx`.
    #[inline(always)]
    pub fn set(&mut self, idx: [isize; D], value: T) {
        self.data[linearize(idx, self.interior, self.ghost)] = value;
    }

    /// Interior extent of the block.
    #[must_use]
    pub const fn interior(&self) -> [usize; D] {
        self.interior
    }

    /// Ghost-ring width of the viewed field.
    #[must_use]
    pub const fn ghost(&self) -> u32 {
        self.ghost
    }
}

/// Visit every interior index of a block extent, dimension 0 fastest. The
/// workhorse of concrete stencil loops.
pub fn for_each_interior<const D: usize>(interior: [usize; D], mut f: impl FnMut([isize; D])) {
    if interior.contains(&0) {
        return;
    }
    let mut idx = [0isize; D];
    loop {
        f(idx);
        let mut d = 0;
        loop {
            idx[d] += 1;
            if (idx[d] as usize) < interior[d] {
                break;
            }
            idx[d] = 0;
            d += 1;
            if d == D {
                return;
            }
        }
    }
}

/// Visit every index in the half-open box `lo..hi` (signed, so ghost ranges
/// are expressible), dimension 0 fastest.
pub fn for_each_box<const D: usize>(lo: [isize; D], hi: [isize; D], mut f: impl FnMut([isize; D])) {
    if (0..D).any(|d| lo[d] >= hi[d]) {
        return;
    }
    let mut idx = lo;
    loop {
        f(idx);
        let mut d = 0;
        loop {
            idx[d] += 1;
            if idx[d] < hi[d] {
                break;
            }
            idx[d] = lo[d];
            d += 1;
            if d == D {
                return;
            }
        }
    }
}

/// Fill one field's ghost cells: halo exchange across interior block faces
/// — wrapping across the domain in periodic dimensions — and mirror
/// (no-flux) at the remaining non-periodic physical boundaries.
///
/// Works dimension by dimension; each sweep `d` writes only `d`-ghost strips
/// and reads only `d`-interior strips, spanning the *full* ghost-inclusive
/// extent in the other dimensions. After all `D` sweeps, edge and corner
/// ghosts are consistent — explicit face-then-corner copies, generalized
/// to any block count and dimension (including a periodic dimension spanned
/// by a single block, which wraps onto itself). Sequential over blocks for
/// now; the sweeps are embarrassingly parallel per block if this ever shows
/// up in profiles.
pub fn fill_ghosts_mirror<T: Scalar, S: StorageBackend<T>, const D: usize>(
    grid: &CartesianGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
) {
    let ghost = state.layout().ghost(handle.index());
    if ghost == 0 {
        return;
    }
    let g = ghost as isize;
    let n_cells = grid.block_cells();
    for d in 0..D {
        // Strip extent: ghost range in dimension d, full ghost-inclusive
        // range everywhere else.
        let mut lo = [0isize; D];
        let mut hi = [0isize; D];
        for e in 0..D {
            lo[e] = -g;
            hi[e] = n_cells[e] as isize + g;
        }
        let n = n_cells[d] as isize;
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            for dir in [-1isize, 1] {
                // Ghost strip on this side: idx[d] ∈ [-g, 0) or [n, n+g).
                (lo[d], hi[d]) = if dir < 0 { (-g, 0) } else { (n, n + g) };
                if let Some(nb) = grid.face_neighbor(block, d, dir) {
                    // My ghost layer k copies the neighbor's interior:
                    // low side: idx[d] = -1-k  ←  neighbor n-1-k
                    // high side: idx[d] = n+k  ←  neighbor k
                    if nb == block {
                        // Periodic wrap onto itself (one block spans this
                        // dimension): same index map, within one slab.
                        // Sources are d-interior entries, writes are
                        // d-ghost entries, so nothing read is overwritten.
                        let slab = state.slab_mut(block, handle);
                        let mut v = grid.view_mut(block, ghost, slab);
                        for_each_box(lo, hi, |idx| {
                            let mut sidx = idx;
                            sidx[d] = if dir < 0 { n + idx[d] } else { idx[d] - n };
                            v.set(idx, v.get(sidx));
                        });
                    } else {
                        let (dst, src) = state.slab_pair_mut(block, nb, handle);
                        let mut vd = grid.view_mut(block, ghost, dst);
                        let vs = grid.view(nb, ghost, src);
                        for_each_box(lo, hi, |idx| {
                            let mut sidx = idx;
                            sidx[d] = if dir < 0 { n + idx[d] } else { idx[d] - n };
                            vd.set(idx, vs.get(sidx));
                        });
                    }
                } else {
                    // Physical boundary: mirror across the face,
                    // ghost -1-k ← interior k (and n+k ← n-1-k).
                    let slab = state.slab_mut(block, handle);
                    let mut v = grid.view_mut(block, ghost, slab);
                    for_each_box(lo, hi, |idx| {
                        let mut sidx = idx;
                        sidx[d] = if dir < 0 {
                            -1 - idx[d]
                        } else {
                            2 * n - 1 - idx[d]
                        };
                        v.set(idx, v.get(sidx));
                    });
                }
            }
        }
    }
}

/// Set one field's interior from a function of the cell-center coordinate
/// — the declarative form of an initial condition:
///
/// ```
/// # use legendre::core::state::StateBuilder;
/// # use legendre::core::storage::{DenseStorage, SystemAllocator};
/// # use legendre::geometry::cartesian::{CartesianGrid, fill_from_fn};
/// # let grid = CartesianGrid::new([8; 2], [4; 2], [0.0; 2], [0.5; 2]).unwrap();
/// # let mut builder = StateBuilder::<f64>::new();
/// # let u = builder.register("u", 1);
/// # let mut state: legendre::core::state::State<f64, DenseStorage<f64>> =
/// #     builder.build(&grid, &SystemAllocator);
/// fill_from_fn(&grid, &mut state, u, |[x, y]| (x * y).cos());
/// ```
///
/// Writes interior cells only; ghosts become consistent through the
/// model's `fill_ghosts` before the first evaluation, as always. `f` is
/// `FnMut` so closures may carry state (e.g. a seeded generator).
pub fn fill_from_fn<T: Scalar, S: StorageBackend<T>, const D: usize>(
    grid: &CartesianGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
    mut f: impl FnMut([f64; D]) -> T,
) {
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        let mut v = state.view_mut(grid, block, handle);
        for_each_interior(grid.block_cells(), |idx| {
            v.set(idx, f(grid.cell_center(block, idx)));
        });
    }
}

impl<const D: usize> Grid for CartesianGrid<D> {
    const DIM: usize = D;
    type Point = [f64; D];
    type Index = [isize; D];
    type View<'a, T: Scalar> = CartesianView<'a, T, D>;
    type ViewMut<'a, T: Scalar> = CartesianViewMut<'a, T, D>;

    fn num_blocks(&self) -> usize {
        self.blocks_per_dim.iter().product()
    }

    fn block_len(&self, _block: BlockId, ghost: u32) -> usize {
        self.dims_with_ghosts(ghost).iter().product()
    }

    fn level(&self, _block: BlockId) -> u8 {
        0
    }

    fn spacing(&self, _block: BlockId) -> [f64; D] {
        self.spacing
    }

    #[allow(clippy::needless_range_loop)]
    fn cell_center(&self, block: BlockId, idx: [isize; D]) -> [f64; D] {
        let coords = self.block_coords(block);
        let mut out = [0.0f64; D];
        for d in 0..D {
            let global = coords[d] * self.block_cells[d];
            out[d] = (global as f64 + idx[d] as f64 + 0.5).mul_add(self.spacing[d], self.origin[d]);
        }
        out
    }

    fn cell_key(&self, _block: BlockId, ghost: u32, offset: usize) -> Option<u64> {
        // Undo the dimension-0-fastest linearization over the ghosted box,
        // reject ghost coordinates, and re-linearize over the interior so
        // the id is independent of this field's ghost width.
        let g = ghost as usize;
        let mut rem = offset;
        let mut key = 0usize;
        let mut stride = 1usize;
        for d in 0..D {
            let ghosted = self.block_cells[d] + 2 * g;
            let c = rem % ghosted;
            rem /= ghosted;
            if c < g || c >= self.block_cells[d] + g {
                return None;
            }
            key += (c - g) * stride;
            stride *= self.block_cells[d];
        }
        debug_assert_eq!(rem, 0, "slab offset out of range");
        Some(key as u64)
    }

    fn view<'a, T: Scalar>(
        &'a self,
        block: BlockId,
        ghost: u32,
        data: &'a [T],
    ) -> CartesianView<'a, T, D> {
        debug_assert_eq!(data.len(), self.block_len(block, ghost));
        CartesianView {
            data,
            interior: self.block_cells,
            ghost,
        }
    }

    fn view_mut<'a, T: Scalar>(
        &'a self,
        block: BlockId,
        ghost: u32,
        data: &'a mut [T],
    ) -> CartesianViewMut<'a, T, D> {
        debug_assert_eq!(data.len(), self.block_len(block, ghost));
        CartesianViewMut {
            data,
            interior: self.block_cells,
            ghost,
        }
    }
}
