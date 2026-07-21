//! The AMR patch hierarchy behind the [`Grid`] trait.
//!
//! An [`AmrGrid`] is a **flat forest of axis-aligned patches**: level 0 is
//! exactly the base `CartesianGrid`'s block tiling, finer levels are
//! [`CellBox`]es of uniform cells with spacing `h_ℓ = h_{ℓ−1}/r_ℓ`. Every
//! patch is a block ([`BlockId`] = patch index, level-major), and views
//! are the plain Cartesian box views — a stencil kernel cannot tell a
//! patch from a uniform-grid block.
//!
//! Invariants, enforced at construction so everything downstream may
//! assume them:
//!
//! - **Disjointness**: same-level patches never overlap (Berger–Rigoutsos
//!   output satisfies this by construction; hand-built hierarchies are
//!   checked).
//! - **Proper nesting** (Berger–Oliger §2): every level-ℓ patch, coarsened
//!   to level ℓ−1 and grown by one cell, lies inside the union of level
//!   ℓ−1 patches — i.e. fine patches keep an interior margin from the
//!   coarse level's edge — except where clipped by the physical domain
//!   boundary.
//!
//! The hierarchy is immutable: a regrid constructs a *new* `AmrGrid` and
//! migrates state (which keeps `Grid: Sync` trivially sound, exactly as
//! for `CartesianGrid`).

use super::cluster::CellBox;
use crate::{
    core::storage::Scalar,
    geometry::{
        GridError,
        cartesian::{CartesianGrid, CartesianView, CartesianViewMut},
        grid::{BlockId, BoxedBlocks, Grid},
    },
};

/// One patch of the hierarchy: a box of uniform cells at one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmrPatch<const D: usize> {
    /// Refinement level (0 = base).
    pub level: u8,
    /// Cell box in *this level's* index space.
    pub bx: CellBox<D>,
}

impl<const D: usize> AmrPatch<D> {
    /// Interior cells per dimension.
    #[must_use]
    pub fn extent(&self) -> [usize; D] {
        std::array::from_fn(|d| self.bx.extent(d) as usize)
    }
}

/// A block-structured AMR hierarchy; see the module docs for invariants.
#[derive(Debug, Clone)]
pub struct AmrGrid<const D: usize> {
    base: CartesianGrid<D>,
    /// `ratios[ℓ]` is the refinement ratio from level ℓ to ℓ+1.
    ratios: Vec<u32>,
    /// All patches, level-major: `BlockId` is an index into this.
    patches: Vec<AmrPatch<D>>,
    /// Precomputed per-level cell spacing.
    level_spacing: Vec<[f64; D]>,
    /// `level_start[l]..level_start[l+1]` are level `l`'s patch indices.
    level_start: Vec<u32>,
    /// Per populated level: cell → owning patch (`u32::MAX` = uncovered),
    /// linearized dimension-0-fastest over the level domain. Ghost fill,
    /// prolongation, and migration do O(1) lookups instead of scanning
    /// patch lists per cell.
    lookup: Vec<Vec<u32>>,
}

impl<const D: usize> AmrGrid<D> {
    /// Build a hierarchy from explicit per-level patch boxes.
    ///
    /// `levels[ℓ]` holds the level-(ℓ+1) patches, in level-(ℓ+1) cell
    /// coordinates; level 0 is always the base grid's own block tiling.
    /// Ratios apply in order; `levels.len() ≤ ratios.len()`.
    ///
    /// # Errors
    ///
    /// [`GridError::AmrRatio`] for a ratio < 2 or more levels than
    /// ratios; [`GridError::AmrEmptyPatch`] for a zero-cell box;
    /// [`GridError::AmrOverlap`] for intersecting same-level boxes;
    /// [`GridError::AmrNotNested`] for a proper-nesting violation.
    ///
    /// # Panics
    ///
    /// Does not panic: the internal `expect` reads the level-0 spacing
    /// pushed unconditionally above it.
    pub fn from_patches(
        base: CartesianGrid<D>,
        ratios: &[u32],
        levels: &[Vec<CellBox<D>>],
    ) -> Result<Self, GridError> {
        if levels.len() > ratios.len() {
            return Err(GridError::AmrRatio {
                levels: levels.len(),
                ratios: ratios.len(),
            });
        }
        for &r in ratios {
            if r < 2 {
                return Err(GridError::AmrRatio {
                    levels: levels.len(),
                    ratios: ratios.len(),
                });
            }
        }
        // The full ratio list is *capacity*: an adaptivity policy may
        // populate levels the initial hierarchy leaves empty.
        let ratios = ratios.to_vec();

        // Level 0: the base grid's blocks as patches.
        let mut patches: Vec<AmrPatch<D>> = (0..base.num_blocks())
            .map(|b| {
                let coords = base.block_coords(BlockId(b as u32));
                let cells = base.block_cells();
                AmrPatch {
                    level: 0,
                    bx: CellBox {
                        lo: std::array::from_fn(|d| (coords[d] * cells[d]) as i64),
                        hi: std::array::from_fn(|d| ((coords[d] + 1) * cells[d]) as i64),
                    },
                }
            })
            .collect();

        // Finer levels, validated against the level below.
        let mut level_start = vec![0u32, patches.len() as u32];
        let mut coarser: Vec<CellBox<D>> = patches.iter().map(|p| p.bx).collect();
        for (l, boxes) in levels.iter().enumerate() {
            let level = (l + 1) as u8;
            let r = i64::from(ratios[l]);
            let domain = level_domain(&base, &ratios, level);
            for (i, bx) in boxes.iter().enumerate() {
                if bx.cells() == 0 {
                    return Err(GridError::AmrEmptyPatch { level });
                }
                if bx.clipped(&domain) != *bx {
                    return Err(GridError::AmrNotNested { level });
                }
                if (0..D).any(|d| bx.lo[d] % r != 0 || bx.hi[d] % r != 0) {
                    return Err(GridError::AmrMisaligned { level });
                }
                for other in &boxes[..i] {
                    if bx.intersects(other) {
                        return Err(GridError::AmrOverlap { level });
                    }
                }
                // Proper nesting: the coarsened box grown by one cell
                // (clipped to the coarse domain) must be covered by the
                // level below.
                let coarse = CellBox {
                    lo: std::array::from_fn(|d| bx.lo[d].div_euclid(r)),
                    hi: std::array::from_fn(|d| (bx.hi[d] + r - 1).div_euclid(r)),
                };
                let need = coarse
                    .grown(1)
                    .clipped(&level_domain(&base, &ratios, level - 1));
                if !covered_by(&need, &coarser) {
                    return Err(GridError::AmrNotNested { level });
                }
            }
            patches.extend(boxes.iter().map(|&bx| AmrPatch { level, bx }));
            level_start.push(patches.len() as u32);
            coarser.clone_from(boxes);
        }

        let mut level_spacing = vec![base.spacing(BlockId(0))];
        for &r in &ratios {
            let prev = *level_spacing.last().expect("level 0 spacing");
            level_spacing.push(std::array::from_fn(|d| prev[d] / f64::from(r)));
        }

        // Paint the per-level ownership maps (construction/regrid only).
        let mut lookup = Vec::with_capacity(level_start.len() - 1);
        for level in 0..level_start.len() - 1 {
            let domain = level_domain(&base, &ratios, level as u8);
            let extent: [usize; D] = std::array::from_fn(|d| domain.hi[d] as usize);
            let mut map = vec![u32::MAX; extent.iter().product()];
            for pid in level_start[level]..level_start[level + 1] {
                let bx = patches[pid as usize].bx;
                paint(&mut map, &extent, &bx, pid);
            }
            lookup.push(map);
        }

        Ok(Self {
            base,
            ratios,
            patches,
            level_spacing,
            level_start,
            lookup,
        })
    }

    /// The base (level-0) grid.
    #[must_use]
    pub const fn base(&self) -> &CartesianGrid<D> {
        &self.base
    }

    /// Refinement ratios between consecutive levels.
    #[must_use]
    pub fn ratios(&self) -> &[u32] {
        &self.ratios
    }

    /// Number of *populated* levels (≥ 1; level 0 is the base).
    #[must_use]
    pub const fn num_levels(&self) -> usize {
        self.level_start.len() - 1
    }

    /// Maximum levels this hierarchy's ratio list allows.
    #[must_use]
    pub const fn max_levels(&self) -> usize {
        self.ratios.len() + 1
    }

    /// Cell spacing at refinement `level` (valid up to `max_levels`, even
    /// for levels not currently populated — a subcycling scheme reads the
    /// stability law at each level's resolution).
    #[must_use]
    pub fn spacing_at_level(&self, level: u8) -> [f64; D] {
        self.level_spacing[level as usize]
    }

    /// All patches, level-major (`BlockId` indexes this slice).
    #[must_use]
    pub fn patches(&self) -> &[AmrPatch<D>] {
        &self.patches
    }

    /// The patch behind a block id.
    #[must_use]
    pub fn patch(&self, block: BlockId) -> &AmrPatch<D> {
        &self.patches[block.index()]
    }

    /// The whole domain as a cell box at level `level`'s resolution.
    #[must_use]
    pub fn level_domain(&self, level: u8) -> CellBox<D> {
        level_domain(&self.base, &self.ratios, level)
    }

    /// The blocks of one level, in id order (empty for levels the
    /// hierarchy does not currently populate — e.g. after a regrid
    /// removed its last patch).
    pub fn blocks_at(&self, level: u8) -> impl Iterator<Item = BlockId> + use<D> {
        let l = level as usize;
        let (start, end) = if l + 1 < self.level_start.len() {
            (self.level_start[l], self.level_start[l + 1])
        } else {
            (0, 0)
        };
        (start..end).map(BlockId)
    }

    /// The level-`level` patch containing `cell`, if any. O(1): a lookup
    /// in the level's precomputed ownership map.
    #[must_use]
    pub fn find_patch(&self, level: u8, cell: [i64; D]) -> Option<BlockId> {
        let map = self.lookup.get(level as usize)?;
        let domain = self.level_domain(level);
        if !domain.contains(cell) {
            return None;
        }
        let mut off = 0usize;
        let mut stride = 1usize;
        for (c, hi) in cell.iter().zip(&domain.hi) {
            off += *c as usize * stride;
            stride *= *hi as usize;
        }
        let pid = map[off];
        (pid != u32::MAX).then_some(BlockId(pid))
    }
}

fn level_domain<const D: usize>(base: &CartesianGrid<D>, ratios: &[u32], level: u8) -> CellBox<D> {
    let scale: i64 = ratios[..level as usize]
        .iter()
        .map(|&r| i64::from(r))
        .product();
    let cells = base.cells();
    CellBox {
        lo: [0; D],
        hi: std::array::from_fn(|d| cells[d] as i64 * scale),
    }
}

/// Paint `bx`'s cells with `pid` in a dimension-0-fastest map of `extent`.
fn paint<const D: usize>(map: &mut [u32], extent: &[usize; D], bx: &CellBox<D>, pid: u32) {
    if (0..D).any(|d| bx.lo[d] >= bx.hi[d]) {
        return;
    }
    let mut c = bx.lo;
    loop {
        let mut off = 0usize;
        let mut stride = 1usize;
        for d in 0..D {
            off += c[d] as usize * stride;
            stride *= extent[d];
        }
        // Runs along axis 0 are contiguous: fill the whole row at once.
        let run = (bx.hi[0] - bx.lo[0]) as usize;
        map[off..off + run].fill(pid);
        c[0] = bx.hi[0] - 1;
        let mut d = 0;
        loop {
            c[d] += 1;
            if c[d] < bx.hi[d] {
                break;
            }
            c[d] = bx.lo[d];
            d += 1;
            if d == D {
                return;
            }
        }
    }
}

/// Whether every cell of `need` lies in some box of `cover`. `need` is a
/// small perimeter box; this runs at construction/regrid, never per step.
fn covered_by<const D: usize>(need: &CellBox<D>, cover: &[CellBox<D>]) -> bool {
    // Walk cells of `need`, skipping ahead within covering boxes along
    // axis 0 for speed.
    let mut idx = need.lo;
    if need.cells() == 0 {
        return true;
    }
    loop {
        match cover.iter().find(|b| b.contains(idx)) {
            None => return false,
            Some(b) => idx[0] = b.hi[0] - 1, // rest of this run is covered
        }
        // Advance to the next cell in `need`, axis 0 fastest.
        let mut d = 0;
        loop {
            idx[d] += 1;
            if idx[d] < need.hi[d] {
                break;
            }
            idx[d] = need.lo[d];
            d += 1;
            if d == D {
                return true;
            }
        }
    }
}

impl<const D: usize> Grid for AmrGrid<D> {
    const DIM: usize = D;
    type Point = [f64; D];
    type Index = [isize; D];
    type View<'a, T: Scalar> = CartesianView<'a, T, D>;
    type ViewMut<'a, T: Scalar> = CartesianViewMut<'a, T, D>;

    fn num_blocks(&self) -> usize {
        self.patches.len()
    }

    fn block_len(&self, block: BlockId, ghost: u32) -> usize {
        let g = ghost as usize;
        self.patch(block)
            .extent()
            .iter()
            .map(|&n| n + 2 * g)
            .product()
    }

    fn level(&self, block: BlockId) -> u8 {
        self.patch(block).level
    }

    fn spacing(&self, block: BlockId) -> [f64; D] {
        self.level_spacing[self.patch(block).level as usize]
    }

    fn finest_spacing(&self) -> [f64; D] {
        // The deepest level the ratio capacity allows, even if unpopulated.
        *self.level_spacing.last().expect("level 0 always exists")
    }

    fn cell_center(&self, block: BlockId, idx: [isize; D]) -> [f64; D] {
        let p = self.patch(block);
        let h = self.level_spacing[p.level as usize];
        let origin = self.base.origin();
        std::array::from_fn(|d| {
            ((p.bx.lo[d] + idx[d] as i64) as f64 + 0.5).mul_add(h[d], origin[d])
        })
    }

    fn cell_key(&self, block: BlockId, ghost: u32, offset: usize) -> Option<u64> {
        // Same de-linearization as the Cartesian grid, over this patch's
        // extent: the id is ghost-independent and per-block.
        let extent = self.patch(block).extent();
        let g = ghost as usize;
        let mut rem = offset;
        let mut key = 0usize;
        let mut stride = 1usize;
        for n in extent {
            let ghosted = n + 2 * g;
            let c = rem % ghosted;
            rem /= ghosted;
            if c < g || c >= n + g {
                return None;
            }
            key += (c - g) * stride;
            stride *= n;
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
        CartesianView::from_raw(data, self.patch(block).extent(), ghost)
    }

    fn view_mut<'a, T: Scalar>(
        &'a self,
        block: BlockId,
        ghost: u32,
        data: &'a mut [T],
    ) -> CartesianViewMut<'a, T, D> {
        debug_assert_eq!(data.len(), self.block_len(block, ghost));
        CartesianViewMut::from_raw_mut(data, self.patch(block).extent(), ghost)
    }
}

impl<const D: usize> BoxedBlocks<D> for AmrGrid<D> {
    fn block_extent(&self, block: BlockId) -> [usize; D] {
        self.patch(block).extent()
    }
}

#[cfg(test)]
// Exact float equality is deliberate: level-0 geometry must be *identical*
// to the base grid's, and level spacings are exact binary halvings.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn base_8x8() -> CartesianGrid<2> {
        CartesianGrid::new([8, 8], [4, 4], [0.0, 0.0], [0.5, 0.5]).unwrap()
    }

    #[test]
    fn level_zero_mirrors_the_base_tiling() {
        let amr = AmrGrid::from_patches(base_8x8(), &[], &[]).unwrap();
        assert_eq!(amr.num_blocks(), 4);
        assert_eq!(amr.num_levels(), 1);
        // Geometry agrees with the base grid block for block.
        let base = base_8x8();
        for b in 0..4 {
            let block = BlockId(b);
            assert_eq!(amr.level(block), 0);
            assert_eq!(amr.spacing(block), base.spacing(block));
            assert_eq!(amr.block_len(block, 1), base.block_len(block, 1));
            assert_eq!(
                amr.cell_center(block, [1, 2]),
                base.cell_center(block, [1, 2])
            );
        }
    }

    #[test]
    fn fine_patch_geometry_nests_in_the_coarse_cells() {
        // One 8×8 fine patch (level-1 coords [4,12)²) over the domain
        // center; coarsened+grown it needs [1,7)², covered by the base.
        let fine = CellBox {
            lo: [4, 4],
            hi: [12, 12],
        };
        let amr = AmrGrid::from_patches(base_8x8(), &[2], &[vec![fine]]).unwrap();
        assert_eq!(amr.num_blocks(), 5);
        let fb = BlockId(4);
        assert_eq!(amr.level(fb), 1);
        assert_eq!(amr.spacing(fb), [0.25, 0.25]);
        assert_eq!(amr.block_len(fb, 1), 10 * 10);

        // The mean of the two fine cell centers spanning coarse cell c
        // is that coarse cell's center, per axis.
        let coarse_center = amr.base().cell_center(BlockId(0), [2, 2]);
        let f0 = amr.cell_center(fb, [0, 0]); // fine cell 4 = coarse cell 2
        let f1 = amr.cell_center(fb, [1, 1]);
        for d in 0..2 {
            assert_relative_eq!(coarse_center[d], 0.5 * (f0[d] + f1[d]), epsilon = 1e-14);
        }
    }

    #[test]
    fn construction_rejects_invalid_hierarchies() {
        let overlap = vec![
            CellBox {
                lo: [4, 4],
                hi: [10, 10],
            },
            CellBox {
                lo: [8, 8],
                hi: [12, 12],
            },
        ];
        assert!(matches!(
            AmrGrid::from_patches(base_8x8(), &[2], &[overlap]),
            Err(GridError::AmrOverlap { level: 1 })
        ));

        // Touching the domain edge is fine (clipped nesting margin)…
        let at_edge = vec![CellBox {
            lo: [0, 0],
            hi: [8, 8],
        }];
        assert!(AmrGrid::from_patches(base_8x8(), &[2], &[at_edge]).is_ok());

        // …but a level-2 patch not properly nested in level 1 is not:
        // level 1 covers [4,12)², so level 2 needs [coarse+margin] inside.
        let l1 = vec![CellBox {
            lo: [4, 4],
            hi: [12, 12],
        }];
        let l2_bad = vec![CellBox {
            lo: [8, 8],
            hi: [16, 16],
        }]; // coarsens to [4,8)+margin ⊄ [4,12) interior? -> lo edge at 4 needs 3
        assert!(matches!(
            AmrGrid::from_patches(base_8x8(), &[2, 2], &[l1.clone(), l2_bad]),
            Err(GridError::AmrNotNested { level: 2 })
        ));
        let l2_good = vec![CellBox {
            lo: [10, 10],
            hi: [18, 18],
        }]; // coarsens to [5,9), grown [4,10) ⊆ [4,12)
        assert!(AmrGrid::from_patches(base_8x8(), &[2, 2], &[l1, l2_good]).is_ok());

        assert!(matches!(
            AmrGrid::from_patches(
                base_8x8(),
                &[1],
                &[vec![CellBox {
                    lo: [0, 0],
                    hi: [4, 4]
                }]]
            ),
            Err(GridError::AmrRatio { .. })
        ));
        assert!(matches!(
            AmrGrid::from_patches(
                base_8x8(),
                &[],
                &[vec![CellBox {
                    lo: [0, 0],
                    hi: [4, 4]
                }]]
            ),
            Err(GridError::AmrRatio { .. })
        ));
    }

    #[test]
    fn out_of_domain_patch_is_rejected() {
        let outside = vec![CellBox {
            lo: [12, 12],
            hi: [20, 20],
        }]; // level-1 domain is [0,16)²
        assert!(matches!(
            AmrGrid::from_patches(base_8x8(), &[2], &[outside]),
            Err(GridError::AmrNotNested { level: 1 })
        ));
    }

    #[test]
    fn cell_keys_enumerate_the_interior_uniquely() {
        let fine = CellBox {
            lo: [4, 4],
            hi: [10, 8],
        }; // 6×4 patch
        let amr = AmrGrid::from_patches(base_8x8(), &[2], &[vec![fine]]).unwrap();
        let block = BlockId(4);
        let ghost = 1;
        let len = amr.block_len(block, ghost);
        let mut seen = std::collections::HashSet::new();
        for off in 0..len {
            if let Some(k) = amr.cell_key(block, ghost, off) {
                assert!(seen.insert(k), "duplicate cell key {k}");
                assert!(k < 24, "key {k} out of interior range");
            }
        }
        assert_eq!(seen.len(), 24, "every interior cell keyed exactly once");
    }
}
