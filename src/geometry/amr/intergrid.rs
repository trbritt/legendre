//! Intergrid operations: the three Berger–Oliger grid interactions.
//!
//! - [`restrict`]: fine→coarse *updating* — conservative averaging of fine
//!   interiors onto the coarse cells beneath them. Mandatory, not
//!   cosmetic: without it the coarse solution disperses under a patch and
//!   contaminates the very values later used for fine-boundary
//!   interpolation.
//! - [`fill_ghosts_mirror`]: ghost filling per level — same-level halo
//!   exchange where an abutting patch exists, mirror (no-flux) reflection
//!   at physical domain boundaries, and coarse→fine **bilinear
//!   prolongation** everywhere else.
//!
//! Models on an [`AmrGrid`] call both, in that order, from their
//! `fill_ghosts` hook:
//!
//! ```text
//! fn fill_ghosts(…) {
//!     amr::restrict(grid, state, self.u);
//!     amr::fill_ghosts_mirror(grid, state, self.u);
//! }
//! ```
//!
//! **Determinism:** every ghost value is computed from *interior* cells of
//! donor patches (exchange copies interiors; prolongation samples coarse
//! interiors), so results are independent of patch visit order, and the
//! sweeps are bit-reproducible under any scheduling.
//!
//! **v1 constraints** (checked by debug assertions): a field's ghost width
//! on refined levels must not exceed the refinement ratio (so the
//! interpolation stencil stays within the properly-nested coarse margin),
//! and the base grid's periodic topology is not yet honored across AMR
//! levels.

use super::{cluster::CellBox, grid::AmrGrid};
use crate::{
    core::{
        state::{FieldHandle, State},
        storage::{Real, StorageBackend},
    },
    geometry::grid::{BlockId, Grid},
};

/// Visit every cell of `bx`, dimension 0 fastest.
fn for_each_cell<const D: usize>(bx: &CellBox<D>, mut f: impl FnMut([i64; D])) {
    if (0..D).any(|d| bx.lo[d] >= bx.hi[d]) {
        return;
    }
    let mut c = bx.lo;
    loop {
        f(c);
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

/// Visit every cell of the ghost ring of `bx` grown by `g`: per axis, the
/// two strips normal to it — earlier axes span the grown range, later
/// axes the interior — a disjoint, complete cover that never touches the
/// interior.
fn for_each_ring_cell<const D: usize>(bx: &CellBox<D>, g: i64, mut f: impl FnMut([i64; D])) {
    for d in 0..D {
        for (lo_d, hi_d) in [(bx.lo[d] - g, bx.lo[d]), (bx.hi[d], bx.hi[d] + g)] {
            let mut strip = *bx;
            for e in 0..d {
                strip.lo[e] = bx.lo[e] - g;
                strip.hi[e] = bx.hi[e] + g;
            }
            strip.lo[d] = lo_d;
            strip.hi[d] = hi_d;
            for_each_cell(&strip, &mut f);
        }
    }
}

/// Reflect a cell coordinate across the physical (no-flux) domain
/// boundary; interior coordinates pass through unchanged.
fn mirror<const D: usize>(cell: [i64; D], domain: &CellBox<D>) -> [i64; D] {
    std::array::from_fn(|d| {
        if cell[d] < domain.lo[d] {
            2 * domain.lo[d] - 1 - cell[d]
        } else if cell[d] >= domain.hi[d] {
            2 * domain.hi[d] - 1 - cell[d]
        } else {
            cell[d]
        }
    })
}

/// Inject fine solutions onto the coarse cells beneath them: each covered
/// coarse cell becomes the mean of its `r^D` fine children. Runs finest →
/// coarsest so multi-level chains propagate all the way down.
pub fn restrict<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
) {
    for fine_level in (1..grid.num_levels() as u8).rev() {
        restrict_level(grid, state, handle, fine_level);
    }
}

/// Restrict a single transition: level `fine_level` onto `fine_level − 1`.
/// The unit the Berger–Oliger subcycle synchronizes with after each fine
/// cycle completes.
pub fn restrict_level<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
    fine_level: u8,
) {
    debug_assert!(fine_level >= 1, "restriction needs a coarser level");
    let ghost = state.layout().ghost(handle.index());
    let r = i64::from(grid.ratios()[fine_level as usize - 1]);
    let inv = T::from_f64(1.0 / (r as f64).powi(D as i32));
    for fb in grid.blocks_at(fine_level) {
        let fp = *grid.patch(fb);
        // Ratio alignment makes this coarsening exact.
        let covered = CellBox {
            lo: std::array::from_fn(|d| fp.bx.lo[d] / r),
            hi: std::array::from_fn(|d| fp.bx.hi[d] / r),
        };
        for cb in grid.blocks_at(fine_level - 1) {
            let cp = *grid.patch(cb);
            let overlap = covered.clipped(&cp.bx);
            if overlap.cells() == 0 {
                continue;
            }
            let (dst, src) = state.slab_pair_mut(cb, fb, handle);
            let mut coarse = grid.view_mut(cb, ghost, dst);
            let fine = grid.view(fb, ghost, src);
            for_each_cell(&overlap, |c| {
                let mut sum = T::ZERO;
                let children = CellBox::<D> {
                    lo: std::array::from_fn(|d| c[d] * r - fp.bx.lo[d]),
                    hi: std::array::from_fn(|d| (c[d] + 1) * r - fp.bx.lo[d]),
                };
                for_each_cell(&children, |k| {
                    sum += fine.get(std::array::from_fn(|d| k[d] as isize));
                });
                let local: [isize; D] = std::array::from_fn(|d| (c[d] - cp.bx.lo[d]) as isize);
                coarse.set(local, sum * inv);
            });
        }
    }
}

/// Fill every ghost cell of every patch: same-level exchange, physical
/// mirror, coarse→fine prolongation (see the module docs). The global-dt
/// path, where every level is at the same time.
pub fn fill_ghosts_mirror<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
) {
    let ghost = state.layout().ghost(handle.index());
    if ghost == 0 {
        return;
    }
    let mut scratch: Vec<([isize; D], T)> = Vec::new();
    for level in 0..grid.num_levels() as u8 {
        fill_one_level(grid, state, None, handle, level, ghost, &mut scratch);
    }
}

/// Fill one level's ghosts under **subcycling**, applying a precomputed
/// [`FillPlan`] (see [`build_fill_plan`]).
///
/// Same-level exchange and physical mirror read the current `state`.
/// `time_interp = Some((old, alpha))` makes the coarse→fine prolongation
/// interpolate *in time* between `old` (the coarser level at the start of
/// its step) and the current `state` (the coarser level after its step),
/// at fraction `alpha ∈ [0, 1]`. Level 0 never prolongs, so it passes
/// `None`.
///
/// The plan carries the transfer topology (which cells each ghost reads,
/// and the prolongation weights) so this reuses it across every substep's
/// fill without recomputing `find_patch`/interpolation stencils.
pub fn fill_level<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    time_interp: Option<(&State<T, S>, f64)>,
    handle: FieldHandle<T>,
    level: u8,
    plan: &FillPlan<D>,
) {
    if state.layout().ghost(handle.index()) == 0 {
        return;
    }
    let mut staged: Vec<T> = Vec::new();
    let alpha = time_interp.map_or(0.0, |(_, a)| a);
    let old = time_interp.map(|(o, _)| o);
    for entry in &plan.blocks[level as usize] {
        staged.clear();
        for op in &entry.ops {
            let value = match op.source {
                FillSource::Copy { block, index } => state.view(grid, block, handle).get(index),
                FillSource::Prolong { start, len } => {
                    let corners = &plan.corners[start..start + len];
                    let interp = |src: &State<T, S>| {
                        corners.iter().fold(T::ZERO, |acc, c| {
                            acc + src.view(grid, c.block, handle).get(c.index)
                                * T::from_f64(c.weight)
                        })
                    };
                    let v_new = interp(state);
                    old.map_or(v_new, |old| {
                        interp(old) * T::from_f64(1.0 - alpha) + v_new * T::from_f64(alpha)
                    })
                }
            };
            staged.push(value);
        }
        let mut v = state.view_mut(grid, entry.block, handle);
        for (op, &value) in entry.ops.iter().zip(&staged) {
            v.set(op.target, value);
        }
    }
}

/// The transfer topology of a per-level ghost fill for one ghost width.
///
/// For each block, it records where each ghost cell reads from — a single
/// interior cell for exchange/mirror, or a weighted set of coarse cells
/// for prolongation. Built once via [`build_fill_plan`] and reused across
/// every substep's fill, so `find_patch` and the interpolation stencils
/// are computed once, not per fill.
pub struct FillPlan<const D: usize> {
    /// `blocks[level]` = the fills for that level's blocks.
    blocks: Vec<Vec<BlockFill<D>>>,
    /// Flat pool of prolongation corners, indexed by `FillSource::Prolong`.
    corners: Vec<Corner<D>>,
}

struct BlockFill<const D: usize> {
    block: BlockId,
    ops: Vec<GhostOp<D>>,
}

struct GhostOp<const D: usize> {
    /// Ghost cell to write, interior-relative to this block.
    target: [isize; D],
    source: FillSource<D>,
}

enum FillSource<const D: usize> {
    /// Copy one interior cell (same-level exchange, or physical mirror).
    Copy { block: BlockId, index: [isize; D] },
    /// Bilinear prolongation from `corners[start..start+len]`.
    Prolong { start: usize, len: usize },
}

struct Corner<const D: usize> {
    block: BlockId,
    index: [isize; D],
    weight: f64,
}

/// Precompute the [`FillPlan`] for fields of the given `ghost` width on
/// `grid` — the per-cell `find_patch` lookups and interpolation weights
/// that would otherwise be recomputed on every fill.
#[must_use]
pub fn build_fill_plan<const D: usize>(grid: &AmrGrid<D>, ghost: u32) -> FillPlan<D> {
    let mut corners = Vec::new();
    let mut blocks = Vec::with_capacity(grid.num_levels());
    let g = i64::from(ghost);
    for level in 0..grid.num_levels() as u8 {
        let domain = grid.level_domain(level);
        let mut level_fills = Vec::new();
        for pb in grid.blocks_at(level) {
            let p = *grid.patch(pb);
            let mut ops = Vec::new();
            for_each_ring_cell(&p.bx, g, |cell| {
                let target: [isize; D] = std::array::from_fn(|d| (cell[d] - p.bx.lo[d]) as isize);
                let dst = mirror(cell, &domain);
                let source = grid.find_patch(level, dst).map_or_else(
                    || {
                        let start = corners.len();
                        prolong_corners(grid, level, dst, &mut corners);
                        FillSource::Prolong {
                            start,
                            len: corners.len() - start,
                        }
                    },
                    |src| {
                        let sp = grid.patch(src);
                        let index = std::array::from_fn(|d| (dst[d] - sp.bx.lo[d]) as isize);
                        FillSource::Copy { block: src, index }
                    },
                );
                ops.push(GhostOp { target, source });
            });
            level_fills.push(BlockFill { block: pb, ops });
        }
        blocks.push(level_fills);
    }
    FillPlan { blocks, corners }
}

/// Record the bilinear prolongation stencil for `cell` at `level` (from
/// level `level-1`) into `corners` — the recording counterpart of
/// [`prolonged`].
fn prolong_corners<const D: usize>(
    grid: &AmrGrid<D>,
    level: u8,
    cell: [i64; D],
    corners: &mut Vec<Corner<D>>,
) {
    let r = f64::from(grid.ratios()[level as usize - 1]);
    let coarse_domain = grid.level_domain(level - 1);
    let mut c0 = [0i64; D];
    let mut w = [0.0f64; D];
    for d in 0..D {
        let pc = (cell[d] as f64 + 0.5) / r - 0.5;
        let f = pc.floor();
        c0[d] = f as i64;
        w[d] = pc - f;
    }
    for corner in 0u32..(1 << D) {
        let mut cc = c0;
        let mut weight = 1.0;
        for d in 0..D {
            if corner >> d & 1 == 1 {
                cc[d] += 1;
                weight *= w[d];
            } else {
                weight *= 1.0 - w[d];
            }
        }
        if weight == 0.0 {
            continue;
        }
        for (d, c) in cc.iter_mut().enumerate() {
            if *c < coarse_domain.lo[d] {
                *c = 2 * coarse_domain.lo[d] - 1 - *c;
            } else if *c >= coarse_domain.hi[d] {
                *c = 2 * coarse_domain.hi[d] - 1 - *c;
            }
        }
        let qb = grid
            .find_patch(level - 1, cc)
            .expect("proper nesting guarantees a coarse donor for prolongation");
        let q = grid.patch(qb);
        let index = std::array::from_fn(|d| (cc[d] - q.bx.lo[d]) as isize);
        corners.push(Corner {
            block: qb,
            index,
            weight,
        });
    }
}

/// The shared per-level ghost fill. `time_interp = Some((old, alpha))`
/// blends the coarse prolongation source in time; `None` prolongs from the
/// current state alone (global dt). Gather-then-write with ring-only
/// iteration (ghosts are a perimeter, not an area); `scratch` is reused
/// across levels.
fn fill_one_level<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    time_interp: Option<(&State<T, S>, f64)>,
    handle: FieldHandle<T>,
    level: u8,
    ghost: u32,
    scratch: &mut Vec<([isize; D], T)>,
) {
    debug_assert!(
        level == 0 || u64::from(ghost) <= u64::from(grid.ratios()[level as usize - 1]),
        "ghost width may not exceed the refinement ratio (interpolation \
         stencil would leave the properly-nested coarse margin)"
    );
    let domain = grid.level_domain(level);
    for pb in grid.blocks_at(level) {
        let p = *grid.patch(pb);
        scratch.clear();
        for_each_ring_cell(&p.bx, i64::from(ghost), |cell| {
            // Physical mirror: reflect out-of-domain coordinates.
            let target: [i64; D] = std::array::from_fn(|d| {
                if cell[d] < domain.lo[d] {
                    2 * domain.lo[d] - 1 - cell[d]
                } else if cell[d] >= domain.hi[d] {
                    2 * domain.hi[d] - 1 - cell[d]
                } else {
                    cell[d]
                }
            });
            let value = grid.find_patch(level, target).map_or_else(
                || {
                    debug_assert!(level > 0, "level 0 tiles the domain");
                    let v_new = prolonged(grid, state, handle, level, target);
                    match time_interp {
                        Some((old, alpha)) => {
                            let v_old = prolonged(grid, old, handle, level, target);
                            v_old * T::from_f64(1.0 - alpha) + v_new * T::from_f64(alpha)
                        }
                        None => v_new,
                    }
                },
                |src| {
                    // Same-level donor (possibly this very patch, for
                    // mirror ghosts): read its interior.
                    let sp = grid.patch(src);
                    let local: [isize; D] =
                        std::array::from_fn(|d| (target[d] - sp.bx.lo[d]) as isize);
                    state.view(grid, src, handle).get(local)
                },
            );
            let local: [isize; D] = std::array::from_fn(|d| (cell[d] - p.bx.lo[d]) as isize);
            scratch.push((local, value));
        });
        let mut v = state.view_mut(grid, pb, handle);
        for &(local, value) in scratch.iter() {
            v.set(local, value);
        }
    }
}

/// Bilinear (per-axis linear) interpolation of the level-`level−1`
/// solution at fine cell `cell`'s center, from coarse *interior* values.
/// Proper nesting plus `ghost ≤ ratio` guarantee every corner donor
/// exists (after mirroring across the physical boundary).
fn prolonged<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &State<T, S>,
    handle: FieldHandle<T>,
    level: u8,
    cell: [i64; D],
) -> T {
    let r = f64::from(grid.ratios()[level as usize - 1]);
    let coarse_domain = grid.level_domain(level - 1);
    let mut c0 = [0i64; D];
    let mut w = [0.0f64; D];
    for d in 0..D {
        let pc = (cell[d] as f64 + 0.5) / r - 0.5;
        let f = pc.floor();
        c0[d] = f as i64;
        w[d] = pc - f;
    }
    let mut acc = T::ZERO;
    for corner in 0u32..(1 << D) {
        let mut cc = c0;
        let mut weight = 1.0;
        for d in 0..D {
            if corner >> d & 1 == 1 {
                cc[d] += 1;
                weight *= w[d];
            } else {
                weight *= 1.0 - w[d];
            }
        }
        if weight == 0.0 {
            continue;
        }
        // Mirror corners that fall outside the physical domain, matching
        // the no-flux boundary rule.
        for (d, c) in cc.iter_mut().enumerate() {
            if *c < coarse_domain.lo[d] {
                *c = 2 * coarse_domain.lo[d] - 1 - *c;
            } else if *c >= coarse_domain.hi[d] {
                *c = 2 * coarse_domain.hi[d] - 1 - *c;
            }
        }
        let qb = grid
            .find_patch(level - 1, cc)
            .expect("proper nesting guarantees a coarse donor for prolongation");
        let q = grid.patch(qb);
        let local: [isize; D] = std::array::from_fn(|d| (cc[d] - q.bx.lo[d]) as isize);
        acc += state.view(grid, qb, handle).get(local) * T::from_f64(weight);
    }
    acc
}
