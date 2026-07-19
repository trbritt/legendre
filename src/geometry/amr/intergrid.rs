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
    geometry::grid::Grid,
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

/// Inject fine solutions onto the coarse cells beneath them: each covered
/// coarse cell becomes the mean of its `r^D` fine children. Runs finest →
/// coarsest so multi-level chains propagate all the way down.
pub fn restrict<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
) {
    let ghost = state.layout().ghost(handle.index());
    for level in (1..grid.num_levels() as u8).rev() {
        let r = i64::from(grid.ratios()[level as usize - 1]);
        let inv = T::from_f64(1.0 / (r as f64).powi(D as i32));
        for fb in grid.blocks_at(level) {
            let fp = *grid.patch(fb);
            // Ratio alignment makes this coarsening exact.
            let covered = CellBox {
                lo: std::array::from_fn(|d| fp.bx.lo[d] / r),
                hi: std::array::from_fn(|d| fp.bx.hi[d] / r),
            };
            for cb in grid.blocks_at(level - 1) {
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
}

/// Fill every ghost cell of every patch: same-level exchange, physical
/// mirror, coarse→fine prolongation (see the module docs).
pub fn fill_ghosts_mirror<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &mut State<T, S>,
    handle: FieldHandle<T>,
) {
    let ghost = state.layout().ghost(handle.index());
    if ghost == 0 {
        return;
    }
    for level in 0..grid.num_levels() as u8 {
        debug_assert!(
            level == 0 || u64::from(ghost) <= u64::from(grid.ratios()[level as usize - 1]),
            "ghost width may not exceed the refinement ratio (interpolation \
             stencil would leave the properly-nested coarse margin)"
        );
        let domain = grid.level_domain(level);
        for pb in grid.blocks_at(level) {
            let p = *grid.patch(pb);
            for_each_cell(&p.bx.grown(i64::from(ghost)), |cell| {
                if p.bx.contains(cell) {
                    return;
                }
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
                        prolonged(grid, state, handle, level, target)
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
                state.view_mut(grid, pb, handle).set(local, value);
            });
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
