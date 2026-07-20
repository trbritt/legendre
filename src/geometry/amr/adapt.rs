//! The Berger–Oliger adaptivity policy: tag → cluster → nest → migrate.
//!
//! [`BergerOliger`] implements [`Adapt`] for [`AmrGrid`]: at the regrid
//! cadence it flags cells through a [`TagCells`] criterion, dilates the
//! flags by the buffer zone (what lets a moving feature stay inside its
//! patch between regrids), clusters them with Berger–Rigoutsos, rebuilds
//! the hierarchy **finest level first** (adding nesting flags so each
//! level properly contains the one above), and migrates the solution onto
//! the new hierarchy — same-level regions by direct copy, newly refined
//! regions by bilinear prolongation from the finest old data beneath them.
//!
//! Everything here is a pure function of the state: tags, clustering, and
//! migration are deterministic, so adaptive runs remain bit-reproducible.
//! Stochastic driver streams re-key across a regrid (block ids change) —
//! the same documented behavior class as any other re-keying.

use super::{
    cluster::{CellBox, ClusterParams, cluster},
    grid::AmrGrid,
};
use crate::{
    core::{
        simulation::Adapt,
        state::{FieldHandle, State},
        storage::{Allocator, Real, StorageBackend},
    },
    geometry::cartesian::for_each_interior,
};

/// A refinement criterion: flags the cells of one level that need finer
/// resolution.
pub trait TagCells<T: Real, const D: usize>: Send + Sync {
    /// Make the state consistent for tagging (restriction, ghost fills of
    /// the fields the criterion reads). Called once per regrid.
    fn prepare<S: StorageBackend<T>>(&self, _grid: &AmrGrid<D>, _state: &mut State<T, S>) {}

    /// Append the level-`level` cells needing refinement, in level-`level`
    /// cell coordinates.
    fn tag_level<S: StorageBackend<T>>(
        &self,
        grid: &AmrGrid<D>,
        state: &State<T, S>,
        level: u8,
        flags: &mut Vec<[i64; D]>,
    );
}

/// Tag where the undivided central-difference gradient of one field
/// exceeds a threshold — the pragmatic criterion for interface-tracking
/// problems (|∇φ|·h is scale-free across levels).
///
/// The field is named, not handled: taggers are constructed before the
/// simulation registers fields, so the handle is resolved from the
/// state's layout at regrid time.
#[derive(Debug, Clone, Copy)]
pub struct GradientTagger {
    /// Registered name of the field whose gradient drives refinement.
    pub field: &'static str,
    /// Flag where `max_d |u(i+e_d) − u(i−e_d)|/2` exceeds this.
    pub threshold: f64,
}

impl GradientTagger {
    fn handle<T: Real, S: StorageBackend<T>>(&self, state: &State<T, S>) -> FieldHandle<T> {
        state
            .field(self.field)
            .unwrap_or_else(|| panic!("GradientTagger field {:?} is not registered", self.field))
    }
}

impl<T: Real, const D: usize> TagCells<T, D> for GradientTagger {
    fn prepare<S: StorageBackend<T>>(&self, grid: &AmrGrid<D>, state: &mut State<T, S>) {
        let h = self.handle(state);
        super::intergrid::restrict(grid, state, h);
        super::intergrid::fill_ghosts_mirror(grid, state, h);
    }

    fn tag_level<S: StorageBackend<T>>(
        &self,
        grid: &AmrGrid<D>,
        state: &State<T, S>,
        level: u8,
        flags: &mut Vec<[i64; D]>,
    ) {
        let h = self.handle(state);
        for pb in grid.blocks_at(level) {
            let p = *grid.patch(pb);
            let v = state.view(grid, pb, h);
            for_each_interior(p.extent(), |idx| {
                let mut g = 0.0f64;
                for d in 0..D {
                    let mut hi = idx;
                    hi[d] += 1;
                    let mut lo = idx;
                    lo[d] -= 1;
                    g = g.max(((v.get(hi) - v.get(lo)).to_f64() * 0.5).abs());
                }
                if g > self.threshold {
                    flags.push(std::array::from_fn(|d| p.bx.lo[d] + idx[d] as i64));
                }
            });
        }
    }
}

/// Knobs of the regrid loop.
#[derive(Debug, Clone, Copy)]
pub struct RegridPolicy {
    /// Regrid every this many steps (the first call, step 0, always
    /// regrids — that is the initial refinement from the initial
    /// conditions).
    pub every: u64,
    /// Buffer zone in coarse cells: flags are dilated by this before
    /// clustering, so features stay inside their patches between regrids
    /// (Berger–Oliger: buffer and cadence trade off directly).
    pub buffer: i64,
    /// Berger–Rigoutsos knobs.
    pub cluster: ClusterParams,
}

impl Default for RegridPolicy {
    fn default() -> Self {
        Self {
            every: 4,
            buffer: 2,
            cluster: ClusterParams::default(),
        }
    }
}

/// The Berger–Oliger adaptivity policy; see the module docs.
#[derive(Debug, Clone)]
pub struct BergerOliger<Tag> {
    tagger: Tag,
    policy: RegridPolicy,
}

impl<Tag> BergerOliger<Tag> {
    /// Assemble the policy from a tagging criterion and regrid knobs.
    pub const fn new(tagger: Tag, policy: RegridPolicy) -> Self {
        Self { tagger, policy }
    }
}

impl<T, S, A, Tag, const D: usize> Adapt<AmrGrid<D>, T, S, A> for BergerOliger<Tag>
where
    T: Real,
    S: StorageBackend<T>,
    A: Allocator<T, Storage = S>,
    Tag: TagCells<T, D>,
{
    fn regrid(
        &mut self,
        grid: &AmrGrid<D>,
        state: &mut State<T, S>,
        alloc: &A,
        step: u64,
    ) -> Option<(AmrGrid<D>, State<T, S>)> {
        if !step.is_multiple_of(self.policy.every) {
            return None;
        }
        self.tagger.prepare(grid, state);

        // Rebuild refined levels finest-first so nesting flags from each
        // new fine level land in the flags of the level being built below
        // it (Berger–Oliger §4: use the most accurate estimates, ensure
        // proper nesting by construction).
        let capacity = grid.max_levels() - 1;
        let mut new_levels: Vec<Vec<CellBox<D>>> = vec![Vec::new(); capacity];
        for target in (1..=capacity).rev() {
            let src_level = (target - 1) as u8;
            let mut flags: Vec<[i64; D]> = Vec::new();
            if (src_level as usize) < grid.num_levels() {
                self.tagger.tag_level(grid, state, src_level, &mut flags);
            }
            // Buffer dilation before clustering keeps B–R's disjointness.
            let b = self.policy.buffer;
            if b > 0 {
                let seeds = std::mem::take(&mut flags);
                for c in seeds {
                    let ball = CellBox {
                        lo: std::array::from_fn(|d| c[d] - b),
                        hi: std::array::from_fn(|d| c[d] + b + 1),
                    };
                    push_cells(&ball, &mut flags);
                }
            }
            // Nesting flags: the just-built finer level, coarsened to
            // `target` and grown by one, must be covered — flag its
            // parent cells at `src_level`.
            if target < capacity && !new_levels[target].is_empty() {
                let r_fine = i64::from(grid.ratios()[target]);
                let r_src = i64::from(grid.ratios()[target - 1]);
                for bx in &new_levels[target] {
                    let need = CellBox::<D> {
                        lo: std::array::from_fn(|d| bx.lo[d] / r_fine - 1),
                        hi: std::array::from_fn(|d| bx.hi[d] / r_fine + 1),
                    };
                    let parents = CellBox::<D> {
                        lo: std::array::from_fn(|d| need.lo[d].div_euclid(r_src)),
                        hi: std::array::from_fn(|d| (need.hi[d] + r_src - 1).div_euclid(r_src)),
                    };
                    push_cells(&parents, &mut flags);
                }
            }
            // Clip to the source level's domain, dedupe, cluster, refine.
            let domain = grid.level_domain(src_level);
            flags.retain(|c| domain.contains(*c));
            flags.sort_unstable();
            flags.dedup();
            if flags.is_empty() {
                continue;
            }
            let r = i64::from(grid.ratios()[target - 1]);
            new_levels[target - 1] = cluster(&mut flags, &self.policy.cluster)
                .into_iter()
                .map(|bx| CellBox {
                    lo: std::array::from_fn(|d| bx.lo[d] * r),
                    hi: std::array::from_fn(|d| bx.hi[d] * r),
                })
                .collect();
        }
        while new_levels.last().is_some_and(Vec::is_empty) {
            new_levels.pop();
        }

        if hierarchy_acceptable(grid, &new_levels) {
            return None;
        }
        let new_grid = AmrGrid::from_patches(grid.base().clone(), grid.ratios(), &new_levels)
            .expect(
                "regrid construction is valid by construction (nesting flags, aligned refinement)",
            );
        let mut new_state = state.reshaped(&new_grid, alloc);
        migrate(grid, state, &new_grid, &mut new_state);
        Some((new_grid, new_state))
    }
}

/// Append every cell of `bx` to `out`.
fn push_cells<const D: usize>(bx: &CellBox<D>, out: &mut Vec<[i64; D]>) {
    if (0..D).any(|d| bx.lo[d] >= bx.hi[d]) {
        return;
    }
    let mut c = bx.lo;
    loop {
        out.push(c);
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

/// Whether the current hierarchy may keep serving the proposed one:
/// same level count, every proposed box already covered by the current
/// level, and the current level not over-refined beyond 3/2 the proposal.
/// Berger–Oliger: "once we have good clusters they do not change very
/// fast" — threshold flicker must not force a rebuild (with its
/// migration, reallocation, and stochastic re-keying) every cadence.
fn hierarchy_acceptable<const D: usize>(
    grid: &AmrGrid<D>,
    new_levels: &[Vec<CellBox<D>>],
) -> bool {
    if new_levels.len() != grid.num_levels() - 1 {
        return false;
    }
    for (l, proposed) in new_levels.iter().enumerate() {
        let current: Vec<CellBox<D>> = grid
            .blocks_at((l + 1) as u8)
            .map(|b| grid.patch(b).bx)
            .collect();
        let proposed_cells: u64 = proposed.iter().map(CellBox::cells).sum();
        let current_cells: u64 = current.iter().map(CellBox::cells).sum();
        if current_cells > proposed_cells + proposed_cells / 2 {
            return false; // over-refined: rebuild to shrink
        }
        for bx in proposed {
            if !box_covered(bx, &current) {
                return false; // the feature escaped the current patches
            }
        }
    }
    true
}

/// Whether every cell of `need` lies in some box of `cover` (run-skipping
/// walk along axis 0; regrid-cadence cost, never per step).
fn box_covered<const D: usize>(need: &CellBox<D>, cover: &[CellBox<D>]) -> bool {
    if need.cells() == 0 {
        return true;
    }
    let mut c = need.lo;
    loop {
        match cover.iter().find(|b| b.contains(c)) {
            None => return false,
            Some(b) => c[0] = b.hi[0] - 1,
        }
        let mut d = 0;
        loop {
            c[d] += 1;
            if c[d] < need.hi[d] {
                break;
            }
            c[d] = need.lo[d];
            d += 1;
            if d == D {
                return true;
            }
        }
    }
}


/// Copy the old solution onto the new hierarchy: level 0 verbatim (the
/// base tiling never changes), refined cells from the same-level old
/// patch when one covers them, otherwise by bilinear interpolation of the
/// finest old data beneath (recursing down levels as needed).
fn migrate<T: Real, S: StorageBackend<T>, const D: usize>(
    old_grid: &AmrGrid<D>,
    old_state: &State<T, S>,
    new_grid: &AmrGrid<D>,
    new_state: &mut State<T, S>,
) {
    for f in 0..old_state.layout().num_fields() {
        let h = State::<T, S>::handle_at(f);
        for nb in new_grid.blocks_at(0) {
            new_state
                .slab_mut(nb, h)
                .copy_from_slice(old_state.slab(nb, h));
        }
        for level in 1..new_grid.num_levels() as u8 {
            for nb in new_grid.blocks_at(level) {
                let np = *new_grid.patch(nb);
                let mut v = new_state.view_mut(new_grid, nb, h);
                for_each_interior(np.extent(), |idx| {
                    let cell: [i64; D] = std::array::from_fn(|d| np.bx.lo[d] + idx[d] as i64);
                    v.set(idx, sample(old_grid, old_state, h, level, cell));
                });
            }
        }
    }
}

/// The old solution at level-`level` cell `cell`: the covering old patch's
/// value if that level reaches here, else bilinear interpolation from one
/// level down (recursively — level 0 always covers).
fn sample<T: Real, S: StorageBackend<T>, const D: usize>(
    grid: &AmrGrid<D>,
    state: &State<T, S>,
    handle: FieldHandle<T>,
    level: u8,
    cell: [i64; D],
) -> T {
    if (level as usize) < grid.num_levels()
        && let Some(b) = grid.find_patch(level, cell)
    {
        let p = grid.patch(b);
        let local: [isize; D] = std::array::from_fn(|d| (cell[d] - p.bx.lo[d]) as isize);
        return state.view(grid, b, handle).get(local);
    }
    debug_assert!(level > 0, "level 0 tiles the domain");
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
        for (d, c) in cc.iter_mut().enumerate() {
            if *c < coarse_domain.lo[d] {
                *c = 2 * coarse_domain.lo[d] - 1 - *c;
            } else if *c >= coarse_domain.hi[d] {
                *c = 2 * coarse_domain.hi[d] - 1 - *c;
            }
        }
        acc += sample(grid, state, handle, level - 1, cc) * T::from_f64(weight);
    }
    acc
}
