//! Simulation state: named fields over blocked storage.
//!
//! - **Block-major layout.** `State` is `blocks × fields`, not `fields ×
//!   blocks`. A block bundles every field slab it owns, so the scheduler can
//!   hand each worker a disjoint `&mut BlockStorage` with no interior
//!   mutability and no unsafe. This is also the AMR migration unit: refining a
//!   block replaces one `BlockStorage` with children.
//!
//! - **Typed handles, untyped storage.** Registration returns a
//!   `FieldHandle<T>` (a typed index); storage stays homogeneous slabs. The
//!   handle is `Copy` and its type parameter keeps field access honest at
//!   compile time without any per-access lookup cost.
//!
//! - **Ghost cells live in the slab.** Each field declares its ghost width at
//!   registration (its maximum stencil support). Slabs are allocated
//!   ghost-inclusive, which makes every state-shaped buffer (integrator stages,
//!   RHS accumulators, noise) *slab-congruent* with the state itself.
//!
//! - **Integrators see a vector space.** Because all state-shaped buffers are
//!   slab-congruent, time integration is pure slab arithmetic ([`State::axpy`],
//!   [`State::copy_from`]) — no spatial indexing, no grid knowledge, no
//!   per-scheme stencil code. Ghost entries of stage buffers hold garbage; they
//!   are refilled before any stencil reads them. This is what makes multi-stage
//!   schemes trivial.

use crate::{
    core::{
        driver::{Driver, DriverKind, DriverSet},
        scheduler::Scheduler,
        storage::{Allocator, Real, Scalar, StorageBackend},
    },
    geometry::grid::{BlockId, Grid},
};
use std::{marker::PhantomData, sync::Arc};

/// Typed index of a registered field.
///
/// Obtained from [`StateBuilder::register`] and stored by the model; valid
/// for every `State` built from that builder's layout (including stage
/// buffers created with [`State::like`]).
#[derive(Debug)]
pub struct FieldHandle<T: Scalar> {
    index: usize,
    _marker: PhantomData<fn() -> T>,
}

// Manual impls: `derive` would bound `T: Clone`/`T: Copy`, but the handle is
// an index — copyable regardless of `T`.
#[allow(clippy::expl_impl_clone_on_copy)]
impl<T: Scalar> Clone for FieldHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Scalar> Copy for FieldHandle<T> {}

impl<T: Scalar> FieldHandle<T> {
    /// Position of this field in the state layout.
    #[inline(always)]
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Declaration of one field.
///
/// Bundles a name for observers/IO with the ghost-ring width the field's
/// stencils require and the field's **driver incidence**: the set of
/// [`Driver`]s that move it. Every buffer-allocation and traffic question
/// reduces to this one set — a *static* field (e.g. a grain-orientation
/// map) is simply one moved by no driver: it lives in the state and in
/// snapshot buffers, but per-driver tendency buffers allocate nothing for
/// it and the vector-space operations skip it.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    /// Field name, used by observers and IO backends as the column name.
    pub name: &'static str,
    /// Ghost-ring width (the maximum stencil support of any operator applied
    /// to this field).
    pub ghost: u32,
    /// The drivers that move this field. Consulted only at allocation
    /// time; per-driver buffers ([`State::like_for`]) carry storage for
    /// exactly the fields a driver moves.
    pub drivers: Vec<Driver>,
}

impl FieldSpec {
    /// Whether `driver` moves this field.
    #[inline]
    #[must_use]
    pub fn is_driven_by(&self, driver: Driver) -> bool {
        self.drivers.contains(&driver)
    }

    /// Whether no driver moves this field (zero time derivative).
    #[inline]
    #[must_use]
    pub const fn is_static(&self) -> bool {
        self.drivers.is_empty()
    }
}

/// Immutable description of a state's fields, shared (via `Arc`) by every
/// state-shaped buffer so congruence is guaranteed by construction.
#[derive(Debug)]
pub struct StateLayout {
    specs: Vec<FieldSpec>,
}

impl StateLayout {
    /// The registered field declarations, in registration order.
    #[must_use]
    pub fn specs(&self) -> &[FieldSpec] {
        &self.specs
    }

    /// Ghost-ring width of the field at `index`.
    #[inline(always)]
    #[must_use]
    pub fn ghost(&self, index: usize) -> u32 {
        self.specs[index].ghost
    }

    /// Number of registered fields.
    #[must_use]
    pub const fn num_fields(&self) -> usize {
        self.specs.len()
    }
}

/// Collects field registrations before allocation. Models register their
/// fields here during simulation setup; this is the only place field handles
/// are minted.
pub struct StateBuilder<T: Scalar> {
    specs: Vec<FieldSpec>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar> Default for StateBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> StateBuilder<T> {
    /// An empty builder with no fields registered.
    #[must_use]
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Register a field moved by the time driver only (the common case:
    /// pure drift dynamics) and return its typed handle.
    pub fn register(&mut self, name: &'static str, ghost: u32) -> FieldHandle<T> {
        self.push(FieldSpec {
            name,
            ghost,
            drivers: vec![Driver::Time],
        })
    }

    /// Register a field with its full driver incidence — every [`Driver`]
    /// that moves it, [`Driver::Time`] included if the field has drift.
    /// Per-driver buffers carry storage only for the fields a driver
    /// moves, so declaring the incidence is what keeps a one-field noise
    /// term from paying whole-state traffic.
    pub fn register_driven(
        &mut self,
        name: &'static str,
        ghost: u32,
        drivers: &[Driver],
    ) -> FieldHandle<T> {
        self.push(FieldSpec {
            name,
            ghost,
            drivers: drivers.to_vec(),
        })
    }

    fn push(&mut self, spec: FieldSpec) -> FieldHandle<T> {
        let index = self.specs.len();
        self.specs.push(spec);
        FieldHandle {
            index,
            _marker: PhantomData,
        }
    }

    /// Register a field with zero time derivative.
    ///
    /// It is integrated past (never touched by `axpy`/noise), and tendency
    /// buffers do not allocate storage for it — at scale this is real
    /// memory: one static field on a 10080² grid is ~0.8 GB per buffer that
    /// stages would otherwise carry.
    pub fn register_static(&mut self, name: &'static str, ghost: u32) -> FieldHandle<T> {
        self.push(FieldSpec {
            name,
            ghost,
            drivers: Vec::new(),
        })
    }

    /// Allocate the state: the single up-front allocation of field memory.
    pub fn build<G: Grid, A: Allocator<T>>(self, grid: &G, alloc: &A) -> State<T, A::Storage> {
        let layout = Arc::new(StateLayout { specs: self.specs });
        let blocks = (0..grid.num_blocks())
            .map(|b| BlockStorage::allocate(&layout, grid, BlockId(b as u32), alloc, |_| true))
            .collect();
        State { layout, blocks }
    }
}

/// Every field slab owned by one block. The unit of parallel dispatch and of
/// AMR migration.
#[derive(Debug)]
pub struct BlockStorage<T: Scalar, S: StorageBackend<T>> {
    fields: Vec<S>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, S: StorageBackend<T>> BlockStorage<T, S> {
    /// Allocate one block's slabs; fields rejected by `keep` get
    /// zero-length storage and are skipped by every vector-space operation.
    fn allocate<G: Grid, A: Allocator<T, Storage = S>>(
        layout: &StateLayout,
        grid: &G,
        block: BlockId,
        alloc: &A,
        keep: impl Fn(&FieldSpec) -> bool,
    ) -> Self {
        Self {
            fields: layout
                .specs
                .iter()
                .map(|spec| {
                    if keep(spec) {
                        alloc.allocate(grid.block_len(block, spec.ghost))
                    } else {
                        alloc.allocate(0)
                    }
                })
                .collect(),
            _marker: PhantomData,
        }
    }

    /// Bind this block's storage to its layout for typed mutable access.
    /// Used by the scheduler when dispatching a block to a worker.
    pub fn bind_mut<'a>(&'a mut self, layout: &'a StateLayout) -> BlockStateMut<'a, T, S> {
        BlockStateMut {
            layout,
            fields: &mut self.fields,
            _marker: PhantomData,
        }
    }
}

impl<T: Real, S: StorageBackend<T>> BlockStorage<T, S> {
    /// Zero every slab of this block. Lets a dispatch fuse "reset the
    /// accumulator" with the work that fills it, instead of paying a
    /// separate whole-buffer pass and barrier.
    pub fn fill_zero(&mut self) {
        for field in &mut self.fields {
            field.as_mut_slice().fill(T::ZERO);
        }
    }
}

/// The complete simulation state (or any state-shaped buffer: integrator
/// stages, RHS accumulators, noise amplitudes).
#[derive(Debug)]
pub struct State<T: Scalar, S: StorageBackend<T>> {
    layout: Arc<StateLayout>,
    blocks: Vec<BlockStorage<T, S>>,
}

impl<T: Scalar, S: StorageBackend<T>> State<T, S> {
    /// The shared field layout this state was built from.
    #[must_use]
    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Number of blocks this state spans.
    #[must_use]
    pub const fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// A congruent, zeroed buffer sharing this state's layout, carrying
    /// every field including statics (snapshot rings, stage *states*).
    #[must_use]
    pub fn like<G: Grid, A: Allocator<T, Storage = S>>(&self, grid: &G, alloc: &A) -> Self {
        self.like_impl(grid, alloc, |_| true)
    }

    /// A congruent *tendency* buffer for one driver: only the fields that
    /// driver moves (see [`StateBuilder::register_driven`]) carry storage;
    /// everything else gets a zero-length slab the vector-space operations
    /// skip. `like_for(Driver::Time)` is the classic dY/dt buffer (static
    /// fields excluded); `like_for(Driver::Wiener(j))` is driver `j`'s
    /// amplitude buffer, so zeroing, amplitude evaluation, and increment
    /// application touch exactly the driven fields' memory — a driver that
    /// moves one field of a large system costs one field of traffic per
    /// step.
    #[must_use]
    pub fn like_for<G: Grid, A: Allocator<T, Storage = S>>(
        &self,
        grid: &G,
        alloc: &A,
        driver: Driver,
    ) -> Self {
        self.like_impl(grid, alloc, |spec| spec.is_driven_by(driver))
    }

    /// A buffer with this state's layout allocated over a *different*
    /// grid's block structure — the regrid migration target. Carries
    /// every field (statics included); the shared layout keeps existing
    /// [`FieldHandle`]s valid on the new state.
    #[must_use]
    pub fn reshaped<G: Grid, A: Allocator<T, Storage = S>>(&self, grid: &G, alloc: &A) -> Self {
        let blocks = (0..grid.num_blocks())
            .map(|b| BlockStorage::allocate(&self.layout, grid, BlockId(b as u32), alloc, |_| true))
            .collect();
        Self {
            layout: Arc::clone(&self.layout),
            blocks,
        }
    }

    /// Mint the handle of the field at `index` (regrid migration sweeps
    /// every field without knowing the model's handles). Kept crate-only:
    /// the builder remains the sole public mint.
    pub(crate) const fn handle_at(index: usize) -> FieldHandle<T> {
        FieldHandle {
            index,
            _marker: PhantomData,
        }
    }

    /// Handle of the field registered under `name`, if any — how
    /// components constructed *before* field registration (adaptivity
    /// taggers, observers) resolve the fields they act on.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<FieldHandle<T>> {
        self.layout
            .specs()
            .iter()
            .position(|s| s.name == name)
            .map(|index| FieldHandle {
                index,
                _marker: PhantomData,
            })
    }

    fn like_impl<G: Grid, A: Allocator<T, Storage = S>>(
        &self,
        grid: &G,
        alloc: &A,
        keep: impl Fn(&FieldSpec) -> bool,
    ) -> Self {
        let blocks = (0..self.blocks.len())
            .map(|b| BlockStorage::allocate(&self.layout, grid, BlockId(b as u32), alloc, &keep))
            .collect();
        Self {
            layout: Arc::clone(&self.layout),
            blocks,
        }
    }

    /// Raw slab of one field on one block (ghost-inclusive).
    #[inline(always)]
    #[must_use]
    pub fn slab(&self, block: BlockId, handle: FieldHandle<T>) -> &[T] {
        self.blocks[block.index()].fields[handle.index()].as_slice()
    }

    /// Mutable raw slab of one field on one block (ghost-inclusive).
    #[inline(always)]
    pub fn slab_mut(&mut self, block: BlockId, handle: FieldHandle<T>) -> &mut [T] {
        self.blocks[block.index()].fields[handle.index()].as_mut_slice()
    }

    /// Typed read view of one field on one block.
    #[inline(always)]
    pub fn view<'a, G: Grid>(
        &'a self,
        grid: &'a G,
        block: BlockId,
        handle: FieldHandle<T>,
    ) -> G::View<'a, T> {
        let ghost = self.layout.ghost(handle.index());
        grid.view(block, ghost, self.slab(block, handle))
    }

    /// Typed mutable view of one field on one block.
    #[inline(always)]
    pub fn view_mut<'a, G: Grid>(
        &'a mut self,
        grid: &'a G,
        block: BlockId,
        handle: FieldHandle<T>,
    ) -> G::ViewMut<'a, T> {
        let ghost = self.layout.ghost(handle.index());
        let slab = self.blocks[block.index()].fields[handle.index()].as_mut_slice();
        grid.view_mut(block, ghost, slab)
    }

    /// Split into layout + per-block storage for parallel dispatch. The
    /// scheduler pairs each `&mut BlockStorage` with a worker; disjointness
    /// is structural.
    pub fn split_blocks_mut(&mut self) -> (&StateLayout, &mut [BlockStorage<T, S>]) {
        (&self.layout, &mut self.blocks)
    }

    /// One field's slab mutably on `dst` and immutably on `src` at once —
    /// the primitive halo exchange is built on.
    ///
    /// # Panics
    ///
    /// Panics if `dst == src` or either block is out of range.
    pub fn slab_pair_mut(
        &mut self,
        dst: BlockId,
        src: BlockId,
        handle: FieldHandle<T>,
    ) -> (&mut [T], &[T]) {
        let [d, s] = self
            .blocks
            .get_disjoint_mut([dst.index(), src.index()])
            .expect("slab_pair_mut requires two distinct, in-range blocks");
        (
            d.fields[handle.index()].as_mut_slice(),
            s.fields[handle.index()].as_slice(),
        )
    }
}

/// Per-block kernels for the vector-space operations. Free functions so the
/// serial and scheduler-driven entry points share one implementation; every
/// operation is keyed by block index only, so results are identical under
/// any scheduling.
fn block_axpy<T: Real, S: StorageBackend<T>>(
    mine: &mut BlockStorage<T, S>,
    theirs: &BlockStorage<T, S>,
    alpha: T,
) {
    for (a, b) in mine.fields.iter_mut().zip(&theirs.fields) {
        if a.is_empty() || b.is_empty() {
            continue; // static field paired with a tendency buffer
        }
        for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
            *x += alpha * *y;
        }
    }
}

fn block_copy<T: Real, S: StorageBackend<T>>(
    mine: &mut BlockStorage<T, S>,
    theirs: &BlockStorage<T, S>,
) {
    for (a, b) in mine.fields.iter_mut().zip(&theirs.fields) {
        if a.is_empty() || b.is_empty() {
            continue; // static field paired with a tendency buffer
        }
        a.as_mut_slice().copy_from_slice(b.as_slice());
    }
}

#[allow(clippy::too_many_arguments)]
fn block_apply_driver<T: Real, S: StorageBackend<T>, G: Grid>(
    grid: &G,
    layout: &StateLayout,
    mine: &mut BlockStorage<T, S>,
    amp: &BlockStorage<T, S>,
    block: BlockId,
    driver: Driver,
    dt: f64,
    seed: u64,
    salt: u64,
) {
    // The driver's kernel owns the measure scaling (dt, √dt·ξ, …); the
    // variant match happens here, once per slab — never per cell.
    for (f, (x, a)) in mine.fields.iter_mut().zip(&amp.fields).enumerate() {
        if x.is_empty() || a.is_empty() {
            continue; // field this buffer carries no storage for
        }
        driver.apply_slab(
            grid,
            block,
            layout.ghost(f),
            dt,
            seed,
            salt,
            a.as_slice(),
            x.as_mut_slice(),
        );
    }
}

impl<T: Real, S: StorageBackend<T>> State<T, S> {
    /// `self += alpha * other`, elementwise over every slab. Requires a
    /// congruent buffer (same layout, same grid).
    pub fn axpy(&mut self, alpha: T, other: &Self) {
        debug_assert!(Arc::ptr_eq(&self.layout, &other.layout));
        for (mine, theirs) in self.blocks.iter_mut().zip(&other.blocks) {
            block_axpy(mine, theirs, alpha);
        }
    }

    /// Scheduler-driven [`State::axpy`]: these vector-space operations are
    /// memory-bound over the whole state, so at large volume they must be
    /// dispatched like any other block work.
    pub fn axpy_with<Sch: Scheduler>(&mut self, scheduler: &Sch, alpha: T, other: &Self) {
        debug_assert!(Arc::ptr_eq(&self.layout, &other.layout));
        scheduler.for_each_block_mut(
            &mut self.blocks,
            || (),
            |id, mine, ()| {
                block_axpy(mine, &other.blocks[id.index()], alpha);
            },
        );
    }

    /// Overwrite every slab from a congruent buffer.
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert!(Arc::ptr_eq(&self.layout, &other.layout));
        for (mine, theirs) in self.blocks.iter_mut().zip(&other.blocks) {
            block_copy(mine, theirs);
        }
    }

    /// Scheduler-driven [`State::copy_from`].
    pub fn copy_from_with<Sch: Scheduler>(&mut self, scheduler: &Sch, other: &Self) {
        debug_assert!(Arc::ptr_eq(&self.layout, &other.layout));
        scheduler.for_each_block_mut(
            &mut self.blocks,
            || (),
            |id, mine, ()| {
                block_copy(mine, &other.blocks[id.index()]);
            },
        );
    }

    /// Zero every slab (used to reset RHS/noise accumulators between stages).
    pub fn fill_zero(&mut self) {
        for block in &mut self.blocks {
            for field in &mut block.fields {
                field.as_mut_slice().fill(T::ZERO);
            }
        }
    }

    /// Scheduler-driven [`State::fill_zero`].
    pub fn fill_zero_with<Sch: Scheduler>(&mut self, scheduler: &Sch) {
        scheduler.for_each_block_mut(
            &mut self.blocks,
            || (),
            |_, block, ()| {
                for field in &mut block.fields {
                    field.as_mut_slice().fill(T::ZERO);
                }
            },
        );
    }

    /// `self += dμ ∘ amplitude` for one driver: each slab goes through the
    /// driver's kernel ([`DriverKind::apply_slab`]), which owns the
    /// measure-correct scaling over a step of size `dt` — a plain `dt`
    /// axpy for [`Driver::Time`], a per-cell `√dt·ξ` increment keyed by
    /// `(seed, salt, stream, block, cell)` for [`Driver::Wiener`] —
    /// deterministic and schedule-independent (see [`crate::util::rng`]).
    /// The cell id comes from [`Grid::cell_key`], so all fields moved by a
    /// stochastic driver receive the same increment at the same cell (the
    /// broadcast correlated systems rely on); zero-amplitude and ghost
    /// entries receive nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_driver<G: Grid>(
        &mut self,
        grid: &G,
        amplitude: &Self,
        driver: Driver,
        dt: f64,
        seed: u64,
        salt: u64,
    ) {
        debug_assert!(Arc::ptr_eq(&self.layout, &amplitude.layout));
        for (b, (mine, amp)) in self.blocks.iter_mut().zip(&amplitude.blocks).enumerate() {
            block_apply_driver(
                grid,
                &self.layout,
                mine,
                amp,
                BlockId(b as u32),
                driver,
                dt,
                seed,
                salt,
            );
        }
    }

    /// Apply one full explicit update — `self += dt·drift + Σᵢ dμᵢ ∘
    /// stochastic[i]` — in a **single** scheduler dispatch: each block
    /// work item applies the drift axpy and every stochastic driver's
    /// kernel back-to-back while its slabs are cache-hot, instead of one
    /// whole-state sweep (and barrier) per driver. `stochastic[i]` is the
    /// amplitude buffer of `N::driver(i)`; results are bitwise identical
    /// to applying the drivers one at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_step_with<G: Grid, N: DriverSet, Sch: Scheduler>(
        &mut self,
        scheduler: &Sch,
        grid: &G,
        drift: &Self,
        stochastic: &[Self],
        dt: f64,
        seed: u64,
        salt: u64,
    ) {
        debug_assert!(Arc::ptr_eq(&self.layout, &drift.layout));
        debug_assert_eq!(stochastic.len(), N::LEN);
        let layout = &self.layout;
        scheduler.for_each_block_mut(
            &mut self.blocks,
            || (),
            |id, mine, ()| {
                block_apply_driver(
                    grid,
                    layout,
                    mine,
                    &drift.blocks[id.index()],
                    id,
                    Driver::Time,
                    dt,
                    seed,
                    salt,
                );
                for (i, amp) in stochastic.iter().enumerate() {
                    block_apply_driver(
                        grid,
                        layout,
                        mine,
                        &amp.blocks[id.index()],
                        id,
                        N::driver(i),
                        dt,
                        seed,
                        salt,
                    );
                }
            },
        );
    }

    /// [`State::apply_step_with`] restricted to blocks at refinement
    /// `level` — the Berger–Oliger per-level state update. Blocks at other
    /// levels are left untouched; noise keys still use absolute block ids,
    /// so a fine level's increments never collide with a coarse level's.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_step_level<G: Grid, N: DriverSet, Sch: Scheduler>(
        &mut self,
        scheduler: &Sch,
        grid: &G,
        drift: &Self,
        stochastic: &[Self],
        dt: f64,
        seed: u64,
        salt: u64,
        level: u8,
    ) {
        debug_assert!(Arc::ptr_eq(&self.layout, &drift.layout));
        debug_assert_eq!(stochastic.len(), N::LEN);
        let layout = &self.layout;
        scheduler.for_each_block_mut(
            &mut self.blocks,
            || (),
            |id, mine, ()| {
                if grid.level(id) != level {
                    return;
                }
                block_apply_driver(
                    grid,
                    layout,
                    mine,
                    &drift.blocks[id.index()],
                    id,
                    Driver::Time,
                    dt,
                    seed,
                    salt,
                );
                for (i, amp) in stochastic.iter().enumerate() {
                    block_apply_driver(
                        grid,
                        layout,
                        mine,
                        &amp.blocks[id.index()],
                        id,
                        N::driver(i),
                        dt,
                        seed,
                        salt,
                    );
                }
            },
        );
    }

    /// Scheduler-driven [`State::apply_driver`]; identical results under
    /// any scheduling because keys depend only on (driver, block, cell).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_driver_with<G: Grid, Sch: Scheduler>(
        &mut self,
        scheduler: &Sch,
        grid: &G,
        amplitude: &Self,
        driver: Driver,
        dt: f64,
        seed: u64,
        salt: u64,
    ) {
        debug_assert!(Arc::ptr_eq(&self.layout, &amplitude.layout));
        let layout = &self.layout;
        scheduler.for_each_block_mut(
            &mut self.blocks,
            || (),
            |id, mine, ()| {
                block_apply_driver(
                    grid,
                    layout,
                    mine,
                    &amplitude.blocks[id.index()],
                    id,
                    driver,
                    dt,
                    seed,
                    salt,
                );
            },
        );
    }
}

/// Mutable access to every field of *one* block, as handed to a model's
/// `rhs_block` for its output. Created via [`BlockStorage::bind_mut`].
pub struct BlockStateMut<'a, T: Scalar, S: StorageBackend<T>> {
    layout: &'a StateLayout,
    fields: &'a mut [S],
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, S: StorageBackend<T>> BlockStateMut<'_, T, S> {
    /// Mutable raw slab of one field (ghost-inclusive).
    #[inline(always)]
    pub fn slab_mut(&mut self, handle: FieldHandle<T>) -> &mut [T] {
        self.fields[handle.index()].as_mut_slice()
    }

    /// Mutable view of one field.
    #[inline(always)]
    pub fn view_mut<'s, G: Grid>(
        &'s mut self,
        grid: &'s G,
        block: BlockId,
        handle: FieldHandle<T>,
    ) -> G::ViewMut<'s, T> {
        let ghost = self.layout.ghost(handle.index());
        grid.view_mut(block, ghost, self.fields[handle.index()].as_mut_slice())
    }

    /// Simultaneously borrow one field mutably and another immutably — the
    /// idiom for coupled tendencies, e.g. a `du/dt` that reads the freshly
    /// written `dφ/dt`.
    ///
    /// # Panics
    ///
    /// Panics if `write == read`.
    pub fn view_split<'s, G: Grid>(
        &'s mut self,
        grid: &'s G,
        block: BlockId,
        write: FieldHandle<T>,
        read: FieldHandle<T>,
    ) -> (G::ViewMut<'s, T>, G::View<'s, T>) {
        assert_ne!(
            write.index(),
            read.index(),
            "cannot split a field with itself"
        );
        let (lo, hi) = if write.index() < read.index() {
            let (lo, hi) = self.fields.split_at_mut(read.index());
            (&mut lo[write.index()], &hi[0])
        } else {
            let (lo, hi) = self.fields.split_at_mut(write.index());
            (&mut hi[0], &lo[read.index()])
        };
        let wg = self.layout.ghost(write.index());
        let rg = self.layout.ghost(read.index());
        (
            grid.view_mut(block, wg, lo.as_mut_slice()),
            grid.view(block, rg, hi.as_slice()),
        )
    }
}
