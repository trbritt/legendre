//! The [`Simulation`]: sole owner of grid, discretization, model,
//! integrator, state, scheduler, and observers (RFC 0001, ownership graph).
//!
//! Construction is the one place allocation happens: field registration →
//! state build → stage buffers. `step()` then delegates: integrator drives
//! model through scheduler; observers are notified after the state update.
//!
//! Scratch memory is a worker-pinned [`ScratchPool`] sized to the
//! scheduler's concurrency, so nothing allocates after construction.

use crate::{
    core::{
        observer::Observer,
        scheduler::Scheduler,
        scratch::ScratchPool,
        state::{State, StateBuilder},
        storage::{Allocator, Scalar, StorageBackend},
    },
    geometry::grid::Grid,
    integrators::{Integrator, StageKind},
    physics::model::{DriverSet, Model},
};

/// An adaptivity policy: decides whether and how to rebuild the grid.
///
/// All refinement knowledge — tagging, clustering, nesting, state
/// migration — lives in implementations of this trait, none in
/// [`Simulation`]: `regrid` returns the new grid **and the already
/// migrated state**, so the simulation only swaps values and rebuilds its
/// stage buffers. `state` is borrowed mutably so policies may make it
/// consistent (restriction, ghost fills) before tagging; the returned
/// state replaces it either way.
pub trait Adapt<G, T: Scalar, S: StorageBackend<T>, A>: Send + Sync {
    /// Called at the top of every step with the number of completed
    /// steps. `None` means keep stepping on the current grid.
    fn regrid(
        &mut self,
        grid: &G,
        state: &mut State<T, S>,
        alloc: &A,
        step: u64,
    ) -> Option<(G, State<T, S>)>;
}

/// The default policy: never adapts. `regrid` monomorphizes to `None`, so
/// uniform-grid simulations pay nothing and keep every existing signature
/// through the default type parameter.
#[derive(Debug, Clone, Copy, Default)]
pub struct Static;

impl<G, T: Scalar, S: StorageBackend<T>, A> Adapt<G, T, S, A> for Static {
    #[inline(always)]
    fn regrid(
        &mut self,
        _grid: &G,
        _state: &mut State<T, S>,
        _alloc: &A,
        _step: u64,
    ) -> Option<(G, State<T, S>)> {
        None
    }
}

/// The boxed observer list of a simulation (see [`Observer`]).
type Observers<G, T, S> = Vec<Box<dyn Observer<G, T, S>>>;

/// Sole owner of every simulation component; see the module docs.
pub struct Simulation<G, D, M, I, Sch, A, R = Static>
where
    G: Grid,
    M: Model<G, D>,
    I: Integrator<G, D, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
    R: Adapt<G, M::Scalar, A::Storage, A>,
{
    adapt: R,
    grid: G,
    disc: D,
    model: M,
    integrator: I,
    scheduler: Sch,
    alloc: A,
    state: State<M::Scalar, A::Storage>,
    stages: Vec<State<M::Scalar, A::Storage>>,
    pool: ScratchPool<M::Scalar, A::Storage>,
    observers: Observers<G, M::Scalar, A::Storage>,
    t: f64,
    step_index: u64,
    epoch: u64,
}

impl<G, D, M, I, Sch, A> Simulation<G, D, M, I, Sch, A>
where
    G: Grid,
    M: Model<G, D>,
    I: Integrator<G, D, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
{
    /// A non-adaptive simulation (the [`Static`] policy): register the
    /// model's fields, allocate state and stage buffers, and assemble.
    ///
    /// # Panics
    ///
    /// See [`Simulation::adaptive`].
    pub fn new(grid: G, disc: D, model: M, integrator: I, scheduler: Sch, alloc: A) -> Self {
        Self::adaptive(grid, disc, model, integrator, scheduler, alloc, Static)
    }
}

impl<G, D, M, I, Sch, A, R> Simulation<G, D, M, I, Sch, A, R>
where
    G: Grid,
    M: Model<G, D>,
    I: Integrator<G, D, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
    R: Adapt<G, M::Scalar, A::Storage, A>,
{
    /// A simulation with an adaptivity policy consulted at the top of
    /// every step; on a regrid the policy's migrated state replaces the
    /// current one and stage buffers and scratch are reallocated (the
    /// only allocating path after construction, amortized over the regrid
    /// interval).
    ///
    /// Observers are not yet regrid-aware: an observer that captured grid
    /// geometry at attach time (e.g. the Parquet writer) will not follow
    /// grid changes.
    ///
    /// # Panics
    ///
    /// Panics if the model declares a stochastic driver that no field is
    /// registered as driven by — that term would silently apply to
    /// nothing (see [`StateBuilder::register_driven`]).
    pub fn adaptive(
        grid: G,
        disc: D,
        mut model: M,
        integrator: I,
        scheduler: Sch,
        alloc: A,
        adapt: R,
    ) -> Self {
        let mut builder = StateBuilder::new();
        model.register_fields(&mut builder);
        let state = builder.build(&grid, &alloc);
        for i in 0..<M::Drivers as DriverSet>::LEN {
            let driver = <M::Drivers as DriverSet>::driver(i);
            assert!(
                state
                    .layout()
                    .specs()
                    .iter()
                    .any(|spec| spec.is_driven_by(driver)),
                "model declares stochastic driver {driver:?} but no field is registered \
                 as driven by it; use StateBuilder::register_driven"
            );
        }
        let stages = Self::build_stages(&integrator, &state, &grid, &alloc);
        let pool = ScratchPool::allocate(
            model.scratch_spec(&grid),
            &alloc,
            scheduler.max_concurrency(),
        );
        Self {
            adapt,
            grid,
            disc,
            model,
            integrator,
            scheduler,
            alloc,
            state,
            stages,
            pool,
            observers: Vec::new(),
            t: 0.0,
            step_index: 0,
            epoch: 0,
        }
    }

    /// Allocate the integrator's stage buffers for `state` on `grid` —
    /// used at construction and again after every regrid.
    fn build_stages(
        integrator: &I,
        state: &State<M::Scalar, A::Storage>,
        grid: &G,
        alloc: &A,
    ) -> Vec<State<M::Scalar, A::Storage>> {
        integrator
            .stage_layout()
            .stages
            .iter()
            .map(|kind| match *kind {
                StageKind::Tendency(driver) => state.like_for(grid, alloc, driver),
                StageKind::State => state.like(grid, alloc),
            })
            .collect()
    }

    /// Register an observer to be notified after every completed step
    /// with the current grid and epoch.
    pub fn attach_observer(&mut self, observer: Box<dyn Observer<G, M::Scalar, A::Storage>>) {
        self.observers.push(observer);
    }

    /// Grid generation: 0 at construction, +1 per adaptive regrid.
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The grid this simulation runs on.
    pub const fn grid(&self) -> &G {
        &self.grid
    }

    /// The model being integrated.
    pub const fn model(&self) -> &M {
        &self.model
    }

    /// Current simulation time.
    pub const fn time(&self) -> f64 {
        self.t
    }

    /// Number of completed steps.
    pub const fn step_index(&self) -> u64 {
        self.step_index
    }

    /// Read access to state (observers, checks).
    pub const fn state(&self) -> &State<M::Scalar, A::Storage> {
        &self.state
    }

    /// Mutable state access for setting initial conditions before stepping.
    pub const fn state_mut(&mut self) -> (&G, &mut State<M::Scalar, A::Storage>) {
        (&self.grid, &mut self.state)
    }

    /// Congruent, zeroed state buffers for observers' snapshot rings.
    pub fn snapshot_buffers(&self, n: usize) -> Vec<State<M::Scalar, A::Storage>> {
        (0..n)
            .map(|_| self.state.like(&self.grid, &self.alloc))
            .collect()
    }

    /// Advance one step of size `dt`, then notify observers. Consults
    /// the adaptivity policy first; a regrid swaps grid and state and
    /// rebuilds stage buffers and scratch before stepping.
    pub fn step(&mut self, dt: f64) {
        if let Some((grid, state)) =
            self.adapt
                .regrid(&self.grid, &mut self.state, &self.alloc, self.step_index)
        {
            self.grid = grid;
            self.state = state;
            self.epoch += 1;
            self.stages =
                Self::build_stages(&self.integrator, &self.state, &self.grid, &self.alloc);
            self.pool = ScratchPool::allocate(
                self.model.scratch_spec(&self.grid),
                &self.alloc,
                self.scheduler.max_concurrency(),
            );
        }
        self.integrator.step(
            &self.model,
            &self.grid,
            &self.disc,
            &self.scheduler,
            &self.pool,
            &mut self.state,
            &mut self.stages,
            self.t,
            dt,
        );
        self.t += dt;
        self.step_index += 1;
        for obs in &mut self.observers {
            obs.observe(self.step_index, self.t, self.epoch, &self.grid, &self.state);
        }
    }

    /// Advisory stable dt from the model, if it declares one.
    pub fn stable_dt(&self) -> Option<f64> {
        self.model.stable_dt(&self.grid)
    }
}
