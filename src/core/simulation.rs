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
        storage::Allocator,
    },
    geometry::grid::Grid,
    integrators::{Integrator, StageKind},
    physics::model::{DriverSet, Model},
};

/// Sole owner of every simulation component; see the module docs.
pub struct Simulation<G, D, M, I, Sch, A>
where
    G: Grid,
    M: Model<G, D>,
    I: Integrator<G, D, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
{
    grid: G,
    disc: D,
    model: M,
    integrator: I,
    scheduler: Sch,
    alloc: A,
    state: State<M::Scalar, A::Storage>,
    stages: Vec<State<M::Scalar, A::Storage>>,
    pool: ScratchPool<M::Scalar, A::Storage>,
    observers: Vec<Box<dyn Observer<M::Scalar, A::Storage>>>,
    t: f64,
    step_index: u64,
}

impl<G, D, M, I, Sch, A> Simulation<G, D, M, I, Sch, A>
where
    G: Grid,
    M: Model<G, D>,
    I: Integrator<G, D, M::Drivers>,
    Sch: Scheduler,
    A: Allocator<M::Scalar>,
{
    /// Register the model's fields, allocate state and stage buffers, and
    /// assemble. The only allocating path in a simulation's lifetime.
    ///
    /// # Panics
    ///
    /// Panics if the model declares a stochastic driver that no field is
    /// registered as driven by — that term would silently apply to
    /// nothing (see [`StateBuilder::register_driven`]).
    pub fn new(grid: G, disc: D, mut model: M, integrator: I, scheduler: Sch, alloc: A) -> Self {
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
        let stages = integrator
            .stage_layout()
            .stages
            .iter()
            .map(|kind| match *kind {
                StageKind::Tendency(driver) => state.like_for(&grid, &alloc, driver),
                StageKind::State => state.like(&grid, &alloc),
            })
            .collect();
        let pool = ScratchPool::allocate(
            model.scratch_spec(&grid),
            &alloc,
            scheduler.max_concurrency(),
        );
        Self {
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
        }
    }

    /// Register an observer to be notified after every completed step.
    pub fn attach_observer(&mut self, observer: Box<dyn Observer<M::Scalar, A::Storage>>) {
        self.observers.push(observer);
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

    /// Advance one step of size `dt`, then notify observers.
    pub fn step(&mut self, dt: f64) {
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
            obs.observe(self.step_index, self.t, &self.state);
        }
    }

    /// Advisory stable dt from the model, if it declares one.
    pub fn stable_dt(&self) -> Option<f64> {
        self.model.stable_dt(&self.grid)
    }
}
