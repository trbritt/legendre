//! `Model::project`: the post-step pointwise-constraint hook.
//!
//! `Simulation::step` calls `project` over the whole state *after* the
//! integrator advance and *before* observers, so a model can enforce
//! constraints the scheme itself does not (e.g. CIR variance positivity).
//! The model here is the minimal stand-in for that use: a constant negative
//! drift that would drive a field below a floor, with `project` clamping it.

// The floor is exactly 0.0, so clamped cells compare exactly equal to it.
#![allow(clippy::float_cmp)]

use legendre::{
    core::{
        observer::Observer,
        scheduler::SerialScheduler,
        scratch::Scratch,
        simulation::Simulation,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::{DenseStorage, StorageBackend, SystemAllocator},
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::ForwardEuler,
    physics::model::{Driver, Model, NoNoise, RhsContext},
};
use std::sync::{Arc, Mutex};

/// `dv/dt = −rate` (constant), with an optional `project` that clamps
/// `v ← max(v, floor)`. With `clamp` off, `v` runs freely negative.
#[derive(Clone)]
struct ClampedDecay {
    rate: f64,
    floor: f64,
    clamp: bool,
    v: Option<FieldHandle<f64>>,
}

impl<D: Sync> Model<CartesianGrid<1>, D> for ClampedDecay {
    type Scalar = f64;
    type Drivers = NoNoise;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.v = Some(builder.register("v", 0));
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        _driver: Driver,
        ctx: &RhsContext<'_, CartesianGrid<1>, D>,
        _state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        let mut dv = out.view_mut(ctx.grid, ctx.block, self.v.unwrap());
        for_each_interior(ctx.grid.block_cells(), |idx| dv.set(idx, -self.rate));
    }

    fn project<S: StorageBackend<f64>>(&self, grid: &CartesianGrid<1>, state: &mut State<f64, S>) {
        if !self.clamp {
            return;
        }
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            let mut v = state.view_mut(grid, block, self.v.unwrap());
            for_each_interior(grid.block_cells(), |idx| v.set(idx, v.get(idx).max(self.floor)));
        }
    }
}

/// Records the minimum field value it is ever shown — used to prove observers
/// see the *projected* state (the projection runs before them).
struct MinObserver {
    v: FieldHandle<f64>,
    min: Arc<Mutex<f64>>,
}

impl<S: StorageBackend<f64>> Observer<CartesianGrid<1>, f64, S> for MinObserver {
    fn observe(&mut self, _step: u64, _t: f64, _epoch: u64, grid: &CartesianGrid<1>, state: &State<f64, S>) {
        let mut m = self.min.lock().unwrap();
        for b in 0..grid.num_blocks() {
            let v = state.view(grid, BlockId(b as u32), self.v);
            for_each_interior(grid.block_cells(), |idx| *m = m.min(v.get(idx)));
        }
    }
}

/// Run the decay model for `steps` from `v ≡ 0.05`; return the final state
/// values and the minimum value any observer was shown.
fn run(clamp: bool, steps: usize) -> (Vec<f64>, f64) {
    let grid = CartesianGrid::new([8], [8], [0.0], [1.0]).unwrap();
    let model = ClampedDecay {
        rate: 1.0,
        floor: 0.0,
        clamp,
        v: None,
    };
    let mut sim = Simulation::new(grid, (), model, ForwardEuler, SerialScheduler, SystemAllocator);

    let v = sim.model().v.unwrap();
    let (g, state) = sim.state_mut();
    for b in 0..g.num_blocks() {
        let block = BlockId(b as u32);
        let mut view = state.view_mut(g, block, v);
        for_each_interior(g.block_cells(), |idx| view.set(idx, 0.05));
    }

    let observed_min = Arc::new(Mutex::new(f64::INFINITY));
    let obs: MinObserver = MinObserver {
        v,
        min: Arc::clone(&observed_min),
    };
    let obs: Box<dyn Observer<CartesianGrid<1>, f64, DenseStorage<f64>>> = Box::new(obs);
    sim.attach_observer(obs);

    let dt = 0.1;
    for _ in 0..steps {
        sim.step(dt);
    }

    let mut final_vals = Vec::new();
    for b in 0..sim.grid().num_blocks() {
        let view = sim.state().view(sim.grid(), BlockId(b as u32), v);
        for_each_interior(sim.grid().block_cells(), |idx| final_vals.push(view.get(idx)));
    }
    let m = *observed_min.lock().unwrap();
    (final_vals, m)
}

#[test]
fn project_clamps_the_state_and_observers_see_it() {
    // dv/dt = −1 from v₀ = 0.05 would go negative within one dt = 0.1 step;
    // the clamp holds it at the floor, and — because projection runs before
    // observers — no observer is ever shown a sub-floor value.
    let (final_vals, observed_min) = run(true, 3);
    for v in &final_vals {
        assert_eq!(*v, 0.0, "clamped field must rest exactly at the floor");
    }
    assert!(
        observed_min >= 0.0,
        "observers must see the projected (clamped) state, not the raw one; saw {observed_min}"
    );
}

#[test]
fn without_project_the_constraint_is_violated() {
    // Same dynamics, projection disabled: the field is free to run negative,
    // which is exactly what makes the clamp above a real correction and not a
    // no-op the dynamics would have satisfied anyway.
    let (final_vals, observed_min) = run(false, 3);
    assert!(
        final_vals.iter().all(|&v| v < 0.0),
        "unprojected decay must cross below the floor"
    );
    assert!(observed_min < 0.0, "observers should see the unclamped excursion");
}
