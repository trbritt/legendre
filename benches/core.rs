//! CodSpeed-tracked benchmarks of the framework's key infrastructure:
//! whole integrator steps on Model C (the realistic composite workload)
//! and the primitive passes they are built from (driver kernels, axpy,
//! ghost fill).
//!
//! Everything runs on `SerialScheduler`: `CodSpeed` measures instruction
//! counts, and single-threaded execution keeps them deterministic.

// `criterion_group!` holds its `Criterion` to the end of `main`; the
// "tighten this drop" suggestion cannot apply inside the macro expansion.
#![allow(clippy::significant_drop_tightening)]

use criterion::{Criterion, criterion_group, criterion_main};
use legendre::{
    core::{
        driver::{Driver, DriverSet},
        scheduler::SerialScheduler,
        simulation::Simulation,
        state::{State, StateBuilder},
        storage::{DenseStorage, SystemAllocator},
    },
    discretization::finite_volume::FiniteVolume,
    geometry::{
        cartesian::{CartesianGrid, fill_ghosts_mirror},
        grid::{BlockId, Grid},
    },
    integrators::{EulerMaruyama, Integrator, RungeKutta4},
    physics::{
        model::{NoNoise, Wiener},
        phasefield::ModelC,
    },
};

const N: usize = 128;
const H: f64 = 0.4;

type ModelCSim<Nz, I> =
    Simulation<CartesianGrid<2>, FiniteVolume, ModelC<Nz>, I, SerialScheduler, SystemAllocator>;

fn model_c_sim<Nz, I>(integrator: I, noise_amplitude: f64) -> (ModelCSim<Nz, I>, f64)
where
    Nz: DriverSet,
    I: Integrator<CartesianGrid<2>, FiniteVolume, Nz>,
{
    let grid = CartesianGrid::new([N; 2], [N / 2; 2], [0.0; 2], [H; 2]).unwrap();
    let mut model = ModelC::<Nz>::classic();
    model.noise_amplitude = noise_amplitude;
    let mut sim = Simulation::new(
        grid,
        FiniteVolume::default(),
        model,
        integrator,
        SerialScheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    {
        let model = sim.model().clone();
        let (grid, state) = sim.state_mut();
        model.initialize(grid, state, [H, H], 10.0 * H, 0.7);
    }
    (sim, dt)
}

fn integrator_steps(c: &mut Criterion) {
    c.bench_function("model_c/step/em_deterministic", |b| {
        let (mut sim, dt) = model_c_sim::<NoNoise, _>(EulerMaruyama { seed: 7 }, 0.0);
        b.iter(|| sim.step(dt));
    });
    c.bench_function("model_c/step/em_wiener", |b| {
        let (mut sim, dt) = model_c_sim::<Wiener<1>, _>(EulerMaruyama { seed: 7 }, 0.05);
        b.iter(|| sim.step(dt));
    });
    c.bench_function("model_c/step/rk4", |b| {
        let (mut sim, dt) = model_c_sim::<NoNoise, _>(RungeKutta4, 0.0);
        b.iter(|| sim.step(dt));
    });
}

/// A two-field state (one noisy, one drift-only) with unit amplitudes,
/// exercising the slab primitives the integrators are built from.
fn primitives(c: &mut Criterion) {
    let grid = CartesianGrid::new([256; 2], [128; 2], [0.0; 2], [1.0; 2]).unwrap();
    let mut builder = StateBuilder::<f64>::new();
    let noisy = builder.register_driven("noisy", 1, &[Driver::Time, Driver::Wiener(0)]);
    let _plain = builder.register("plain", 1);
    let mut state: State<f64, DenseStorage<f64>> = builder.build(&grid, &SystemAllocator);

    let mut drift = state.like_for(&grid, &SystemAllocator, Driver::Time);
    let mut amp = state.like_for(&grid, &SystemAllocator, Driver::Wiener(0));
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        drift.slab_mut(block, noisy).fill(1.0);
        amp.slab_mut(block, noisy).fill(1.0);
    }

    c.bench_function("state/apply_driver/time", |b| {
        b.iter(|| state.apply_driver(&grid, &drift, Driver::Time, 1e-3, 7, 0));
    });
    c.bench_function("state/apply_driver/wiener", |b| {
        b.iter(|| state.apply_driver(&grid, &amp, Driver::Wiener(0), 1e-3, 7, 0));
    });
    c.bench_function("state/axpy", |b| {
        b.iter(|| state.axpy(1e-3, &drift));
    });
    c.bench_function("ghosts/fill_mirror", |b| {
        b.iter(|| fill_ghosts_mirror(&grid, &mut state, noisy));
    });
}

criterion_group!(benches, integrator_steps, primitives);
criterion_main!(benches);
