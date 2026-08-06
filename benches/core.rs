//! CodSpeed-tracked benchmarks of the framework's key infrastructure:
//! whole integrator steps on Model C (the realistic composite workload)
//! and the primitive passes they are built from (driver kernels, axpy,
//! ghost fill).
//!
//! Everything runs on `SerialScheduler`: `CodSpeed` measures instruction
//! counts, and single-threaded execution keeps them deterministic.
//!
//! # Flamegraph profiling
//!
//! With the `dev-profiling` feature, criterion's profiling hook samples
//! the benchmark with `pprof` and writes a flamegraph, giving optimization
//! work evidence instead of guesses:
//!
//! ```text
//! cargo bench --features dev-profiling --bench core -- \
//!     --profile-time 10 "model_c/step/em_deterministic"
//! # -> target/criterion/model_c/step/em_deterministic/profile/flamegraph.svg
//! ```
//!
//! Profiling only engages under `--profile-time`; plain `cargo bench` and
//! the `CodSpeed` CI run are unaffected.

// `criterion_group!` holds its `Criterion` to the end of `main` (the
// "tighten this drop" suggestion cannot apply inside the macro expansion)
// and generates an undocumentable `fn benches`; neither lint is
// meaningful for a bench harness.
#![allow(clippy::significant_drop_tightening)]
#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use legendre::{
    core::{
        driver::{Driver, DriverSet},
        monte_carlo::path_grid,
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
        simulation::Simulation,
        state::{FieldHandle, State, StateBuilder},
        storage::{DenseStorage, SystemAllocator},
    },
    discretization::finite_volume::FiniteVolume,
    geometry::{
        amr::{AmrGrid, BergerOliger, ClusterParams, GradientTagger, RegridPolicy, cluster},
        cartesian::{CartesianGrid, fill_ghosts_mirror, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::{EulerMaruyama, ForwardEuler, Integrator, RungeKutta4, Subcycling},
    physics::{
        market_making::{DepthTables, HjbMarketMaker, MarketMakerParams, MarketMakingEnsemble},
        model::{NoNoise, Wiener},
        phasefield::ModelC,
    },
};
use std::sync::Arc;

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

/// Criterion `Profiler` that samples the benchmark under `--profile-time`
/// and writes `profile/flamegraph.svg` next to the benchmark's report.
#[cfg(feature = "dev-profiling")]
mod profiling {
    use criterion::profiler::Profiler;
    use pprof::ProfilerGuard;
    use std::{fs::File, path::Path};

    pub struct Flamegraph<'a> {
        frequency: i32,
        active: Option<ProfilerGuard<'a>>,
    }

    impl Flamegraph<'_> {
        pub const fn new(frequency: i32) -> Self {
            Self {
                frequency,
                active: None,
            }
        }
    }

    impl Profiler for Flamegraph<'_> {
        fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
            self.active = Some(ProfilerGuard::new(self.frequency).expect("start pprof sampler"));
        }

        fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
            let Some(profiler) = self.active.take() else {
                return;
            };
            std::fs::create_dir_all(benchmark_dir).expect("create profile dir");
            let file =
                File::create(benchmark_dir.join("flamegraph.svg")).expect("create flamegraph.svg");
            profiler
                .report()
                .build()
                .expect("build pprof report")
                .flamegraph(file)
                .expect("write flamegraph");
        }
    }
}

fn config() -> Criterion {
    // 997 Hz: a prime sampling rate avoids lock-step with periodic work.
    #[cfg(feature = "dev-profiling")]
    {
        Criterion::default().with_profiler(profiling::Flamegraph::new(997))
    }
    #[cfg(not(feature = "dev-profiling"))]
    {
        Criterion::default()
    }
}

/// The AMR machinery: the Berger–Rigoutsos kernel on an interface-shaped
/// flag set, and a whole adaptive Model C step (amortized regrids,
/// migration, intergrid transfers included).
fn amr(c: &mut Criterion) {
    c.bench_function("amr/cluster/interface_ring", |b| {
        // A ring of flagged cells, the shape interface tracking produces.
        let flags: Vec<[i64; 2]> = (0..128 * 128)
            .map(|i| [i % 128, i / 128])
            .filter(|&[x, y]| {
                let (dx, dy) = ((x - 64) as f64, (y - 64) as f64);
                let r = dx.hypot(dy);
                (40.0..44.0).contains(&r)
            })
            .collect();
        let params = ClusterParams {
            efficiency: 0.8,
            min_width: 4,
        };
        b.iter(|| {
            let mut work = flags.clone();
            cluster(&mut work, &params)
        });
    });

    c.bench_function("amr/step/model_c_adaptive", |b| {
        let base = CartesianGrid::new([N; 2], [N / 2; 2], [0.0; 2], [H; 2]).unwrap();
        let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
        let mut sim = Simulation::adaptive(
            grid,
            FiniteVolume::default(),
            ModelC::<NoNoise>::classic(),
            EulerMaruyama { seed: 7 },
            SerialScheduler,
            SystemAllocator,
            BergerOliger::new(
                GradientTagger {
                    field: "phi",
                    threshold: 0.15,
                },
                RegridPolicy {
                    every: 4,
                    buffer: 2,
                    cluster: ClusterParams {
                        efficiency: 0.8,
                        min_width: 4,
                    },
                },
            ),
        );
        let dt = sim.stable_dt().unwrap();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize(grid.base(), state, [H, H], 10.0 * H, 0.7);
        }
        b.iter(|| sim.step(dt));
    });

    c.bench_function("amr/step/model_c_subcycled", |b| {
        let base = CartesianGrid::new([N; 2], [N / 2; 2], [0.0; 2], [H; 2]).unwrap();
        let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
        let mut sim = Simulation::adaptive(
            grid,
            FiniteVolume::default(),
            ModelC::<NoNoise>::classic(),
            Subcycling { seed: 7 },
            SerialScheduler,
            SystemAllocator,
            BergerOliger::new(
                GradientTagger {
                    field: "phi",
                    threshold: 0.15,
                },
                RegridPolicy {
                    every: 4,
                    buffer: 2,
                    cluster: ClusterParams {
                        efficiency: 0.8,
                        min_width: 4,
                    },
                },
            ),
        );
        let dt = sim.stable_dt().unwrap();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize(grid.base(), state, [H, H], 10.0 * H, 0.7);
        }
        b.iter(|| sim.step(dt));
    });
}

// --- Market making: the HJB policy solve, the controlled path ensemble, and
// the live quote lookup — the three latency surfaces of the strategy. ---

const NU_CELLS: usize = 100;
const NU_BLOCK: usize = 25;
const HORIZON: f64 = 3.0;

type HjbSim = Simulation<CartesianGrid<1>, (), HjbMarketMaker, ForwardEuler, SerialScheduler, SystemAllocator>;
type EnsembleSim =
    Simulation<CartesianGrid<1>, (), MarketMakingEnsemble, EulerMaruyama, SerialScheduler, SystemAllocator>;

fn nu_grid() -> CartesianGrid<1> {
    let (lo, hi) = (0.001, 2.0);
    CartesianGrid::new([NU_CELLS], [NU_BLOCK], [lo], [(hi - lo) / NU_CELLS as f64]).unwrap()
}

fn hjb_sim(params: MarketMakerParams) -> (HjbSim, f64) {
    let sim = Simulation::new(
        nu_grid(),
        (),
        HjbMarketMaker::new(params),
        ForwardEuler,
        SerialScheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    (sim, dt)
}

/// The value surface as `surface[q_index][ν_cell]`.
fn capture(
    grid: &CartesianGrid<1>,
    state: &State<f64, DenseStorage<f64>>,
    handles: &[FieldHandle<f64>],
) -> Vec<Vec<f64>> {
    handles
        .iter()
        .map(|&h| {
            let mut col = Vec::new();
            for b in 0..grid.num_blocks() {
                let v = state.view(grid, BlockId(b as u32), h);
                for_each_interior(grid.block_cells(), |idx| col.push(v.get(idx)));
            }
            col
        })
        .collect()
}

/// Solve the HJB to the horizon (setup only) and bake the optimal-quote tables.
fn depth_tables(params: MarketMakerParams) -> DepthTables {
    let (mut sim, dt) = hjb_sim(params);
    let steps = (HORIZON / dt).ceil() as usize;
    let handles = sim.model().handles().to_vec();
    let mut taus = vec![0.0];
    let mut surfaces = vec![capture(sim.grid(), sim.state(), &handles)];
    for i in 0..steps {
        sim.step(dt);
        taus.push((i + 1) as f64 * dt);
        surfaces.push(capture(sim.grid(), sim.state(), &handles));
    }
    DepthTables::build(&HjbMarketMaker::new(params), &nu_grid(), &taus, &surfaces)
}

fn ensemble_sim(params: MarketMakerParams, tables: Arc<DepthTables>) -> EnsembleSim {
    let grid = path_grid(20_000, 4000).unwrap();
    let mut sim = Simulation::new(
        grid,
        (),
        MarketMakingEnsemble::new(params, tables, HORIZON),
        EulerMaruyama { seed: 7 },
        SerialScheduler,
        SystemAllocator,
    );
    let model = sim.model().clone();
    let (g, state) = sim.state_mut();
    model.initialize(g, state, params.theta, 100.0);
    sim
}

/// Full backward-time HJB re-solve to the horizon (state reset to the zero
/// terminal condition each iteration) — the policy-refresh latency itself.
fn bench_hjb_solve<Sch: Scheduler>(
    c: &mut Criterion,
    name: &str,
    scheduler: Sch,
    cells: usize,
    block: usize,
) {
    let (lo, hi) = (0.001, 2.0);
    let grid = CartesianGrid::new([cells], [block], [lo], [(hi - lo) / cells as f64]).unwrap();
    let mut sim = Simulation::new(
        grid,
        (),
        HjbMarketMaker::new(MarketMakerParams::default()),
        ForwardEuler,
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    let steps = (HORIZON / dt).ceil() as usize;
    c.bench_function(name, |b| {
        b.iter(|| {
            sim.state_mut().1.fill_zero();
            for _ in 0..steps {
                sim.step(dt);
            }
        });
    });
}

fn market_making(c: &mut Criterion) {
    let params = MarketMakerParams::default();

    // Policy-refresh cost: one backward-time HJB step (a full re-solve is
    // `steps` of these — the freshness bound when recalibrating intraday).
    c.bench_function("mm/hjb/step", |b| {
        let (mut sim, dt) = hjb_sim(params);
        b.iter(|| sim.step(dt));
    });

    // Full re-solve latency: the cheap, bit-exact levers (parallelism, grid
    // resolution) measured against the serial baseline.
    bench_hjb_solve(c, "mm/hjb/solve/serial_n100", SerialScheduler, 100, 25);
    bench_hjb_solve(c, "mm/hjb/solve/rayon_n100", RayonScheduler, 100, 10);
    bench_hjb_solve(c, "mm/hjb/solve/serial_n50", SerialScheduler, 50, 25);
    bench_hjb_solve(c, "mm/hjb/solve/serial_n200", SerialScheduler, 200, 25);

    // Backtest throughput: one Euler–Maruyama step of the controlled ensemble.
    c.bench_function("mm/ensemble/step", |b| {
        let (_, dt) = hjb_sim(params);
        let mut sim = ensemble_sim(params, Arc::new(depth_tables(params)));
        b.iter(|| sim.step(dt));
    });

    // Live quote: the single lookup that turns (τ, ν, q) into (δᵇ, δˢ).
    c.bench_function("mm/depth_tables/lookup", |b| {
        let tables = depth_tables(params);
        b.iter(|| black_box(tables.lookup(black_box(1.5), black_box(0.9), black_box(2))));
    });
}

criterion_group! {
    name = benches;
    config = config();
    targets = integrator_steps, primitives, amr, market_making
}
criterion_main!(benches);
