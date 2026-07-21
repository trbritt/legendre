//! Dendritic solidification / nucleation with the shipped phase-field model.
//!
//! ```text
//! cargo run --release --example model_c -- --help
//! ```
//!
//! Defaults reproduce a classic single-dendrite setup (630² cells as 5×5
//! blocks, h = 0.4, corner seed, u₀ = 0.7). `--seeds N` with N > 1 scatters
//! N random seeds across the domain for macroscale nucleation studies;
//! combine with `--noise` for stochastic sidebranching and `--orient` for
//! per-grain crystallographic orientations. Output is Parquet snapshots
//! written by the async observer pipeline; a progress bar reports rate,
//! ETA, and per-field statistics as the run proceeds.
//!
//! Render the results to a movie with the bundled Python script:
//!
//! ```text
//! python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt
//! .venv/bin/python scripts/render_model_c.py data/model_c --out dendrite.mp4
//! ```

use clap::{Parser, ValueEnum};
use legendre::{
    core::{
        observer::AsyncObserver,
        scheduler::RayonScheduler,
        simulation::Simulation,
        storage::{DenseStorage, SystemAllocator},
    },
    discretization::finite_volume::FiniteVolume,
    geometry::{
        amr::{AmrGrid, BergerOliger, ClusterParams, GradientTagger, RegridPolicy},
        cartesian::CartesianGrid,
        grid::Grid,
    },
    integrators::{EulerMaruyama, Integrator, RungeKutta4, Subcycling},
    io::{
        parquet::ParquetObserver,
        progress::{FieldStat, FieldStatsSink, ProgressObserver, progress_bar},
    },
    physics::{
        model::{DriverSet, NoNoise, Wiener},
        phasefield::{Grain, ModelC},
    },
    util::rng::{mix_key, unit_open},
};
use std::error::Error;

/// Cell spacing (isotropic).
const H: f64 = 0.4;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Scheme {
    /// Euler–Maruyama: first-order drift, √dt-correct noise.
    Em,
    /// Classic RK4 drift; deterministic only (pair --noise with em).
    Rk4,
}

/// Dendritic solidification: coupled phase field φ and thermal field u with
/// 4-fold anisotropic surface energy, on a block-decomposed 2D grid.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Interior cells per side (must be divisible by --block)
    #[arg(long, default_value_t = 630)]
    cells: usize,

    /// Block size per side
    #[arg(long, default_value_t = 126)]
    block: usize,

    /// Simulated end time
    #[arg(long, default_value_t = 3000.0)]
    time: f64,

    /// Snapshot cadence in steps
    #[arg(long, default_value_t = 1000)]
    every: u64,

    /// Number of nucleation seeds (1 = classic corner seed)
    #[arg(long, default_value_t = 1)]
    seeds: usize,

    /// Give each grain a random crystallographic orientation in [0, π/2)
    #[arg(long)]
    orient: bool,

    /// Additive noise amplitude on φ (0 disables the stochastic term)
    #[arg(long, default_value_t = 0.0)]
    noise: f64,

    /// Seed for the deterministic noise generator and seed placement
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Snapshot ring depth (bounds observer memory and lag)
    #[arg(long, default_value_t = 4)]
    ring: usize,

    /// Output directory for Parquet snapshots
    #[arg(long, default_value = "data/model_c")]
    out: String,

    /// Time integrator
    #[arg(long, value_enum, default_value_t = Scheme::Em)]
    integrator: Scheme,

    /// Adaptively refine a 2x level over the base grid (Berger-Oliger),
    /// tracking the interface via |grad phi|. Snapshots carry per-cell
    /// level/patch columns; render with `--patches` to outline them.
    #[arg(long)]
    amr: bool,

    /// With --amr, subcycle in time (Berger-Oliger): the coarse level
    /// takes big steps, the fine level r²=4 substeps, so the finest dt is
    /// not imposed on the whole domain. Euler-Maruyama only (not rk4).
    #[arg(long)]
    subcycle: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.amr {
        if args.orient {
            return Err("--amr with --orient is not supported yet (static-field \
                        migration across regrids blurs theta0)"
                .into());
        }
        if args.subcycle {
            // Subcycling is single-stage (Euler-Maruyama family) only.
            if matches!(args.integrator, Scheme::Rk4) {
                return Err("--subcycle does not support rk4 (multi-stage)".into());
            }
            return if args.noise == 0.0 {
                run_amr::<NoNoise, _>(Subcycling { seed: args.seed }, &args)
            } else {
                run_amr::<Wiener<1>, _>(Subcycling { seed: args.seed }, &args)
            };
        }
        return match args.integrator {
            Scheme::Rk4 => {
                if args.noise != 0.0 {
                    return Err(
                        "--integrator rk4 is deterministic-only; use em with --noise".into(),
                    );
                }
                run_amr::<NoNoise, _>(RungeKutta4, &args)
            }
            Scheme::Em if args.noise == 0.0 => {
                run_amr::<NoNoise, _>(EulerMaruyama { seed: args.seed }, &args)
            }
            Scheme::Em => run_amr::<Wiener<1>, _>(EulerMaruyama { seed: args.seed }, &args),
        };
    }
    match args.integrator {
        // The driver contract makes RK4 deterministic-only at the type
        // level; a stochastic run must pair --noise with Euler–Maruyama.
        Scheme::Rk4 => {
            if args.noise != 0.0 {
                return Err("--integrator rk4 is deterministic-only; use em with --noise".into());
            }
            run::<NoNoise, _>(RungeKutta4, &args)
        }
        // Zero amplitude means the model *has no Wiener driver*: select the
        // NoNoise instantiation so the run pays nothing for the absent term
        // (exactly what the pre-driver `has_noise()` runtime gate did).
        Scheme::Em if args.noise == 0.0 => {
            run::<NoNoise, _>(EulerMaruyama { seed: args.seed }, &args)
        }
        Scheme::Em => run::<Wiener<1>, _>(EulerMaruyama { seed: args.seed }, &args),
    }
}

fn seed_positions(args: &Args) -> Vec<Grain> {
    let radius = 10.0 * H;
    let orientation = |i: u64| {
        if args.orient {
            // 4-fold symmetry: [0, π/2) is the full fundamental domain.
            unit_open(mix_key(args.seed, &[i, 2])) * std::f64::consts::FRAC_PI_2
        } else {
            0.0
        }
    };
    if args.seeds <= 1 {
        return vec![Grain {
            center: [H, H],
            radius,
            orientation: orientation(0),
        }];
    }
    let extent = args.cells as f64 * H;
    let margin = 2.0 * radius;
    let span = 2.0f64.mul_add(-margin, extent);
    (0..args.seeds as u64)
        .map(|i| {
            let x = margin + unit_open(mix_key(args.seed, &[i, 0])) * span;
            let y = margin + unit_open(mix_key(args.seed, &[i, 1])) * span;
            Grain {
                center: [x, y],
                radius,
                orientation: orientation(i),
            }
        })
        .collect()
}

fn run<N, I>(integrator: I, args: &Args) -> Result<(), Box<dyn Error>>
where
    N: DriverSet,
    I: Integrator<CartesianGrid<2>, FiniteVolume, N> + 'static,
{
    let grid = CartesianGrid::new(
        [args.cells, args.cells],
        [args.block, args.block],
        [0.0, 0.0],
        [H, H],
    )?;
    let mut model = ModelC::<N>::classic();
    if args.orient {
        model = model.with_orientations();
    }
    model.noise_amplitude = args.noise;

    let mut sim = Simulation::new(
        grid,
        FiniteVolume::default(),
        model,
        integrator,
        RayonScheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().expect("the model declares a stable dt");
    let steps = (args.time / dt).ceil() as u64;

    let seeds = seed_positions(args);
    println!(
        "model_c: {cells}² cells, {nblocks} blocks of {block}², {nseeds} seeds \
         (orientations: {orient}), dt = {dt:.5}, {steps} steps to t = {time}, \
         integrator {scheme:?}, snapshots every {every} -> {out}/",
        cells = args.cells,
        nblocks = sim.grid().num_blocks(),
        block = args.block,
        nseeds = seeds.len(),
        orient = if args.orient {
            "random"
        } else {
            "axis-aligned"
        },
        time = args.time,
        scheme = args.integrator,
        every = args.every,
        out = args.out,
    );

    let (phi_h, u_h) = (sim.model().phi(), sim.model().u());
    {
        let model = sim.model().clone();
        let (grid, state) = sim.state_mut();
        model.initialize_grains(grid, state, &seeds, 0.7);
    }

    // Observer pipeline: progress each step on the solver thread; Parquet +
    // statistics on the background runtime at snapshot cadence.
    let bar = progress_bar(steps);
    // Keep the bar redrawing even when individual steps are slow (large
    // grids) or the solver thread is briefly busy copying a snapshot.
    bar.enable_steady_tick(std::time::Duration::from_millis(200));
    let mut parquet = ParquetObserver::new(
        sim.grid().clone(),
        vec![("phi", phi_h), ("u", u_h)],
        1, // cadence owned by AsyncObserver
        &args.out,
    )?;
    if let Some(theta_h) = sim.model().theta0() {
        // θ₀ is time-invariant: written once per grid epoch, used for
        // grain coloring in renders.
        parquet = parquet.with_static(vec![("theta0", theta_h)]);
    }
    let stats = FieldStatsSink::new(
        sim.grid().clone(),
        vec![
            FieldStat {
                name: "phi",
                handle: phi_h,
                fraction_above: Some(0.0),
            },
            FieldStat {
                name: "u",
                handle: u_h,
                fraction_above: None,
            },
        ],
        bar.clone(),
    );
    let buffers = sim.snapshot_buffers(args.ring);
    let pipeline: AsyncObserver<f64, DenseStorage<f64>> = AsyncObserver::new(
        args.every,
        buffers,
        vec![Box::new(parquet), Box::new(stats)],
    );
    sim.attach_observer(Box::new(ProgressObserver::new(bar)));
    sim.attach_observer(Box::new(pipeline));

    for _ in 0..steps {
        sim.step(dt);
    }
    drop(sim); // drains the observer pipeline (final snapshots + finish)

    println!(
        "done. render with: python3 scripts/render_model_c.py {out}",
        out = args.out
    );
    Ok(())
}

/// The adaptive run: coarse base grid, one 2x refinement level tracking
/// the interface. Observation is synchronous (the async snapshot ring
/// cannot follow regrids), so prefer a generous --every.
fn run_amr<N, I>(integrator: I, args: &Args) -> Result<(), Box<dyn Error>>
where
    N: DriverSet,
    I: Integrator<AmrGrid<2>, FiniteVolume, N> + 'static,
{
    let base = CartesianGrid::new(
        [args.cells, args.cells],
        [args.block, args.block],
        [0.0, 0.0],
        [H, H],
    )?;
    let grid = AmrGrid::from_patches(base, &[2], &[])?;
    let mut model = ModelC::<N>::classic();
    model.noise_amplitude = args.noise;

    let mut sim = Simulation::adaptive(
        grid,
        FiniteVolume::default(),
        model,
        integrator,
        RayonScheduler,
        SystemAllocator,
        BergerOliger::new(
            GradientTagger {
                field: "phi",
                threshold: 0.15,
            },
            RegridPolicy {
                every: 8,
                buffer: 2,
                // min_width trades patch-tracking tightness against
                // work-item granularity; 4 coarse cells hugs the
                // interface band closely at demo scales.
                cluster: ClusterParams {
                    efficiency: 0.8,
                    min_width: 4,
                },
            },
        ),
    );
    let dt = sim.stable_dt().expect("the model declares a stable dt");
    let steps = (args.time / dt).ceil() as u64;

    let seeds = seed_positions(args);
    println!(
        "model_c (AMR): {cells}\u{b2} base cells + one 2x level, {nseeds} seeds, \
         dt = {dt:.5}, {steps} steps to t = {time}, integrator {scheme:?}, \
         snapshots every {every} -> {out}/",
        cells = args.cells,
        nseeds = seeds.len(),
        time = args.time,
        scheme = args.integrator,
        every = args.every,
        out = args.out,
    );

    let (phi_h, u_h) = (sim.model().phi(), sim.model().u());
    {
        let model = sim.model().clone();
        let (grid, state) = sim.state_mut();
        model.initialize_grains(grid.base(), state, &seeds, 0.7);
    }

    let bar = progress_bar(steps);
    bar.enable_steady_tick(std::time::Duration::from_millis(200));
    let parquet = ParquetObserver::new(
        sim.grid().clone(),
        vec![("phi", phi_h), ("u", u_h)],
        args.every,
        &args.out,
    )?;
    sim.attach_observer(Box::new(ProgressObserver::new(bar)));
    sim.attach_observer(Box::new(parquet));

    for _ in 0..steps {
        sim.step(dt);
    }
    drop(sim);

    println!(
        "done. render with: python3 scripts/render_model_c.py {out} --patches",
        out = args.out
    );
    Ok(())
}
