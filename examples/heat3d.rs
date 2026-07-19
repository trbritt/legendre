//! 3D heat diffusion with the full observation pipeline — the
//! dimension-generality proof for the async observers.
//!
//! A Gaussian hot spot diffuses in a 96³ box with no-flux boundaries;
//! progress bar on the solver thread, statistics + Parquet snapshots
//! (columns step, t, epoch, u, plus per-epoch x, y, z) on the background
//! runtime.
//!
//! ```text
//! cargo run --release --example heat3d -- --help
//! ```

use clap::Parser;
use legendre::{
    core::{
        observer::AsyncObserver,
        scheduler::RayonScheduler,
        scratch::Scratch,
        simulation::Simulation,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::{DenseStorage, StorageBackend, SystemAllocator},
    },
    discretization::{
        finite_difference::FiniteDifference,
        operators::{Discretizes, Laplacian},
        stencil::Stencil,
    },
    geometry::{
        cartesian::{CartesianGrid, fill_ghosts_mirror, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::ForwardEuler,
    io::{
        parquet::ParquetObserver,
        progress::{FieldStat, FieldStatsSink, ProgressObserver, progress_bar},
    },
    physics::model::{Driver, Model, NoNoise, RhsContext},
};
use std::error::Error;

struct Heat3 {
    kappa: f64,
    u: Option<FieldHandle<f64>>,
}

impl<P: Discretizes<CartesianGrid<3>, Laplacian>> Model<CartesianGrid<3>, P> for Heat3 {
    type Scalar = f64;
    type Drivers = NoNoise;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.u = Some(builder.register("u", 1));
    }

    fn fill_ghosts<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<3>,
        state: &mut State<f64, S>,
        _t: f64,
    ) {
        fill_ghosts_mirror(grid, state, self.u.unwrap());
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        _driver: Driver,
        ctx: &RhsContext<'_, CartesianGrid<3>, P>,
        state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        let u = self.u.unwrap();
        let stencil = ctx.disc.build(ctx.grid, Laplacian);
        let input = state.view(ctx.grid, ctx.block, u);
        let mut output = out.view_mut(ctx.grid, ctx.block, u);
        stencil.apply(ctx.grid, ctx.block, input, &mut output);
        for_each_interior(input.interior(), |idx| {
            output.set(idx, output.get(idx) * self.kappa);
        });
    }

    fn stable_dt(&self, grid: &CartesianGrid<3>) -> Option<f64> {
        let h = grid.spacing(BlockId(0));
        let min_h = h.iter().copied().fold(f64::MAX, f64::min);
        Some(0.8 / 6.0 * min_h * min_h / self.kappa) // r_total = 0.4 < 0.5
    }
}

/// 3D heat diffusion of a Gaussian hot spot, with async Parquet snapshots
/// and live statistics.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Interior cells per side (must be divisible by --block)
    #[arg(long, default_value_t = 96)]
    cells: usize,

    /// Block size per side
    #[arg(long, default_value_t = 32)]
    block: usize,

    /// Simulated end time
    #[arg(long, default_value_t = 2.0)]
    time: f64,

    /// Snapshot cadence in steps
    #[arg(long, default_value_t = 200)]
    every: u64,

    /// Output directory for Parquet snapshots
    #[arg(long, default_value = "data/heat3d")]
    out: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let spacing = 0.1;
    let grid = CartesianGrid::new([args.cells; 3], [args.block; 3], [0.0; 3], [spacing; 3])?;
    let model = Heat3 {
        kappa: 0.7,
        u: None,
    };
    let mut sim = Simulation::new(
        grid,
        FiniteDifference,
        model,
        ForwardEuler,
        RayonScheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().expect("the model declares a stable dt");
    let steps = (args.time / dt).ceil() as u64;

    println!(
        "heat3d: {cells}³ cells, {nblocks} blocks of {block}³, dt = {dt:.5}, \
         {steps} steps to t = {time}, snapshots every {every} -> {out}/",
        cells = args.cells,
        nblocks = sim.grid().num_blocks(),
        block = args.block,
        time = args.time,
        every = args.every,
        out = args.out,
    );

    // Gaussian hot spot at the domain center.
    let u_h = sim.model().u.unwrap();
    let extent = args.cells as f64 * spacing;
    let center = extent / 2.0;
    let sigma2 = (0.08 * extent).powi(2);
    {
        let (grid, state) = sim.state_mut();
        for b in 0..grid.num_blocks() {
            let blk = BlockId(b as u32);
            let mut v = state.view_mut(grid, blk, u_h);
            for_each_interior(grid.block_cells(), |idx| {
                let [x, y, z] = grid.cell_center(blk, idx);
                let r2 = (x - center).mul_add(
                    x - center,
                    (y - center).mul_add(y - center, (z - center).powi(2)),
                );
                v.set(idx, (-r2 / (2.0 * sigma2)).exp());
            });
        }
    }

    let bar = progress_bar(steps);
    bar.enable_steady_tick(std::time::Duration::from_millis(200));
    let parquet = ParquetObserver::new(
        sim.grid().clone(),
        vec![("u", u_h)],
        1, // cadence owned by AsyncObserver
        &args.out,
    )?;
    let stats = FieldStatsSink::new(
        sim.grid().clone(),
        vec![FieldStat {
            name: "u",
            handle: u_h,
            fraction_above: Some(0.1),
        }],
        bar.clone(),
    );
    let buffers = sim.snapshot_buffers(3);
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
    drop(sim);
    Ok(())
}
