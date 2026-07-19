//! `CodSpeed` benchmarks for the `legendre` PDE framework.
//!
//! These benchmarks exercise the full solver hot path — `Simulation::step`
//! driving `Integrator` → `Scheduler` → `Model::rhs_block` → `Discretizes` →
//! `Stencil` → views → `axpy` — on the dimension-generic heat equation
//! `∂u/∂t = κ∇²u` with no-flux (mirror) boundaries. The heat model is the
//! canonical framework example: benchmarking it measures the abstraction
//! machinery every model shares rather than any one physics kernel.
//!
//! Coverage:
//! - Time stepping in 2D and 3D across a range of grid sizes.
//! - Forward Euler (one rhs per step) vs. Runge–Kutta 4 (four stages).
//! - Serial vs. Rayon scheduling of the same workload.
//! - Grid construction and initial-condition fill (setup cost).

use legendre::{
    core::{
        scheduler::{RayonScheduler, SerialScheduler},
        scratch::Scratch,
        simulation::Simulation,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::{StorageBackend, SystemAllocator},
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
    integrators::{ForwardEuler, RungeKutta4},
    physics::model::{Model, RhsContext},
};

fn main() {
    divan::main();
}

/// Dimension-generic isotropic heat equation, generic over any discretization
/// policy that can realize a Laplacian.
struct Heat<const D: usize> {
    kappa: f64,
    u: Option<FieldHandle<f64>>,
}

impl<const D: usize, P> Model<CartesianGrid<D>, P> for Heat<D>
where
    P: Discretizes<CartesianGrid<D>, Laplacian>,
{
    type Scalar = f64;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.u = Some(builder.register("u", 1));
    }

    fn fill_ghosts<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<D>,
        state: &mut State<f64, S>,
        _t: f64,
    ) {
        fill_ghosts_mirror(grid, state, self.u.unwrap());
    }

    fn rhs_block<S: StorageBackend<f64>>(
        &self,
        ctx: &RhsContext<'_, CartesianGrid<D>, P>,
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

    fn stable_dt(&self, grid: &CartesianGrid<D>) -> Option<f64> {
        let h = grid.spacing(BlockId(0));
        let min_h = h.iter().copied().fold(f64::MAX, f64::min);
        Some(0.8 / (2.0 * D as f64) * min_h * min_h / self.kappa)
    }
}

const KAPPA: f64 = 0.7;

/// Build a `D`-dimensional cubic grid of `n` interior cells per side, split
/// into blocks of `block` cells per side.
fn make_grid<const D: usize>(n: usize, block: usize) -> CartesianGrid<D> {
    let cells = [n; D];
    let blocks = [block; D];
    let origin = [0.0; D];
    let spacing = [0.1; D];
    CartesianGrid::new(cells, blocks, origin, spacing).unwrap()
}

/// Seed the field with a smooth cosine profile along the first axis — an
/// eigenmode of the discrete Laplacian, so the initial data is representative
/// of a real run without diverging.
fn seed_initial_condition<const D: usize, I, Sch>(
    sim: &mut Simulation<CartesianGrid<D>, FiniteDifference, Heat<D>, I, Sch, SystemAllocator>,
) where
    I: legendre::integrators::Integrator<CartesianGrid<D>, FiniteDifference>,
    Sch: legendre::core::scheduler::Scheduler,
    FiniteDifference: Discretizes<CartesianGrid<D>, Laplacian>,
{
    let u = sim.model().u.unwrap();
    let kx = std::f64::consts::PI / (32.0 * 0.1);
    let (grid, state) = sim.state_mut();
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        let mut v = state.view_mut(grid, block, u);
        for_each_interior(grid.block_cells(), |idx| {
            let x = grid.cell_center(block, idx)[0];
            v.set(idx, (kx * x).cos());
        });
    }
}

/// Run `steps` timesteps of a fully-configured heat simulation and return it
/// so divan's `black_box` sees the result.
fn run_heat<const D: usize, Sch, I>(
    n: usize,
    block: usize,
    scheduler: Sch,
    integrator: I,
    steps: usize,
) -> f64
where
    Sch: legendre::core::scheduler::Scheduler,
    I: legendre::integrators::Integrator<CartesianGrid<D>, FiniteDifference>,
    FiniteDifference: Discretizes<CartesianGrid<D>, Laplacian>,
{
    let grid = make_grid::<D>(n, block);
    let model = Heat::<D> {
        kappa: KAPPA,
        u: None,
    };
    let mut sim = Simulation::new(
        grid,
        FiniteDifference,
        model,
        integrator,
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    seed_initial_condition(&mut sim);
    for _ in 0..steps {
        sim.step(dt);
    }
    // Return one field value so the compiler cannot elide the work.
    let u = sim.model().u.unwrap();
    sim.state().view(sim.grid(), BlockId(0), u).get([0; D])
}

/// 2D heat: 25 forward-Euler steps, serial scheduler, over grid sizes.
#[divan::bench(args = [32, 64, 128])]
fn heat_2d_euler_serial(bencher: divan::Bencher, n: usize) {
    bencher.bench(|| {
        divan::black_box(run_heat::<2, _, _>(
            n,
            32,
            SerialScheduler,
            ForwardEuler,
            25,
        ))
    });
}

/// 2D heat: 25 forward-Euler steps, Rayon scheduler, over grid sizes.
#[divan::bench(args = [32, 64, 128])]
fn heat_2d_euler_rayon(bencher: divan::Bencher, n: usize) {
    bencher
        .bench(|| divan::black_box(run_heat::<2, _, _>(n, 32, RayonScheduler, ForwardEuler, 25)));
}

/// 2D heat: 10 RK4 steps (four rhs evaluations per step), serial scheduler.
#[divan::bench(args = [32, 64, 128])]
fn heat_2d_rk4_serial(bencher: divan::Bencher, n: usize) {
    bencher.bench(|| {
        divan::black_box(run_heat::<2, _, _>(
            n,
            32,
            SerialScheduler,
            RungeKutta4::default(),
            10,
        ))
    });
}

/// 3D heat: 10 forward-Euler steps, serial scheduler, over grid sizes.
#[divan::bench(args = [32, 64])]
fn heat_3d_euler_serial(bencher: divan::Bencher, n: usize) {
    bencher.bench(|| {
        divan::black_box(run_heat::<3, _, _>(
            n,
            32,
            SerialScheduler,
            ForwardEuler,
            10,
        ))
    });
}

/// 3D heat: 10 forward-Euler steps, Rayon scheduler, over grid sizes.
#[divan::bench(args = [32, 64])]
fn heat_3d_euler_rayon(bencher: divan::Bencher, n: usize) {
    bencher
        .bench(|| divan::black_box(run_heat::<3, _, _>(n, 32, RayonScheduler, ForwardEuler, 10)));
}

/// Setup cost: grid construction plus initial-condition fill (no stepping).
#[divan::bench(args = [64, 128, 256])]
fn heat_2d_setup(bencher: divan::Bencher, n: usize) {
    bencher.bench(|| {
        let grid = make_grid::<2>(n, 32);
        let model = Heat::<2> {
            kappa: KAPPA,
            u: None,
        };
        let mut sim = Simulation::new(
            grid,
            FiniteDifference,
            model,
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        seed_initial_condition(&mut sim);
        let u = sim.model().u.unwrap();
        divan::black_box(sim.state().view(sim.grid(), BlockId(0), u).get([0, 0]))
    });
}
