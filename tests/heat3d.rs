//! 3D validation: the framework's dimension-generic path end-to-end.
//!
//! Same construction as the 2D heat test: `u₀ = cos(kx·x)` is an exact
//! eigenvector of the 7-point discrete Laplacian under cell-centered mirror
//! ghosts, so forward Euler must reproduce `(1 − dt·κ·λ)ⁿ` to rounding
//! error — on a 2×2×2 block decomposition, which exercises 3D halo
//! exchange (including the dimension-sweep corner/edge resolution) exactly.

use legendre::{
    core::{
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
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
    integrators::ForwardEuler,
    physics::model::{Driver, Model, NoNoise, RhsContext},
};

/// Isotropic heat equation, generic over spatial dimension: the model code
/// below never names a dimension — `Heat<2>` and `Heat<3>` are the same
/// source.
struct Heat<const D: usize> {
    kappa: f64,
    u: Option<FieldHandle<f64>>,
}

impl<const D: usize, P> Model<CartesianGrid<D>, P> for Heat<D>
where
    P: Discretizes<CartesianGrid<D>, Laplacian>,
{
    type Scalar = f64;
    type Noise = NoNoise;

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

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        _driver: Driver,
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
        Some(0.5 / (2.0 * D as f64) * min_h * min_h / self.kappa)
    }
}

fn run_heat3d<Sch: Scheduler>(scheduler: Sch, steps: usize) -> (Vec<f64>, Vec<f64>) {
    const N: usize = 16;
    let h = 0.1;
    let kappa = 0.7;
    let grid = CartesianGrid::new([N; 3], [N / 2; 3], [0.0; 3], [h; 3]).unwrap();
    let kx = std::f64::consts::PI / (N as f64 * h);

    let model = Heat::<3> { kappa, u: None };
    let mut sim = Simulation::new(
        grid,
        FiniteDifference,
        model,
        ForwardEuler,
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();

    let u = sim.model().u.unwrap();
    let (grid, state) = sim.state_mut();
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        let mut v = state.view_mut(grid, block, u);
        for_each_interior(grid.block_cells(), |idx| {
            let x = grid.cell_center(block, idx)[0];
            v.set(idx, (kx * x).cos());
        });
    }

    for _ in 0..steps {
        sim.step(dt);
    }

    let lambda = 2.0f64.mul_add(-(kx * h).cos(), 2.0) / (h * h);
    let factor = (dt * kappa).mul_add(-lambda, 1.0).powi(steps as i32);
    let mut got = Vec::new();
    let mut expected = Vec::new();
    for b in 0..sim.grid().num_blocks() {
        let block = BlockId(b as u32);
        let v = sim.state().view(sim.grid(), block, u);
        for_each_interior(sim.grid().block_cells(), |idx| {
            let x = sim.grid().cell_center(block, idx)[0];
            got.push(v.get(idx));
            expected.push(factor * (kx * x).cos());
        });
    }
    (got, expected)
}

#[test]
fn heat3d_matches_discrete_eigenmode_decay() {
    let (got, expected) = run_heat3d(SerialScheduler, 200);
    for (g, e) in got.iter().zip(&expected) {
        approx::assert_relative_eq!(g, e, max_relative = 1e-10, epsilon = 1e-12);
    }
}

#[test]
fn heat3d_parallel_is_bitwise_identical_to_serial() {
    let (serial, _) = run_heat3d(SerialScheduler, 100);
    let (parallel, _) = run_heat3d(RayonScheduler, 100);
    assert_eq!(serial, parallel, "scheduling must not change results");
}
