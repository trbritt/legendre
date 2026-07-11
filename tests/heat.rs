//! End-to-end validation of the trait hierarchy on the 2D heat equation.
//!
//! `u_t = κ∇²u` with no-flux (mirror) boundaries. The initial condition
//! `u₀ = cos(kx·x)` with `kx = π/Lx` is an exact eigenvector of the
//! 5-point discrete Laplacian under cell-centered mirror ghosts, so forward
//! Euler must reproduce the discrete decay factor
//! `(1 − dt·κ·λ)ⁿ, λ = (2 − 2cos(kx·h))/h²` to rounding error. This pins the
//! whole chain: `Simulation` → `Integrator` → `Scheduler` →
//! `Model::rhs_block` → `Discretizes` → `Stencil` → views → `axpy`.

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
    integrators::{ForwardEuler, Integrator, RungeKutta4},
    physics::model::{Model, RhsContext},
};

/// Isotropic heat equation on a 2D Cartesian grid, generic over any
/// discretization policy that can realize a Laplacian.
struct Heat {
    kappa: f64,
    u: Option<FieldHandle<f64>>,
}

impl<D> Model<CartesianGrid<2>, D> for Heat
where
    D: Discretizes<CartesianGrid<2>, Laplacian>,
{
    type Scalar = f64;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.u = Some(builder.register("u", 1));
    }

    fn fill_ghosts<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<2>,
        state: &mut State<f64, S>,
        _t: f64,
    ) {
        fill_ghosts_mirror(grid, state, self.u.unwrap());
    }

    fn rhs_block<S: StorageBackend<f64>>(
        &self,
        ctx: &RhsContext<'_, CartesianGrid<2>, D>,
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

    fn stable_dt(&self, grid: &CartesianGrid<2>) -> Option<f64> {
        let h = grid.spacing(BlockId(0));
        Some(0.25 * h[0].min(h[1]).powi(2) / self.kappa)
    }
}

/// Run `steps` of the heat model; `per_step_factor(z)` is the scheme's
/// amplification of a linear mode per step, z = dt·κ·λ (forward Euler:
/// 1 − z; RK4: the quartic Taylor polynomial of e^−z — both *exact* for
/// the discrete eigenmode, pinning integrator wiring to rounding error).
fn run_heat<Sch, I>(
    scheduler: Sch,
    integrator: I,
    steps: usize,
    per_step_factor: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>, [usize; 2])
where
    Sch: Scheduler,
    I: Integrator<CartesianGrid<2>, FiniteDifference>,
{
    const N: usize = 32;
    let h = 0.1;
    let kappa = 0.7;
    let grid = CartesianGrid::new([N, N], [N / 4, N / 4], [0.0, 0.0], [h, h]).unwrap();
    let kx = std::f64::consts::PI / (N as f64 * h);

    let model = Heat { kappa, u: None };
    let mut sim = Simulation::new(
        grid,
        FiniteDifference,
        model,
        integrator,
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();

    // Initial condition: the discrete Neumann eigenmode cos(kx·x).
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

    // Exact discrete decay of the eigenmode under the given scheme.
    let lambda = 2.0f64.mul_add(-(kx * h).cos(), 2.0) / (h * h);
    let factor = per_step_factor(dt * kappa * lambda).powi(steps as i32);
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
    (got, expected, [N, N])
}

fn euler_factor(z: f64) -> f64 {
    1.0 - z
}

fn rk4_factor(z: f64) -> f64 {
    1.0 - z + z * z / 2.0 - z * z * z / 6.0 + z * z * z * z / 24.0
}

#[test]
fn heat_matches_discrete_eigenmode_decay() {
    let (got, expected, _) = run_heat(SerialScheduler, ForwardEuler, 200, euler_factor);
    for (g, e) in got.iter().zip(&expected) {
        approx::assert_relative_eq!(g, e, max_relative = 1e-10, epsilon = 1e-12);
    }
}

#[test]
fn rk4_matches_discrete_eigenmode_decay() {
    let (got, expected, _) = run_heat(SerialScheduler, RungeKutta4::default(), 200, rk4_factor);
    for (g, e) in got.iter().zip(&expected) {
        approx::assert_relative_eq!(g, e, max_relative = 1e-10, epsilon = 1e-12);
    }
}

#[test]
fn parallel_scheduler_is_bitwise_identical_to_serial() {
    let (serial, _, _) = run_heat(SerialScheduler, ForwardEuler, 100, euler_factor);
    let (parallel, _, _) = run_heat(RayonScheduler, ForwardEuler, 100, euler_factor);
    assert_eq!(serial, parallel, "scheduling must not change results");
}

#[test]
fn rk4_parallel_is_bitwise_identical_to_serial() {
    let (serial, _, _) = run_heat(SerialScheduler, RungeKutta4::default(), 100, rk4_factor);
    let (parallel, _, _) = run_heat(RayonScheduler, RungeKutta4::default(), 100, rk4_factor);
    assert_eq!(serial, parallel, "scheduling must not change results");
}
