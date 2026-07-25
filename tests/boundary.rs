//! Physical boundary conditions beyond no-flux: exact ghost values for
//! Dirichlet and inhomogeneous-flux fills, their coexistence with interior
//! halo exchange, the block-vs-ghost-width guard, and a model *configured*
//! with mixed per-face conditions (the answer to "how do I set a boundary
//! condition"): its `fill_ghosts` returns a [`FaceBc`] per (dimension, side).
//!
//! Exact float equality is the point: the ghost algebra is closed-form, so
//! every boundary value is reproducible to the bit. The remaining allows keep
//! the test arithmetic written the clear, textbook way.
#![allow(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::many_single_char_names
)]

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
        boundary::FaceBc,
        cartesian::{CartesianGrid, fill_from_fn, fill_ghosts_bc},
        grid::{BlockId, Grid},
    },
    integrators::ForwardEuler,
    physics::model::{Driver, Model, NoNoise, RhsContext},
};

const H: f64 = 0.5;

/// Build a single-block grid + one field of the given ghost width, its
/// interior set to `f(x, y)`.
fn one_block_state(
    n: usize,
    ghost: u32,
    f: impl FnMut([f64; 2]) -> f64,
) -> (
    CartesianGrid<2>,
    State<f64, <SystemAllocator as legendre::core::storage::Allocator<f64>>::Storage>,
    FieldHandle<f64>,
) {
    let grid = CartesianGrid::single_block([n, n], [0.0, 0.0], [H, H]).unwrap();
    let mut builder = StateBuilder::<f64>::new();
    let u = builder.register("u", ghost);
    let mut state = builder.build(&grid, &SystemAllocator);
    fill_from_fn(&grid, &mut state, u, f);
    (grid, state, u)
}

/// Dirichlet ghosts are the odd reflection about the face value, at *every*
/// ghost layer — so a ghost-width-2 field (biharmonic support) fills both.
#[test]
fn dirichlet_ghosts_exact_including_second_layer() {
    let value = 2.5;
    let (grid, mut state, u) = one_block_state(8, 2, |[x, y]| 1.0 + 2.0 * x - 0.5 * y);
    fill_ghosts_bc(&grid, &mut state, u, |_, _| FaceBc::Dirichlet(value));

    let v = state.view(&grid, BlockId(0), u);
    for j in 0..8isize {
        // Low-x face: ghost(-1-k) = 2·value − interior(k).
        assert_eq!(v.get([-1, j]), 2.0 * value - v.get([0, j]));
        assert_eq!(v.get([-2, j]), 2.0 * value - v.get([1, j]));
        // High-x face: ghost(n+k) = 2·value − interior(n-1-k).
        assert_eq!(v.get([8, j]), 2.0 * value - v.get([7, j]));
        assert_eq!(v.get([9, j]), 2.0 * value - v.get([6, j]));
    }
}

/// Inhomogeneous flux ghosts add the `(2k+1)·h·g` ramp to the mirror image,
/// so the one-sided normal gradient across the face is exactly `g`.
#[test]
fn flux_ghosts_exact_including_second_layer() {
    let g = 1.5;
    let (grid, mut state, u) = one_block_state(8, 2, |[x, y]| 0.3 * x + 0.7 * y);
    fill_ghosts_bc(&grid, &mut state, u, |_, _| FaceBc::Flux(g));

    let v = state.view(&grid, BlockId(0), u);
    for j in 0..8isize {
        // ghost(-1-k) = interior(k) + (2k+1)·h·g.
        assert_eq!(v.get([-1, j]), v.get([0, j]) + 1.0 * H * g);
        assert_eq!(v.get([-2, j]), v.get([1, j]) + 3.0 * H * g);
        // Innermost layer realizes the prescribed gradient across the face
        // (a subtraction, so compare within a few ULP, not bit-exact).
        approx::assert_relative_eq!((v.get([-1, j]) - v.get([0, j])) / H, g, epsilon = 1e-12);
    }
    // Flux(0) must be exactly the no-flux mirror.
    let (grid2, mut s2, u2) = one_block_state(8, 1, |[x, _]| x * x);
    fill_ghosts_bc(&grid2, &mut s2, u2, |_, _| FaceBc::Flux(0.0));
    let v2 = s2.view(&grid2, BlockId(0), u2);
    assert_eq!(v2.get([-1, 3]), v2.get([0, 3]));
}

/// The boundary condition applies only at *physical* faces; interior block
/// faces still exchange the neighbor's data. On a 2×2 block grid, block 0's
/// low faces are Dirichlet while its high faces copy neighbors' interiors.
#[test]
fn mixed_bc_and_halo_exchange_coexist() {
    let value = -1.0;
    let grid = CartesianGrid::new([8, 8], [4, 4], [0.0, 0.0], [H, H]).unwrap();
    let mut builder = StateBuilder::<f64>::new();
    let u = builder.register("u", 1);
    let mut state = builder.build(&grid, &SystemAllocator);
    fill_from_fn(&grid, &mut state, u, |[x, y]| 1.0 + x + 2.0 * y);
    fill_ghosts_bc(&grid, &mut state, u, |_, _| FaceBc::Dirichlet(value));

    // Blocks: index = cx + 2·cy. Block 0 = (0,0); its +x neighbor is block 1.
    let b0 = state.view(&grid, BlockId(0), u);
    let b1 = state.view(&grid, BlockId(1), u);
    for j in 0..4isize {
        // Low-x face is physical: Dirichlet reflection.
        assert_eq!(b0.get([-1, j]), 2.0 * value - b0.get([0, j]));
        // High-x face is interior: exchange copies block 1's first column,
        // *not* the boundary condition.
        assert_eq!(b0.get([4, j]), b1.get([0, j]));
    }
}

/// The guard: a ghost ring wider than the block would reflect onto the
/// block's own ghosts (garbage), so the fill must refuse it.
#[test]
#[should_panic(expected = "ghost width")]
fn ghost_wider_than_block_panics() {
    // Block is 3 cells thick; a ghost-4 field cannot be mirror-filled.
    let (grid, mut state, u) = one_block_state(3, 4, |_| 0.0);
    fill_ghosts_bc(&grid, &mut state, u, |_, _| FaceBc::Mirror);
}

// --- A model configured with mixed boundary conditions ---------------------

/// Heat with Dirichlet ends in x (`lo`, `hi`) and no-flux walls in y. The
/// boundary condition is *data the model owns* and returns per face.
struct HeatBc {
    kappa: f64,
    lo: f64,
    hi: f64,
    u: Option<FieldHandle<f64>>,
}

impl<D> Model<CartesianGrid<2>, D> for HeatBc
where
    D: Discretizes<CartesianGrid<2>, Laplacian>,
{
    type Scalar = f64;
    type Drivers = NoNoise;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.u = Some(builder.register("u", 1));
    }

    fn fill_ghosts<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<2>,
        state: &mut State<f64, S>,
        _t: f64,
    ) {
        fill_ghosts_bc(grid, state, self.u.unwrap(), |dim, side| {
            match (dim, side) {
                (0, -1) => FaceBc::Dirichlet(self.lo), // x = 0 wall
                (0, 1) => FaceBc::Dirichlet(self.hi),  // x = L wall
                _ => FaceBc::Mirror,                   // insulated in y
            }
        });
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        _driver: Driver,
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
        for_each(input.interior(), |idx| {
            output.set(idx, output.get(idx) * self.kappa);
        });
    }

    fn stable_dt(&self, h: [f64; 2]) -> Option<f64> {
        Some(0.25 * h[0].min(h[1]).powi(2) / self.kappa)
    }
}

// Small local `for_each_interior` to avoid one more import line.
fn for_each(interior: [usize; 2], mut f: impl FnMut([isize; 2])) {
    for j in 0..interior[1] as isize {
        for i in 0..interior[0] as isize {
            f([i, j]);
        }
    }
}

/// The exact linear profile between the two Dirichlet walls is a discrete
/// fixed point: a Dirichlet ghost is the linear extrapolation of a linear
/// field, so its Laplacian — hence dU/dt — is zero everywhere, boundary
/// cells included. Run and confirm nothing moves (to rounding error). This
/// pins the physical correctness of the Dirichlet ghost, not just its
/// algebra.
fn run_fixed_point<Sch: Scheduler>(scheduler: Sch) -> (Vec<f64>, Vec<f64>) {
    const N: usize = 16;
    let (a, b) = (0.3, 1.7);
    let kappa = 0.7;
    let l = N as f64 * H;
    let grid = CartesianGrid::new([N, N], [N / 2, N / 2], [0.0, 0.0], [H, H]).unwrap();

    let mut sim = Simulation::new(
        grid,
        FiniteDifference,
        HeatBc {
            kappa,
            lo: a,
            hi: b,
            u: None,
        },
        ForwardEuler,
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    let u = sim.model().u.unwrap();

    // Exact steady profile u(x) = a + (b − a)·x/L at cell centers.
    let profile = |x: f64| (b - a).mul_add(x / l, a);
    {
        let (grid, state) = sim.state_mut();
        for blk in 0..grid.num_blocks() {
            let block = BlockId(blk as u32);
            let mut v = state.view_mut(grid, block, u);
            for_each(grid.block_cells(), |idx| {
                v.set(idx, profile(grid.cell_center(block, idx)[0]));
            });
        }
    }

    for _ in 0..200 {
        sim.step(dt);
    }

    let mut got = Vec::new();
    let mut expected = Vec::new();
    for blk in 0..sim.grid().num_blocks() {
        let block = BlockId(blk as u32);
        let v = sim.state().view(sim.grid(), block, u);
        for_each(sim.grid().block_cells(), |idx| {
            got.push(v.get(idx));
            expected.push(profile(sim.grid().cell_center(block, idx)[0]));
        });
    }
    (got, expected)
}

#[test]
fn linear_profile_is_a_fixed_point_under_dirichlet() {
    let (got, expected) = run_fixed_point(SerialScheduler);
    for (g, e) in got.iter().zip(&expected) {
        approx::assert_relative_eq!(g, e, max_relative = 1e-11, epsilon = 1e-12);
    }
}

#[test]
fn dirichlet_model_is_bitwise_scheduler_independent() {
    let (serial, _) = run_fixed_point(SerialScheduler);
    let (parallel, _) = run_fixed_point(RayonScheduler);
    assert_eq!(
        serial, parallel,
        "the BC fill must not depend on scheduling"
    );
}
