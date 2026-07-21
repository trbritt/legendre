//! Periodic topology and the declarative initial-condition helper.
//!
//! Periodicity is topology: `face_neighbor` wraps, so the existing halo
//! exchange handles periodic dimensions with no new fill code. These tests
//! pin the wrap at every level — neighbor queries, exact ghost values
//! (multi-block, single-block self-wrap, wide ghosts, mixed
//! periodic/mirror dimensions), an eigenmode decay of the heat equation
//! under periodic boundaries exact to rounding error, and bitwise
//! scheduler-independence.

// Exact float equality is the point where used: ghost values are copies.
#![allow(clippy::float_cmp)]

use legendre::{
    core::{
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
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
        cartesian::{CartesianGrid, fill_from_fn, fill_ghosts_mirror, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::ForwardEuler,
    physics::model::{Driver, Model, NoNoise, RhsContext},
};

type DenseState = State<f64, DenseStorage<f64>>;

mod topology {
    use super::*;

    #[test]
    fn face_neighbor_wraps_only_periodic_dimensions() {
        // 2×2 blocks, periodic in x only.
        let grid = CartesianGrid::new([4, 4], [2, 2], [0.0; 2], [1.0; 2])
            .unwrap()
            .with_periodic([true, false]);
        // Interior faces unchanged.
        assert_eq!(grid.face_neighbor(BlockId(0), 0, 1), Some(BlockId(1)));
        // x wraps: block 0's low face reaches block 1, block 1's high face
        // reaches block 0.
        assert_eq!(grid.face_neighbor(BlockId(0), 0, -1), Some(BlockId(1)));
        assert_eq!(grid.face_neighbor(BlockId(1), 0, 1), Some(BlockId(0)));
        // y does not wrap.
        assert_eq!(grid.face_neighbor(BlockId(0), 1, -1), None);
        assert_eq!(grid.face_neighbor(BlockId(2), 1, 1), None);
    }

    #[test]
    fn single_block_periodic_dimension_wraps_onto_itself() {
        let grid = CartesianGrid::new([4], [4], [0.0], [1.0])
            .unwrap()
            .with_periodic([true]);
        assert_eq!(grid.face_neighbor(BlockId(0), 0, 1), Some(BlockId(0)));
        assert_eq!(grid.face_neighbor(BlockId(0), 0, -1), Some(BlockId(0)));
        // Non-periodic single block stays a domain boundary.
        let plain = CartesianGrid::new([4], [4], [0.0], [1.0]).unwrap();
        assert_eq!(plain.face_neighbor(BlockId(0), 0, 1), None);
    }
}

mod ghosts {
    use super::*;

    /// 1D, two blocks of four cells, u = global cell index: the wrap must
    /// deliver the far end's interior into the boundary ghosts.
    #[test]
    fn periodic_ghost_values_are_exact_across_blocks() {
        let grid = CartesianGrid::new([8], [4], [0.0], [1.0])
            .unwrap()
            .with_periodic([true]);
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 1);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        fill_from_fn(&grid, &mut state, u, |[x]| x - 0.5); // global index
        fill_ghosts_mirror(&grid, &mut state, u);

        let v0 = state.view(&grid, BlockId(0), u);
        let v1 = state.view(&grid, BlockId(1), u);
        // Interior faces exchange as before.
        assert_eq!(v0.get([4]), 4.0);
        assert_eq!(v1.get([-1]), 3.0);
        // Domain faces wrap instead of mirroring.
        assert_eq!(v0.get([-1]), 7.0, "low ghost wraps to last cell");
        assert_eq!(v1.get([4]), 0.0, "high ghost wraps to first cell");
    }

    /// One block spanning a periodic dimension: the self-wrap branch, with
    /// a two-layer ghost ring.
    #[test]
    fn single_block_self_wrap_fills_every_layer() {
        let grid = CartesianGrid::new([4], [4], [0.0], [1.0])
            .unwrap()
            .with_periodic([true]);
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 2);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        fill_from_fn(&grid, &mut state, u, |[x]| x - 0.5);
        fill_ghosts_mirror(&grid, &mut state, u);

        let v = state.view(&grid, BlockId(0), u);
        assert_eq!(v.get([-1]), 3.0);
        assert_eq!(v.get([-2]), 2.0);
        assert_eq!(v.get([4]), 0.0);
        assert_eq!(v.get([5]), 1.0);
    }

    /// Mixed dimensions: periodic in x, mirror in y — including the corner
    /// ghosts the dimension sweep must resolve consistently.
    #[test]
    fn mixed_periodic_and_mirror_dimensions() {
        let grid = CartesianGrid::new([4, 4], [4, 4], [0.0; 2], [1.0; 2])
            .unwrap()
            .with_periodic([true, false]);
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 1);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        // u(i, j) = 10j + i from cell centers.
        fill_from_fn(&grid, &mut state, u, |[x, y]| {
            10.0f64.mul_add(y - 0.5, x - 0.5)
        });
        fill_ghosts_mirror(&grid, &mut state, u);

        let v = state.view(&grid, BlockId(0), u);
        // x wraps: ghost (-1, j) ← (3, j).
        assert_eq!(v.get([-1, 0]), 3.0);
        assert_eq!(v.get([4, 2]), 20.0);
        // y mirrors: ghost (i, -1) ← (i, 0).
        assert_eq!(v.get([1, -1]), 1.0);
        assert_eq!(v.get([2, 4]), 32.0);
        // Corner: wrap in x, mirror in y — (-1, -1) ← (3, 0).
        assert_eq!(v.get([-1, -1]), 3.0);
        assert_eq!(v.get([4, 4]), 30.0);
    }
}

mod init {
    use super::*;

    #[test]
    fn fill_from_fn_writes_every_interior_cell_center() {
        let grid = CartesianGrid::new([4, 4], [2, 2], [1.0, -1.0], [0.5, 0.25]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 1);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        fill_from_fn(&grid, &mut state, u, |[x, y]| 100.0f64.mul_add(x, y));

        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            let v = state.view(&grid, block, u);
            for_each_interior(grid.block_cells(), |idx| {
                let [x, y] = grid.cell_center(block, idx);
                assert_eq!(v.get(idx), 100.0f64.mul_add(x, y));
            });
        }
    }
}

mod physics {
    use super::*;

    /// Heat equation, generic over dimension, mirror-or-wrap boundaries
    /// decided entirely by the grid's topology.
    struct Heat {
        kappa: f64,
        u: Option<FieldHandle<f64>>,
    }

    impl<D> Model<CartesianGrid<2>, D> for Heat
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
            fill_ghosts_mirror(grid, state, self.u.unwrap());
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
            for_each_interior(input.interior(), |idx| {
                output.set(idx, output.get(idx) * self.kappa);
            });
        }

        fn stable_dt(&self, h: [f64; 2]) -> Option<f64> {
            Some(0.25 * h[0].min(h[1]).powi(2) / self.kappa)
        }
    }

    /// Run the fully periodic eigenmode `u₀ = cos(2πx/L)` — an exact
    /// eigenvector of the periodic 5-point Laplacian — and return
    /// (got, expected) under forward Euler's discrete decay factor.
    fn run_periodic_mode(blocks_per_side: usize, steps: usize) -> (Vec<f64>, Vec<f64>) {
        const N: usize = 32;
        let h = 0.1;
        let kappa = 0.7;
        let grid = CartesianGrid::new([N, N], [N / blocks_per_side; 2], [0.0; 2], [h; 2])
            .unwrap()
            .with_periodic([true, true]);
        // Full period across the domain: forbidden under mirror boundaries,
        // exact under periodic ones.
        let kx = 2.0 * std::f64::consts::PI / (N as f64 * h);

        let model = Heat { kappa, u: None };
        let mut sim = Simulation::new(
            grid,
            FiniteDifference,
            model,
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let dt = sim.stable_dt().unwrap();
        let u = sim.model().u.unwrap();
        {
            let (grid, state) = sim.state_mut();
            fill_from_fn(grid, state, u, |[x, _]| (kx * x).cos());
        }
        for _ in 0..steps {
            sim.step(dt);
        }

        let lambda = 2.0f64.mul_add(-(kx * h).cos(), 2.0) / (h * h);
        let factor = (dt * kappa).mul_add(-lambda, 1.0).powi(steps as i32);
        let (mut got, mut expected) = (Vec::new(), Vec::new());
        for b in 0..sim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let v = sim.state().view(sim.grid(), block, u);
            for_each_interior(sim.grid().block_cells(), |idx| {
                let [x, _] = sim.grid().cell_center(block, idx);
                got.push(v.get(idx));
                expected.push(factor * (kx * x).cos());
            });
        }
        (got, expected)
    }

    #[test]
    fn periodic_eigenmode_decays_exactly_multi_block() {
        let (got, expected) = run_periodic_mode(2, 200);
        for (g, e) in got.iter().zip(&expected) {
            approx::assert_relative_eq!(g, e, max_relative = 1e-10, epsilon = 1e-12);
        }
    }

    #[test]
    fn periodic_eigenmode_decays_exactly_single_block() {
        // One block per side: exercises the self-wrap ghost branch through
        // the whole solver chain.
        let (got, expected) = run_periodic_mode(1, 200);
        for (g, e) in got.iter().zip(&expected) {
            approx::assert_relative_eq!(g, e, max_relative = 1e-10, epsilon = 1e-12);
        }
    }

    #[test]
    fn periodic_run_is_bitwise_identical_across_schedulers() {
        fn run<Sch: Scheduler>(scheduler: Sch) -> Vec<f64> {
            const N: usize = 32;
            let grid = CartesianGrid::new([N, N], [N / 2; 2], [0.0; 2], [0.1; 2])
                .unwrap()
                .with_periodic([true, true]);
            let kx = 2.0 * std::f64::consts::PI / (N as f64 * 0.1);
            let model = Heat {
                kappa: 0.7,
                u: None,
            };
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
            {
                let (grid, state) = sim.state_mut();
                fill_from_fn(grid, state, u, |[x, y]| (kx * x).cos() + (kx * y).sin());
            }
            for _ in 0..100 {
                sim.step(dt);
            }
            let mut out = Vec::new();
            for b in 0..sim.grid().num_blocks() {
                let v = sim.state().view(sim.grid(), BlockId(b as u32), u);
                for_each_interior(sim.grid().block_cells(), |idx| out.push(v.get(idx)));
            }
            out
        }
        assert_eq!(
            run(SerialScheduler),
            run(RayonScheduler),
            "scheduling must not change periodic results"
        );
    }
}
