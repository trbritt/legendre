//! AMR Phase B validation: a patch is indistinguishable from a
//! uniform-grid block. Stencil kernels applied through the `AmrGrid`
//! trait impl must reproduce the `CartesianGrid` results bit for bit, and
//! the whole `Simulation` stack runs on an `AmrGrid` unchanged.

// Bitwise identity is the property under test.
#![allow(clippy::float_cmp)]

use legendre::{
    core::{
        scheduler::SerialScheduler,
        scratch::Scratch,
        simulation::Simulation,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::{DenseStorage, StorageBackend, SystemAllocator},
    },
    discretization::{
        finite_difference::{CentralLaplacian, FiniteDifference},
        finite_volume::KarmaRappelFlux,
        stencil::Stencil,
    },
    geometry::{
        amr::{AmrGrid, CellBox},
        cartesian::CartesianGrid,
        grid::{BlockId, Grid},
    },
    integrators::ForwardEuler,
    physics::model::{Driver, Model, NoNoise, RhsContext},
};

type DenseState<G> = (G, State<f64, DenseStorage<f64>>, FieldHandle<f64>);

/// A state with one ghost-1 field on `grid`, every slab entry (ghosts
/// included) filled by a deterministic function of its offset.
fn filled_state<G: Grid>(grid: G) -> DenseState<G> {
    let mut builder = StateBuilder::<f64>::new();
    let u = builder.register("u", 1);
    let mut state: State<f64, DenseStorage<f64>> = builder.build(&grid, &SystemAllocator);
    for b in 0..grid.num_blocks() {
        for (i, v) in state.slab_mut(BlockId(b as u32), u).iter_mut().enumerate() {
            *v = (((i * 37 + b * 101) % 89) as f64).mul_add(0.037, -1.2);
        }
    }
    (grid, state, u)
}

/// A 6×6 level-1 patch at spacing 0.25 (base 8×8 at 0.5, ratio 2).
fn one_patch_amr() -> AmrGrid<2> {
    let base = CartesianGrid::new([8, 8], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
    AmrGrid::from_patches(
        base,
        &[2],
        &[vec![CellBox {
            lo: [4, 4],
            hi: [10, 10],
        }]],
    )
    .unwrap()
}

/// The uniform twin of that patch: same extent, same spacing.
fn patch_twin() -> CartesianGrid<2> {
    CartesianGrid::single_block([6, 6], [0.0, 0.0], [0.25, 0.25]).unwrap()
}

#[test]
fn laplacian_on_a_patch_is_bitwise_identical_to_uniform() {
    let (amr, amr_state, ua) = filled_state(one_patch_amr());
    let (uni, uni_state, uu) = filled_state(patch_twin());
    let patch = BlockId(1); // block 0 is the base block, 1 the patch

    // Same slab layout by construction…
    assert_eq!(amr.block_len(patch, 1), uni.block_len(BlockId(0), 1));
    // …but different block index: refill the patch slab to match the twin.
    let mut amr_state = amr_state;
    let twin: Vec<f64> = uni_state.slab(BlockId(0), uu).to_vec();
    amr_state.slab_mut(patch, ua).copy_from_slice(&twin);

    let mut amr_out = amr_state.like(&amr, &SystemAllocator);
    let mut uni_out = uni_state.like(&uni, &SystemAllocator);
    CentralLaplacian.apply(
        &amr,
        patch,
        amr_state.view(&amr, patch, ua),
        &mut amr_out.view_mut(&amr, patch, ua),
    );
    CentralLaplacian.apply(
        &uni,
        BlockId(0),
        uni_state.view(&uni, BlockId(0), uu),
        &mut uni_out.view_mut(&uni, BlockId(0), uu),
    );
    assert_eq!(
        amr_out.slab(patch, ua),
        uni_out.slab(BlockId(0), uu),
        "patch Laplacian must equal the uniform twin bitwise"
    );
}

#[test]
fn karma_rappel_on_a_patch_is_bitwise_identical_to_uniform() {
    let (amr, amr_state, ua) = filled_state(one_patch_amr());
    let (uni, uni_state, uu) = filled_state(patch_twin());
    let patch = BlockId(1);
    let mut amr_state = amr_state;
    let twin: Vec<f64> = uni_state.slab(BlockId(0), uu).to_vec();
    amr_state.slab_mut(patch, ua).copy_from_slice(&twin);

    let flux = KarmaRappelFlux::new(0.06, 1e-8);
    let mut amr_out = amr_state.like(&amr, &SystemAllocator);
    let mut uni_out = uni_state.like(&uni, &SystemAllocator);
    flux.apply(
        &amr,
        patch,
        amr_state.view(&amr, patch, ua),
        &mut amr_out.view_mut(&amr, patch, ua),
    );
    flux.apply(
        &uni,
        BlockId(0),
        uni_state.view(&uni, BlockId(0), uu),
        &mut uni_out.view_mut(&uni, BlockId(0), uu),
    );
    assert_eq!(amr_out.slab(patch, ua), uni_out.slab(BlockId(0), uu));
}

/// du/dt = −u³ on a single cell: the whole Simulation/Integrator stack on
/// an `AmrGrid`, compared bitwise against the Cartesian run. (The model
/// is written once per grid type; the bodies are identical because the
/// views are the same concrete type.)
mod simulation_stack {
    use super::*;

    struct Cubic {
        u: Option<FieldHandle<f64>>,
    }

    macro_rules! impl_cubic {
        ($grid:ty) => {
            impl<D: Sync> Model<$grid, D> for Cubic {
                type Scalar = f64;
                type Drivers = NoNoise;

                fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
                    self.u = Some(builder.register("u", 0));
                }

                fn vector_field_block<S: StorageBackend<f64>>(
                    &self,
                    _driver: Driver,
                    ctx: &RhsContext<'_, $grid, D>,
                    state: &State<f64, S>,
                    out: &mut BlockStateMut<'_, f64, S>,
                    _scratch: &mut Scratch<f64, S>,
                ) {
                    let u = self.u.unwrap();
                    let v = state.view(ctx.grid, ctx.block, u);
                    let mut dv = out.view_mut(ctx.grid, ctx.block, u);
                    dv.set([0], -v.get([0]).powi(3));
                }
            }
        };
    }
    impl_cubic!(CartesianGrid<1>);
    impl_cubic!(AmrGrid<1>);

    fn run<G>(grid: G, u0: f64, steps: usize) -> f64
    where
        G: Grid<Index = [isize; 1]>,
        Cubic: Model<G, FiniteDifference, Scalar = f64, Drivers = NoNoise>,
    {
        let mut sim = Simulation::new(
            grid,
            FiniteDifference,
            Cubic { u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let u = sim.model().u.unwrap();
        {
            let (_, state) = sim.state_mut();
            state.slab_mut(BlockId(0), u)[0] = u0;
        }
        for _ in 0..steps {
            sim.step(0.01);
        }
        sim.state().slab(BlockId(0), u)[0]
    }

    #[test]
    fn simulation_on_amr_matches_cartesian_bitwise() {
        let cart = CartesianGrid::new([1], [1], [0.0], [1.0]).unwrap();
        let base = CartesianGrid::new([1], [1], [0.0], [1.0]).unwrap();
        let amr = AmrGrid::from_patches(base, &[], &[]).unwrap();
        assert_eq!(run(cart, 1.0, 100), run(amr, 1.0, 100));
    }
}

/// Phase C: the intergrid operations.
mod intergrid {
    use super::*;
    use legendre::geometry::{
        amr::{fill_ghosts_mirror as amr_fill, restrict},
        cartesian::for_each_interior,
    };

    /// Write `f(cell center)` into every interior cell of every patch.
    fn fill_by_center(
        grid: &AmrGrid<2>,
        state: &mut State<f64, DenseStorage<f64>>,
        u: FieldHandle<f64>,
        f: impl Fn([f64; 2]) -> f64,
    ) {
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            let ext = grid.patch(block).extent();
            let mut v = state.view_mut(grid, block, u);
            for_each_interior(ext, |idx| v.set(idx, f(grid.cell_center(block, idx))));
        }
    }

    /// Exchange and prolongation reproduce a global linear field exactly
    /// in every ghost cell (bilinear interpolation and interior copies
    /// are exact on linears), and restriction reproduces it on covered
    /// coarse cells (the mean of linear children is the center value).
    #[test]
    fn linear_field_ghosts_and_restriction_are_exact() {
        let lin = |[x, y]: [f64; 2]| 3.0f64.mul_add(x, 2.0) - y;
        let base = CartesianGrid::new([8, 8], [4, 4], [0.0, 0.0], [0.5, 0.5]).unwrap();
        // Two abutting fine patches away from the physical boundary:
        // exchange across their shared face, prolongation elsewhere.
        let grid = AmrGrid::from_patches(
            base,
            &[2],
            &[vec![
                CellBox {
                    lo: [4, 4],
                    hi: [8, 12],
                },
                CellBox {
                    lo: [8, 4],
                    hi: [12, 12],
                },
            ]],
        )
        .unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 1);
        let mut state: State<f64, DenseStorage<f64>> = builder.build(&grid, &SystemAllocator);
        fill_by_center(&grid, &mut state, u, lin);

        // Corrupt the coarse cells beneath the patches: restriction must
        // rebuild them from the fine data.
        {
            let mut v = state.view_mut(&grid, BlockId(0), u);
            for_each_interior([4, 4], |idx| {
                if idx[0] >= 2 && idx[1] >= 2 {
                    v.set(idx, 99.0);
                }
            });
        }
        restrict(&grid, &mut state, u);
        amr_fill(&grid, &mut state, u);

        // Every ghost cell of both fine patches carries the exact linear
        // value at its center.
        for fb in grid.blocks_at(1) {
            let ext = grid.patch(fb).extent();
            let v = state.view(&grid, fb, u);
            for i in -1..=(ext[0] as isize) {
                for j in -1..=(ext[1] as isize) {
                    let interior = i >= 0 && j >= 0 && i < ext[0] as isize && j < ext[1] as isize;
                    if interior {
                        continue;
                    }
                    let c = grid.cell_center(fb, [i, j]);
                    approx::assert_relative_eq!(v.get([i, j]), lin(c), epsilon = 1e-12);
                }
            }
        }
        // The corrupted coarse cells were rebuilt exactly.
        let v = state.view(&grid, BlockId(0), u);
        for_each_interior([4, 4], |idx| {
            let c = grid.cell_center(BlockId(0), idx);
            approx::assert_relative_eq!(v.get(idx), lin(c), epsilon = 1e-12);
        });
    }

    pub struct Heat {
        pub kappa: f64,
        pub u: Option<FieldHandle<f64>>,
    }

    impl<D> Model<AmrGrid<2>, D> for Heat
    where
        D: legendre::discretization::operators::Discretizes<
                AmrGrid<2>,
                legendre::discretization::operators::Laplacian,
            >,
    {
        type Scalar = f64;
        type Drivers = NoNoise;

        fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
            self.u = Some(builder.register("u", 1));
        }

        fn fill_ghosts<S: StorageBackend<f64>>(
            &self,
            grid: &AmrGrid<2>,
            state: &mut State<f64, S>,
            _t: f64,
        ) {
            restrict(grid, state, self.u.unwrap());
            amr_fill(grid, state, self.u.unwrap());
        }

        fn vector_field_block<S: StorageBackend<f64>>(
            &self,
            _driver: Driver,
            ctx: &RhsContext<'_, AmrGrid<2>, D>,
            state: &State<f64, S>,
            out: &mut BlockStateMut<'_, f64, S>,
            _scratch: &mut Scratch<f64, S>,
        ) {
            use legendre::discretization::{operators::Laplacian, stencil::Stencil};
            let u = self.u.unwrap();
            let stencil = ctx.disc.build(ctx.grid, Laplacian);
            let input = state.view(ctx.grid, ctx.block, u);
            let mut output = out.view_mut(ctx.grid, ctx.block, u);
            stencil.apply(ctx.grid, ctx.block, input, &mut output);
            let ext = ctx.grid.patch(ctx.block).extent();
            for_each_interior(ext, |idx| {
                output.set(idx, output.get(idx) * self.kappa);
            });
        }

        fn stable_dt(&self, h: [f64; 2]) -> Option<f64> {
            Some(0.25 * h[0].min(h[1]).powi(2) / self.kappa)
        }
    }

    /// Fine patches tiling the whole domain: the fine level *is* a
    /// uniform grid, so its evolution must match the uniform fine run
    /// bitwise (fine ghosts come only from exchange and physical mirror,
    /// never prolongation).
    #[test]
    fn full_fine_tiling_matches_uniform_grid_bitwise() {
        let kappa = 0.7;
        let hf = 0.25;
        let dt = 0.2 * hf * hf / kappa;
        let init = |[x, y]: [f64; 2]| (0.7 * x).cos() + (1.3 * y).sin();

        // AMR: 8×8 base at h=0.5, fully tiled by two 8×16 fine patches.
        let base = CartesianGrid::new([8, 8], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let grid = AmrGrid::from_patches(
            base,
            &[2],
            &[vec![
                CellBox {
                    lo: [0, 0],
                    hi: [8, 16],
                },
                CellBox {
                    lo: [8, 0],
                    hi: [16, 16],
                },
            ]],
        )
        .unwrap();
        let mut sim = Simulation::new(
            grid,
            FiniteDifference,
            Heat { kappa, u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let u = sim.model().u.unwrap();
        {
            let (grid, state) = sim.state_mut();
            for b in 0..grid.num_blocks() {
                let block = BlockId(b as u32);
                let ext = grid.patch(block).extent();
                let mut v = state.view_mut(grid, block, u);
                for_each_interior(ext, |idx| v.set(idx, init(grid.cell_center(block, idx))));
            }
        }
        for _ in 0..100 {
            sim.step(dt);
        }
        // Gather the fine level by global cell index.
        let mut amr_field = vec![[0.0f64; 16]; 16];
        for fb in sim.grid().blocks_at(1) {
            let p = *sim.grid().patch(fb);
            let v = sim.state().view(sim.grid(), fb, u);
            for_each_interior(p.extent(), |idx| {
                let gx = (p.bx.lo[0] + idx[0] as i64) as usize;
                let gy = (p.bx.lo[1] + idx[1] as i64) as usize;
                amr_field[gy][gx] = v.get(idx);
            });
        }

        // Uniform twin: 16×16 at h=0.25, same decomposition.
        let uni = CartesianGrid::new([16, 16], [8, 16], [0.0, 0.0], [hf, hf]).unwrap();
        let mut usim = Simulation::new(
            uni,
            FiniteDifference,
            UniformHeat { kappa, u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let uu = usim.model().u.unwrap();
        {
            let (grid, state) = usim.state_mut();
            legendre::geometry::cartesian::fill_from_fn(grid, state, uu, init);
        }
        for _ in 0..100 {
            usim.step(dt);
        }
        for b in 0..usim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let v = usim.state().view(usim.grid(), block, uu);
            for_each_interior(usim.grid().block_cells(), |idx| {
                let [x, y] = usim.grid().cell_center(block, idx);
                let gx = (x / hf - 0.5).round() as usize;
                let gy = (y / hf - 0.5).round() as usize;
                assert_eq!(
                    amr_field[gy][gx],
                    v.get(idx),
                    "cell ({gx},{gy}) diverged from the uniform run"
                );
            });
        }
    }

    pub struct UniformHeat {
        pub kappa: f64,
        pub u: Option<FieldHandle<f64>>,
    }

    impl<D> Model<CartesianGrid<2>, D> for UniformHeat
    where
        D: legendre::discretization::operators::Discretizes<
                CartesianGrid<2>,
                legendre::discretization::operators::Laplacian,
            >,
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
            legendre::geometry::cartesian::fill_ghosts_mirror(grid, state, self.u.unwrap());
        }

        fn vector_field_block<S: StorageBackend<f64>>(
            &self,
            _driver: Driver,
            ctx: &RhsContext<'_, CartesianGrid<2>, D>,
            state: &State<f64, S>,
            out: &mut BlockStateMut<'_, f64, S>,
            _scratch: &mut Scratch<f64, S>,
        ) {
            use legendre::discretization::{operators::Laplacian, stencil::Stencil};
            let u = self.u.unwrap();
            let stencil = ctx.disc.build(ctx.grid, Laplacian);
            let input = state.view(ctx.grid, ctx.block, u);
            let mut output = out.view_mut(ctx.grid, ctx.block, u);
            stencil.apply(ctx.grid, ctx.block, input, &mut output);
            for_each_interior(input.interior(), |idx| {
                output.set(idx, output.get(idx) * self.kappa);
            });
        }
    }

    /// A genuinely refined run — one fine patch over a Gaussian bump —
    /// stays physical: finite everywhere, maximum principle respected,
    /// heat decays.
    #[test]
    fn interface_run_stays_physical() {
        let kappa = 0.7;
        let hf = 0.25;
        let dt = 0.2 * hf * hf / kappa;
        let base = CartesianGrid::new([16, 16], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let grid = AmrGrid::from_patches(
            base,
            &[2],
            &[vec![CellBox {
                lo: [8, 8],
                hi: [24, 24],
            }]],
        )
        .unwrap();
        let init = |[x, y]: [f64; 2]| {
            let (dx, dy) = (x - 4.0, y - 4.0);
            (-dx.mul_add(dx, dy * dy) / 1.5).exp()
        };
        let mut sim = Simulation::new(
            grid,
            FiniteDifference,
            Heat { kappa, u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let u = sim.model().u.unwrap();
        {
            let (grid, state) = sim.state_mut();
            for b in 0..grid.num_blocks() {
                let block = BlockId(b as u32);
                let ext = grid.patch(block).extent();
                let mut v = state.view_mut(grid, block, u);
                for_each_interior(ext, |idx| v.set(idx, init(grid.cell_center(block, idx))));
            }
        }
        for _ in 0..200 {
            sim.step(dt);
        }
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for b in 0..sim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let ext = sim.grid().patch(block).extent();
            let v = sim.state().view(sim.grid(), block, u);
            for_each_interior(ext, |idx| {
                let x = v.get(idx);
                assert!(x.is_finite(), "non-finite value at {block:?} {idx:?}");
                lo = lo.min(x);
                hi = hi.max(x);
            });
        }
        assert!(
            hi < 0.9,
            "heat must decay from the initial peak of 1: max {hi}"
        );
        assert!(hi > 0.0 && lo > -1e-9, "maximum principle: [{lo}, {hi}]");
    }
}

/// Phase D: the Berger–Oliger adaptivity policy on the plain `Simulation`.
mod adaptive {
    use super::intergrid::Heat;
    use super::*;
    use legendre::geometry::{
        amr::{BergerOliger, ClusterParams, GradientTagger, RegridPolicy, TagCells, restrict},
        cartesian::for_each_interior,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    fn gaussian([x, y]: [f64; 2]) -> f64 {
        let (dx, dy) = (x - 4.0, y - 4.0);
        (-dx.mul_add(dx, dy * dy) / 0.4).exp()
    }

    fn fill_amr(
        grid: &AmrGrid<2>,
        state: &mut State<f64, DenseStorage<f64>>,
        u: FieldHandle<f64>,
        f: impl Fn([f64; 2]) -> f64,
    ) {
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            let ext = grid.patch(block).extent();
            let mut v = state.view_mut(grid, block, u);
            for_each_interior(ext, |idx| v.set(idx, f(grid.cell_center(block, idx))));
        }
    }

    const fn policy() -> BergerOliger<GradientTagger> {
        BergerOliger::new(
            GradientTagger {
                field: "u",
                threshold: 0.05,
            },
            RegridPolicy {
                every: 4,
                buffer: 2,
                cluster: ClusterParams {
                    efficiency: 0.7,
                    min_width: 2,
                },
            },
        )
    }

    /// The very first step regrids from the initial conditions: every
    /// high-gradient base cell ends up interior to a level-1 patch, and
    /// the run stays refined and physical through later regrids.
    #[test]
    fn initial_regrid_refines_around_the_feature() {
        let base = CartesianGrid::new([16, 16], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
        let mut sim = Simulation::adaptive(
            grid,
            FiniteDifference,
            Heat {
                kappa: 0.7,
                u: None,
            },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
            policy(),
        );
        let u = sim.model().u.unwrap();
        {
            let (grid, state) = sim.state_mut();
            fill_amr(grid, state, u, gaussian);
        }
        let dt = 0.2 * 0.25 * 0.25 / 0.7;
        sim.step(dt); // step 0: initial refinement from the ICs

        assert_eq!(
            sim.grid().num_levels(),
            2,
            "the bump must trigger refinement"
        );
        // Every base cell whose IC gradient beats the threshold sits under
        // a fine patch.
        let g = sim.grid();
        for cell_y in 0..16i64 {
            for cell_x in 0..16i64 {
                let center = [
                    0.5f64.mul_add(cell_x as f64, 0.25),
                    0.5f64.mul_add(cell_y as f64, 0.25),
                ];
                let gx = (gaussian([center[0] + 0.5, center[1]])
                    - gaussian([center[0] - 0.5, center[1]]))
                    / 2.0;
                let gy = (gaussian([center[0], center[1] + 0.5])
                    - gaussian([center[0], center[1] - 0.5]))
                    / 2.0;
                if gx.abs().max(gy.abs()) > 0.06 {
                    let fine = [cell_x * 2, cell_y * 2];
                    assert!(
                        g.find_patch(1, fine).is_some(),
                        "high-gradient cell ({cell_x},{cell_y}) not refined"
                    );
                }
            }
        }

        // A few more steps: the bump is still sharp, refinement persists.
        // (Run long enough and the policy correctly *de-refines* the
        // flattened Gaussian — that behavior is pinned in the comparison
        // test below.)
        for _ in 0..8 {
            sim.step(dt);
        }
        assert_eq!(
            sim.grid().num_levels(),
            2,
            "feature still present, still refined"
        );
        for b in 0..sim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let ext = sim.grid().patch(block).extent();
            let v = sim.state().view(sim.grid(), block, u);
            for_each_interior(ext, |idx| {
                let x = v.get(idx);
                assert!(
                    x.is_finite() && (-1e-9..=1.0).contains(&x),
                    "unphysical {x}"
                );
            });
        }
    }

    /// The whole point of AMR, as one assertion: against a restricted
    /// uniform-fine reference, the adaptive run beats the uniform-coarse
    /// run — at a fraction of the fine run's cell count.
    #[test]
    fn adaptive_beats_coarse_against_fine_reference() {
        let kappa = 0.7;
        let dt = 0.2 * 0.25 * 0.25 / kappa;
        let steps = 60;

        // Uniform fine 32² reference, restricted to 16².
        let fine = CartesianGrid::new([32, 32], [16, 16], [0.0, 0.0], [0.25, 0.25]).unwrap();
        let mut fsim = Simulation::new(
            fine,
            FiniteDifference,
            super::intergrid::UniformHeat { kappa, u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let fu = fsim.model().u.unwrap();
        {
            let (grid, state) = fsim.state_mut();
            legendre::geometry::cartesian::fill_from_fn(grid, state, fu, gaussian);
        }
        for _ in 0..steps {
            fsim.step(dt);
        }
        let mut reference = vec![[0.0f64; 16]; 16];
        for b in 0..fsim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let v = fsim.state().view(fsim.grid(), block, fu);
            for_each_interior(fsim.grid().block_cells(), |idx| {
                let [x, y] = fsim.grid().cell_center(block, idx);
                let (cx, cy) = ((x / 0.5) as usize, (y / 0.5) as usize);
                reference[cy][cx] = v.get(idx).mul_add(0.25, reference[cy][cx]);
            });
        }

        // Uniform coarse 16².
        let coarse = CartesianGrid::new([16, 16], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let mut csim = Simulation::new(
            coarse,
            FiniteDifference,
            super::intergrid::UniformHeat { kappa, u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let cu = csim.model().u.unwrap();
        {
            let (grid, state) = csim.state_mut();
            legendre::geometry::cartesian::fill_from_fn(grid, state, cu, gaussian);
        }
        for _ in 0..steps {
            csim.step(dt);
        }

        // Adaptive: coarse base, one refinement level, |∇u| tagging.
        let base = CartesianGrid::new([16, 16], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
        let mut asim = Simulation::adaptive(
            grid,
            FiniteDifference,
            Heat { kappa, u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
            policy(),
        );
        let au = asim.model().u.unwrap();
        {
            let (grid, state) = asim.state_mut();
            fill_amr(grid, state, au, gaussian);
        }
        let mut refined_cells = 0usize;
        for _ in 0..steps {
            asim.step(dt);
            refined_cells = refined_cells.max(
                asim.grid()
                    .blocks_at(1)
                    .map(|b| asim.grid().patch(b).bx.cells() as usize)
                    .sum(),
            );
        }
        // Compose the adaptive solution onto the coarse cells.
        {
            let (grid, state) = asim.state_mut();
            restrict(grid, state, au);
        }
        let (mut err_adaptive, mut err_coarse) = (0.0f64, 0.0f64);
        for b in asim.grid().blocks_at(0) {
            let p = *asim.grid().patch(b);
            let va = asim.state().view(asim.grid(), b, au);
            let vc = csim.state().view(csim.grid(), b, cu);
            for_each_interior(p.extent(), |idx| {
                let gx = (p.bx.lo[0] + idx[0] as i64) as usize;
                let gy = (p.bx.lo[1] + idx[1] as i64) as usize;
                err_adaptive = err_adaptive.max((va.get(idx) - reference[gy][gx]).abs());
                err_coarse = err_coarse.max((vc.get(idx) - reference[gy][gx]).abs());
            });
        }
        approx::assert_relative_eq!(err_adaptive, 0.001_680_900f64, epsilon = 1e-6);
        approx::assert_relative_eq!(err_coarse, 0.003_716_000f64, epsilon = 1e-6);

        // We heavily refine from the default otherwise of 32x32=1024
        assert_eq!(
            refined_cells, 400,
            "refinement must stay partial: {refined_cells} fine cells"
        );
    }

    /// Migration is exact on linear fields, through both paths: copy from
    /// the old same-level patch, and prolongation of newly refined
    /// regions — pinned by moving a forced refinement box mid-run under a
    /// do-nothing model.
    struct BoxTagger {
        calls: AtomicU64,
    }

    impl TagCells<f64, 2> for BoxTagger {
        fn tag_level<S: StorageBackend<f64>>(
            &self,
            _grid: &AmrGrid<2>,
            _state: &State<f64, S>,
            level: u8,
            flags: &mut Vec<[i64; 2]>,
        ) {
            if level != 0 {
                return;
            }
            // First regrid: box [2,6)²; later: shifted box [4,8)².
            let shift = i64::from(self.calls.fetch_add(1, Ordering::Relaxed) > 0) * 2;
            for y in 2 + shift..6 + shift {
                for x in 2 + shift..6 + shift {
                    flags.push([x, y]);
                }
            }
        }
    }

    struct Frozen {
        u: Option<FieldHandle<f64>>,
    }

    impl<D: Sync> Model<AmrGrid<2>, D> for Frozen {
        type Scalar = f64;
        type Drivers = NoNoise;

        fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
            self.u = Some(builder.register("u", 1));
        }

        fn vector_field_block<S: StorageBackend<f64>>(
            &self,
            _driver: Driver,
            ctx: &RhsContext<'_, AmrGrid<2>, D>,
            _state: &State<f64, S>,
            out: &mut BlockStateMut<'_, f64, S>,
            _scratch: &mut Scratch<f64, S>,
        ) {
            let u = self.u.unwrap();
            let ext = ctx.grid.patch(ctx.block).extent();
            let mut dv = out.view_mut(ctx.grid, ctx.block, u);
            for_each_interior(ext, |idx| dv.set(idx, 0.0));
        }
    }

    #[test]
    fn migration_preserves_linear_fields_exactly() {
        let lin = |[x, y]: [f64; 2]| 3.0f64.mul_add(x, 2.0) - y;
        let base = CartesianGrid::new([16, 16], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
        let mut sim = Simulation::adaptive(
            grid,
            FiniteDifference,
            Frozen { u: None },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
            BergerOliger::new(
                BoxTagger {
                    calls: AtomicU64::new(0),
                },
                RegridPolicy {
                    every: 1,
                    buffer: 0,
                    cluster: ClusterParams {
                        efficiency: 0.9,
                        min_width: 2,
                    },
                },
            ),
        );
        let u = sim.model().u.unwrap();
        {
            let (grid, state) = sim.state_mut();
            fill_amr(grid, state, u, lin);
        }
        let dt = 1e-3;
        sim.step(dt); // regrid to [2,6)² (prolongation path), step frozen
        sim.step(dt); // regrid to [4,8)² (copy overlap + prolong the rest)

        assert_eq!(sim.grid().num_levels(), 2);
        let fine: Vec<BlockId> = sim.grid().blocks_at(1).collect();
        assert!(!fine.is_empty());
        for fb in fine {
            let p = *sim.grid().patch(fb);
            assert_eq!(
                p.bx,
                CellBox {
                    lo: [8, 8],
                    hi: [16, 16]
                },
                "shifted box expected"
            );
            let v = sim.state().view(sim.grid(), fb, u);
            for_each_interior(p.extent(), |idx| {
                let c = sim.grid().cell_center(fb, idx);
                approx::assert_relative_eq!(v.get(idx), lin(c), epsilon = 1e-12);
            });
        }
    }
}

/// The flagship physics under adaptivity: Model C's interface is tracked
/// by |∇φ| tagging, stays physical, and keeps the refinement partial.
mod model_c_adaptive {
    use super::*;
    use legendre::{
        discretization::finite_volume::FiniteVolume,
        geometry::{
            amr::{BergerOliger, ClusterParams, GradientTagger, RegridPolicy},
            cartesian::for_each_interior,
        },
        integrators::EulerMaruyama,
        physics::phasefield::ModelC,
    };

    #[test]
    fn dendrite_interface_is_tracked_by_refinement() {
        let base = CartesianGrid::new([48, 48], [24, 24], [0.0, 0.0], [0.4, 0.4]).unwrap();
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
        let phi_h = sim.model().phi();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize(grid.base(), state, [0.4, 0.4], 4.0, 0.7);
        }
        for _ in 0..150 {
            sim.step(dt);
        }

        let g = sim.grid();
        assert_eq!(g.num_levels(), 2, "the interface must be refined");
        let refined: u64 = g.blocks_at(1).map(|b| g.patch(b).bx.cells()).sum();
        assert!(
            refined > 0 && refined < 96 * 96,
            "refinement must be partial: {refined} fine cells"
        );

        // Every level-0 interface cell (per the tagging criterion) sits
        // under a fine patch, and the solution stays physical everywhere.
        for b in 0..g.num_blocks() {
            let block = BlockId(b as u32);
            let level = g.level(block);
            let p = *g.patch(block);
            let v = sim.state().view(g, block, phi_h);
            for_each_interior(p.extent(), |idx| {
                let x = v.get(idx);
                assert!(x.is_finite() && x.abs() < 1.2, "unphysical phi {x}");
                if level == 0 {
                    let gx = 0.5 * (v.get([idx[0] + 1, idx[1]]) - v.get([idx[0] - 1, idx[1]]));
                    let gy = 0.5 * (v.get([idx[0], idx[1] + 1]) - v.get([idx[0], idx[1] - 1]));
                    if gx.abs().max(gy.abs()) > 0.2 {
                        let fine = [
                            (p.bx.lo[0] + idx[0] as i64) * 2,
                            (p.bx.lo[1] + idx[1] as i64) * 2,
                        ];
                        assert!(
                            g.find_patch(1, fine).is_some(),
                            "interface cell {fine:?} not refined"
                        );
                    }
                }
            });
        }
    }
}

/// Subcycling: coarse levels take large steps, fine levels subcycle.
mod subcycling {
    use super::intergrid::{Heat, UniformHeat};
    use super::*;
    use legendre::{
        core::scheduler::{RayonScheduler, Scheduler},
        geometry::{
            amr::{BergerOliger, ClusterParams, GradientTagger, RegridPolicy},
            cartesian::{fill_from_fn, for_each_interior},
        },
        integrators::Subcycling,
    };

    const KAPPA: f64 = 0.7;

    fn init([x, y]: [f64; 2]) -> f64 {
        (0.6 * x).cos() + (1.1 * y).sin()
    }

    /// Fully fine-tiled subcycled run: the fine level's ghosts are all
    /// same-level exchange or physical mirror (never prolongation), so
    /// subcycling steps it independently `n` times per coarse step — it
    /// must match a uniform-fine grid stepped at the substep dt, bit for
    /// bit.
    #[test]
    fn fully_tiled_subcycled_fine_matches_uniform_fine_bitwise() {
        const COARSE_STEPS: usize = 25;
        // AMR: 8² base at h = 0.5, fully tiled by two level-1 patches
        // (h = 0.25, ratio 2 ⇒ 4 substeps for the parabolic heat model).
        let base = CartesianGrid::new([8, 8], [8, 8], [0.0, 0.0], [0.5, 0.5]).unwrap();
        let grid = AmrGrid::from_patches(
            base,
            &[2],
            &[vec![
                CellBox {
                    lo: [0, 0],
                    hi: [8, 16],
                },
                CellBox {
                    lo: [8, 0],
                    hi: [16, 16],
                },
            ]],
        )
        .unwrap();
        let mut sim = Simulation::new(
            grid,
            FiniteDifference,
            Heat {
                kappa: KAPPA,
                u: None,
            },
            Subcycling { seed: 0 },
            SerialScheduler,
            SystemAllocator,
        );
        let u = sim.model().u.unwrap();
        let dt_coarse = sim.stable_dt().unwrap(); // level-0 (coarse) dt
        {
            let (grid, state) = sim.state_mut();
            for b in 0..grid.num_blocks() {
                let block = BlockId(b as u32);
                let ext = grid.patch(block).extent();
                let mut v = state.view_mut(grid, block, u);
                for_each_interior(ext, |idx| v.set(idx, init(grid.cell_center(block, idx))));
            }
        }
        for _ in 0..COARSE_STEPS {
            sim.step(dt_coarse);
        }
        let mut amr_field = vec![[0.0f64; 16]; 16];
        for fb in sim.grid().blocks_at(1) {
            let p = *sim.grid().patch(fb);
            let v = sim.state().view(sim.grid(), fb, u);
            for_each_interior(p.extent(), |idx| {
                let gx = (p.bx.lo[0] + idx[0] as i64) as usize;
                let gy = (p.bx.lo[1] + idx[1] as i64) as usize;
                amr_field[gy][gx] = v.get(idx);
            });
        }

        // Uniform 16² at h = 0.25, stepped at *exactly* dt_coarse / 4 (the
        // same expression the recursion uses) for 4× as many steps.
        let hf = 0.25;
        let dt_fine = dt_coarse / 4.0;
        let uni = CartesianGrid::new([16, 16], [8, 16], [0.0, 0.0], [hf, hf]).unwrap();
        let mut usim = Simulation::new(
            uni,
            FiniteDifference,
            UniformHeat {
                kappa: KAPPA,
                u: None,
            },
            ForwardEuler,
            SerialScheduler,
            SystemAllocator,
        );
        let uu = usim.model().u.unwrap();
        {
            let (grid, state) = usim.state_mut();
            fill_from_fn(grid, state, uu, init);
        }
        for _ in 0..COARSE_STEPS * 4 {
            usim.step(dt_fine);
        }
        for b in 0..usim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let v = usim.state().view(usim.grid(), block, uu);
            for_each_interior(usim.grid().block_cells(), |idx| {
                let [x, y] = usim.grid().cell_center(block, idx);
                let gx = (x / hf - 0.5).round() as usize;
                let gy = (y / hf - 0.5).round() as usize;
                assert_eq!(
                    amr_field[gy][gx],
                    v.get(idx),
                    "subcycled fine cell ({gx},{gy}) diverged from the uniform-fine run"
                );
            });
        }
    }

    /// A genuinely refined, adaptive, subcycled run (prolongation + time
    /// interpolation exercised) must be bit-identical under serial and
    /// parallel scheduling.
    #[test]
    fn adaptive_subcycled_run_is_scheduler_independent() {
        fn run<Sch: Scheduler>(scheduler: Sch) -> Vec<f64> {
            let base = CartesianGrid::new([32, 32], [16, 16], [0.0, 0.0], [0.5, 0.5]).unwrap();
            let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
            let mut sim = Simulation::adaptive(
                grid,
                FiniteDifference,
                Heat {
                    kappa: KAPPA,
                    u: None,
                },
                Subcycling { seed: 0 },
                scheduler,
                SystemAllocator,
                BergerOliger::new(
                    GradientTagger {
                        field: "u",
                        threshold: 0.03,
                    },
                    RegridPolicy {
                        every: 2,
                        buffer: 2,
                        cluster: ClusterParams {
                            efficiency: 0.7,
                            min_width: 4,
                        },
                    },
                ),
            );
            let u = sim.model().u.unwrap();
            let dt = sim.stable_dt().unwrap();
            {
                let (grid, state) = sim.state_mut();
                for b in 0..grid.num_blocks() {
                    let block = BlockId(b as u32);
                    let ext = grid.patch(block).extent();
                    let mut v = state.view_mut(grid, block, u);
                    for_each_interior(ext, |idx| {
                        let [x, y] = grid.cell_center(block, idx);
                        let (dx, dy) = (x - 8.0, y - 8.0);
                        v.set(idx, (-dx.mul_add(dx, dy * dy) / 3.0).exp());
                    });
                }
            }
            for _ in 0..30 {
                sim.step(dt);
            }
            // The coarse (level-0) solution, restricted-consistent.
            let mut out = Vec::new();
            for b in sim.grid().blocks_at(0) {
                let v = sim.state().view(sim.grid(), b, u);
                let ext = sim.grid().patch(b).extent();
                for_each_interior(ext, |idx| out.push(v.get(idx)));
            }
            out
        }
        assert_eq!(
            run(SerialScheduler),
            run(RayonScheduler),
            "subcycled run must not depend on scheduling"
        );
    }

    /// Model C dendrite, adaptive + subcycled, stays physical — the fine
    /// interface takes r² = 4 substeps per coarse step.
    #[test]
    fn model_c_subcycled_dendrite_is_physical() {
        use legendre::{discretization::finite_volume::FiniteVolume, physics::phasefield::ModelC};
        let base = CartesianGrid::new([48, 48], [24, 24], [0.0, 0.0], [0.4, 0.4]).unwrap();
        let grid = AmrGrid::from_patches(base, &[2], &[]).unwrap();
        let mut sim = Simulation::adaptive(
            grid,
            FiniteVolume::default(),
            ModelC::<NoNoise>::classic(),
            Subcycling { seed: 0 },
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
        let phi_h = sim.model().phi();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize(grid.base(), state, [0.4, 0.4], 4.0, 0.7);
        }
        for _ in 0..40 {
            sim.step(dt);
        }
        assert_eq!(sim.grid().num_levels(), 2, "interface must be refined");
        for b in 0..sim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let ext = sim.grid().patch(block).extent();
            let v = sim.state().view(sim.grid(), block, phi_h);
            for_each_interior(ext, |idx| {
                let p = v.get(idx);
                assert!(p.is_finite() && p.abs() < 1.2, "unphysical phi {p}");
            });
        }
    }
}
