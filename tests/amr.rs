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
