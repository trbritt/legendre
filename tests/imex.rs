//! Validation of the IMEX (implicit–explicit) Euler scheme on the
//! market-making HJB: the implicit ν advection–diffusion solve must break the
//! explicit `dt ∝ dν²` step wall while still producing the same optimal
//! policy, and must degenerate to forward Euler on a model with no stiff part.

#![allow(clippy::suboptimal_flops, clippy::float_cmp)]

use legendre::{
    core::{
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
        scratch::Scratch,
        simulation::Simulation,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::{StorageBackend, SystemAllocator},
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::{ForwardEuler, ImexEuler},
    physics::{
        market_making::{HjbMarketMaker, MarketMakerParams},
        model::{Driver, Model, NoNoise, RhsContext},
    },
};

const NU_MIN: f64 = 0.001;
const NU_MAX: f64 = 2.0;

fn nu_grid(cells: usize, block: usize) -> CartesianGrid<1> {
    let dnu = (NU_MAX - NU_MIN) / cells as f64;
    CartesianGrid::new([cells], [block], [NU_MIN], [dnu]).unwrap()
}

/// Solve the HJB to `horizon` under the given integrator (using the
/// integrator's own suggested dt), returning the value surface as
/// `u[q_index][ν_cell]` and the number of steps taken.
fn solve_imex<Sch: Scheduler>(
    params: MarketMakerParams,
    grid: CartesianGrid<1>,
    scheduler: Sch,
    horizon: f64,
) -> (Vec<Vec<f64>>, usize) {
    let mut sim = Simulation::new(
        grid,
        (),
        HjbMarketMaker::new(params),
        ImexEuler,
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    let steps = (horizon / dt).ceil() as usize;
    for _ in 0..steps {
        sim.step(dt);
    }
    (surface(&sim), steps)
}

fn solve_explicit(
    params: MarketMakerParams,
    grid: CartesianGrid<1>,
    horizon: f64,
) -> (Vec<Vec<f64>>, usize) {
    let mut sim = Simulation::new(
        grid,
        (),
        HjbMarketMaker::new(params),
        ForwardEuler,
        SerialScheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();
    let steps = (horizon / dt).ceil() as usize;
    for _ in 0..steps {
        sim.step(dt);
    }
    (surface(&sim), steps)
}

fn surface<I>(
    sim: &Simulation<CartesianGrid<1>, (), HjbMarketMaker, I, impl Scheduler, SystemAllocator>,
) -> Vec<Vec<f64>>
where
    I: legendre::integrators::Integrator<CartesianGrid<1>, (), NoNoise>,
{
    let handles = sim.model().handles().to_vec();
    handles
        .iter()
        .map(|&h| {
            let mut col = Vec::new();
            for b in 0..sim.grid().num_blocks() {
                let v = sim.state().view(sim.grid(), BlockId(b as u32), h);
                for_each_interior(sim.grid().block_cells(), |idx| col.push(v.get(idx)));
            }
            col
        })
        .collect()
}

#[test]
fn imex_takes_far_fewer_steps_than_explicit() {
    // The explicit scheme is CFL-bound at dt ∝ dν²; IMEX is bound only by the
    // nonstiff fill reaction, independent of dν. On a fine grid that is a
    // large step-count reduction — the whole point for policy-refresh latency.
    let params = MarketMakerParams::default();
    let (_, imex_steps) = solve_imex(params, nu_grid(100, 25), SerialScheduler, 3.0);
    let (_, exp_steps) = solve_explicit(params, nu_grid(100, 25), 3.0);
    assert!(
        exp_steps >= 10 * imex_steps,
        "IMEX ({imex_steps} steps) should need ≥10× fewer than explicit ({exp_steps})"
    );
}

#[test]
fn imex_recovers_the_explicit_optimal_policy() {
    // Both schemes are first-order in dt but solve the same PDE, so their
    // value surfaces — and thus the optimal quotes read off them — must agree
    // to O(dt). Compare the optimal depths (the quantity that actually drives
    // trading) at a mid-variance node across all inventory levels.
    let params = MarketMakerParams::default();
    let grid = nu_grid(100, 25);
    let (u_imex, _) = solve_imex(params, grid.clone(), SerialScheduler, 3.0);
    let (u_exp, _) = solve_explicit(params, grid, 3.0);

    let model = HjbMarketMaker::new(params);
    let cell = u_imex[0].len() / 2;
    let nq = params.num_levels();
    for qi in 0..nq {
        let q = params.q_min + qi as isize;
        let up = |u: &[Vec<f64>]| (q < params.q_max).then(|| u[qi + 1][cell]);
        let dn = |u: &[Vec<f64>]| (q > params.q_min).then(|| u[qi - 1][cell]);
        let (bi, ai) = model.optimal_depths(q, u_imex[qi][cell], up(&u_imex), dn(&u_imex));
        let (be, ae) = model.optimal_depths(q, u_exp[qi][cell], up(&u_exp), dn(&u_exp));
        // Withdrawn (∞) sides must agree on being withdrawn; finite depths
        // agree to a few percent (first-order-in-dt discretization gap).
        assert_eq!(
            bi.is_finite(),
            be.is_finite(),
            "bid withdrawal disagrees at q={q}"
        );
        assert_eq!(
            ai.is_finite(),
            ae.is_finite(),
            "ask withdrawal disagrees at q={q}"
        );
        if bi.is_finite() {
            assert!(
                (bi - be).abs() < 0.02,
                "bid depth q={q}: imex {bi:.5} vs exp {be:.5}"
            );
        }
        if ai.is_finite() {
            assert!(
                (ai - ae).abs() < 0.02,
                "ask depth q={q}: imex {ai:.5} vs exp {ae:.5}"
            );
        }
    }
}

#[test]
fn imex_stays_bounded_on_a_fine_grid() {
    // A grid so fine the explicit dt would be minuscule; IMEX must still be
    // stable at its dν-independent step and produce a finite, sensible surface.
    let params = MarketMakerParams::default();
    let (u, steps) = solve_imex(params, nu_grid(400, 50), SerialScheduler, 3.0);
    assert!(
        steps < 100,
        "fine-grid IMEX should still take few steps, took {steps}"
    );
    for col in &u {
        for &v in col {
            assert!(
                v.is_finite() && v.abs() < 100.0,
                "IMEX surface value {v} out of range"
            );
        }
    }
}

#[test]
fn imex_solve_is_bitwise_scheduler_independent() {
    let params = MarketMakerParams::default();
    let (s, _) = solve_imex(params, nu_grid(100, 20), SerialScheduler, 2.0);
    let (p, _) = solve_imex(params, nu_grid(100, 20), RayonScheduler, 2.0);
    assert_eq!(s, p, "IMEX must be schedule-independent");
}

/// A stiff-free 1D decay model: `dv/dt = −rate·v`, no `stiff_rows`.
#[derive(Clone)]
struct Decay {
    rate: f64,
    v: Option<FieldHandle<f64>>,
}

impl<D: Sync> Model<CartesianGrid<1>, D> for Decay {
    type Scalar = f64;
    type Drivers = NoNoise;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.v = Some(builder.register("v", 0));
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        _driver: Driver,
        ctx: &RhsContext<'_, CartesianGrid<1>, D>,
        state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        let v = state.view(ctx.grid, ctx.block, self.v.unwrap());
        let mut dv = out.view_mut(ctx.grid, ctx.block, self.v.unwrap());
        for_each_interior(ctx.grid.block_cells(), |idx| {
            dv.set(idx, -self.rate * v.get(idx));
        });
    }
}

#[test]
fn imex_degenerates_to_forward_euler_without_stiff_rows() {
    // With no stiff part declared, IMEX is exactly forward Euler — bit for bit.
    let grid = CartesianGrid::new([16], [8], [0.0], [1.0]).unwrap();
    let run = |imex: bool| {
        let seed_model = Decay { rate: 0.5, v: None };
        macro_rules! go {
            ($integ:expr) => {{
                let mut sim = Simulation::new(
                    grid.clone(),
                    (),
                    seed_model.clone(),
                    $integ,
                    SerialScheduler,
                    SystemAllocator,
                );
                let v = sim.model().v.unwrap();
                let (g, state) = sim.state_mut();
                for b in 0..g.num_blocks() {
                    let mut view = state.view_mut(g, BlockId(b as u32), v);
                    for_each_interior(g.block_cells(), |idx| view.set(idx, 1.0));
                }
                for _ in 0..20 {
                    sim.step(0.1);
                }
                let mut out = Vec::new();
                for b in 0..sim.grid().num_blocks() {
                    let view = sim.state().view(sim.grid(), BlockId(b as u32), v);
                    for_each_interior(sim.grid().block_cells(), |idx| out.push(view.get(idx)));
                }
                out
            }};
        }
        if imex {
            go!(ImexEuler)
        } else {
            go!(ForwardEuler)
        }
    };
    assert_eq!(
        run(true),
        run(false),
        "IMEX with no stiff rows must equal forward Euler"
    );
}
