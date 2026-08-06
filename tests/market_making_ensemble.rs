//! End-to-end validation of the controlled market-making system: solve the
//! HJB for the optimal quotes, bake the depth tables, then run the controlled
//! jump-diffusion ensemble through the generic Monte Carlo harness and check
//! the economics an optimal maker must exhibit — bounded inventory, profit
//! from the spread, no directional bias under a symmetric book.

#![allow(clippy::suboptimal_flops, clippy::float_cmp)]

use legendre::{
    core::{
        monte_carlo::{MonteCarlo, path_grid},
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
        simulation::Simulation,
        storage::SystemAllocator,
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::{EulerMaruyama, ForwardEuler},
    physics::{
        market_making::{
            DepthTables, HjbMarketMaker, MarketMakerParams, MarketMakingEnsemble, Side,
        },
        model::Model,
    },
};
use std::sync::Arc;

const NU_MIN: f64 = 0.001;
const NU_MAX: f64 = 2.0;

fn nu_grid(cells: usize, block: usize) -> CartesianGrid<1> {
    let dnu = (NU_MAX - NU_MIN) / cells as f64;
    CartesianGrid::new([cells], [block], [NU_MIN], [dnu]).unwrap()
}

/// Capture the whole value surface as `surface[q_index][ν_cell]`.
fn capture(
    grid: &CartesianGrid<1>,
    state: &legendre::core::state::State<f64, legendre::core::storage::DenseStorage<f64>>,
    handles: &[legendre::core::state::FieldHandle<f64>],
) -> Vec<Vec<f64>> {
    handles
        .iter()
        .map(|&h| {
            let mut col = Vec::new();
            for b in 0..grid.num_blocks() {
                let v = state.view(grid, BlockId(b as u32), h);
                for_each_interior(grid.block_cells(), |idx| col.push(v.get(idx)));
            }
            col
        })
        .collect()
}

/// Solve the HJB forward in τ, recording the value surface every step; return
/// the ν-grid, the τ levels, and the surface frames.
fn solve_hjb(
    params: MarketMakerParams,
    dt: f64,
    steps: usize,
) -> (CartesianGrid<1>, Vec<f64>, Vec<Vec<Vec<f64>>>) {
    let grid = nu_grid(80, 20);
    let mut sim = Simulation::new(
        grid.clone(),
        (),
        HjbMarketMaker::new(params),
        ForwardEuler,
        SerialScheduler,
        SystemAllocator,
    );
    let handles = sim.model().handles().to_vec();
    let mut taus = vec![0.0];
    let mut surfaces = vec![capture(&grid, sim.state(), &handles)];
    for i in 0..steps {
        sim.step(dt);
        taus.push((i + 1) as f64 * dt);
        surfaces.push(capture(&grid, sim.state(), &handles));
    }
    (grid, taus, surfaces)
}

/// The controlled ensemble after the full run: terminal (inv, wealth) samples.
struct Terminal {
    inv: Vec<f64>,
    wealth_mean: f64,
    wealth_std: f64,
}

fn simulate<Sch: Scheduler>(
    params: MarketMakerParams,
    horizon: f64,
    nu0: f64,
    s0: f64,
    paths: usize,
    seed: u64,
    scheduler: Sch,
) -> Terminal {
    // Stage 1: HJB solve → depth tables.
    let dt = {
        let model = HjbMarketMaker::new(params);
        let g = nu_grid(80, 20);
        <HjbMarketMaker as Model<CartesianGrid<1>, ()>>::stable_dt(&model, g.spacing(BlockId(0)))
            .unwrap()
    };
    let steps = (horizon / dt).ceil() as usize;
    let (nu_g, taus, surfaces) = solve_hjb(params, dt, steps);
    let tables = Arc::new(DepthTables::build(
        &HjbMarketMaker::new(params),
        &nu_g,
        &taus,
        &surfaces,
    ));

    // Stage 2: controlled ensemble on the generic Monte Carlo harness.
    let grid = path_grid(paths, 4000).unwrap();
    let model = MarketMakingEnsemble::new(params, Arc::clone(&tables), horizon);
    let sim = Simulation::new(
        grid,
        (),
        model,
        EulerMaruyama { seed },
        scheduler,
        SystemAllocator,
    );
    let mut mc = MonteCarlo::new(sim);
    {
        let model = mc.simulation().model().clone();
        let (g, state) = mc.simulation_mut().state_mut();
        model.initialize(g, state, nu0, s0);
    }
    mc.run(steps, dt);

    let model = mc.simulation().model().clone();
    let (inv_h, cash_h, mid_h) = (model.inv(), model.cash(), model.mid());
    let wealth = mc.stats(|p| model.terminal_wealth(p.get(cash_h), p.get(inv_h), p.get(mid_h)));

    let mut inv = Vec::with_capacity(paths);
    for b in 0..mc.simulation().grid().num_blocks() {
        let v = mc
            .simulation()
            .state()
            .view(mc.simulation().grid(), BlockId(b as u32), inv_h);
        for_each_interior(mc.simulation().grid().block_cells(), |idx| {
            inv.push(v.get(idx))
        });
    }
    Terminal {
        inv,
        wealth_mean: wealth.mean,
        wealth_std: wealth.std(),
    }
}

#[test]
fn inventory_stays_within_the_quoting_bounds() {
    // The optimal policy withdraws the buy quote at q_max and the sell quote
    // at q_min (infinite depth ⇒ zero fill intensity), so inventory can never
    // leave [q_min, q_max] — a hard invariant, checked on every path.
    let params = MarketMakerParams::default();
    let term = simulate(params, 3.0, params.theta, 100.0, 20_000, 1, SerialScheduler);
    assert!(
        term.inv
            .iter()
            .all(|&q| q >= params.q_min as f64 - 1e-9 && q <= params.q_max as f64 + 1e-9),
        "inventory left [{}, {}]",
        params.q_min,
        params.q_max
    );
    // Inventory is integer-valued (fills move it by exactly ±1).
    assert!(term.inv.iter().all(|&q| q == q.round()));
}

#[test]
fn market_maker_earns_positive_expected_wealth() {
    // Running the HJB-optimal quotes, the maker harvests the bid–ask spread:
    // mean terminal wealth is clearly positive, with genuine path dispersion.
    let params = MarketMakerParams::default();
    let term = simulate(params, 3.0, params.theta, 100.0, 40_000, 2, SerialScheduler);
    assert!(
        term.wealth_mean > 0.5,
        "expected terminal wealth {:.4} should be solidly positive",
        term.wealth_mean
    );
    assert!(term.wealth_std > 0.0, "wealth must vary across paths");
}

#[test]
fn symmetric_book_has_no_inventory_bias() {
    // A symmetric book (μ = 0, equal buy/sell intensities) quotes symmetrically
    // about q = 0, so the controlled inventory has no directional drift: the
    // ensemble-mean terminal inventory sits at zero to Monte Carlo error.
    let side = Side {
        lambda: 1.5,
        kappa: 2.0,
        epsilon: 0.004,
    };
    let params = MarketMakerParams {
        mu: 0.0,
        sell: side,
        buy: side,
        ..MarketMakerParams::default()
    };
    let term = simulate(params, 3.0, params.theta, 100.0, 40_000, 3, SerialScheduler);
    let mean_inv = term.inv.iter().sum::<f64>() / term.inv.len() as f64;
    assert!(
        mean_inv.abs() < 0.05,
        "symmetric book should hold mean inventory near 0, got {mean_inv:.4}"
    );
}

#[test]
fn controlled_ensemble_is_scheduler_independent() {
    let params = MarketMakerParams::default();
    let s = simulate(
        params,
        2.0,
        params.theta,
        100.0,
        16_000,
        42,
        SerialScheduler,
    );
    let p = simulate(params, 2.0, params.theta, 100.0, 16_000, 42, RayonScheduler);
    assert_eq!(s.inv, p.inv, "inventory paths must be schedule-independent");
    assert_eq!(
        s.wealth_mean, p.wealth_mean,
        "wealth must be schedule-independent"
    );
}
