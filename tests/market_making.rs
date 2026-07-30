//! Behavioural validation of the stochastic-volatility market-making HJB
//! solver.
//!
//! These are not "does one Euler match another Euler" checks — they pick
//! concrete, economically meaningful parameterizations and assert the
//! properties an *optimal* market maker's value function and quotes must
//! have, each justified from the model rather than by re-deriving the
//! discretization:
//!
//! - **Spread floor.** At flat inventory the maker earns the myopic optimal
//!   spread on both sides at a rate with a closed form.
//! - **Inventory aversion.** Carrying inventory is penalized, so a symmetric
//!   book values flat inventory most and less the further from it.
//! - **Book symmetry.** A symmetric book values `+q` and `−q` identically.
//! - **Inventory skew.** A long maker quotes a tighter ask than bid (to
//!   offload) and a short maker the reverse.
//! - **Determinism.** The solve is bitwise scheduler-independent.

#![allow(clippy::float_cmp)]

use legendre::{
    core::{
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
        simulation::Simulation,
        storage::SystemAllocator,
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::ForwardEuler,
    physics::market_making::{HjbMarketMaker, MarketMakerParams, Side},
};

const NU_MIN: f64 = 0.001;
const NU_MAX: f64 = 2.0;

fn nu_grid(cells: usize, block: usize) -> CartesianGrid<1> {
    let dnu = (NU_MAX - NU_MIN) / cells as f64;
    CartesianGrid::new([cells], [block], [NU_MIN], [dnu]).unwrap()
}

/// The model's stable dt on `grid` (the CFL is a pure function of spacing).
fn stable_dt(params: MarketMakerParams, grid: &CartesianGrid<1>) -> f64 {
    use legendre::physics::model::Model;
    let model = HjbMarketMaker::new(params);
    <HjbMarketMaker as Model<CartesianGrid<1>, ()>>::stable_dt(&model, grid.spacing(BlockId(0)))
        .unwrap()
}

/// Solve the HJB forward in τ for `steps` and return the value surface as
/// `u[q_index][cell]` (`q_index` 0 ⇔ `q_min`), in global cell order.
fn solve<Sch: Scheduler>(
    params: MarketMakerParams,
    grid: CartesianGrid<1>,
    scheduler: Sch,
    dt: f64,
    steps: usize,
) -> Vec<Vec<f64>> {
    let mut sim = Simulation::new(
        grid,
        (),
        HjbMarketMaker::new(params),
        ForwardEuler,
        scheduler,
        SystemAllocator,
    );
    for _ in 0..steps {
        sim.step(dt);
    }
    let handles: Vec<_> = sim.model().handles().to_vec();
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

/// Per-unit-time profit of a maker quoting the myopic optimal spread on both
/// sides at flat inventory (u ≡ 0): `λˢ·e^{−κˢ(εˢ+α)}/κˢ + λᵇ·e^{−κᵇ(εᵇ+α)}/κᵇ`.
fn spread_floor(p: &MarketMakerParams) -> f64 {
    p.sell.lambda * (-p.sell.kappa * (p.sell.epsilon + p.alpha)).exp() / p.sell.kappa
        + p.buy.lambda * (-p.buy.kappa * (p.buy.epsilon + p.alpha)).exp() / p.buy.kappa
}

#[test]
fn flat_inventory_earns_the_closed_form_spread_floor() {
    // At q = 0 the μq drift and the ψσ²νq² / αq² inventory penalties all
    // vanish, and from the u ≡ 0 terminal condition the ν-derivatives are
    // zero too. So the very first backward-time step is *exactly* dt times
    // the sum of the two side-Hamiltonians at u = 0 — the spread floor —
    // with no ν dependence and no reassociation noise.
    let p = MarketMakerParams::default();
    let floor = spread_floor(&p);
    let dt = 1e-3;
    let q0 = (-p.q_min) as usize;

    let one = solve(p, nu_grid(16, 16), SerialScheduler, dt, 1);
    for cell in &one[q0] {
        assert_eq!(
            *cell,
            dt * floor,
            "first step at flat inventory must equal dt·(spread floor)"
        );
    }

    // Over a short horizon it keeps compounding at ≈ that rate: the only
    // correction is the (small, positive) inventory-value gap u(0) − u(±1)
    // entering the Hamiltonian exponents, which mildly slows growth — so the
    // realized value is a touch *below* floor·τ, never above it.
    let (steps, cells) = (100, 16);
    let tau = dt * steps as f64;
    let horizon = solve(p, nu_grid(cells, cells), SerialScheduler, dt, steps);
    for cell in &horizon[q0] {
        let ratio = cell / (floor * tau);
        assert!(
            (0.9..=1.0).contains(&ratio),
            "flat-inventory value {cell} should track the spread floor {} (ratio {ratio:.4})",
            floor * tau
        );
    }
}

#[test]
fn symmetric_book_values_are_inventory_symmetric() {
    // Equal, symmetric buy/ask intensities and zero midprice drift make the
    // equation invariant under q ↔ −q, so u(ν, q) = u(ν, −q) exactly.
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
    let grid = nu_grid(32, 16);
    let dt = stable_dt(params, &grid);
    let u = solve(params, grid, SerialScheduler, dt, 300);

    let n = params.num_levels();
    for qi in 0..n {
        let mirror = n - 1 - qi; // q ↔ −q
        for (a, b) in u[qi].iter().zip(&u[mirror]) {
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                "u(ν, q) must equal u(ν, −q) for a symmetric book: {a} vs {b}"
            );
        }
    }
}

#[test]
fn holding_inventory_is_penalized() {
    // A symmetric, inventory-averse book (ψ > 0): every unit of inventory
    // carries the running penalty ψσ²νq² and the terminal penalty αq², while
    // the spread earned per step is largest when both sides are quoted
    // symmetrically at q = 0. So the value must peak at flat inventory and
    // fall away monotonically as |q| grows.
    let side = Side {
        lambda: 1.5,
        kappa: 2.0,
        epsilon: 0.004,
    };
    let params = MarketMakerParams {
        mu: 0.0,
        psi: 0.5, // pronounced inventory aversion
        sell: side,
        buy: side,
        ..MarketMakerParams::default()
    };
    let grid = nu_grid(40, 20);
    let dt = stable_dt(params, &grid);
    let u = solve(params, grid, SerialScheduler, dt, 400);

    let q0 = (-params.q_min) as usize;
    let cells = u[0].len();
    for cell in 0..cells {
        // Walk outward from flat inventory to the long bound; each step out
        // must not increase the value.
        for qi in q0..params.num_levels() - 1 {
            assert!(
                u[qi][cell] >= u[qi + 1][cell] - 1e-12,
                "value must not increase moving from q={} to q={} (cell {cell}): {} -> {}",
                qi as isize + params.q_min,
                qi as isize + 1 + params.q_min,
                u[qi][cell],
                u[qi + 1][cell],
            );
        }
        // Flat inventory is the global maximum over inventory.
        let best = u.iter().map(|c| c[cell]).fold(f64::MIN, f64::max);
        assert!(
            (u[q0][cell] - best).abs() < 1e-12,
            "flat inventory must be the most valuable state at cell {cell}"
        );
    }
}

#[test]
fn long_inventory_skews_quotes_toward_offloading() {
    // Solve a *symmetric* book (so any quote asymmetry is inventory-driven,
    // not book- or drift-driven), then read the optimal quotes off the
    // surface at a mid-variance cell. A long maker (q > 0) wants to shed
    // inventory, so it must quote a tighter ask than bid; a short maker the
    // reverse; a flat book quotes symmetrically; and, by q ↔ −q symmetry, the
    // long and short quote pairs are exact mirror images. Depth asymmetry is
    // the model's inventory control.
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
    let grid = nu_grid(64, 16);
    let dt = stable_dt(params, &grid);
    let steps = (5.0 / dt).ceil() as usize; // a few units of horizon
    let u = solve(params, grid, RayonScheduler, dt, steps);

    let model = HjbMarketMaker::new(params);
    let cell = u[0].len() / 2; // mid-variance node
    let depths = |q: isize| {
        let qi = (q - params.q_min) as usize;
        let up = (q < params.q_max).then(|| u[qi + 1][cell]);
        let dn = (q > params.q_min).then(|| u[qi - 1][cell]);
        model.optimal_depths(q, u[qi][cell], up, dn)
    };

    let (bid0, ask0) = depths(0);
    assert!(
        (bid0 - ask0).abs() < 1e-9,
        "a symmetric flat book quotes symmetrically: bid {bid0} vs ask {ask0}"
    );

    for q in 1..params.q_max {
        let (bid_long, ask_long) = depths(q);
        assert!(
            ask_long < bid_long,
            "long q={q}: ask {ask_long} must be tighter than bid {bid_long} (offload)"
        );
        let (bid_short, ask_short) = depths(-q);
        assert!(
            bid_short < ask_short,
            "short q={}: bid {bid_short} must be tighter than ask {ask_short} (cover)",
            -q
        );
        // Exact q ↔ −q mirror: the long book's ask is the short book's bid.
        assert!(
            (ask_long - bid_short).abs() < 1e-9 && (bid_long - ask_short).abs() < 1e-9,
            "long/short quotes must mirror: ({bid_long},{ask_long}) vs ({bid_short},{ask_short})"
        );
    }
}

#[test]
fn positive_inventory_drift_biases_the_maker_long() {
    // Isolate the midprice drift: a book with symmetric intensities but
    // μ > 0. Holding a long position now earns μq·dt, so near flat inventory
    // the value *rises* with q — u(1) > u(0) > u(−1). That makes the maker
    // quote a tighter bid than ask even at q = 0: it wants to accumulate the
    // appreciating asset. (This is why the default sell-heavy, μ > 0 book
    // does *not* quote symmetrically at flat inventory.)
    let side = Side {
        lambda: 1.5,
        kappa: 2.0,
        epsilon: 0.004,
    };
    let params = MarketMakerParams {
        mu: 0.05,
        psi: 0.1,
        sell: side,
        buy: side,
        ..MarketMakerParams::default()
    };
    let grid = nu_grid(64, 16);
    let dt = stable_dt(params, &grid);
    let steps = (5.0 / dt).ceil() as usize;
    let u = solve(params, grid, SerialScheduler, dt, steps);

    let q0 = (-params.q_min) as usize;
    let cell = u[0].len() / 2;
    // Value rises with inventory around flat: being long is rewarded.
    assert!(
        u[q0 + 1][cell] > u[q0][cell] && u[q0][cell] > u[q0 - 1][cell],
        "μ > 0 must make value increase with inventory near q = 0: {} < {} < {}",
        u[q0 - 1][cell],
        u[q0][cell],
        u[q0 + 1][cell],
    );

    let model = HjbMarketMaker::new(params);
    let up = Some(u[q0 + 1][cell]);
    let dn = Some(u[q0 - 1][cell]);
    let (bid, ask) = model.optimal_depths(0, u[q0][cell], up, dn);
    assert!(
        bid < ask,
        "μ > 0 flat book must quote a tighter bid than ask (eager to buy): bid {bid} vs ask {ask}"
    );
}

#[test]
fn optimal_depths_clamp_and_withdraw() {
    // Closed-form corner behaviour of the quote map, independent of any solve.
    let params = MarketMakerParams::default();
    let mut model = HjbMarketMaker::new(params);
    {
        use legendre::{core::state::StateBuilder, physics::model::Model};
        let mut builder = StateBuilder::new();
        <HjbMarketMaker as Model<CartesianGrid<1>, ()>>::register_fields(&mut model, &mut builder);
    }

    // Flat value surface ⇒ the myopic base spread 1/κ + ε + α on each side.
    let (bid, ask) = model.optimal_depths(0, 0.0, Some(0.0), Some(0.0));
    assert!((bid - (1.0 / 2.0 + 0.004 + 0.01)).abs() < 1e-15);
    assert!((ask - (1.0 / 2.0 + 0.004 + 0.01)).abs() < 1e-15);

    // A steep value gain for unloading clamps the ask at the minimum spread.
    let (_, ask_clamped) = model.optimal_depths(0, 0.0, Some(0.0), Some(10.0));
    assert_eq!(ask_clamped, params.min_spread);

    // Withdrawn sides at the inventory bounds are infinite and fill at rate 0.
    let (bid_top, _) = model.optimal_depths(params.q_max, 0.0, None, Some(0.0));
    assert!(bid_top.is_infinite());
    assert_eq!(params.buy.fill_rate(bid_top), 0.0);
    assert!(params.buy.fill_rate(0.5) > 0.0);
}

#[test]
fn solve_is_bitwise_scheduler_independent() {
    // Reproducibility is a hard framework guarantee, not an approximation.
    let params = MarketMakerParams::default();
    let dt = 0.01;
    let serial = solve(params, nu_grid(64, 16), SerialScheduler, dt, 200);
    let parallel = solve(params, nu_grid(64, 16), RayonScheduler, dt, 200);
    assert_eq!(serial, parallel, "scheduling must not change results");
}
