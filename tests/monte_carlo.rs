//! The generic Monte Carlo harness ([`MonteCarlo`], [`ensemble_stats`]): it
//! drives *any* model as a path ensemble and reduces a per-path functional to
//! summary statistics. Exercised here with the library's stochastic-volatility
//! model, but the harness itself is model-agnostic.

#![allow(clippy::suboptimal_flops, clippy::float_cmp)]

use legendre::{
    core::{
        monte_carlo::{MonteCarlo, Stats, ensemble_stats, path_grid},
        scheduler::SerialScheduler,
        simulation::Simulation,
        storage::SystemAllocator,
    },
    integrators::EulerMaruyama,
    physics::stochastic_vol::{StochVolParams, StochVolPaths},
};

/// Wrap a stochastic-vol ensemble started at (`nu0`, `s0`).
fn ensemble(
    params: StochVolParams,
    paths: usize,
    block: usize,
    nu0: f64,
    s0: f64,
    seed: u64,
) -> MonteCarlo<1, (), StochVolPaths, EulerMaruyama, SerialScheduler, SystemAllocator> {
    let grid = path_grid(paths, block).unwrap();
    let sim = Simulation::new(
        grid,
        (),
        StochVolPaths::new(params),
        EulerMaruyama { seed },
        SerialScheduler,
        SystemAllocator,
    );
    let mut mc = MonteCarlo::new(sim);
    let model = mc.simulation().model().clone();
    let (g, state) = mc.simulation_mut().state_mut();
    model.initialize(g, state, nu0, s0);
    mc
}

#[test]
fn run_and_reduce_recovers_the_cir_mean() {
    // Drive the ensemble through the harness and reduce the terminal variance
    // field; the mean must match the analytic CIR law E[ν_t]=θ+(ν₀−θ)e^{−kt}.
    let params = StochVolParams::default();
    let (nu0, dt, steps) = (2.0, 0.01, 100);
    let mut mc = ensemble(params, 40_000, 4000, nu0, 100.0, 7);
    mc.run(steps, dt);

    let nu = mc.simulation().model().nu();
    let stats = mc.stats(|p| p.get(nu));

    let t = dt * steps as f64;
    let expected = params.theta + (nu0 - params.theta) * (-params.k_speed * t).exp();
    assert_eq!(stats.count, 40_000, "every path is one sample");
    assert!(
        (stats.mean - expected).abs() < 0.01,
        "ensemble mean {:.5} should match analytic {expected:.5}",
        stats.mean
    );
    // A genuine distribution formed: spread is positive and brackets the mean.
    assert!(stats.std() > 0.0 && stats.min < stats.mean && stats.mean < stats.max);
}

#[test]
fn stats_summarize_a_multi_field_payoff() {
    // A functional over *two* fields — the harness reads a whole path, not a
    // single field. E[ν + S] = E[ν] + E[S]; with ν₀ = θ, E[ν]≈θ and E[S]=s0.
    let params = StochVolParams {
        mu: 0.0,
        ..StochVolParams::default()
    };
    let (dt, steps, s0) = (0.01, 100, 100.0);
    let mut mc = ensemble(params, 40_000, 4000, params.theta, s0, 3);
    mc.run(steps, dt);

    let (nu, mid) = (mc.simulation().model().nu(), mc.simulation().model().mid());
    let stats = mc.stats(|p| p.get(nu) + p.get(mid));
    assert!(
        (stats.mean - (params.theta + s0)).abs() < 0.05,
        "E[ν+S] {:.4} should be θ+S₀ {:.4}",
        stats.mean,
        params.theta + s0
    );
}

#[test]
fn constant_functional_has_zero_variance() {
    // Pure reducer arithmetic, independent of any dynamics: a constant payoff
    // gives mean = c, zero variance, and min = max = c over exactly `count`
    // paths.
    let mc = ensemble(StochVolParams::default(), 8_000, 4000, 1.0, 100.0, 1);
    let stats = mc.stats(|_| 3.5);
    assert_eq!(stats.count, 8_000);
    assert_eq!(stats.mean, 3.5);
    assert_eq!(stats.variance, 0.0);
    assert_eq!(stats.min, 3.5);
    assert_eq!(stats.max, 3.5);
}

#[test]
fn ensemble_stats_free_function_matches_the_wrapper() {
    // `MonteCarlo::stats` is exactly `ensemble_stats` over the wrapped state.
    let mc = ensemble(StochVolParams::default(), 8_000, 4000, 1.5, 100.0, 2);
    let nu = mc.simulation().model().nu();
    let via_wrapper = mc.stats(|p| p.get(nu));
    let direct: Stats = ensemble_stats(mc.simulation().grid(), mc.simulation().state(), |p| {
        p.get(nu)
    });
    assert_eq!(via_wrapper, direct);
}

#[test]
fn path_grid_requires_divisible_sizes() {
    assert!(path_grid(100_000, 4000).is_ok());
    assert!(path_grid(100, 0).is_err());
    assert!(path_grid(100, 7).is_err(), "paths must divide into blocks");
}
