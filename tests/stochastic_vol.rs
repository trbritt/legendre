//! Behavioural validation of the stochastic-volatility path ensemble
//! ([`StochVolPaths`]): the analytic CIR mean-reversion law, variance
//! positivity, an unbiased price drift, and bitwise scheduler determinism.
//! Each cell is one independent Monte Carlo path; assertions are on ensemble
//! statistics, checked against closed-form expectations to Monte Carlo error.

// Reference expectations are written as plain textbook arithmetic so they
// stay readable and structurally independent of the model's `mul_add` code.
#![allow(clippy::suboptimal_flops)]

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
    integrators::EulerMaruyama,
    physics::stochastic_vol::{StochVolParams, StochVolPaths},
};

/// Simulate `paths` paths for `steps` of size `dt` from (`nu0`, `s0`); return
/// the terminal (ν, S) samples.
#[allow(clippy::too_many_arguments)]
fn run<Sch: Scheduler>(
    params: StochVolParams,
    paths: usize,
    block: usize,
    nu0: f64,
    s0: f64,
    dt: f64,
    steps: usize,
    seed: u64,
    scheduler: Sch,
) -> (Vec<f64>, Vec<f64>) {
    let grid = CartesianGrid::new([paths], [block], [0.0], [1.0]).unwrap();
    let mut sim = Simulation::new(
        grid,
        (),
        StochVolPaths::new(params),
        EulerMaruyama { seed },
        scheduler,
        SystemAllocator,
    );
    {
        let model = sim.model().clone();
        let (grid, state) = sim.state_mut();
        model.initialize(grid, state, nu0, s0);
    }
    for _ in 0..steps {
        sim.step(dt);
    }
    let (nu_h, mid_h) = (sim.model().nu(), sim.model().mid());
    let mut nu = Vec::with_capacity(paths);
    let mut mid = Vec::with_capacity(paths);
    for b in 0..sim.grid().num_blocks() {
        let block = BlockId(b as u32);
        let vnu = sim.state().view(sim.grid(), block, nu_h);
        let vmid = sim.state().view(sim.grid(), block, mid_h);
        for_each_interior(sim.grid().block_cells(), |idx| {
            nu.push(vnu.get(idx));
            mid.push(vmid.get(idx));
        });
    }
    (nu, mid)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

#[test]
fn cir_variance_mean_reverts_to_theta() {
    // The CIR mean obeys E[ν_t] = θ + (ν₀ − θ)·e^{−k t}, exactly and
    // independently of σ (the noise is zero-mean). Start well above θ and
    // check the ensemble mean tracks that analytic decay — a genuine
    // property of the process, not a re-run of the integrator.
    let params = StochVolParams {
        mu: 0.0,
        k_speed: 1.0,
        theta: 1.0,
        sigma: 0.1,
    };
    let (nu0, dt, steps) = (2.0, 0.01, 100);
    let t = dt * steps as f64;
    let (nu, _) = run(
        params,
        40_000,
        4000,
        nu0,
        100.0,
        dt,
        steps,
        7,
        SerialScheduler,
    );

    let expected = params.theta + (nu0 - params.theta) * (-params.k_speed * t).exp();
    let got = mean(&nu);
    assert!(
        (got - expected).abs() < 0.01,
        "CIR ensemble mean {got:.5} should track analytic {expected:.5} at t={t}"
    );
    // And it has actually reverted a long way from the start toward θ.
    assert!(
        got < nu0 - 0.5 && got > params.theta,
        "mean {got} must lie between θ and ν₀"
    );
}

#[test]
fn variance_stays_positive_under_heavy_vol_of_vol() {
    // Stress the positivity guard: a small starting variance with a large
    // vol-of-vol repeatedly pushes ν below zero, which full truncation plus
    // the reflecting `project` must absorb — no path may end negative.
    let params = StochVolParams {
        mu: 0.0,
        k_speed: 1.0,
        theta: 0.2,
        sigma: 0.8,
    };
    let (nu, _) = run(
        params,
        20_000,
        4000,
        0.02,
        100.0,
        0.01,
        200,
        11,
        SerialScheduler,
    );
    assert!(
        nu.iter().all(|&v| v >= 0.0),
        "CIR positivity violated: min ν = {}",
        nu.iter().copied().fold(f64::INFINITY, f64::min)
    );
}

#[test]
fn midprice_drift_is_unbiased() {
    // The price has deterministic drift μ and zero-mean diffusion, so
    // E[S_T] = S₀ + μT exactly; the ensemble mean matches to MC error. Its
    // dispersion is set by the variance level (vol = √ν), so a non-trivial
    // spread must develop — the price is not frozen.
    let params = StochVolParams {
        mu: 0.05,
        k_speed: 1.0,
        theta: 1.0,
        sigma: 0.1,
    };
    let (s0, dt, steps) = (100.0, 0.01, 100);
    let t = dt * steps as f64;
    let (_, mid) = run(
        params,
        40_000,
        4000,
        params.theta,
        s0,
        dt,
        steps,
        3,
        SerialScheduler,
    );

    let got = mean(&mid);
    let expected = s0 + params.mu * t;
    assert!(
        (got - expected).abs() < 0.02,
        "midprice ensemble mean {got:.4} should equal S₀+μT {expected:.4}"
    );
    // With ν ≈ θ over [0,T], Var[S_T] ≈ ∫₀^T ν dt ≈ θ·T; order of magnitude.
    let var = variance(&mid);
    let target = params.theta * t;
    assert!(
        (0.5 * target..2.0 * target).contains(&var),
        "price variance {var:.4} should be of order θ·T {target:.4}"
    );
}

#[test]
fn paths_are_bitwise_scheduler_independent() {
    // Counter-based noise ⇒ identical results regardless of thread count.
    let params = StochVolParams::default();
    let s = run(
        params,
        16_000,
        4000,
        1.0,
        100.0,
        0.01,
        50,
        42,
        SerialScheduler,
    );
    let p = run(
        params,
        16_000,
        4000,
        1.0,
        100.0,
        0.01,
        50,
        42,
        RayonScheduler,
    );
    assert_eq!(s, p, "scheduling must not change the sampled paths");
}
