//! Behavioural validation of the kernel-owned jump driver ([`Driver::Jump`])
//! and the mixed [`WienerJump`] driver set.
//!
//! The model is a per-cell jump-diffusion: a drift-diffusion `w` moved by the
//! time and Wiener drivers, plus a point process that, on each fire,
//! increments a counter and adds a fixed payoff to an accumulator — both
//! moved by one jump driver, at a per-path intensity `λ` read from state.
//! Assertions are on ensemble statistics against closed-form expectations,
//! plus the exact multi-field lockstep the driver guarantees.

// Analytic expectations are plain textbook arithmetic; the lockstep check is
// an exact-equality property.
#![allow(clippy::suboptimal_flops, clippy::float_cmp)]

use legendre::{
    core::{
        driver::WienerJump,
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
    integrators::EulerMaruyama,
    physics::model::{Driver, Model, RhsContext},
};

const MU: f64 = 0.1; // drift of w
const SIGMA: f64 = 0.5; // diffusion of w
const PAYOFF: f64 = 2.0; // jump size added to `jsum` per fire

/// A per-cell jump-diffusion driven by one Wiener process and one jump
/// process (`WienerJump<1, 1>`):
/// `dw = μ dt + σ dW`, and on each fire `count += 1`, `jsum += PAYOFF`, with
/// per-path firing intensity `λ` held in the static field `lam`.
#[derive(Clone)]
struct JumpDiffusion {
    w: Option<FieldHandle<f64>>,
    count: Option<FieldHandle<f64>>,
    jsum: Option<FieldHandle<f64>>,
    lam: Option<FieldHandle<f64>>,
}

impl<D: Sync> Model<CartesianGrid<1>, D> for JumpDiffusion {
    type Scalar = f64;
    type Drivers = WienerJump<1, 1>;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.w = Some(builder.register_driven("w", 0, &[Driver::Time, Driver::Wiener(0)]));
        self.count = Some(builder.register_driven("count", 0, &[Driver::Jump(0)]));
        self.jsum = Some(builder.register_driven("jsum", 0, &[Driver::Jump(0)]));
        self.lam = Some(builder.register_static("lam", 0));
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        driver: Driver,
        ctx: &RhsContext<'_, CartesianGrid<1>, D>,
        state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        let (grid, block) = (ctx.grid, ctx.block);
        match driver {
            // Drift of w (the only time-driven field).
            Driver::Time => {
                let mut dw = out.view_mut(grid, block, self.w.unwrap());
                for_each_interior(grid.block_cells(), |idx| dw.set(idx, MU));
            }
            // Diffusion amplitude of w.
            Driver::Wiener(0) => {
                let mut amp = out.view_mut(grid, block, self.w.unwrap());
                for_each_interior(grid.block_cells(), |idx| amp.set(idx, SIGMA));
            }
            // Jump: increments for the two coupled fields + the per-cell
            // intensity read from state.
            Driver::Jump(0) => {
                let mut dcount = out.view_mut(grid, block, self.count.unwrap());
                for_each_interior(grid.block_cells(), |idx| dcount.set(idx, 1.0));
                let mut djsum = out.view_mut(grid, block, self.jsum.unwrap());
                for_each_interior(grid.block_cells(), |idx| djsum.set(idx, PAYOFF));

                let lam = state.view(grid, block, self.lam.unwrap());
                let mut rate = out.intensity_mut(grid, block);
                for_each_interior(grid.block_cells(), |idx| rate.set(idx, lam.get(idx)));
            }
            Driver::Wiener(_) | Driver::Jump(_) => unreachable!("declares WienerJump<1,1>"),
        }
    }
}

struct Paths {
    count: Vec<f64>,
    jsum: Vec<f64>,
    w: Vec<f64>,
}

/// Simulate `paths` jump-diffusion paths, intensity `lam_of(path)`.
#[allow(clippy::too_many_arguments)]
fn run<Sch: Scheduler>(
    lam_of: impl Fn(usize) -> f64,
    paths: usize,
    block: usize,
    dt: f64,
    steps: usize,
    seed: u64,
    scheduler: Sch,
) -> Paths {
    let grid = CartesianGrid::new([paths], [block], [0.0], [1.0]).unwrap();
    let model = JumpDiffusion {
        w: None,
        count: None,
        jsum: None,
        lam: None,
    };
    let mut sim = Simulation::new(
        grid,
        (),
        model,
        EulerMaruyama { seed },
        scheduler,
        SystemAllocator,
    );

    let lam_h = sim.model().lam.unwrap();
    let (g, state) = sim.state_mut();
    for b in 0..g.num_blocks() {
        let blk = BlockId(b as u32);
        let base = b * block;
        let mut v = state.view_mut(g, blk, lam_h);
        for_each_interior(g.block_cells(), |idx| {
            v.set(idx, lam_of(base + idx[0] as usize))
        });
    }

    for _ in 0..steps {
        sim.step(dt);
    }

    let (count_h, jsum_h, w_h) = (
        sim.model().count.unwrap(),
        sim.model().jsum.unwrap(),
        sim.model().w.unwrap(),
    );
    let (mut count, mut jsum, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for b in 0..sim.grid().num_blocks() {
        let blk = BlockId(b as u32);
        let (vc, vj, vw) = (
            sim.state().view(sim.grid(), blk, count_h),
            sim.state().view(sim.grid(), blk, jsum_h),
            sim.state().view(sim.grid(), blk, w_h),
        );
        for_each_interior(sim.grid().block_cells(), |idx| {
            count.push(vc.get(idx));
            jsum.push(vj.get(idx));
            w.push(vw.get(idx));
        });
    }
    Paths { count, jsum, w }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

/// Expected count of a Bernoulli-thinned Poisson process over `steps` steps
/// of size `dt` at intensity `lam`: each step fires with probability
/// `1 − e^{−λ dt}`, so `E[count] = steps·(1 − e^{−λ dt})` exactly.
fn thinned_mean(lam: f64, dt: f64, steps: usize) -> f64 {
    steps as f64 * (1.0 - (-lam * dt).exp())
}

#[test]
fn jump_count_matches_thinned_poisson_mean() {
    let (lam, dt, steps) = (2.0, 0.01, 100);
    let p = run(|_| lam, 50_000, 5000, dt, steps, 1, SerialScheduler);
    let got = mean(&p.count);
    let expected = thinned_mean(lam, dt, steps);
    assert!(
        (got - expected).abs() < 0.03,
        "mean count {got:.4} should match thinned-Poisson {expected:.4}"
    );
    // The counter is an integer count on every path.
    assert!(
        p.count.iter().all(|&c| c == c.round() && c >= 0.0),
        "counts must be non-negative integers"
    );
}

#[test]
fn coupled_fields_move_on_the_same_fire() {
    // The jump moves `count` and `jsum` together, so on *every* path the two
    // are locked: `jsum == count·PAYOFF` exactly, with no statistical slack.
    // This is the multi-field guarantee — one fire, all the driver's fields.
    let p = run(|_| 3.0, 20_000, 5000, 0.01, 100, 5, SerialScheduler);
    for (c, j) in p.count.iter().zip(&p.jsum) {
        assert_eq!(
            *j,
            c * PAYOFF,
            "coupled fields diverged: count {c}, jsum {j}"
        );
    }
}

#[test]
fn intensity_is_read_per_cell() {
    // Two path groups with different per-cell intensities in the *same* run:
    // each group's mean count must match its own thinned-Poisson expectation,
    // proving the kernel thins against the per-cell λ the model wrote.
    let (lo, hi, dt, steps) = (1.0, 5.0, 0.01, 100);
    let n = 40_000;
    let p = run(
        |i| if i < n / 2 { lo } else { hi },
        n,
        5000,
        dt,
        steps,
        9,
        SerialScheduler,
    );
    let lo_mean = mean(&p.count[..n / 2]);
    let hi_mean = mean(&p.count[n / 2..]);
    assert!(
        (lo_mean - thinned_mean(lo, dt, steps)).abs() < 0.05,
        "low-λ group mean {lo_mean:.4} vs {:.4}",
        thinned_mean(lo, dt, steps)
    );
    assert!(
        (hi_mean - thinned_mean(hi, dt, steps)).abs() < 0.05,
        "high-λ group mean {hi_mean:.4} vs {:.4}",
        thinned_mean(hi, dt, steps)
    );
    assert!(hi_mean > lo_mean, "higher intensity must fire more often");
}

#[test]
fn zero_intensity_never_fires() {
    let p = run(|_| 0.0, 10_000, 5000, 0.01, 200, 3, SerialScheduler);
    assert!(
        p.count.iter().all(|&c| c == 0.0) && p.jsum.iter().all(|&j| j == 0.0),
        "a zero-intensity jump driver must never fire"
    );
}

#[test]
fn wiener_and_jump_coexist_in_a_mixed_set() {
    // In the same WienerJump<1,1> model, the diffusion `w` keeps its own
    // statistics — E[w]=μT, Var[w]=σ²T — undisturbed by the jump driver.
    let (dt, steps) = (0.01, 100);
    let t = dt * steps as f64;
    let p = run(|_| 3.0, 50_000, 5000, dt, steps, 7, SerialScheduler);
    let (m, v) = (mean(&p.w), variance(&p.w));
    assert!(
        (m - MU * t).abs() < 0.02,
        "E[w] {m:.4} should be μT {:.4}",
        MU * t
    );
    assert!(
        (v - SIGMA * SIGMA * t).abs() < 0.02,
        "Var[w] {v:.4} should be σ²T {:.4}",
        SIGMA * SIGMA * t
    );
}

#[test]
fn jump_paths_are_bitwise_scheduler_independent() {
    let s = run(|_| 2.5, 16_000, 4000, 0.01, 60, 42, SerialScheduler);
    let p = run(|_| 2.5, 16_000, 4000, 0.01, 60, 42, RayonScheduler);
    assert_eq!(s.count, p.count, "counts must be schedule-independent");
    assert_eq!(s.jsum, p.jsum, "jump sums must be schedule-independent");
    assert_eq!(s.w, p.w, "diffusion must be schedule-independent");
}
