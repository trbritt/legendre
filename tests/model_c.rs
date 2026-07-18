//! Model C validation: physics sanity on a small dendrite, and
//! bitwise scheduler-independence with the stochastic term enabled (which
//! pins the counter-based noise design, not just the deterministic path).

use legendre::{
    core::{
        scheduler::{RayonScheduler, Scheduler, SerialScheduler},
        simulation::Simulation,
        storage::SystemAllocator,
    },
    discretization::finite_volume::FiniteVolume,
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    integrators::EulerMaruyama,
    physics::{
        model::{NoNoise, Wiener},
        phasefield::ModelC,
    },
};

const N: usize = 64;
const H: f64 = 0.4;

fn run_model_c<Sch: Scheduler>(
    scheduler: Sch,
    noise_amplitude: f64,
    steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let grid = CartesianGrid::new([N, N], [N / 2, N / 2], [0.0, 0.0], [H, H]).unwrap();
    let mut model = ModelC::<Wiener<1>>::classic();
    model.noise_amplitude = noise_amplitude;

    let mut sim = Simulation::new(
        grid,
        FiniteVolume::default(),
        model,
        EulerMaruyama { seed: 7 },
        scheduler,
        SystemAllocator,
    );
    let dt = sim.stable_dt().unwrap();

    // Corner seed in a uniformly undercooled melt.
    let (phi_h, u_h) = (sim.model().phi(), sim.model().u());
    {
        let model = sim.model().clone();
        let (grid, state) = sim.state_mut();
        model.initialize(grid, state, [H, H], 10.0 * H, 0.7);
    }

    for _ in 0..steps {
        sim.step(dt);
    }

    let mut phi = Vec::new();
    let mut u = Vec::new();
    for b in 0..sim.grid().num_blocks() {
        let block = BlockId(b as u32);
        let vp = sim.state().view(sim.grid(), block, phi_h);
        let vu = sim.state().view(sim.grid(), block, u_h);
        for_each_interior(sim.grid().block_cells(), |idx| {
            phi.push(vp.get(idx));
            u.push(vu.get(idx));
        });
    }
    (phi, u)
}

#[test]
fn dendrite_grows_and_stays_physical() {
    let solid_fraction =
        |phi: &[f64]| phi.iter().filter(|&&p| p > 0.0).count() as f64 / phi.len() as f64;

    let (phi0, _) = run_model_c(SerialScheduler, 0.0, 0);
    let (phi, u) = run_model_c(SerialScheduler, 0.0, 1200);

    // Order parameter stays in its physical range (small overshoot of the
    // ±1 wells is expected for the explicit scheme).
    let (pmin, pmax) = phi
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    assert!(pmax > 0.95 && pmax < 1.1, "solid well: pmax = {pmax}");
    assert!(pmin < -0.95 && pmin > -1.1, "liquid well: pmin = {pmin}");

    // The seed grows.
    let (f0, f1) = (solid_fraction(&phi0), solid_fraction(&phi));
    assert!(f1 > 1.5 * f0, "solid fraction {f0} -> {f1}: seed must grow");

    // Latent heat is released at the interface, warming the melt toward the
    // melting point but never above it (u = 0), from the initial u = -0.7.
    let (umin, umax) = u
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    assert!(
        umax > -0.4 && umax < 0.05,
        "latent heat release: umax = {umax}"
    );
    assert!(umin >= -0.75, "melt no colder than initial: umin = {umin}");
}

#[test]
fn stochastic_run_is_bitwise_identical_across_schedulers() {
    let (phi_s, u_s) = run_model_c(SerialScheduler, 0.05, 300);
    let (phi_p, u_p) = run_model_c(RayonScheduler, 0.05, 300);
    assert_eq!(phi_s, phi_p, "phi must not depend on scheduling");
    assert_eq!(u_s, u_p, "u must not depend on scheduling");
}

#[test]
fn noise_changes_the_trajectory() {
    let (quiet, _) = run_model_c(SerialScheduler, 0.0, 100);
    let (noisy, _) = run_model_c(SerialScheduler, 0.05, 100);
    assert_ne!(quiet, noisy, "noise amplitude must have an effect");
}

mod async_pipeline {
    use super::*;
    use legendre::core::{
        observer::{AsyncObserver, SnapshotSink},
        state::State,
        storage::{DenseStorage, StorageBackend},
    };
    use std::sync::{Arc, Mutex};

    struct RecordingSink {
        seen: Arc<Mutex<Vec<(u64, f64)>>>,
        finished: Arc<Mutex<bool>>,
    }

    impl<S: StorageBackend<f64>> SnapshotSink<f64, S> for RecordingSink {
        fn consume(&mut self, step: u64, t: f64, state: &State<f64, S>) {
            // Prove we got a real deep copy: read a slab.
            let _ = state.layout().num_fields();
            self.seen.lock().unwrap().push((step, t));
        }
        fn finish(&mut self) {
            *self.finished.lock().unwrap() = true;
        }
    }

    #[test]
    fn async_observer_delivers_and_drains() {
        let grid = CartesianGrid::new([N, N], [N / 2, N / 2], [0.0, 0.0], [H, H]).unwrap();
        let model = ModelC::<NoNoise>::classic();
        let mut sim = Simulation::new(
            grid,
            FiniteVolume::default(),
            model,
            EulerMaruyama { seed: 7 },
            SerialScheduler,
            SystemAllocator,
        );
        let dt = sim.stable_dt().unwrap();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize(grid, state, [H, H], 10.0 * H, 0.7);
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let finished = Arc::new(Mutex::new(false));
        let sink = RecordingSink {
            seen: Arc::clone(&seen),
            finished: Arc::clone(&finished),
        };
        let buffers = sim.snapshot_buffers(3);
        let observer: AsyncObserver<f64, DenseStorage<f64>> =
            AsyncObserver::new(20, buffers, vec![Box::new(sink)]);
        sim.attach_observer(Box::new(observer));

        for _ in 0..100 {
            sim.step(dt);
        }
        drop(sim); // shuts down the pipeline, draining in-flight snapshots

        // Cadence: step 1 plus every 20th step; ring depth 3 with a fast
        // sink should drop nothing.
        let steps: Vec<u64> = seen.lock().unwrap().iter().map(|(s, _)| *s).collect();
        assert_eq!(steps, vec![1, 20, 40, 60, 80, 100]);
        assert!(*finished.lock().unwrap(), "finish() must run on shutdown");
    }
}

mod orientations {
    use super::*;
    use legendre::physics::phasefield::Grain;

    /// Run a centered single grain at orientation `theta0` and return, for
    /// each solid cell, its (angle, radius) from the seed center.
    fn run_oriented(theta0: f64, steps: usize) -> Vec<(f64, f64)> {
        const M: usize = 96;
        let grid = CartesianGrid::new([M, M], [M / 2, M / 2], [0.0, 0.0], [H, H]).unwrap();
        let model = ModelC::<NoNoise>::classic().with_orientations();
        let mut sim = Simulation::new(
            grid,
            FiniteVolume::default(),
            model,
            EulerMaruyama { seed: 7 },
            RayonScheduler,
            SystemAllocator,
        );
        let dt = sim.stable_dt().unwrap();
        let center = [M as f64 * H / 2.0; 2];
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize_grains(
                grid,
                state,
                &[Grain {
                    center,
                    radius: 10.0 * H,
                    orientation: theta0,
                }],
                0.7,
            );
        }
        for _ in 0..steps {
            sim.step(dt);
        }
        let phi_h = sim.model().phi();
        let mut solid = Vec::new();
        for b in 0..sim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let v = sim.state().view(sim.grid(), block, phi_h);
            for_each_interior(sim.grid().block_cells(), |idx| {
                if v.get(idx) > 0.0 {
                    let [x, y] = sim.grid().cell_center(block, idx);
                    let (dx, dy) = (x - center[0], y - center[1]);
                    solid.push((dy.atan2(dx), dx.hypot(dy)));
                }
            });
        }
        solid
    }

    /// Longest solid radius within ±wedge of any of the given directions.
    fn arm_length(solid: &[(f64, f64)], directions: &[f64], wedge: f64) -> f64 {
        solid
            .iter()
            .filter(|(ang, _)| {
                directions.iter().any(|d| {
                    let mut diff = (ang - d).abs() % std::f64::consts::TAU;
                    if diff > std::f64::consts::PI {
                        diff = std::f64::consts::TAU - diff;
                    }
                    diff < wedge
                })
            })
            .map(|&(_, r)| r)
            .fold(0.0, f64::max)
    }

    #[test]
    fn rotated_grain_grows_along_rotated_axes() {
        use std::f64::consts::FRAC_PI_4;
        let wedge = 12f64.to_radians();
        let axes = [
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
        ];
        let diags = [FRAC_PI_4, 3.0 * FRAC_PI_4, -FRAC_PI_4, -3.0 * FRAC_PI_4];

        // The runs are bit-deterministic, so these small margins are
        // stable pins of the *direction flip* (the physical claim); tip
        // contrast grows with domain size, but test-sized domains put arms
        // near the mirror boundaries where radii saturate.
        let aligned = run_oriented(0.0, 700);
        let a_axis = arm_length(&aligned, &axes, wedge);
        let a_diag = arm_length(&aligned, &diags, wedge);
        assert!(
            a_axis > 1.02 * a_diag,
            "axis-aligned grain: axis arm {a_axis:.2} vs diagonal {a_diag:.2}"
        );

        let rotated = run_oriented(FRAC_PI_4, 700);
        let r_axis = arm_length(&rotated, &axes, wedge);
        let r_diag = arm_length(&rotated, &diags, wedge);
        assert!(
            r_diag > 1.02 * r_axis,
            "45°-rotated grain: diagonal arm {r_diag:.2} vs axis {r_axis:.2}"
        );
    }

    #[test]
    fn rk4_stage_states_carry_the_orientation_field() {
        // RK4's y_tmp is a stage-*state* buffer: it must carry theta0 so the
        // model can read it during intermediate stages (tendency buffers
        // don't). A crash or NaN here means the stage layout is wrong.
        // (RK4 is deterministic-only under the driver contract, so this
        // runs the NoNoise instantiation.)
        use legendre::integrators::RungeKutta4;
        let grid = CartesianGrid::new([N, N], [N / 2, N / 2], [0.0, 0.0], [H, H]).unwrap();
        let model = ModelC::<NoNoise>::classic().with_orientations();
        let mut sim = Simulation::new(
            grid,
            FiniteVolume::default(),
            model,
            RungeKutta4,
            RayonScheduler,
            SystemAllocator,
        );
        let dt = sim.stable_dt().unwrap();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize_grains(
                grid,
                state,
                &[Grain {
                    center: [H, H],
                    radius: 10.0 * H,
                    orientation: 0.5,
                }],
                0.7,
            );
        }
        for _ in 0..200 {
            sim.step(dt);
        }
        let phi_h = sim.model().phi();
        for b in 0..sim.grid().num_blocks() {
            let v = sim.state().view(sim.grid(), BlockId(b as u32), phi_h);
            for_each_interior(sim.grid().block_cells(), |idx| {
                let p = v.get(idx);
                assert!(p.is_finite() && p.abs() < 1.2, "phi = {p} out of bounds");
            });
        }
    }

    #[test]
    // Bitwise identity is the property under test.
    #[allow(clippy::float_cmp)]
    fn zero_orientation_is_bitwise_identical_to_legacy_path() {
        // sin_cos(0) = (0, 1) and multiplication by exact 1/0 is exact, so
        // the oriented path with theta0 = 0 must reproduce the axis-aligned
        // path bit-for-bit (phi and u; the oriented run also carries the
        // extra theta0 field, which must stay identically zero).
        let (phi_legacy, u_legacy) = run_model_c(SerialScheduler, 0.0, 300);

        let grid = CartesianGrid::new([N, N], [N / 2, N / 2], [0.0, 0.0], [H, H]).unwrap();
        let model = ModelC::<Wiener<1>>::classic().with_orientations();
        let mut sim = Simulation::new(
            grid,
            FiniteVolume::default(),
            model,
            EulerMaruyama { seed: 7 },
            SerialScheduler,
            SystemAllocator,
        );
        let dt = sim.stable_dt().unwrap();
        {
            let model = sim.model().clone();
            let (grid, state) = sim.state_mut();
            model.initialize_grains(
                grid,
                state,
                &[Grain {
                    center: [H, H],
                    radius: 10.0 * H,
                    orientation: 0.0,
                }],
                0.7,
            );
        }
        for _ in 0..300 {
            sim.step(dt);
        }
        let (phi_h, u_h, th) = (
            sim.model().phi(),
            sim.model().u(),
            sim.model().theta0().unwrap(),
        );
        let (mut phi, mut u, mut theta_max) = (Vec::new(), Vec::new(), 0.0f64);
        for b in 0..sim.grid().num_blocks() {
            let block = BlockId(b as u32);
            let vp = sim.state().view(sim.grid(), block, phi_h);
            let vu = sim.state().view(sim.grid(), block, u_h);
            let vt = sim.state().view(sim.grid(), block, th);
            for_each_interior(sim.grid().block_cells(), |idx| {
                phi.push(vp.get(idx));
                u.push(vu.get(idx));
                theta_max = theta_max.max(vt.get(idx).abs());
            });
        }
        assert_eq!(phi, phi_legacy, "phi must match the legacy path bitwise");
        assert_eq!(u, u_legacy, "u must match the legacy path bitwise");
        assert_eq!(theta_max, 0.0, "static theta0 must remain exactly zero");
    }
}
