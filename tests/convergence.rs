//! Convergence-order verification on a nonlinear scalar ODE.
//!
//! The eigenmode tests pin the schemes *exactly* on linear problems; this
//! suite pins the *order* on a nonlinear one, where any mis-weighted stage
//! would show up as a wrong error ratio. The problem is `du/dt = −u³`,
//! `u(0) = 1`, whose exact solution is `u(t) = 1/√(1 + 2t)` — run on a
//! single-cell grid so the "PDE" is the ODE.

use legendre::{
    core::{
        scheduler::SerialScheduler,
        scratch::Scratch,
        simulation::Simulation,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::{StorageBackend, SystemAllocator},
    },
    geometry::{cartesian::CartesianGrid, grid::BlockId},
    integrators::{EulerMaruyama, ForwardEuler, Integrator, RungeKutta4},
    physics::model::{Model, RhsContext},
};

struct Cubic {
    u: Option<FieldHandle<f64>>,
}

impl<D: Sync> Model<CartesianGrid<1>, D> for Cubic {
    type Scalar = f64;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.u = Some(builder.register("u", 0)); // no stencil, no ghosts
    }

    fn rhs_block<S: StorageBackend<f64>>(
        &self,
        ctx: &RhsContext<'_, CartesianGrid<1>, D>,
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

fn run<I: Integrator<CartesianGrid<1>, ()>>(integrator: I, dt: f64, t_end: f64) -> f64 {
    let grid = CartesianGrid::new([1], [1], [0.0], [1.0]).unwrap();
    let mut sim = Simulation::new(
        grid,
        (),
        Cubic { u: None },
        integrator,
        SerialScheduler,
        SystemAllocator,
    );
    let u = sim.model().u.unwrap();
    {
        let (grid, state) = sim.state_mut();
        let mut v = state.view_mut(grid, BlockId(0), u);
        v.set([0], 1.0);
    }
    let steps = (t_end / dt).round() as u64;
    for _ in 0..steps {
        sim.step(dt);
    }
    sim.state().view(sim.grid(), BlockId(0), u).get([0])
}

fn error(integrator: impl Integrator<CartesianGrid<1>, ()>, dt: f64) -> f64 {
    const T_END: f64 = 0.5;
    let exact = 1.0 / 2.0f64.mul_add(T_END, 1.0).sqrt();
    (run(integrator, dt, T_END) - exact).abs()
}

/// Observed order p from halving dt: p = log2(e(dt)/e(dt/2)).
fn observed_order(integrator: impl Integrator<CartesianGrid<1>, ()> + Copy, dt: f64) -> f64 {
    (error(integrator, dt) / error(integrator, dt / 2.0)).log2()
}

#[test]
fn forward_euler_is_first_order() {
    let p = observed_order(ForwardEuler, 0.02);
    assert!((0.9..1.1).contains(&p), "expected order ≈ 1, got {p}");
}

#[test]
fn rk4_is_fourth_order() {
    let p = observed_order(RungeKutta4::default(), 0.02);
    assert!((3.7..4.3).contains(&p), "expected order ≈ 4, got {p}");
}

/// With no stochastic term declared, Euler–Maruyama degenerates to forward
/// Euler exactly.
#[test]
// Bitwise identity is the property under test.
#[allow(clippy::float_cmp)]
fn euler_maruyama_without_noise_is_forward_euler() {
    let em = run(EulerMaruyama { seed: 9 }, 0.01, 0.5);
    let fe = run(ForwardEuler, 0.01, 0.5);
    assert_eq!(em, fe);
}
