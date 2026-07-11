use crate::{
    core::{
        scheduler::Scheduler,
        scratch::ScratchPool,
        state::State,
        storage::{Real, StorageBackend},
    },
    geometry::grid::Grid,
    integrators::{Integrator, StageLayout, eval_noise, eval_rhs},
    physics::model::Model,
};

/// Classic fourth-order Runge–Kutta.
///
/// Deterministic drift at O(dt⁴); if the model declares noise,
/// `√dt·b(Y)∘ξ` is added after the drift update (amplitude evaluated at the
/// pre-update state), which preserves the weak Euler–Maruyama treatment of
/// the stochastic term.
#[derive(Debug, Clone, Copy, Default)]
pub struct RungeKutta4 {
    /// Seed of the counter-based noise generator (stochastic models only).
    pub seed: u64,
}

impl<G: Grid, D: Sync> Integrator<G, D> for RungeKutta4 {
    fn stage_layout(&self) -> StageLayout {
        StageLayout {
            tendency: 4,    // k1..k4
            stage_state: 1, // y_tmp (also reused as the noise amplitude)
        }
    }

    fn step<M, S, Sch>(
        &self,
        model: &M,
        grid: &G,
        disc: &D,
        scheduler: &Sch,
        pool: &ScratchPool<M::Scalar, S>,
        state: &mut State<M::Scalar, S>,
        stages: &mut [State<M::Scalar, S>],
        t: f64,
        dt: f64,
    ) where
        M: Model<G, D>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let half_dt = M::Scalar::from_f64(0.5 * dt);
        let (k1, rest) = stages.split_first_mut().expect("stage buffers");
        let (k2, rest) = rest.split_first_mut().expect("stage buffers");
        let (k3, rest) = rest.split_first_mut().expect("stage buffers");
        let (k4, rest) = rest.split_first_mut().expect("stage buffers");
        let y_tmp = &mut rest[0];

        eval_rhs(model, grid, disc, scheduler, pool, state, k1, t);

        y_tmp.copy_from_with(scheduler, state);
        y_tmp.axpy_with(scheduler, half_dt, k1);
        eval_rhs(
            model,
            grid,
            disc,
            scheduler,
            pool,
            y_tmp,
            k2,
            0.5f64.mul_add(dt, t),
        );

        y_tmp.copy_from_with(scheduler, state);
        y_tmp.axpy_with(scheduler, half_dt, k2);
        eval_rhs(
            model,
            grid,
            disc,
            scheduler,
            pool,
            y_tmp,
            k3,
            0.5f64.mul_add(dt, t),
        );

        y_tmp.copy_from_with(scheduler, state);
        y_tmp.axpy_with(scheduler, M::Scalar::from_f64(dt), k3);
        eval_rhs(model, grid, disc, scheduler, pool, y_tmp, k4, t + dt);

        // y_tmp is free now; reuse it for the noise amplitude, evaluated at
        // the pre-update state before the drift combine mutates it.
        if model.has_noise() {
            eval_noise(model, grid, disc, scheduler, pool, state, y_tmp, t);
        }

        let sixth = M::Scalar::from_f64(dt / 6.0);
        let third = M::Scalar::from_f64(dt / 3.0);
        state.axpy_with(scheduler, sixth, k1);
        state.axpy_with(scheduler, third, k2);
        state.axpy_with(scheduler, third, k3);
        state.axpy_with(scheduler, sixth, k4);

        if model.has_noise() {
            state.add_noise_with(
                scheduler,
                y_tmp,
                M::Scalar::from_f64(dt.sqrt()),
                self.seed,
                t.to_bits(),
            );
        }
    }
}
