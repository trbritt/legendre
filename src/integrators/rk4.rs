use crate::{
    core::{
        scheduler::Scheduler,
        scratch::ScratchPool,
        state::State,
        storage::{Real, StorageBackend},
    },
    geometry::grid::Grid,
    integrators::{Integrator, StageLayout, eval_drift},
    physics::model::{Model, NoNoise},
};

/// Classic fourth-order Runge–Kutta.
///
/// Deterministic models only: a stochastic term would reduce the scheme to
/// weak first order anyway, so stochastic systems pair with
/// [`EulerMaruyama`](crate::integrators::EulerMaruyama) (or a future
/// higher-order stochastic scheme) instead — enforced at compile time by
/// the `NoNoise` driver set.
#[derive(Debug, Clone, Copy, Default)]
pub struct RungeKutta4;

impl<G: Grid, D: Sync> Integrator<G, D, NoNoise> for RungeKutta4 {
    fn stage_layout(&self) -> StageLayout {
        StageLayout {
            tendency: 4,    // k1..k4
            stage_state: 1, // y_tmp
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
        M: Model<G, D, Noise = NoNoise>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let half_dt = M::Scalar::from_f64(0.5 * dt);
        let (k1, rest) = stages.split_first_mut().expect("stage buffers");
        let (k2, rest) = rest.split_first_mut().expect("stage buffers");
        let (k3, rest) = rest.split_first_mut().expect("stage buffers");
        let (k4, rest) = rest.split_first_mut().expect("stage buffers");
        let y_tmp = &mut rest[0];

        eval_drift(model, grid, disc, scheduler, pool, state, k1, t);

        y_tmp.copy_from_with(scheduler, state);
        y_tmp.axpy_with(scheduler, half_dt, k1);
        eval_drift(
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
        eval_drift(
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
        eval_drift(model, grid, disc, scheduler, pool, y_tmp, k4, t + dt);

        let sixth = M::Scalar::from_f64(dt / 6.0);
        let third = M::Scalar::from_f64(dt / 3.0);
        state.axpy_with(scheduler, sixth, k1);
        state.axpy_with(scheduler, third, k2);
        state.axpy_with(scheduler, third, k3);
        state.axpy_with(scheduler, sixth, k4);
    }
}
