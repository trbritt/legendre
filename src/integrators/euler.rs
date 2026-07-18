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

/// Explicit forward Euler: `Y ← Y + dt·f(Y, t)`.
///
/// Deterministic models only; for a stochastic system use
/// [`EulerMaruyama`](crate::integrators::EulerMaruyama), which degenerates
/// to this scheme when the model has no Wiener drivers.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardEuler;

impl<G: Grid, D: Sync> Integrator<G, D, NoNoise> for ForwardEuler {
    fn stage_layout(&self) -> StageLayout {
        StageLayout {
            tendency: 1,
            stage_state: 0,
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
        let k = &mut stages[0];
        eval_drift(model, grid, disc, scheduler, pool, state, k, t);
        state.axpy_with(scheduler, M::Scalar::from_f64(dt), k);
    }
}
