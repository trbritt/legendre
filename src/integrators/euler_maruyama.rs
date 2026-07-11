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

/// Euler–Maruyama: `Y ← Y + dt·f(Y, t) + √dt·b(Y)∘ξ`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerMaruyama {
    /// Seed of the counter-based noise generator.
    pub seed: u64,
}

impl<G: Grid, D: Sync> Integrator<G, D> for EulerMaruyama {
    fn stage_layout(&self) -> StageLayout {
        StageLayout {
            tendency: 2, // drift + noise amplitude
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
        M: Model<G, D>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let (k, rest) = stages.split_first_mut().expect("stage buffers");
        eval_rhs(model, grid, disc, scheduler, pool, state, k, t);
        if model.has_noise() {
            eval_noise(model, grid, disc, scheduler, pool, state, &mut rest[0], t);
        }
        state.axpy_with(scheduler, M::Scalar::from_f64(dt), k);
        if model.has_noise() {
            state.add_noise_with(
                scheduler,
                &rest[0],
                M::Scalar::from_f64(dt.sqrt()),
                self.seed,
                t.to_bits(),
            );
        }
    }
}
