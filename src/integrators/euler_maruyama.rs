use crate::{
    core::{
        scheduler::Scheduler,
        scratch::ScratchPool,
        state::State,
        storage::{Real, StorageBackend},
    },
    geometry::grid::Grid,
    integrators::{Integrator, StageLayout, eval_drift, eval_wiener},
    physics::model::{Model, NoiseSpec},
};

/// Euler–Maruyama: `Y ← Y + dt·V₀(Y, t) + Σⱼ √dt·Vⱼ(Y, t)∘ξⱼ`.
///
/// Implemented for every driver set: with `M` Wiener drivers it evaluates
/// all `M` amplitude fields at the pre-update state (Itô) and applies each
/// with its own per-cell increment stream; with
/// [`NoNoise`](crate::physics::model::NoNoise) it degenerates to forward
/// Euler exactly.
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerMaruyama {
    /// Seed of the counter-based noise generator.
    pub seed: u64,
}

impl<G: Grid, D: Sync, N: NoiseSpec> Integrator<G, D, N> for EulerMaruyama {
    fn stage_layout(&self) -> StageLayout {
        StageLayout {
            tendency: 1 + N::WIENER_DIM, // drift + one amplitude per driver
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
        M: Model<G, D, Noise = N>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let (k, wiener) = stages.split_first_mut().expect("stage buffers");
        // Every field — drift and all amplitudes — is evaluated at the
        // pre-update state before any update is applied (Itô).
        eval_drift(model, grid, disc, scheduler, pool, state, k, t);
        for (j, amp) in wiener.iter_mut().enumerate() {
            eval_wiener(model, grid, disc, scheduler, pool, state, amp, t, j);
        }
        state.axpy_with(scheduler, M::Scalar::from_f64(dt), k);
        let sqrt_dt = M::Scalar::from_f64(dt.sqrt());
        for (j, amp) in wiener.iter().enumerate() {
            state.add_wiener_with(scheduler, grid, amp, sqrt_dt, self.seed, t.to_bits(), j);
        }
    }
}
