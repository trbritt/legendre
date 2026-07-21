use crate::{
    core::{
        driver::Driver, scheduler::Scheduler, scratch::ScratchPool, state::State,
        storage::StorageBackend,
    },
    geometry::grid::Grid,
    integrators::{Integrator, StageKind, StageLayout, eval_drift, eval_tendency},
    physics::model::{DriverSet, Model},
};

/// Euler–Maruyama: `Y ← Y + dt·V₀(Y, t) + Σ_d dμ_d ∘ V_d(Y, t)`.
///
/// Implemented for every driver set, and *kind-agnostic*: it evaluates one
/// amplitude field per driver at the pre-update state (Itô) and applies
/// the whole update through the drivers' own kernels
/// ([`crate::core::driver::DriverKind`]) in a single fused dispatch, so
/// any first-order explicit measure composes here without a new scheme.
/// With [`NoNoise`](crate::physics::model::NoNoise) it degenerates to
/// forward Euler exactly.
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerMaruyama {
    /// Seed of the counter-based noise generator.
    pub seed: u64,
}

impl<G: Grid, D: Sync, N: DriverSet> Integrator<G, D, N> for EulerMaruyama {
    fn stage_layout(&self, _grid: &G) -> StageLayout {
        let mut stages = Vec::with_capacity(1 + N::LEN);
        stages.push(StageKind::Tendency(Driver::Time));
        stages.extend((0..N::LEN).map(|i| StageKind::Tendency(N::driver(i))));
        StageLayout { stages }
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
        M: Model<G, D, Drivers = N>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let (k, stochastic) = stages.split_first_mut().expect("stage buffers");
        // Every field — drift and all amplitudes — is evaluated at the
        // pre-update state before any update is applied (Itô).
        eval_drift(model, grid, disc, scheduler, pool, state, k, t);
        for (i, amp) in stochastic.iter_mut().enumerate() {
            eval_tendency(
                model,
                grid,
                disc,
                scheduler,
                pool,
                state,
                amp,
                t,
                N::driver(i),
            );
        }
        // Drift + every driver applied in one dispatch, block-local and
        // cache-hot (see State::apply_step_with).
        state.apply_step_with::<G, N, Sch>(
            scheduler,
            grid,
            k,
            stochastic,
            dt,
            self.seed,
            t.to_bits(),
        );
    }
}
