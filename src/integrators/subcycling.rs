//! Berger–Oliger subcycled time integration for [`AmrGrid`] hierarchies.
//!
//! Coarse levels take large steps; each finer level takes several small
//! substeps within one coarse step, so the finest grid's restrictive `dt`
//! is never imposed on the whole domain. For a **parabolic** problem
//! (`dt ∝ h²`, every diffusion-dominated model legendre ships) a level of
//! spatial ratio `r` takes `r²` substeps per parent step.
//!
//! One coarse step is the recursion
//!
//! ```text
//! advance(level, t, dt):
//!   snapshot level              # its state at t, for children's time interp
//!   fill level's ghosts         # same-level + physical; coarse source
//!                               #   interpolated in time from the parent
//!   step level by dt            # Euler–Maruyama, restricted to this level
//!   if a finer level exists:
//!     for s in 0..r²:           # child substeps
//!       advance(level+1, t + s·dt/r², dt/r²)   # coarse source at α = s/r²
//!     restrict(level+1 → level) # fine solution wins beneath the patch
//! ```
//!
//! `Subcycling::step(dt)` advances the whole hierarchy by one **level-0**
//! step `dt`; `Simulation::stable_dt()` returns the right coarse dt for it
//! (via [`Integrator::suggested_dt`](crate::integrators::Integrator::suggested_dt),
//! the coarsest level's stable dt), and the per-level substep counts are
//! derived from the model's stability law at each level's spacing.
//!
//! **v1 scope.** The scheme is fixed to Euler–Maruyama (forward Euler for
//! `NoNoise`); regridding happens only at level-0 (synchronized)
//! boundaries via the usual [`Adapt`](crate::core::simulation::Adapt)
//! policy; boundary conditions are the standard AMR mirror. Non-commutative
//! multi-Wiener corrections, arbitrary base schemes, and finer-cadence
//! regridding are future work.

use crate::{
    core::{
        driver::Driver, scheduler::Scheduler, scratch::ScratchPool, state::FieldHandle,
        state::State, storage::StorageBackend,
    },
    geometry::amr::{self, AmrGrid},
    integrators::{Integrator, StageKind, StageLayout, eval_drift_level, eval_tendency_level},
    physics::model::{DriverSet, Model},
};

/// Berger–Oliger subcycling integrator; see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Subcycling {
    /// Seed of the counter-based noise generator.
    pub seed: u64,
}

impl<D2: Sync, N: DriverSet, const D: usize> Integrator<AmrGrid<D>, D2, N> for Subcycling {
    fn suggested_dt<M>(&self, model: &M, grid: &AmrGrid<D>) -> Option<f64>
    where
        M: Model<AmrGrid<D>, D2, Drivers = N>,
    {
        // The level-0 (coarsest) stable dt; finer levels are refined in
        // time per level inside `step`.
        model.stable_dt(grid.spacing_at_level(0))
    }

    fn stage_layout(&self, grid: &AmrGrid<D>) -> StageLayout {
        // Drift + one amplitude per Wiener driver, then one
        // time-interpolation snapshot per level. `max_levels` (not the
        // currently populated count) keeps the buffer count — and so the
        // allocation — stable across regrids.
        let mut stages = Vec::with_capacity(1 + N::LEN + grid.max_levels());
        stages.push(StageKind::Tendency(Driver::Time));
        stages.extend((0..N::LEN).map(|i| StageKind::Tendency(N::driver(i))));
        stages.extend(std::iter::repeat_n(StageKind::State, grid.max_levels()));
        StageLayout { stages }
    }

    fn step<M, S, Sch>(
        &self,
        model: &M,
        grid: &AmrGrid<D>,
        disc: &D2,
        scheduler: &Sch,
        pool: &ScratchPool<M::Scalar, S>,
        state: &mut State<M::Scalar, S>,
        stages: &mut [State<M::Scalar, S>],
        t: f64,
        dt: f64,
    ) where
        M: Model<AmrGrid<D>, D2, Drivers = N>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let (tendency, snapshots) = stages.split_at_mut(1 + N::LEN);
        let (drift, amps) = tendency.split_first_mut().expect("drift stage buffer");

        // Every dynamic field is filled/restricted; ghost-0 fields (e.g. a
        // static orientation map) no-op inside the transfers.
        let fields: Vec<FieldHandle<M::Scalar>> = state
            .layout()
            .specs()
            .iter()
            .enumerate()
            .filter(|(_, spec)| !spec.is_static())
            .map(|(i, _)| State::<M::Scalar, S>::handle_at(i))
            .collect();

        // One ghost-fill plan per distinct ghost width, built once for this
        // coarse step and reused across every substep's fill — the
        // per-cell find_patch/interpolation work happens once, not per
        // fill. Paired with each field's plan index.
        let mut widths: Vec<u32> = fields
            .iter()
            .map(|h| state.layout().ghost(h.index()))
            .collect();
        widths.sort_unstable();
        widths.dedup();
        let plans: Vec<amr::FillPlan<D>> = widths
            .iter()
            .map(|&w| amr::build_fill_plan(grid, w))
            .collect();
        let field_plan: Vec<(FieldHandle<M::Scalar>, usize)> = fields
            .iter()
            .map(|&h| {
                let w = state.layout().ghost(h.index());
                (h, widths.iter().position(|&x| x == w).unwrap())
            })
            .collect();

        advance(
            self.seed,
            model,
            grid,
            disc,
            scheduler,
            pool,
            state,
            drift,
            amps,
            snapshots,
            &field_plan,
            &plans,
            0,
            t,
            dt,
            0.0,
        );
    }
}

/// Advance level `level` by `dt`, starting at time `t`; `alpha` is the
/// fraction of the *parent's* step already elapsed at `t`, used to
/// interpolate this level's coarse ghost source in time. Recurses into
/// finer levels (subcycling them) and restricts them back on completion.
#[allow(clippy::too_many_arguments)]
fn advance<M, S, Sch, D2, const D: usize, N>(
    seed: u64,
    model: &M,
    grid: &AmrGrid<D>,
    disc: &D2,
    scheduler: &Sch,
    pool: &ScratchPool<M::Scalar, S>,
    state: &mut State<M::Scalar, S>,
    drift: &mut State<M::Scalar, S>,
    amps: &mut [State<M::Scalar, S>],
    snapshots: &mut [State<M::Scalar, S>],
    field_plan: &[(FieldHandle<M::Scalar>, usize)],
    plans: &[amr::FillPlan<D>],
    level: u8,
    t: f64,
    dt: f64,
    alpha: f64,
) where
    M: Model<AmrGrid<D>, D2, Drivers = N>,
    S: StorageBackend<M::Scalar>,
    Sch: Scheduler,
    D2: Sync,
    N: DriverSet,
{
    let l = level as usize;

    // 1. Snapshot this level's pre-step state — the coarse source our
    //    children interpolate from in time.
    snapshots[l].copy_from_with(scheduler, state);

    // 2. Fill this level's ghosts. Level 0 has no coarser parent; finer
    //    levels interpolate their prolongation source between the parent
    //    snapshot (at α) and the parent's post-step state.
    for &(h, pi) in field_plan {
        let interp = (level > 0).then(|| (&snapshots[l - 1] as &State<M::Scalar, S>, alpha));
        amr::fill_level(grid, state, interp, h, level, &plans[pi]);
    }

    // 3. Step this level by dt (Euler–Maruyama, restricted to its blocks).
    let salt = t.to_bits();
    eval_drift_level(model, grid, disc, scheduler, pool, state, drift, t, level);
    for (i, amp) in amps.iter_mut().enumerate() {
        eval_tendency_level(
            model,
            grid,
            disc,
            scheduler,
            pool,
            state,
            amp,
            t,
            N::driver(i),
            level,
        );
    }
    state.apply_step_level::<AmrGrid<D>, N, Sch>(
        scheduler, grid, drift, amps, dt, seed, salt, level,
    );

    // 4. Subcycle the next finer level, if present. The substep count is
    //    the ratio of the model's stability limits at the two spacings —
    //    r² for a parabolic model, r for hyperbolic, derived not assumed.
    let child = level + 1;
    if (child as usize) < grid.num_levels() {
        let dt_coarse = model
            .stable_dt(grid.spacing_at_level(level))
            .expect("subcycling requires Model::stable_dt");
        let dt_fine = model
            .stable_dt(grid.spacing_at_level(child))
            .expect("subcycling requires Model::stable_dt");
        let n = (dt_coarse / dt_fine).round().max(1.0) as u64;
        let dt_child = dt / n as f64;
        for s in 0..n {
            let child_t = (s as f64).mul_add(dt_child, t);
            let child_alpha = s as f64 / n as f64;
            advance(
                seed,
                model,
                grid,
                disc,
                scheduler,
                pool,
                state,
                drift,
                amps,
                snapshots,
                field_plan,
                plans,
                child,
                child_t,
                dt_child,
                child_alpha,
            );
        }
        // 5. Fine solution wins beneath the patch.
        for &(h, _) in field_plan {
            amr::restrict_level(grid, state, h, child);
        }
    }
}
