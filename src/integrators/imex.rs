use crate::{
    core::{
        scheduler::Scheduler,
        scratch::ScratchPool,
        state::State,
        storage::{Real, StorageBackend},
    },
    geometry::{
        cartesian::CartesianGrid,
        grid::{BlockId, Grid},
    },
    integrators::{Integrator, StageKind, StageLayout, StepCtx},
    physics::model::{Driver, Model, NoNoise, StiffRow},
};

/// First-order IMEX (forward–backward) Euler on a 1D Cartesian grid:
///
/// ```text
/// (I − dt·L)·Yⁿ⁺¹ = Yⁿ + dt·(V₀(Yⁿ) − L·Yⁿ)
/// ```
///
/// The model's stiff linear tridiagonal part `L` (declared via
/// [`Model::stiff_rows`]) is integrated implicitly by a Thomas solve per
/// field line, so the timestep is bounded by the *nonstiff remainder* alone
/// ([`Model::stable_dt_nonstiff`]) — for a diffusion-stiff model this breaks
/// the explicit `dt ∝ h²` wall entirely. The model still defines its
/// dynamics once, in [`Model::vector_field_block`]: the scheme evaluates the
/// full drift and subtracts `L·Yⁿ` itself, so the split cannot drift out of
/// sync with the physics.
///
/// Fields with no stiff rows take the plain explicit update; with no stiff
/// field anywhere the scheme reproduces
/// [`ForwardEuler`](crate::integrators::ForwardEuler) bit for bit.
/// Backward Euler on an upwind advection–diffusion operator is an M-matrix
/// solve: unconditionally stable and monotone.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImexEuler;

impl<D: Sync> Integrator<CartesianGrid<1>, D, NoNoise> for ImexEuler {
    fn stage_layout(&self, _grid: &CartesianGrid<1>) -> StageLayout {
        StageLayout {
            stages: vec![StageKind::Tendency(Driver::Time)],
        }
    }

    /// The nonstiff-remainder bound when the model declares one — the whole
    /// point of the scheme — falling back to the explicit bound otherwise.
    fn suggested_dt<M>(&self, model: &M, grid: &CartesianGrid<1>) -> Option<f64>
    where
        M: Model<CartesianGrid<1>, D, Drivers = NoNoise>,
    {
        model
            .stable_dt_nonstiff(grid.finest_spacing())
            .or_else(|| model.stable_dt(grid.finest_spacing()))
    }

    fn step<M, S, Sch>(
        &self,
        model: &M,
        grid: &CartesianGrid<1>,
        disc: &D,
        scheduler: &Sch,
        pool: &ScratchPool<M::Scalar, S>,
        state: &mut State<M::Scalar, S>,
        stages: &mut [State<M::Scalar, S>],
        t: f64,
        dt: f64,
    ) where
        M: Model<CartesianGrid<1>, D, Drivers = NoNoise>,
        S: StorageBackend<M::Scalar>,
        Sch: Scheduler,
    {
        let ctx = StepCtx {
            model,
            grid,
            disc,
            scheduler,
            pool,
        };
        let drift = &mut stages[0];
        // Full drift V₀(Yⁿ) — ghosts filled, model evaluated once, exactly
        // as in the explicit schemes.
        ctx.eval_drift(state, drift, t);

        let bc = grid.block_cells()[0];
        let ncells = grid.cells()[0];
        let nb = grid.num_blocks();

        // Line workspaces (a few line-length vectors per step — the 1D
        // implicit solve is global across blocks, so these are not
        // block-shaped stage buffers; at line lengths the allocation is
        // noise next to the drift evaluation).
        let mut rows: Vec<StiffRow> = vec![[0.0; 3]; ncells];
        let mut line = vec![0.0f64; ncells];
        let mut rhs = vec![0.0f64; ncells];
        let mut cp = vec![0.0f64; ncells];

        let num_fields = state.layout().num_fields();
        for fi in 0..num_fields {
            if !state.layout().specs()[fi].is_driven_by(Driver::Time) {
                continue;
            }
            let handle = State::<M::Scalar, S>::handle_at(fi);

            // Gather this field's stiff rows, block by block.
            let mut any_stiff = false;
            for b in 0..nb {
                let seg = &mut rows[b * bc..(b + 1) * bc];
                if model.stiff_rows(grid, BlockId(b as u32), handle, seg) {
                    any_stiff = true;
                } else {
                    seg.fill([0.0; 3]);
                }
            }

            if !any_stiff {
                // No stiff part: the plain forward-Euler update (interior
                // cells; ghosts are refilled before any read).
                let scale = M::Scalar::from_f64(dt);
                for b in 0..nb {
                    let block = BlockId(b as u32);
                    let kv = drift.view(grid, block, handle);
                    let mut uv = state.view_mut(grid, block, handle);
                    for i in 0..bc {
                        let idx = [i as isize];
                        uv.set(idx, uv.get(idx) + scale * kv.get(idx));
                    }
                }
                continue;
            }
            assert!(
                !grid.periodic()[0],
                "ImexEuler's tridiagonal solve does not support a periodic dimension \
                 (the matrix would be cyclic)"
            );

            // Gather Yⁿ and build rhs = Yⁿ + dt·(V₀(Yⁿ) − L·Yⁿ).
            for b in 0..nb {
                let uv = state.view(grid, BlockId(b as u32), handle);
                for i in 0..bc {
                    line[b * bc + i] = uv.get([i as isize]).to_f64();
                }
            }
            for b in 0..nb {
                let kv = drift.view(grid, BlockId(b as u32), handle);
                for i in 0..bc {
                    let gc = b * bc + i;
                    let lo = if gc > 0 { line[gc - 1] } else { 0.0 };
                    let hi = if gc + 1 < ncells { line[gc + 1] } else { 0.0 };
                    let lu =
                        rows[gc][0].mul_add(lo, rows[gc][1].mul_add(line[gc], rows[gc][2] * hi));
                    rhs[gc] = dt.mul_add(kv.get([i as isize]).to_f64() - lu, line[gc]);
                }
            }

            // Thomas solve of (I − dt·L)·x = rhs. The upwind/diffusion sign
            // structure makes I − dt·L strictly diagonally dominant, so no
            // pivoting is needed.
            let denom0 = dt.mul_add(-rows[0][1], 1.0);
            cp[0] = -dt * rows[0][2] / denom0;
            rhs[0] /= denom0;
            for i in 1..ncells {
                let sub = -dt * rows[i][0];
                let pivot = sub.mul_add(-cp[i - 1], dt.mul_add(-rows[i][1], 1.0));
                cp[i] = -dt * rows[i][2] / pivot;
                rhs[i] = sub.mul_add(-rhs[i - 1], rhs[i]) / pivot;
            }
            for i in (0..ncells - 1).rev() {
                rhs[i] = cp[i].mul_add(-rhs[i + 1], rhs[i]);
            }

            // Scatter Yⁿ⁺¹.
            for b in 0..nb {
                let mut uv = state.view_mut(grid, BlockId(b as u32), handle);
                for i in 0..bc {
                    uv.set([i as isize], M::Scalar::from_f64(rhs[b * bc + i]));
                }
            }
        }
    }
}
