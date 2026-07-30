//! Stochastic-volatility Monte Carlo paths: a CIR variance driving an
//! arithmetic price, one independent path per grid cell.
//!
//! ```text
//! dν = k(θ − ν) dt + σ√ν dW₁      (CIR / Heston variance, mean-reverting)
//! dS = μ dt      +   √ν dW₂       (arithmetic midprice, stochastic vol)
//! ```
//!
//! This is a **0-dimensional SDE per cell** — cells do not couple, so the
//! grid is used purely as an embarrassingly-parallel ensemble of paths (one
//! path per interior cell). Fields therefore carry no ghosts, the model needs
//! no `fill_ghosts` and no CFL `stable_dt`, and it rides entirely on the
//! framework's generic stochastic machinery: `ν` is registered as driven by
//! the time driver and Wiener process 0, `S` by the time driver and Wiener
//! process 1, so [`crate::integrators::EulerMaruyama`] evaluates one
//! amplitude per driver at the pre-update state (Itô) and applies `√dt·ξ`
//! itself. The two Wiener processes are independent (no leverage
//! correlation); a correlated driver is a model-level extension (mix the two
//! increments in the amplitudes — the Cholesky factor lives in the model).
//!
//! **CIR positivity** is handled the standard way: full truncation (`ν⁺` in
//! the drift and diffusion coefficients) plus a reflecting [`Model::project`]
//! that maps any noise-induced excursion `ν < 0` back to `0` after each step.

use crate::{
    core::{
        scratch::Scratch,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::StorageBackend,
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    physics::model::{Driver, Model, RhsContext, Wiener},
};

/// Parameters of the [`StochVolPaths`] model.
#[derive(Debug, Clone, Copy)]
pub struct StochVolParams {
    /// Price drift μ.
    pub mu: f64,
    /// Variance mean-reversion speed k.
    pub k_speed: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Volatility of variance σ.
    pub sigma: f64,
}

impl Default for StochVolParams {
    fn default() -> Self {
        Self {
            mu: 0.0,
            k_speed: 1.0,
            theta: 1.0,
            sigma: 0.1,
        }
    }
}

/// A stochastic-volatility path ensemble (CIR variance + arithmetic price),
/// one path per grid cell. See the module docs for the dynamics.
///
/// Driven by two independent Wiener processes ([`Wiener<2>`]); pairs with any
/// stochastic integrator (e.g. [`crate::integrators::EulerMaruyama`]).
#[derive(Debug, Clone)]
pub struct StochVolPaths {
    /// Model parameters.
    pub params: StochVolParams,
    nu: Option<FieldHandle<f64>>,
    mid: Option<FieldHandle<f64>>,
}

impl StochVolPaths {
    /// A model with the given parameters; fields are registered by
    /// [`crate::core::simulation::Simulation::new`].
    #[must_use]
    pub const fn new(params: StochVolParams) -> Self {
        Self {
            params,
            nu: None,
            mid: None,
        }
    }

    /// Handle of the variance field ν.
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet.
    #[must_use]
    pub const fn nu(&self) -> FieldHandle<f64> {
        self.nu.expect("model fields not yet registered")
    }

    /// Handle of the midprice field S.
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet.
    #[must_use]
    pub const fn mid(&self) -> FieldHandle<f64> {
        self.mid.expect("model fields not yet registered")
    }

    /// Start every path at variance `nu0` and price `s0`.
    pub fn initialize<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<1>,
        state: &mut State<f64, S>,
        nu0: f64,
        s0: f64,
    ) {
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            for (handle, value) in [(self.nu(), nu0), (self.mid(), s0)] {
                let mut v = state.view_mut(grid, block, handle);
                for_each_interior(grid.block_cells(), |idx| v.set(idx, value));
            }
        }
    }
}

impl<D: Sync> Model<CartesianGrid<1>, D> for StochVolPaths {
    type Scalar = f64;
    type Drivers = Wiener<2>;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        // ν moves under the clock and Wiener 0; S under the clock and Wiener
        // 1. Ghost width 0: paths never read a neighbour.
        self.nu = Some(builder.register_driven("nu", 0, &[Driver::Time, Driver::Wiener(0)]));
        self.mid = Some(builder.register_driven("mid", 0, &[Driver::Time, Driver::Wiener(1)]));
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        driver: Driver,
        ctx: &RhsContext<'_, CartesianGrid<1>, D>,
        state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        let p = &self.params;
        let (grid, block) = (ctx.grid, ctx.block);
        let nu = state.view(grid, block, self.nu());
        match driver {
            // Drift of both fields (full-truncation ν⁺ in the CIR drift).
            Driver::Time => {
                let mut dnu = out.view_mut(grid, block, self.nu());
                for_each_interior(nu.interior(), |idx| {
                    dnu.set(idx, p.k_speed * (p.theta - nu.get(idx).max(0.0)));
                });
                let mut dmid = out.view_mut(grid, block, self.mid());
                for_each_interior(nu.interior(), |idx| dmid.set(idx, p.mu));
            }
            // Variance diffusion amplitude σ√ν⁺ (this buffer carries ν only).
            Driver::Wiener(0) => {
                let mut amp = out.view_mut(grid, block, self.nu());
                for_each_interior(nu.interior(), |idx| {
                    amp.set(idx, p.sigma * nu.get(idx).max(0.0).sqrt());
                });
            }
            // Price diffusion amplitude √ν⁺ (this buffer carries S only).
            Driver::Wiener(1) => {
                let mut amp = out.view_mut(grid, block, self.mid());
                for_each_interior(nu.interior(), |idx| {
                    amp.set(idx, nu.get(idx).max(0.0).sqrt())
                });
            }
            Driver::Wiener(_) => unreachable!("StochVolPaths declares Wiener<2>"),
        }
    }

    /// CIR positivity: reflect any noise-induced excursion `ν < 0` back to 0.
    fn project<S: StorageBackend<f64>>(&self, grid: &CartesianGrid<1>, state: &mut State<f64, S>) {
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            let mut v = state.view_mut(grid, block, self.nu());
            for_each_interior(grid.block_cells(), |idx| v.set(idx, v.get(idx).max(0.0)));
        }
    }
}
