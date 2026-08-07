//! Optimal market making under stochastic volatility
//! (Cartea–Jaimungal-style limit-order-book models).
//!
//! A market maker quotes bid/ask depths δ± around a midprice whose variance
//! ν follows a CIR process. Market orders lift the quotes at exponential
//! intensities `λ·e^(−κδ)`, moving the (bounded, discrete) inventory
//! `q ∈ [q_min, q_max]` by ±1. With exponential intensities the Hamiltonian's
//! optimization over δ has a closed form, and after the standard ansatz the
//! HJB for the value function u(t, ν, q) reduces to, in backward time
//! τ = T − t:
//!
//! ```text
//! ∂u/∂τ = μq − ψσ²νq² + k(θ−ν)·∂νu + ½σ²ν·∂ννu + Hˢ(u; q) + Hᵇ(u; q)
//! ```
//!
//! with terminal condition u(τ = 0) = 0, mean-reversion speed k toward the
//! long-run variance θ, vol-of-vol σ, inventory drift μq, running inventory
//! penalty ψ, and one optimized fill term per side:
//!
//! ```text
//! Hˢ = λˢ(qεˢ + e^(−κˢ(εˢ + α(1−2q) − u(q−1) + u(q)))/κˢ)   q > q_min,
//! Hᵇ = λᵇ(−qεᵇ + e^(−κᵇ(εᵇ + α(1+2q) − u(q+1) + u(q)))/κᵇ)  q < q_max,
//! ```
//!
//! degenerating to `λˢqεˢ` / `−λᵇqεᵇ` at the inventory bounds where the
//! side is not quoted (α is the terminal liquidation penalty, ε± the fee
//! parameters).
//!
//! **Discretization.** ν lives on a [`CartesianGrid<1>`]; the discrete
//! inventory is **one field per q level** — the q-coupling above is ordinary
//! cross-field reads, exactly like any coupled multi-field model. The
//! ν-drift is upwinded on the sign of (θ−ν), the diffusion is central, and
//! the ν boundaries use linear extrapolation
//! ([`fill_ghosts_extrapolate`]), all inside the [`Driver::Time`] vector
//! field (an upwind stencil family for the policy layer is roadmap; this
//! model predates it). The system is deterministic — the HJB is a PDE — so
//! the driver set is [`NoNoise`]; the *stochasticity* lives in the controlled
//! path simulation that consumes the optimal depths (see the `market_making`
//! example).
//!
//! **Side conventions** (the classic trap — fixed here by naming):
//!
//! | side           | fill moves q | intensity params | depth from    |
//! |----------------|--------------|------------------|---------------|
//! | `sell` (ask)   | q → q−1      | λˢ, κˢ, εˢ       | u(q−1) − u(q) |
//! | `buy`  (bid)   | q → q+1      | λᵇ, κᵇ, εᵇ       | u(q+1) − u(q) |
//!
//! Optimal depths (used to drive a path simulation of the controlled
//! system; see the `market_making` example):
//!
//! ```text
//! δᵇ = 1/κᵇ + εᵇ + α(1+2q) − u(q+1) + u(q)     (∞ at q = q_max)
//! δˢ = 1/κˢ + εˢ + α(1−2q) − u(q−1) + u(q)     (∞ at q = q_min)
//! ```
//!
//! both clamped from below by `min_spread`.

use crate::{
    core::{
        driver::WienerJump,
        scratch::Scratch,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::StorageBackend,
    },
    geometry::{
        cartesian::{CartesianGrid, fill_ghosts_extrapolate, for_each_interior},
        grid::{BlockId, Grid},
    },
    physics::model::{Driver, Model, NoNoise, RhsContext, StiffRow},
};
use std::sync::Arc;

/// Intensity parameters of one quoting side.
#[derive(Debug, Clone, Copy)]
pub struct Side {
    /// Baseline market-order arrival rate λ.
    pub lambda: f64,
    /// Fill-probability decay κ per unit of quoted depth.
    pub kappa: f64,
    /// Fee / adverse-selection parameter ε.
    pub epsilon: f64,
}

impl Side {
    /// Fill intensity at quoted depth δ: `λ·e^(−κδ)` (0 for an infinite —
    /// i.e. withdrawn — quote).
    #[must_use]
    pub fn fill_rate(&self, depth: f64) -> f64 {
        if depth.is_finite() {
            self.lambda * (-self.kappa * depth).exp()
        } else {
            0.0
        }
    }
}

/// Parameters of the stochastic-volatility market-making model.
#[derive(Debug, Clone, Copy)]
pub struct MarketMakerParams {
    /// Midprice drift per unit inventory μ.
    pub mu: f64,
    /// Running inventory penalty ψ (weights σ²νq²).
    pub psi: f64,
    /// Terminal liquidation penalty α.
    pub alpha: f64,
    /// Variance mean-reversion speed k.
    pub k_speed: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Volatility of variance σ.
    pub sigma: f64,
    /// Ask side: fills decrease inventory.
    pub sell: Side,
    /// Bid side: fills increase inventory.
    pub buy: Side,
    /// Minimum inventory (selling forbidden here).
    pub q_min: isize,
    /// Maximum inventory (buying forbidden here).
    pub q_max: isize,
    /// Floor on quoted depths.
    pub min_spread: f64,
    /// Variance-domain lower edge — the ν-extent over which the explicit
    /// scheme's CFL bound is taken ([`Model::stable_dt`]). The grid this
    /// model runs on must span `[nu_min, nu_max]`.
    pub nu_min: f64,
    /// Variance-domain upper edge (see [`Self::nu_min`]).
    pub nu_max: f64,
}

impl Default for MarketMakerParams {
    /// The reference parameterization: T-horizon-agnostic model constants
    /// for a liquid book with mild inventory aversion.
    fn default() -> Self {
        Self {
            mu: 0.01,
            psi: 0.1,
            alpha: 0.01,
            k_speed: 1.0,
            theta: 1.0,
            sigma: 0.1,
            sell: Side {
                lambda: 2.0,
                kappa: 2.0,
                epsilon: 0.004,
            },
            buy: Side {
                lambda: 1.0,
                kappa: 2.0,
                epsilon: 0.004,
            },
            q_min: -5,
            q_max: 5,
            min_spread: 1e-4,
            nu_min: 0.001,
            nu_max: 2.0,
        }
    }
}

impl MarketMakerParams {
    /// Number of inventory levels (fields) this model carries.
    #[must_use]
    pub const fn num_levels(&self) -> usize {
        (self.q_max - self.q_min + 1) as usize
    }
}

/// The HJB model: value function u(τ, ν, q).
///
/// Lives on a 1D variance grid with one field per inventory level,
/// integrated **forward in τ = T − t** from the zero terminal condition (a
/// freshly built state — no initialization needed). Deterministic
/// (`Drivers = NoNoise`). See the module docs for the equation and
/// conventions.
#[derive(Debug, Clone)]
pub struct HjbMarketMaker {
    /// Model parameters.
    pub params: MarketMakerParams,
    handles: Vec<FieldHandle<f64>>,
}

impl HjbMarketMaker {
    /// A model with the given parameters; fields are registered by
    /// [`crate::core::simulation::Simulation::new`].
    #[must_use]
    pub const fn new(params: MarketMakerParams) -> Self {
        Self {
            params,
            handles: Vec::new(),
        }
    }

    /// Field handles in inventory order (index 0 ⇔ `q_min`).
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet.
    #[must_use]
    pub fn handles(&self) -> &[FieldHandle<f64>] {
        assert!(!self.handles.is_empty(), "model fields not yet registered");
        &self.handles
    }

    /// Handle of the value field at inventory `q`.
    ///
    /// # Panics
    ///
    /// Panics if `q` is outside `[q_min, q_max]` or fields are unregistered.
    #[must_use]
    pub fn handle(&self, q: isize) -> FieldHandle<f64> {
        self.handles()[(q - self.params.q_min) as usize]
    }

    /// Optimal quoted depths `(δᵇ bid, δˢ ask)` at inventory `q` from the
    /// value at this level (`u_q`) and its inventory neighbors (`u_up` =
    /// u(q+1), `u_dn` = u(q−1); `None` at the respective bound, where the
    /// side is withdrawn and the depth is `∞`). Clamped by `min_spread`.
    #[must_use]
    pub fn optimal_depths(
        &self,
        q: isize,
        u_q: f64,
        u_up: Option<f64>,
        u_dn: Option<f64>,
    ) -> (f64, f64) {
        let p = &self.params;
        let qf = q as f64;
        let bid = u_up.map_or(f64::INFINITY, |uu| {
            (p.alpha
                .mul_add(2.0f64.mul_add(qf, 1.0), 1.0 / p.buy.kappa + p.buy.epsilon)
                - uu
                + u_q)
                .max(p.min_spread)
        });
        let ask = u_dn.map_or(f64::INFINITY, |ud| {
            (p.alpha.mul_add(
                (-2.0f64).mul_add(qf, 1.0),
                1.0 / p.sell.kappa + p.sell.epsilon,
            ) - ud
                + u_q)
                .max(p.min_spread)
        });
        (bid, ask)
    }
}

impl<D: Sync> Model<CartesianGrid<1>, D> for HjbMarketMaker {
    type Scalar = f64;
    type Drivers = NoNoise;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        // One field per inventory level. Field names are `&'static str`, so
        // the runtime-sized set leaks its (tiny, setup-time-only) names.
        self.handles = (self.params.q_min..=self.params.q_max)
            .map(|q| {
                let name: &'static str = Box::leak(format!("u_q{q}").into_boxed_str());
                builder.register(name, 1)
            })
            .collect();
    }

    fn fill_ghosts<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<1>,
        state: &mut State<f64, S>,
        _t: f64,
    ) {
        for &h in &self.handles {
            fill_ghosts_extrapolate(grid, state, h);
        }
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        _driver: Driver, // NoNoise ⇒ always `Driver::Time` (the HJB drift)
        ctx: &RhsContext<'_, CartesianGrid<1>, D>,
        state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        let p = &self.params;
        let (grid, block) = (ctx.grid, ctx.block);
        let dnu = grid.spacing(block)[0];
        let inv_dnu = 1.0 / dnu;
        let inv_dnu2 = inv_dnu * inv_dnu;
        let sigma2 = p.sigma * p.sigma;

        for (qi, &h) in self.handles.iter().enumerate() {
            let q = p.q_min + qi as isize;
            let qf = q as f64;
            let u = state.view(grid, block, h);
            let up = (q < p.q_max).then(|| state.view(grid, block, self.handles[qi + 1]));
            let dn = (q > p.q_min).then(|| state.view(grid, block, self.handles[qi - 1]));
            let mut du = out.view_mut(grid, block, h);

            for_each_interior(u.interior(), |idx| {
                let [i] = idx;
                let nu = grid.cell_center(block, idx)[0];
                let u_c = u.get(idx);
                let u_p = u.get([i + 1]);
                let u_m = u.get([i - 1]);

                // Upwinded mean-reversion drift + central diffusion in ν.
                let drift_nu = p.k_speed * (p.theta - nu);
                let d_nu = if drift_nu > 0.0 {
                    (u_p - u_c) * inv_dnu
                } else {
                    (u_c - u_m) * inv_dnu
                };
                let dd_nu = 2.0f64.mul_add(-u_c, u_p + u_m) * inv_dnu2;

                // Optimized fill terms (module docs); withdrawn at bounds.
                let sell = dn
                    .as_ref()
                    .map_or(p.sell.lambda * qf * p.sell.epsilon, |dn| {
                        let arg = p.alpha.mul_add((-2.0f64).mul_add(qf, 1.0), p.sell.epsilon)
                            - dn.get(idx)
                            + u_c;
                        p.sell.lambda
                            * (qf.mul_add(
                                p.sell.epsilon,
                                (-p.sell.kappa * arg).exp() / p.sell.kappa,
                            ))
                    });
                let buy = up
                    .as_ref()
                    .map_or(-p.buy.lambda * qf * p.buy.epsilon, |up| {
                        let arg = p.alpha.mul_add(2.0f64.mul_add(qf, 1.0), p.buy.epsilon)
                            - up.get(idx)
                            + u_c;
                        p.buy.lambda
                            * ((-qf)
                                .mul_add(p.buy.epsilon, (-p.buy.kappa * arg).exp() / p.buy.kappa))
                    });

                let source = p.mu.mul_add(qf, -(p.psi * sigma2 * nu * qf * qf));
                let value =
                    (0.5 * sigma2 * nu).mul_add(dd_nu, drift_nu.mul_add(d_nu, source)) + sell + buy;
                du.set(idx, value);
            });
        }
    }

    /// The stiff ν advection–diffusion operator `L`, one tridiagonal row per
    /// interior cell — the linear part of the drift that carries the explicit
    /// scheme's `dt ∝ dν²` stiffness. Identical for every inventory level
    /// (the ν dynamics do not depend on q), so `field` is ignored; only the
    /// nonlinear, cross-`q` fill terms and the source are left to the
    /// explicit remainder. The linear-extrapolation ν boundary
    /// ([`fill_ghosts_extrapolate`], `ghost = 2·edge − inner`) is folded into
    /// the first and last domain rows, since the implicit solve sees no
    /// ghosts. The rows sum to zero (a pure transport operator), and
    /// `I − dt·L` is an M-matrix — the backward solve is unconditionally
    /// stable and monotone.
    fn stiff_rows(
        &self,
        grid: &CartesianGrid<1>,
        block: BlockId,
        _field: FieldHandle<f64>,
        rows: &mut [StiffRow],
    ) -> bool {
        let p = &self.params;
        let dnu = grid.spacing(block)[0];
        let inv_dnu = 1.0 / dnu;
        let inv_dnu2 = inv_dnu * inv_dnu;
        let sigma2 = p.sigma * p.sigma;
        let bc = grid.block_cells()[0];
        let n = grid.cells()[0];
        let base = block.index() * bc;

        for (i, row) in rows.iter_mut().enumerate() {
            let nu = grid.cell_center(block, [i as isize])[0];
            let a = 0.5 * sigma2 * nu * inv_dnu2; // central diffusion
            let drift = p.k_speed * (p.theta - nu); // upwind advection
            let adv = drift * inv_dnu;
            // Interior tridiagonal (matches the explicit RHS exactly):
            //   diffusion a·(u₋ − 2u + u₊); upwind adv on the sign of drift.
            let mut lo = a + if drift <= 0.0 { -adv } else { 0.0 };
            let mut ctr = 2.0f64.mul_add(-a, -drift.abs() * inv_dnu);
            let mut hi = a + if drift > 0.0 { adv } else { 0.0 };

            // Fold the linear-extrapolation ghost (2·edge − inner) at the
            // physical domain ends into the interior stencil.
            let global = base + i;
            if global == 0 {
                ctr += 2.0f64.mul_add(lo, ctr);
                hi -= lo;
                lo = 0.0;
            }
            if global == n - 1 {
                ctr += 2.0f64.mul_add(hi, ctr);
                lo -= hi;
                hi = 0.0;
            }
            *row = [lo, ctr, hi];
        }
        true
    }

    /// Stable dt of the nonstiff remainder (`N = V₀ − L·Y` = source + fill
    /// terms): a bounded reaction with **no dν dependence**, so an IMEX
    /// scheme's step is capped by the fill intensity, not the grid. Gershgorin
    /// on the fill Jacobian bounds the spectral radius by `2·(λˢ + λᵇ)`; a 0.4
    /// safety factor gives the bound below.
    fn stable_dt_nonstiff(&self, _spacing: [f64; 1]) -> Option<f64> {
        let rate = self.params.sell.lambda + self.params.buy.lambda;
        (rate > 0.0).then(|| 0.4 / rate)
    }

    /// Binding CFL bound of the explicit upwind/central scheme. Diffusion
    /// and advection restrictions are **joint**, not separate:
    /// `dt · max_ν (σ²ν/dν² + |k(θ−ν)|/dν) ≤ 1`, taken with a 0.8 safety
    /// factor. (Checking the two limits independently — the textbook
    /// shortcut — admits timesteps up to 2× too large near the ν where both
    /// terms peak, and the scheme then blows up mid-horizon.)
    ///
    /// The ν-extent is the model's own `[nu_min, nu_max]` (the grid must
    /// span it): a pure function of the spacing `dν`, so subcycling can
    /// evaluate it per level. Taking the bound at the domain *edges* rather
    /// than the outermost cell centers is conservative by half a cell.
    fn stable_dt(&self, spacing: [f64; 1]) -> Option<f64> {
        let p = &self.params;
        let dnu = spacing[0];
        let sigma2 = p.sigma * p.sigma;
        // bound(ν) = σ²ν/dν² + |k(θ−ν)|/dν is piecewise linear in ν with a
        // kink at θ: its maximum over [ν_min, ν_max] is at an endpoint or θ.
        let bound = |nu: f64| sigma2 * nu / dnu.powi(2) + (p.k_speed * (p.theta - nu)).abs() / dnu;
        let worst = bound(p.nu_min)
            .max(bound(p.nu_max))
            .max(bound(p.theta.clamp(p.nu_min, p.nu_max)));
        (worst > 0.0).then(|| 0.8 / worst)
    }
}

/// Optimal quoted depths δᵇ/δˢ on the `(τ, ν, q)` lattice, with nearest-node
/// lookup — the interface between the [`HjbMarketMaker`] solve and the
/// controlled path ensemble ([`MarketMakingEnsemble`]).
///
/// Built from the recorded HJB value surface: for each recorded backward-time
/// frame it bakes [`HjbMarketMaker::optimal_depths`] into `bid`/`ask` tables
/// indexed `[frame][q_index·n_ν + ν_cell]`, with `f64::INFINITY` marking a
/// withdrawn side. Cheap, allocation-free reads during the path simulation.
#[derive(Debug, Clone)]
pub struct DepthTables {
    dtau: f64,
    nu_min: f64,
    dnu: f64,
    n_nu: usize,
    q_min: isize,
    q_max: isize,
    bid: Vec<Vec<f64>>,
    ask: Vec<Vec<f64>>,
}

impl DepthTables {
    /// Bake the depth tables from the recorded value surface. `taus` are the
    /// recorded backward-time levels (uniformly spaced, ascending from 0), and
    /// `surfaces[frame][q_index][ν_cell]` is the value `u` at each — exactly
    /// what a surface-recording observer accumulates over the HJB solve.
    ///
    /// # Panics
    ///
    /// Panics if a surface frame is not `num_levels × grid.cells()` shaped.
    #[must_use]
    pub fn build(
        model: &HjbMarketMaker,
        grid: &CartesianGrid<1>,
        taus: &[f64],
        surfaces: &[Vec<Vec<f64>>],
    ) -> Self {
        let p = &model.params;
        let n_nu = grid.cells()[0];
        let nq = p.num_levels();
        let dtau = if taus.len() > 1 {
            taus[1] - taus[0]
        } else {
            f64::MAX
        };

        let (mut bid, mut ask) = (
            Vec::with_capacity(surfaces.len()),
            Vec::with_capacity(surfaces.len()),
        );
        for surface in surfaces {
            let mut b = vec![0.0; nq * n_nu];
            let mut a = vec![0.0; nq * n_nu];
            for qi in 0..nq {
                let q = p.q_min + qi as isize;
                for cell in 0..n_nu {
                    let u_q = surface[qi][cell];
                    let u_up = (q < p.q_max).then(|| surface[qi + 1][cell]);
                    let u_dn = (q > p.q_min).then(|| surface[qi - 1][cell]);
                    let (db, da) = model.optimal_depths(q, u_q, u_up, u_dn);
                    b[qi * n_nu + cell] = db;
                    a[qi * n_nu + cell] = da;
                }
            }
            bid.push(b);
            ask.push(a);
        }

        // Cell centers sit at nu_min + (i + 0.5)·dν; recover the domain edge.
        let dnu = grid.spacing(BlockId(0))[0];
        let nu_min = 0.5f64.mul_add(-dnu, grid.cell_center(BlockId(0), [0])[0]);
        Self {
            dtau,
            nu_min,
            dnu,
            n_nu,
            q_min: p.q_min,
            q_max: p.q_max,
            bid,
            ask,
        }
    }

    /// `(δᵇ, δˢ)` at remaining horizon `tau`, variance `nu`, inventory `q`
    /// (nearest lattice node, clamped into range). A withdrawn side reads
    /// `f64::INFINITY`.
    #[must_use]
    pub fn lookup(&self, tau: f64, nu: f64, q: isize) -> (f64, f64) {
        let t = ((tau / self.dtau).round().max(0.0) as usize).min(self.bid.len() - 1);
        let cell =
            (((nu - self.nu_min) / self.dnu - 0.5).round().max(0.0) as usize).min(self.n_nu - 1);
        let qi = (q.clamp(self.q_min, self.q_max) - self.q_min) as usize;
        let at = qi * self.n_nu + cell;
        (self.bid[t][at], self.ask[t][at])
    }
}

/// The **controlled** market-making system as a Monte Carlo path ensemble.
///
/// One path per grid cell (a 0-dimensional SDE per cell; see
/// [`crate::physics::stochastic_vol`]). Pairs with the
/// [`crate::core::monte_carlo`] harness.
///
/// Per path: a CIR variance and diffusive midprice (as in
/// [`crate::physics::stochastic_vol::StochVolPaths`]) driven by two Wiener
/// processes, plus two jump channels — bid and ask fills — that fire at the
/// optimal quotes looked up from a [`DepthTables`]. A bid fill (rate
/// `λᵇ·e^{−κᵇδᵇ}`) does `q += 1`, `cash −= S − δᵇ`; an ask fill does `q −= 1`,
/// `cash += S + δˢ`. Withdrawn sides read an infinite depth, hence zero
/// intensity, so inventory stays in `[q_min, q_max]` with no explicit clamp.
///
/// Driver set [`WienerJump<2, 2>`]: `nu`←Wiener 0, `mid`←Wiener 1, bid←Jump 0,
/// ask←Jump 1.
#[derive(Debug, Clone)]
pub struct MarketMakingEnsemble {
    /// Model parameters (shared with the HJB solve).
    pub params: MarketMakerParams,
    /// Optimal quotes from the HJB solve.
    pub tables: Arc<DepthTables>,
    /// Trading horizon `T`; remaining time `τ = T − t` indexes the tables.
    pub horizon: f64,
    nu: Option<FieldHandle<f64>>,
    mid: Option<FieldHandle<f64>>,
    inv: Option<FieldHandle<f64>>,
    cash: Option<FieldHandle<f64>>,
}

impl MarketMakingEnsemble {
    /// A controlled-system ensemble for the given parameters, optimal-quote
    /// tables, and trading horizon.
    #[must_use]
    pub const fn new(params: MarketMakerParams, tables: Arc<DepthTables>, horizon: f64) -> Self {
        Self {
            params,
            tables,
            horizon,
            nu: None,
            mid: None,
            inv: None,
            cash: None,
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

    /// Handle of the inventory field q.
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet.
    #[must_use]
    pub const fn inv(&self) -> FieldHandle<f64> {
        self.inv.expect("model fields not yet registered")
    }

    /// Handle of the cash field.
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet.
    #[must_use]
    pub const fn cash(&self) -> FieldHandle<f64> {
        self.cash.expect("model fields not yet registered")
    }

    /// Start every path at `(ν₀, S₀)` with zero inventory and cash.
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

    /// Terminal wealth `W = cash + q·S − α·q²` of one path — the mark-to-market
    /// value after liquidating inventory `q` at the midprice under the terminal
    /// penalty. The natural functional to reduce over the ensemble.
    #[must_use]
    pub fn terminal_wealth(&self, cash: f64, inv: f64, mid: f64) -> f64 {
        (self.params.alpha * inv).mul_add(-inv, inv.mul_add(mid, cash))
    }
}

impl<D: Sync> Model<CartesianGrid<1>, D> for MarketMakingEnsemble {
    type Scalar = f64;
    type Drivers = WienerJump<2, 2>;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        self.nu = Some(builder.register_driven("nu", 0, &[Driver::Time, Driver::Wiener(0)]));
        self.mid = Some(builder.register_driven("mid", 0, &[Driver::Time, Driver::Wiener(1)]));
        // Inventory and cash are moved only by the two fill channels.
        self.inv = Some(builder.register_driven("inv", 0, &[Driver::Jump(0), Driver::Jump(1)]));
        self.cash = Some(builder.register_driven("cash", 0, &[Driver::Jump(0), Driver::Jump(1)]));
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
            // CIR variance + midprice drift (full-truncation ν⁺).
            Driver::Time => {
                let mut dnu = out.view_mut(grid, block, self.nu());
                for_each_interior(nu.interior(), |idx| {
                    dnu.set(idx, p.k_speed * (p.theta - nu.get(idx).max(0.0)));
                });
                let mut dmid = out.view_mut(grid, block, self.mid());
                for_each_interior(nu.interior(), |idx| dmid.set(idx, p.mu));
            }
            // Variance diffusion σ√ν⁺.
            Driver::Wiener(0) => {
                let mut amp = out.view_mut(grid, block, self.nu());
                for_each_interior(nu.interior(), |idx| {
                    amp.set(idx, p.sigma * nu.get(idx).max(0.0).sqrt());
                });
            }
            // Midprice diffusion √ν⁺.
            Driver::Wiener(1) => {
                let mut amp = out.view_mut(grid, block, self.mid());
                for_each_interior(nu.interior(), |idx| {
                    amp.set(idx, nu.get(idx).max(0.0).sqrt());
                });
            }
            // Fills: bid (Jump 0) lifts inventory, ask (Jump 1) sheds it. On a
            // fire the kernel applies both increments together; a withdrawn
            // side (infinite depth) yields zero intensity, so it never fires
            // and inventory stays bounded with no explicit clamp. Written in
            // three cheap passes (the buffer is pre-zeroed, so unwritten cells
            // stay 0) since the increment and intensity slabs cannot be held
            // mutably at once.
            Driver::Jump(channel @ (0 | 1)) => {
                let tau = self.horizon - ctx.t;
                let (side, dq) = if channel == 0 {
                    (&p.buy, 1.0)
                } else {
                    (&p.sell, -1.0)
                };
                let (mid, inv) = (
                    state.view(grid, block, self.mid()),
                    state.view(grid, block, self.inv()),
                );
                // δ this channel quotes at, given the current (τ, ν, q).
                let depth_at = |idx: [isize; 1]| {
                    let q = inv.get(idx).round() as isize;
                    let (bid, ask) = self.tables.lookup(tau, nu.get(idx), q);
                    if channel == 0 { bid } else { ask }
                };
                // Inventory step is the constant ±1 (ignored where it does not fire).
                {
                    let mut dinv = out.view_mut(grid, block, self.inv());
                    for_each_interior(nu.interior(), |idx| dinv.set(idx, dq));
                }
                // Cash step: bid → δᵇ − S, ask → S + δˢ (only where quoted).
                {
                    let mut dcash = out.view_mut(grid, block, self.cash());
                    for_each_interior(nu.interior(), |idx| {
                        let depth = depth_at(idx);
                        if depth.is_finite() {
                            dcash.set(idx, dq.mul_add(-mid.get(idx), depth));
                        }
                    });
                }
                // Firing intensity λ·e^{−κδ} (zero on a withdrawn, infinite δ).
                {
                    let mut rate = out.intensity_mut(grid, block);
                    for_each_interior(nu.interior(), |idx| {
                        rate.set(idx, side.fill_rate(depth_at(idx)));
                    });
                }
            }
            Driver::Wiener(_) | Driver::Jump(_) => {
                unreachable!("MarketMakingEnsemble declares WienerJump<2,2>")
            }
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
