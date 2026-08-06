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
        scratch::Scratch,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::StorageBackend,
    },
    geometry::{
        cartesian::{CartesianGrid, fill_ghosts_extrapolate, for_each_interior},
        grid::Grid,
    },
    physics::model::{Driver, Model, NoNoise, RhsContext},
};

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
