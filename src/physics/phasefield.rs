//! Phase-field models. First member: **Model C** dendritic solidification
//! (Karma & Rappel, PRE 57, 4323 (1998)).
//!
//! Coupled fields on a 2D Cartesian grid:
//!
//! ```text
//! τ(n)·∂φ/∂t = ∇·[A²∇φ − A A′ ∂⊥φ] + φ − φ³ − λ·u·(1 − φ²)²   (+ noise)
//!     ∂u/∂t = D∇²u + ½·∂φ/∂t
//! ```
//!
//! with 4-fold anisotropy A(θ) = ā(1 + ε′cos 4θ), relaxation time
//! τ(n) = A(n)², thermal diffusivity D = a₂λ, and mirror (no-flux)
//! boundaries.
//!
//! The model owns τ(n) = A(n)² — the anisotropy appears in the *physics*
//! (free energy), so the mobility prefactor is evaluated here, while the
//! discretization policy owns how the anisotropic divergence is realized as
//! a stencil. The two share the mathematical constants but not code paths;
//! `anisotropy_tol` should match the policy's regularization threshold.

use crate::{
    core::{
        scratch::Scratch,
        state::{BlockStateMut, FieldHandle, State, StateBuilder},
        storage::StorageBackend,
    },
    discretization::{
        finite_volume::KarmaRappelFlux,
        operators::{AnisotropicDivergence, Discretizes, Laplacian},
        stencil::Stencil,
    },
    geometry::{
        cartesian::{CartesianGrid, fill_ghosts_mirror, for_each_interior},
        grid::{BlockId, Grid},
    },
    physics::model::{Driver, DriverSet, Model, NoNoise, RhsContext},
};
use std::marker::PhantomData;

/// Karma–Rappel thin-interface asymptotics constant a₁.
pub const A1: f64 = 0.8836;
/// Karma–Rappel thin-interface asymptotics constant a₂ (sets D = a₂λ).
pub const A2: f64 = 0.6267;

/// Karma–Rappel dendritic solidification: a coupled order parameter φ and
/// dimensionless temperature u with 4-fold anisotropic surface energy; see
/// the module docs for the equations.
///
/// Generic over its driver set `N`: the default `ModelC` (= `ModelC<NoNoise>`)
/// is deterministic and pairs with any integrator;
/// `ModelC<Wiener<1>>` adds the additive noise term on φ and pairs with
/// stochastic integrators only.
#[derive(Debug, Clone)]
pub struct ModelC<N: DriverSet = NoNoise> {
    /// 4-fold anisotropy strength ε₄.
    pub eps4: f64,
    /// Coupling λ between undercooling and the order parameter.
    pub lambda: f64,
    /// Thermal diffusivity D = a₂λ (thin-interface relation).
    pub d_thermal: f64,
    /// Amplitude of the additive noise on φ. Takes effect only when the
    /// driver set is `Wiener<1>`; a `NoNoise` model has no stochastic term
    /// to scale.
    pub noise_amplitude: f64,
    /// |∇φ|⁴ threshold below which the interface is treated as flat.
    pub anisotropy_tol: f64,
    // Derived: A(θ) = a_bar·(1 + eps_prime·cos 4θ).
    a_bar: f64,
    eps_prime: f64,
    orientations: bool,
    phi: Option<FieldHandle<f64>>,
    u: Option<FieldHandle<f64>>,
    theta: Option<FieldHandle<f64>>,
    _noise: PhantomData<fn() -> N>,
}

/// One nucleation site: a solid seed with its own crystallographic
/// orientation (radians; the 4-fold anisotropy makes it π/2-periodic).
#[derive(Debug, Clone, Copy)]
pub struct Grain {
    /// Seed center in physical coordinates.
    pub center: [f64; 2],
    /// Seed radius (the tanh profile's half-width sits at this radius).
    pub radius: f64,
    /// Crystallographic orientation θ₀ in radians.
    pub orientation: f64,
}

impl<N: DriverSet> ModelC<N> {
    /// A visually striking parameter set: ε₄ = 0.06, λ = 3.19, no noise.
    #[must_use]
    pub fn classic() -> Self {
        Self::new(0.06, 3.19, 0.0)
    }

    /// A model with the given anisotropy strength, coupling, and additive
    /// noise amplitude; D = a₂λ per the thin-interface relation.
    #[must_use]
    pub fn new(eps4: f64, lambda: f64, noise_amplitude: f64) -> Self {
        let a_bar = 3.0f64.mul_add(-eps4, 1.0);
        Self {
            eps4,
            lambda,
            d_thermal: A2 * lambda,
            noise_amplitude,
            anisotropy_tol: 1e-8,
            a_bar,
            eps_prime: 3.0 * eps4 / a_bar,
            orientations: false,
            phi: None,
            u: None,
            theta: None,
            _noise: PhantomData,
        }
    }

    /// Enable per-grain crystallographic orientations: registers a static
    /// orientation field θ₀(x) (assigned by [`Self::initialize_grains`])
    /// and evaluates the anisotropy as A(θ − θ₀). Costs one extra field of
    /// state (+50% memory) and a sincos per cell per stage.
    #[must_use]
    pub const fn with_orientations(mut self) -> Self {
        self.orientations = true;
        self
    }

    /// Handle of the order parameter φ.
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet (i.e.
    /// before [`crate::core::simulation::Simulation::new`]).
    #[must_use]
    pub const fn phi(&self) -> FieldHandle<f64> {
        self.phi.expect("model fields not yet registered")
    }

    /// Handle of the dimensionless temperature u.
    ///
    /// # Panics
    ///
    /// Panics if the model's fields have not been registered yet (i.e.
    /// before [`crate::core::simulation::Simulation::new`]).
    #[must_use]
    pub const fn u(&self) -> FieldHandle<f64> {
        self.u.expect("model fields not yet registered")
    }

    /// Handle of the grain-orientation field θ₀ (orientations enabled only).
    #[must_use]
    pub const fn theta0(&self) -> Option<FieldHandle<f64>> {
        self.theta
    }

    /// Cell-centered anisotropy A(∇φ) for the τ(n) = A² mobility prefactor,
    /// from central differences (scale-invariant in the raw differences).
    #[inline(always)]
    fn center_anisotropy(
        &self,
        phi: &<CartesianGrid<2> as Grid>::View<'_, f64>,
        [i, j]: [isize; 2],
    ) -> f64 {
        let dx = 0.5 * (phi.get([i + 1, j]) - phi.get([i - 1, j]));
        let dy = 0.5 * (phi.get([i, j + 1]) - phi.get([i, j - 1]));
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let sum = dx2 + dy2;
        let mag2 = sum * sum;
        if mag2 <= self.anisotropy_tol {
            self.a_bar
        } else {
            self.a_bar * (1.0 + self.eps_prime * (dx2 * dx2 + dy2 * dy2) / mag2)
        }
    }

    /// A standard initial condition: a single solid seed
    /// `φ₀ = −tanh((|x − c| − r₀)/√2)` at `seed_center` with radius
    /// `seed_radius`, in a uniformly undercooled melt `u = −u₀`.
    pub fn initialize<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<2>,
        state: &mut State<f64, S>,
        seed_center: [f64; 2],
        seed_radius: f64,
        u0: f64,
    ) {
        self.initialize_seeds(grid, state, &[(seed_center, seed_radius)], u0);
    }

    /// Multi-seed initial condition for nucleation studies: each seed is a
    /// `(center, radius)` tanh profile, combined by pointwise max (solid
    /// wins where seeds overlap), in a uniformly undercooled melt `u = −u₀`.
    ///
    /// Cost is O(cells + blocks·seeds), not O(cells·seeds): a tanh profile
    /// is −1 to f64 precision once `r − r₀ > 19√2 ≈ 27`, so each block only
    /// evaluates the seeds whose influence region intersects it.
    pub fn initialize_seeds<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<2>,
        state: &mut State<f64, S>,
        seeds: &[([f64; 2], f64)],
        u0: f64,
    ) {
        let grains: Vec<Grain> = seeds
            .iter()
            .map(|&(center, radius)| Grain {
                center,
                radius,
                orientation: 0.0,
            })
            .collect();
        self.initialize_grains(grid, state, &grains, u0);
    }

    /// Nucleation initial condition with per-grain orientations. φ is the
    /// pointwise max of the seed tanh profiles; if orientations are enabled,
    /// θ₀(x) is assigned by **nearest-seed Voronoi** — the impingement
    /// partition, so each growing dendrite carries the anisotropy of the
    /// grain that will claim that region. Melt is uniformly undercooled to
    /// `u = −u₀`.
    ///
    /// Both passes are O(cells + blocks·grains): the tanh profile saturates
    /// within `r₀ + 27` of a seed, and a grain is a nearest-seed candidate
    /// on a block only if its closest approach beats every grain's farthest
    /// corner.
    pub fn initialize_grains<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<2>,
        state: &mut State<f64, S>,
        grains: &[Grain],
        u0: f64,
    ) {
        const CUTOFF: f64 = 27.0;
        let (phi_h, u_h) = (self.phi(), self.u());
        let interior = grid.block_cells();
        let [nx, ny] = interior;
        for b in 0..grid.num_blocks() {
            let block = BlockId(b as u32);
            let [hx, hy] = grid.spacing(block);
            let c0 = grid.cell_center(block, [0, 0]);
            let c1 = grid.cell_center(block, [nx as isize - 1, ny as isize - 1]);
            let (lo, hi) = (
                [0.5f64.mul_add(-hx, c0[0]), 0.5f64.mul_add(-hy, c0[1])],
                [0.5f64.mul_add(hx, c1[0]), 0.5f64.mul_add(hy, c1[1])],
            );
            // distance from a point to this block's bounding box (0 inside)
            let dist_box = |cx: f64, cy: f64| {
                let dx = (lo[0] - cx).max(cx - hi[0]).max(0.0);
                let dy = (lo[1] - cy).max(cy - hi[1]).max(0.0);
                dx.hypot(dy)
            };
            // distance from a point to this block's farthest corner
            let dist_far = |cx: f64, cy: f64| {
                let dx = (cx - lo[0]).abs().max((cx - hi[0]).abs());
                let dy = (cy - lo[1]).abs().max((cy - hi[1]).abs());
                dx.hypot(dy)
            };

            let near: Vec<Grain> = grains
                .iter()
                .copied()
                .filter(|g| dist_box(g.center[0], g.center[1]) <= g.radius + CUTOFF)
                .collect();
            {
                let mut v = state.view_mut(grid, block, phi_h);
                if near.is_empty() {
                    for_each_interior(interior, |idx| v.set(idx, -1.0));
                } else {
                    for_each_interior(interior, |idx| {
                        let [x, y] = grid.cell_center(block, idx);
                        let phi = near
                            .iter()
                            .map(|g| {
                                let r = (x - g.center[0]).hypot(y - g.center[1]);
                                -((r - g.radius) / f64::sqrt(2.0)).tanh()
                            })
                            .fold(-1.0f64, f64::max);
                        v.set(idx, phi);
                    });
                }
            }
            {
                let mut v = state.view_mut(grid, block, u_h);
                for_each_interior(interior, |idx| v.set(idx, -u0));
            }
            if let Some(theta_h) = self.theta {
                let threshold = grains
                    .iter()
                    .map(|g| dist_far(g.center[0], g.center[1]))
                    .fold(f64::MAX, f64::min);
                let candidates: Vec<Grain> = grains
                    .iter()
                    .copied()
                    .filter(|g| dist_box(g.center[0], g.center[1]) <= threshold)
                    .collect();
                let mut v = state.view_mut(grid, block, theta_h);
                for_each_interior(interior, |idx| {
                    let [x, y] = grid.cell_center(block, idx);
                    let nearest = candidates
                        .iter()
                        .min_by(|a, b| {
                            let da = (y - a.center[1])
                                .mul_add(y - a.center[1], (x - a.center[0]).powi(2));
                            let db = (y - b.center[1])
                                .mul_add(y - b.center[1], (x - b.center[0]).powi(2));
                            da.total_cmp(&db)
                        })
                        .map_or(0.0, |g| g.orientation);
                    v.set(idx, nearest);
                });
            }
        }
    }
}

// The `Stencil = KarmaRappelFlux` bound exists because the oriented
// (multi-grain) path calls `apply_oriented`, a two-input stencil the
// single-input `Stencil` trait cannot yet express — a known trait-surface
// limitation. Axis-aligned Model C uses only the generic trait surface.
impl<D, N: DriverSet> Model<CartesianGrid<2>, D> for ModelC<N>
where
    D: Discretizes<CartesianGrid<2>, AnisotropicDivergence, Stencil = KarmaRappelFlux>
        + Discretizes<CartesianGrid<2>, Laplacian>,
{
    type Scalar = f64;
    type Drivers = N;

    fn register_fields(&mut self, builder: &mut StateBuilder<f64>) {
        // φ is the only field the (optional) Wiener driver moves: its
        // amplitude buffer carries φ storage alone, so the stochastic
        // term never pays u-sized traffic. Harmless under NoNoise, where
        // the Wiener driver simply never runs.
        self.phi = Some(builder.register_driven("phi", 1, &[Driver::Time, Driver::Wiener(0)]));
        self.u = Some(builder.register("u", 1));
        if self.orientations {
            // Static per-cell crystal orientation; ghost width 0 because
            // the stencil reads it at cell centers only. Registered static:
            // tendency buffers carry no storage for it and the integrator
            // skips it in every vector-space sweep.
            self.theta = Some(builder.register_static("theta0", 0));
        }
    }

    fn fill_ghosts<S: StorageBackend<f64>>(
        &self,
        grid: &CartesianGrid<2>,
        state: &mut State<f64, S>,
        _t: f64,
    ) {
        fill_ghosts_mirror(grid, state, self.phi());
        fill_ghosts_mirror(grid, state, self.u());
    }

    fn vector_field_block<S: StorageBackend<f64>>(
        &self,
        driver: Driver,
        ctx: &RhsContext<'_, CartesianGrid<2>, D>,
        state: &State<f64, S>,
        out: &mut BlockStateMut<'_, f64, S>,
        _scratch: &mut Scratch<f64, S>,
    ) {
        if let Driver::Wiener(_) = driver {
            // Additive noise on φ only, constant amplitude.
            let mut amp = out.view_mut(ctx.grid, ctx.block, self.phi());
            for_each_interior(amp.interior(), |idx| amp.set(idx, self.noise_amplitude));
            return;
        }
        let (grid, block) = (ctx.grid, ctx.block);
        let (phi_h, u_h) = (self.phi(), self.u());
        let phi = state.view(grid, block, phi_h);
        let u = state.view(grid, block, u_h);

        // dφ/dt = [∇·J − g′(φ) − λ·u·P′(φ)] / A(∇φ − frame θ₀)²
        {
            let flux = ctx
                .disc
                .build(grid, AnisotropicDivergence { eps4: self.eps4 });
            let mut dphi = out.view_mut(grid, block, phi_h);
            if let Some(theta_h) = self.theta {
                // Multi-grain path: anisotropy in each grain's crystal frame.
                let theta = state.view(grid, block, theta_h);
                flux.apply_oriented(grid.spacing(block), phi, theta, &mut dphi);
                for_each_interior(phi.interior(), |idx| {
                    let p = phi.get(idx);
                    let g_prime = (p * p).mul_add(p, -p);
                    let one_m_p2 = p.mul_add(-p, 1.0);
                    let p_prime = one_m_p2 * one_m_p2;
                    let (s0, c0) = theta.get(idx).sin_cos();
                    let a = flux.center_anisotropy_rotated(&phi, idx, c0, s0);
                    let val = (self.lambda * u.get(idx)).mul_add(-p_prime, dphi.get(idx) - g_prime)
                        / (a * a);
                    dphi.set(idx, val);
                });
            } else {
                flux.apply(grid, block, phi, &mut dphi);
                for_each_interior(phi.interior(), |idx| {
                    let p = phi.get(idx);
                    let g_prime = (p * p).mul_add(p, -p);
                    let one_m_p2 = p.mul_add(-p, 1.0);
                    let p_prime = one_m_p2 * one_m_p2;
                    let a = self.center_anisotropy(&phi, idx);
                    let val = (self.lambda * u.get(idx)).mul_add(-p_prime, dphi.get(idx) - g_prime)
                        / (a * a);
                    dphi.set(idx, val);
                });
            }
        }

        // θ₀ is registered static: tendency buffers allocate nothing for it
        // and the integrator's vector ops skip it, so no tendency needs to
        // be written. (fill on the empty slab would be a no-op anyway.)

        // du/dt = D∇²u + ½·dφ/dt (reads the freshly written dφ/dt).
        let lap = ctx.disc.build(grid, Laplacian);
        let (mut du, dphi) = out.view_split(grid, block, u_h, phi_h);
        lap.apply(grid, block, u, &mut du);
        for_each_interior(u.interior(), |idx| {
            du.set(
                idx,
                du.get(idx).mul_add(self.d_thermal, 0.5 * dphi.get(idx)),
            );
        });
    }

    /// dt = 0.2·h²/D — deliberately *below* the model's 0.25·h²/D. That
    /// choice sits exactly on the 2D limit of the 5-point thermal update,
    /// where the grid-scale checkerboard mode has amplification −1
    /// (undamped); latent-heat release at the interface forces that mode
    /// coherently, so it grows secularly and eventually blows up. r = 0.2
    /// damps it (factor −0.6) with 20% margin.
    fn stable_dt(&self, grid: &CartesianGrid<2>) -> Option<f64> {
        let [hx, hy] = grid.spacing(BlockId(0));
        Some(0.2 * hx.min(hy).powi(2) / self.d_thermal)
    }
}
