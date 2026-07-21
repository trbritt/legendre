//! Finite-volume discretizations on rectilinear grids.
//!
//! Kernels operate on the concrete Cartesian box views, so every grid
//! family whose blocks are uniform boxes (the Cartesian grid, AMR patches)
//! shares one implementation; per-grid `Stencil`/`Discretizes` impls are
//! thin shims.
//!
//! First member: the Karma–Rappel anisotropic divergence used by
//! phase-field solidification models (Karma & Rappel, PRE 57, 4323 (1998)).
//!
//! For each cell, gradients are evaluated at the four face centers (normal
//! component by direct difference, transverse component by corner
//! averaging — a 9-point stencil), the anisotropy function and its angular
//! derivative are evaluated per face, and the update is the conservative
//! flux divergence `(J_R − J_L + J_T − J_B)/h²`.
//!
//! All face derivatives are *raw differences* (not divided by h): the
//! anisotropy factors are ratios that are scale-invariant in them, and the
//! single `1/h²` at the end restores dimensions.

use super::{
    operators::{AnisotropicDivergence, Discretizes, Laplacian},
    stencil::Stencil,
};
use crate::{
    core::storage::Real,
    discretization::finite_difference::CentralLaplacian,
    geometry::{
        amr::AmrGrid,
        cartesian::{CartesianGrid, CartesianView, CartesianViewMut, for_each_interior},
        grid::{BlockId, Grid},
    },
};

/// Finite-volume policy on rectilinear grids.
///
/// `tol` is the |∇φ|⁴ threshold below which the anisotropy is taken
/// isotropic (the interface-free limit), a purely numerical regularization —
/// hence policy state, not tag state.
#[derive(Debug, Clone, Copy)]
pub struct FiniteVolume {
    /// |∇φ|⁴ regularization threshold for the anisotropy evaluation.
    pub tol: f64,
}

impl Default for FiniteVolume {
    fn default() -> Self {
        Self { tol: 1e-8 }
    }
}

impl Discretizes<CartesianGrid<2>, AnisotropicDivergence> for FiniteVolume {
    type Stencil = KarmaRappelFlux;

    fn build(&self, grid: &CartesianGrid<2>, op: AnisotropicDivergence) -> KarmaRappelFlux {
        let [hx, hy] = grid.spacing(BlockId(0));
        debug_assert!(
            (hx - hy).abs() < 1e-12 * hx.abs(),
            "Karma–Rappel flux assumes isotropic spacing"
        );
        KarmaRappelFlux::new(op.eps4, self.tol)
    }
}

/// On a uniform Cartesian grid the FV Laplacian degenerates to the central
/// 5-point stencil, so the policy reuses it; a diffusion model bound on
/// `Discretizes<G, Laplacian>` runs under either policy unchanged.
impl<const D: usize> Discretizes<CartesianGrid<D>, Laplacian> for FiniteVolume {
    type Stencil = CentralLaplacian;

    fn build(&self, _grid: &CartesianGrid<D>, _op: Laplacian) -> CentralLaplacian {
        CentralLaplacian
    }
}

impl Discretizes<AmrGrid<2>, AnisotropicDivergence> for FiniteVolume {
    type Stencil = KarmaRappelFlux;

    fn build(&self, grid: &AmrGrid<2>, op: AnisotropicDivergence) -> KarmaRappelFlux {
        let [hx, hy] = grid.spacing(BlockId(0));
        debug_assert!(
            (hx - hy).abs() < 1e-12 * hx.abs(),
            "Karma-Rappel flux assumes isotropic spacing"
        );
        KarmaRappelFlux::new(op.eps4, self.tol)
    }
}

impl<const D: usize> Discretizes<AmrGrid<D>, Laplacian> for FiniteVolume {
    type Stencil = CentralLaplacian;

    fn build(&self, _grid: &AmrGrid<D>, _op: Laplacian) -> CentralLaplacian {
        CentralLaplacian
    }
}

/// 9-point anisotropic flux-divergence stencil with 4-fold symmetry.
///
/// Precomputed from `eps4`: `ā = 1 − 3ε₄`, `ε′ = 3ε₄/ā` (so that
/// A(θ) = ā(1 + ε′ cos 4θ)), and `a₁₂ = 4āε′` for the angular derivative.
#[derive(Debug, Clone, Copy)]
pub struct KarmaRappelFlux {
    a_bar: f64,
    eps_prime: f64,
    a12: f64,
    tol: f64,
}

impl KarmaRappelFlux {
    /// Precompute the anisotropy constants from ε₄ and the regularization
    /// threshold.
    #[must_use]
    pub fn new(eps4: f64, tol: f64) -> Self {
        let a_bar = 3.0f64.mul_add(-eps4, 1.0);
        let eps_prime = 3.0 * eps4 / a_bar;
        Self {
            a_bar,
            eps_prime,
            a12: 4.0 * a_bar * eps_prime,
            tol,
        }
    }

    /// Anisotropy A and its angular derivative dA/dθ (÷|∇φ|² normalization
    /// folded in), from raw face differences (dx, dy).
    /// Below the regularization threshold the interface is flat: A = ā,
    /// dA = 0.
    #[inline(always)]
    fn face<T: Real>(&self, dx: T, dy: T) -> (T, T) {
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let sum = dx2 + dy2;
        let mag2 = sum * sum;
        if mag2 <= T::from_f64(self.tol) {
            (T::from_f64(self.a_bar), T::ZERO)
        } else {
            let a = T::from_f64(self.a_bar)
                * (T::ONE + T::from_f64(self.eps_prime) * (dx2 * dx2 + dy2 * dy2) / mag2);
            let da = -(T::from_f64(self.a12)) * dx * dy * (dx2 - dy2) / mag2;
            (a, da)
        }
    }

    /// Cell-centered anisotropy A(∇φ) from central differences — the same
    /// regularized function models need for the 1/A² mobility prefactor.
    #[inline(always)]
    #[must_use]
    pub fn center_anisotropy<T: Real>(
        &self,
        input: &CartesianView<'_, T, 2>,
        idx: [isize; 2],
    ) -> T {
        let [i, j] = idx;
        let half = T::from_f64(0.5);
        let dx = half * (input.get([i + 1, j]) - input.get([i - 1, j]));
        let dy = half * (input.get([i, j + 1]) - input.get([i, j - 1]));
        self.face(dx, dy).0
    }

    /// [`Self::center_anisotropy`] in a crystal frame rotated by θ₀
    /// (supplied as `cos θ₀`, `sin θ₀`).
    #[inline(always)]
    pub fn center_anisotropy_rotated<T: Real>(
        &self,
        input: &CartesianView<'_, T, 2>,
        idx: [isize; 2],
        cos0: T,
        sin0: T,
    ) -> T {
        let [i, j] = idx;
        let half = T::from_f64(0.5);
        let dx = half * (input.get([i + 1, j]) - input.get([i - 1, j]));
        let dy = half * (input.get([i, j + 1]) - input.get([i, j - 1]));
        let (rx, ry) = rotate(cos0, sin0, dx, dy);
        self.face(rx, ry).0
    }

    /// Flux divergence with a per-cell crystal orientation field θ₀(x)
    /// (multi-grain nucleation). The variational flux keeps *lab-frame*
    /// gradient components,
    ///
    /// ```text
    /// Jx = A²·∂xφ − A·A′·∂yφ,   Jy = A²·∂yφ + A·A′·∂xφ,
    /// ```
    ///
    /// while A and A′ are functions of the interface angle *relative to the
    /// crystal axes*: they are evaluated from face gradients rotated by
    /// −θ₀. θ₀ is read at the cell center for all four faces, so fluxes are
    /// exactly conservative within a grain and approximate only across
    /// grain boundaries (where a single-order-parameter model is
    /// approximate anyway).
    pub fn apply_oriented<T: Real>(
        &self,
        [hx, hy]: [f64; 2],
        input: CartesianView<'_, T, 2>,
        orientation: CartesianView<'_, T, 2>,
        output: &mut CartesianViewMut<'_, T, 2>,
    ) {
        let inv_h2 = T::from_f64(1.0 / (hx * hy));
        let quarter = T::from_f64(0.25);
        let p = |i: isize, j: isize| input.get([i, j]);

        for_each_interior(input.interior(), |[i, j]| {
            let theta0 = orientation.get([i, j]).to_f64();
            let (s0, c0) = theta0.sin_cos();
            let (c0, s0) = (T::from_f64(c0), T::from_f64(s0));

            let derx_r = p(i + 1, j) - p(i, j);
            let derx_l = p(i, j) - p(i - 1, j);
            let derx_t = quarter * (p(i + 1, j + 1) - p(i - 1, j + 1) + p(i + 1, j) - p(i - 1, j));
            let derx_b = quarter * (p(i + 1, j) - p(i - 1, j) + p(i + 1, j - 1) - p(i - 1, j - 1));

            let dery_t = p(i, j + 1) - p(i, j);
            let dery_b = p(i, j) - p(i, j - 1);
            let dery_r = quarter * (p(i + 1, j + 1) - p(i + 1, j - 1) + p(i, j + 1) - p(i, j - 1));
            let dery_l = quarter * (p(i, j + 1) - p(i, j - 1) + p(i - 1, j + 1) - p(i - 1, j - 1));

            // Anisotropy in the crystal frame, fluxes in the lab frame.
            let (rx, ry) = rotate(c0, s0, derx_r, dery_r);
            let (a_r, da_r) = self.face(rx, ry);
            let (rx, ry) = rotate(c0, s0, derx_l, dery_l);
            let (a_l, da_l) = self.face(rx, ry);
            let (rx, ry) = rotate(c0, s0, derx_t, dery_t);
            let (a_t, da_t) = self.face(rx, ry);
            let (rx, ry) = rotate(c0, s0, derx_b, dery_b);
            let (a_b, da_b) = self.face(rx, ry);

            let j_r = a_r * (a_r * derx_r - da_r * dery_r);
            let j_l = a_l * (a_l * derx_l - da_l * dery_l);
            let j_t = a_t * (a_t * dery_t + da_t * derx_t);
            let j_b = a_b * (a_b * dery_b + da_b * derx_b);

            output.set([i, j], (j_r - j_l + j_t - j_b) * inv_h2);
        });
    }
}

/// Rotate lab-frame gradient components into a crystal frame at angle θ₀:
/// `(dx', dy') = (c₀·dx + s₀·dy, −s₀·dx + c₀·dy)`.
#[inline(always)]
fn rotate<T: Real>(cos0: T, sin0: T, dx: T, dy: T) -> (T, T) {
    (cos0 * dx + sin0 * dy, cos0 * dy - sin0 * dx)
}

impl KarmaRappelFlux {
    /// The axis-aligned kernel behind `Stencil::apply` for every
    /// rectilinear grid family.
    fn kernel<T: Real>(
        &self,
        [hx, hy]: [f64; 2],
        input: &CartesianView<'_, T, 2>,
        output: &mut CartesianViewMut<'_, T, 2>,
    ) {
        let inv_h2 = T::from_f64(1.0 / (hx * hy));
        let quarter = T::from_f64(0.25);
        let p = |i: isize, j: isize| input.get([i, j]);

        for_each_interior(input.interior(), |[i, j]| {
            // Face-centered gradients from raw differences: normal component
            // direct, transverse component corner-averaged.
            let derx_r = p(i + 1, j) - p(i, j);
            let derx_l = p(i, j) - p(i - 1, j);
            let derx_t = quarter * (p(i + 1, j + 1) - p(i - 1, j + 1) + p(i + 1, j) - p(i - 1, j));
            let derx_b = quarter * (p(i + 1, j) - p(i - 1, j) + p(i + 1, j - 1) - p(i - 1, j - 1));

            let dery_t = p(i, j + 1) - p(i, j);
            let dery_b = p(i, j) - p(i, j - 1);
            let dery_r = quarter * (p(i + 1, j + 1) - p(i + 1, j - 1) + p(i, j + 1) - p(i, j - 1));
            let dery_l = quarter * (p(i, j + 1) - p(i, j - 1) + p(i - 1, j + 1) - p(i - 1, j - 1));

            let (a_r, da_r) = self.face(derx_r, dery_r);
            let (a_l, da_l) = self.face(derx_l, dery_l);
            let (a_t, da_t) = self.face(derx_t, dery_t);
            let (a_b, da_b) = self.face(derx_b, dery_b);

            // Conservative face fluxes.
            let j_r = a_r * (a_r * derx_r - da_r * dery_r);
            let j_l = a_l * (a_l * derx_l - da_l * dery_l);
            let j_t = a_t * (a_t * dery_t + da_t * derx_t);
            let j_b = a_b * (a_b * dery_b + da_b * derx_b);

            output.set([i, j], (j_r - j_l + j_t - j_b) * inv_h2);
        });
    }
}

impl Stencil<CartesianGrid<2>> for KarmaRappelFlux {
    fn ghost_width(&self) -> u32 {
        1
    }

    fn apply<T: Real>(
        &self,
        grid: &CartesianGrid<2>,
        block: BlockId,
        input: <CartesianGrid<2> as Grid>::View<'_, T>,
        output: &mut <CartesianGrid<2> as Grid>::ViewMut<'_, T>,
    ) {
        self.kernel(grid.spacing(block), &input, output);
    }
}

impl Stencil<AmrGrid<2>> for KarmaRappelFlux {
    fn ghost_width(&self) -> u32 {
        1
    }

    fn apply<T: Real>(
        &self,
        grid: &AmrGrid<2>,
        block: BlockId,
        input: <AmrGrid<2> as Grid>::View<'_, T>,
        output: &mut <AmrGrid<2> as Grid>::ViewMut<'_, T>,
    ) {
        self.kernel(grid.spacing(block), &input, output);
    }
}
