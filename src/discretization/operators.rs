//! Operators as mathematical tags; discretizations as policies.
//!
//! `Grid + Discretization → Operators`
//! with an *open* universe of operators. Instead of one `Discretization`
//! trait with a fixed method list (which every new operator would break),
//! each (policy, operator, grid) triple is a separate impl of
//! [`Discretizes`]:
//!
//! ```text
//! impl Discretizes<CartesianGrid<2>, Laplacian> for FiniteDifference { … }
//! impl Discretizes<QuadTree,        Laplacian> for FiniteVolume     { … }
//! ```
//!
//! A model states its requirements as bounds — e.g. a diffusion model is
//! `impl<G, D> Model<G, D> for Diffusion where D: Discretizes<G, Laplacian>`
//! — and never learns which scheme satisfied them. Everything resolves at
//! compile time; there is no operator registry and no dynamic dispatch.
//!
//! Operator tags may carry mathematical parameters (e.g. a gradient
//! component index) but never numerical ones — those belong to the policy.

use super::stencil::Stencil;
use crate::geometry::grid::Grid;

/// ∇²
#[derive(Debug, Clone, Copy, Default)]
pub struct Laplacian;

/// ∂/∂`x_d` — one component of ∇.
#[derive(Debug, Clone, Copy)]
pub struct Gradient(pub usize);

/// ∇· of a face-flux function (the finite-volume workhorse; an anisotropic
/// surface-energy term is a `Divergence` with a nonlinear flux).
#[derive(Debug, Clone, Copy, Default)]
pub struct Divergence;

/// ∇·[A(n)² ∇φ − A(n)A′(n) ∂⊥φ] — the anisotropic surface-energy divergence
/// of phase-field models, with m-fold anisotropy A(θ) = ā(1 + ε′ cos mθ).
///
/// `eps4` is the 4-fold anisotropy strength (a *mathematical* parameter, so
/// it lives on the tag); regularization thresholds are numerical and belong
/// to the policy.
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicDivergence {
    /// 4-fold anisotropy strength ε₄.
    pub eps4: f64,
}

/// A discretization policy that knows how to realize operator `Op` on grid
/// `G` as a concrete [`Stencil`].
pub trait Discretizes<G: Grid, Op>: Send + Sync {
    /// The stencil type realizing `Op` on `G` under this policy.
    type Stencil: Stencil<G>;

    /// Build the stencil. Called at setup; the result is reused every step.
    fn build(&self, grid: &G, op: Op) -> Self::Stencil;
}
