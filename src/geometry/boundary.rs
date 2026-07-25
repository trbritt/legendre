//! Physical boundary conditions for cell-centered ghost fills.
//!
//! A ghost cell outside the physical domain is filled from its **mirror
//! image** inside the domain, transformed by the boundary condition on the
//! face it sits behind. Every condition here is a per-layer *linear
//! reflection* of the mirror-image interior value `src`, so they compose with
//! the dimension-split ghost sweep the grid families already run and cost the
//! same as the no-flux mirror:
//!
//! ```text
//! ghost(k) = a·src + b        (k = ghost layer, 0 = innermost)
//! ```
//!
//! | Condition                | face relation              | a  | b                 |
//! |--------------------------|----------------------------|----|-------------------|
//! | [`FaceBc::Mirror`]       | ∂u/∂n = 0 (no-flux)        |  1 | 0                 |
//! | [`FaceBc::Dirichlet`]    | u = v at the face          | −1 | 2v                |
//! | [`FaceBc::Flux`]         | ∂u/∂n = g (outward normal) |  1 | (2k+1)·h·g        |
//!
//! `src` is the value at the mirror-image interior cell (found by reflecting
//! the ghost coordinate across the boundary with [`reflect_coord`]); `h` is
//! the cell spacing normal to the face; `2k+1` is the odd cell distance
//! between a ghost layer and its source ([`reflect_steps`]). Robin and
//! time-dependent conditions slot into the same closure-driven fill: a model
//! returns a `FaceBc` per (dimension, side) — and, for time-dependent
//! forcing, closes over the evaluation time it is handed in `fill_ghosts`.

use crate::core::storage::Real;

/// A physical boundary condition on one non-periodic domain face.
///
/// For cell-centered ghost fills; see the module docs for the reflection each
/// one applies. Periodic faces are *topology*, not a `FaceBc`: the grid wraps
/// them before any boundary rule is consulted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaceBc {
    /// Homogeneous Neumann (no-flux): even reflection, `ghost = src`. The
    /// default for conservative diffusion — total quantity is preserved.
    Mirror,
    /// Dirichlet: the field equals `value` *at the face*. Odd reflection
    /// about `value`, `ghost = 2·value − src`.
    Dirichlet(f64),
    /// Inhomogeneous Neumann: the outward normal derivative ∂u/∂n equals
    /// `flux` at the face. `ghost = src + (2k+1)·h·flux`. `Flux(0.0)` is
    /// exactly [`FaceBc::Mirror`].
    Flux(f64),
}

impl FaceBc {
    /// The ghost value from its mirror-image interior source `src`, the odd
    /// cell distance `steps` = 2k+1 to that source, and the normal spacing
    /// `h`. See the module docs for the per-condition formula.
    #[inline]
    #[must_use]
    pub(crate) fn ghost_value<T: Real>(self, src: T, steps: i64, h: f64) -> T {
        match self {
            Self::Mirror => src,
            Self::Dirichlet(value) => T::from_f64(2.0 * value) - src,
            Self::Flux(flux) => src + T::from_f64(steps as f64 * h * flux),
        }
    }
}

/// Reflect coordinate `c` across a `[lo, hi)` interior interval's boundaries
/// (the no-flux image); interior coordinates pass through unchanged. The one
/// place the mirror-index arithmetic lives — shared by the Cartesian and AMR
/// ghost fills and their prolongation-corner clamps.
#[inline]
#[must_use]
pub(crate) const fn reflect_coord(c: i64, lo: i64, hi: i64) -> i64 {
    if c < lo {
        2 * lo - 1 - c
    } else if c >= hi {
        2 * hi - 1 - c
    } else {
        c
    }
}

/// Odd cell distance `2k+1` between an out-of-interior coordinate `c` and its
/// [`reflect_coord`] image across a `[lo, hi)` interval; `1` for interior
/// coordinates. Used to place inhomogeneous-flux ghost layers.
#[inline]
#[must_use]
pub(crate) const fn reflect_steps(c: i64, lo: i64, hi: i64) -> i64 {
    if c < lo {
        2 * (lo - c) - 1
    } else if c >= hi {
        2 * (c - hi) + 1
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    // Exact float equality is deliberate: these check closed-form ghost
    // algebra, and a couple of clarity-first expressions skip `mul_add`.
    #![allow(clippy::float_cmp, clippy::suboptimal_flops)]
    use super::*;

    #[test]
    fn reflect_coord_matches_hand_math() {
        // interior [0, n): low ghosts -1,-2,-3 -> 0,1,2 ; high n,n+1 -> n-1,n-2.
        let n = 5;
        assert_eq!(reflect_coord(-1, 0, n), 0);
        assert_eq!(reflect_coord(-2, 0, n), 1);
        assert_eq!(reflect_coord(-3, 0, n), 2);
        assert_eq!(reflect_coord(n, 0, n), n - 1);
        assert_eq!(reflect_coord(n + 1, 0, n), n - 2);
        // interior passes through
        assert_eq!(reflect_coord(2, 0, n), 2);
    }

    #[test]
    fn reflect_steps_is_odd_distance() {
        let n = 5;
        assert_eq!(reflect_steps(-1, 0, n), 1);
        assert_eq!(reflect_steps(-2, 0, n), 3);
        assert_eq!(reflect_steps(-3, 0, n), 5);
        assert_eq!(reflect_steps(n, 0, n), 1);
        assert_eq!(reflect_steps(n + 1, 0, n), 3);
    }

    #[test]
    fn mirror_is_flux_zero() {
        let src = 1.25_f64;
        assert_eq!(FaceBc::Mirror.ghost_value(src, 1, 0.4), src);
        assert_eq!(FaceBc::Flux(0.0).ghost_value(src, 3, 0.4), src);
    }

    #[test]
    fn dirichlet_puts_value_on_the_face() {
        // (ghost + interior)/2 == value  at the innermost layer (steps=1).
        let interior = 0.3_f64;
        let value = 2.0_f64;
        let ghost = FaceBc::Dirichlet(value).ghost_value(interior, 1, 0.4);
        assert!((0.5 * (ghost + interior) - value).abs() < 1e-15);
    }

    #[test]
    fn flux_sets_the_normal_gradient() {
        // (ghost - interior)/h == g  at the innermost layer (outward normal,
        // low face: the one-sided gradient across the boundary face).
        let interior = 0.3_f64;
        let (g, h) = (1.5_f64, 0.4_f64);
        let ghost = FaceBc::Flux(g).ghost_value(interior, 1, h);
        assert!(((ghost - interior) / h - g).abs() < 1e-15);
    }
}
