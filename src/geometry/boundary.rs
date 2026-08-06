//! Physical boundary conditions for cell-centered ghost fills.
//!
//! A ghost cell outside the physical domain is filled from the field's
//! **interior cells along the outward normal**, transformed by the boundary
//! condition on the face it sits behind. Each condition is handed the ghost
//! layer `k` (0 = innermost), the normal spacing `h`, and a lazy accessor
//! `interior(j)` returning the `j`-th interior cell measured inward from the
//! face (`j = 0` the edge cell, `j = 1` the next one in), and returns the
//! ghost value:
//!
//! | Condition                | face relation              | ghost(k)                     |
//! |--------------------------|----------------------------|------------------------------|
//! | [`FaceBc::Mirror`]       | ∂u/∂n = 0 (no-flux)        | `interior(k)`                |
//! | [`FaceBc::Dirichlet`]    | u = v at the face          | `2v − interior(k)`           |
//! | [`FaceBc::Flux`]         | ∂u/∂n = g (outward normal) | `interior(k) + (2k+1)·h·g`   |
//! | [`FaceBc::Extrapolate`]  | ∂²u/∂n² = 0 (linear)       | `(k+2)·interior(0) − (k+1)·interior(1)` |
//!
//! The first three read a single interior cell (`interior(k)`, the mirror
//! image found by reflecting the ghost coordinate across the boundary — see
//! [`reflect_coord`]); [`FaceBc::Extrapolate`] is the multi-point member,
//! reading the two edge cells to continue the field linearly (a transparent
//! outflow condition — no spurious curvature at the face). The accessor is
//! lazy so a condition reads only the cells it needs, keeping the no-flux
//! mirror a single copy. Robin and time-dependent conditions slot into the
//! same closure-driven fill: a model returns a `FaceBc` per (dimension, side)
//! — and, for time-dependent forcing, closes over the evaluation time it is
//! handed in `fill_ghosts`.

use crate::core::storage::Real;

/// A physical boundary condition on one non-periodic domain face.
///
/// For cell-centered ghost fills; see the module docs for the reflection each
/// one applies. Periodic faces are *topology*, not a `FaceBc`: the grid wraps
/// them before any boundary rule is consulted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaceBc {
    /// Homogeneous Neumann (no-flux): even reflection, `ghost = interior(k)`.
    /// The default for conservative diffusion — total quantity is preserved.
    Mirror,
    /// Dirichlet: the field equals `value` *at the face*. Odd reflection
    /// about `value`, `ghost = 2·value − interior(k)`.
    Dirichlet(f64),
    /// Inhomogeneous Neumann: the outward normal derivative ∂u/∂n equals
    /// `flux` at the face. `ghost = interior(k) + (2k+1)·h·flux`. `Flux(0.0)`
    /// is exactly [`FaceBc::Mirror`].
    Flux(f64),
    /// Linear extrapolation (zero second normal derivative): continue the
    /// field linearly from the two edge cells,
    /// `ghost = (k+2)·interior(0) − (k+1)·interior(1)`. A transparent
    /// outflow condition; requires at least two interior cells across the
    /// block in the normal direction.
    Extrapolate,
}

impl FaceBc {
    /// The ghost value at layer `k` (0 = innermost), from the normal spacing
    /// `h` and a lazy accessor to the interior cells along the outward normal
    /// (`interior(j)` = the `j`-th interior cell, `j = 0` the edge). See the
    /// module docs for the per-condition formula.
    #[inline]
    #[must_use]
    pub(crate) fn ghost_value<T: Real>(self, k: i64, h: f64, interior: impl Fn(i64) -> T) -> T {
        match self {
            Self::Mirror => interior(k),
            Self::Dirichlet(value) => T::from_f64(2.0 * value) - interior(k),
            Self::Flux(flux) => interior(k) + T::from_f64((2 * k + 1) as f64 * h * flux),
            Self::Extrapolate => {
                T::from_f64((k + 2) as f64) * interior(0)
                    - T::from_f64((k + 1) as f64) * interior(1)
            }
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
    fn mirror_is_flux_zero() {
        // A monotone interior line; mirror at layer k reads interior(k).
        let interior = |j: i64| 1.25 + j as f64;
        assert_eq!(FaceBc::Mirror.ghost_value(0, 0.4, interior), interior(0));
        assert_eq!(FaceBc::Flux(0.0).ghost_value(2, 0.4, interior), interior(2));
    }

    #[test]
    fn dirichlet_puts_value_on_the_face() {
        // (ghost + interior)/2 == value  at the innermost layer (k = 0).
        let edge = 0.3_f64;
        let value = 2.0_f64;
        let ghost = FaceBc::Dirichlet(value).ghost_value(0, 0.4, |_| edge);
        assert!((0.5 * (ghost + edge) - value).abs() < 1e-15);
    }

    #[test]
    fn flux_sets_the_normal_gradient() {
        // (ghost - interior)/h == g  at the innermost layer (outward normal,
        // low face: the one-sided gradient across the boundary face).
        let edge = 0.3_f64;
        let (g, h) = (1.5_f64, 0.4_f64);
        let ghost = FaceBc::Flux(g).ghost_value(0, h, |_| edge);
        assert!(((ghost - edge) / h - g).abs() < 1e-15);
    }

    #[test]
    fn extrapolate_continues_the_line() {
        // A field linear in the normal coordinate is reproduced exactly at
        // every ghost layer: edge - inner is the per-cell slope, and layer k
        // sits k+1 cells beyond the edge.
        let (edge, slope) = (2.0_f64, 0.5_f64);
        let interior = |j: i64| edge - slope * j as f64; // interior(0)=edge, interior(1)=edge-slope
        for k in 0..3 {
            let ghost = FaceBc::Extrapolate.ghost_value(k, 0.4, interior);
            assert!((ghost - (edge + slope * (k + 1) as f64)).abs() < 1e-15);
        }
    }
}
