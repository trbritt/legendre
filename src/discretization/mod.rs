//! Numerical schemes as policies.
//!
//! Operator tags, the `Discretizes` policy trait, the `Stencil` execution
//! abstraction, and concrete scheme families (central finite differences
//! and the finite-volume family).

pub mod finite_difference;
pub mod finite_volume;
pub mod operators;
pub mod stencil;
