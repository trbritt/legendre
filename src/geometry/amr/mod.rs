//! Block-structured adaptive mesh refinement (Berger–Oliger lineage).
//!
//! Refinement is strictly **rectilinear**: patches are axis-aligned boxes
//! of uniform cells at each level — never a general placement of points.
//! The pieces:
//!
//! - [`cluster`]: the Berger–Rigoutsos grid-generation kernel — flagged
//!   cells → disjoint enclosing boxes.
//! - [`AmrGrid`]: a patch hierarchy behind the same [`Grid`] trait every
//!   uniform grid implements; patches are blocks, views are the plain
//!   Cartesian box views, so stencil kernels run on a patch unmodified.
//!   The hierarchy is immutable — a regrid builds a *new* grid and
//!   migrates state, exactly as `CartesianGrid` documents.
//!
//! [`Grid`]: crate::geometry::grid::Grid

mod adapt;
mod cluster;
mod grid;
mod intergrid;

pub use adapt::{BergerOliger, GradientTagger, RegridPolicy, TagCells};
pub use cluster::{CellBox, ClusterParams, cluster};
pub use grid::{AmrGrid, AmrPatch};
pub use intergrid::{fill_ghosts_mirror, fill_level, restrict, restrict_level};
