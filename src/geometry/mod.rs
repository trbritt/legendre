//! Grids and topology: the `Grid` trait and concrete grid families.
//! Quadtree/octree AMR grids land here later behind the same trait.

pub mod cartesian;
pub mod grid;

/// Errors constructing a grid.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GridError {
    /// A dimension has zero cells or zero block cells.
    #[error("empty dimension {0}")]
    EmptyDimension(usize),
    /// The domain is not evenly tiled by the requested block size.
    #[error(
        "cells[{dimension}] = {cells} not divisible by block_cells[{dimension}] = {block_cells}"
    )]
    IndivisibleDimension {
        /// The offending dimension.
        dimension: usize,
        /// Interior cells requested in that dimension.
        cells: usize,
        /// Block cells requested in that dimension.
        block_cells: usize,
    },
    /// Cell spacing must be a positive finite number.
    #[error("invalid spacing {spacing} in dimension {dimension}")]
    InvalidSpacing {
        /// The offending dimension.
        dimension: usize,
        /// The rejected spacing value.
        spacing: f64,
    },
}
