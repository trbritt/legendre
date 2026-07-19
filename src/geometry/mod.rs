//! Grids and topology: the `Grid` trait and concrete grid families.
//! Block-structured AMR grids land here behind the same trait.

pub mod amr;
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
    /// AMR refinement ratios must be ≥ 2, one per level transition.
    #[error("invalid AMR ratios: {levels} refined levels, {ratios} ratios (each must be >= 2)")]
    AmrRatio {
        /// Refined levels requested.
        levels: usize,
        /// Ratios supplied.
        ratios: usize,
    },
    /// An AMR patch has zero cells.
    #[error("empty AMR patch at level {level}")]
    AmrEmptyPatch {
        /// The offending level.
        level: u8,
    },
    /// Two same-level AMR patches overlap.
    #[error("overlapping AMR patches at level {level}")]
    AmrOverlap {
        /// The offending level.
        level: u8,
    },
    /// An AMR patch violates proper nesting (Berger–Oliger: coarsened and
    /// grown by one cell, it must lie in the union of the level below,
    /// except at the physical domain boundary).
    #[error("AMR patch at level {level} is not properly nested in the level below")]
    AmrNotNested {
        /// The offending level.
        level: u8,
    },
}
