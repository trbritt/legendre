//! Second-order central finite differences on Cartesian grids.

use super::{
    operators::{Discretizes, Laplacian},
    stencil::Stencil,
};
use crate::{
    core::storage::Real,
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
};
/// Central finite-difference policy (second order) on Cartesian grids.
#[derive(Debug, Clone, Copy, Default)]
pub struct FiniteDifference;

impl<const D: usize> Discretizes<CartesianGrid<D>, Laplacian> for FiniteDifference {
    type Stencil = CentralLaplacian;

    fn build(&self, _grid: &CartesianGrid<D>, _op: Laplacian) -> CentralLaplacian {
        CentralLaplacian
    }
}

/// (2·D+1)-point central Laplacian. Holds no geometry: spacing is read from
/// `(grid, block)` at apply time, so the same stencil serves every
/// refinement level.
#[derive(Debug, Clone, Copy, Default)]
pub struct CentralLaplacian;

impl<const D: usize> Stencil<CartesianGrid<D>> for CentralLaplacian {
    fn ghost_width(&self) -> u32 {
        1
    }

    fn apply<T: Real>(
        &self,
        grid: &CartesianGrid<D>,
        block: BlockId,
        input: <CartesianGrid<D> as Grid>::View<'_, T>,
        output: &mut <CartesianGrid<D> as Grid>::ViewMut<'_, T>,
    ) {
        let h = grid.spacing(block);
        let inv_h2: [T; D] = std::array::from_fn(|d| T::from_f64(1.0 / (h[d] * h[d])));
        for_each_interior(input.interior(), |idx| {
            let center = input.get(idx);
            let mut acc = T::ZERO;
            for d in 0..D {
                let mut plus = idx;
                plus[d] += 1;
                let mut minus = idx;
                minus[d] -= 1;
                acc += (input.get(plus) - (center + center) + input.get(minus)) * inv_h2[d];
            }
            output.set(idx, acc);
        });
    }
}
