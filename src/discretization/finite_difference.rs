//! Second-order central finite differences on rectilinear grids.
//!
//! Kernels are free functions over the concrete Cartesian box views, so
//! every grid family whose blocks are uniform boxes (the Cartesian grid,
//! AMR patches) shares one implementation; the per-grid `Stencil` /
//! `Discretizes` impls are thin shims that read `(grid, block)` geometry
//! and delegate.

use super::{
    operators::{Discretizes, Laplacian},
    stencil::Stencil,
};
use crate::{
    core::storage::Real,
    geometry::{
        amr::AmrGrid,
        cartesian::{CartesianGrid, CartesianView, CartesianViewMut, for_each_interior},
        grid::{BlockId, Grid},
    },
};
/// Central finite-difference policy (second order) on rectilinear grids.
#[derive(Debug, Clone, Copy, Default)]
pub struct FiniteDifference;

impl<const D: usize> Discretizes<CartesianGrid<D>, Laplacian> for FiniteDifference {
    type Stencil = CentralLaplacian;

    fn build(&self, _grid: &CartesianGrid<D>, _op: Laplacian) -> CentralLaplacian {
        CentralLaplacian
    }
}

impl<const D: usize> Discretizes<AmrGrid<D>, Laplacian> for FiniteDifference {
    type Stencil = CentralLaplacian;

    fn build(&self, _grid: &AmrGrid<D>, _op: Laplacian) -> CentralLaplacian {
        CentralLaplacian
    }
}

/// (2·D+1)-point central Laplacian. Holds no geometry: spacing is read from
/// `(grid, block)` at apply time, so the same stencil serves every
/// refinement level.
#[derive(Debug, Clone, Copy, Default)]
pub struct CentralLaplacian;

/// The kernel: ∇²`input` → interior of `output`, given the block's cell
/// spacing.
pub(crate) fn central_laplacian_kernel<T: Real, const D: usize>(
    h: [f64; D],
    input: &CartesianView<'_, T, D>,
    output: &mut CartesianViewMut<'_, T, D>,
) {
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
        central_laplacian_kernel(grid.spacing(block), &input, output);
    }
}

impl<const D: usize> Stencil<AmrGrid<D>> for CentralLaplacian {
    fn ghost_width(&self) -> u32 {
        1
    }

    fn apply<T: Real>(
        &self,
        grid: &AmrGrid<D>,
        block: BlockId,
        input: <AmrGrid<D> as Grid>::View<'_, T>,
        output: &mut <AmrGrid<D> as Grid>::ViewMut<'_, T>,
    ) {
        central_laplacian_kernel(grid.spacing(block), &input, output);
    }
}
