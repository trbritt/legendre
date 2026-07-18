//! Unit-level validation of the core mechanics: grid construction errors,
//! block topology, iteration order, halo exchange / mirror boundaries,
//! static-field buffer layout, split borrows, and the vector-space
//! operations integrators are built on.

// Exact float equality is the point throughout: every assertion compares
// values that were *copied* (halo exchange, axpy with exact inputs), where
// any difference is a real defect.
#![allow(clippy::float_cmp)]

use legendre::{
    core::{
        state::{State, StateBuilder},
        storage::{DenseStorage, SystemAllocator},
    },
    geometry::{
        GridError,
        cartesian::{CartesianGrid, fill_ghosts_mirror, for_each_box, for_each_interior},
        grid::{BlockId, Grid},
    },
};

type DenseState = State<f64, DenseStorage<f64>>;

mod grid {
    use super::*;

    #[test]
    fn construction_rejects_bad_inputs() {
        assert!(matches!(
            CartesianGrid::new([0, 4], [1, 2], [0.0; 2], [1.0; 2]),
            Err(GridError::EmptyDimension(0))
        ));
        assert!(matches!(
            CartesianGrid::new([4, 4], [3, 2], [0.0; 2], [1.0; 2]),
            Err(GridError::IndivisibleDimension { dimension: 0, .. })
        ));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                CartesianGrid::new([4, 4], [2, 2], [0.0; 2], [1.0, bad]),
                Err(GridError::InvalidSpacing { dimension: 1, .. })
            ));
        }
    }

    #[test]
    fn block_topology_is_dimension_0_fastest() {
        let grid = CartesianGrid::new([4, 4], [2, 2], [0.0; 2], [1.0; 2]).unwrap();
        assert_eq!(grid.num_blocks(), 4);
        // Layout: 0 1 / 2 3 (dimension 0 fastest).
        assert_eq!(grid.block_coords(BlockId(1)), [1, 0]);
        assert_eq!(grid.block_coords(BlockId(2)), [0, 1]);
        assert_eq!(grid.face_neighbor(BlockId(0), 0, 1), Some(BlockId(1)));
        assert_eq!(grid.face_neighbor(BlockId(0), 1, 1), Some(BlockId(2)));
        assert_eq!(grid.face_neighbor(BlockId(0), 0, -1), None);
        assert_eq!(grid.face_neighbor(BlockId(3), 1, 1), None);
        assert_eq!(grid.face_neighbor(BlockId(3), 0, -1), Some(BlockId(2)));
    }

    #[test]
    fn cell_centers_are_cell_centered() {
        let grid = CartesianGrid::new([4], [2], [10.0], [0.5]).unwrap();
        // Global cell i has center origin + (i + 0.5)·h.
        assert_eq!(grid.cell_center(BlockId(0), [0]), [10.25]);
        assert_eq!(grid.cell_center(BlockId(1), [1]), [11.75]);
    }

    #[test]
    fn iteration_is_dimension_0_fastest() {
        let mut seen = Vec::new();
        for_each_interior([2, 2], |idx| seen.push(idx));
        assert_eq!(seen, vec![[0, 0], [1, 0], [0, 1], [1, 1]]);

        let mut boxed = Vec::new();
        for_each_box([-1, 0], [1, 1], |idx| boxed.push(idx));
        assert_eq!(boxed, vec![[-1, 0], [0, 0]]);

        // Degenerate extents visit nothing.
        for_each_interior([0, 3], |_| panic!("empty extent must not iterate"));
        for_each_box([2, 0], [2, 5], |_| panic!("empty box must not iterate"));
    }
}

mod ghosts {
    use super::*;

    /// 1D, two blocks of four cells, u = global cell index. Pins the exact
    /// halo-exchange and mirror-boundary values.
    #[test]
    fn halo_and_mirror_values_are_exact() {
        let grid = CartesianGrid::new([8], [4], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 1);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);

        for b in 0..2 {
            let mut v = state.view_mut(&grid, BlockId(b), u);
            for_each_interior([4], |[i]| {
                v.set([i], (b as usize * 4 + i as usize) as f64);
            });
        }
        fill_ghosts_mirror(&grid, &mut state, u);

        let v0 = state.view(&grid, BlockId(0), u);
        let v1 = state.view(&grid, BlockId(1), u);
        // Physical boundaries mirror the adjacent interior cell.
        assert_eq!(v0.get([-1]), 0.0);
        assert_eq!(v1.get([4]), 7.0);
        // Interior faces exchange neighbor interiors.
        assert_eq!(v0.get([4]), 4.0);
        assert_eq!(v1.get([-1]), 3.0);
    }

    /// A ghost-width-2 field mirrors two layers and exchanges two layers.
    #[test]
    fn wide_ghosts_fill_every_layer() {
        let grid = CartesianGrid::new([8], [4], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let u = builder.register("u", 2);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);

        for b in 0..2 {
            let mut v = state.view_mut(&grid, BlockId(b), u);
            for_each_interior([4], |[i]| {
                v.set([i], (b as usize * 4 + i as usize) as f64);
            });
        }
        fill_ghosts_mirror(&grid, &mut state, u);

        let v0 = state.view(&grid, BlockId(0), u);
        // Mirror: ghost -1-k ← interior k.
        assert_eq!(v0.get([-1]), 0.0);
        assert_eq!(v0.get([-2]), 1.0);
        // Exchange: ghost n+k ← neighbor interior k.
        assert_eq!(v0.get([4]), 4.0);
        assert_eq!(v0.get([5]), 5.0);
    }
}

mod state_ops {
    use super::*;

    fn two_field_state() -> (
        CartesianGrid<1>,
        DenseState,
        legendre::core::state::FieldHandle<f64>,
        legendre::core::state::FieldHandle<f64>,
    ) {
        let grid = CartesianGrid::new([4], [4], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let dynamic = builder.register("dyn", 0);
        let static_f = builder.register_static("static", 0);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        {
            let mut v = state.view_mut(&grid, BlockId(0), dynamic);
            for_each_interior([4], |idx| v.set(idx, 1.0));
        }
        {
            let mut v = state.view_mut(&grid, BlockId(0), static_f);
            for_each_interior([4], |idx| v.set(idx, 5.0));
        }
        (grid, state, dynamic, static_f)
    }

    #[test]
    fn tendency_buffers_carry_no_static_storage() {
        let (grid, state, dynamic, static_f) = two_field_state();
        let tendency = state.like_tendency(&grid, &SystemAllocator);
        assert_eq!(tendency.slab(BlockId(0), static_f).len(), 0);
        assert_eq!(tendency.slab(BlockId(0), dynamic).len(), 4);
        // Full buffers carry everything.
        let full = state.like(&grid, &SystemAllocator);
        assert_eq!(full.slab(BlockId(0), static_f).len(), 4);
    }

    #[test]
    fn axpy_and_noise_skip_static_fields() {
        let (grid, mut state, dynamic, static_f) = two_field_state();
        let mut tendency = state.like_tendency(&grid, &SystemAllocator);
        tendency.slab_mut(BlockId(0), dynamic).fill(2.0);

        state.axpy(10.0, &tendency);
        let vd = state.view(&grid, BlockId(0), dynamic);
        let vs = state.view(&grid, BlockId(0), static_f);
        for_each_interior([4], |idx| {
            assert_eq!(vd.get(idx), 21.0, "dynamic: 1 + 10·2");
            assert_eq!(vs.get(idx), 5.0, "static fields must be skipped");
        });

        // Noise with unit amplitude on the dynamic field only.
        let mut amp = state.like_tendency(&grid, &SystemAllocator);
        amp.slab_mut(BlockId(0), dynamic).fill(1.0);
        let before_static: Vec<f64> = state.slab(BlockId(0), static_f).to_vec();
        let before_dynamic: Vec<f64> = state.slab(BlockId(0), dynamic).to_vec();
        state.add_wiener(&grid, &amp, 1.0, 42, 0, 0);
        assert_eq!(
            state.slab(BlockId(0), static_f),
            &before_static[..],
            "static fields receive no noise"
        );
        assert_ne!(
            state.slab(BlockId(0), dynamic),
            &before_dynamic[..],
            "dynamic fields do"
        );
    }

    #[test]
    fn zero_amplitude_entries_receive_no_noise() {
        let (grid, mut state, dynamic, _) = two_field_state();
        let amp = state.like_tendency(&grid, &SystemAllocator); // all zero
        let before: Vec<f64> = state.slab(BlockId(0), dynamic).to_vec();
        state.add_wiener(&grid, &amp, 1.0, 42, 0, 0);
        assert_eq!(state.slab(BlockId(0), dynamic), &before[..]);
    }

    /// The driver-broadcast contract: one Wiener increment per (cell,
    /// driver), identical across every field the driver moves — even when
    /// the fields' slab layouts differ (different ghost widths) — and
    /// ghost entries never receive noise. Distinct drivers use distinct
    /// increment streams.
    #[test]
    fn wiener_increments_broadcast_per_cell_across_fields() {
        let grid = CartesianGrid::new([4], [4], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let a = builder.register("a", 0);
        let b = builder.register("b", 2); // different ghost width than a
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);

        // Unit amplitude everywhere, ghost entries of b included: the
        // cell-key gate (not the zero-amplitude gate) must keep ghosts
        // noise-free.
        let mut amp = state.like_tendency(&grid, &SystemAllocator);
        amp.slab_mut(BlockId(0), a).fill(1.0);
        amp.slab_mut(BlockId(0), b).fill(1.0);

        state.add_wiener(&grid, &amp, 1.0, 42, 3, 0);
        let va = state.view(&grid, BlockId(0), a);
        let vb = state.view(&grid, BlockId(0), b);
        for_each_interior([4], |idx| {
            assert_ne!(va.get(idx), 0.0, "interior cells receive noise");
            assert_eq!(
                va.get(idx),
                vb.get(idx),
                "same driver, same cell ⇒ same increment on every field"
            );
        });
        for k in [-2isize, -1, 4, 5] {
            assert_eq!(vb.get([k]), 0.0, "ghost entries receive no noise");
        }

        // A different driver index draws from a different stream.
        let mut other = state.like(&grid, &SystemAllocator);
        other.fill_zero();
        other.add_wiener(&grid, &amp, 1.0, 42, 3, 1);
        let vo = other.view(&grid, BlockId(0), a);
        let mut all_equal = true;
        for_each_interior([4], |idx| {
            if vo.get(idx) != va.get(idx) {
                all_equal = false;
            }
        });
        assert!(!all_equal, "drivers must have independent streams");
    }

    #[test]
    fn view_split_borrows_both_orders() {
        let grid = CartesianGrid::new([2], [2], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let a = builder.register("a", 0);
        let b = builder.register("b", 0);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        state.slab_mut(BlockId(0), a).fill(1.0);
        state.slab_mut(BlockId(0), b).fill(2.0);

        let (layout, blocks) = state.split_blocks_mut();
        let mut block = blocks[0].bind_mut(layout);
        {
            // write earlier field, read later field
            let (mut wa, rb) = block.view_split(&grid, BlockId(0), a, b);
            wa.set([0], rb.get([0]) + 10.0);
        }
        {
            // write later field, read earlier field
            let (mut wb, ra) = block.view_split(&grid, BlockId(0), b, a);
            wb.set([1], ra.get([1]) + 100.0);
        }
        assert_eq!(state.view(&grid, BlockId(0), a).get([0]), 12.0);
        assert_eq!(state.view(&grid, BlockId(0), b).get([1]), 101.0);
    }

    #[test]
    #[should_panic(expected = "cannot split a field with itself")]
    fn view_split_rejects_same_field() {
        let grid = CartesianGrid::new([2], [2], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let a = builder.register("a", 0);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        let (layout, blocks) = state.split_blocks_mut();
        let mut block = blocks[0].bind_mut(layout);
        let _ = block.view_split(&grid, BlockId(0), a, a);
    }

    #[test]
    #[should_panic(expected = "two distinct")]
    fn slab_pair_rejects_same_block() {
        let grid = CartesianGrid::new([4], [4], [0.0], [1.0]).unwrap();
        let mut builder = StateBuilder::<f64>::new();
        let a = builder.register("a", 0);
        let mut state: DenseState = builder.build(&grid, &SystemAllocator);
        let _ = state.slab_pair_mut(BlockId(0), BlockId(0), a);
    }

    #[test]
    fn fill_zero_resets_every_slab() {
        let (grid, mut state, dynamic, static_f) = two_field_state();
        state.fill_zero();
        let vd = state.view(&grid, BlockId(0), dynamic);
        let vs = state.view(&grid, BlockId(0), static_f);
        for_each_interior([4], |idx| {
            assert_eq!(vd.get(idx), 0.0);
            assert_eq!(vs.get(idx), 0.0);
        });
    }
}
