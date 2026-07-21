//! Round-trip validation of the Parquet snapshot format: the per-epoch
//! static file (coordinates + time-invariant fields) and the slim per-step
//! snapshot (step, t, epoch, dynamic fields), joined by row order.

use legendre::{
    core::{
        observer::Observer,
        state::{FieldHandle, State, StateBuilder},
        storage::{DenseStorage, SystemAllocator},
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    io::parquet::ParquetObserver,
};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;

type DenseState = State<f64, DenseStorage<f64>>;

/// Read one column of doubles (by name) from every row of a parquet file.
fn read_doubles(path: &std::path::Path, column: &str) -> Vec<f64> {
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let schema = reader.metadata().file_metadata().schema();
    let idx = schema
        .get_fields()
        .iter()
        .position(|f| f.name() == column)
        .unwrap_or_else(|| panic!("column {column} not found"));
    reader
        .get_row_iter(None)
        .unwrap()
        .map(|row| row.unwrap().get_double(idx).unwrap())
        .collect()
}

fn read_longs(path: &std::path::Path, column: &str) -> Vec<i64> {
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let schema = reader.metadata().file_metadata().schema();
    let idx = schema
        .get_fields()
        .iter()
        .position(|f| f.name() == column)
        .unwrap_or_else(|| panic!("column {column} not found"));
    reader
        .get_row_iter(None)
        .unwrap()
        .map(|row| row.unwrap().get_long(idx).unwrap())
        .collect()
}

/// Build a two-field state (dynamic phi, static theta0) with distinct,
/// position-dependent values.
fn build_state(grid: &CartesianGrid<2>) -> (DenseState, FieldHandle<f64>, FieldHandle<f64>) {
    let mut builder = StateBuilder::<f64>::new();
    let phi = builder.register("phi", 1);
    let theta = builder.register_static("theta0", 0);
    let mut state: DenseState = builder.build(grid, &SystemAllocator);
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        {
            let mut v = state.view_mut(grid, block, phi);
            for_each_interior(grid.block_cells(), |idx| {
                let [x, y] = grid.cell_center(block, idx);
                v.set(idx, 100.0f64.mul_add(x, y));
            });
        }
        {
            let mut v = state.view_mut(grid, block, theta);
            for_each_interior(grid.block_cells(), |idx| {
                let [x, y] = grid.cell_center(block, idx);
                v.set(idx, -1000.0f64.mul_add(y, x));
            });
        }
    }
    (state, phi, theta)
}

/// The writer's row order (block-major, dimension-0 fastest) applied to any
/// per-cell function.
fn in_row_order(grid: &CartesianGrid<2>, mut f: impl FnMut(BlockId, [isize; 2])) {
    for b in 0..grid.num_blocks() {
        let block = BlockId(b as u32);
        for_each_interior(grid.block_cells(), |idx| f(block, idx));
    }
}

#[test]
fn snapshot_roundtrips_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let grid = CartesianGrid::new([4, 4], [2, 2], [0.0, 0.0], [0.5, 0.5]).unwrap();
    let (state, phi, theta) = build_state(&grid);

    let mut obs = ParquetObserver::new(grid.clone(), vec![("phi", phi)], 2, dir.path())
        .unwrap()
        .with_static(vec![("theta0", theta)]);

    // Cadence: step 1 always fires; step 3 is skipped; step 4 fires.
    obs.observe(1, 0.25, 0, &grid, &state);
    obs.observe(3, 0.75, 0, &grid, &state);
    obs.observe(4, 1.0, 0, &grid, &state);

    let static_path = dir.path().join("static_0000000.parquet");
    let snap1 = dir.path().join("snap_0000001.parquet");
    let snap3 = dir.path().join("snap_0000003.parquet");
    let snap4 = dir.path().join("snap_0000004.parquet");
    assert!(
        static_path.exists(),
        "per-epoch static file must be written"
    );
    assert!(snap1.exists() && snap4.exists());
    assert!(!snap3.exists(), "off-cadence steps must not snapshot");

    // Expected values in writer row order.
    let (mut xs, mut ys, mut phis, mut thetas) = (vec![], vec![], vec![], vec![]);
    in_row_order(&grid, |block, idx| {
        let [x, y] = grid.cell_center(block, idx);
        xs.push(x);
        ys.push(y);
        phis.push(100.0f64.mul_add(x, y));
        thetas.push(-(1000.0f64.mul_add(y, x)));
    });

    // Static file: coordinates + static fields, exact (doubles round-trip
    // bit-for-bit through parquet).
    assert_eq!(read_doubles(&static_path, "x"), xs);
    assert_eq!(read_doubles(&static_path, "y"), ys);
    assert_eq!(read_doubles(&static_path, "theta0"), thetas);

    // Snapshots: constant prefix columns + dynamic fields, joined by row
    // order to the static file.
    assert_eq!(read_doubles(&snap1, "phi"), phis);
    assert_eq!(read_doubles(&snap4, "phi"), phis);
    assert_eq!(read_longs(&snap1, "step"), vec![1; 16]);
    assert_eq!(read_longs(&snap1, "epoch"), vec![0; 16]);
    assert_eq!(read_doubles(&snap4, "t"), vec![1.0; 16]);
}

#[test]
fn constructor_rejects_empty_field_lists() {
    let dir = tempfile::tempdir().unwrap();
    let grid = CartesianGrid::new([4, 4], [2, 2], [0.0, 0.0], [0.5, 0.5]).unwrap();
    assert!(ParquetObserver::new(grid, vec![], 1, dir.path()).is_err());
}
