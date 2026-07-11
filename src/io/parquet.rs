//! Parquet snapshot observer for Cartesian simulations in any dimension
//! (coordinate columns `x`, `y`, `z` up to D = 3).
//!
//! ## File layout: per-epoch statics + slim per-step snapshots
//!
//! ```text
//! run_dir/
//! ├── static_0000000.parquet   x, y[, z], static fields (θ₀, …)  — one per grid epoch
//! ├── snap_0000001.parquet     step, t, epoch, dynamic fields
//! ├── snap_0004000.parquet     …
//! ```
//!
//! Coordinates and model-declared static fields never change between
//! regrids, so they are written **once per grid epoch** and joined to
//! snapshots by row order (block-major, interior cells, deterministic).
//! Every snapshot carries a constant `epoch` column (RLE-encoded, ~free)
//! naming the static file its rows align with. A uniform-grid run has
//! exactly one epoch; under AMR each regrid bumps the epoch and emits a
//! fresh static file, so the on-disk contract already accommodates
//! refinement. This cuts both steady-state disk and — more importantly —
//! the writer's transient memory, which previously materialized coordinate
//! columns for every snapshot.
//!
//! ## Bounded memory
//!
//! Rows are assembled and flushed in **bounded row groups** (4M rows,
//! ≈ 32 MB per column): the writer's footprint is flat in domain size,
//! never the multi-GB whole-domain columns that can evict the simulation's
//! own state from RAM on large runs.
//!
//! One file per snapshot (rather than row groups appended to a single file)
//! keeps every completed snapshot readable even if the run is killed, since
//! a parquet file is only valid once its footer is written.

use crate::{
    core::{
        observer::{Observer, SnapshotSink},
        state::{FieldHandle, State},
        storage::StorageBackend,
    },
    geometry::{
        cartesian::{CartesianGrid, for_each_interior},
        grid::{BlockId, Grid},
    },
    io::ObserverError,
};
use parquet::{
    basic::Compression,
    data_type::{DoubleType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use std::{fs::File, path::PathBuf, sync::Arc};

const COORD_NAMES: [&str; 3] = ["x", "y", "z"];

/// Rows per parquet row group: bounds the writer's transient memory.
const ROWS_PER_GROUP: usize = 1 << 22;

/// Writes state snapshots as Parquet files; see the module docs for the
/// on-disk layout.
pub struct ParquetObserver<const D: usize = 2> {
    grid: CartesianGrid<D>,
    dynamic: Vec<(&'static str, FieldHandle<f64>)>,
    statics: Vec<(&'static str, FieldHandle<f64>)>,
    every: u64,
    dir: PathBuf,
    props: Arc<WriterProperties>,
    /// Grid generation. Fixed at 0 for uniform grids; AMR regrids will bump
    /// this (and reset `static_written`) when adaptive grids land.
    epoch: u64,
    static_written: bool,
}

impl<const D: usize> ParquetObserver<D> {
    /// Snapshot the *dynamic* `fields` every `every` steps (plus step 1)
    /// into `dir`. Coordinates are always emitted to the per-epoch static
    /// file; add time-invariant fields to it with [`Self::with_static`].
    ///
    /// # Errors
    ///
    /// Returns an [`ObserverError`] if `D` is not 1, 2, or 3, if `fields`
    /// is empty, or if `dir` cannot be created.
    pub fn new(
        grid: CartesianGrid<D>,
        fields: Vec<(&'static str, FieldHandle<f64>)>,
        every: u64,
        dir: impl Into<PathBuf>,
    ) -> Result<Self, ObserverError> {
        if !(D >= 1 && D <= 3) {
            return Err(ObserverError::InvalidDimension(D));
        }
        if fields.is_empty() {
            return Err(ObserverError::NoValidFields);
        }
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            grid,
            dynamic: fields,
            statics: Vec::new(),
            every,
            dir,
            props: Arc::new(
                WriterProperties::builder()
                    .set_compression(Compression::SNAPPY)
                    .build(),
            ),
            epoch: 0,
            static_written: false,
        })
    }

    /// Declare time-invariant fields (e.g. a grain-orientation map θ₀) to
    /// be written once per grid epoch instead of once per snapshot.
    #[must_use]
    pub fn with_static(mut self, fields: Vec<(&'static str, FieldHandle<f64>)>) -> Self {
        self.statics = fields;
        self
    }

    fn schema(cols: &[String]) -> Result<Arc<parquet::schema::types::Type>, ObserverError> {
        let body: String = cols.concat();
        Ok(Arc::new(parse_message_type(&format!(
            "message snapshot {{ {body} }}"
        ))?))
    }

    /// Stream field columns block by block, flushing a row group whenever
    /// the buffers reach [`ROWS_PER_GROUP`]. `prefix` writes the leading
    /// constant columns (step/t/epoch) for `n` rows of a group.
    fn write_streamed<S, P>(
        &self,
        path: &PathBuf,
        schema: Arc<parquet::schema::types::Type>,
        state: &State<f64, S>,
        with_coords: bool,
        fields: &[(&'static str, FieldHandle<f64>)],
        prefix: P,
    ) -> Result<(), ObserverError>
    where
        S: StorageBackend<f64>,
        P: Fn(
            &mut parquet::file::writer::SerializedRowGroupWriter<'_, File>,
            usize,
        ) -> Result<(), ObserverError>,
    {
        let file = File::create(path)?;
        let mut writer = SerializedFileWriter::new(file, schema, self.props.clone())?;

        let block_cells: usize = self.grid.block_cells().iter().product();
        let cap = ROWS_PER_GROUP + block_cells;
        let n_coords = if with_coords { D } else { 0 };
        let mut bufs: Vec<Vec<f64>> = (0..n_coords + fields.len())
            .map(|_| Vec::with_capacity(cap))
            .collect();

        let mut flush = |bufs: &mut Vec<Vec<f64>>| -> Result<(), ObserverError> {
            let n = bufs[0].len();
            if n == 0 {
                return Ok(());
            }
            let mut rg = writer.next_row_group()?;
            prefix(&mut rg, n)?;
            for data in bufs.iter() {
                let mut col = rg
                    .next_column()?
                    .ok_or_else(|| ObserverError::Internal("missing column"))?;
                col.typed::<DoubleType>().write_batch(data, None, None)?;
                col.close()?;
            }
            rg.close()?;
            for b in bufs.iter_mut() {
                b.clear();
            }
            Ok(())
        };

        for b in 0..self.grid.num_blocks() {
            let block = BlockId(b as u32);
            if with_coords {
                for_each_interior(self.grid.block_cells(), |idx| {
                    let center = self.grid.cell_center(block, idx);
                    for (d, buf) in bufs[..D].iter_mut().enumerate() {
                        buf.push(center[d]);
                    }
                });
            }
            for ((_, handle), buf) in fields.iter().zip(&mut bufs[n_coords..]) {
                let v = state.view(&self.grid, block, *handle);
                for_each_interior(self.grid.block_cells(), |idx| buf.push(v.get(idx)));
            }
            if bufs[0].len() >= ROWS_PER_GROUP {
                flush(&mut bufs)?;
            }
        }
        flush(&mut bufs)?;
        writer.close()?;
        Ok(())
    }

    /// Coordinates + static fields for the current grid epoch.
    fn write_static<S: StorageBackend<f64>>(
        &self,
        state: &State<f64, S>,
    ) -> Result<(), ObserverError> {
        let mut cols: Vec<String> = COORD_NAMES[..D]
            .iter()
            .map(|n| format!("required double {n};\n"))
            .collect();
        cols.extend(
            self.statics
                .iter()
                .map(|(n, _)| format!("required double {n};\n")),
        );
        let path = self.dir.join(format!("static_{:07}.parquet", self.epoch));
        self.write_streamed(
            &path,
            Self::schema(&cols)?,
            state,
            true,
            &self.statics,
            |_rg, _n| Ok(()),
        )
    }

    fn write_dynamic<S: StorageBackend<f64>>(
        &self,
        step: u64,
        t: f64,
        state: &State<f64, S>,
    ) -> Result<(), ObserverError> {
        let mut cols = vec![
            "required int64 step;\n".to_string(),
            "required double t;\n".to_string(),
            "required int64 epoch;\n".to_string(),
        ];
        cols.extend(
            self.dynamic
                .iter()
                .map(|(n, _)| format!("required double {n};\n")),
        );
        let path = self.dir.join(format!("snap_{step:07}.parquet"));
        let epoch = self.epoch;
        self.write_streamed(
            &path,
            Self::schema(&cols)?,
            state,
            false,
            &self.dynamic,
            move |rg, n| {
                let steps = vec![step as i64; n];
                let ts = vec![t; n];
                let epochs = vec![epoch as i64; n];
                let mut col = rg
                    .next_column()?
                    .ok_or_else(|| ObserverError::Internal("missing step column"))?;
                col.typed::<Int64Type>().write_batch(&steps, None, None)?;
                col.close()?;
                let mut col = rg
                    .next_column()?
                    .ok_or_else(|| ObserverError::Internal("missing t column"))?;
                col.typed::<DoubleType>().write_batch(&ts, None, None)?;
                col.close()?;
                let mut col = rg
                    .next_column()?
                    .ok_or_else(|| ObserverError::Internal("missing epoch column"))?;
                col.typed::<Int64Type>().write_batch(&epochs, None, None)?;
                col.close()?;
                Ok(())
            },
        )
    }

    fn write_snapshot<S: StorageBackend<f64>>(
        &mut self,
        step: u64,
        t: f64,
        state: &State<f64, S>,
    ) -> Result<(), ObserverError> {
        if !self.static_written {
            self.write_static(state)?;
            self.static_written = true;
        }
        self.write_dynamic(step, t, state)
    }
}

impl<S: StorageBackend<f64>, const D: usize> Observer<f64, S> for ParquetObserver<D> {
    fn observe(&mut self, step: u64, t: f64, state: &State<f64, S>) {
        if step != 1 && !step.is_multiple_of(self.every) {
            return;
        }
        if let Err(e) = self.write_snapshot(step, t, state) {
            tracing::error!(step, "parquet snapshot failed: {e:#}");
        }
    }
}

/// As a [`SnapshotSink`] the observer writes every snapshot it receives —
/// cadence is owned by the [`crate::core::observer::AsyncObserver`] feeding
/// it (construct with `every = 1` in that case).
impl<S: StorageBackend<f64>, const D: usize> SnapshotSink<f64, S> for ParquetObserver<D> {
    fn consume(&mut self, step: u64, t: f64, state: &State<f64, S>) {
        if let Err(e) = self.write_snapshot(step, t, state) {
            tracing::error!(step, "parquet snapshot failed: {e:#}");
        }
    }
}
