#!/usr/bin/env python3
"""Render a legendre Parquet run directory to a movie.

Reads the snapshot format written by ``legendre::io::parquet::ParquetObserver``:

    run_dir/
    |-- static_<epoch>.parquet   x, y[, z], level, patch + static fields
    `-- snap_<step>.parquet      step, t, epoch + dynamic fields

Static files carry the coordinates (and any time-invariant fields) once per
grid epoch; snapshots are joined to them by row order. 3D runs are rendered
as the mid-plane slice normal to z.

AMR runs are composited finest-wins: every cell paints an
(h_level / h_finest)^2 pixel block on the finest-level lattice, coarse
levels first — a uniform grid is just the single-level case of the same
path. Adaptive runs write one static file per regrid epoch; each frame uses
its snapshot's epoch. Pass --patches to outline refined patches.

Typical usage (after `python3 -m venv .venv && .venv/bin/pip install -r
scripts/requirements.txt`):

    .venv/bin/python scripts/render_model_c.py data/model_c --out dendrite.mp4
    .venv/bin/python scripts/render_model_c.py data/model_c --field u
    .venv/bin/python scripts/render_model_c.py data/model_c --grains

Requires ffmpeg on PATH for .mp4 output; pass an --out ending in .gif to use
the (slower, larger) Pillow writer instead.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

SNAP_RE = re.compile(r"snap_(\d+)\.parquet$")
STATIC_RE = re.compile(r"static_(\d+)\.parquet$")


def fail(msg: str) -> "sys.NoReturn":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def find_run_files(run_dir: Path) -> tuple[dict[int, Path], list[tuple[int, Path]]]:
    """Return ({epoch: static_path}, [(step, snap_path), ...] sorted by step)."""
    statics: dict[int, Path] = {}
    snaps: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if m := STATIC_RE.search(p.name):
            statics[int(m.group(1))] = p
        elif m := SNAP_RE.search(p.name):
            snaps.append((int(m.group(1)), p))
    if not statics:
        fail(
            f"no static_<epoch>.parquet in {run_dir} — is this a legendre run directory?"
        )
    if not snaps:
        fail(f"no snap_<step>.parquet files in {run_dir}")
    snaps.sort()
    return statics, snaps


class GridIndex:
    """Maps the writer's row order onto a dense (ny, nx) image grid.

    Row order is block-major and dimension-0 fastest, so rows are *not* in
    raster order; instead of assuming the layout, values are painted through
    their coordinates, which is exact for any block decomposition. Under AMR
    every cell paints an (h_level / h_finest)^2 pixel block on the
    finest-level lattice, coarse levels first (finest wins); a uniform grid
    is the single-level case of the same path.
    """

    def __init__(self, static_path: Path):
        table = pq.read_table(static_path)
        self.columns = set(table.column_names)
        x = table["x"].to_numpy()
        y = table["y"].to_numpy()
        self.is_3d = "z" in self.columns
        level = (
            table["level"].to_numpy().astype(int)
            if "level" in self.columns
            else np.zeros(len(x), dtype=int)
        )
        patch = (
            table["patch"].to_numpy().astype(int)
            if "patch" in self.columns
            else np.zeros(len(x), dtype=int)
        )

        if self.is_3d:
            if level.max() > 0:
                fail("3D AMR rendering is not supported yet")
            # Mid-plane slice: keep rows whose z is the unique value closest
            # to the domain midpoint.
            z = table["z"].to_numpy()
            zs = np.unique(z)
            z_mid = zs[len(zs) // 2]
            self.mask = z == z_mid
            self.z_mid = z_mid
            x, y, level, patch = x[self.mask], y[self.mask], level[self.mask], patch[self.mask]
        else:
            self.mask = slice(None)

        # Per-level spacing from that level's own coordinate lattice.
        levels = np.unique(level)
        spacing = {}
        for lv in levels:
            xs_l = np.unique(x[level == lv])
            spacing[lv] = float(np.min(np.diff(xs_l))) if len(xs_l) > 1 else 1.0
        h_fine = spacing[levels.max()]

        # Domain edges from cell centers +- h/2, then the finest lattice.
        half = np.array([spacing[lv] / 2.0 for lv in level])
        lo_x, hi_x = float(np.min(x - half)), float(np.max(x + half))
        lo_y, hi_y = float(np.min(y - half)), float(np.max(y + half))
        nx = round((hi_x - lo_x) / h_fine)
        ny = round((hi_y - lo_y) / h_fine)
        self.extent = (lo_x, hi_x, lo_y, hi_y)
        self.shape = (ny, nx)

        # Painting order: coarse first, so finer levels overwrite.
        self.order = np.argsort(level, kind="stable")
        lvo = level[self.order]
        # Pixel-block origin and size per (ordered) cell.
        k = np.array([round(spacing[lv] / h_fine) for lv in lvo])
        self.k = k
        self.ix0 = np.round((x[self.order] - half[self.order] - lo_x) / h_fine).astype(int)
        self.iy0 = np.round((y[self.order] - half[self.order] - lo_y) / h_fine).astype(int)

        # Patch outlines (levels >= 1) for --patches.
        self.patch_boxes = []
        for lv in levels:
            if lv == 0:
                continue
            for pid in np.unique(patch[level == lv]):
                sel = (level == lv) & (patch == pid)
                h = spacing[lv]
                x0, x1 = float(np.min(x[sel]) - h / 2), float(np.max(x[sel]) + h / 2)
                y0, y1 = float(np.min(y[sel]) - h / 2), float(np.max(y[sel]) + h / 2)
                self.patch_boxes.append((int(lv), x0, y0, x1 - x0, y1 - y0))

        self.statics = {
            name: self.to_image(table[name].to_numpy()[self.mask])
            for name in self.columns - {"x", "y", "z", "level", "patch"}
        }

    def to_image(self, values: np.ndarray) -> np.ndarray:
        img = np.full(self.shape, np.nan)
        vals = np.asarray(values)[self.order]
        for kk in np.unique(self.k):
            sel = self.k == kk
            for dy in range(kk):
                for dx in range(kk):
                    img[self.iy0[sel] + dy, self.ix0[sel] + dx] = vals[sel]
        return img


def downsample(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Stride-subsample so max(shape) <= max_dim (rendering only)."""
    stride = max(1, int(np.ceil(max(img.shape) / max_dim)))
    return img[::stride, ::stride]


def grain_composite(
    phi: np.ndarray, theta0: np.ndarray, cmap_melt
) -> np.ndarray:
    """Color solid (phi > 0) by grain orientation hue, melt by phi."""
    from matplotlib.colors import hsv_to_rgb

    rgb = cmap_melt((phi + 1.0) / 2.0)[..., :3]
    solid = phi > 0.0
    # theta0 in [0, pi/2) -> hue in [0, 1).
    hue = (theta0 / (np.pi / 2.0)) % 1.0
    sat = np.full_like(hue, 0.85)
    val = np.clip((phi + 1.0) / 2.0, 0.0, 1.0)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb[solid] = hsv_to_rgb(hsv[solid])
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="directory of a legendre Parquet run")
    ap.add_argument("--out", type=Path, default=None, help="output movie path (.mp4 or .gif; default <run_dir>.mp4)")
    ap.add_argument("--field", default="phi", help="dynamic field to render (default: phi)")
    ap.add_argument("--fps", type=int, default=12, help="frames per second (default: 12)")
    ap.add_argument("--cmap", default="magma", help="matplotlib colormap (default: magma)")
    ap.add_argument("--max-dim", type=int, default=1080, help="downsample frames above this size (default: 1080)")
    ap.add_argument("--grains", action="store_true", help="color solid regions by grain orientation (needs a theta0 static field)")
    ap.add_argument("--patches", action="store_true", help="outline refined AMR patches")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        fail(f"{args.run_dir} is not a directory")
    statics, snaps = find_run_files(args.run_dir)
    out = args.out or args.run_dir.with_suffix(".mp4")

    # Uniform-grid runs have one epoch; index every epoch that appears.
    grids = {epoch: GridIndex(path) for epoch, path in statics.items()}
    first = next(iter(grids.values()))
    if args.grains and "theta0" not in first.statics:
        fail("--grains needs a theta0 static field (run the example with --orient)")

    # Global color scale from the first and last snapshots.
    def field_column(path: Path) -> np.ndarray:
        table = pq.read_table(path, columns=[args.field, "epoch"])
        if args.field not in table.column_names:
            fail(f"field {args.field!r} not in {path.name}")
        return table[args.field].to_numpy(), int(table["epoch"][0].as_py())

    lo, hi = np.inf, -np.inf
    for _, path in (snaps[0], snaps[-1]):
        vals, _ = field_column(path)
        lo, hi = min(lo, float(np.min(vals))), max(hi, float(np.max(vals)))
    if lo == hi:
        hi = lo + 1.0

    cmap = plt.get_cmap(args.cmap)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
    title = ax.set_title("", fontsize=10, family="monospace")

    def frame_image(step_path: Path):
        vals, epoch = field_column(step_path)
        grid = grids[epoch]
        t = pq.read_table(step_path, columns=["t"])["t"][0].as_py()
        img = grid.to_image(np.asarray(vals)[grid.mask])
        if args.grains:
            rgb = grain_composite(img, grid.statics["theta0"], cmap)
            return downsample(rgb, args.max_dim), t
        return downsample(img, args.max_dim), t

    img0, t0 = frame_image(snaps[0][1])
    if args.grains:
        artist = ax.imshow(img0, origin="lower", extent=first.extent)
    else:
        artist = ax.imshow(
            img0, origin="lower", extent=first.extent, cmap=cmap, vmin=lo, vmax=hi
        )

    outline_state = {"epoch": None, "artists": []}

    def draw_patches(epoch: int) -> None:
        if not args.patches or outline_state["epoch"] == epoch:
            return
        from matplotlib.patches import Rectangle

        for a in outline_state["artists"]:
            a.remove()
        outline_state["artists"] = [
            ax.add_patch(
                Rectangle((x0, y0), w, h, fill=False, ec="white", lw=0.6, alpha=0.8)
            )
            for (_lv, x0, y0, w, h) in grids[epoch].patch_boxes
        ]
        outline_state["epoch"] = epoch

    def update(i: int):
        step, path = snaps[i]
        img, t = frame_image(path)
        epoch = int(pq.read_table(path, columns=["epoch"])["epoch"][0].as_py())
        draw_patches(epoch)
        artist.set_data(img)
        title.set_text(f"{args.field}  step {step}  t = {t:.1f}")
        return artist, title

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=len(snaps), blit=False)

    if out.suffix == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
    else:
        if not animation.FFMpegWriter.isAvailable():
            fail("ffmpeg not found on PATH; install it or use an --out ending in .gif")
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)

    print(f"rendering {len(snaps)} frames from {args.run_dir} -> {out}")
    anim.save(out, writer=writer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
