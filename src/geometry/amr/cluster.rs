//! Berger–Rigoutsos point clustering (IEEE Trans. SMC 21(5), 1991): the
//! grid-generation kernel of Clawpack-lineage AMR — a pure function from
//! flagged cells to disjoint enclosing boxes.
//!
//! The algorithm, per box (worklist, starting from the flags' bounding
//! box, every candidate first shrunk to its flags' bounds):
//!
//! 1. **Accept** if efficiency = flagged/cells ≥ the threshold.
//! 2. **Holes**: if any per-axis *signature* (count of flags per
//!    coordinate plane) has an interior zero, split at the hole nearest
//!    the box middle — this separates islands, and is exploited before
//!    any inflection search.
//! 3. **Inflections**: otherwise split at the zero crossing of the
//!    signature's second difference with the largest jump — the most
//!    prominent edge in the flag distribution.
//! 4. **Bisection fallback**: otherwise bisect the longest axis (the
//!    paper's own fix for its 45°-stripe anomaly, which otherwise floors
//!    efficiency at 50%).
//! 5. Accept unsplittable boxes as-is.
//!
//! Splits partition the flags, so the output boxes are **disjoint** and
//! **cover** every flag by construction; both sides of any split are
//! nonempty (a shrunk box has flags on every face), so the recursion
//! terminates. Cost is O(boxes · flags), run at regrid cadence — never in
//! the step hot loop.

/// An axis-aligned box of cells, half-open: `lo[d] ≤ c[d] < hi[d]`.
///
/// Coordinates are *cell indices at one refinement level*; the box knows
/// nothing about physical geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellBox<const D: usize> {
    /// Inclusive lower cell index per axis.
    pub lo: [i64; D],
    /// Exclusive upper cell index per axis.
    pub hi: [i64; D],
}

impl<const D: usize> CellBox<D> {
    /// The tight bounding box of a nonempty set of cells.
    ///
    /// # Panics
    ///
    /// Panics if `cells` is empty.
    #[must_use]
    pub fn bounding(cells: &[[i64; D]]) -> Self {
        assert!(!cells.is_empty(), "bounding box of no cells");
        let mut lo = cells[0];
        let mut hi = cells[0];
        for c in cells {
            for d in 0..D {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        for h in &mut hi {
            *h += 1; // half-open
        }
        Self { lo, hi }
    }

    /// Extent (number of cells) along axis `d`.
    #[inline]
    #[must_use]
    pub const fn extent(&self, d: usize) -> u64 {
        (self.hi[d] - self.lo[d]) as u64
    }

    /// Total number of cells (saturating on overflow).
    #[must_use]
    pub fn cells(&self) -> u64 {
        (0..D).fold(1u64, |acc, d| acc.saturating_mul(self.extent(d)))
    }

    /// Whether `c` lies inside the box.
    #[inline]
    #[must_use]
    pub fn contains(&self, c: [i64; D]) -> bool {
        (0..D).all(|d| self.lo[d] <= c[d] && c[d] < self.hi[d])
    }

    /// Whether the two boxes share any cell.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        (0..D).all(|d| self.lo[d] < other.hi[d] && other.lo[d] < self.hi[d])
    }

    /// The box grown by `b` cells on every face (the regrid *buffer
    /// zone*: what lets a moving feature stay inside its patch between
    /// regrids).
    #[must_use]
    pub const fn grown(&self, b: i64) -> Self {
        let mut out = *self;
        let mut d = 0;
        while d < D {
            out.lo[d] -= b;
            out.hi[d] += b;
            d += 1;
        }
        out
    }

    /// The box clipped to `domain` (empty extents possible if disjoint).
    #[must_use]
    pub fn clipped(&self, domain: &Self) -> Self {
        let mut out = *self;
        for d in 0..D {
            out.lo[d] = out.lo[d].max(domain.lo[d]);
            out.hi[d] = out.hi[d].min(domain.hi[d]).max(out.lo[d]);
        }
        out
    }
}

/// Knobs of the clustering algorithm.
#[derive(Debug, Clone, Copy)]
pub struct ClusterParams {
    /// Minimum acceptable flagged/cells ratio of an output box, in
    /// (0, 1]. Berger–Rigoutsos report 0.85–1.00 achieved in practice;
    /// 0.7–0.85 are typical thresholds.
    pub efficiency: f64,
    /// Minimum box width a *split* may produce, per axis (bounding-box
    /// shrinkage may still yield thinner boxes — a single row of flags is
    /// one cell tall). Keeps work items worth dispatching.
    pub min_width: usize,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self {
            efficiency: 0.8,
            min_width: 4,
        }
    }
}

/// Cluster flagged cells into disjoint axis-aligned boxes covering all of
/// them (Berger–Rigoutsos; see the module docs for the algorithm).
///
/// `flags` is reordered in place (splits partition it, quicksort-style);
/// the multiset of cells is unchanged. Duplicate cells are permitted but
/// inflate efficiency estimates — pass each flagged cell once.
#[must_use]
pub fn cluster<const D: usize>(flags: &mut [[i64; D]], params: &ClusterParams) -> Vec<CellBox<D>> {
    debug_assert!(
        params.efficiency > 0.0 && params.efficiency <= 1.0,
        "efficiency must be in (0, 1]"
    );
    debug_assert!(params.min_width >= 1, "min_width must be at least 1");
    let mut out = Vec::new();
    if !flags.is_empty() {
        split_or_accept(flags, params, &mut out);
    }
    out
}

fn split_or_accept<const D: usize>(
    flags: &mut [[i64; D]],
    params: &ClusterParams,
    out: &mut Vec<CellBox<D>>,
) {
    let bx = CellBox::bounding(flags);
    let efficiency = flags.len() as f64 / bx.cells() as f64;
    if efficiency >= params.efficiency {
        out.push(bx);
        return;
    }

    // Per-axis signatures: flags per coordinate plane.
    let sigs = signatures(&bx, flags);

    let split = find_hole(&bx, &sigs)
        .or_else(|| find_inflection(&bx, &sigs, params.min_width))
        .or_else(|| bisection(&bx, params.min_width));
    let Some((axis, plane)) = split else {
        out.push(bx); // unsplittable: accept below threshold
        return;
    };

    // Partition in place: cells with coordinate < plane go left. A shrunk
    // box has flags on every face, so both sides are provably nonempty
    // for hole/inflection/bisection splits alike.
    let mid = partition_in_place(flags, axis, plane);
    debug_assert!(mid > 0 && mid < flags.len(), "split produced an empty side");
    let (left, right) = flags.split_at_mut(mid);
    split_or_accept(left, params, out);
    split_or_accept(right, params, out);
}

/// Signature `Σ_d[i]`: number of flags in plane `lo[d] + i` of axis `d`.
fn signatures<const D: usize>(bx: &CellBox<D>, flags: &[[i64; D]]) -> [Vec<u32>; D] {
    let mut sigs: [Vec<u32>; D] = std::array::from_fn(|d| vec![0; bx.extent(d) as usize]);
    for c in flags {
        for d in 0..D {
            sigs[d][(c[d] - bx.lo[d]) as usize] += 1;
        }
    }
    sigs
}

/// The most central interior zero of any signature, as a split plane
/// `(axis, plane)` meaning "cells with coordinate < plane go left".
/// Splitting *at* the hole cell puts it (empty) on the right; the
/// subsequent bounding-box shrink discards it.
fn find_hole<const D: usize>(bx: &CellBox<D>, sigs: &[Vec<u32>; D]) -> Option<(usize, i64)> {
    let mut best: Option<(usize, i64, u64)> = None; // (axis, plane, centrality)
    for (d, sig) in sigs.iter().enumerate() {
        let n = sig.len();
        for (i, &s) in sig.iter().enumerate().take(n - 1).skip(1) {
            if s == 0 {
                // Distance to the nearer end: larger is more central.
                let centrality = (i as u64).min((n - 1 - i) as u64);
                if best.is_none_or(|(_, _, c)| centrality > c) {
                    best = Some((d, bx.lo[d] + i as i64, centrality));
                }
            }
        }
    }
    best.map(|(d, plane, _)| (d, plane))
}

/// The strongest zero crossing of any signature's second difference,
/// among split positions leaving at least `min_width` cells on each side.
fn find_inflection<const D: usize>(
    bx: &CellBox<D>,
    sigs: &[Vec<u32>; D],
    min_width: usize,
) -> Option<(usize, i64)> {
    let mut best: Option<(usize, i64, i64, u64)> = None; // (axis, plane, strength, centrality)
    for (d, sig) in sigs.iter().enumerate() {
        let n = sig.len();
        if n < 4 {
            continue; // no two interior second differences to compare
        }
        // Δ[i] = Σ[i+1] − 2Σ[i] + Σ[i−1], defined for i in 1..n−1.
        let delta = |i: usize| -> i64 {
            i64::from(sig[i + 1]) - 2 * i64::from(sig[i]) + i64::from(sig[i - 1])
        };
        for i in 1..n - 2 {
            let (a, b) = (delta(i), delta(i + 1));
            if a * b < 0 {
                // Crossing between cells i and i+1: split plane i+1.
                let plane = i + 1;
                if plane < min_width || n - plane < min_width {
                    continue;
                }
                let strength = (b - a).abs();
                let centrality = (plane as u64).min((n - plane) as u64);
                let better = best
                    .is_none_or(|(_, _, s, c)| strength > s || (strength == s && centrality > c));
                if better {
                    best = Some((d, bx.lo[d] + plane as i64, strength, centrality));
                }
            }
        }
    }
    best.map(|(d, plane, _, _)| (d, plane))
}

/// Midpoint split of the longest axis with room for two `min_width`
/// halves; `None` if no axis has room (the box is accepted as-is).
fn bisection<const D: usize>(bx: &CellBox<D>, min_width: usize) -> Option<(usize, i64)> {
    let d = (0..D).max_by_key(|&d| bx.extent(d))?;
    if bx.extent(d) < 2 * min_width as u64 {
        return None;
    }
    Some((d, bx.lo[d] + (bx.extent(d) / 2) as i64))
}

/// Two-pointer partition: cells with `c[axis] < plane` move to the front;
/// returns the split index.
fn partition_in_place<const D: usize>(cells: &mut [[i64; D]], axis: usize, plane: i64) -> usize {
    let mut lt = 0;
    for j in 0..cells.len() {
        if cells[j][axis] < plane {
            cells.swap(lt, j);
            lt += 1;
        }
    }
    lt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::rng::{mix_key, splitmix64};

    /// Flags from ASCII art: `x` is flagged, anything else is not.
    /// Row r, column c maps to cell [c, r].
    fn art(rows: &[&str]) -> Vec<[i64; 2]> {
        let mut out = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                if ch == 'x' {
                    out.push([c as i64, r as i64]);
                }
            }
        }
        out
    }

    fn total_cells(boxes: &[CellBox<2>]) -> u64 {
        boxes.iter().map(CellBox::cells).sum()
    }

    /// The structural guarantees: pairwise disjoint boxes covering every
    /// flag.
    fn assert_cover_disjoint<const D: usize>(flags: &[[i64; D]], boxes: &[CellBox<D>]) {
        for f in flags {
            let n = boxes.iter().filter(|b| b.contains(*f)).count();
            assert_eq!(n, 1, "flag {f:?} covered by {n} boxes");
        }
        for (i, a) in boxes.iter().enumerate() {
            for b in &boxes[i + 1..] {
                assert!(!a.intersects(b), "boxes overlap: {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn dense_rectangle_is_one_full_box() {
        let mut flags = art(&["xxxx", "xxxx", "xxxx"]);
        let boxes = cluster(&mut flags, &ClusterParams::default());
        assert_eq!(
            boxes,
            vec![CellBox {
                lo: [0, 0],
                hi: [4, 3]
            }]
        );
        assert_cover_disjoint(&flags, &boxes);
    }

    #[test]
    fn islands_split_at_holes() {
        // Two blobs separated by empty columns (B–R: holes are exploited
        // before any inflection search).
        let mut flags = art(&[
            "xx....xxx", //
            "xx....xxx", //
            "......xxx", //
        ]);
        let boxes = cluster(&mut flags, &ClusterParams::default());
        assert_eq!(boxes.len(), 2, "{boxes:?}");
        assert_cover_disjoint(&flags, &boxes);
        // Each island is boxed tightly.
        assert_eq!(total_cells(&boxes), flags.len() as u64);
    }

    #[test]
    fn l_shape_splits_at_inflection() {
        // An L has no holes; the inflection in the signature is the
        // corner, and two boxes cover it with 100% efficiency.
        let mut flags = art(&[
            "xx......", //
            "xx......", //
            "xx......", //
            "xx......", //
            "xxxxxxxx", //
            "xxxxxxxx", //
        ]);
        let n = flags.len() as u64;
        let boxes = cluster(
            &mut flags,
            &ClusterParams {
                efficiency: 0.95,
                min_width: 2,
            },
        );
        assert_cover_disjoint(&flags, &boxes);
        assert_eq!(
            total_cells(&boxes),
            n,
            "L must be covered exactly: {boxes:?}"
        );
        assert_eq!(boxes.len(), 2, "{boxes:?}");
    }

    #[test]
    fn v_shape_is_covered_efficiently() {
        let mut flags = art(&[
            "x.....x", //
            "x.....x", //
            ".x...x.", //
            ".x...x.", //
            "..x.x..", //
            "..xxx..", //
        ]);
        let n = flags.len();
        let boxes = cluster(
            &mut flags,
            &ClusterParams {
                efficiency: 0.7,
                min_width: 1,
            },
        );
        assert_cover_disjoint(&flags, &boxes);
        let eff = n as f64 / total_cells(&boxes) as f64;
        assert!(eff >= 0.7, "V-shape efficiency {eff:.2} with {boxes:?}");
    }

    #[test]
    fn diagonal_stripe_bisects_past_the_anomaly() {
        // The paper's anomalous case: a 45° stripe has hole-free
        // signatures with zero second difference — only the bisection
        // fallback makes progress. Without it, efficiency floors at
        // 8/64; with it, boxes shrink until acceptably tight.
        let mut flags: Vec<[i64; 2]> = (0..8).map(|i| [i, i]).collect();
        let boxes = cluster(
            &mut flags,
            &ClusterParams {
                efficiency: 0.8,
                min_width: 1,
            },
        );
        assert_cover_disjoint(&flags, &boxes);
        let eff = flags.len() as f64 / total_cells(&boxes) as f64;
        assert!(eff >= 0.8, "diagonal efficiency {eff:.2} with {boxes:?}");

        // With a width floor, boxes stop splitting early and efficiency
        // settles at the floor's 50% — accepted below threshold rather
        // than violating min_width.
        let mut flags: Vec<[i64; 2]> = (0..8).map(|i| [i, i]).collect();
        let boxes = cluster(
            &mut flags,
            &ClusterParams {
                efficiency: 0.8,
                min_width: 2,
            },
        );
        assert_cover_disjoint(&flags, &boxes);
        for b in &boxes {
            let eff = boxes_flags(&flags, b) as f64 / b.cells() as f64;
            assert!(eff >= 0.5, "box {b:?} efficiency {eff:.2}");
        }
    }

    fn boxes_flags<const D: usize>(flags: &[[i64; D]], b: &CellBox<D>) -> usize {
        flags.iter().filter(|f| b.contains(**f)).count()
    }

    #[test]
    fn three_dimensional_islands() {
        // Two 2×2×2 blobs separated along z: the same code, one more
        // signature.
        let mut flags = Vec::new();
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    flags.push([x, y, z]);
                    flags.push([x, y, z + 6]);
                }
            }
        }
        let boxes = cluster(&mut flags, &ClusterParams::default());
        assert_eq!(boxes.len(), 2, "{boxes:?}");
        assert_cover_disjoint(&flags, &boxes);
        assert_eq!(boxes.iter().map(CellBox::cells).sum::<u64>(), 16);
    }

    #[test]
    fn output_is_independent_of_input_order() {
        let flags = art(&[
            "xx....xxx", //
            "xx..x.xxx", //
            "....x.xxx", //
        ]);
        let mut a = flags.clone();
        let mut b = flags;
        // Deterministic shuffle via the counter-based mixer.
        let n = b.len();
        for i in (1..n).rev() {
            let j = (splitmix64(mix_key(9, &[i as u64])) % (i as u64 + 1)) as usize;
            b.swap(i, j);
        }
        let params = ClusterParams {
            efficiency: 0.8,
            min_width: 1,
        };
        let mut boxes_a = cluster(&mut a, &params);
        let mut boxes_b = cluster(&mut b, &params);
        boxes_a.sort_unstable();
        boxes_b.sort_unstable();
        assert_eq!(boxes_a, boxes_b);
    }

    #[test]
    fn grown_and_clipped_boxes() {
        let b = CellBox {
            lo: [2, 2],
            hi: [4, 4],
        };
        let domain = CellBox {
            lo: [0, 0],
            hi: [5, 5],
        };
        assert_eq!(
            b.grown(2).clipped(&domain),
            CellBox {
                lo: [0, 0],
                hi: [5, 5]
            }
        );
        assert_eq!(
            b.grown(1),
            CellBox {
                lo: [1, 1],
                hi: [5, 5]
            }
        );
        // Disjoint clip yields an empty (zero-extent) box, not a panic.
        let far = CellBox {
            lo: [10, 10],
            hi: [12, 12],
        };
        assert_eq!(far.clipped(&domain).cells(), 0);
    }

    #[test]
    fn empty_input_yields_no_boxes() {
        let boxes = cluster::<2>(&mut [], &ClusterParams::default());
        assert!(boxes.is_empty());
    }
}
